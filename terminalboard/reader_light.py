"""Self-contained pure-Python TensorBoard event reader (the ``--light`` path).

No tensorboard / tensorflow / protobuf dependency. It decodes just enough of
two formats to extract scalars:

  * TFRecord framing:  <uint64 length><uint32 crc><payload><uint32 crc>
  * protobuf wire format for the Event / Summary / Value / TensorProto messages

Scalars appear in event files in two shapes:
  * legacy: Summary.Value.simple_value (float32, field 2)
  * TF2:    Summary.Value.tensor (a 0-d TensorProto, field 8)
Both are handled. CRCs are not verified (we instead stop cleanly on any
truncated/partial trailing record, which is what live-tailing needs).
"""
from __future__ import annotations

import struct
from typing import Dict, Iterator, List, Optional, Tuple

from .model import Run, ScalarSeries

# --- protobuf wire-format primitives ---------------------------------------


def _read_varint(buf: bytes, pos: int) -> Tuple[int, int]:
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


def _iter_fields(buf: bytes) -> Iterator[Tuple[int, int, object]]:
    """Yield (field_number, wire_type, value) for each field in a message.

    value is an int for varints, or raw bytes for fixed64/fixed32/length-delimited.
    """
    pos = 0
    n = len(buf)
    while pos < n:
        key, pos = _read_varint(buf, pos)
        field = key >> 3
        wt = key & 0x7
        if wt == 0:  # varint
            val, pos = _read_varint(buf, pos)
            yield field, wt, val
        elif wt == 1:  # 64-bit
            yield field, wt, buf[pos:pos + 8]
            pos += 8
        elif wt == 2:  # length-delimited
            ln, pos = _read_varint(buf, pos)
            yield field, wt, buf[pos:pos + ln]
            pos += ln
        elif wt == 5:  # 32-bit
            yield field, wt, buf[pos:pos + 4]
            pos += 4
        else:
            raise ValueError(f"unsupported wire type {wt}")


# --- message decoders -------------------------------------------------------

# TensorProto field numbers
_F_DTYPE, _F_CONTENT = 1, 4
_F_FLOAT_VAL, _F_DOUBLE_VAL, _F_INT_VAL, _F_STRING_VAL = 5, 6, 7, 8
DT_FLOAT, DT_DOUBLE, DT_STRING = 1, 2, 7


def _tensor_dtype(buf: bytes) -> Optional[int]:
    for f, wt, v in _iter_fields(buf):
        if f == _F_DTYPE and wt == 0:
            return v
    return None


def _tensor_doubles(buf: bytes) -> List[float]:
    """All numeric values in a TensorProto (content, double_val, or float_val)."""
    dtype = content = None
    doubles: List[float] = []
    floats: List[float] = []
    for f, wt, v in _iter_fields(buf):
        if f == _F_DTYPE and wt == 0:
            dtype = v
        elif f == _F_CONTENT and wt == 2:
            content = v
        elif f == _F_DOUBLE_VAL:
            if wt == 2:
                doubles += list(struct.unpack(f"<{len(v) // 8}d", v))
            elif wt == 1:
                doubles.append(struct.unpack("<d", v)[0])
        elif f == _F_FLOAT_VAL:
            if wt == 2:
                floats += list(struct.unpack(f"<{len(v) // 4}f", v))
            elif wt == 5:
                floats.append(struct.unpack("<f", v)[0])
    if content:
        if dtype == DT_DOUBLE:
            return list(struct.unpack(f"<{len(content) // 8}d", content))
        return list(struct.unpack(f"<{len(content) // 4}f", content))  # DT_FLOAT
    return doubles or floats


def _tensor_to_float(buf: bytes) -> Optional[float]:
    """Extract a single scalar from a TensorProto."""
    vals = _tensor_doubles(buf)
    return vals[0] if vals else None


def _tensor_strings(buf: bytes) -> List[str]:
    return [bytes(v).decode("utf-8", "replace")
            for f, wt, v in _iter_fields(buf)
            if f == _F_STRING_VAL and wt == 2]


def _tensor_to_histogram(buf: bytes):
    """TF2 histogram tensor: shape [N,3] rows of (left, right, count)."""
    vals = _tensor_doubles(buf)
    if not vals or len(vals) % 3 != 0:
        return None
    edges = [vals[i + 1] for i in range(0, len(vals), 3)]   # right edges
    counts = [vals[i + 2] for i in range(0, len(vals), 3)]
    return edges, counts


def _histo_proto(buf: bytes):
    """Legacy HistogramProto: bucket_limit (field 6) + bucket counts (field 7)."""
    edges: List[float] = []
    counts: List[float] = []
    for f, wt, v in _iter_fields(buf):
        if f == 6:
            if wt == 2:
                edges += list(struct.unpack(f"<{len(v) // 8}d", v))
            elif wt == 1:
                edges.append(struct.unpack("<d", v)[0])
        elif f == 7:
            if wt == 2:
                counts += list(struct.unpack(f"<{len(v) // 8}d", v))
            elif wt == 1:
                counts.append(struct.unpack("<d", v)[0])
    return (edges, counts) if edges and counts else None


def _plugin_data(meta_buf: bytes):
    """SummaryMetadata.plugin_data(1) -> (plugin_name(1), content(2) bytes)."""
    name = None
    content = b""
    for f, wt, v in _iter_fields(meta_buf):
        if f == 1 and wt == 2:                       # plugin_data
            for pf, pwt, pv in _iter_fields(v):
                if pf == 1 and pwt == 2:             # plugin_name
                    name = bytes(pv).decode("utf-8", "replace")
                elif pf == 2 and pwt == 2:           # content
                    content = bytes(pv)
    return name, content


def _tensor_to_prcurve(buf: bytes):
    """PR-curve tensor: shape [6, N] = tp/fp/tn/fn/precision/recall rows."""
    vals = _tensor_doubles(buf)
    if not vals or len(vals) % 6 != 0:
        return None
    n = len(vals) // 6
    precision = vals[4 * n:5 * n]
    recall = vals[5 * n:6 * n]
    if not precision or not recall:
        return None
    return precision, recall


# --- HParams plugin protos --------------------------------------------------

def _struct_value(buf: bytes):
    """google.protobuf.Value -> python scalar (number / string / bool / None)."""
    for f, wt, v in _iter_fields(buf):
        if f == 2 and wt == 1:                       # number_value (double)
            return struct.unpack("<d", v)[0]
        if f == 3 and wt == 2:                       # string_value
            return bytes(v).decode("utf-8", "replace")
        if f == 4 and wt == 0:                       # bool_value
            return bool(v)
    return None


def _parse_hparams(content: bytes):
    """HParamsPluginData content -> ('values', {name: val}) for a run's
    session_start_info, or ('experiment', {hparams: [...], metrics: [...]}) for
    the experiment definition, else None."""
    for f, wt, v in _iter_fields(content):
        if f == 3 and wt == 2:                       # session_start_info
            values = {}
            for sf, swt, sv in _iter_fields(v):
                if sf == 1 and swt == 2:             # map<string, Value> entry
                    key = val = None
                    for ef, ewt, ev in _iter_fields(sv):
                        if ef == 1 and ewt == 2:
                            key = bytes(ev).decode("utf-8", "replace")
                        elif ef == 2 and ewt == 2:
                            val = _struct_value(ev)
                    if key is not None:
                        values[key] = val
            return "values", values
        if f == 2 and wt == 2:                       # experiment
            hps, metrics = [], []
            for ef, ewt, ev in _iter_fields(v):
                if ef == 5 and ewt == 2:             # hparam_infos
                    for hf, hwt, hv in _iter_fields(ev):
                        if hf == 1 and hwt == 2:     # HParamInfo.name
                            hps.append(bytes(hv).decode("utf-8", "replace"))
                elif ef == 6 and ewt == 2:           # metric_infos
                    for mf, mwt, mv in _iter_fields(ev):
                        if mf == 1 and mwt == 2:     # MetricName name
                            for nf, nwt, nv in _iter_fields(mv):
                                if nf == 2 and nwt == 2:   # name.tag
                                    metrics.append(
                                        bytes(nv).decode("utf-8", "replace"))
            return "experiment", {"hparams": hps, "metrics": metrics}
    return None


def _parse_value(buf: bytes):
    """Decode a Summary.Value -> (tag, kind, payload) or None.

    kind is 'scalar' (payload float), 'text' (payload str), or 'histogram'
    (payload (edges, counts)).
    """
    tag = simple = tensor = histo = plugin = None
    content = b""
    for f, wt, v in _iter_fields(buf):
        if f == 1 and wt == 2:           # tag (string)
            tag = bytes(v).decode("utf-8", "replace")
        elif f == 2 and wt == 5:         # simple_value (float32)
            simple = struct.unpack("<f", v)[0]
        elif f == 5 and wt == 2:         # histo (legacy HistogramProto)
            histo = v
        elif f == 8 and wt == 2:         # tensor (TensorProto)
            tensor = v
        elif f == 9 and wt == 2:         # metadata (SummaryMetadata)
            plugin, content = _plugin_data(v)
    if tag is None:
        return None
    if plugin == "hparams":              # data lives in the metadata, not a value
        return tag, "hparams", content
    if histo is not None:
        hp = _histo_proto(histo)
        if hp is not None:
            return tag, "histogram", hp
    if plugin == "pr_curves" and tensor is not None:
        pr = _tensor_to_prcurve(tensor)
        if pr is not None:
            return tag, "pr_curve", pr
    if simple is not None:
        return tag, "scalar", simple
    if tensor is not None:
        if plugin == "text" or _tensor_dtype(tensor) == DT_STRING:
            strs = _tensor_strings(tensor)
            return tag, "text", (strs[0] if strs else "")
        if plugin == "histograms":
            hp = _tensor_to_histogram(tensor)
            if hp is not None:
                return tag, "histogram", hp
        val = _tensor_to_float(tensor)
        if val is not None:
            return tag, "scalar", val
    return None


def _parse_event(buf: bytes):
    """Decode an Event -> (step, wall_time, [(tag, kind, payload), ...])."""
    step: Optional[int] = None
    wall_time = 0.0
    values = []
    for f, wt, v in _iter_fields(buf):
        if f == 1 and wt == 1:           # wall_time (double)
            wall_time = struct.unpack("<d", v)[0]
        elif f == 2 and wt == 0:         # step (int64)
            step = v
        elif f == 5 and wt == 2:         # summary (Summary message)
            for sf, swt, sv in _iter_fields(v):
                if sf == 1 and swt == 2:  # repeated Value value = 1
                    parsed = _parse_value(sv)
                    if parsed is not None:
                        values.append(parsed)
    return step, wall_time, values


# --- TFRecord framing -------------------------------------------------------


def _read_records(data: bytes, offset: int) -> Tuple[List[bytes], int]:
    """Return (complete payloads, new_offset). Stops at the first incomplete
    record so partial/in-progress writes are simply retried next poll."""
    payloads: List[bytes] = []
    n = len(data)
    pos = offset
    while pos + 12 <= n:
        (length,) = struct.unpack("<Q", data[pos:pos + 8])
        end = pos + 12 + length + 4
        if end > n:
            break  # record still being written
        payload = data[pos + 12:pos + 12 + length]
        payloads.append(payload)
        pos = end
    return payloads, pos


# --- public reader ----------------------------------------------------------


class LightEventFile:
    """Tails one event file, remembering how far it has read."""

    def __init__(self, path: str):
        self.path = path
        self._offset = 0

    def read_new(self) -> List[Tuple[Optional[int], float, List[Tuple[str, float]]]]:
        try:
            with open(self.path, "rb") as fh:
                data = fh.read()
        except OSError:
            return []
        if len(data) <= self._offset:
            return []
        payloads, new_offset = _read_records(data, self._offset)
        self._offset = new_offset
        events = []
        for p in payloads:
            try:
                events.append(_parse_event(p))
            except (IndexError, struct.error, ValueError):
                continue  # skip a malformed record, keep going
        return events


def collect_run(run: Run, files: List[str], state: Dict[str, LightEventFile]) -> None:
    """Read any new events from this run's files into run.series (in place)."""
    for path in files:
        ef = state.get(path)
        if ef is None:
            ef = LightEventFile(path)
            state[path] = ef
        for step, wall_time, values in ef.read_new():
            if step is None:
                step = 0          # proto3 omits step==0; record it at 0
            for tag, kind, payload in values:
                if kind == "hparams":       # not a series — fold into run metadata
                    info = _parse_hparams(payload)
                    if info is None:
                        continue
                    what, data = info
                    if what == "values":
                        run.hparams.update(data)
                    elif what == "experiment" and data.get("hparams"):
                        run.hparam_info = data
                    continue
                s = run.get(tag, kind)
                if kind in ("histogram", "pr_curve"):
                    s.append(step, payload[0], payload[1], wall_time)
                else:                       # scalar (float) or text (str)
                    s.append(step, payload, wall_time)
