#!/usr/bin/env python3
"""Generate a demo TensorBoard logdir exercising every type terminalboard
supports: scalars (curves), text summaries, and histograms (heatmaps) — across
several experiments so you can test overlay, z-order, colors, and filtering.

    python3 gen_demo_logs.py            # writes ./demo_logs/
    terminalboard demo_logs             # view it

Self-contained: writes valid TFRecord/protobuf event files (masked CRC32C), so
it needs no tensorflow/torch and works with both terminalboard parsers.
"""
import json
import math
import os
import struct

# --- masked CRC32C (TensorFlow TFRecord framing) ---------------------------
_CRC = []
for _i in range(256):
    _c = _i
    for _ in range(8):
        _c = (_c >> 1) ^ (0x82F63B78 & -(_c & 1))
    _CRC.append(_c & 0xFFFFFFFF)


def _crc32c(data):
    c = 0xFFFFFFFF
    for b in data:
        c = (c >> 8) ^ _CRC[(c ^ b) & 0xFF]
    return c ^ 0xFFFFFFFF


def _mask(data):
    c = _crc32c(data)
    return (((c >> 15) | (c << 17)) + 0xA282EAD8) & 0xFFFFFFFF


# --- protobuf wire helpers --------------------------------------------------
def _varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | 0x80 if n else b)
        if not n:
            return bytes(out)


def _key(field, wire):
    return _varint((field << 3) | wire)


def _ld(field, data):
    return _key(field, 2) + _varint(len(data)) + data


def scalar_value(tag, val):
    return _ld(1, tag.encode()) + _key(2, 5) + struct.pack("<f", val)


def text_value(tag, text):
    tensor = _key(1, 0) + _varint(7) + _ld(8, text.encode())   # DT_STRING
    return _ld(1, tag.encode()) + _ld(8, tensor)


def histogram_value(tag, edges, counts):
    hp = _key(6, 2) + _varint(len(edges) * 8) + struct.pack(f"<{len(edges)}d", *edges)
    hp += _key(7, 2) + _varint(len(counts) * 8) + struct.pack(f"<{len(counts)}d", *counts)
    return _ld(1, tag.encode()) + _ld(5, hp)


def _metadata(plugin, content=b""):
    return _ld(1, _ld(1, plugin.encode()) + _ld(2, content))


def pr_curve_value(tag, precision, recall):
    n = len(precision)
    rows = [0.0] * (4 * n) + list(precision) + list(recall)   # [6, N] DT_FLOAT
    floats = struct.pack(f"<{len(rows)}f", *rows)
    tensor = _key(1, 0) + _varint(1) + _key(5, 2) + _varint(len(floats)) + floats
    return _ld(1, tag.encode()) + _ld(8, tensor) + _ld(9, _metadata("pr_curves"))


def _struct_value(val):
    if isinstance(val, bool):
        return _key(4, 0) + _varint(1 if val else 0)
    if isinstance(val, (int, float)):
        return _key(2, 1) + struct.pack("<d", float(val))
    return _ld(3, str(val).encode())


def hparams_session(values):
    entries = b"".join(_ld(1, _ld(1, k.encode()) + _ld(2, _struct_value(v)))
                       for k, v in values.items())
    meta = _metadata("hparams", _ld(3, entries))
    return _ld(1, b"_hparams_/session_start_info") + _ld(9, meta)


def hparams_experiment(hparam_names, metric_tags):
    exp = b"".join(_ld(5, _ld(1, n.encode())) for n in hparam_names)
    exp += b"".join(_ld(6, _ld(1, _ld(2, t.encode()))) for t in metric_tags)
    return _ld(1, b"_hparams_/experiment") + _ld(9, _metadata("hparams", _ld(2, exp)))


def _event(step, wall, values):
    e = _key(1, 1) + struct.pack("<d", wall) + _key(2, 0) + _varint(step)
    return e + _ld(5, b"".join(_ld(1, v) for v in values))


def _record(payload):
    length = struct.pack("<Q", len(payload))
    return (length + struct.pack("<I", _mask(length))
            + payload + struct.pack("<I", _mask(payload)))


def write_run(path, records, sps=12.0, base=1_700_000_000.0):
    # sps = wall-clock seconds per step, so the x-axis 'time' mode differs from
    # 'step' (and runs at different speeds look different on the time axis).
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        fv = _key(1, 1) + struct.pack("<d", base) + _ld(3, b"brain.Event:2")
        f.write(_record(fv))
        for step, values in records:
            f.write(_record(_event(step, base + step * sps, values)))


# --- a deterministic pseudo-random (no imports / reproducible) --------------
def _rng(seed):
    s = seed & 0xFFFFFFFF
    while True:
        s = (1103515245 * s + 12345) & 0x7FFFFFFF
        yield s / 0x7FFFFFFF


def _gauss_hist(edges, mean, std, total):
    """Bucket counts for a Gaussian(mean,std) over the given right-edges."""
    counts = []
    prev = edges[0] - (edges[1] - edges[0])
    for e in edges:
        mid = (prev + e) / 2
        counts.append(total * math.exp(-((mid - mean) ** 2) / (2 * std * std)))
        prev = e
    return counts


# --- experiment definitions -------------------------------------------------
EXPERIMENTS = {
    # name: (loss_scale, lr, noise_seed, config)
    "baseline":            (1.0, 1e-3, 1, {"lr": 1e-3, "dropout": 0.1, "model": "resnet50"}),
    "high_lr":             (1.3, 3e-3, 2, {"lr": 3e-3, "dropout": 0.1, "model": "resnet50"}),
    "ablation_nodropout":  (0.9, 1e-3, 3, {"lr": 1e-3, "dropout": 0.0, "model": "resnet50"}),
}

STEPS = list(range(0, 1000, 10))           # 100 points for scalars
HIST_STEPS = list(range(0, 1000, 40))      # fewer, for histograms
EDGES = [(-4.0 + 8.0 * i / 30) for i in range(1, 31)]   # 30 buckets over [-4,4]


def make_records(name):
    scale, lr, seed, config = EXPERIMENTS[name]
    rnd = _rng(seed)
    records = []
    for s in STEPS:
        t = s / 1000.0
        loss = scale * (3.0 * math.exp(-3 * t) + 0.05) + 0.05 * (next(rnd) - 0.5)
        acc = (1 - math.exp(-4 * t)) * (0.95 if "nodropout" in name else 0.9)
        acc += 0.02 * (next(rnd) - 0.5)
        vals = [
            scalar_value("train/loss", max(0.0, loss)),
            scalar_value("train/accuracy", min(1.0, max(0.0, acc))),
            scalar_value("val/loss", max(0.0, loss * 1.1 + 0.05)),
            scalar_value("val/accuracy", min(1.0, max(0.0, acc - 0.03))),
            scalar_value("train/lr", lr * (0.5 + 0.5 * math.cos(math.pi * t))),
            scalar_value("train/grad_norm", 5.0 * math.exp(-2 * t) + 0.1),
            scalar_value("system/gpu_mem_gb", 0.0),          # a flat series
        ]
        if s == 0:                                            # text at step 0
            vals.append(text_value("config/json", json.dumps(config, indent=2)))
            vals.append(text_value(
                "notes/run", f"# {name}\n\nLearning rate **{lr}**, "
                f"dropout {config['dropout']}.\nStarted as a demo run."))
            # HParams: this run's hyperparameters (+ the experiment definition).
            vals.append(hparams_experiment(
                ["lr", "dropout", "model"], ["val/accuracy", "val/loss"]))
            vals.append(hparams_session(
                {"lr": lr, "dropout": config["dropout"], "model": config["model"]}))
        records.append((s, vals))

    for s in HIST_STEPS:                                      # drifting histograms
        t = s / 1000.0
        # weights: centered, slowly narrowing
        w_counts = _gauss_hist(EDGES, mean=0.3 * math.sin(3 * t),
                               std=1.2 - 0.6 * t, total=1000)
        # gradients: shrinking spread as training converges
        g_counts = _gauss_hist(EDGES, mean=0.0, std=max(0.15, 1.5 * math.exp(-2 * t)),
                               total=1000)
        # PR curve sharpening as accuracy improves (recall grid, precision falls
        # off later as training progresses).
        recall = [i / 10 for i in range(11)]
        sharp = min(0.95, 0.55 + 0.8 * t)
        precision = [min(1.0, sharp + (1 - sharp) * (1 - r) ** (1 + 6 * t))
                     for r in recall]
        records.append((s, [
            histogram_value("weights/layer0", EDGES, w_counts),
            histogram_value("grad/layer0", EDGES, g_counts),
            pr_curve_value("pr/classifier", precision, recall),
        ]))
    records.sort(key=lambda r: r[0])
    return records


def main():
    out = os.path.abspath("demo_logs")
    sps = {"baseline": 12.0, "high_lr": 9.0, "ablation_nodropout": 15.0}
    for name in EXPERIMENTS:
        path = os.path.join(out, name, "events.out.tfevents.1700000000.demo.1.0")
        write_run(path, make_records(name), sps=sps.get(name, 12.0))
    print(f"Wrote {len(EXPERIMENTS)} runs to {out}/")
    print("Scalars: train/loss, train/accuracy, val/loss, val/accuracy, "
          "train/lr, train/grad_norm, system/gpu_mem_gb (flat)")
    print("Text:    config/json, notes/run")
    print("Hists:   weights/layer0, grad/layer0  (heatmaps; press b for bands)")
    print("PR:      pr/classifier")
    print("HParams: lr, dropout, model  (press P for the table)")
    print(f"\nView it:  terminalboard {out}")


if __name__ == "__main__":
    main()
