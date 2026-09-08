"""Run discovery and the two reader backends.

A *run* is a directory that directly contains one or more TensorBoard event
files (``*.tfevents.*``), matching TensorBoard's own grouping. A logdir may hold
many runs in nested subdirectories; all are discovered recursively.

Two backends, same :class:`~terminalboard.model.Run` output:
  * :class:`TBReader`    — default, uses tensorboard's EventAccumulator.
  * :class:`LightReader` — ``--light``, the dependency-free pure-Python parser.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

from .model import Run


def _is_event_file(name: str) -> bool:
    return ".tfevents." in name


def discover_runs(logdir: str) -> Dict[str, List[str]]:
    """Map run-name -> list of event-file paths, scanning logdir recursively."""
    logdir = os.path.abspath(logdir)
    runs: Dict[str, List[str]] = {}
    for dirpath, _dirs, files in os.walk(logdir):
        event_files = sorted(
            os.path.join(dirpath, f) for f in files if _is_event_file(f)
        )
        if not event_files:
            continue
        rel = os.path.relpath(dirpath, logdir)
        name = "." if rel == "." else rel
        runs[name] = event_files
    return runs


class BaseReader:
    """Common discovery; subclasses fill in event parsing in ``poll``."""

    def __init__(self, logdir: str):
        self.logdir = os.path.abspath(logdir)
        self.runs: Dict[str, Run] = {}

    def poll(self) -> Dict[str, Run]:  # pragma: no cover - interface
        raise NotImplementedError

    def all_tags(self) -> List[str]:
        tags = set()
        for run in self.runs.values():
            tags.update(run.series.keys())
        return sorted(tags)


class LightReader(BaseReader):
    """Pure-Python incremental reader. Tails files between polls."""

    def __init__(self, logdir: str):
        super().__init__(logdir)
        from .reader_light import LightEventFile  # noqa: F401  (state typing)

        self._file_state: Dict[str, "object"] = {}

    def poll(self) -> Dict[str, Run]:
        from .reader_light import collect_run

        discovered = discover_runs(self.logdir)
        for name, files in discovered.items():
            run = self.runs.get(name)
            if run is None:
                run = Run(name=name, path=os.path.join(self.logdir, name))
                self.runs[name] = run
            collect_run(run, files, self._file_state)
        return self.runs


class TBReader(BaseReader):
    """EventAccumulator-backed reader. Robust across summary encodings."""

    def __init__(self, logdir: str):
        super().__init__(logdir)
        self._accumulators: Dict[str, "object"] = {}

    def poll(self) -> Dict[str, Run]:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
        from .model import HistogramSeries, ScalarSeries, TextSeries

        discovered = discover_runs(self.logdir)
        for name in discovered:
            path = os.path.join(self.logdir, name)
            acc = self._accumulators.get(name)
            if acc is None:
                acc = EventAccumulator(
                    path, size_guidance={"scalars": 0, "histograms": 0, "tensors": 0}
                )
                self._accumulators[name] = acc
            acc.Reload()

            run = Run(name=name, path=path)
            tags = acc.Tags()

            for tag in tags.get("scalars", []):
                ev = acc.Scalars(tag)
                run.series[tag] = ScalarSeries(
                    tag=tag,
                    steps=[e.step for e in ev],
                    values=[e.value for e in ev],
                    wall_times=[e.wall_time for e in ev],
                )

            for tag in tags.get("histograms", []):
                ev = acc.Histograms(tag)
                hs = HistogramSeries(tag=tag)
                for e in ev:
                    h = e.histogram_value
                    hs.append(e.step, list(h.bucket_limit), list(h.bucket),
                              e.wall_time)
                run.series[tag] = hs

            # Text summaries are tensors registered under the 'text' plugin.
            try:
                text_tags = list(acc.PluginTagToContent("text").keys())
            except (KeyError, Exception):
                text_tags = []
            for tag in text_tags:
                try:
                    ev = acc.Tensors(tag)
                except (KeyError, Exception):
                    continue
                ts = TextSeries(tag=tag)
                for e in ev:
                    vals = getattr(e.tensor_proto, "string_val", [])
                    text = vals[0].decode("utf-8", "replace") if vals else ""
                    ts.append(e.step, text, e.wall_time)
                run.series[tag] = ts

            self.runs[name] = run
        return self.runs


def make_reader(logdir: str, use_tb: bool = False) -> BaseReader:
    """Pick a backend.

    Default is the dependency-free pure-Python parser. ``use_tb`` selects the
    tensorboard EventAccumulator (more battle-tested across exotic encodings),
    falling back to the light reader with a note if tensorboard isn't installed.
    """
    if not use_tb:
        return LightReader(logdir)
    try:
        import tensorboard  # noqa: F401
    except ImportError:
        print(
            "terminalboard: --tb requested but tensorboard isn't installed; "
            "using the built-in parser instead. "
            "(pip install 'terminalboard[tb]')",
            file=sys.stderr,
        )
        return LightReader(logdir)
    return TBReader(logdir)
