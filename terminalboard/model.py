"""Shared, backend-agnostic data model for time-series.

Both the ``--light`` (pure-Python) and ``--tb`` (tensorboard) readers produce
these same structures, so the renderers never need to know which parser ran.

Three series kinds, each with a ``kind`` class attribute the renderers dispatch
on: scalars (curves), text summaries, and histograms (drawn as a heatmap of the
distribution over steps).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union


@dataclass
class ScalarSeries:
    """A single scalar tag's time-series within one run."""

    kind = "scalar"
    tag: str
    steps: List[int] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    wall_times: List[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.steps)

    def append(self, step: int, value: float, wall_time: float = 0.0) -> None:
        self.steps.append(step)
        self.values.append(value)
        self.wall_times.append(wall_time)


@dataclass
class TextSeries:
    """A text summary's history within one run."""

    kind = "text"
    tag: str
    steps: List[int] = field(default_factory=list)
    texts: List[str] = field(default_factory=list)
    wall_times: List[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.steps)

    def append(self, step: int, text: str, wall_time: float = 0.0) -> None:
        self.steps.append(step)
        self.texts.append(text)
        self.wall_times.append(wall_time)


@dataclass
class HistogramSeries:
    """A histogram's history within one run.

    Each entry is ``(edges, counts)``: ``edges`` are the right-hand bucket limits
    and ``counts`` the per-bucket population at that step.
    """

    kind = "histogram"
    tag: str
    steps: List[int] = field(default_factory=list)
    buckets: List[Tuple[List[float], List[float]]] = field(default_factory=list)
    wall_times: List[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.steps)

    def append(self, step: int, edges, counts, wall_time: float = 0.0) -> None:
        self.steps.append(step)
        self.buckets.append((list(edges), list(counts)))
        self.wall_times.append(wall_time)


@dataclass
class PRCurveSeries:
    """A precision-recall curve's history within one run.

    Each entry is ``(precision, recall)``: equal-length arrays sampled across
    classification thresholds, so a single step plots as one P-R curve.
    """

    kind = "pr_curve"
    tag: str
    steps: List[int] = field(default_factory=list)
    precision: List[List[float]] = field(default_factory=list)
    recall: List[List[float]] = field(default_factory=list)
    wall_times: List[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.steps)

    def append(self, step: int, precision, recall, wall_time: float = 0.0) -> None:
        self.steps.append(step)
        self.precision.append(list(precision))
        self.recall.append(list(recall))
        self.wall_times.append(wall_time)


Series = Union[ScalarSeries, TextSeries, HistogramSeries, PRCurveSeries]
_SERIES_BY_KIND = {
    "scalar": ScalarSeries,
    "text": TextSeries,
    "histogram": HistogramSeries,
    "pr_curve": PRCurveSeries,
}


@dataclass
class Run:
    """One TensorBoard run: a directory's worth of event files."""

    name: str  # path relative to the logdir (or "." for the logdir itself)
    path: str  # absolute directory path
    series: Dict[str, Series] = field(default_factory=dict)
    # HParams plugin: this run's hyperparameter values (name -> value), and the
    # experiment definition (column order + metric tags) if it carried one.
    hparams: Dict[str, object] = field(default_factory=dict)
    hparam_info: Optional[Dict[str, list]] = None

    def tags(self) -> List[str]:
        return sorted(self.series.keys())

    def get(self, tag: str, kind: str = "scalar") -> Series:
        """Get (creating if needed) the series for ``tag`` of the given kind."""
        s = self.series.get(tag)
        if s is None:
            s = _SERIES_BY_KIND[kind](tag)
            self.series[tag] = s
        return s
