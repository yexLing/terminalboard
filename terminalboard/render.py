"""Rendering: turn selected series into terminal output.

Panels are rendered independently into fixed-size text blocks and then tiled into
a grid, so a page can freely mix **scalars** (curves), **text** summaries, and
**histograms** (drawn as a heatmap of the distribution over steps). Everything is
pure text (plotext braille/Unicode + custom block widgets) — no images, works
over any terminal.
"""
from __future__ import annotations

import math
import re
import textwrap
from typing import Dict, List, Optional, Sequence

from .model import Run

# A small, readable palette reused across runs/overlays. Each entry pairs a
# plotext color name (for the curve) with the matching ANSI SGR code (for the
# run legend / markers we draw ourselves).
_RUN_STYLES = [
    ("cyan", "36"), ("green", "32"), ("magenta", "35"), ("yellow", "33"),
    ("blue", "34"), ("red", "31"), ("white", "37"),
]
_PALETTE = [name for name, _ in _RUN_STYLES]
_SHADES = " ░▒▓█"
_EMPTY_MSG = "\n  (nothing matches the current filter yet…)\n"

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


# --- generic helpers --------------------------------------------------------

def ema(values: Sequence[float], alpha: float) -> List[float]:
    """Exponential moving average; ``alpha`` in [0, 1) is the smoothing weight."""
    if alpha <= 0 or not values:
        return list(values)
    out: List[float] = []
    m = values[0]
    for v in values:
        m = alpha * m + (1 - alpha) * v
        out.append(m)
    return out


def subsample(xs: Sequence, ys: Sequence, max_points: int):
    """Evenly thin a series to at most ``max_points`` for fast drawing."""
    n = len(xs)
    if max_points <= 0 or n <= max_points:
        return list(xs), list(ys)
    step = n / max_points
    idx = [int(i * step) for i in range(max_points)]
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return [xs[i] for i in idx], [ys[i] for i in idx]


def shorten_tag(tag: str, maxlen: int) -> str:
    """Fit a tag into ``maxlen`` cells — prefer the trailing path segment, then a
    leading-ellipsis truncate."""
    if maxlen <= 0 or len(tag) <= maxlen:
        return tag
    leaf = tag.rsplit("/", 1)[-1]
    if len(leaf) <= maxlen:
        return leaf
    if maxlen <= 1:
        return leaf[:maxlen]
    return "…" + leaf[-(maxlen - 1):]


def title_tag(tag: str, maxlen: int) -> str:
    """Show the FULL tag path; if too wide, keep the tail (leaf + as much of the
    namespace as fits) with a leading ellipsis — never just the leaf."""
    if maxlen <= 0 or len(tag) <= maxlen:
        return tag
    if maxlen <= 1:
        return tag[-maxlen:]
    return "…" + tag[-(maxlen - 1):]


def wrap_title(tag: str, w: int, max_lines: int):
    """Wrap a tag across up to ``max_lines`` lines of width ``w`` (char-wise),
    **balancing** the line lengths so every line is roughly equal — that way each
    line, once centered, shows margins on both sides instead of the first line
    filling the whole width.

    If it doesn't fit in ``max_lines`` lines of width ``w``, the last line keeps
    the tail (leaf) with a leading ellipsis so the most specific part stays
    visible."""
    w = max(1, w)
    need = max(1, -(-len(tag) // w))          # min lines to fit at width w
    if need <= max_lines:
        lines = need
        size = -(-len(tag) // lines)          # even chunk size = ceil(len/lines)
        return [tag[i:i + size] for i in range(0, len(tag), size)] or [""]
    # Too long even for max_lines: fill the first lines, keep the leaf on the last.
    head = [tag[i:i + w] for i in range(0, (max_lines - 1) * w, w)]
    rest = tag[(max_lines - 1) * w:]
    last = rest if len(rest) <= w else "…" + rest[-(w - 1):]
    return head + [last]


_TITLE_MARGIN = 3      # keep the title this many cols short of the panel edge so
                       # adjacent titles have a clear gap (the plot below fills w)


def _title_w(w: int) -> int:
    return max(4, w - _TITLE_MARGIN)


def title_lines_needed(tag: str, w: int) -> int:
    iw = _title_w(w)
    return max(1, -(-len(tag) // max(1, iw)))    # ceil(len/inner_width)


def _title_block(tag: str, w: int, rows: int):
    """Bold title lines, wrapped to a margin-narrowed width and **centered** within
    the full panel width (exactly ``rows`` lines; empty list when rows<=0)."""
    if rows <= 0:
        return []
    out = []
    for c in wrap_title(tag, _title_w(w), rows):
        pad = max(0, (w - len(c)) // 2)          # equal-ish margins on both sides
        out.append(_fit(" " * pad + "\033[1m" + c + "\033[0m", w))
    while len(out) < rows:
        out.append(" " * w)
    return out[:rows]


def grid_dims(n: int, max_cols: int = 3):
    """Choose (rows, cols) for ``n`` panels."""
    if n <= 0:
        return 1, 1
    cols = min(max_cols, n)
    rows = math.ceil(n / cols)
    return rows, cols


def _reset_plotext(plt) -> None:
    """Fully reset plotext's global figure between plots (clear_figure only
    re-inits the active subplot, leaking state across calls)."""
    try:
        import plotext._core as _pc
        _pc._figure.__init__()
    except Exception:
        plt.clear_figure()


def run_legend_lines(run_order, run_colors, width: int) -> List[str]:
    """Map each run to its (stable) color, with **full names** — flowing onto as
    many lines as needed (no truncation)."""
    lines: List[str] = []
    cur, cur_w = "", 0
    for name in run_order:
        code = _RUN_STYLES[run_colors.get(name, 0) % len(_RUN_STYLES)][1]
        seg = f"\033[{code}m──\033[0m {name}"
        seg_w = 3 + len(name)          # "── " + name
        sep_w = 3 if cur else 2        # gap between entries / left margin
        if cur and cur_w + sep_w + seg_w > width:
            lines.append(("  " + cur))
            cur, cur_w = seg, seg_w
        else:
            cur = (cur + "   " + seg) if cur else seg
            cur_w += sep_w + seg_w
    if cur:
        lines.append("  " + cur)
    return lines or [""]


# --- block / tiling helpers (ANSI-aware) ------------------------------------

def _vis_len(s: str) -> int:
    return len(_ANSI_RE.sub("", s))


def _fit(s: str, w: int) -> str:
    """Pad/truncate ``s`` to exactly ``w`` visible columns, keeping ANSI codes."""
    vl = _vis_len(s)
    if vl == w:
        return s
    if vl < w:
        return s + " " * (w - vl)
    out, count, i, n = [], 0, 0, len(s)
    while i < n and count < w:
        m = _ANSI_RE.match(s, i)
        if m:
            out.append(m.group())
            i = m.end()
            continue
        out.append(s[i])
        count += 1
        i += 1
    return "".join(out) + "\033[0m"


def _to_block(s: str, w: int, h: int) -> List[str]:
    lines = [_fit(line, w) for line in s.split("\n")[:h]]
    while len(lines) < h:
        lines.append(" " * w)
    return lines


def _empty_block(w: int, h: int) -> List[str]:
    return [" " * w for _ in range(h)]


def _tile(blocks: List[List[str]], rows: int, cols: int, h: int,
          gutter: int = 1) -> str:
    sep = " " * gutter
    out = []
    for r in range(rows):
        for line in range(h):
            out.append(sep.join(blocks[r * cols + c][line] for c in range(cols)))
    return "\n".join(out)


def _even_indices(n: int, k: int) -> List[int]:
    if n <= 0:
        return []
    if n <= k:
        return list(range(n))
    return [int(i * n / k) for i in range(k)]


def _short_num(x: float) -> str:
    ax = abs(x)
    if ax != 0 and (ax >= 1000 or ax < 0.01):
        return f"{x:.1e}"
    return f"{x:.2f}"


def _pairs(runs, order, tag):
    """[(run_name, series)] for runs (in draw order) that have ``tag``."""
    return [(n, runs[n].series[tag]) for n in order if tag in runs[n].series]


# --- panel widgets ----------------------------------------------------------

def _xs(s, xaxis):
    """X values for a scalar series: by step, or relative wall-time (seconds)."""
    if xaxis == "time" and getattr(s, "wall_times", None):
        t0 = s.wall_times[0]
        return [wt - t0 for wt in s.wall_times]
    return s.steps


def _scalar_block(tag, pairs, run_color, w, h, smooth, marker, theme,
                  max_points, cursor=None, xaxis="step", logy=False,
                  title_rows=1) -> List[str]:
    import plotext as plt

    h = max(2, h)
    tblock = _title_block(tag, w, title_rows)     # our own (wrapped) full-path title
    plot_h = max(1, h - len(tblock))
    _reset_plotext(plt)
    plt.plotsize(w, plot_h)
    plt.theme(theme)
    vmin = vmax = None
    for run_name, s in pairs:
        if not len(s):
            continue
        xs, ys = subsample(_xs(s, xaxis), ema(s.values, smooth), max_points)
        color = _PALETTE[run_color.get(run_name, 0) % len(_PALETTE)]
        plt.plot(xs, ys, marker=marker, color=color)   # draw order = z-order
        if ys:
            lo, hi = min(ys), max(ys)
            vmin = lo if vmin is None else min(vmin, lo)
            vmax = hi if vmax is None else max(vmax, hi)
    do_log = logy and vmin is not None and vmin > 0   # log needs positive values
    if do_log:
        try:
            plt.yscale("log")
        except Exception:
            do_log = False
    # A flat series gives plotext a zero-height axis whose tick search can hang.
    if not do_log and vmin is not None and (vmax - vmin) <= abs(vmin) * 1e-9 + 1e-12:
        pad = abs(vmin) * 0.5 or 1.0
        plt.ylim(vmin - pad, vmax + pad)
    if cursor is not None:
        try:
            plt.vertical_line(cursor, color="white")
        except Exception:
            pass
    # Title rendered by us (not plotext) so the full path is never dropped.
    return tblock + _to_block(plt.build(), w, plot_h)


def _text_block(tag, pairs, run_color, w, h, multi_run, title_rows=1) -> List[str]:
    lines = list(_title_block(tag, w, title_rows))
    body: List[str] = []
    for run_name, s in pairs:
        if not len(s):
            continue
        if multi_run:
            code = _RUN_STYLES[run_color.get(run_name, 0) % len(_RUN_STYLES)][1]
            body.append(f"\033[{code}m● {shorten_tag(run_name, max(4, w - 2))}\033[0m")
        latest = s.texts[-1] if s.texts else ""
        for para in (latest.splitlines() or [""]):
            body.extend(textwrap.wrap(para, w) or [""])
        if multi_run:
            body.append("")
    if not body:
        body = ["(no text)"]
    for line in body[:h - len(lines)]:
        lines.append(_fit(line, w))
    while len(lines) < h:
        lines.append(" " * w)
    return lines[:h]


def _rebin(edges, counts, lo, hi, bins) -> List[float]:
    """Distribute bucket counts onto a fixed ``bins``-row grid over [lo, hi]."""
    out = [0.0] * bins
    if hi <= lo or not counts:
        return out
    binw = (hi - lo) / bins
    for i, c in enumerate(counts):
        if c <= 0:
            continue
        left = edges[i - 1] if i > 0 else lo
        right = edges[i] if i < len(edges) else hi
        a, b = max(lo, left), min(hi, right)
        if b <= a:
            continue
        span = (right - left) or binw
        first = max(0, int((a - lo) / binw))
        last = min(bins - 1, int((b - lo) / binw))
        for bi in range(first, last + 1):
            bl = lo + bi * binw
            ov = min(bl + binw, b) - max(bl, a)
            if ov > 0:
                out[bi] += c * (ov / span)
    return out


def _histogram_block(tag, pairs, run_color, w, h, multi_run, title_rows=1) -> List[str]:
    chosen = None
    for run_name, s in pairs:        # last with data = the one "on top"
        if len(s):
            chosen = (run_name, s)
    if chosen is None:
        return _empty_block(w, h)
    run_name, s = chosen
    lines = list(_title_block(tag, w, title_rows))   # color identifies the run

    all_edges = [e for (edges, _c) in s.buckets for e in edges]
    if not all_edges:
        return _empty_block(w, h)
    lo, hi = min(all_edges), max(all_edges)
    if hi <= lo:
        hi = lo + 1.0
    plot_h = max(2, h - len(lines) - 1)     # title rows + 1 x-axis line
    ylab_w = 8
    plot_w = max(4, w - ylab_w - 1)

    cols = []
    gmax = 0.0
    for ci in _even_indices(len(s.buckets), plot_w):
        edges, counts = s.buckets[ci]
        col = _rebin(edges, counts, lo, hi, plot_h)
        gmax = max(gmax, max(col) if col else 0.0)
        cols.append(col)
    gmax = gmax or 1.0

    for r in range(plot_h):
        bin_i = plot_h - 1 - r        # top row = highest value bin
        cells = "".join(
            _SHADES[min(len(_SHADES) - 1, int((col[bin_i] / gmax) * len(_SHADES)))]
            for col in cols
        )
        lab = _short_num(hi) if r == 0 else (_short_num(lo) if r == plot_h - 1 else "")
        lines.append(_fit(f"{lab:>{ylab_w}} \033[36m{cells}\033[0m", w))

    steps = s.steps
    xaxis = f"{'':>{ylab_w}} {steps[0]}…{steps[-1]}" if steps else ""
    lines.append(_fit(xaxis, w))
    while len(lines) < h:
        lines.append(" " * w)
    return lines[:h]


def _percentile(edges, counts, q: float) -> float:
    """Value at cumulative fraction ``q`` of a bucketed distribution (edges are
    right-hand limits), linearly interpolated within the spanning bucket."""
    total = sum(counts)
    if total <= 0 or not edges:
        return edges[len(edges) // 2] if edges else 0.0
    target = q * total
    cum = 0.0
    for i, c in enumerate(counts):
        if cum + c >= target and c > 0:
            left = edges[i - 1] if i > 0 else (
                edges[0] - (edges[1] - edges[0] if len(edges) > 1 else 1.0))
            return left + (edges[i] - left) * ((target - cum) / c)
        cum += c
    return edges[-1]


def _distribution_block(tag, pairs, run_color, w, h, theme, max_points,
                        title_rows=1) -> List[str]:
    """Percentile bands (0/25/50/75/100) over steps — the 'distributions' view of
    the same histogram data; median in white, the spread in the run's color."""
    import plotext as plt
    chosen = None
    for run_name, s in pairs:        # last with data = on top
        if len(s):
            chosen = (run_name, s)
    if chosen is None:
        return _empty_block(w, h)
    run_name, s = chosen
    h = max(2, h)
    tblock = _title_block(tag, w, title_rows)
    plot_h = max(1, h - len(tblock))
    steps = s.steps
    band = {q: [_percentile(e, c, q) for (e, c) in s.buckets]
            for q in (0.0, 0.25, 0.5, 0.75, 1.0)}
    _reset_plotext(plt)
    plt.plotsize(w, plot_h)
    plt.theme(theme)
    color = _PALETTE[run_color.get(run_name, 0) % len(_PALETTE)]
    for q in (0.0, 0.25, 0.75, 1.0):
        xs, ys = subsample(steps, band[q], max_points)
        plt.plot(xs, ys, marker="braille", color=color)
    xs, ys = subsample(steps, band[0.5], max_points)
    plt.plot(xs, ys, marker="braille", color="white")          # median
    return tblock + _to_block(plt.build(), w, plot_h)


def _nearest_step_index(steps, step) -> int:
    best_i, best_d = 0, None
    for i, st in enumerate(steps):
        d = abs(st - step)
        if best_d is None or d < best_d:
            best_i, best_d = i, d
    return best_i


def _prcurve_block(tag, pairs, run_color, w, h, theme, title_rows=1,
                   step=None) -> List[str]:
    """Precision (y) vs recall (x) curve(s) — one per run, at the latest step (or
    ``step`` if given)."""
    import plotext as plt
    h = max(2, h)
    tblock = _title_block(tag, w, title_rows)
    plot_h = max(1, h - len(tblock))
    _reset_plotext(plt)
    plt.plotsize(w, plot_h)
    plt.theme(theme)
    drew = False
    for run_name, s in pairs:
        if not len(s):
            continue
        idx = (_nearest_step_index(s.steps, step) if step is not None
               else len(s.steps) - 1)
        pts = sorted(zip(s.recall[idx], s.precision[idx]))     # by recall asc
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        color = _PALETTE[run_color.get(run_name, 0) % len(_PALETTE)]
        plt.plot(xs, ys, marker="braille", color=color)
        drew = True
    if not drew:
        return _empty_block(w, h)
    try:
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("recall")
    except Exception:
        pass
    return tblock + _to_block(plt.build(), w, plot_h)


def hparams_table(col_titles, data_rows, width, height, scroll=0):
    """Render an hparams grid: bold header, rule, then one row per run (vertically
    scrollable). Returns (text, total_rows)."""
    ncol = len(col_titles)
    widths = []
    for c in range(ncol):
        wmax = len(str(col_titles[c]))
        for row in data_rows:
            if c < len(row):
                wmax = max(wmax, len(str(row[c])))
        widths.append(max(3, min(wmax, 22)))

    def fmt(cells, bold=False):
        parts = []
        for c in range(ncol):
            cell = str(cells[c]) if c < len(cells) else ""
            if len(cell) > widths[c]:
                cell = cell[:widths[c] - 1] + "…"
            parts.append(f"{cell:<{widths[c]}}")
        line = "  ".join(parts)
        if bold:
            line = "\033[1m" + line + "\033[0m"
        return _fit(line, width)

    rule = _fit("─" * min(width, sum(widths) + 2 * max(0, ncol - 1)), width)
    head = [fmt(col_titles, bold=True), rule]
    body = [fmt(r) for r in data_rows]
    total = len(body)
    avail = max(1, height - len(head))
    scroll = max(0, min(scroll, max(0, total - avail)))
    lines = head + body[scroll:scroll + avail]
    while len(lines) < height:
        lines.append("")
    return "\n".join(lines[:height]), total


# --- renderers --------------------------------------------------------------

class Renderer:
    name = "base"

    def frame(self, runs, tags, *, smooth=0.0, max_cols=3, width=0, height=0,
              run_colors=None, run_order=None, focus=-1) -> str:  # pragma: no cover
        """Return the rendered body as a single string (no printing).

        ``run_colors`` maps run -> stable color index; ``run_order`` is the draw
        order (last = on top) used for z-order cycling; ``focus`` is the index of
        the panel to highlight (-1 = none).
        """
        raise NotImplementedError


def _highlight_block(block: List[str]) -> List[str]:
    """Reverse-video a panel's title line to mark it as focused."""
    if block:
        block[0] = "\033[7m" + _ANSI_RE.sub("", block[0]) + "\033[0m"
    return block


class TextRenderer(Renderer):
    """plotext + custom block widgets, tiled into a grid (default)."""

    name = "text"

    def __init__(self, theme: str = "pro", marker: str = "braille",
                 max_points: int = 400):
        self.theme = theme
        self.marker = marker
        self.max_points = max_points

    def frame(self, runs, tags, *, smooth=0.0, max_cols=3, width=0, height=0,
              run_colors=None, run_order=None, focus=-1, xaxis="step",
              logy=False, hist_mode="heatmap") -> str:
        if not tags:
            return _EMPTY_MSG
        if not width or not height:
            import shutil
            tw, th = shutil.get_terminal_size((100, 30))
            width = width or tw
            height = height or max(4, th - 2)

        run_color = run_colors or {n: i for i, n in enumerate(sorted(runs))}
        order = [n for n in (run_order or sorted(runs)) if n in runs]
        multi_run = len(runs) > 1
        # Full run names, wrapped over as many lines as needed (capped so the
        # legend can't swallow the whole screen).
        legend_lines = (run_legend_lines(sorted(runs), run_color, width)
                        if multi_run else [])
        legend_lines = legend_lines[:max(1, height // 3)] if legend_lines else []
        legend_rows = len(legend_lines)

        rows, cols = grid_dims(len(tags), max_cols)
        gutter = 1
        panel_w = max(16, (width - (cols - 1) * gutter) // cols)
        panel_h = max(4, (height - legend_rows) // rows)
        # Uniform title height across the page (so rows stay aligned): as many
        # lines as the longest visible tag needs, capped at 3 and never eating
        # more than (panel_h - 2) so the plot keeps at least two rows.
        title_rows = max(1, min(3, panel_h - 2,
                                max((title_lines_needed(t, panel_w) for t in tags),
                                    default=1)))

        blocks: List[List[str]] = []
        for idx in range(rows * cols):
            if idx >= len(tags):
                blocks.append(_empty_block(panel_w, panel_h))
                continue
            tag = tags[idx]
            pairs = _pairs(runs, order, tag)
            kind = pairs[0][1].kind if pairs else "scalar"
            if kind == "text":
                block = _text_block(tag, pairs, run_color, panel_w,
                                    panel_h, multi_run, title_rows=title_rows)
            elif kind == "histogram":
                if hist_mode == "dist":
                    block = _distribution_block(tag, pairs, run_color, panel_w,
                                                panel_h, self.theme,
                                                self.max_points, title_rows)
                else:
                    block = _histogram_block(tag, pairs, run_color, panel_w,
                                             panel_h, multi_run,
                                             title_rows=title_rows)
            elif kind == "pr_curve":
                block = _prcurve_block(tag, pairs, run_color, panel_w, panel_h,
                                       self.theme, title_rows=title_rows)
            else:
                block = _scalar_block(tag, pairs, run_color, panel_w,
                                      panel_h, smooth, self.marker, self.theme,
                                      self.max_points, xaxis=xaxis, logy=logy,
                                      title_rows=title_rows)
            if idx == focus:
                block = _highlight_block(block)
            blocks.append(block)
        body = _tile(blocks, rows, cols, panel_h, gutter)
        if legend_lines:
            return "\n".join(legend_lines) + "\n" + body
        return body

    def detail_scalar(self, runs, tag, *, order, run_color, w, h, smooth,
                      cursor_x, xaxis="step", logy=False) -> str:
        """Full-screen scalar plot with a vertical cursor at ``cursor_x`` (in the
        active x-axis domain)."""
        pairs = _pairs(runs, order, tag)
        block = _scalar_block(tag, pairs, run_color, w, h, smooth, self.marker,
                              self.theme, self.max_points, cursor=cursor_x,
                              xaxis=xaxis, logy=logy, title_rows=0)
        return "\n".join(block)

    def detail_prcurve(self, runs, tag, *, order, run_color, w, h, step) -> str:
        """Full-screen P-R curve overlay (all runs) at ``step``."""
        pairs = _pairs(runs, order, tag)
        block = _prcurve_block(tag, pairs, run_color, w, h, self.theme,
                               title_rows=0, step=step)
        return "\n".join(block)
