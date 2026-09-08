"""The interactive live dashboard loop."""
from __future__ import annotations

import bisect
import fnmatch
import hashlib
import json
import math
import os
import re
import shutil
import textwrap
import threading
import time
from typing import List, Optional

from . import llm as _llm

from .keys import KeyReader
from .reader import BaseReader
from .render import Renderer, grid_dims, _fit
from .screen import Screen

# Zoom ladder: (rows, cols) per page, from most-zoomed-in (1 big panel) to
# most-zoomed-out (36 small panels). Panel counts: 1,2,4,6,9,12,16,24,36.
_ZOOM_LADDER = [
    (1, 1), (1, 2), (2, 2), (2, 3), (3, 3), (3, 4), (4, 4), (4, 6), (6, 6),
]

# Type selector ('c'): cycle through None (all) and each series kind.
_KIND_CYCLE = [None, "scalar", "histogram", "text", "pr_curve"]
_KIND_LABEL = {None: "all", "scalar": "scalars", "histogram": "histograms",
               "text": "text", "pr_curve": "pr-curves"}


def _fmt_hparam(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)

# Characters that count as part of a "word" for word-wise cursor motion / delete.
_WORDCHARS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
)


def _word_match(word: str, name: str) -> bool:
    """Match a single filter word against ``name``.

    ``!w`` negates; ``/re/`` is a regex; ``* ? [ ]`` make it a glob; otherwise a
    case-insensitive substring.
    """
    neg = word.startswith("!")
    if neg:
        word = word[1:]
    if not word:
        return True
    if len(word) >= 2 and word.startswith("/") and word.endswith("/"):
        try:
            ok = re.search(word[1:-1], name, re.IGNORECASE) is not None
        except re.error:
            ok = False
    elif any(c in word for c in "*?["):
        ok = fnmatch.fnmatch(name.lower(), word.lower())
    else:
        ok = word.lower() in name.lower()
    return (not ok) if neg else ok


def match_filter(patterns: Optional[str], name: str) -> bool:
    """Match ``name`` against a small filter grammar (empty matches everything).

    * If the **whole** pattern is wrapped in ``/.../`` it's a single regex
      (``re.search``, case-insensitive) — use this for regexes containing
      ``|`` or spaces, e.g. ``/^train/(loss|lr)$/``.
    * Otherwise: ``|`` or ``,`` separate **OR** alternatives; whitespace or ``&``
      within an alternative is **AND**; a word is a case-insensitive **substring**
      (``loss`` → ``train/loss``); ``* ? [ ]`` make it a glob; a per-word
      ``/regex/`` (without ``|``/spaces) is also a regex.
    * **``!word`` is a GLOBAL exclude** — it applies to the whole match no matter
      which alternative it sits in. So ``a | b | c !d`` means *(a or b or c) and
      not d*. A filter of only excludes (``!d``) keeps everything but ``d``.
    """
    if not patterns:
        return True
    p = patterns.strip()
    if len(p) >= 2 and p.startswith("/") and p.endswith("/"):
        try:
            return re.search(p[1:-1], name, re.IGNORECASE) is not None
        except re.error:
            return False
    pos_groups = []      # OR over these; each is a list of AND-ed positive words
    negatives = []       # global excludes (collected from every alternative)
    saw_term = False
    for term in re.split(r"[|,]", patterns):
        words = [w for w in re.split(r"[\s&]+", term) if w]
        if not words:
            continue
        saw_term = True
        negatives.extend(w for w in words if w.startswith("!"))
        pos = [w for w in words if not w.startswith("!")]
        if pos:
            pos_groups.append(pos)
    if not saw_term:                              # all-separators → match all
        return True
    if not all(_word_match(w, name) for w in negatives):   # any exclude hit?
        return False
    if not pos_groups:                            # only excludes → keep the rest
        return True
    return any(all(_word_match(w, name) for w in pos) for pos in pos_groups)


def _prev_word(buf: list, pos: int) -> int:
    """Index of the start of the word before ``pos`` (skip seps, then word)."""
    i = pos
    while i > 0 and buf[i - 1] not in _WORDCHARS:
        i -= 1
    while i > 0 and buf[i - 1] in _WORDCHARS:
        i -= 1
    return i


def _next_word(buf: list, pos: int) -> int:
    """Index just past the end of the word at/after ``pos``."""
    n = len(buf)
    i = pos
    while i < n and buf[i] not in _WORDCHARS:
        i += 1
    while i < n and buf[i] in _WORDCHARS:
        i += 1
    return i


class App:
    def __init__(
        self,
        reader: BaseReader,
        renderer: Renderer,
        *,
        tag_filter: Optional[str] = None,
        run_filter: Optional[str] = None,
        smooth: float = 0.6,
        cols: int = 3,
        rows: int = 2,
        interval: float = 2.0,
        xaxis: str = "step",
        logy: bool = False,
        csv_dir: str = "",
        restore: bool = False,
        restore_exclude=(),
    ):
        self.reader = reader
        self.renderer = renderer
        self.tag_filter = tag_filter
        self.run_filter = run_filter
        self.smooth = smooth
        # Stable color index per run: assigned once and never reshuffled, so an
        # experiment keeps its color regardless of filtering or new runs.
        self._run_color_index: dict = {}
        # Per-kind history of applied filter patterns (recalled with up/down).
        self._filter_history = {"tags": [], "runs": []}
        # Rotation of the run draw order (z-order); 'o' cycles which run is on top.
        self._order_rot = 0
        self.interval = interval
        # Cursor over the matching-tags list (drives the focused panel + page).
        self._focus = 0
        # Detail (drill-down) state: a tag string when zoomed in, else None.
        self._detail: Optional[str] = None
        self._detail_run = 0   # which experiment is shown in detail (text/heatmap)
        self._scroll = 0       # scroll offset in a text detail view
        self._cursor = 0       # x-cursor index in a scalar detail view
        self.xaxis = xaxis if xaxis in ("step", "time") else "step"
        self.logy = bool(logy)  # log-scale y for scalar panels
        self._textdiff = False  # text detail: show only keys that differ across runs
        self._status = ""       # transient status line (e.g. after CSV export)
        self._csv_dir = csv_dir  # default folder pre-filled in the CSV save prompt
        self._distmode = False  # histograms shown as distribution bands (b)
        self._kind_filter = None  # type selector: None=all, else a single kind (c)
        self._hparams = False   # full-screen HParams table mode (P)
        self._hparams_total = 0  # row count (for scroll clamping)
        # LLM assistant (optional): config is loaded lazily on first use.
        self._llm_config = None
        self._llm_complete = None   # test hook: inject a fake completion callable
        self._llm_history = []      # prior (user/assistant) turns for the 1-shot ask
        # Chat sidebar (A): a persistent, multi-session conversation on the right.
        self._chat_open = False
        self._chat_full = False     # full-screen chat (hide the dashboard)
        self._dash_cache = None     # (sig, lines) — dashboard reuse while typing
        self._chat_sessions = [{"name": "chat 1", "messages": []}]
        self._chat_active = 0
        self._chat_input = []       # current input buffer (chars)
        self._chat_pos = 0
        self._chat_scroll = 0       # transcript lines scrolled up from the bottom
        self._chat_partial = None   # streaming assistant text in flight (or None)
        self._chat_drafts = []      # sent-message history (Up/Down recall)
        self._chat_draft_idx = 0
        self._cols_override = None   # render the dashboard at this width (split view)
        # Start at the ladder rung closest to the requested grid's panel count.
        target = max(1, rows) * max(1, cols)
        self._zoom = min(
            range(len(_ZOOM_LADDER)),
            key=lambda i: abs(_ZOOM_LADDER[i][0] * _ZOOM_LADDER[i][1] - target),
        )
        self.rows, self.cols = _ZOOM_LADDER[self._zoom]
        # Per-logdir view persistence: restore the last session's filters/zoom/
        # smoothing/etc. (CLI-explicit options win, via restore_exclude).
        self._restore = restore
        if restore:
            self._load_view(exclude=restore_exclude)

    # -- view persistence ----------------------------------------------------

    def _view_state_file(self) -> str:
        base = (os.environ.get("XDG_STATE_HOME")
                or os.path.expanduser("~/.local/state"))
        logdir = getattr(self.reader, "logdir", "") or ""
        h = hashlib.sha1(logdir.encode()).hexdigest()[:12]
        name = os.path.basename(logdir.rstrip("/")) or "root"
        return os.path.join(base, "terminalboard", "views", f"{name}-{h}.json")

    def _save_view(self) -> None:
        try:
            path = self._view_state_file()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            state = {
                "logdir": getattr(self.reader, "logdir", ""),
                "tag_filter": self.tag_filter, "run_filter": self.run_filter,
                "smooth": self.smooth, "xaxis": self.xaxis, "logy": self.logy,
                "order_rot": self._order_rot, "zoom": self._zoom,
                "focus": self._focus, "distmode": self._distmode,
                "kind_filter": self._kind_filter,
                "chat": self._chat_sessions, "chat_active": self._chat_active,
            }
            with open(path, "w") as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass            # persistence is best-effort; never break the session

    def _load_view(self, exclude=()) -> None:
        try:
            path = self._view_state_file()
            if not os.path.isfile(path):
                return
            with open(path) as f:
                s = json.load(f)
        except Exception:
            return
        if not isinstance(s, dict):
            return
        ex = set(exclude)
        try:                        # an old/foreign state file must never crash us
            if "tag_filter" not in ex and "tag_filter" in s:
                self.tag_filter = s["tag_filter"] or None
            if "run_filter" not in ex and "run_filter" in s:
                self.run_filter = s["run_filter"] or None
            if "smooth" not in ex and isinstance(s.get("smooth"), (int, float)):
                self.smooth = max(0.0, min(0.99, float(s["smooth"])))
            if "xaxis" not in ex and s.get("xaxis") in ("step", "time"):
                self.xaxis = s["xaxis"]
            if "logy" not in ex and isinstance(s.get("logy"), bool):
                self.logy = s["logy"]
            if "order_rot" not in ex and isinstance(s.get("order_rot"), int):
                self._order_rot = s["order_rot"]
            if ("zoom" not in ex and isinstance(s.get("zoom"), int)
                    and 0 <= s["zoom"] < len(_ZOOM_LADDER)):
                self._zoom = s["zoom"]
                self.rows, self.cols = _ZOOM_LADDER[self._zoom]
            if "focus" not in ex and isinstance(s.get("focus"), int):
                self._focus = max(0, s["focus"])
            if "distmode" not in ex and isinstance(s.get("distmode"), bool):
                self._distmode = s["distmode"]
            # NB: None is a valid kind_filter (= "all") AND is in _KIND_CYCLE, so
            # an absent key must be checked explicitly — don't rely on .get().
            if ("kind_filter" not in ex and "kind_filter" in s
                    and s["kind_filter"] in _KIND_CYCLE):
                self._kind_filter = s["kind_filter"]
            chat = s.get("chat")
            if isinstance(chat, list) and chat and all(isinstance(c, dict) for c in chat):
                self._chat_sessions = [{"name": str(c.get("name", "chat")),
                                        "messages": c.get("messages", [])}
                                       for c in chat]
                self._chat_active = min(max(0, int(s.get("chat_active", 0))),
                                        len(self._chat_sessions) - 1)
        except Exception:
            pass

    # -- tag selection -------------------------------------------------------

    def _kind_of(self, tag: str) -> Optional[str]:
        """The series kind for ``tag`` (from the first visible run that has it)."""
        for run in self._visible_runs().values():
            s = run.series.get(tag)
            if s is not None:
                return s.kind
        return None

    def _matching_tags(self) -> List[str]:
        # A tag is shown only if at least one *visible* run actually has it.
        visible = self._visible_runs()
        tags = sorted({t for run in visible.values() for t in run.series})
        tags = [t for t in tags if match_filter(self.tag_filter, t)]
        if self._kind_filter:               # type selector (c)
            tags = [t for t in tags if self._kind_of(t) == self._kind_filter]
        return tags

    def _visible_runs(self):
        runs = self.reader.runs
        if not self.run_filter:
            return runs
        return {n: r for n, r in runs.items() if match_filter(self.run_filter, n)}

    def _run_colors(self) -> dict:
        # Assign a stable color index to any run we haven't seen yet (sorted, so
        # the first assignment is deterministic; later runs only ever append).
        for name in sorted(self.reader.runs):
            if name not in self._run_color_index:
                self._run_color_index[name] = len(self._run_color_index)
        return self._run_color_index

    def _run_order(self) -> List[str]:
        # Draw order (last drawn = on top); 'o' rotates which run is on top.
        names = sorted(self._visible_runs().keys())
        if not names:
            return names
        k = self._order_rot % len(names)
        return names[k:] + names[:k]

    def _layout(self):
        """Resolve the focus cursor into (tags, page slice, page idx, n_pages,
        focus-cell-within-page)."""
        tags = self._matching_tags()
        per_page = max(1, self.cols * self.rows)
        n = len(tags)
        self._focus = max(0, min(self._focus, max(0, n - 1)))
        n_pages = max(1, (n + per_page - 1) // per_page)
        page = self._focus // per_page
        start = page * per_page
        return tags, tags[start:start + per_page], page, n_pages, self._focus - start

    # -- rendering -----------------------------------------------------------

    def _header(self, tags: List[str], page: int, n_pages: int) -> str:
        n_vis = len(self._visible_runs())
        n_all = len(self.reader.runs)
        runs_str = f"{n_vis}/{n_all}" if self.run_filter else str(n_all)
        tflt = self.tag_filter or "*"
        eflt = self.run_filter or "*"
        kind = ""
        if self._kind_filter:
            kind = f"  \033[1;35mtype={_KIND_LABEL[self._kind_filter]}\033[0m (\033[36mc\033[0m)"
        dist = "  \033[35mdist\033[0m" if self._distmode else ""
        return (
            f"\033[1mterminalboard\033[0m  "
            f"exp={runs_str} (\033[36mf\033[0m:{eflt})  "
            f"tags={len(tags)} (\033[36mt\033[0m:{tflt}){kind}  "
            f"page {page + 1}/{n_pages}  "
            f"smooth={self.smooth:.2f}  x={self.xaxis}  "
            f"y={'log' if self.logy else 'lin'}{dist}"
        )

    def _footer(self) -> str:
        if self._chat_open:                 # the chat pane has the keyboard
            return ("\033[2m🤖 chat on the right — type & Enter · Esc closes it "
                    "· /help for commands\033[0m")
        per_page = self.rows * self.cols
        return (
            "\033[2m[arrows]focus [Enter]inspect [n/p]page [f/t]ilter "
            f"[c]type [z/Z]zoom({per_page}) [o]rder [b]dist [+/-/0]smooth "
            "[x]axis [l]og [w]csv [P]hparams [a]chat🤖 [H]elp [q/Esc]uit\033[0m"
        )

    def _prompt_footer(self, label: str, text: str, pos: int, kind: str,
                       warn: bool) -> str:
        # Draw the input with a reverse-video block cursor at ``pos``.
        if pos < len(text):
            shown = text[:pos] + "\033[7m" + text[pos] + "\033[0m" + text[pos + 1:]
        else:
            shown = text + "\033[7m \033[0m"
        if kind not in ("tags", "runs"):                  # generic input prompt
            return (f"\033[1m{label}>\033[0m {shown}  "
                    "\033[2m(Enter save · Esc cancel)\033[0m")
        if warn:
            status = "\033[1;31m✗ no matches — pattern not applied\033[0m"
        else:
            if kind == "tags":
                n, unit = len(self._matching_tags()), "tags"
            else:
                n, unit = len(self._visible_runs()), "experiments"
            status = (f"\033[2m({n} {unit} · ←→ move · ↑↓ history · "
                      f"Enter apply · Esc cancel)\033[0m")
        return f"\033[1m{label}>\033[0m {shown}  {status}"

    def _count_matches(self, kind: str, value) -> int:
        if kind == "runs":
            return sum(1 for n in self.reader.runs if match_filter(value, n))
        tags = {t for run in self._visible_runs().values() for t in run.series}
        return sum(1 for t in tags if match_filter(value, t))

    def _parse_chunk(self, s):
        """Split one input chunk into a list of key tokens.

        A single ``os.read`` can return several keypresses at once (key
        auto-repeat, fast typing, paste). Each token is a nav token
        (UP/DOWN/LEFT/RIGHT/HOME/END/DEL/WORD-LEFT/WORD-RIGHT/WORD-DEL-BACK/
        WORD-DEL-FWD/ESC/IGNORE) or a single ordinary character. Handles CSI
        (ESC ``[``) and SS3 (ESC ``O``), including modified arrows
        (Alt/Ctrl+←/→ as ESC ``[1;3D`` / ``[1;5D``) and Alt-b/f/d.
        """
        plain = {"A": "UP", "B": "DOWN", "C": "RIGHT", "D": "LEFT",
                 "H": "HOME", "F": "END"}
        wordkey = {"C": "WORD-RIGHT", "D": "WORD-LEFT"}
        numtilde = {"3": "DEL", "1": "HOME", "7": "HOME", "4": "END", "8": "END",
                    "5": "PGUP", "6": "PGDN"}
        tokens = []
        i, n = 0, len(s)
        while i < n:
            c = s[i]
            if c != "\x1b":
                tokens.append(c)
                i += 1
                continue
            if i + 1 >= n:
                tokens.append("ESC")
                i += 1
                continue
            nxt = s[i + 1]
            if nxt in "[O":                 # CSI / SS3
                j = i + 2
                params = ""
                while j < n and (s[j].isdigit() or s[j] == ";"):
                    params += s[j]
                    j += 1
                if j >= n:
                    tokens.append("ESC")
                    break
                fin = s[j]
                mod = params.split(";")[-1] not in ("", "1") if ";" in params else False
                if fin == "~":
                    tokens.append(numtilde.get(params.split(";")[0], "IGNORE"))
                elif mod and fin in wordkey:   # Alt/Ctrl + Left/Right => by word
                    tokens.append(wordkey[fin])
                elif fin in plain:
                    tokens.append(plain[fin])
                else:
                    tokens.append("IGNORE")
                i = j + 1
            elif nxt in ("b", "B"):         # Alt-b
                tokens.append("WORD-LEFT")
                i += 2
            elif nxt in ("f", "F"):         # Alt-f
                tokens.append("WORD-RIGHT")
                i += 2
            elif nxt == "d":                # Alt-d
                tokens.append("WORD-DEL-FWD")
                i += 2
            elif nxt in ("\x7f", "\b"):     # Alt-Backspace
                tokens.append("WORD-DEL-BACK")
                i += 2
            else:
                tokens.append("ESC")        # lone Esc
                i += 1
        return tokens

    def _termsize(self):
        """Terminal (cols, rows) — cols narrowed to ``_cols_override`` while the
        dashboard is being rendered into the left pane of the chat split."""
        cols, rows = shutil.get_terminal_size((100, 30))
        if self._cols_override:
            cols = self._cols_override
        return cols, rows

    def _build_frame(self, prompt=None) -> str:
        if self._chat_open and prompt is None:
            return self._compose_split()
        return self._dashboard_frame(prompt)

    def _dashboard_frame(self, prompt=None) -> str:
        if self._hparams and prompt is None:
            return self._build_hparams_frame()
        if self._detail is not None and prompt is None:
            return self._build_detail_frame()
        cols, rows = self._termsize()
        all_tags, page_tags, page, n_pages, focus_cell = self._layout()
        header = self._header(all_tags, page, n_pages)
        footer = (self._prompt_footer(*prompt) if prompt
                  else self._with_status(self._footer()))
        # Reserve the header + footer rows; the body must fit the rest so the
        # whole frame is never taller than the terminal (overflow scrolls and
        # would misalign the in-place repaint, leaving stale curves behind).
        body = self.renderer.frame(
            self._visible_runs(), page_tags, smooth=self.smooth, max_cols=self.cols,
            width=cols, height=max(4, rows - 2), run_colors=self._run_colors(),
            run_order=self._run_order(), xaxis=self.xaxis, logy=self.logy,
            focus=(-1 if prompt else focus_cell),
            hist_mode=("dist" if self._distmode else "heatmap"),
        )
        frame = f"{header}\n{body}\n{footer}"
        return self._crop(frame, rows)

    # -- HParams table view --------------------------------------------------

    def _hparams_columns(self):
        """(ordered hparam names, metric tags) — from the experiment def if any,
        else the union of values seen across runs."""
        runs = self._visible_runs()
        info = next((r.hparam_info for r in runs.values() if r.hparam_info), None)
        seen = {k for r in runs.values() for k in r.hparams}
        if info and info.get("hparams"):
            ordered = [h for h in info["hparams"] if h in seen or True]
            extra = sorted(k for k in seen if k not in info["hparams"])
            metrics = [m for m in info.get("metrics", [])
                       if any(m in r.series for r in runs.values())]
            return ordered + extra, metrics
        return sorted(seen), []

    def _hparams_rows(self, hps, metrics):
        runs = self._visible_runs()
        rows = []
        for name in sorted(runs):
            r = runs[name]
            if not r.hparams:
                continue
            cells = [name]
            for h in hps:
                v = r.hparams.get(h)
                cells.append("" if v is None else _fmt_hparam(v))
            for m in metrics:
                s = r.series.get(m)
                cells.append(f"{s.values[-1]:.4g}" if s and s.values else "")
            rows.append(cells)
        return rows

    def _build_hparams_frame(self) -> str:
        from .render import hparams_table
        cols, rows = self._termsize()
        hps, metrics = self._hparams_columns()
        data = self._hparams_rows(hps, metrics)
        header = (f"\033[1mterminalboard\033[0m  HParams  "
                  f"runs={len(data)}  hparams={len(hps)}  metrics={len(metrics)}")
        footer = self._with_status(
            "\033[2m↑/↓ PgUp/PgDn scroll · P/Esc back · q quit\033[0m")
        if not data:
            body = "\n  (no HParams logged in this logdir)"
            return self._crop(f"{header}\n{body}\n{footer}", rows)
        titles = ["run"] + list(hps) + list(metrics)
        body, total = hparams_table(titles, data, cols, max(3, rows - 2),
                                    scroll=self._scroll)
        self._hparams_total = total
        return self._crop(f"{header}\n{body}\n{footer}", rows)

    # -- LLM assistant: context + action executor ---------------------------

    def _zoom_to(self, panels) -> None:
        try:
            target = max(1, int(panels))
        except (TypeError, ValueError):
            return
        self._zoom = min(range(len(_ZOOM_LADDER)),
                         key=lambda i: abs(_ZOOM_LADDER[i][0] * _ZOOM_LADDER[i][1]
                                           - target))
        self.rows, self.cols = _ZOOM_LADDER[self._zoom]

    def _llm_context(self) -> str:
        """A compact JSON summary of what's on screen — fed to the model so it can
        craft precise filters and grounded analysis."""
        runs = self._visible_runs()
        by_kind: dict = {}
        for r in runs.values():
            for t, s in r.series.items():
                by_kind.setdefault(s.kind, set()).add(t)
        tags_by_kind = {k: sorted(v)[:60] for k, v in by_kind.items()}
        # Per scalar tag: last/min/max + trend + non-finite flag across runs.
        scalars: dict = {}
        for t in sorted(by_kind.get("scalar", []))[:40]:
            per = {}
            for n, r in list(runs.items())[:12]:
                s = r.series.get(t)
                if s is not None and getattr(s, "values", None):
                    vals = s.values
                    delta = vals[-1] - vals[0]
                    trend = ("down" if delta < -1e-12 else
                             "up" if delta > 1e-12 else "flat")
                    per[n] = {"last": round(vals[-1], 6),
                              "min": round(min(vals), 6),
                              "max": round(max(vals), 6),
                              "steps": len(vals), "trend": trend}
                    if not all(math.isfinite(v) for v in vals):
                        per[n]["nonfinite"] = True
            if per:
                scalars[t] = per
        hps, metrics = self._hparams_columns()
        all_tags, page_tags, page, n_pages, focus_cell = self._layout()
        mode = ("hparams" if self._hparams else
                "detail" if self._detail else "grid")
        ctx = {
            "view": {                       # what the user is looking at right now
                "mode": mode,
                "focused_tag": (page_tags[focus_cell]
                                if page_tags and focus_cell < len(page_tags) else None),
                "open_detail": self._detail,
                "visible_tags": page_tags,           # the panels on this page
                "visible_count": len(page_tags),
                "total_matching_tags": len(all_tags),
                "page": page + 1, "pages": n_pages,
                "grid": f"{self.rows}x{self.cols}",
            },
            "state": {
                "tag_filter": self.tag_filter, "experiment_filter": self.run_filter,
                "type": self._kind_filter or "all",
                "smoothing": self.smooth, "xaxis": self.xaxis, "logy": self.logy,
                "distribution": self._distmode,
            },
            "experiments": sorted(runs)[:24],
            "tags_by_kind": tags_by_kind,
            "scalars": scalars,
            "hparams": {"columns": hps, "metrics": metrics,
                        "rows": self._hparams_rows(hps, metrics)[:24]} if hps else {},
        }
        return json.dumps(ctx, default=str)

    def _llm_apply_action(self, name: str, args: dict) -> Optional[str]:
        """Apply one validated tool call; return a short human description."""
        a = args or {}
        if name == "set_tag_filter":
            self.tag_filter = (a.get("pattern") or None)
            self._focus = 0
            return f"tags={self.tag_filter or '*'}"
        if name == "set_experiment_filter":
            self.run_filter = (a.get("pattern") or None)
            self._focus = 0
            return f"experiments={self.run_filter or '*'}"
        if name == "set_type":
            k = a.get("kind")
            self._kind_filter = None if k in (None, "all") else k
            self._focus = 0
            return f"type={_KIND_LABEL.get(self._kind_filter, 'all')}"
        if name == "set_smoothing":
            try:
                self.smooth = max(0.0, min(0.99, float(a.get("value"))))
            except (TypeError, ValueError):
                return None
            return f"smooth={self.smooth:.2f}"
        if name == "set_zoom":
            self._zoom_to(a.get("panels"))
            return f"zoom={self.rows * self.cols} panels"
        if name == "set_xaxis":
            self.xaxis = "time" if a.get("axis") == "time" else "step"
            return f"x={self.xaxis}"
        if name == "set_logy":
            self.logy = bool(a.get("on"))
            return f"logy={'on' if self.logy else 'off'}"
        if name == "set_distribution":
            self._distmode = bool(a.get("on"))
            return f"dist={'on' if self._distmode else 'off'}"
        if name == "open_detail":
            tag = a.get("tag")
            if tag and tag in self._matching_tags():
                self._detail = tag
                self._detail_run = 0
                self._scroll = 0
                track = self._scalar_track(tag, self._detail_runs())
                self._cursor = max(0, (len(track) - 1) // 2)
                return f"open {tag}"
            return None
        if name == "close_detail":
            self._detail = None
            return "close detail"
        if name == "open_hparams":
            self._hparams = True
            self._scroll = 0
            return "open hparams"
        if name == "goto_page":
            try:
                p = max(1, int(a.get("page")))
            except (TypeError, ValueError):
                return None
            self._focus = (p - 1) * max(1, self.rows * self.cols)
            return f"page {p}"
        return None

    def _llm_run(self, question: str, on_delta=None):
        """Build messages, call the model (streaming if ``on_delta`` given), apply
        actions. Returns (text, applied, usage). Kept for the unit tests; the live
        chat uses _chat_complete."""
        cfg = self._llm_config
        msgs = _llm.build_messages(self._llm_context(), question, self._llm_history)
        tools = _llm.build_tools()
        if on_delta is not None:
            result = _llm.ask_stream(cfg, msgs, tools, complete=self._llm_complete,
                                     on_delta=on_delta)
        else:
            result = _llm.ask(cfg, msgs, tools, complete=self._llm_complete)
        applied = []
        for nm, ar in result["tool_calls"]:
            desc = self._llm_apply_action(nm, ar)
            if desc:
                applied.append(desc)
        # Remember the turn for follow-ups — include what was done so a later
        # "now zoom into that" / "compare to baseline" has the thread of actions.
        note = result["text"] or ""
        if applied:
            note = (note + "\n" if note else "") + "[did: " + "; ".join(applied) + "]"
        self._llm_history.append({"role": "user", "content": question})
        self._llm_history.append({"role": "assistant", "content": note or "(ok)"})
        self._llm_history = self._llm_history[-8:]
        return result["text"], applied, result.get("usage")

    # -- chat sidebar -------------------------------------------------------

    def _chat_width(self, cols: int) -> int:
        """Width of the right chat pane (the dashboard gets the rest)."""
        return max(26, min(54, cols // 3))

    def _toggle_chat(self, screen, keys) -> None:
        """A: open the chat sidebar (close it with Esc). Runs setup the 1st time."""
        if self._chat_open:
            self._chat_open = False
            return
        if not _llm.is_available():
            self._show_text_overlay(screen, keys, "LLM assistant — not installed", [
                "The chat assistant needs the optional LLM extra:",
                "", "    pip install 'terminalboard[llm]'", "",
                "Then press A again and pick a model + key (e.g. ollama/llama3,",
                "gpt-5.4-nano, anthropic/claude-haiku-4-5).",
            ])
            return
        if self._llm_config is None:
            self._llm_config = _llm.load_config()
        if self._llm_config is None or not self._llm_config.ok():
            if not self._llm_setup(screen, keys):
                return
        self._chat_open = True
        self._status = ""

    def _chat_session(self):
        self._chat_active = max(0, min(self._chat_active, len(self._chat_sessions) - 1))
        return self._chat_sessions[self._chat_active]

    def _dash_sig(self):
        """Everything that affects the dashboard pane (so it can be cached while
        the user just types in the chat)."""
        total = last = 0
        for run in self.reader.runs.values():
            for s in run.series.values():
                total += len(s)
                if s.steps:
                    last = max(last, s.steps[-1])
        return (total, last, self._focus, self._detail_run, self._scroll,
                self._cursor, self._textdiff, self.tag_filter, self.run_filter,
                self._kind_filter, round(self.smooth, 3), self.rows, self.cols,
                self._order_rot, self._detail, self._hparams, self.xaxis,
                self.logy, self._distmode, self._status)

    def _compose_split(self) -> str:
        """Render the dashboard (left, narrowed) beside the chat pane (right).
        The dashboard is cached so typing in the chat only repaints the right.
        In full-screen mode the chat takes the whole screen."""
        cols, rows = shutil.get_terminal_size((100, 30))
        if self._chat_full:
            return "\n".join(self._chat_pane(cols, rows))
        chat_w = self._chat_width(cols)
        dash_w = max(24, cols - chat_w - 1)
        dsig = (dash_w, rows) + self._dash_sig()
        if self._dash_cache and self._dash_cache[0] == dsig:
            left = self._dash_cache[1]
        else:
            self._cols_override = dash_w
            try:
                left = self._dashboard_frame().split("\n")
            finally:
                self._cols_override = None
            left = (left + [""] * rows)[:rows]
            self._dash_cache = (dsig, left)
        chat = self._chat_pane(chat_w, rows)
        bar = "\033[90m│\033[0m"
        out = [f"{_fit(left[r], dash_w)}{bar}{chat[r]}" for r in range(rows)]
        return "\n".join(out)

    def _chat_pane(self, w: int, h: int):
        """The chat column as exactly ``h`` lines of visible width ``w``."""
        sess = self._chat_session()
        scrolled = "  \033[2m↑more\033[0m" if self._chat_scroll > 0 else ""
        title = f"💬 {sess['name']} ({self._chat_active + 1}/{len(self._chat_sessions)})"
        head = _fit("\033[1;35m" + title + "\033[0m" + scrolled, w)
        rule = "\033[90m" + "─" * w + "\033[0m"
        body_h = max(1, h - 4)                       # head + rule + body + hint + input
        lines = self._chat_transcript(w)
        self._chat_scroll = max(0, min(self._chat_scroll, len(lines) - body_h))
        end = max(0, len(lines) - self._chat_scroll)
        start = max(0, end - body_h)
        view = lines[start:end]
        view = view + [""] * (body_h - len(view))
        full = "split" if self._chat_full else "full"
        if w >= 44:
            hint = (f"\033[2mEnter send · ↑/↓ scroll · ^F {full} · Esc close · "
                    "/help\033[0m")
        else:
            hint = f"\033[2m↑/↓ scroll · ^F {full} · Esc · /help\033[0m"
        hint = _fit(hint, w)
        # input line, with a sliding window so the cursor is always visible
        plain = "".join(self._chat_input)
        avail = max(8, w - 2)
        s = 0
        if len(plain) > avail:
            s = max(0, min(self._chat_pos - avail // 2, len(plain) - avail))
        win = plain[s:s + avail]
        wpos = self._chat_pos - s
        lead = "…" if s > 0 else ""
        if wpos < len(win):
            shown = win[:wpos] + "\033[7m" + win[wpos] + "\033[0m" + win[wpos + 1:]
        else:
            shown = win + "\033[7m \033[0m"
        prompt = _fit("\033[1;36m›\033[0m " + lead + shown, w)
        return [head, rule] + [_fit(x, w) for x in view] + [hint, prompt]

    def _md(self, line: str) -> str:
        """Light inline markdown: **bold** and `code`."""
        line = re.sub(r"\*\*(.+?)\*\*", "\033[1m\\1\033[0m", line)
        line = re.sub(r"`([^`]+)`", "\033[36m\\1\033[0m", line)
        return line

    def _chat_greeting(self, w: int):
        out = [_fit("\033[1;35m🤖 assistant\033[0m", w), ""]
        for ln in ("Ask about your runs — I can filter, open, compare, "
                   "analyze, and drive the dashboard.").split("\n"):
            for x in textwrap.wrap(ln, w):
                out.append(_fit("\033[2m" + x + "\033[0m", w))
        out += ["", _fit("\033[2me.g.  which run is overfitting?\033[0m", w),
                _fit("\033[2m      show val losses, smoothed\033[0m", w),
                _fit("\033[2m      open the pr curve and rate it\033[0m", w), ""]
        out.append(_fit("\033[2msessions: /new /next /prev /rename /clear · "
                        "/help\033[0m", w))
        return out

    def _chat_transcript(self, w: int):
        """Wrapped, role-labeled transcript lines for the active session."""
        sess = self._chat_session()
        if not sess["messages"] and self._chat_partial is None:
            return self._chat_greeting(w)
        out = []
        for m in sess["messages"]:
            role = m["role"]
            if role == "user":
                out.append(_fit("\033[1;36myou\033[0m", w))
                body, md = m["content"], False
            elif role == "assistant":
                out.append(_fit("\033[1;35m🤖\033[0m", w))
                body, md = (m.get("text") or m["content"]), True
            else:                                   # note (display-only)
                for ln in textwrap.wrap(m["content"], w) or [""]:
                    out.append(_fit("\033[2m" + ln + "\033[0m", w))
                continue
            out += self._render_body(body, w, md)
            if m.get("applied"):
                out.append(_fit("\033[2m↳ " + " · ".join(m["applied"]) + "\033[0m", w))
            if m.get("meta"):
                out.append(_fit("\033[2m" + m["meta"] + "\033[0m", w))
            out.append("")
        if self._chat_partial is not None:          # streaming in flight
            out.append(_fit("\033[1;35m🤖\033[0m", w))
            out += self._render_body(self._chat_partial or "…", w, md=True)
        return out

    def _render_body(self, body: str, w: int, md: bool):
        out = []
        for para in body.split("\n"):
            bold = False
            p = para
            if md and p.lstrip().startswith("#"):           # heading
                p, bold = p.lstrip("# ").rstrip(), True
            elif md and p.lstrip()[:2] in ("- ", "* "):     # bullet
                p = "• " + p.lstrip()[2:]
            for ln in (textwrap.wrap(p, w) or [""]):
                if md:
                    ln = self._md(ln)
                if bold:
                    ln = "\033[1m" + ln + "\033[0m"
                out.append(_fit(ln, w))
        return out

    def _chat_complete(self, sess, on_delta):
        """One streamed turn for ``sess`` (its last message is the user prompt)."""
        sys_msg = (_llm.SYSTEM_PROMPT + "\n\nCurrent dashboard context:\n"
                   + self._llm_context())
        history = [{"role": m["role"], "content": m["content"]}
                   for m in sess["messages"] if m["role"] in ("user", "assistant")]
        msgs = [{"role": "system", "content": sys_msg}] + history
        result = _llm.ask_stream(self._llm_config, msgs, _llm.build_tools(),
                                 complete=self._llm_complete, on_delta=on_delta)
        applied = []
        for nm, ar in result["tool_calls"]:
            desc = self._llm_apply_action(nm, ar)
            if desc:
                applied.append(desc)
        note = result["text"] or ""
        if applied:
            note = (note + "\n" if note else "") + "[did: " + "; ".join(applied) + "]"
        sess["messages"].append({"role": "assistant", "content": note or "(ok)",
                                 "text": result["text"], "applied": applied})
        return result["text"], applied, result.get("usage")

    def _chat_dispatch(self, screen, keys, question: str) -> None:
        """Send a user message and stream the reply into the active session."""
        sess = self._chat_session()
        sess["messages"].append({"role": "user", "content": question})
        self._chat_drafts.append(question)
        self._chat_draft_idx = len(self._chat_drafts)
        self._chat_scroll = 0
        out: dict = {"partial": ""}
        lock = threading.Lock()

        def on_delta(frag):
            with lock:
                out["partial"] += frag

        def work():
            try:
                out["res"] = self._chat_complete(sess, on_delta)
            except Exception as e:
                out["err"] = e

        t0 = time.monotonic()
        th = threading.Thread(target=work, daemon=True)
        th.start()
        frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        i = 0
        while th.is_alive():
            with lock:
                self._chat_partial = out["partial"] + f"  \033[36m{frames[i % 10]}\033[0m"
            screen.draw(self._build_frame(), hard=True)
            i += 1
            chunk = keys.get(0.1)
            if chunk and any(t == "ESC" for t in self._parse_chunk(chunk)):
                break                               # stop watching (daemon finishes)
        self._chat_partial = None
        if "err" in out:
            sess["messages"].append({"role": "note",
                                     "content": "error: " + _llm.friendly_error(out["err"])})
            return
        if "res" not in out:                        # cancelled mid-stream
            sess["messages"].append({"role": "note", "content": "(cancelled)"})
            return
        _text, applied, usage = out["res"]
        elapsed = time.monotonic() - t0
        bits = []
        us = _llm.usage_summary(usage)
        cost = _llm.estimate_cost(self._llm_config.model, usage)
        if us:
            bits.append(us)
        if cost:
            bits.append(f"${cost:.4f}")
        bits.append(f"{elapsed:.1f}s")
        if sess["messages"] and sess["messages"][-1]["role"] == "assistant":
            sess["messages"][-1]["meta"] = " · ".join(bits)   # feedback in-pane
        self._chat_scroll = 0

    def _chat_command(self, text: str, screen, keys) -> None:
        """Slash commands for session management."""
        parts = text[1:].split(maxsplit=1)
        cmd = (parts[0].lower() if parts else "")
        arg = parts[1].strip() if len(parts) > 1 else ""
        S = self._chat_sessions

        def note(msg):
            self._chat_session()["messages"].append({"role": "note", "content": msg})

        if cmd in ("new", "n"):
            S.append({"name": f"chat {len(S) + 1}", "messages": []})
            self._chat_active = len(S) - 1
            self._chat_scroll = 0
        elif cmd in ("next",):
            self._chat_active = (self._chat_active + 1) % len(S)
            self._chat_scroll = 0
        elif cmd in ("prev", "p"):
            self._chat_active = (self._chat_active - 1) % len(S)
            self._chat_scroll = 0
        elif cmd in ("delete", "del"):
            if len(S) > 1:
                S.pop(self._chat_active)
                self._chat_active = max(0, self._chat_active - 1)
            else:
                S[0]["messages"] = []
            self._chat_scroll = 0
        elif cmd in ("rename",) and arg:
            self._chat_session()["name"] = arg[:24]
        elif cmd in ("clear",):
            self._chat_session()["messages"] = []
            self._chat_scroll = 0
        elif cmd in ("full", "wide"):
            self._chat_full = True
        elif cmd in ("split", "side"):
            self._chat_full = False
        elif cmd in ("model",):
            self._llm_setup(screen, keys)
        elif cmd in ("close", "q"):
            self._chat_open = False
        elif cmd in ("sessions", "ls"):
            note("sessions: " + ", ".join(
                f"{i+1}.{s['name']}" for i, s in enumerate(S)))
        else:
            note("commands: /new  /next  /prev  /delete  /rename <name>  /clear  "
                 "/sessions  /full  /split  /model  /close    "
                 "(↑/↓ scroll · ^F full · Esc close)")

    def _handle_chat_key(self, screen, keys, tok: str):
        """Keys while the chat sidebar is open. Esc CLOSES it (never quits the
        app). Returns 'quit' only on Ctrl-C/D."""
        if tok in ("\x03", "\x04"):
            return "quit"
        if tok == "ESC":                            # close the sidebar
            self._chat_open = False
            return None
        if tok == "\x06":                           # ^F: full-screen ↔ split
            self._chat_full = not self._chat_full
            return None
        if tok in ("\r", "\n"):
            text = "".join(self._chat_input).strip()
            self._chat_input = []
            self._chat_pos = 0
            if not text:
                return None
            if text.startswith("/"):
                self._chat_command(text, screen, keys)
            else:
                self._chat_dispatch(screen, keys, text)
            return None
        buf = self._chat_input
        page = max(1, shutil.get_terminal_size((100, 30))[1] - 6)
        if tok == "LEFT":
            self._chat_pos = max(0, self._chat_pos - 1)
        elif tok == "RIGHT":
            self._chat_pos = min(len(buf), self._chat_pos + 1)
        elif tok in ("HOME", "\x01"):               # Home / ^A
            self._chat_pos = 0
        elif tok in ("END", "\x05"):                # End / ^E
            self._chat_pos = len(buf)
        elif tok in ("WORD-LEFT",):
            self._chat_pos = _prev_word(buf, self._chat_pos)
        elif tok in ("WORD-RIGHT",):
            self._chat_pos = _next_word(buf, self._chat_pos)
        elif tok in ("\x7f", "\b", "\x08"):         # backspace
            if self._chat_pos > 0:
                del buf[self._chat_pos - 1]
                self._chat_pos -= 1
        elif tok == "DEL":
            if self._chat_pos < len(buf):
                del buf[self._chat_pos]
        elif tok in ("\x17", "WORD-DEL-BACK"):      # ^W: delete word back
            start = _prev_word(buf, self._chat_pos)
            del buf[start:self._chat_pos]
            self._chat_pos = start
        elif tok == "WORD-DEL-FWD":                 # Alt-d: delete word fwd
            del buf[self._chat_pos:_next_word(buf, self._chat_pos)]
        elif tok == "\x15":                         # ^U: clear the line
            buf.clear()
            self._chat_pos = 0
        elif tok == "\x0b":                         # ^K: kill to end
            del buf[self._chat_pos:]
        elif tok == "UP":                           # scroll the transcript up
            self._chat_scroll += 1
        elif tok == "DOWN":
            self._chat_scroll = max(0, self._chat_scroll - 1)
        elif tok == "PGUP":
            self._chat_scroll += page
        elif tok == "PGDN":
            self._chat_scroll = max(0, self._chat_scroll - page)
        elif tok == "\x10":                         # ^P: recall previous message
            if self._chat_drafts and self._chat_draft_idx > 0:
                self._chat_draft_idx -= 1
                self._chat_input = list(self._chat_drafts[self._chat_draft_idx])
                self._chat_pos = len(self._chat_input)
        elif tok == "\x0e":                         # ^N: next message / clear
            if self._chat_draft_idx < len(self._chat_drafts) - 1:
                self._chat_draft_idx += 1
                self._chat_input = list(self._chat_drafts[self._chat_draft_idx])
                self._chat_pos = len(self._chat_input)
            else:
                self._chat_draft_idx = len(self._chat_drafts)
                self._chat_input = []
                self._chat_pos = 0
        elif isinstance(tok, str) and tok.isprintable():
            for c in tok:
                buf.insert(self._chat_pos, c)
                self._chat_pos += 1
        return None

    @staticmethod
    def _crop(frame: str, rows: int) -> str:
        # Hard safety crop: never exceed the terminal height (line wrap is off,
        # so width takes care of itself).
        lines = frame.split("\n")
        return "\n".join(lines[:rows])

    def _with_status(self, footer: str) -> str:
        # Status goes at the FRONT so it's never clipped off the right edge of a
        # narrow terminal (line-wrap is off; the static key-hints can truncate).
        if self._status:
            return f"\033[1;32m{self._status}\033[0m   {footer}"
        return footer

    def _current_tag(self) -> Optional[str]:
        if self._detail is not None:
            return self._detail
        tags = self._matching_tags()
        return tags[self._focus] if tags and self._focus < len(tags) else None

    def _csv_default_path(self, tag: str) -> str:
        """Default save path: <csv_dir>/<sanitized-tag>.csv (csv_dir from config)."""
        import os
        name = tag.strip("/").replace("/", "_") + ".csv"
        base = os.path.expanduser(self._csv_dir) if self._csv_dir else ""
        return os.path.join(base, name) if base else name

    def _do_csv(self, screen, keys) -> None:
        """Prompt for a path (pre-filled from config) and export the focused tag."""
        tag = self._current_tag()
        if not tag:
            self._status = "nothing to export"
            return
        path = self._input_prompt(screen, keys, "save CSV",
                                  self._csv_default_path(tag))
        self._status = "" if path is None else self._export_csv(path)

    def _export_csv(self, path: Optional[str] = None) -> str:
        """Write the focused/detail scalar tag to ``path`` (default if None)."""
        import csv
        import os
        tag = self._current_tag()
        if not tag:
            return "nothing to export"
        runs = self._visible_runs()
        names = [n for n in sorted(runs)
                 if tag in runs[n].series and runs[n].series[tag].kind == "scalar"]
        if not names:
            return f"'{tag}' is not a scalar — CSV skipped"
        lut = {n: dict(zip(runs[n].series[tag].steps, runs[n].series[tag].values))
               for n in names}
        steps = sorted({s for n in names for s in lut[n]})
        path = os.path.expanduser(path or self._csv_default_path(tag))
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["step"] + names)
                for s in steps:
                    w.writerow([s] + [lut[n].get(s, "") for n in names])
        except OSError as e:
            return f"export failed: {e}"
        return f"✓ wrote {path} ({len(steps)} rows)"

    def _input_prompt(self, screen, keys, label, initial):
        """Modal single-line text input (returns the string, or None on Esc)."""
        buf = list(initial or "")
        pos = len(buf)
        pending = []

        def next_key():
            nonlocal pending
            if not pending:
                chunk = keys.get(30)
                if chunk is None:
                    return None
                if chunk == "\x1b":
                    more = keys.get(0.03)
                    if more:
                        chunk += more
                pending = self._parse_chunk(chunk)
            return pending.pop(0) if pending else None

        while True:
            screen.draw(self._build_frame(
                prompt=(label, "".join(buf), pos, "input", False)), hard=True)
            key = next_key()
            if key is None:
                continue
            if key in ("\r", "\n"):
                return "".join(buf).strip()
            if key in ("\x03", "\x04", "ESC"):
                return None
            if key in ("\x7f", "\b", "\x08"):
                if pos > 0:
                    del buf[pos - 1]
                    pos -= 1
            elif key == "DEL":
                if pos < len(buf):
                    del buf[pos]
            elif key == "LEFT":
                pos = max(0, pos - 1)
            elif key == "RIGHT":
                pos = min(len(buf), pos + 1)
            elif key in ("HOME", "\x01"):
                pos = 0
            elif key in ("END", "\x05"):
                pos = len(buf)
            elif key == "\x15":
                buf, pos = [], 0
            elif key == "\x0b":
                del buf[pos:]
            elif key == "WORD-LEFT":
                pos = _prev_word(buf, pos)
            elif key == "WORD-RIGHT":
                pos = _next_word(buf, pos)
            elif key in ("WORD-DEL-BACK", "\x17"):
                start = _prev_word(buf, pos)
                del buf[start:pos]
                pos = start
            elif key == "WORD-DEL-FWD":
                del buf[pos:_next_word(buf, pos)]
            elif isinstance(key, str) and key.isprintable():
                for c in key:
                    buf.insert(pos, c)
                    pos += 1

    # -- detail (drill-down) view -------------------------------------------

    def _detail_runs(self):
        """Runs (sorted) that have the detail tag."""
        runs = self._visible_runs()
        return [n for n in sorted(runs) if self._detail in runs[n].series]

    def _build_detail_frame(self) -> str:
        cols, rows = self._termsize()
        tag = self._detail
        runs = self._visible_runs()
        names = self._detail_runs()
        if not names:                       # tag vanished (filter/data) — bail out
            self._detail = None
            return self._build_frame()
        kind = runs[names[0]].series[tag].kind
        body_h = max(2, rows - 2)

        if kind == "text":
            if self._textdiff and len(names) > 1:
                header, body = self._text_diff_detail(tag, names, cols, body_h)
                footer = "\033[2m↑/↓ scroll · d full text · Esc back\033[0m"
            else:
                sel = names[self._detail_run % len(names)]
                header, body = self._text_detail(tag, sel, len(names), cols, body_h)
                diff = " · d diff" if len(names) > 1 else ""
                footer = ("\033[2m↑/↓ scroll · PgUp/PgDn · ←/→ switch exp"
                          f"{diff} · Esc back\033[0m")
        elif kind == "scalar":
            return self._scalar_detail(tag, names, cols, rows, body_h)
        elif kind == "pr_curve":
            track = self._scalar_track(tag, names)           # union of pr steps
            self._cursor = max(0, min(self._cursor, max(0, len(track) - 1)))
            step = track[self._cursor] if track else None
            header = (f"\033[1m{tag}\033[0m  step {step}  "
                      f"({self._cursor + 1}/{max(1, len(track))})  "
                      f"exps={len(names)}  kind={kind}")
            body = self.renderer.detail_prcurve(
                runs, tag, order=self._run_order(), run_color=self._run_colors(),
                w=cols, h=body_h, step=step)
            footer = "\033[2m←/→ step · Home/End · Esc back\033[0m"
        else:                                            # histogram
            sel = names[self._detail_run % len(names)]
            order = [n for n in names if n != sel] + [sel]   # selected on top
            view = "distribution" if self._distmode else "histogram"
            header = (f"\033[1m{tag}\033[0m  [{sel}]  "
                      f"exp {self._detail_run % len(names) + 1}/{len(names)}  "
                      f"{view}")
            body = self.renderer.frame(
                runs, [tag], smooth=self.smooth, max_cols=1,
                width=cols, height=body_h, run_colors=self._run_colors(),
                run_order=order, hist_mode=("dist" if self._distmode else "heatmap"),
            )
            switch = "←/→ switch exp · " if len(names) > 1 else ""
            footer = f"\033[2m{switch}b {view}↔ · Esc back\033[0m"
        return self._crop(f"{header}\n{body}\n{self._with_status(footer)}", rows)

    # -- scalar detail with a TensorBoard-style x-cursor + readout -----------

    @staticmethod
    def _fmt_reltime(secs: float) -> str:
        secs = max(0, int(secs))
        if secs < 60:
            return f"+{secs}s"
        m, s = divmod(secs, 60)
        if m < 60:
            return f"+{m}m{s:02d}s"
        h, m = divmod(m, 60)
        return f"+{h}h{m:02d}m"

    def _scalar_track(self, tag, names) -> List[int]:
        """Cursor stops = the UNION of every visible run's steps, so ←/→ can reach
        the last point among all experiments (not just one run's last step)."""
        runs = self._visible_runs()
        steps = set()
        for n in names:
            steps.update(runs[n].series[tag].steps)
        return sorted(steps)

    def _cursor_time(self, tag, names, cstep) -> float:
        """Relative wall-time for the cursor on the time axis: take the
        furthest-reaching run's time at the step nearest ``cstep``."""
        runs = self._visible_runs()
        best = None                      # (reach, reltime)
        for n in names:
            s = runs[n].series[tag]
            if not s.steps or not s.wall_times:
                continue
            i = self._nearest_index(s.steps, cstep)
            if i >= len(s.wall_times):
                continue
            reach, reltime = s.steps[-1], s.wall_times[i] - s.wall_times[0]
            if best is None or reach > best[0]:
                best = (reach, reltime)
        return best[1] if best else cstep

    def _nearest_index(self, steps, target) -> int:
        i = bisect.bisect_left(steps, target)
        if i >= len(steps):
            return len(steps) - 1
        if i > 0 and abs(steps[i - 1] - target) <= abs(steps[i] - target):
            return i - 1
        return i

    def _scalar_detail(self, tag, names, cols, rows, body_h) -> str:
        from .render import ema, _RUN_STYLES
        runs = self._visible_runs()
        rc = self._run_colors()
        track = self._scalar_track(tag, names)
        if not track:
            return self._crop(f"\033[1m{tag}\033[0m\n  (no data)\n"
                              "\033[2mEsc back\033[0m", rows)
        self._cursor = max(0, min(self._cursor, len(track) - 1))
        cstep = track[self._cursor]

        # Per-run readout at the cursor step (full run names, aligned).
        readout: List[str] = []
        shown = names[:8]
        namew = max((len(n) for n in shown), default=4)
        for n in shown:
            s = runs[n].series[tag]
            if not s.steps:
                continue
            i = self._nearest_index(s.steps, cstep)
            val = s.values[i]
            sm = ema(s.values, self.smooth)[i] if self.smooth > 0 else val
            rt = ""
            if s.wall_times and i < len(s.wall_times):
                rt = "  t " + self._fmt_reltime(s.wall_times[i] - s.wall_times[0])
            code = _RUN_STYLES[rc.get(n, 0) % len(_RUN_STYLES)][1]
            readout.append(
                f"\033[{code}m●\033[0m {n:<{namew}}  step {s.steps[i]:>8}  "
                f"value {val:< 12.5g} smoothed {sm:< 12.5g}{rt}"
            )

        # Cursor x in the active axis domain (so the vertical line lands right).
        cx = self._cursor_time(tag, names, cstep) if self.xaxis == "time" else cstep

        plot_h = max(2, body_h - len(readout))
        plot = self.renderer.detail_scalar(
            runs, tag, order=self._run_order(), run_color=rc,
            w=cols, h=plot_h, smooth=self.smooth, cursor_x=cx,
            xaxis=self.xaxis, logy=self.logy,
        )
        axes = f"x={self.xaxis} y={'log' if self.logy else 'lin'}"
        header = (f"\033[1m{tag}\033[0m  cursor @ step {cstep}  "
                  f"({self._cursor + 1}/{len(track)})  exps={len(names)}  {axes}")
        footer = self._with_status(
            "\033[2m←/→ cursor · Shift+←/→ fast · Home/End · "
            "+/- smooth · x axis · l log · w csv · Esc back\033[0m")
        return self._crop("\n".join([header, plot] + readout + [footer]), rows)

    def _scroll_view(self, lines, w, h):
        """Clamp self._scroll and return (h fitted lines, total)."""
        total = len(lines)
        self._scroll = max(0, min(self._scroll, max(0, total - h)))
        view = [l for l in lines[self._scroll:self._scroll + h]]
        view += [""] * (h - len(view))
        return view, total

    def _text_detail(self, tag, run_name, n_runs, w, h):
        series = self._visible_runs()[run_name].series[tag]
        text = series.texts[-1] if series.texts else ""
        wrapped: List[str] = []
        for para in text.split("\n"):
            wrapped.extend(textwrap.wrap(para, w) or [""])
        view, total = self._scroll_view(wrapped, w, h)
        idx = self._detail_run % max(1, n_runs)
        header = (f"\033[1m{tag}\033[0m  [{run_name}]  exp {idx + 1}/{n_runs}  "
                  f"lines {self._scroll + 1}–{min(total, self._scroll + h)}/{total}")
        return header, "\n".join(view)

    @staticmethod
    def _parse_kv(text):
        """Pull key→value pairs from config-ish text (JSON / `k: v` / `k = v`)."""
        d = {}
        for line in text.splitlines():
            line = line.strip().rstrip(",")
            m = re.match(r'^"?([\w./\- ]+?)"?\s*[:=]\s*(.+)$', line)
            if m:
                d[m.group(1).strip()] = m.group(2).strip()
        return d

    def _text_diff_detail(self, tag, names, w, h):
        from .render import _RUN_STYLES
        runs = self._visible_runs()
        rc = self._run_colors()
        parsed = {n: self._parse_kv(runs[n].series[tag].texts[-1]
                                    if runs[n].series[tag].texts else "")
                  for n in names}
        keys = sorted({k for d in parsed.values() for k in d})
        lines: List[str] = []
        for k in keys:
            vals = [parsed[n].get(k, "—") for n in names]
            if len(set(vals)) <= 1:
                continue                                # identical → not a diff
            lines.append(f"\033[1m{k}\033[0m")
            for n in names:
                code = _RUN_STYLES[rc.get(n, 0) % len(_RUN_STYLES)][1]
                lines.append(f"  \033[{code}m●\033[0m {n[:18]:<18} "
                             f"{parsed[n].get(k, '—')}")
        if not lines:
            lines = ["(no differing keys — configs are identical, or not key:value)"]
        view, total = self._scroll_view(lines, w, h)
        header = (f"\033[1m{tag}\033[0m  diff across {len(names)} experiments  "
                  f"lines {self._scroll + 1}–{min(total, self._scroll + h)}/{total}")
        return header, "\n".join(view)

    def _view_sig(self):
        """Layout-level state — a change here triggers a *hard* clear so a new
        page/grid/detail can never leave residue. Excludes scroll / exp-switch /
        in-page focus moves, which repaint softly (no flash)."""
        per_page = max(1, self.rows * self.cols)
        page = self._focus // per_page
        return (page, round(self.smooth, 3), self.rows, self.cols,
                self.tag_filter, self.run_filter, self.renderer.name,
                self._order_rot, self._detail, self.xaxis, self.logy,
                self._distmode, self._kind_filter, self._hparams,
                self._chat_open, self._chat_full,
                shutil.get_terminal_size((100, 30)))

    def _chat_sig(self):
        sess = self._chat_sessions[self._chat_active] if self._chat_open else None
        return (self._chat_active, "".join(self._chat_input), self._chat_pos,
                self._chat_scroll, len(sess["messages"]) if sess else 0,
                len(self._chat_sessions))

    def _signature(self):
        """Everything that affects the frame — a change triggers a (soft or hard)
        repaint; no change means no repaint, so an idle dashboard never flickers."""
        total = 0
        last_step = 0
        for run in self.reader.runs.values():
            for s in run.series.values():
                total += len(s)
                if s.steps:
                    last_step = max(last_step, s.steps[-1])
        return (total, last_step, self._focus, self._detail_run, self._scroll,
                self._cursor, self._textdiff, self._status,
                self._chat_sig()) + self._view_sig()

    def render_once(self) -> None:
        self.reader.poll()
        print(self._build_frame())

    # -- interactive loop ----------------------------------------------------

    def run(self, *, once: bool = False) -> None:
        if once:
            self.render_once()
            return

        try:
            self._run_loop()
        finally:
            if self._restore:
                self._save_view()

    def _run_loop(self) -> None:
        with Screen() as screen, KeyReader() as keys:
            last_sig = None
            last_view = None
            while True:
                self.reader.poll()
                sig = self._signature()
                if sig != last_sig:
                    view = self._view_sig()
                    # Hard-clear on a layout change; soft in-place on data-only.
                    screen.draw(self._build_frame(), hard=(view != last_view))
                    last_sig, last_view = sig, view

                # Wait out the interval, but react instantly to keypresses.
                deadline = time.monotonic() + self.interval
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    chunk = keys.get(remaining)
                    if chunk is None:
                        break
                    for tok in self._parse_chunk(chunk):
                        if self._chat_open:
                            if self._handle_chat_key(screen, keys, tok) == "quit":
                                return
                        elif self._hparams:
                            if self._handle_hparams_key(screen, keys, tok) == "quit":
                                return
                        elif self._detail is not None:
                            if self._handle_detail_key(screen, keys, tok) == "quit":
                                return
                        elif self._handle_grid_key(screen, keys, tok):
                            return  # quit
                    # A keypress may have changed the view: repaint now, hard-
                    # clearing if the layout changed so no stale plots remain.
                    view = self._view_sig()
                    screen.draw(self._build_frame(), hard=(view != last_view))
                    last_sig, last_view = self._signature(), view
                    deadline = time.monotonic() + self.interval

    def _edit_filter(self, screen, keys, kind: str) -> None:
        """Modal line-editor for a tag or experiment filter.

        Live preview (re-filters as you type) — but if a pattern matches nothing
        the layout is *kept* and a red warning is shown, instead of collapsing to
        an empty screen and yanking the input box to the top. ←/→ move the
        cursor, ↑/↓ recall previous patterns.
        """
        attr = "tag_filter" if kind == "tags" else "run_filter"
        label = "filter tags" if kind == "tags" else "filter experiments"
        original = getattr(self, attr)
        last_valid = original
        buf = list(original or "")
        pos = len(buf)
        hist = self._filter_history[kind]
        # The current input normally lives in a "draft" slot just past the newest
        # entry. But if it already equals an existing history entry (the usual
        # case — the editor pre-fills with the active filter, which is in the
        # history), start *on* that entry: then Up goes to the previous filter
        # (not a redundant repeat of what's shown) and Down to the next, with no
        # duplicate "extra" empty slot.
        cur = "".join(buf)
        if cur and cur in hist:
            hist_idx = hist.index(cur)
            has_draft = False
            draft = None
        else:
            hist_idx = len(hist)   # the live-draft slot
            has_draft = True
            draft = list(buf)
        pending = []           # tokens decoded from one read, drained one by one

        def next_key():
            nonlocal pending
            if not pending:
                chunk = keys.get(30)
                if chunk is None:
                    return None
                if chunk == "\x1b":         # maybe a split escape sequence
                    more = keys.get(0.03)
                    if more:
                        chunk += more
                pending = self._parse_chunk(chunk)
            return pending.pop(0) if pending else None

        while True:
            typed = "".join(buf).strip() or None
            warn = typed is not None and self._count_matches(kind, typed) == 0
            if not warn:
                setattr(self, attr, typed)  # commit -> layout updates live
                last_valid = typed
                self._focus = 0
            # When warn: keep the last valid filter committed (layout frozen).
            screen.draw(
                self._build_frame(prompt=(label, "".join(buf), pos, kind, warn)),
                hard=True,
            )

            key = next_key()
            if key is None:
                continue
            if key in ("\r", "\n"):                       # apply last valid
                setattr(self, attr, last_valid)
                if last_valid:                            # unique, most-recent-last
                    if last_valid in hist:
                        hist.remove(last_valid)
                    hist.append(last_valid)
                return
            if key == "ESC":                              # cancel: restore
                setattr(self, attr, original)
                self._focus = 0
                return
            if key in ("\x7f", "\b", "\x08"):             # backspace
                if pos > 0:
                    del buf[pos - 1]
                    pos -= 1
            elif key == "DEL":
                if pos < len(buf):
                    del buf[pos]
            elif key == "LEFT":
                pos = max(0, pos - 1)
            elif key == "RIGHT":
                pos = min(len(buf), pos + 1)
            elif key in ("HOME", "\x01"):                 # Home / Ctrl-A
                pos = 0
            elif key in ("END", "\x05"):                  # End / Ctrl-E
                pos = len(buf)
            elif key == "\x15":                           # Ctrl-U: clear line
                buf, pos = [], 0
            elif key == "\x0b":                           # Ctrl-K: kill to end
                del buf[pos:]
            elif key == "WORD-LEFT":
                pos = _prev_word(buf, pos)
            elif key == "WORD-RIGHT":
                pos = _next_word(buf, pos)
            elif key in ("WORD-DEL-BACK", "\x17"):        # Ctrl-W / Alt-Backspace
                start = _prev_word(buf, pos)
                del buf[start:pos]
                pos = start
            elif key == "WORD-DEL-FWD":                   # Alt-d
                del buf[pos:_next_word(buf, pos)]
            elif key == "UP":
                if hist_idx > 0:
                    if hist_idx == len(hist):
                        draft = list(buf)                 # stash the live draft
                    hist_idx -= 1
                    buf, pos = list(hist[hist_idx]), len(hist[hist_idx])
            elif key == "DOWN":
                # Don't descend into the draft slot when there isn't one (the
                # input started on the newest entry) — that's the "extra
                # position" bug. With a draft, Down can return to it.
                top = len(hist) if has_draft else len(hist) - 1
                if hist_idx < top:
                    hist_idx += 1
                    nxt = hist[hist_idx] if hist_idx < len(hist) else (draft or [])
                    buf, pos = list(nxt), len(nxt)
            elif isinstance(key, str) and key.isprintable():
                # May be several chars at once (fast typing or a paste).
                for c in key:
                    buf.insert(pos, c)
                    pos += 1

    def _handle_grid_key(self, screen, keys, tok: str) -> bool:
        """Handle a key in the grid (overview). Return True to quit."""
        if tok != "w":
            self._status = ""
        per_page = max(1, self.rows * self.cols)
        n = len(self._matching_tags())
        last = max(0, n - 1)
        if tok == "w":
            self._do_csv(screen, keys)
        elif tok in ("f", "t"):
            self._edit_filter(screen, keys, "tags" if tok == "t" else "runs")
        elif tok in ("H", "?"):
            self._show_help(screen, keys)
        elif tok == "P":                                # HParams table mode
            self._hparams = True
            self._scroll = 0
        elif tok in ("a", "A"):                         # chat assistant
            self._toggle_chat(screen, keys)
        elif tok in ("q", "Q", "\x03", "\x04", "ESC"):
            return True
        elif tok in ("\r", "\n"):                       # inspect focused panel
            if n:
                self._detail = self._matching_tags()[min(self._focus, last)]
                self._detail_run = 0
                self._scroll = 0
                # Start the x-cursor in the MIDDLE so it's obvious it can move
                # both ways (a cursor parked at the far right looks static).
                names = self._detail_runs()
                track = self._scalar_track(self._detail, names) if names else []
                self._cursor = max(0, (len(track) - 1) // 2)
        elif tok == "LEFT":
            self._focus = max(0, self._focus - 1)
        elif tok == "RIGHT":
            self._focus = min(last, self._focus + 1)
        elif tok == "UP":
            self._focus = max(0, self._focus - self.cols)
        elif tok == "DOWN":
            self._focus = min(last, self._focus + self.cols)
        elif tok in ("n", " ", "j"):                    # page down
            self._focus = min(last, self._focus + per_page)
        elif tok in ("p", "k"):                         # page up
            self._focus = max(0, self._focus - per_page)
        elif len(tok) == 1:
            self._handle_view_key(tok)
        return False

    def _handle_view_key(self, ch: str) -> None:
        """Smoothing / zoom / z-order — shared view options."""
        if ch in ("+", "="):
            self.smooth = min(0.99, round(self.smooth + 0.05, 2))
        elif ch == "-":
            self.smooth = max(0.0, round(self.smooth - 0.05, 2))
        elif ch == "0":
            self.smooth = 0.0
        elif ch == "z":
            self._zoom = min(len(_ZOOM_LADDER) - 1, self._zoom + 1)
            self.rows, self.cols = _ZOOM_LADDER[self._zoom]
        elif ch == "Z":
            self._zoom = max(0, self._zoom - 1)
            self.rows, self.cols = _ZOOM_LADDER[self._zoom]
        elif ch == "o":
            self._order_rot += 1
        elif ch == "l":
            self.logy = not self.logy
        elif ch == "x":
            self.xaxis = "time" if self.xaxis == "step" else "step"
        elif ch == "b":                              # histograms ↔ distributions
            self._distmode = not self._distmode
        elif ch == "c":                              # cycle the type selector
            i = _KIND_CYCLE.index(self._kind_filter) if self._kind_filter \
                in _KIND_CYCLE else 0
            self._kind_filter = _KIND_CYCLE[(i + 1) % len(_KIND_CYCLE)]
            self._focus = 0

    def _handle_detail_key(self, screen, keys, tok: str):
        """Handle a key in the detail (drill-down) view.

        Returns 'quit' only on Ctrl-C/Ctrl-D; Esc just goes *back* to the grid
        (press Esc again there to quit). 'q' does nothing here."""
        if tok in ("\x03", "\x04"):                     # Ctrl-C / Ctrl-D
            return "quit"
        if tok != "w":
            self._status = ""
        if tok in ("ESC", "\r", "\n"):                  # back to grid
            self._detail = None
            self._scroll = 0
            return None
        if tok == "w":
            self._do_csv(screen, keys)
            return None
        if tok in ("a", "A"):                           # chat about the open tag
            self._toggle_chat(screen, keys)
            return None
        names = self._detail_runs()
        if not names:
            return None
        kind = self._visible_runs()[names[0]].series[self._detail].kind
        nruns = len(names)

        if kind in ("scalar", "pr_curve"):              # ←/→ move the x-cursor
            steps = max(1, len(self._scalar_track(self._detail, names)))
            fast = max(2, steps // 25)                   # Shift/Pg jump amount
            if tok == "LEFT":
                self._cursor -= 1
            elif tok == "RIGHT":
                self._cursor += 1
            elif tok in ("WORD-LEFT", "PGUP"):           # Shift+← / PgUp: fast
                self._cursor -= fast
            elif tok in ("WORD-RIGHT", "PGDN"):          # Shift+→ / PgDn: fast
                self._cursor += fast
            elif tok == "HOME":
                self._cursor = 0
            elif tok == "END":
                self._cursor = steps - 1
            elif len(tok) == 1:
                self._handle_view_key(tok)
            self._cursor = max(0, min(self._cursor, steps - 1))
        elif kind == "text":                            # ↑/↓ scroll, ←/→ switch exp
            if tok == "UP":
                self._scroll = max(0, self._scroll - 1)
            elif tok == "DOWN":
                self._scroll += 1
            elif tok == "PGUP":
                self._scroll = max(0, self._scroll - 10)
            elif tok == "PGDN":
                self._scroll += 10
            elif tok == "HOME":
                self._scroll = 0
            elif tok == "END":
                self._scroll = 10 ** 9
            elif tok == "LEFT":
                self._detail_run = (self._detail_run - 1) % nruns
                self._scroll = 0
            elif tok == "RIGHT":
                self._detail_run = (self._detail_run + 1) % nruns
                self._scroll = 0
            elif tok == "d":                            # toggle config-diff
                self._textdiff = not self._textdiff
                self._scroll = 0
            elif len(tok) == 1:
                self._handle_view_key(tok)
        else:                                           # histogram: ←/→ switch exp
            if tok == "LEFT":
                self._detail_run = (self._detail_run - 1) % nruns
            elif tok == "RIGHT":
                self._detail_run = (self._detail_run + 1) % nruns
            elif len(tok) == 1:
                self._handle_view_key(tok)
        return None

    def _handle_hparams_key(self, screen, keys, tok: str):
        """Keys in the HParams table view. Returns 'quit' on q/Ctrl-C/D."""
        if tok in ("\x03", "\x04", "q", "Q"):
            return "quit"
        self._status = ""
        if tok in ("ESC", "P", "\r", "\n"):             # back to grid
            self._hparams = False
            self._scroll = 0
            return None
        if tok in ("a", "A"):                           # chat about the table
            self._toggle_chat(screen, keys)
            return None
        page = max(1, shutil.get_terminal_size((100, 30))[1] - 4)
        if tok == "UP":
            self._scroll -= 1
        elif tok == "DOWN":
            self._scroll += 1
        elif tok == "PGUP":
            self._scroll -= page
        elif tok == "PGDN":
            self._scroll += page
        elif tok == "HOME":
            self._scroll = 0
        elif tok == "END":
            self._scroll = 10 ** 9
        self._scroll = max(0, min(self._scroll, max(0, self._hparams_total - 1)))
        return None

    # -- help overlay --------------------------------------------------------

    def _help_lines(self) -> List[str]:
        """Single-column help: one binding per line, key → action, with color."""
        BOLD, DIM, RST = "\033[1m", "\033[2m", "\033[0m"
        HDR = "\033[1;33m"          # bold yellow section header
        KEY = "\033[1;36m"          # bold cyan key
        KW = 15                     # key column width

        def hdr(t):
            return f"{HDR}{t}{RST}"

        def row(k, d):
            return f"  {KEY}{k:<{KW}}{RST} {d}"

        def note(t):
            return f"  {DIM}{t}{RST}"

        L: List[str] = []
        L.append(f"{BOLD}terminalboard{RST} {DIM}— keyboard help{RST}")
        L.append("")
        L.append(hdr("Navigation"))
        L += [
            row("←/↑/↓/→", "move focus between panels"),
            row("Enter", "inspect focused panel (full screen)"),
            row("n / Space / j", "next page"),
            row("p / k", "previous page"),
            row("c", "type selector: all → scalars → histograms → …"),
            row("z / Z", "zoom out / in (panels per page)"),
            row("o", "cycle curve order (which run is on top)"),
            row("b", "histograms ↔ distribution bands"),
            row("x", "x-axis: step ↔ time"),
            row("l", "toggle log-scale Y"),
            row("w", "export focused scalar to CSV"),
            row("P", "HParams table (runs × hyperparams × metrics)"),
            row("a", "quick one-shot ask (overlay answer)"),
            row("A", "chat sidebar (multi-session; Tab to focus/leave)"),
            row("r", "refresh now"),
            row("q / Esc", "quit"),
        ]
        L.append("")
        L.append(hdr("Detail view — curve") + DIM + "  (after Enter)" + RST)
        L += [
            row("←/→", "move cursor (readout: value / step / time)"),
            row("Shift+←/→", "move cursor faster"),
            row("Home / End", "jump to first / last point"),
        ]
        L.append(hdr("Detail view — text"))
        L += [
            row("↑/↓  PgUp/PgDn", "scroll"),
            row("←/→", "switch experiment"),
            row("d", "config diff (only the keys that differ)"),
        ]
        L.append(hdr("Detail view — histogram"))
        L += [row("←/→", "switch experiment"), row("b", "heatmap ↔ distribution")]
        L.append(hdr("Detail view — pr-curve"))
        L += [row("←/→", "step through training (Home/End)")]
        L.append(row("Esc", "back to grid (Esc again to quit)"))
        L.append("")
        L.append(hdr("Smoothing"))
        L += [
            row("+ / =", "more"),
            row("-", "less"),
            row("0", "off"),
        ]
        L.append("")
        L.append(hdr("Filtering"))
        L += [
            row("t", "edit tag filter"),
            row("f", "edit experiment filter"),
        ]
        L.append(note("in the filter prompt:"))
        L += [
            row("←/→", "move cursor"),
            row("↑/↓", "recall previous patterns"),
            row("Home/End", "line start / end  (also ^A / ^E)"),
            row("^W / ^K / ^U", "delete word / kill-to-end / clear"),
            row("Alt/Ctrl+←/→", "move by word"),
            row("Enter / Esc", "apply / cancel"),
        ]
        L.append("")
        L.append(hdr("Filter syntax") + DIM + "  (tags and experiments)" + RST)
        L += [
            row("word", "case-insensitive substring  (loss → train/loss)"),
            row("a b", "AND  (both must match)"),
            row("a | b , c", "OR  (| or , separate alternatives)"),
            row("* ? [ ]", "glob wildcards  (train/*loss*)"),
            row("!word", "NOT  (exclude)"),
            row("/regex/", "regex; wrap the WHOLE filter for | or spaces"),
        ]
        L.append("")
        L.append(hdr("Chat assistant") + DIM + "  (a / A — needs the [llm] extra)"
                 + RST)
        L += [
            row("a / A", "open the chat sidebar (Esc closes it)"),
            row("Enter", "send the message"),
            row("↑/↓  PgUp/PgDn", "scroll the transcript"),
            row("^F", "full-screen chat ↔ split (or /full · /split)"),
            row("^P / ^N", "recall previous / next message"),
            row("^W/^U/^A/^E", "delete word / clear / start / end"),
        ]
        L.append(note("it sees the live view (focused/visible tags, counts) and "
                      "all log data,"))
        L.append(note("answers + drives the dashboard, and keeps multiple sessions:"))
        L.append(note("  /new /next /prev /delete /rename <n> /clear /sessions "
                      "/model /close"))
        L.append("")
        L.append(hdr("Plot types"))
        L.append(note("scalars · text · histograms (heatmap/distribution) ·"))
        L.append(note("pr-curves · hparams table"))
        L.append("")
        L.append(hdr("View state"))
        L.append(note("filters, zoom, smoothing, axis, order and focus are saved"))
        L.append(note("per-logdir and restored next time  (start fresh: --reset-view)"))
        return L

    def _show_help(self, screen, keys) -> None:
        cols, rows = shutil.get_terminal_size((100, 30))
        lines = self._help_lines()
        body_h = max(1, rows - 1)              # reserve a row for the footer
        maxscroll = max(0, len(lines) - body_h)
        scroll = 0
        while True:
            view = lines[scroll:scroll + body_h]
            view += [""] * (body_h - len(view))
            if maxscroll:
                pos = f"{scroll + 1}-{min(len(lines), scroll + body_h)}/{len(lines)}"
                foot = (f"\033[2m  ↑/↓ PgUp/PgDn scroll · "
                        f"any other key to return   ({pos})\033[0m")
            else:
                foot = "\033[2m  press any key to return…\033[0m"
            screen.draw("\n".join(view + [foot]), hard=True)
            chunk = keys.get(30)
            if chunk is None:
                continue
            if not maxscroll:                  # everything fits → any key returns
                return
            done = False
            for tok in self._parse_chunk(chunk):
                if tok == "UP":
                    scroll -= 1
                elif tok == "DOWN":
                    scroll += 1
                elif tok == "PGUP":
                    scroll -= body_h
                elif tok == "PGDN":
                    scroll += body_h
                elif tok == "HOME":
                    scroll = 0
                elif tok == "END":
                    scroll = maxscroll
                else:                          # any non-scroll key returns
                    done = True
                    break
                scroll = max(0, min(scroll, maxscroll))
            if done:
                return

    # -- text overlay (help / not-installed notices) ------------------------

    def _show_text_overlay(self, screen, keys, title: str, lines: List[str]) -> None:
        cols, rows = shutil.get_terminal_size((100, 30))
        body = [f"\033[1m{title}\033[0m", ""] + lines
        body_h = max(1, rows - 1)
        maxscroll = max(0, len(body) - body_h)
        scroll = 0
        while True:
            view = body[scroll:scroll + body_h]
            view += [""] * (body_h - len(view))
            foot = "\033[2m  ↑/↓ PgUp/PgDn scroll · any other key to close\033[0m"
            screen.draw("\n".join(view + [foot]), hard=True)
            chunk = keys.get(30)
            if chunk is None:
                continue
            if not maxscroll:
                return
            done = False
            for tok in self._parse_chunk(chunk):
                if tok == "UP":
                    scroll -= 1
                elif tok == "DOWN":
                    scroll += 1
                elif tok == "PGUP":
                    scroll -= body_h
                elif tok == "PGDN":
                    scroll += body_h
                elif tok == "HOME":
                    scroll = 0
                elif tok == "END":
                    scroll = maxscroll
                else:
                    done = True
                    break
                scroll = max(0, min(scroll, maxscroll))
            if done:
                return

    def _model_list(self, query: str):
        """Picker rows for ``query``: (display, value, provider). Substring match
        over the catalog (curated first), plus a 'use exactly' custom row."""
        q = query.strip().lower()
        cat = _llm.model_catalog()
        if not q:
            hits = cat[:9]
        else:
            noisy = ("azure", "bedrock", "vercel", "fireworks", "novita",
                     "deepinfra", "sagemaker")
            curated = {m: i for i, (m, _) in enumerate(_llm.CURATED_MODELS)}
            matches = [(m, p) for (m, p) in cat
                       if q in m.lower() or q in p.lower()]
            # rank: curated picks (in our order) first, then name-prefix, then
            # plain providers, then shortest.
            matches.sort(key=lambda mp: (
                0 if mp[0] in curated else 1,
                curated.get(mp[0], 0),
                0 if mp[0].lower().startswith(q) else 1,
                1 if any(n in (mp[1] or "").lower() for n in noisy) else 0,
                len(mp[0]), mp[0]))
            hits = matches[:9]
        rows = [(m, m, p) for (m, p) in hits]
        qs = query.strip()
        if qs and qs.lower() not in (m.lower() for m, _, _ in rows):
            rows.append((f'use "{qs}" exactly', qs, "custom"))
        return rows

    def _llm_setup(self, screen, keys) -> bool:
        """Modal: pick/search a Model, enter the key/base, validate, save."""
        cur = self._llm_config or _llm.LLMConfig()
        fields = [["Model", list(cur.model)],
                  ["API key", list(cur.api_key)],
                  ["API base (optional)", list(cur.api_base)]]
        fi = 0
        pos = len(fields[0][1])
        pick = 0
        selected = False        # whole text field selected (key/base only)
        error = ""
        testing = False

        def focus(new_fi):
            nonlocal fi, pos, selected, pick
            fi = new_fi % len(fields)
            pos = len(fields[fi][1])
            selected = fi != 0 and len(fields[fi][1]) > 0   # select-all key/base
            pick = 0

        while True:
            lst = self._model_list("".join(fields[0][1])) if fi == 0 else []
            if lst:
                pick = max(0, min(pick, len(lst) - 1))
            screen.draw(self._llm_setup_frame(fields, fi, pos, selected, pick,
                                              lst, error, testing), hard=True)
            if testing:
                cfg = _llm.LLMConfig("".join(fields[0][1]).strip(),
                                     "".join(fields[1][1]).strip(),
                                     "".join(fields[2][1]).strip())
                ok, err = _llm.validate(cfg, complete=self._llm_complete)
                testing = False
                if ok:
                    _llm.save_config(cfg)
                    self._llm_config = cfg
                    return True
                error = (err or "validation failed")[:200]
                continue
            chunk = keys.get(30)
            if chunk is None:
                continue
            for tok in self._parse_chunk(chunk):
                buf = fields[fi][1]
                if tok in ("\x03", "\x04", "ESC"):
                    return False
                if fi == 0:                             # --- model search/picker ---
                    if tok in ("\r", "\n"):
                        if lst:
                            fields[0][1] = list(lst[pick][1])
                        focus(1)
                        break
                    if tok == "UP":
                        pick = max(0, pick - 1)
                    elif tok == "DOWN":
                        pick = min(len(lst) - 1, pick + 1) if lst else 0
                    elif tok == "\t":
                        focus(1)
                    elif tok == "LEFT":
                        pos = max(0, pos - 1)
                    elif tok == "RIGHT":
                        pos = min(len(buf), pos + 1)
                    elif tok == "HOME":
                        pos = 0
                    elif tok == "END":
                        pos = len(buf)
                    elif tok in ("\x7f", "\b", "\x08"):
                        if pos > 0:
                            del buf[pos - 1]
                            pos -= 1
                        pick = 0
                    elif tok == "DEL":
                        if pos < len(buf):
                            del buf[pos]
                        pick = 0
                    elif tok in ("\x15",):              # ^U clear search
                        buf.clear()
                        pos = pick = 0
                    elif isinstance(tok, str) and tok.isprintable():
                        for c in tok:
                            buf.insert(pos, c)
                            pos += 1
                        pick = 0
                    continue
                # --- key / base text fields (select-all + sliding window) ---
                if tok in ("\r", "\n"):
                    if "".join(fields[0][1]).strip():
                        testing, error = True, ""
                    else:
                        error = "pick a model first"
                        focus(0)
                    break
                if tok in ("DOWN", "\t"):
                    focus(fi + 1)
                elif tok == "UP":
                    focus(fi - 1)
                elif tok == "LEFT":
                    pos = 0 if selected else max(0, pos - 1)
                    selected = False
                elif tok == "RIGHT":
                    pos = len(buf) if selected else min(len(buf), pos + 1)
                    selected = False
                elif tok == "HOME":
                    pos, selected = 0, False
                elif tok == "END":
                    pos, selected = len(buf), False
                elif tok in ("\x7f", "\b", "\x08"):
                    if selected:
                        buf.clear()
                        pos, selected = 0, False
                    elif pos > 0:
                        del buf[pos - 1]
                        pos -= 1
                elif tok == "DEL":
                    if selected:
                        buf.clear()
                        pos, selected = 0, False
                    elif pos < len(buf):
                        del buf[pos]
                elif isinstance(tok, str) and tok.isprintable():
                    if selected:
                        buf.clear()
                        pos, selected = 0, False
                    for c in tok:
                        buf.insert(pos, c)
                        pos += 1

    def _llm_setup_frame(self, fields, fi, pos, selected, pick, lst, error,
                         testing) -> str:
        cols, rows = shutil.get_terminal_size((100, 30))
        L = ["\033[1mterminalboard — set up the LLM assistant\033[0m", ""]
        L.append("\033[2mType to search models (e.g. deepseek, qwen, claude, gpt) — "
                 "a small/cheap one is plenty \033[0m^_^")
        L.append("")
        avail = max(12, cols - 28)
        for idx, (label, buf) in enumerate(fields):
            plain = ("•" * len(buf)) if idx == 1 else "".join(buf)
            focused = idx == fi and not testing
            if focused:
                start = 0
                if len(plain) > avail:
                    start = max(0, min(pos - avail // 2, len(plain) - avail))
                win = plain[start:start + avail]
                wpos = pos - start
                lead = "…" if start > 0 else ""
                trail = "…" if start + avail < len(plain) else ""
                if selected and plain:
                    cur = "\033[7m" + win + "\033[0m"
                elif wpos < len(win):
                    cur = win[:wpos] + "\033[7m" + win[wpos] + "\033[0m" + win[wpos + 1:]
                else:
                    cur = win + "\033[7m \033[0m"
                tag = "search" if idx == 0 else label
                L.append(f"\033[1;36m▶\033[0m {tag:<22}{lead}{cur}{trail}")
            else:
                shown = plain if len(plain) <= avail else plain[:avail - 1] + "…"
                L.append(f"  {label:<22}{shown or '—'}")
            if idx == 0 and focused:                    # the model picker list
                nw = max(20, min(48, cols - 22))        # name column width
                for i, (disp, _val, prov) in enumerate(lst):
                    name = disp if len(disp) <= nw else disp[:nw - 1] + "…"
                    if i == pick:
                        row = _fit(f"  ▸ {name:<{nw}}  {prov}", cols - 1)
                        L.append("\033[7m" + row + "\033[0m")
                    else:
                        L.append(_fit(f"    {name:<{nw}}  \033[2m{prov}\033[0m",
                                      cols - 1))
                if not lst:
                    L.append("\033[2m    (no matches — type a custom model string)"
                             "\033[0m")
        L.append("")
        keypath = _llm.config_path().replace(os.path.expanduser("~"), "~")
        L.append(f"\033[2m🔒 key stored locally at {keypath} (chmod 600) · "
                 "API base blank for hosted providers\033[0m")
        L.append("\033[2m⚠ queries send your tag names + metric summaries to "
                 "this provider\033[0m")
        if testing:
            L.append("\033[1;33mtesting…\033[0m")
        elif error:
            L.append(f"\033[1;31m{error}\033[0m")
        L.append("")
        if fi == 0:
            L.append("\033[2m↑/↓ pick · Enter choose · Tab → key · Esc cancel\033[0m")
        else:
            L.append("\033[2mTab/↑↓ field · Enter test & save · Esc cancel\033[0m")
        return self._crop("\n".join(L), rows)
