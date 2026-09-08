# Changelog

## Unreleased

## 0.5.2 — 2026-06-13

### Changed
- **Filter `!word` is now a global exclude.** It applies to the whole filter no
  matter which OR alternative it's written in, so
  `a | b | c !d` means *(a or b or c) and not d* (previously the `!d` only bound
  to its own `c` alternative). Positive OR/AND matching is unchanged; a filter of
  only excludes still keeps everything else.

## 0.5.1 — 2026-06-13

### Fixed
- **Startup crash** (`KeyError: 'kind_filter'`) when restoring a per-logdir view
  state written by an older version (pre-0.4, before the type selector existed):
  `None` is a valid `kind_filter` *and* a member of the cycle, so an absent key
  slipped past the guard. Now the key's presence is checked explicitly, and the
  whole state-apply step is wrapped so a stale/foreign state file can never crash
  startup. Workaround on older builds: `--reset-view` or delete
  `~/.local/state/terminalboard/views/*.json`.

## 0.5.0 — 2026-06-13

### Added
- **Chat sidebar** (`a` / `A`) — a persistent, **multi-session** conversation
  panel on the right; the dashboard re-tiles into the remaining width and is
  cached so typing only repaints the chat. The chat sees the **live view**
  (focused tag, what's visible, counts, mode) plus all log data, **answers and
  drives the dashboard** in the same turn, and streams its reply into the
  transcript (light markdown). **`Esc` closes it** (never quits); the input has a
  full editor (`^W`/`^U`/`^A`/`^E`/word-motion) with a sliding window so the
  cursor stays on-screen; `↑/↓` recall messages, `PgUp/PgDn` scroll. Slash
  commands manage sessions — `/new`, `/next`, `/prev`, `/delete`, `/rename`,
  `/clear`, `/sessions`, `/model`, `/close` — persisted per-logdir.
- **LLM assistant** (`a`) — *optional*, `pip install 'terminalboard[llm]'`.
  Ask in natural language; the model both **navigates** the dashboard (filter
  tags/experiments, pick a type, smooth, zoom, open a tag, open HParams, …) and
  **analyzes** your results, in one turn. Powered by **LiteLLM**, so any provider
  works (OpenAI / Anthropic / Gemini / OpenRouter / local Ollama / …) — you pick
  the model string + key in a first-run setup form (`A` to reconfigure). Actions
  are a typed whitelist (no shell). Answers **stream** as they arrive, with
  follow-up memory (so "now zoom into that" works), context-aware "explain this
  panel" from the detail view, and a tokens/cost/latency readout. ⚠️ queries send
  tag names + metric summaries to your chosen provider; use a local model
  (Ollama) to keep everything on-box.

## 0.4.0 — 2026-06-09

### Added
- **Distributions view** (`b`): toggle histogram panels between the heatmap and
  percentile bands (0/25/50/75/100 over steps; median highlighted) — the same
  data TensorBoard shows under "Distributions". Works in the grid and detail.
- **PR curves**: the `pr_curves` plugin (precision-vs-recall) is parsed and drawn
  as a curve; the detail view steps through training with `←/→`.
- **HParams table** (`P`): a full-screen, scrollable table of runs × hyper-
  parameters × final metric values, parsed from the `hparams` plugin.
- **Type selector** (`c`): cycle the grid between all types / scalars /
  histograms / text / pr-curves — a quick filter by data type.
- The bundled demo (`examples/gen_demo_logs.py`) now also emits a PR curve and
  HParams so all five types are visible out of the box.

## 0.3.0 — 2026-06-08

### Added
- **Log-Y** (`l`) and **x-axis step↔time** (`x`) toggles for scalar panels.
- **Config diff** in the text detail view (`d`): show only the config keys that
  differ across experiments.
- **CSV export** (`w`): write the focused scalar tag to `<tag>.csv`.
- **Config file** (`~/.config/terminalboard.toml` / `$TERMINALBOARD_CONFIG`) for
  defaults (smooth, grid, interval, tags, experiments, xaxis, logy, tb).
- Bundled **demo generator** (`examples/gen_demo_logs.py`) and a **GIF recording
  script** (`scripts/record_demo.sh`); `uvx`/`pipx run` note in the README.

### Added
- **Per-logdir view persistence**: filters, zoom, smoothing, x-axis, log-Y,
  curve order and focus are saved on exit (under `$XDG_STATE_HOME` /
  `~/.local/state/terminalboard/`) and restored when you reopen the same logdir.
  Explicit CLI flags still win; `--reset-view` ignores the saved state (and
  `restore = false` in the config disables persistence).

### Fixed
- **Scalar detail cursor** now ranges over the **union of all visible runs'
  steps**, so `←/→`/`End` can reach the last point among *all* experiments —
  previously it stopped at one run's final step even when others had data
  further right. It also **starts in the middle** of the range (so it's clear it
  can move both ways) instead of parked at the far right.

### Changed
- **Legend** now shows **full experiment names**, wrapping over multiple lines
  instead of truncating with `…` — so names are readable when filtering.
- **Panel titles** show the **full tag path**, wrapping over up to 3 lines (with
  a uniform, row-aligned height across the page); when still too long the last
  line keeps the leaf via leading ellipsis. Scalar titles are drawn by us, so a
  wide title is never dropped by plotext.

## 0.2.1

### Fixed / changed
- **Filter regex**: when the *whole* filter is wrapped in `/.../` it's now a real
  regex (`re.search`, case-insensitive), so `|` and spaces work
  (e.g. `/^train\/(loss|lr)$/`). A per-word `/regex/` (without `|`/spaces) still
  works inside boolean expressions.
- Help overlay clarifies the regex rule; package summary now mentions text and
  histogram support.

## 0.2.0

### Added
- **Plot types** beyond scalars: **text summaries** and **histograms** (drawn as
  a heatmap of the distribution over steps), mixed freely in the grid.
- **Focus + drill-down**: arrow keys move a highlighted panel; **Enter** inspects
  a tag full-screen, **Esc** goes back.
  - **Scalar detail** has a TensorBoard-style **x-cursor** — `←/→` move one data
    point, `Shift+←/→` jump fast — with a per-experiment **value / smoothed /
    step / wall-time** readout.
  - **Histogram / text detail**: `←/→` switch experiment; text is **scrollable**
    (`↑/↓`, `PgUp/PgDn`, `Home/End`), one experiment per screen.
- **Curve z-order** (`o`): cycle which overlapping experiment is drawn on top.
- **Richer filter grammar**: `|`/`,` = OR, space/`&` = AND, `!` = NOT,
  `/regex/`, and `* ? [ ]` globs.
- **Readline editing** in filter prompts: `^W`, `^K`, `Alt`/`Ctrl`+arrows, and
  per-kind history.
- **Help overlay** on `H` / `?`.

### Changed
- The dependency-free **pure-Python parser is now the default**; `--tb` opts into
  the tensorboard `EventAccumulator`.
- **`Esc`** quits from the grid (and goes back one level from a detail view).

### Removed
- The **`--hq` image renderer** and the `matplotlib` / iTerm2 inline-image
  dependencies — everything is pure text now (lighter, faster, one render path).

## 0.1.0 – 0.1.3

Initial PyPI releases: live scalar dashboard in braille/Unicode (plus an early
`--hq` image mode), pure-Python and tensorboard parsers, multi-experiment overlay
with stable per-run colors, zoom, live tag/experiment filtering, and flicker-free
repaints.
