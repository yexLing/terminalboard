# Contributing to terminalboard

Thanks for your interest! Contributions — bug reports, fixes, features, docs —
are welcome.

## Workflow (fork & pull request)

You don't need write access to this repo. The standard flow:

1. **Fork** the repo on GitHub (top-right *Fork* button).
2. **Clone** your fork and create a branch:
   ```bash
   git clone https://github.com/<you>/terminalboard.git
   cd terminalboard
   git checkout -b my-feature
   ```
3. **Set up a dev environment:**
   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -e '.[tb,dev]'
   ```
4. **Make your change**, and check it runs:
   ```bash
   .venv/bin/terminalboard path/to/tb_logs --once          # text renderer
   .venv/bin/terminalboard path/to/tb_logs --light --once  # pure-Python parser
   ```
5. **Run the tests** (CI runs these on every PR):
   ```bash
   .venv/bin/pip install -e '.[full,dev]'   # adds pytest + build/twine
   .venv/bin/pytest -q
   ```
   The suite is self-contained — it generates a tiny synthetic event file, so no
   real logs are needed.
6. **Commit and push** to your fork:
   ```bash
   git commit -am "Describe your change"
   git push origin my-feature
   ```
7. **Open a Pull Request** from your branch to `yexLing/terminalboard:main`.
   Describe what changed and why; link any related issue.

A maintainer reviews, may request changes, then merges. Your commits keep your
authorship.

## Guidelines

- **Keep the base install light.** Only `plotext` is a hard dependency; the
  built-in parser and the renderer must work with no heavy deps. `tensorboard`
  stays behind the `[tb]` extra and must be imported lazily (inside the function
  that needs it), so startup stays fast.
- **Match the surrounding style** — small, focused modules; clear names; comments
  that explain *why*, not *what*.
- **One logical change per PR.** Smaller PRs review and merge faster.
- **Update docs** (`README.md`) when you change CLI flags or behavior.
- Be kind in reviews and issues.

## Project layout

```
terminalboard/
  model.py         scalar / text / histogram series data model
  reader_light.py  pure-Python TFRecord + protobuf parser (default)
  reader.py        run discovery + tensorboard backend (--tb)
  render.py        plotext text renderer: scalar curves, text, histogram heatmaps
  screen.py        flicker-free terminal painter (alt screen + sync output)
  keys.py          raw-fd key reader
  app.py           live loop, paging/zoom, interactive filters
  cli.py           argparse front end
```

## Reporting bugs / requesting features

Open a GitHub Issue. For bugs, include: what you ran, what happened, what you
expected, your terminal (e.g. iTerm2/tmux), and a snippet of the event-file
layout if relevant.

## License

By contributing you agree your contributions are licensed under the project's
[MIT License](LICENSE).
