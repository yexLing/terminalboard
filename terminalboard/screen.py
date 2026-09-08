"""Flicker-free terminal painter.

The naive "clear the whole screen, then print" approach flashes badly: every
frame the terminal goes blank and then fills back in. This avoids that:

  * draw on the **alternate screen buffer** and hide the cursor;
  * each frame, move the cursor **home and overwrite in place** (clearing to the
    end of each line and below the frame) instead of blanking first;
  * wrap each repaint in **synchronized output** (DEC private mode 2026) so the
    terminal presents the whole frame atomically — no tearing, no flash.

All of these degrade gracefully: unsupported sequences are simply ignored, and
on a non-TTY the painter just writes plain text.
"""
from __future__ import annotations

import sys

_ALT_ON = "\033[?1049h"
_ALT_OFF = "\033[?1049l"
_HIDE = "\033[?25l"
_SHOW = "\033[?25h"
_WRAP_OFF = "\033[?7l"    # disable line wrap: clip overlong lines at the margin
_WRAP_ON = "\033[?7h"
_SYNC_ON = "\033[?2026h"
_SYNC_OFF = "\033[?2026l"
_HOME = "\033[H"
_CLEAR_EOL = "\033[K"     # erase to end of line
_CLEAR_BELOW = "\033[J"   # erase from cursor to end of screen
_CLEAR_ALL = "\033[2J"    # erase the whole screen


class Screen:
    def __init__(self, use_alt: bool = True):
        self._tty = sys.stdout.isatty()
        self.use_alt = use_alt and self._tty

    def __enter__(self) -> "Screen":
        if self._tty:
            seq = (_ALT_ON if self.use_alt else "") + _HIDE + _WRAP_OFF
            sys.stdout.write(seq)
            sys.stdout.flush()
        return self

    def __exit__(self, *exc) -> None:
        if self._tty:
            seq = _WRAP_ON + _SHOW + (_ALT_OFF if self.use_alt else "")
            sys.stdout.write(seq)
            sys.stdout.flush()

    def draw(self, frame: str, *, hard: bool = False) -> None:
        """Repaint the screen with ``frame``.

        ``hard=False`` overwrites in place (flicker-free) — used for live data
        updates where the layout is unchanged. ``hard=True`` first clears the
        whole screen — used when the view changes (page/grid/smoothing) so a
        new layout can never leave residue from the old one. Both are wrapped
        in synchronized output, so even the hard clear presents atomically.
        """
        if not self._tty:
            sys.stdout.write(frame + "\n")
            sys.stdout.flush()
            return
        # Clear to end of each line so a shorter new line can't leave stale
        # characters behind; clear below to drop any now-unused trailing rows.
        body = frame.replace("\n", _CLEAR_EOL + "\n")
        clear = _CLEAR_ALL if hard else ""
        out = _SYNC_ON + clear + _HOME + body + _CLEAR_EOL + _CLEAR_BELOW + _SYNC_OFF
        sys.stdout.write(out)
        sys.stdout.flush()
