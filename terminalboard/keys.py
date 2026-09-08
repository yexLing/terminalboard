"""Non-blocking key reader for the interactive live loop.

Puts the terminal in cbreak mode and reads input with a timeout so the loop can
also wake up to refresh on its interval. A no-op fallback is used when stdin is
not a TTY (pipes, ``--once``, CI).

Reads come straight from the file descriptor with ``os.read`` rather than
``sys.stdin.read`` — the latter buffers inside Python's text layer, so the tail
of a multi-byte escape sequence (e.g. an arrow key's ``[A``) would sit in that
buffer where ``select`` can't see it, and the sequence would be misread as a
lone Esc. Reading the raw fd keeps ``select`` and the read in sync, so a whole
escape sequence arrives in a single ``get()``.
"""
from __future__ import annotations

import os
import select
import sys
from typing import Optional


class KeyReader:
    def __init__(self):
        self._isatty = sys.stdin.isatty()
        self._fd = None
        self._saved = None

    def __enter__(self) -> "KeyReader":
        if not self._isatty:
            return self
        try:
            import termios
            import tty

            self._fd = sys.stdin.fileno()
            self._saved = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        except Exception:
            self._isatty = False
        return self

    def __exit__(self, *exc) -> None:
        if self._saved is not None and self._fd is not None:
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)

    def get(self, timeout: float) -> Optional[str]:
        """Return the next input chunk within ``timeout`` seconds, else None.

        A chunk is usually a single character, but a key that emits an escape
        sequence (arrows, Home/End, …) is returned whole, e.g. ``"\\x1b[A"``.
        """
        if not self._isatty:
            if timeout > 0:
                import time

                time.sleep(timeout)
            return None
        r, _, _ = select.select([self._fd], [], [], timeout)
        if not r:
            return None
        try:
            data = os.read(self._fd, 64)
        except OSError:
            return None
        if not data:
            return None
        return data.decode("utf-8", "ignore")
