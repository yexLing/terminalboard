#!/usr/bin/env bash
# Record a demo GIF of terminalboard.
#
# Requires:
#   - asciinema   (pip install asciinema)
#   - agg         (asciinema -> gif; `cargo install agg` or grab a release:
#                  https://github.com/asciinema/agg)
#
# Usage:
#   scripts/record_demo.sh [LOGDIR]
#
# It generates a demo logdir if none is given, opens terminalboard for you to
# drive (page, filter with t/f, Enter to inspect a curve and move the cursor,
# Esc/q to quit), then converts the recording to demo.gif.
set -euo pipefail

LOGDIR="${1:-demo_logs}"
if [ ! -d "$LOGDIR" ]; then
  echo "Generating $LOGDIR ..."
  python3 "$(dirname "$0")/../examples/gen_demo_logs.py"
fi

echo "Recording — interact with terminalboard, then press q to finish."
asciinema rec --overwrite -c "terminalboard $LOGDIR" demo.cast

echo "Converting to demo.gif ..."
agg --theme monokai demo.cast demo.gif
echo "Wrote demo.gif"
