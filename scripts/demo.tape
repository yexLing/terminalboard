# terminalboard demo — reproducible GIF via vhs (https://github.com/charmbracelet/vhs)
#
#   # one-time:
#   #   macOS:  brew install vhs
#   #   linux:  download a release from github.com/charmbracelet/vhs/releases
#   #           (needs `ttyd` and `ffmpeg` on PATH)
#   pip install -e .                 # so `terminalboard` is on PATH
#   vhs scripts/demo.tape            # writes demo.gif
#
# Uses the SYNTHETIC demo_logs (never your real ../tb_logs — tag names leak
# architecture).

Output demo.gif

Set FontSize 15
Set Width 1200
Set Height 700
Set Theme "Monokai"
Set Padding 12
Set TypingSpeed 60ms

# --- set up (hidden): generate the demo data, clear the screen ---
Hide
Type "python3 examples/gen_demo_logs.py >/dev/null 2>&1" Enter
Sleep 1s
Type "clear" Enter
Show

# --- launch ---
Type "terminalboard demo_logs" Enter
Sleep 3s

# page through the grid
Type "n" Sleep 1500ms
Type "n" Sleep 1500ms
Type "p" Sleep 1200ms

# zoom in a step
Type "Z" Sleep 1800ms

# filter tags -> only losses
Type "t" Sleep 700ms
Type "loss" Sleep 1200ms
Enter Sleep 2200ms

# focus a panel and inspect it full-screen, then sweep the cursor
Right Sleep 500ms
Enter Sleep 1800ms
Left Left Left Left Sleep 1500ms
Escape Sleep 1000ms

# histograms -> distribution bands
Type "t" Sleep 500ms
Ctrl+U
Type "layer0" Sleep 1000ms
Enter Sleep 1500ms
Type "b" Sleep 2000ms

# the HParams table
Type "P" Sleep 2500ms
Escape Sleep 800ms

# done
Type "q"
Sleep 1200ms
