import os

# The sketch tests open arcade windows; render offscreen so they run without a display.
os.environ.setdefault("ARCADE_HEADLESS", "1")
