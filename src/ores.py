import time
from config import KEY_DELAY, MENU_DELAY, SCROLL_DELAY
from input_utils import tap

def ore_module(pages: int = 2):
    # Open resources
    tap("shift+1", MENU_DELAY)
    tap("f1", MENU_DELAY)  # ores tab

    # Reset to top (safe even if already at top)
    for _ in range(10):
        tap("num8", KEY_DELAY)

    row_keys = ["1", "2", "3", "4"]
    scroll_key = "num2"  # BlueStacks swipe down

    for page in range(pages):
        for key in row_keys:
            tap(key, KEY_DELAY)
            tap("num5", KEY_DELAY)  # open sell
            tap("num9", KEY_DELAY)  # slider to ~100%
            tap("num5", KEY_DELAY)  # execute sell

        if page < pages - 1:
            tap(scroll_key, SCROLL_DELAY)
            time.sleep(MENU_DELAY)

    # Close resources
    tap("shift+1", MENU_DELAY)
    tap("shift+1", MENU_DELAY)
