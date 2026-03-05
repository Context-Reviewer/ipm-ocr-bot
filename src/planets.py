import time
from config import KEY_DELAY, MENU_DELAY, SCROLL_DELAY
from input_utils import tap

def planet_module(planets: int = 15):
    # Open planet menu
    tap("p", MENU_DELAY)

    # Reset left a bunch so we're near planet 1
    for _ in range(30):
        tap("-", KEY_DELAY)
    time.sleep(MENU_DELAY)

    # Upgrade loop
    for _ in range(planets):
        tap("ctrl+1", KEY_DELAY)  # mining
        tap("ctrl+2", KEY_DELAY)  # speed
        tap("ctrl+3", KEY_DELAY)  # cargo
        tap("=", SCROLL_DELAY)    # next planet

    # Exit planet menu (your known escape)
    tap("shift+1", MENU_DELAY)
    tap("shift+1", MENU_DELAY)
