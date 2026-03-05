import time
import keyboard
from config import KEY_DELAY, MENU_DELAY

def tap(key: str, delay: float = KEY_DELAY):
    keyboard.send(key)
    time.sleep(delay)

def reset_ui():
    """
    Force the UI back to a known safe state.
    Prevents drift if a module leaves a menu open.
    """
    tap("shift+1", MENU_DELAY)
    tap("shift+1", MENU_DELAY)

    tap("shift+2", MENU_DELAY)
    tap("shift+2", MENU_DELAY)

    tap("shift+1", MENU_DELAY)
    tap("shift+1", MENU_DELAY)
