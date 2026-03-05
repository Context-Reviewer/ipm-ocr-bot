import time
import ctypes
import keyboard
from config import KEY_DELAY, MENU_DELAY

INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002

VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_NUMPAD0 = 0x60
VK_OEM_MINUS = 0xBD
VK_OEM_PLUS = 0xBB

user32 = ctypes.windll.user32

DEBUG = False

class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]

class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("ki", KEYBDINPUT),
    ]

def _send_key(vk: int, flags: int):
    inp = INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(wVk=vk, wScan=0, dwFlags=flags, time=0, dwExtraInfo=None))
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

def _key_down(vk: int):
    _send_key(vk, 0)

def _key_up(vk: int):
    _send_key(vk, KEYEVENTF_KEYUP)

def _key_press(vk: int):
    _key_down(vk)
    _key_up(vk)

def normalize_key(key: str) -> str:
    if not isinstance(key, str):
        return ""
    t = key.strip().lower().replace(" ", "")
    num_map = {
        "num2": "numpad 2",
        "num5": "numpad 5",
        "num7": "numpad 7",
        "num8": "numpad 8",
        "num9": "numpad 9",
    }
    return num_map.get(t, key.strip())

def _vk_for_numpad(numpad_key: str):
    t = numpad_key.strip().lower()
    if t.startswith("numpad"):
        parts = t.split()
        if len(parts) == 2 and parts[1].isdigit():
            digit = int(parts[1])
            if 0 <= digit <= 9:
                return VK_NUMPAD0 + digit
    return None

def tap(key: str, delay: float = KEY_DELAY, debug: bool = False):
    if debug or DEBUG:
        print(f"TAP {key}")

    normalized = normalize_key(key)
    vk = _vk_for_numpad(normalized)
    if vk is not None:
        _key_press(vk)
        time.sleep(delay)
        return

    keyboard.send(key)
    time.sleep(delay)

def tap_key(key: str) -> None:
    tap(key)

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
