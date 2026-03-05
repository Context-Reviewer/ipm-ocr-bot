from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import win32gui
import win32con


@dataclass(frozen=True)
class ClientRect:
    left: int
    top: int
    width: int
    height: int


def _enum_windows():
    hwnds = []

    def cb(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            hwnds.append(hwnd)
        return True

    win32gui.EnumWindows(cb, None)
    return hwnds


def find_window_by_title_substring(sub: str) -> Optional[int]:
    sub_lower = sub.lower()
    for hwnd in _enum_windows():
        try:
            title = win32gui.GetWindowText(hwnd) or ""
        except Exception:
            continue
        if sub_lower in title.lower():
            return hwnd
    return None


def get_client_rect_screen(hwnd: int) -> Optional[ClientRect]:
    if not hwnd:
        return None

    try:
        # Client rect in client coords
        left, top, right, bottom = win32gui.GetClientRect(hwnd)
        w = max(0, right - left)
        h = max(0, bottom - top)
        if w == 0 or h == 0:
            return None

        # Convert client (0,0) -> screen coords
        screen_left, screen_top = win32gui.ClientToScreen(hwnd, (0, 0))
        return ClientRect(left=screen_left, top=screen_top, width=w, height=h)
    except Exception:
        return None


def get_bluestacks_client_rect(title_hint: str = "BlueStacks App Player") -> Optional[ClientRect]:
    hwnd = find_window_by_title_substring(title_hint)
    if not hwnd:
        return None
    return get_client_rect_screen(hwnd)
