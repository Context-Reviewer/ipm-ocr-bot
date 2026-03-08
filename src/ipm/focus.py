from __future__ import annotations

import ctypes
import time

from .config import FocusConfig
from window_win32 import activate_window_by_title_substring


def get_active_window_title() -> str:
    try:
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        if not hwnd:
            return ""
        length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return ""
        buf = ctypes.create_unicode_buffer(length + 1)
        ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
        return buf.value
    except Exception:
        return ""


def title_matches_focus_target(title: str, cfg: FocusConfig) -> bool:
    if not title:
        return False
    if cfg.window_substring and cfg.window_substring.lower() not in title.lower():
        return False
    for blocked in cfg.excluded_substrings:
        if blocked and blocked.lower() in title.lower():
            return False
    return True


def is_focus_ok(cfg: FocusConfig) -> bool:
    if not cfg.required:
        return True
    return title_matches_focus_target(get_active_window_title(), cfg)


def ensure_focus(cfg: FocusConfig) -> bool:
    if is_focus_ok(cfg):
        return True
    if not cfg.auto_activate:
        return False
    if activate_window_by_title_substring(cfg.window_substring):
        time.sleep(cfg.activate_retry_delay)
    return is_focus_ok(cfg)
