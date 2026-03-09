from __future__ import annotations

import ctypes
import time
from dataclasses import dataclass

from .config import FocusConfig
from window_win32 import ActivationResult, activate_window_by_title_substring_result


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


@dataclass(frozen=True)
class FocusResult:
    ok: bool
    reason: str
    active_title_before: str
    active_title_after: str
    activation_status: str = ""
    activation_error: str = ""
    target_title: str = ""


def ensure_focus_result(cfg: FocusConfig) -> FocusResult:
    active_title_before = get_active_window_title()
    if not cfg.required:
        return FocusResult(
            ok=True,
            reason="focus_not_required",
            active_title_before=active_title_before,
            active_title_after=active_title_before,
        )
    if title_matches_focus_target(active_title_before, cfg):
        return FocusResult(
            ok=True,
            reason="already_focused",
            active_title_before=active_title_before,
            active_title_after=active_title_before,
        )
    if not cfg.auto_activate:
        return FocusResult(
            ok=False,
            reason="auto_activate_disabled",
            active_title_before=active_title_before,
            active_title_after=active_title_before,
        )

    activation = activate_window_by_title_substring_result(cfg.window_substring)
    if activation.ok:
        time.sleep(cfg.activate_retry_delay)
    active_title_after = get_active_window_title()
    if title_matches_focus_target(active_title_after, cfg):
        return FocusResult(
            ok=True,
            reason="focused_after_activation",
            active_title_before=active_title_before,
            active_title_after=active_title_after,
            activation_status=activation.status,
            activation_error=activation.error_message,
            target_title=activation.target_title,
        )
    return FocusResult(
        ok=False,
        reason=activation.status or "focus_mismatch",
        active_title_before=active_title_before,
        active_title_after=active_title_after,
        activation_status=activation.status,
        activation_error=activation.error_message,
        target_title=activation.target_title,
    )


def ensure_focus(cfg: FocusConfig) -> bool:
    return ensure_focus_result(cfg).ok
