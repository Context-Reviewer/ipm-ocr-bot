from __future__ import annotations

import ctypes
from dataclasses import dataclass
import time
from typing import Optional

import win32gui
import win32con

import config


@dataclass(frozen=True)
class ClientRect:
    left: int
    top: int
    width: int
    height: int


@dataclass(frozen=True)
class ActivationResult:
    ok: bool
    status: str
    hwnd: int | None
    active_title_before: str
    active_title_after: str
    target_title: str = ""
    error_message: str = ""
    used_attach_thread_input: bool = False


_RECT_CACHE: dict[str, tuple[float, ClientRect]] = {}
_USER32 = ctypes.WinDLL("user32", use_last_error=True)
_KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)


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


def _get_window_title(hwnd: int) -> str:
    if not hwnd:
        return ""
    try:
        return win32gui.GetWindowText(hwnd) or ""
    except Exception:
        return ""


def get_foreground_window_title() -> str:
    try:
        hwnd = win32gui.GetForegroundWindow()
    except Exception:
        return ""
    return _get_window_title(hwnd)


def _get_foreground_window_handle() -> int:
    try:
        return int(win32gui.GetForegroundWindow())
    except Exception:
        return 0


def _get_current_thread_id() -> int:
    try:
        return int(_KERNEL32.GetCurrentThreadId())
    except Exception:
        return 0


def _get_window_thread_id(hwnd: int) -> int:
    if not hwnd:
        return 0
    try:
        return int(_USER32.GetWindowThreadProcessId(int(hwnd), None))
    except Exception:
        return 0


def _attach_thread_input_pair(source_thread_id: int, target_thread_id: int, attach: bool) -> tuple[bool, str]:
    if source_thread_id <= 0 or target_thread_id <= 0 or source_thread_id == target_thread_id:
        return True, ""
    try:
        try:
            ctypes.set_last_error(0)
        except Exception:
            pass
        result = int(_USER32.AttachThreadInput(int(source_thread_id), int(target_thread_id), int(bool(attach))))
        if result:
            return True, ""
        error_code = int(ctypes.get_last_error())
        if error_code:
            return False, f"AttachThreadInput failed error={error_code}"
        return False, "AttachThreadInput failed"
    except Exception as exc:
        return False, str(exc)


def _run_activation_sequence(
    hwnd: int,
    *,
    active_title_before: str,
    target_title: str,
    success_status: str,
    mismatch_status: str,
    setforeground_failed_status: str,
    used_attach_thread_input: bool,
) -> ActivationResult:
    try:
        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        else:
            win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
        try:
            win32gui.BringWindowToTop(hwnd)
        except Exception:
            pass
        win32gui.SetForegroundWindow(hwnd)
    except Exception as exc:
        return ActivationResult(
            ok=False,
            status=setforeground_failed_status,
            hwnd=hwnd,
            active_title_before=active_title_before,
            active_title_after=get_foreground_window_title(),
            target_title=target_title,
            error_message=str(exc),
            used_attach_thread_input=used_attach_thread_input,
        )

    active_title_after = get_foreground_window_title()
    active_hwnd = _get_foreground_window_handle()
    if active_hwnd == hwnd:
        return ActivationResult(
            ok=True,
            status=success_status,
            hwnd=hwnd,
            active_title_before=active_title_before,
            active_title_after=active_title_after,
            target_title=target_title,
            used_attach_thread_input=used_attach_thread_input,
        )
    return ActivationResult(
        ok=False,
        status=mismatch_status,
        hwnd=hwnd,
        active_title_before=active_title_before,
        active_title_after=active_title_after,
        target_title=target_title,
        used_attach_thread_input=used_attach_thread_input,
    )


def _activate_window_direct(hwnd: int, *, active_title_before: str, target_title: str) -> ActivationResult:
    return _run_activation_sequence(
        hwnd,
        active_title_before=active_title_before,
        target_title=target_title,
        success_status="activated",
        mismatch_status="foreground_mismatch",
        setforeground_failed_status="setforeground_failed",
        used_attach_thread_input=False,
    )


def _activate_window_with_attach_thread_input(
    hwnd: int,
    *,
    active_title_before: str,
    target_title: str,
) -> ActivationResult:
    current_thread_id = _get_current_thread_id()
    foreground_hwnd = _get_foreground_window_handle()
    foreground_thread_id = _get_window_thread_id(foreground_hwnd)
    target_thread_id = _get_window_thread_id(hwnd)
    attached_pairs: list[tuple[int, int]] = []
    try:
        for source_thread_id, linked_thread_id in (
            (current_thread_id, foreground_thread_id),
            (current_thread_id, target_thread_id),
        ):
            ok, error_message = _attach_thread_input_pair(source_thread_id, linked_thread_id, True)
            if not ok:
                return ActivationResult(
                    ok=False,
                    status="attach_thread_input_failed",
                    hwnd=hwnd,
                    active_title_before=active_title_before,
                    active_title_after=get_foreground_window_title(),
                    target_title=target_title,
                    error_message=error_message,
                    used_attach_thread_input=True,
                )
            if source_thread_id > 0 and linked_thread_id > 0 and source_thread_id != linked_thread_id:
                attached_pairs.append((source_thread_id, linked_thread_id))
        return _run_activation_sequence(
            hwnd,
            active_title_before=active_title_before,
            target_title=target_title,
            success_status="activated_via_attach_thread_input",
            mismatch_status="foreground_mismatch_after_attach_thread_input",
            setforeground_failed_status="setforeground_failed_after_attach_thread_input",
            used_attach_thread_input=True,
        )
    finally:
        for source_thread_id, linked_thread_id in reversed(attached_pairs):
            _attach_thread_input_pair(source_thread_id, linked_thread_id, False)


def activate_window_result(hwnd: int) -> ActivationResult:
    active_title_before = get_foreground_window_title()
    target_title = _get_window_title(hwnd)
    if not hwnd:
        return ActivationResult(
            ok=False,
            status="window_not_found",
            hwnd=None,
            active_title_before=active_title_before,
            active_title_after=active_title_before,
        )
    direct_result = _activate_window_direct(
        hwnd,
        active_title_before=active_title_before,
        target_title=target_title,
    )
    if direct_result.ok or direct_result.status not in {"setforeground_failed", "foreground_mismatch"}:
        return direct_result
    return _activate_window_with_attach_thread_input(
        hwnd,
        active_title_before=active_title_before,
        target_title=target_title,
    )


def activate_window(hwnd: int) -> bool:
    return activate_window_result(hwnd).ok


def activate_window_by_title_substring(sub: str) -> bool:
    return activate_window_by_title_substring_result(sub).ok


def activate_window_by_title_substring_result(sub: str) -> ActivationResult:
    hwnd = find_window_by_title_substring(sub)
    if not hwnd:
        active_title = get_foreground_window_title()
        return ActivationResult(
            ok=False,
            status="window_not_found",
            hwnd=None,
            active_title_before=active_title,
            active_title_after=active_title,
        )
    return activate_window_result(hwnd)


def _title_matches(hwnd: int, sub: str) -> bool:
    try:
        title = win32gui.GetWindowText(hwnd) or ""
    except Exception:
        return False
    return sub.lower() in title.lower()


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
    now = time.monotonic()
    ttl = float(getattr(config, "WINDOW_RECT_CACHE_TTL", 0.5) or 0.0)
    cached = _RECT_CACHE.get(title_hint)
    if cached and cached[0] > now:
        return cached[1]

    hwnd = None
    try:
        fg = win32gui.GetForegroundWindow()
        if fg and _title_matches(fg, title_hint):
            hwnd = fg
    except Exception:
        hwnd = None

    if not hwnd:
        hwnd = find_window_by_title_substring(title_hint)
    if not hwnd:
        return None
    rect = get_client_rect_screen(hwnd)
    if rect and ttl > 0:
        _RECT_CACHE[title_hint] = (now + ttl, rect)
    return rect
