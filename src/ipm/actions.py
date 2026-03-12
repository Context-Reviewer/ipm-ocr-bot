from __future__ import annotations

import time
import ctypes
import re

import keyboard
from PIL import Image, ImageStat

from .config import RuntimeConfig
from .focus import ensure_focus
from .rects import RectStore
from window_win32 import get_bluestacks_client_rect


INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_ABSOLUTE = 0x8000
SM_CXSCREEN = 0
SM_CYSCREEN = 1


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("mi", MOUSEINPUT),
    ]


class ActionDriver:
    def __init__(self, config: RuntimeConfig, rects: RectStore | None = None, capture_backend: object | None = None) -> None:
        self.config = config
        self.rects = rects
        self.capture_backend = capture_backend

    @staticmethod
    def _normalize_key(key: str) -> str:
        normalized = str(key or "").strip().lower().replace(" ", "")
        numpad_map = {
            "num0": "numpad 0",
            "num1": "numpad 1",
            "num2": "numpad 2",
            "num3": "numpad 3",
            "num4": "numpad 4",
            "num5": "numpad 5",
            "num6": "numpad 6",
            "num7": "numpad 7",
            "num8": "numpad 8",
            "num9": "numpad 9",
        }
        return numpad_map.get(normalized, str(key).strip())

    def _invalidate_capture(self) -> None:
        if self.capture_backend is None:
            return
        invalidate = getattr(self.capture_backend, "invalidate", None)
        if callable(invalidate):
            invalidate()

    def _capture_screen(self) -> Image.Image | None:
        if self.capture_backend is None:
            return None
        capture_screen = getattr(self.capture_backend, "capture_screen", None)
        if not callable(capture_screen):
            return None
        return capture_screen()

    def _capture_rect_image(self, rect_key: str) -> Image.Image | None:
        image = self._capture_screen()
        if image is None or self.rects is None:
            return None
        rect = self.rects.get(rect_key)
        if rect is None:
            return None
        x, y, w, h = rect
        if w <= 0 or h <= 0:
            return None
        return image.crop((x, y, x + w, y + h))

    @staticmethod
    def _crop_signal_stats(image: Image.Image | None, *, bright_threshold: int) -> dict[str, float] | None:
        if image is None:
            return None
        grayscale = image.convert("L")
        stat = ImageStat.Stat(grayscale)
        histogram = grayscale.histogram()
        total = max(1.0, float(sum(histogram)))
        extrema = grayscale.getextrema()
        return {
            "dynamic_range": float(int(extrema[1]) - int(extrema[0])),
            "bright_fraction": sum(float(histogram[index]) for index in range(int(bright_threshold), 256)) / total,
        }

    def _rect_has_signal(
        self,
        rect_key: str,
        *,
        min_dynamic_range: int,
        min_bright_fraction: float,
        bright_threshold: int,
    ) -> bool:
        stats = self._crop_signal_stats(self._capture_rect_image(rect_key), bright_threshold=bright_threshold)
        if stats is None:
            return False
        return (
            stats["dynamic_range"] >= float(min_dynamic_range)
            and stats["bright_fraction"] >= float(min_bright_fraction)
        )

    def _resources_panel_visible(self) -> bool:
        return self._rect_has_signal(
            "RESOURCES",
            min_dynamic_range=120,
            min_bright_fraction=0.05,
            bright_threshold=150,
        )

    def _production_panel_visible(self) -> bool:
        return self._rect_has_signal(
            "PRODUCTION",
            min_dynamic_range=120,
            min_bright_fraction=0.05,
            bright_threshold=150,
        )

    def _production_tab_active(self, rect_key: str) -> bool:
        if not self._production_panel_visible():
            return False
        return self._rect_has_signal(
            rect_key,
            min_dynamic_range=160,
            min_bright_fraction=0.12,
            bright_threshold=185,
        )

    def _wait_for_state(self, checker, *, attempts: int = 3, delay: float | None = None) -> bool:
        settle_delay = float(delay if delay is not None else self.config.actions.menu_delay_seconds)
        for attempt_index in range(max(1, int(attempts))):
            if checker():
                return True
            if attempt_index < attempts - 1:
                time.sleep(settle_delay)
                self._invalidate_capture()
        return checker()

    @staticmethod
    def _screen_to_absolute_units(x: int, y: int) -> tuple[int, int]:
        user32 = ctypes.windll.user32
        screen_w = int(user32.GetSystemMetrics(SM_CXSCREEN))
        screen_h = int(user32.GetSystemMetrics(SM_CYSCREEN))
        abs_x = int(int(x) * 65535 / max(1, screen_w - 1))
        abs_y = int(int(y) * 65535 / max(1, screen_h - 1))
        return abs_x, abs_y

    @classmethod
    def _send_mouse_input(cls, *, x: int, y: int, flags: int) -> None:
        abs_x, abs_y = cls._screen_to_absolute_units(x, y)
        user32 = ctypes.windll.user32
        event = INPUT(
            type=INPUT_MOUSE,
            mi=MOUSEINPUT(
                dx=abs_x,
                dy=abs_y,
                mouseData=0,
                dwFlags=int(flags) | MOUSEEVENTF_ABSOLUTE,
                time=0,
                dwExtraInfo=None,
            ),
        )
        user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(event))

    def _open_production_panel(self) -> bool:
        if self._wait_for_state(self._production_panel_visible, attempts=1):
            return True
        if self.click_rect_center("PRODUCTION", delay=self.config.actions.menu_delay_seconds):
            if self._wait_for_state(self._production_panel_visible):
                return True
        if self.send_key(self.config.actions.open_production_key, delay=self.config.actions.menu_delay_seconds):
            if self._wait_for_state(self._production_panel_visible):
                return True
        return False

    @staticmethod
    def _numpad_vk(key: str) -> int | None:
        match = re.fullmatch(r"numpad\s+([0-9])", str(key or "").strip().lower())
        if not match:
            return None
        return 0x60 + int(match.group(1))

    @staticmethod
    def _press_vk(vk: int) -> None:
        user32 = ctypes.windll.user32
        user32.keybd_event(vk, 0, 0, 0)
        time.sleep(0.02)
        user32.keybd_event(vk, 0, 0x0002, 0)

    def send_key(self, key: str, *, delay: float | None = None) -> bool:
        if not ensure_focus(self.config.focus):
            return False
        normalized = self._normalize_key(key)
        try:
            vk = self._numpad_vk(normalized)
            if vk is not None:
                self._press_vk(vk)
            else:
                keyboard.send(normalized)
        except Exception:
            return False
        time.sleep(float(delay if delay is not None else self.config.actions.key_delay_seconds))
        self._invalidate_capture()
        return True

    def reset_ui(self) -> None:
        self.send_key(self.config.actions.open_resources_key, delay=self.config.actions.menu_delay_seconds)
        self.send_key(self.config.actions.open_resources_key, delay=self.config.actions.menu_delay_seconds)
        self.send_key(self.config.actions.open_production_key, delay=self.config.actions.menu_delay_seconds)
        self.send_key(self.config.actions.open_production_key, delay=self.config.actions.menu_delay_seconds)

    def open_planet_menu(self) -> bool:
        return self.send_key(self.config.actions.open_planet_menu_key, delay=self.config.actions.menu_delay_seconds)

    def close_planet_panel(self) -> bool:
        if self.click_rect_center("PLANET_PANEL_CLOSE", delay=self.config.actions.menu_delay_seconds):
            return True
        ok = self.send_key(self.config.actions.open_resources_key, delay=self.config.actions.menu_delay_seconds)
        ok = self.send_key(self.config.actions.open_resources_key, delay=self.config.actions.menu_delay_seconds) and ok
        return ok

    def open_ores_panel(self) -> bool:
        ok = self.send_key(self.config.actions.open_resources_key, delay=self.config.actions.menu_delay_seconds)
        ok = self.send_key(self.config.actions.ores_tab_key, delay=self.config.actions.menu_delay_seconds) and ok
        return ok

    def open_alloys_panel(self) -> bool:
        ok = self.send_key(self.config.actions.open_resources_key, delay=self.config.actions.menu_delay_seconds)
        ok = self.send_key(self.config.actions.alloys_tab_key, delay=self.config.actions.menu_delay_seconds) and ok
        return ok

    def open_items_panel(self) -> bool:
        ok = self.send_key(self.config.actions.open_resources_key, delay=self.config.actions.menu_delay_seconds)
        ok = self.send_key(self.config.actions.items_tab_key, delay=self.config.actions.menu_delay_seconds) and ok
        return ok

    def open_smelter_panel(self) -> bool:
        if not self._open_production_panel():
            return False
        if self._wait_for_state(lambda: self._production_tab_active("PRODUCTION_SMELT_TAB"), attempts=1):
            return True
        if self.click_rect_center("PRODUCTION_SMELT_TAB", delay=self.config.actions.menu_delay_seconds):
            if self._wait_for_state(lambda: self._production_tab_active("PRODUCTION_SMELT_TAB")):
                return True
        if self.send_key(self.config.actions.smelt_tab_key, delay=self.config.actions.menu_delay_seconds):
            if self._wait_for_state(lambda: self._production_tab_active("PRODUCTION_SMELT_TAB")):
                return True
        return False

    def open_crafter_panel(self) -> bool:
        if not self._open_production_panel():
            return False
        if self._wait_for_state(lambda: self._production_tab_active("PRODUCTION_CRAFT_TAB"), attempts=1):
            return True
        if self.click_rect_center("PRODUCTION_CRAFT_TAB", delay=self.config.actions.menu_delay_seconds):
            if self._wait_for_state(lambda: self._production_tab_active("PRODUCTION_CRAFT_TAB")):
                return True
        if self.send_key(self.config.actions.craft_tab_key, delay=self.config.actions.menu_delay_seconds):
            if self._wait_for_state(lambda: self._production_tab_active("PRODUCTION_CRAFT_TAB")):
                return True
        return False

    def close_ores_panel(self) -> None:
        # The ores task always opens the Resources panel first, so one toggle closes it.
        self.send_key(self.config.actions.open_resources_key, delay=self.config.actions.menu_delay_seconds)

    def scroll_resource_list_down(self) -> bool:
        return self.send_key(self.config.actions.scroll_down_key, delay=self.config.actions.scroll_delay_seconds)

    def scroll_resource_list_up(self) -> bool:
        return self.send_key(self.config.actions.scroll_up_key, delay=self.config.actions.scroll_delay_seconds)

    def increase_planet_stat(self, stat: str) -> bool:
        key = {
            "M": self.config.actions.increase_mining_key,
            "S": self.config.actions.increase_speed_key,
            "C": self.config.actions.increase_cargo_key,
        }.get(str(stat).upper())
        if not key:
            return False
        return self.send_key(key, delay=self.config.actions.key_delay_seconds)

    def select_ore_row(self, row_index: int) -> bool:
        key = self.config.actions.ore_select_keys.get(int(row_index))
        if not key:
            return False
        return self.send_key(key, delay=self.config.actions.key_delay_seconds)

    def choose_sell_fraction(self, fraction: float) -> bool:
        presets = sorted(self.config.actions.sell_fraction_keys.items(), key=lambda item: item[0])
        key = None
        for preset_fraction, preset_key in presets:
            if fraction >= preset_fraction:
                key = preset_key
        if key is None:
            return False
        return self.send_key(key, delay=self.config.actions.key_delay_seconds)

    def execute_sell(self) -> bool:
        return self.send_key(self.config.actions.sell_confirm_key, delay=self.config.actions.key_delay_seconds)

    def open_sell_dialog(self) -> bool:
        return self.send_key(self.config.actions.sell_open_key, delay=self.config.actions.key_delay_seconds)

    def _click_client_point(self, point: tuple[int, int], *, delay: float | None = None) -> bool:
        if not ensure_focus(self.config.focus):
            return False
        client = get_bluestacks_client_rect(self.config.capture.window_title)
        if client is None:
            return False
        abs_x = int(client.left + point[0])
        abs_y = int(client.top + point[1])
        self._send_mouse_input(x=abs_x, y=abs_y, flags=MOUSEEVENTF_MOVE)
        time.sleep(0.03)
        self._send_mouse_input(x=abs_x, y=abs_y, flags=MOUSEEVENTF_MOVE | MOUSEEVENTF_LEFTDOWN)
        time.sleep(0.03)
        self._send_mouse_input(x=abs_x, y=abs_y, flags=MOUSEEVENTF_MOVE | MOUSEEVENTF_LEFTUP)
        time.sleep(float(delay if delay is not None else self.config.actions.scroll_delay_seconds))
        self._invalidate_capture()
        return True

    def click_client_point(self, point: tuple[int, int], *, delay: float | None = None) -> bool:
        return self._click_client_point(point, delay=delay)

    def click_rect_center(self, rect_key: str, *, delay: float | None = None) -> bool:
        if self.rects is None:
            return False
        rect = self.rects.get(rect_key)
        if rect is None:
            return False
        x, y, w, h = rect
        return self._click_client_point((x + (w // 2), y + (h // 2)), delay=delay)

    def next_planet(self) -> bool:
        return self.click_rect_center("CYCLE_PLANETS_RIGHT", delay=self.config.actions.scroll_delay_seconds)

    def previous_planet(self) -> bool:
        return self.click_rect_center("CYCLE_PLANETS_LEFT", delay=self.config.actions.scroll_delay_seconds)
