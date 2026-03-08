from __future__ import annotations

import time
import ctypes
import re

import keyboard

from .config import RuntimeConfig
from .focus import ensure_focus
from .rects import RectStore
from window_win32 import get_bluestacks_client_rect


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

    def close_ores_panel(self) -> None:
        # The ores task always opens the Resources panel first, so one toggle closes it.
        self.send_key(self.config.actions.open_resources_key, delay=self.config.actions.menu_delay_seconds)

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
        user32 = ctypes.windll.user32
        user32.SetCursorPos(abs_x, abs_y)
        time.sleep(0.03)
        user32.mouse_event(0x0002, 0, 0, 0, 0)
        time.sleep(0.03)
        user32.mouse_event(0x0004, 0, 0, 0, 0)
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
