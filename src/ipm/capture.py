from __future__ import annotations

from dataclasses import dataclass
import io
import subprocess
import time
from typing import Optional, Protocol

from PIL import Image, ImageGrab

from .rects import Rect
from window_win32 import get_bluestacks_client_rect


class CaptureBackend(Protocol):
    name: str

    def available(self) -> bool:
        ...

    def capture_screen(self) -> Optional[Image.Image]:
        ...

    def capture_bbox(self, bbox: Rect) -> Optional[Image.Image]:
        ...

    def capture_client_bbox(self, bbox: Rect) -> Optional[Image.Image]:
        ...

    def invalidate(self) -> None:
        ...


@dataclass(slots=True)
class DesktopCaptureBackend:
    name: str = "desktop"
    window_title: str = "BlueStacks App Player"
    cache_ttl_seconds: float = 0.2
    _cached_at: float = 0.0
    _cached_screen: Optional[Image.Image] = None

    def available(self) -> bool:
        return get_bluestacks_client_rect(self.window_title) is not None

    def invalidate(self) -> None:
        self._cached_at = 0.0
        self._cached_screen = None

    def capture_screen(self) -> Optional[Image.Image]:
        now = time.monotonic()
        if self._cached_screen is not None and (now - self._cached_at) <= self.cache_ttl_seconds:
            return self._cached_screen.copy()
        rect = get_bluestacks_client_rect(self.window_title)
        if rect is None:
            return None
        try:
            image = ImageGrab.grab(
                bbox=(rect.left, rect.top, rect.left + rect.width, rect.top + rect.height)
            ).convert("RGB")
        except Exception:
            return None
        self._cached_at = now
        self._cached_screen = image
        return image.copy()

    def capture_bbox(self, bbox: Rect) -> Optional[Image.Image]:
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return None
        try:
            return ImageGrab.grab(bbox=(x, y, x + w, y + h)).convert("RGB")
        except Exception:
            return None

    def capture_client_bbox(self, bbox: Rect) -> Optional[Image.Image]:
        frame = self.capture_screen()
        if frame is None:
            return None
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return None
        return frame.crop((x, y, x + w, y + h))


@dataclass(slots=True)
class AdbCaptureBackend:
    name: str = "adb"
    adb_path: str = "adb"
    serial: str = ""
    target_resolution: tuple[int, int] | None = None
    cache_ttl_seconds: float = 0.2
    _cached_at: float = 0.0
    _cached_screen: Optional[Image.Image] = None

    def _adb_command(self) -> list[str]:
        command = [self.adb_path]
        if self.serial:
            command.extend(["-s", self.serial])
        return command

    def available(self) -> bool:
        try:
            result = subprocess.run(
                self._adb_command() + ["get-state"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except Exception:
            return False
        return result.returncode == 0 and "device" in (result.stdout or "").lower()

    def invalidate(self) -> None:
        self._cached_at = 0.0
        self._cached_screen = None

    def capture_screen(self) -> Optional[Image.Image]:
        now = time.monotonic()
        if self._cached_screen is not None and (now - self._cached_at) <= self.cache_ttl_seconds:
            return self._cached_screen.copy()
        try:
            result = subprocess.run(
                self._adb_command() + ["exec-out", "screencap", "-p"],
                capture_output=True,
                timeout=8,
                check=False,
            )
        except Exception:
            return None
        if result.returncode != 0 or not result.stdout:
            return None
        try:
            image = Image.open(io.BytesIO(result.stdout)).convert("RGB")
            image.load()
        except Exception:
            return None
        self._cached_at = now
        self._cached_screen = image
        return image.copy()

    def capture_bbox(self, bbox: Rect) -> Optional[Image.Image]:
        frame = self.capture_screen()
        if frame is None:
            return None
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return None
        return frame.crop((x, y, x + w, y + h))

    def capture_client_bbox(self, bbox: Rect) -> Optional[Image.Image]:
        frame = self.capture_screen()
        if frame is None:
            return None
        x, y, w, h = bbox
        if self.target_resolution:
            src_w, src_h = frame.size
            tgt_w, tgt_h = self.target_resolution
            if tgt_w > 0 and tgt_h > 0 and (src_w != tgt_w or src_h != tgt_h):
                scale_x = src_w / tgt_w
                scale_y = src_h / tgt_h
                x = int(round(x * scale_x))
                y = int(round(y * scale_y))
                w = int(round(w * scale_x))
                h = int(round(h * scale_y))
        if w <= 0 or h <= 0:
            return None
        return frame.crop((x, y, x + w, y + h))


def create_capture_backend(
    name: str,
    *,
    serial: str = "",
    adb_path: str = "adb",
    target_resolution: tuple[int, int] | None = None,
    window_title: str = "BlueStacks App Player",
    cache_ttl_seconds: float = 0.2,
) -> CaptureBackend:
    normalized = str(name or "desktop").lower()
    if normalized == "adb":
        return AdbCaptureBackend(
            adb_path=adb_path,
            serial=serial,
            target_resolution=target_resolution,
            cache_ttl_seconds=cache_ttl_seconds,
        )
    return DesktopCaptureBackend(
        window_title=window_title,
        cache_ttl_seconds=cache_ttl_seconds,
    )
