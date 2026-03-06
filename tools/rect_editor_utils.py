from __future__ import annotations

import colorsys
import hashlib
from typing import Tuple


def color_for_name(name: str) -> Tuple[int, int, int]:
    """Deterministic RGB color from name."""
    if not name:
        return (0, 200, 200)
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()
    hue = int(digest[:8], 16) % 360
    r, g, b = colorsys.hsv_to_rgb(hue / 360.0, 0.65, 0.9)
    return (int(r * 255), int(g * 255), int(b * 255))


def snap_value(value: float, grid: int) -> float:
    if grid <= 0:
        return value
    return round(value / grid) * grid


def snap_rect(x: float, y: float, w: float, h: float, grid: int) -> tuple[float, float, float, float]:
    return (
        snap_value(x, grid),
        snap_value(y, grid),
        snap_value(w, grid),
        snap_value(h, grid),
    )
