from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import config
import ocr


@dataclass(frozen=True)
class PlanetLevels:
    mining: int
    speed: int
    cargo: int


def read_planet_levels(panel_label: str = "PLANET_STATS_PANEL") -> Optional[PlanetLevels]:
    """
    Reads Mining/Speed/Cargo levels from the planet stats panel.
    Fail-closed: returns None if any field can't be parsed.
    """
    panel_bbox = getattr(config, panel_label, None)
    if not isinstance(panel_bbox, (tuple, list)) or len(panel_bbox) != 4:
        return None
    x, y, w, h = panel_bbox
    if w <= 0 or h <= 0:
        return None

    h1 = h // 3
    h2 = h // 3
    h3 = h - h1 - h2
    blocks = [
        (x, y, w, h1),
        (x, y + h1, w, h2),
        (x, y + h1 + h2, w, h3),
    ]

    vals: list[int] = []
    for i, bbox in enumerate(blocks, start=1):
        v = ocr.ocr_read_number(bbox, mode="generic", debug_tag=f"planet_level_{i}")
        if v is None:
            return None
        try:
            iv = int(v)
        except Exception:
            return None
        if iv <= 0:
            return None
        vals.append(iv)

    return PlanetLevels(mining=vals[0], speed=vals[1], cargo=vals[2])


def read_hud_cash() -> Optional[int]:
    bbox = getattr(config, "RECT_HUD_CASH", None)
    return ocr.ocr_read_number(bbox, mode="hud_cash", debug_tag="hud_cash")
