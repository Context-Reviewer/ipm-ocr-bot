from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Optional

import config
import ocr
import perception
import template_number_reader


@dataclass(frozen=True)
class PlanetLevels:
    mining: int
    speed: int
    cargo: int


_LEVEL_RE = re.compile(r"(?:Lv\.?\s*)?(\d+)", re.IGNORECASE)
_ROW_LEVEL_RE = re.compile(r"Lv\.?\s*(\d+)", re.IGNORECASE)
_PLANET_TITLE_ID_RE = re.compile(r"^\D*(\d{1,2})\b")
_PLANET_TITLE_TEXT_RE = re.compile(r"[^0-9A-Z. ]+")


def _planet_title_number_bbox() -> Optional[tuple[int, int, int, int]]:
    title_bbox = getattr(config, "PLANET_TITLE", "PLANET_TITLE")
    resolved = ocr.resolve_bbox(title_bbox)
    if not isinstance(resolved, (tuple, list)) or len(resolved) != 4:
        return None
    x, y, w, h = resolved
    width_ratio = float(getattr(config, "PLANET_TITLE_NUMBER_WIDTH_RATIO", 0.36))
    pad_x = int(getattr(config, "PLANET_TITLE_NUMBER_PAD_X", 4))
    pad_y = int(getattr(config, "PLANET_TITLE_NUMBER_PAD_Y", 4))
    num_w = max(12, int(round(w * width_ratio)))
    return (
        int(x + pad_x),
        int(y + pad_y),
        max(1, int(num_w - (2 * pad_x))),
        max(1, int(h - (2 * pad_y))),
    )


def _parse_planet_title_id_from_text(text: str | None) -> Optional[int]:
    if not text:
        return None
    first_line = str(text).splitlines()[0].strip()
    match = _PLANET_TITLE_ID_RE.search(first_line)
    if not match:
        return None
    try:
        value = int(match.group(1))
    except Exception:
        return None
    return value if value > 0 else None


def _normalize_planet_title_text(text: str | None) -> str:
    if not text:
        return ""
    normalized = str(text).upper().replace("\n", " ").strip()
    normalized = normalized.translate(
        str.maketrans({
            "|": "I",
            "!": "I",
            ":": ".",
            ";": ".",
            ",": ".",
        })
    )
    normalized = _PLANET_TITLE_TEXT_RE.sub(" ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _read_planet_title_text_from_bbox(
    bbox,
    *,
    mode: str = "planet_title",
    prefer_template: bool = False,
) -> str:
    if bbox is None:
        return ""

    if prefer_template:
        img, _meta = ocr.capture_bbox(bbox)
        ok, _reason = ocr.validate_crop(img, bbox, mode)
        if ok and img is not None:
            template_text, _template_score = template_number_reader.read_text(img, mode=mode)
            normalized = _normalize_planet_title_text(template_text)
            if normalized:
                return normalized

    text = _normalize_planet_title_text(ocr.ocr_read_text(bbox, mode=mode))
    if text:
        return text

    dbg = ocr.ocr_read_debug(bbox, mode=mode)
    return _normalize_planet_title_text(dbg.get("text"))


def _read_planet_title_number_text() -> str:
    number_bbox = _planet_title_number_bbox()
    if number_bbox is None:
        return ""

    text = _read_planet_title_text_from_bbox(number_bbox, mode="planet_title", prefer_template=True)
    if _parse_planet_title_id_from_text(text) is not None:
        return text

    return ""


def read_planet_title_text() -> str:
    title_bbox = getattr(config, "PLANET_TITLE", "PLANET_TITLE")
    texts: list[str] = []

    text = _read_planet_title_number_text()
    if text:
        texts.append(text)

    vision_text = _normalize_planet_title_text(perception.read_planet_title_text(title_bbox).text)
    if vision_text:
        texts.append(vision_text)

    text = _read_planet_title_text_from_bbox(title_bbox, mode="planet_title")
    if text:
        texts.append(text)

    if not texts:
        text = _read_planet_title_text_from_bbox(title_bbox, mode="generic")
        if text:
            texts.append(text)

    if not texts:
        return ""

    def _score_title(candidate: str) -> tuple[int, int]:
        has_id = 1 if _parse_planet_title_id_from_text(candidate) is not None else 0
        return (has_id, len(candidate))

    texts.sort(key=_score_title, reverse=True)
    return texts[0]


def _read_level_from_bbox(bbox, tag: str) -> Optional[int]:
    # Primary: single centered crop with the template matcher. The tiny level fields
    # are sensitive to stabilization/offset probes, so trust the centered crop first.
    img, meta = ocr.capture_bbox(bbox)
    ok, _reason = ocr.validate_crop(img, bbox, "level")
    if ok and img is not None:
        text, _score = template_number_reader.read_text(img, mode="level")
        if text is not None:
            try:
                iv = int(text)
            except Exception:
                iv = None
            if iv and iv > 0:
                return iv

    # Secondary: digit-only OCR with a few offsets for alignment drift.
    try:
        x, y, w, h = bbox
    except Exception:
        x = y = w = h = None
    if x is not None:
        for dy in (0, -3, 3):
            for dx in (0, -2, 2):
                v = ocr.ocr_read_number((x + dx, y + dy, w, h), mode="level", debug_tag=tag)
                if v is not None:
                    try:
                        iv = int(v)
                    except Exception:
                        iv = None
                    if iv and iv > 0:
                        return iv

    # Fallback: parse digits from generic OCR text (handles "Lv. 41").
    dbg = ocr.ocr_read_debug(bbox, mode="generic")
    text = (dbg.get("text") or "").strip()
    if text:
        m = _LEVEL_RE.search(text)
        if m:
            try:
                iv = int(m.group(1))
                return iv if iv > 0 else None
            except Exception:
                return None
    return None


def _row_bbox_from_fields(level_key: str, value_key: str) -> Optional[tuple[int, int, int, int]]:
    explicit_key = {
        ("MINING_LVL", "MINING_RATE"): "MINING_ROW",
        ("SHIP_LVL", "SHIP_SPEED"): "SHIP_ROW",
        ("CARGO_LVL", "CARGO_CAPACITY"): "CARGO_ROW",
    }.get((level_key, value_key))
    if explicit_key:
        explicit_rect = ocr.resolve_bbox(explicit_key)
        if isinstance(explicit_rect, (tuple, list)) and len(explicit_rect) == 4:
            return explicit_rect

    level_rect = ocr.resolve_bbox(level_key)
    value_rect = ocr.resolve_bbox(value_key)
    if not all(isinstance(r, (tuple, list)) and len(r) == 4 for r in (level_rect, value_rect)):
        return None
    lx, ly, lw, lh = level_rect
    vx, vy, vw, vh = value_rect
    left = min(lx, vx) - int(getattr(config, "PLANET_ROW_PAD_LEFT", 40))
    top = min(ly, vy) - int(getattr(config, "PLANET_ROW_PAD_TOP", 18))
    right = max(lx + lw, vx + vw) + int(getattr(config, "PLANET_ROW_PAD_RIGHT", 14))
    bottom = max(ly + lh, vy + vh) + int(getattr(config, "PLANET_ROW_PAD_BOTTOM", 10))
    return (left, top, max(1, right - left), max(1, bottom - top))


def _read_level_from_row(row_bbox, tag: str) -> Optional[int]:
    text = ocr.ocr_read_text(row_bbox, mode="generic")
    if text:
        m = _ROW_LEVEL_RE.search(text)
        if m:
            try:
                value = int(m.group(1))
            except Exception:
                value = None
            if value and value > 0:
                return value
    dbg = ocr.ocr_read_debug(row_bbox, mode="generic")
    text = (dbg.get("text") or "").strip()
    if not text:
        return None
    m = _ROW_LEVEL_RE.search(text)
    if not m:
        return None
    try:
        value = int(m.group(1))
    except Exception:
        return None
    return value if value > 0 else None


def _read_level_template_only(bbox) -> Optional[int]:
    img, _meta = ocr.capture_bbox(bbox)
    ok, _reason = ocr.validate_crop(img, bbox, "level")
    if not ok or img is None:
        return None
    text, _score = template_number_reader.read_text(img, mode="level")
    if text is None:
        return None
    try:
        value = int(text)
    except Exception:
        return None
    return value if value > 0 else None


def read_planet_levels(panel_label: str = "PLANET_STATS_PANEL") -> Optional[PlanetLevels]:
    """
    Reads Mining/Speed/Cargo levels from the planet stats panel.
    Fail-closed: returns None if any field can't be parsed.
    """
    mining_row = _row_bbox_from_fields("MINING_LVL", "MINING_RATE")
    ship_row = _row_bbox_from_fields("SHIP_LVL", "SHIP_SPEED")
    cargo_row = _row_bbox_from_fields("CARGO_LVL", "CARGO_CAPACITY")
    if all(isinstance(r, (tuple, list)) and len(r) == 4 for r in (mining_row, ship_row, cargo_row)):
        m = _read_level_from_row(mining_row, "planet_mining_row")
        s = _read_level_from_row(ship_row, "planet_ship_row")
        c = _read_level_from_row(cargo_row, "planet_cargo_row")
        if m and s and c:
            return PlanetLevels(mining=m, speed=s, cargo=c)

    # Fallback to explicit per-field rects.
    mining_rect = ocr.resolve_bbox("MINING_LVL")
    speed_rect = ocr.resolve_bbox("SHIP_LVL")
    cargo_rect = ocr.resolve_bbox("CARGO_LVL")
    if all(isinstance(r, (tuple, list)) and len(r) == 4 for r in (mining_rect, speed_rect, cargo_rect)):
        m = _read_level_from_bbox(mining_rect, "planet_mining_lvl")
        s = _read_level_from_bbox(speed_rect, "planet_ship_lvl")
        c = _read_level_from_bbox(cargo_rect, "planet_cargo_lvl")
        if m and s and c:
            return PlanetLevels(mining=m, speed=s, cargo=c)
        return None

    panel_spec = getattr(config, panel_label, panel_label)
    panel_bbox = ocr.resolve_bbox(panel_spec)
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


def read_planet_levels_fast() -> Optional[PlanetLevels]:
    mining_row = _row_bbox_from_fields("MINING_LVL", "MINING_RATE")
    ship_row = _row_bbox_from_fields("SHIP_LVL", "SHIP_SPEED")
    cargo_row = _row_bbox_from_fields("CARGO_LVL", "CARGO_CAPACITY")
    if all(isinstance(r, (tuple, list)) and len(r) == 4 for r in (mining_row, ship_row, cargo_row)):
        m = _read_level_from_row(mining_row, "planet_mining_row_fast")
        s = _read_level_from_row(ship_row, "planet_ship_row_fast")
        c = _read_level_from_row(cargo_row, "planet_cargo_row_fast")
        if m and s and c:
            return PlanetLevels(mining=m, speed=s, cargo=c)

    mining_rect = ocr.resolve_bbox("MINING_LVL")
    speed_rect = ocr.resolve_bbox("SHIP_LVL")
    cargo_rect = ocr.resolve_bbox("CARGO_LVL")
    if not all(isinstance(r, (tuple, list)) and len(r) == 4 for r in (mining_rect, speed_rect, cargo_rect)):
        return None

    m = _read_level_template_only(mining_rect)
    s = _read_level_template_only(speed_rect)
    c = _read_level_template_only(cargo_rect)
    if m and s and c:
        return PlanetLevels(mining=m, speed=s, cargo=c)
    return None


def read_hud_cash() -> Optional[int]:
    bbox = getattr(config, "RECT_HUD_CASH", "HUD_CASH")
    value = ocr.ocr_read_number(bbox, mode="hud_cash", debug_tag="hud_cash")
    if value is not None:
        return value
    try:
        value, _result = perception.read_number_value(
            bbox,
            mode="hud_cash",
            prompt=str(getattr(config, "PERCEPTION_HUD_PRICE_PROMPT", "")),
        )
    except Exception:
        value = None
    return value


def read_upgrade_button_cost(stat: str) -> Optional[int]:
    rect_key = {
        "M": "UPGRADE_MINING",
        "S": "UPGRADE_SPEED",
        "C": "UPGRADE_CARGO",
    }.get(str(stat).upper())
    if not rect_key:
        return None
    value = ocr.ocr_read_number(rect_key, mode="hud_cash", debug_tag=f"upgrade_cost_{stat.lower()}")
    if value is not None:
        return value
    try:
        value, _result = perception.read_number_value(
            rect_key,
            mode="hud_cash",
            prompt=str(getattr(config, "PERCEPTION_HUD_PRICE_PROMPT", "")),
        )
    except Exception:
        value = None
    return value


def read_planet_title_id() -> Optional[int]:
    fast_value = _parse_planet_title_id_from_text(_read_planet_title_number_text())
    if fast_value is not None:
        return fast_value
    return _parse_planet_title_id_from_text(read_planet_title_text())


def read_planet_title_text_stable(*, samples: int | None = None, delay: float | None = None) -> str:
    sample_count = max(1, int(samples or getattr(config, "PLANET_TITLE_READ_SAMPLES", 3)))
    sample_delay = max(0.0, float(delay if delay is not None else getattr(config, "PLANET_TITLE_READ_DELAY", 0.04)))
    counts: dict[str, int] = {}

    for idx in range(sample_count):
        value = read_planet_title_text()
        if value:
            counts[value] = counts.get(value, 0) + 1
        if idx < sample_count - 1 and sample_delay > 0:
            time.sleep(sample_delay)

    if not counts:
        return ""

    ranked = sorted(
        counts.items(),
        key=lambda item: (-item[1], -len(item[0]), item[0]),
    )
    best_value, best_count = ranked[0]
    if best_count >= 2 or sum(counts.values()) == 1:
        return best_value
    return ""


def read_planet_title_id_stable(*, samples: int | None = None, delay: float | None = None) -> Optional[int]:
    sample_count = max(1, int(samples or getattr(config, "PLANET_TITLE_READ_SAMPLES", 3)))
    sample_delay = max(0.0, float(delay if delay is not None else getattr(config, "PLANET_TITLE_READ_DELAY", 0.04)))
    counts: dict[int, int] = {}
    for idx in range(sample_count):
        value = _parse_planet_title_id_from_text(_read_planet_title_number_text())
        if value is not None:
            counts[value] = counts.get(value, 0) + 1
        if idx < sample_count - 1 and sample_delay > 0:
            time.sleep(sample_delay)

    if counts:
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        best_value, best_count = ranked[0]
        if best_count >= 2 or sum(counts.values()) == 1:
            return best_value

    text = read_planet_title_text_stable(samples=samples, delay=delay)
    if text:
        value = _parse_planet_title_id_from_text(text)
        if value is not None:
            return value
    return None
