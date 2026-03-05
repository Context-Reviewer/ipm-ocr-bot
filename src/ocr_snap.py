from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pytesseract
from PIL import Image

import config
from ocr_utils import grab_bbox, parse_compact_number

if config.TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD

_OUT_DIR = Path(__file__).resolve().parent / "out"
_OUT_DIR.mkdir(exist_ok=True)

_LV_RE = re.compile(r"Lv\.?\s*(\d+)", re.IGNORECASE)


def _maybe_save(name: str, img: np.ndarray) -> None:
    if not getattr(config, "OCR_SNAP_DEBUG", False):
        return
    cv2.imwrite(str(_OUT_DIR / name), img)


def _sanitize(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")


def _resolve_bbox(label_or_bbox):
    if isinstance(label_or_bbox, (tuple, list)) and len(label_or_bbox) == 4:
        return tuple(label_or_bbox)
    if isinstance(label_or_bbox, str):
        if hasattr(config, label_or_bbox):
            return getattr(config, label_or_bbox)
        bboxes = getattr(config, "OCR_SNAP_BBOXES", None)
        if isinstance(bboxes, dict) and label_or_bbox in bboxes:
            return bboxes[label_or_bbox]
    raise ValueError(f"Unknown bbox label: {label_or_bbox}")


def _to_bgr(img) -> np.ndarray:
    if isinstance(img, Image.Image):
        arr = np.array(img)
    else:
        arr = img
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


# ---- Common preprocessing ----

def _prep_text(img_bgr: np.ndarray, scale: int = 3) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    up = cv2.resize(img_bgr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    return bw


def _ocr_text(bw: np.ndarray, psm: int = 6, whitelist: str | None = None) -> str:
    cfg = f"--psm {psm}"
    if whitelist:
        cfg += f" -c tessedit_char_whitelist={whitelist}"
    return pytesseract.image_to_string(bw, config=cfg).strip()


# ---- Public snapshot helpers ----

def snap(label_or_bbox) -> np.ndarray:
    """Grab a bbox by label or raw bbox (x, y, w, h)."""
    bbox = _resolve_bbox(label_or_bbox)
    img = grab_bbox(bbox)
    bgr = _to_bgr(img)
    name = _sanitize(str(label_or_bbox))
    _maybe_save(f"{name}_raw.png", bgr)
    return bgr


# ---- Typed readers ----

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
    panel = snap(panel_label)
    h, w = panel.shape[:2]

    blocks = [
        panel[0 : h // 3, 0:w],
        panel[h // 3 : (2 * h) // 3, 0:w],
        panel[(2 * h) // 3 : h, 0:w],
    ]

    vals: list[int] = []
    for i, b in enumerate(blocks, start=1):
        bw = _prep_text(b, scale=4)
        _maybe_save(f"{panel_label}_block{i}_bw.png", bw)

        txt = _ocr_text(bw, psm=6, whitelist="Lv.0123456789")
        m = _LV_RE.search(txt)
        if not m:
            _maybe_save(f"{panel_label}_block{i}_raw.png", b)
            return None
        try:
            v = int(m.group(1))
        except ValueError:
            return None
        if v <= 0:
            return None
        vals.append(v)

    return PlanetLevels(mining=vals[0], speed=vals[1], cargo=vals[2])

def read_hud_cash() -> Optional[int]:
    bbox = getattr(config, "RECT_HUD_CASH", None)
    if not bbox:
        return None
    img = grab_bbox(bbox)
    bgr = _to_bgr(img)
    _maybe_save("HUD_CASH_raw.png", bgr)
    bw = _prep_text(bgr, scale=3)
    _maybe_save("HUD_CASH_bw.png", bw)
    txt = _ocr_text(bw, psm=7, whitelist="0123456789.,$KMBTqQsSO")
    return parse_compact_number(txt)


def read_planet_resource_table() -> Optional[list[dict]]:
    """Placeholder for future planet resource table OCR."""
    return None


def read_cash() -> Optional[int]:
    """Placeholder for future cash OCR."""
    return None
