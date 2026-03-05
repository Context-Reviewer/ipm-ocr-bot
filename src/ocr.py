from __future__ import annotations

import re
import time
import statistics
from pathlib import Path
from typing import Optional
from decimal import Decimal, InvalidOperation

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageGrab, ImageOps

import config

if config.TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD

_OUT_DIR = Path(__file__).resolve().parent / "out"
_OUT_DIR.mkdir(exist_ok=True)

_SUFFIX_MULTS = {
    "K": 1_000,
    "M": 1_000_000,
    "B": 1_000_000_000,
    "T": 1_000_000_000_000,
    "q": 1_000_000_000_000_000,
    "Q": 1_000_000_000_000_000_000,
    "s": 1_000_000_000_000_000_000_000,
    "S": 1_000_000_000_000_000_000_000_000,
    "O": 1_000_000_000_000_000_000_000_000_000,
    "k": 1_000,
    "m": 1_000_000,
    "b": 1_000_000_000,
    "t": 1_000_000_000_000,
}


def _bbox_key(bbox) -> str:
    try:
        x, y, w, h = bbox
        return f"x{x}_y{y}_w{w}_h{h}"
    except Exception:
        return "bbox_invalid"


def _sanitize(tag: str) -> str:
    if not tag:
        return ""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", tag).strip("_")


def _debug_save(mode: str, bbox, step: str, img, tag: str | None = None) -> None:
    if not getattr(config, "OCR_SNAP_DEBUG", False):
        return
    name = f"{mode}"
    if tag:
        name += f"_{_sanitize(tag)}"
    name += f"_{_bbox_key(bbox)}_{step}.png"

    if isinstance(img, Image.Image):
        arr = np.array(img)
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(_OUT_DIR / name), arr)
        return
    if isinstance(img, np.ndarray):
        cv2.imwrite(str(_OUT_DIR / name), img)


def capture_bbox(bbox) -> tuple[Optional[Image.Image], dict]:
    if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
        return None, {"ok": False, "reason": "invalid_bbox"}
    x, y, w, h = bbox
    try:
        if int(w) <= 0 or int(h) <= 0:
            return None, {"ok": False, "reason": "zero_size"}
        img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
    except Exception as exc:
        return None, {"ok": False, "reason": f"grab_error:{exc.__class__.__name__}"}
    if img is None:
        return None, {"ok": False, "reason": "grab_none"}
    try:
        if img.size[0] <= 0 or img.size[1] <= 0:
            return None, {"ok": False, "reason": "empty_image"}
    except Exception:
        return None, {"ok": False, "reason": "size_error"}
    return img, {"ok": True, "reason": "ok"}


def validate_crop(img, bbox, mode: str) -> tuple[bool, str]:
    if img is None:
        return False, "img_none"
    if isinstance(img, Image.Image):
        if img.size[0] <= 0 or img.size[1] <= 0:
            return False, "img_empty"
        return True, "ok"
    if isinstance(img, np.ndarray):
        if img.size == 0:
            return False, "arr_empty"
        return True, "ok"
    return False, "unsupported_type"


def parse_compact_number(text: str | None) -> Optional[int]:
    if text is None:
        return None
    cleaned = text.strip().replace(",", "").replace(" ", "").replace("\n", "").replace("$", "")
    if not cleaned:
        return None
    match = re.match(r"^(\d+(?:\.\d+)?)([KMBTqQsSOkmbt]?)$", cleaned)
    if not match:
        return None
    num_str, suffix = match.groups()
    try:
        value = Decimal(num_str)
    except (InvalidOperation, ValueError):
        return None
    mult = Decimal(_SUFFIX_MULTS.get(suffix, 1))
    return int(value * mult)


def _to_bgr(img) -> Optional[np.ndarray]:
    if img is None:
        return None
    if isinstance(img, Image.Image):
        arr = np.array(img)
    else:
        arr = img
    if arr is None or arr.size == 0:
        return None
    if arr.ndim == 2:
        try:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        except Exception:
            return None
    if arr.shape[2] == 4:
        try:
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        except Exception:
            return None
    try:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def _prep_hud_cash(img) -> Optional[np.ndarray]:
    bgr = _to_bgr(img)
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    if h <= 0 or w <= 0:
        return None
    up = cv2.resize(bgr, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    return bw


def _prep_ore_qty(img) -> Optional[np.ndarray]:
    if img is None:
        return None
    if not isinstance(img, Image.Image):
        try:
            img = Image.fromarray(img)
        except Exception:
            return None
    gray = ImageOps.grayscale(img)
    contrast = ImageOps.autocontrast(gray)
    bw = contrast.point(lambda p: 255 if p > 160 else 0)
    arr = np.array(bw)
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(arr, kernel, iterations=1)
    return dilated


def _prep_generic(img) -> Optional[np.ndarray]:
    bgr = _to_bgr(img)
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    if h <= 0 or w <= 0:
        return None
    up = cv2.resize(bgr, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return bw


def _ocr_text(img: np.ndarray, *, psm: int, whitelist: str) -> str:
    cfg = f"--psm {psm} -c tessedit_char_whitelist={whitelist}"
    return pytesseract.image_to_string(img, config=cfg).strip()


def _log_fail(mode: str, bbox, reason: str) -> None:
    print(f"[OCR] mode={mode} bbox={bbox} capture=empty (reason={reason}) -> None")


def _read_number_once(bbox, *, mode: str, debug_tag: str | None = None) -> Optional[int]:
    img, meta = capture_bbox(bbox)
    ok, reason = validate_crop(img, bbox, mode)
    if not ok:
        _log_fail(mode, bbox, reason)
        return None

    _debug_save(mode, bbox, "raw", img, debug_tag)

    if mode == "hud_cash":
        bw = _prep_hud_cash(img)
        whitelist = "0123456789.,$KMBTqQsSO"
        psm = 7
    elif mode == "ore_qty":
        bw = _prep_ore_qty(img)
        whitelist = "0123456789.,KMBTqQsSO"
        psm = 7
    else:
        bw = _prep_generic(img)
        whitelist = "0123456789.,$KMBTqQsSO"
        psm = 6

    if bw is None or (isinstance(bw, np.ndarray) and bw.size == 0):
        _log_fail(mode, bbox, "prep_empty")
        return None

    _debug_save(mode, bbox, "bw", bw, debug_tag)

    text = _ocr_text(bw, psm=psm, whitelist=whitelist)
    value = parse_compact_number(text)
    if value is None:
        print(f"[OCR] mode={mode} bbox={bbox} text=\"{text}\" parsed=None -> None")
        return None
    return value


def _read_number_with_offsets(bbox, offsets, *, mode: str, debug_tag: str | None = None) -> Optional[int]:
    x, y, w, h = bbox
    for dy in offsets:
        shifted = (x, y + dy, w, h)
        val = _read_number_once(shifted, mode=mode, debug_tag=debug_tag)
        if val is not None:
            return val
    return None


def _read_ore_qty_median(bbox, debug_tag: str | None = None) -> Optional[int]:
    values = []
    for i in range(config.ORE_QTY_SAMPLES):
        val = _read_number_with_offsets(
            bbox,
            config.OCR_QTY_Y_OFFSETS,
            mode="ore_qty",
            debug_tag=f"{debug_tag}_s{i}" if debug_tag else None,
        )
        if val is not None:
            values.append(val)
        if i < config.ORE_QTY_SAMPLES - 1:
            time.sleep(config.ORE_QTY_SAMPLE_DELAY)

    if len(values) < config.ORE_QTY_MIN_VALID_SAMPLES:
        print(f"[OCR] mode=ore_qty bbox={bbox} samples={len(values)} < min_valid -> None")
        return None

    median_val = statistics.median(values)
    if median_val == 0:
        return 0 if max(values) == 0 else None

    rel_spread = (max(values) - min(values)) / median_val
    if rel_spread > config.ORE_QTY_MAX_REL_SPREAD:
        print(f"[OCR] mode=ore_qty bbox={bbox} spread={rel_spread:.3f} -> None")
        return None

    return int(median_val)


def ocr_read_number(bbox, *, mode: str, debug_tag: str | None = None) -> Optional[int]:
    if bbox is None:
        _log_fail(mode, bbox, "bbox_none")
        return None
    if mode == "ore_qty":
        return _read_ore_qty_median(bbox, debug_tag)
    return _read_number_once(bbox, mode=mode, debug_tag=debug_tag)

def preprocess_for_mode(img, mode: str) -> Optional[np.ndarray]:
    if mode == "ore_qty":
        return _prep_ore_qty(img)
    if mode == "hud_cash":
        return _prep_hud_cash(img)
    return _prep_generic(img)
