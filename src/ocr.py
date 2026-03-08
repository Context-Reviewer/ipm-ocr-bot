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
import template_number_reader
from rect_store import RectStore
from window_win32 import get_bluestacks_client_rect

if config.TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD

_OUT_DIR = Path(__file__).resolve().parent / "out"
_OUT_DIR.mkdir(exist_ok=True)

_RECT_STORE: RectStore | None = None
_RAPIDOCR_ENGINE = None
_RAPIDOCR_UNAVAILABLE = False

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

_PLANET_TITLE_TEXT_RE = re.compile(r"[^0-9A-Z. ]+")


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


def _load_rects() -> RectStore | None:
    global _RECT_STORE
    if _RECT_STORE is not None:
        return _RECT_STORE
    path = Path(getattr(config, "RECTS_JSON_PATH", "rects.json"))
    if not path.exists():
        return None
    _RECT_STORE = RectStore.load(path)
    return _RECT_STORE


def rel_to_screen_bbox(rel_bbox, title_hint: str = "BlueStacks App Player"):
    c = get_bluestacks_client_rect(title_hint)
    if not c:
        return None
    x, y, w, h = rel_bbox
    return (c.left + x, c.top + y, w, h)


def _resolve_bbox(bbox):
    title_hint = getattr(config, "BLUESTACKS_TITLE_HINT", "BlueStacks App Player")
    if isinstance(bbox, str):
        store = _load_rects()
        if store is None:
            return None, "rects_missing"
        rect = store.rects.get(bbox)
        if rect is None:
            return None, "rect_not_found"
        screen_bbox = rel_to_screen_bbox(rect, title_hint=title_hint)
        return (screen_bbox, "ok") if screen_bbox else (None, "client_not_found")

    if isinstance(bbox, (tuple, list)) and len(bbox) == 4:
        if getattr(config, "RECTS_USE_CLIENT", False):
            screen_bbox = rel_to_screen_bbox(bbox, title_hint=title_hint)
            return (screen_bbox, "ok") if screen_bbox else (None, "client_not_found")
        return bbox, "ok"

    return None, "invalid_bbox"


def resolve_bbox(bbox):
    resolved, _reason = _resolve_bbox(bbox)
    return resolved


def resolve_client_bbox(bbox):
    if isinstance(bbox, str):
        store = _load_rects()
        if store is None:
            return None
        rect = store.rects.get(bbox)
        if isinstance(rect, (tuple, list)) and len(rect) == 4:
            return tuple(int(v) for v in rect)
        return None
    if isinstance(bbox, (tuple, list)) and len(bbox) == 4:
        return tuple(int(v) for v in bbox)
    return None


def capture_bbox(bbox) -> tuple[Optional[Image.Image], dict]:
    resolved, reason = _resolve_bbox(bbox)
    if not resolved:
        return None, {"ok": False, "reason": reason}
    x, y, w, h = resolved
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
    cleaned = text.strip().replace(" ", "").replace("\n", "").replace("$", "")
    if not cleaned:
        return None

    candidates = [cleaned]
    token_matches = re.findall(r"[0-9A-Za-z.,]+", cleaned)
    candidates.extend(token_matches)

    for candidate in candidates:
        value = _parse_compact_candidate(candidate)
        if value is not None:
            return value
    return None


def parse_compact_number_for_mode(text: str | None, *, mode: str) -> Optional[int]:
    if mode != "ore_qty":
        return parse_compact_number(text)
    if text is None:
        return None
    cleaned = text.strip().replace(" ", "").replace("\n", "").replace("$", "")
    if not cleaned:
        return None

    candidates = [cleaned]
    candidates.extend(re.findall(r"[0-9A-Za-z.,]+", cleaned))
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        value = _parse_compact_candidate(
            candidate,
            digit_map={
                "O": "0",
                "o": "0",
                "Q": "0",
                "D": "0",
                "I": "1",
                "l": "1",
                "|": "1",
                "!": "1",
                "S": "5",
                "s": "5",
                "B": "8",
            },
            allowed_suffixes={"", "K", "M", "B", "T", "k", "m", "b", "t"},
            infer_decimal=True,
        )
        if value is not None:
            return value
    return None


def _parse_compact_candidate(
    candidate: str,
    *,
    digit_map: dict[str, str] | None = None,
    allowed_suffixes: set[str] | None = None,
    infer_decimal: bool = False,
) -> Optional[int]:
    if not candidate:
        return None

    candidate = candidate.strip().strip(".,:;")
    if not candidate:
        return None

    suffix = ""
    if candidate and candidate[-1] in _SUFFIX_MULTS:
        suffix = candidate[-1]
        body = candidate[:-1]
    else:
        body = candidate

    if allowed_suffixes is not None and suffix not in allowed_suffixes:
        return None

    body = body.translate(str.maketrans(digit_map or {
        "O": "0",
        "o": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "l": "1",
        "|": "1",
        "!": "1",
    }))
    body = body.strip(".,")
    if not body:
        return None

    if "," in body and "." not in body:
        body = re.sub(r"(?<=\d),(?=\d)", ".", body)
    else:
        body = body.replace(",", "")

    if body.count(".") > 1:
        first = body.find(".")
        body = body[: first + 1] + body[first + 1 :].replace(".", "")

    if infer_decimal and "." not in body and suffix and body.isdigit() and len(body) >= 4:
        body = f"{body[:-2]}.{body[-2:]}"

    if not re.fullmatch(r"\d+(?:\.\d+)?", body):
        return None

    try:
        value = Decimal(body)
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


def _prep_ore_qty_gray(img) -> Optional[np.ndarray]:
    bgr = _to_bgr(img)
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    if h <= 0 or w <= 0:
        return None
    up = cv2.resize(bgr, (w * 6, h * 6), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)


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


def _prep_planet_title(img) -> Optional[np.ndarray]:
    bgr = _to_bgr(img)
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    if h <= 0 or w <= 0:
        return None
    up = cv2.resize(bgr, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return bw


def _ocr_text(img: np.ndarray, *, psm: int, whitelist: str) -> str:
    cfg = f"--psm {psm} -c tessedit_char_whitelist={whitelist}"
    return pytesseract.image_to_string(img, config=cfg).strip()


def _normalize_text_for_mode(text: str | None, *, mode: str) -> str:
    if not text:
        return ""
    normalized = str(text).replace("\r", "\n").strip()
    if not normalized:
        return ""
    if mode == "planet_title":
        normalized = normalized.upper().replace("\n", " ")
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
    else:
        normalized = normalized.replace("\n", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _score_text_candidate(text: str, *, mode: str) -> tuple:
    if not text:
        return (-1,)
    if mode == "planet_title":
        has_id = 1 if re.search(r"^\D*(\d{1,2})\b", text) else 0
        alpha_count = sum(1 for ch in text if ch.isalpha())
        digit_count = sum(1 for ch in text if ch.isdigit())
        dot_count = text.count(".")
        return (has_id, alpha_count > 0, digit_count > 0, dot_count > 0, len(text), alpha_count, digit_count)
    return (len(text),)


def _choose_best_text_candidate(candidates: list[str], *, mode: str) -> str:
    normalized = []
    for candidate in candidates:
        value = _normalize_text_for_mode(candidate, mode=mode)
        if value:
            normalized.append(value)
    if not normalized:
        return ""
    unique = list(dict.fromkeys(normalized))
    unique.sort(key=lambda item: _score_text_candidate(item, mode=mode), reverse=True)
    return unique[0]


def _get_rapidocr():
    global _RAPIDOCR_ENGINE, _RAPIDOCR_UNAVAILABLE
    if _RAPIDOCR_UNAVAILABLE:
        return None
    if _RAPIDOCR_ENGINE is not None:
        return _RAPIDOCR_ENGINE
    try:
        from rapidocr_onnxruntime import RapidOCR
    except Exception:
        _RAPIDOCR_UNAVAILABLE = True
        return None
    try:
        _RAPIDOCR_ENGINE = RapidOCR()
    except Exception:
        _RAPIDOCR_UNAVAILABLE = True
        return None
    return _RAPIDOCR_ENGINE


def _ocr_text_rapidocr(img) -> str:
    engine = _get_rapidocr()
    if engine is None:
        return ""
    try:
        if isinstance(img, Image.Image):
            arr = np.array(img)
        else:
            arr = img
        result, _ = engine(arr)
    except Exception:
        return ""
    if not result:
        return ""
    texts = []
    for item in result:
        if isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[1], str):
            texts.append(item[1])
    return " ".join(texts).strip()


def _text_ocr_variants(img, *, mode: str) -> list[str]:
    if img is None:
        return []

    candidates: list[str] = []

    if mode in {"level", "ore_qty", "planet_title"}:
        template_text, _template_score = template_number_reader.read_text(img, mode=mode)
        if template_text:
            candidates.append(template_text)

    variant_specs: list[tuple[Optional[np.ndarray], int, str]] = []
    if mode == "hud_cash":
        variant_specs = [(_prep_hud_cash(img), 7, "0123456789.,$KMBTqQsSO")]
    elif mode == "ore_qty":
        variant_specs = [
            (_prep_ore_qty(img), 7, "0123456789.,KMBTqQsSO"),
            (_prep_ore_qty_gray(img), 7, "0123456789.,KMBTqQsSO"),
        ]
    elif mode == "level":
        variant_specs = [(_prep_generic(img), 7, "0123456789")]
    elif mode == "planet_title":
        variant_specs = [
            (_prep_planet_title(img), 7, "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ. "),
            (_prep_generic(img), 7, "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ. "),
        ]
    else:
        variant_specs = [(_prep_generic(img), 6, "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:$ /")]

    for variant, psm, whitelist in variant_specs:
        if variant is None or (isinstance(variant, np.ndarray) and variant.size == 0):
            continue
        text = _ocr_text(variant, psm=psm, whitelist=whitelist).strip()
        if text:
            candidates.append(text)

    if mode == "ore_qty":
        rapid_candidates = [_ocr_text_rapidocr(_prep_ore_qty_gray(img)), _ocr_text_rapidocr(img)]
    else:
        rapid_candidates = [_ocr_text_rapidocr(img)]
    for text in rapid_candidates:
        if text:
            candidates.append(text)
    return candidates


def _read_text_once(bbox, *, mode: str) -> str:
    if bbox is None:
        return ""
    img, _meta = capture_bbox(bbox)
    ok, _reason = validate_crop(img, bbox, mode)
    if not ok:
        return ""
    return _choose_best_text_candidate(_text_ocr_variants(img, mode=mode), mode=mode)


def _log_fail(mode: str, bbox, reason: str) -> None:
    print(f"[OCR] mode={mode} bbox={bbox} capture=empty (reason={reason}) -> None")


def _read_number_once(bbox, *, mode: str, debug_tag: str | None = None) -> Optional[int]:
    img, meta = capture_bbox(bbox)
    ok, reason = validate_crop(img, bbox, mode)
    if not ok:
        _log_fail(mode, bbox, reason)
        return None

    _debug_save(mode, bbox, "raw", img, debug_tag)

    if mode in {"level", "ore_qty"}:
        template_text, template_score = template_number_reader.read_text(img, mode=mode)
        if template_text:
            value = parse_compact_number_for_mode(template_text, mode=mode)
            if value is not None:
                return value

    if mode == "hud_cash":
        bw = _prep_hud_cash(img)
        whitelist = "0123456789.,$KMBTqQsSO"
        psm = 7
        variants = [("bw", bw)]
    elif mode == "ore_qty":
        bw = _prep_ore_qty(img)
        whitelist = "0123456789.,KMBTqQsSO"
        psm = 7
        variants = [("bw", bw), ("gray", _prep_ore_qty_gray(img))]
    elif mode == "level":
        bw = _prep_generic(img)
        whitelist = "0123456789"
        psm = 7
        variants = [("bw", bw)]
    elif mode == "planet_title":
        bw = _prep_planet_title(img)
        whitelist = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ. "
        psm = 7
        variants = [("bw", bw), ("generic", _prep_generic(img))]
    else:
        bw = _prep_generic(img)
        whitelist = "0123456789.,$KMBTqQsSO"
        psm = 6
        variants = [("bw", bw)]

    last_text = ""
    had_variant = False
    for step_name, variant in variants:
        if variant is None or (isinstance(variant, np.ndarray) and variant.size == 0):
            continue
        had_variant = True
        _debug_save(mode, bbox, step_name, variant, debug_tag)
        text = _ocr_text(variant, psm=psm, whitelist=whitelist)
        last_text = text
        value = parse_compact_number_for_mode(text, mode=mode)
        if value is not None:
            return value

    rapidocr_variants = []
    if mode == "ore_qty":
        rapidocr_variants = [("raw", img), ("gray", _prep_ore_qty_gray(img))]
    elif mode in {"level", "hud_cash", "planet_title"}:
        rapidocr_variants = [("raw", img)]

    for step_name, variant in rapidocr_variants:
        if variant is None or (isinstance(variant, np.ndarray) and variant.size == 0):
            continue
        text = _ocr_text_rapidocr(variant)
        if not text:
            continue
        last_text = text
        value = parse_compact_number_for_mode(text, mode=mode)
        if value is not None:
            return value

    if not had_variant:
        _log_fail(mode, bbox, "prep_empty")
        return None

    print(f"[OCR] mode={mode} bbox={bbox} text=\"{last_text}\" parsed=None -> None")
    return None


def _read_number_with_offsets(bbox, offsets, *, mode: str, debug_tag: str | None = None) -> Optional[int]:
    x, y, w, h = bbox
    for dy in offsets:
        shifted = (x, y + dy, w, h)
        val = _read_number_once(shifted, mode=mode, debug_tag=debug_tag)
        if val is not None:
            return val
    return None


def _read_ore_qty_median(bbox, debug_tag: str | None = None) -> Optional[int]:
    resolved, reason = _resolve_bbox(bbox)
    if not resolved:
        _log_fail("ore_qty", bbox, reason)
        return None
    bbox = resolved
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


def _read_number_stable(
    bbox,
    *,
    mode: str,
    attempts: int,
    min_valid: int,
    max_rel_spread: float,
    delay: float,
    debug_tag: str | None = None,
) -> Optional[int]:
    values = []
    attempts = max(1, int(attempts))
    min_valid = max(1, int(min_valid))
    for i in range(attempts):
        tag = f"{debug_tag}_s{i}" if debug_tag else None
        val = _read_number_once(bbox, mode=mode, debug_tag=tag)
        if val is not None:
            values.append(val)
        if i < attempts - 1 and delay > 0:
            time.sleep(delay)

    if len(values) < min_valid:
        print(f"[OCR] mode={mode} bbox={bbox} samples={len(values)} < min_valid -> None")
        return None

    median_val = statistics.median(values)
    if median_val == 0:
        return 0 if max(values) == 0 else None

    rel_spread = (max(values) - min(values)) / median_val
    if rel_spread > max_rel_spread:
        print(f"[OCR] mode={mode} bbox={bbox} spread={rel_spread:.3f} -> None")
        return None

    return int(median_val)


def ocr_read_number(bbox, *, mode: str, debug_tag: str | None = None) -> Optional[int]:
    if bbox is None:
        _log_fail(mode, bbox, "bbox_none")
        return None
    if mode == "ore_qty":
        return _read_ore_qty_median(bbox, debug_tag)
    if mode in {"hud_cash", "level"} and getattr(config, "OCR_STABLE_SAMPLES", 1) > 1:
        return _read_number_stable(
            bbox,
            mode=mode,
            attempts=getattr(config, "OCR_STABLE_SAMPLES", 3),
            min_valid=getattr(config, "OCR_STABLE_MIN_VALID_SAMPLES", 2),
            max_rel_spread=getattr(config, "OCR_STABLE_MAX_REL_SPREAD", 0.15),
            delay=getattr(config, "OCR_STABLE_SAMPLE_DELAY", 0.05),
            debug_tag=debug_tag,
        )
    return _read_number_once(bbox, mode=mode, debug_tag=debug_tag)

def preprocess_for_mode(img, mode: str) -> Optional[np.ndarray]:
    if mode == "ore_qty":
        return _prep_ore_qty(img)
    if mode == "hud_cash":
        return _prep_hud_cash(img)
    if mode == "level":
        return _prep_generic(img)
    return _prep_generic(img)


def ocr_read_debug(bbox, *, mode: str) -> dict:
    """Single-pass OCR with raw text + parsed value for UI preview."""
    if bbox is None:
        return {"ok": False, "reason": "bbox_none", "text": "", "value": None}
    img, _meta = capture_bbox(bbox)
    ok, reason = validate_crop(img, bbox, mode)
    if not ok:
        return {"ok": False, "reason": reason, "text": "", "value": None}

    if mode == "hud_cash":
        bw = _prep_hud_cash(img)
        whitelist = "0123456789.,$KMBTqQsSO"
        psm = 7
    elif mode == "ore_qty":
        bw = _prep_ore_qty(img)
        whitelist = "0123456789.,KMBTqQsSO"
        psm = 7
    elif mode == "level":
        bw = _prep_generic(img)
        whitelist = "0123456789"
        psm = 7
    elif mode == "planet_title":
        bw = _prep_planet_title(img)
        whitelist = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ. "
        psm = 7
    else:
        bw = _prep_generic(img)
        whitelist = "0123456789.,$KMBTqQsSO"
        psm = 6

    if bw is None or (isinstance(bw, np.ndarray) and bw.size == 0):
        return {"ok": False, "reason": "prep_empty", "text": "", "value": None}

    candidates = _text_ocr_variants(img, mode=mode)
    best_text = _choose_best_text_candidate(candidates, mode=mode)
    if mode == "planet_title" and best_text:
        return {"ok": True, "reason": "ok", "text": best_text, "value": None}
    value = parse_compact_number_for_mode(best_text, mode=mode)
    if value is not None:
        return {"ok": True, "reason": "ok", "text": best_text, "value": value}
    return {"ok": False, "reason": "parse_fail", "text": best_text, "value": None}


def ocr_read_text(bbox, *, mode: str = "generic") -> str:
    """Single-pass OCR text read without numeric parsing."""
    return _read_text_once(bbox, mode=mode)


def ocr_read_text_stable(
    bbox,
    *,
    mode: str = "generic",
    attempts: int | None = None,
    min_valid: int | None = None,
    delay: float | None = None,
) -> str:
    if bbox is None:
        return ""
    attempts = max(1, int(attempts or getattr(config, "OCR_TEXT_STABLE_SAMPLES", 3)))
    min_valid = max(1, int(min_valid or getattr(config, "OCR_TEXT_STABLE_MIN_VALID_SAMPLES", 2)))
    delay = max(0.0, float(delay if delay is not None else getattr(config, "OCR_TEXT_STABLE_SAMPLE_DELAY", 0.04)))

    counts: dict[str, int] = {}
    for idx in range(attempts):
        text = _read_text_once(bbox, mode=mode)
        if text:
            counts[text] = counts.get(text, 0) + 1
        if idx < attempts - 1 and delay > 0:
            time.sleep(delay)

    if not counts:
        return ""

    ranked = sorted(
        counts.items(),
        key=lambda item: (item[1],) + _score_text_candidate(item[0], mode=mode),
        reverse=True,
    )
    best_text, best_count = ranked[0]
    if best_count < min_valid and len(counts) > 1:
        return ""
    return best_text
