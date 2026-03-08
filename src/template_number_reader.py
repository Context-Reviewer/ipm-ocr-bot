from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


_TEMPLATE_DIR = Path(__file__).resolve().parent / "assets" / "text_samples"
_CANVAS_H = 32
_CANVAS_W = 24
_ALLOWED = {
    "level": set("0123456789"),
    "ore_qty": set("0123456789KMBT"),
    "planet_title": set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ."),
}
_MODE_THRESHOLDS = {
    "level": 0.58,
    "ore_qty": 0.58,
    "planet_title": 0.50,
}

_SAMPLE_SPECS = (
    ("level", "level_23.png", "23", 80),
    ("level", "level_12.png", "12", 80),
    ("level", "level_14.png", "14", 80),
    ("level", "level_18.png", "18", 80),
    ("level", "level_27.png", "27", 80),
    ("ore_qty", "ore_67021K.png", "67021K", 100),
    ("ore_qty", "ore_722K.png", "722K", 100),
    ("ore_qty", "ore_4305K.png", "4305K", 100),
)


def _to_gray(img) -> np.ndarray:
    if isinstance(img, Image.Image):
        arr = np.array(img.convert("RGB"))
    else:
        arr = np.array(img)
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def _mask_from_gray(gray: np.ndarray, *, mode: str, threshold: int | None = None) -> np.ndarray:
    if threshold is None:
        threshold = 80 if mode == "level" else 96 if mode == "planet_title" else 100
    mask = ((gray > threshold).astype(np.uint8) * 255)
    return mask


def _component_boxes(mask: np.ndarray, *, mode: str) -> list[tuple[int, int, int, int, int]]:
    num, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    height, width = mask.shape
    boxes: list[tuple[int, int, int, int, int]] = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < 8 or h < 3 or w < 2:
            continue
        if w > width * 0.8 and h <= 2:
            continue
        if y < max(2, int(height * 0.15)) and h <= 2:
            continue
        # Decimal dots are tiny and the ore parser can infer decimals from suffixed values.
        if mode == "ore_qty" and w <= 3 and area <= 24:
            continue
        boxes.append((int(x), int(y), int(w), int(h), int(area)))
    boxes.sort(key=lambda item: (item[0], item[1]))
    return boxes


def _normalize_component(component_mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(component_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((_CANVAS_H, _CANVAS_W), dtype=np.uint8)
    cropped = component_mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    h, w = cropped.shape
    scale = min(20 / max(w, 1), 28 / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((_CANVAS_H, _CANVAS_W), dtype=np.uint8)
    off_y = (_CANVAS_H - new_h) // 2
    off_x = (_CANVAS_W - new_w) // 2
    canvas[off_y : off_y + new_h, off_x : off_x + new_w] = (resized > 0).astype(np.uint8)
    return canvas


def _load_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\Arial.ttf",
        "arialbd.ttf",
        "Arial.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, 28)
        except Exception:
            continue
    return ImageFont.load_default()


def _render_synthetic_template(ch: str, *, font_size: int = 28) -> np.ndarray:
    font = _load_font()
    if isinstance(font, ImageFont.FreeTypeFont):
        try:
            font = ImageFont.truetype(font.path, font_size)
        except Exception:
            pass
    img = Image.new("L", (50, 60), 0)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), ch, font=font)
    draw.text((6 - bbox[0], 6 - bbox[1]), ch, fill=255, font=font)
    return _normalize_component(np.array(img))


@lru_cache(maxsize=1)
def _template_bank() -> dict[str, list[np.ndarray]]:
    bank: dict[str, list[np.ndarray]] = {}
    for mode, filename, text, threshold in _SAMPLE_SPECS:
        path = _TEMPLATE_DIR / filename
        if not path.exists():
            continue
        gray = _to_gray(Image.open(path))
        mask = _mask_from_gray(gray, mode=mode, threshold=threshold)
        boxes = _component_boxes(mask, mode=mode)
        if len(boxes) != len(text):
            continue
        for box, ch in zip(boxes, text):
            x, y, w, h, _area = box
            bank.setdefault(ch, []).append(_normalize_component(mask[y : y + h, x : x + w]))

    for ch in "0123456789KMBT":
        if ch not in bank:
            bank[ch] = [_render_synthetic_template(ch)]
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ.":
        if ch not in bank:
            bank[ch] = [_render_synthetic_template(ch, font_size=26)]
    return bank


def _match_component(component_mask: np.ndarray, *, allowed: set[str]) -> tuple[Optional[str], float]:
    normalized = _normalize_component(component_mask)
    best_char: Optional[str] = None
    best_score = -1.0
    for ch, templates in _template_bank().items():
        if ch not in allowed:
            continue
        for template in templates:
            inter = np.logical_and(normalized, template).sum()
            union = np.logical_or(normalized, template).sum() or 1
            score = float(inter / union)
            if score > best_score:
                best_char = ch
                best_score = score
    return best_char, best_score


def read_text(img, *, mode: str) -> tuple[Optional[str], float]:
    allowed = _ALLOWED.get(mode)
    if not allowed:
        return None, 0.0
    gray = _to_gray(img)
    mask = _mask_from_gray(gray, mode=mode)
    boxes = _component_boxes(mask, mode=mode)
    if not boxes:
        return None, 0.0

    chars: list[str] = []
    scores: list[float] = []
    for x, y, w, h, _area in boxes:
        ch, score = _match_component(mask[y : y + h, x : x + w], allowed=allowed)
        if ch is None:
            return None, 0.0
        chars.append(ch)
        scores.append(score)

    text = "".join(chars)
    if not text:
        return None, 0.0
    min_score = min(scores)
    avg_score = sum(scores) / len(scores)
    threshold = _MODE_THRESHOLDS.get(mode, 0.58)
    if min_score < threshold:
        return None, min_score
    return text, avg_score
