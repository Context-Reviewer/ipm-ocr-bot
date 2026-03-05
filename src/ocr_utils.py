import re
import time
import statistics
import cv2
import numpy as np
from PIL import Image, ImageGrab, ImageOps
import pytesseract
import config

if config.TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD

def grab_bbox(bbox):
    x, y, w, h = bbox
    return ImageGrab.grab(bbox=(x, y, x + w, y + h))

def preprocess(img):
    gray = ImageOps.grayscale(img)
    contrast = ImageOps.autocontrast(gray)
    bw = contrast.point(lambda p: 255 if p > 160 else 0)
    arr = np.array(bw)
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(arr, kernel, iterations=1)
    return Image.fromarray(dilated)

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

def parse_qty(text):
    if text is None:
        return None
    cleaned = text.strip().replace(",", "").replace(" ", "").replace("\n", "").replace("$", "")
    if not cleaned:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)([KMBTqQsSOkmbt]?)", cleaned)
    if not match:
        return None
    num_str, suffix = match.groups()
    try:
        value = float(num_str)
    except ValueError:
        return None
    mult = _SUFFIX_MULTS.get(suffix, 1)
    return int(value * mult)

def parse_compact_number(text):
    if text is None:
        return None
    cleaned = text.strip().replace(",", "").replace(" ", "").replace("\n", "").replace("$", "")
    if not cleaned:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)([KMBTqQsSOkmbt]?)", cleaned)
    if not match:
        return None
    num_str, suffix = match.groups()
    try:
        value = float(num_str)
    except ValueError:
        return None
    mult = _SUFFIX_MULTS.get(suffix, 1)
    return int(value * mult)

def ocr_qty_once(bbox):
    qty, _dy = ocr_qty_from_bbox_with_y_scan(bbox, offsets=config.OCR_QTY_Y_OFFSETS)
    return qty

def ocr_qty_from_bbox_with_y_scan(bbox, *, offsets):
    x, y, w, h = bbox
    config_str = "--psm 7 -c tessedit_char_whitelist=0123456789.KM"
    for dy in offsets:
        shifted = (x, y + dy, w, h)
        img = grab_bbox(shifted)
        img = preprocess(img)
        text = pytesseract.image_to_string(img, config=config_str)
        qty = parse_qty(text)
        if qty is not None:
            return qty, dy
    return None, None

def ocr_qty_median(bbox, samples, delay, min_valid, max_rel_spread):
    values = []
    for i in range(samples):
        val = ocr_qty_once(bbox)
        if val is not None:
            values.append(val)
        if i < samples - 1:
            time.sleep(delay)

    if len(values) < min_valid:
        return None

    median_val = statistics.median(values)
    if median_val == 0:
        return 0 if max(values) == 0 else None

    rel_spread = (max(values) - min(values)) / median_val
    if rel_spread > max_rel_spread:
        return None

    return int(median_val)
