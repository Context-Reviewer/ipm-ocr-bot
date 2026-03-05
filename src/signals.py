from PIL import ImageGrab
import numpy as np

RECT_MINING = (1796, 730, 8, 8)
RECT_SPEED = (1793, 819, 8, 8)
RECT_CARGO = (1794, 914, 8, 8)
RECT_ORES_SCROLLBAR_TOP = (0, 0, 10, 10)

def sample_rect(rect):
    x, y, w, h = rect
    img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
    return np.array(img)

def is_cyan_present(rect):
    pixels = sample_rect(rect)
    if pixels.size == 0:
        return False
    r = pixels[:, :, 0]
    g = pixels[:, :, 1]
    b = pixels[:, :, 2]
    mask = (r < 120) & (g > 150) & (b > 150)
    ratio = mask.sum() / mask.size
    return ratio >= 0.20

def mining_available() -> bool:
    return is_cyan_present(RECT_MINING)

def speed_available() -> bool:
    return is_cyan_present(RECT_SPEED)

def cargo_available() -> bool:
    return is_cyan_present(RECT_CARGO)

def mean_luma(rect) -> float:
    pixels = sample_rect(rect)
    if pixels.size == 0:
        return 0.0
    r = pixels[:, :, 0].astype(np.float32)
    g = pixels[:, :, 1].astype(np.float32)
    b = pixels[:, :, 2].astype(np.float32)
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return float(y.mean())

def mean_abs_diff(rect, prev_pixels):
    curr = sample_rect(rect)
    if prev_pixels is None or curr.size == 0:
        return (1e9, curr)
    diff = float(np.mean(np.abs(curr.astype(np.int16) - prev_pixels.astype(np.int16))))
    return (diff, curr)
