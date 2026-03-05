from PIL import ImageGrab
import numpy as np
import config

RECT_MINING = (1796, 730, 8, 8)
RECT_SPEED = (1793, 819, 8, 8)
RECT_CARGO = (1794, 914, 8, 8)
RECT_ORES_SCROLLBAR_TOP = (0, 0, 10, 10)

def sample_rect(rect) -> np.ndarray:
    try:
        x, y, w, h = rect
        img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
        return np.array(img)
    except Exception:
        return np.array([])

def is_cyan_present(rect) -> bool:
    if not isinstance(rect, (tuple, list)) or len(rect) != 4:
        return False
    x, y, w, h = rect
    if int(w) <= 0 or int(h) <= 0:
        return False
    pixels = sample_rect(rect)
    if pixels.size == 0 or pixels.ndim != 3 or pixels.shape[2] < 3:
        return False

    height, width = pixels.shape[0], pixels.shape[1]
    if height <= 0 or width <= 0:
        return False
    sx = 1 if width > 2 else 0

    samples = []
    votes = 0
    n = 5
    for i in range(n):
        sy = int(round((i + 1) * (height - 1) / (n + 1)))
        r, g, b = pixels[sy, sx, 0], pixels[sy, sx, 1], pixels[sy, sx, 2]
        samples.append((int(r), int(g), int(b)))
        if r < 80 and g > 170 and b > 190:
            votes += 1

    affordable = votes >= 3
    if getattr(config, "CYAN_DEBUG", False):
        print(f"[CYAN] rect={rect} samples={samples} votes={votes} -> affordable={affordable}")
    return affordable

def mining_available() -> bool:
    return is_cyan_present(RECT_MINING)

def speed_available() -> bool:
    return is_cyan_present(RECT_SPEED)

def cargo_available() -> bool:
    return is_cyan_present(RECT_CARGO)
