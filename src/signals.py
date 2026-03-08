from PIL import ImageGrab
import numpy as np
from pathlib import Path
import config
from rect_store import RectStore
from window_win32 import get_bluestacks_client_rect

RECT_MINING = "UPGRADE_MINING"
RECT_SPEED = "UPGRADE_SPEED"
RECT_CARGO = "UPGRADE_CARGO"
RECT_ORES_SCROLLBAR_TOP = (0, 0, 10, 10)

_RECT_STORE: RectStore | None = None


def _load_rects() -> RectStore | None:
    global _RECT_STORE
    if _RECT_STORE is not None:
        return _RECT_STORE
    path = Path(getattr(config, "RECTS_JSON_PATH", "rects.json"))
    if not path.exists():
        return None
    _RECT_STORE = RectStore.load(path)
    return _RECT_STORE


def _resolve_rect(rect):
    title_hint = getattr(config, "BLUESTACKS_TITLE_HINT", "BlueStacks App Player")
    if isinstance(rect, str):
        store = _load_rects()
        if store is None:
            return None
        rel = store.rects.get(rect)
        if rel is None:
            return None
        c = get_bluestacks_client_rect(title_hint)
        if not c:
            return None
        x, y, w, h = rel
        return (c.left + x, c.top + y, w, h)

    if isinstance(rect, (tuple, list)) and len(rect) == 4:
        if getattr(config, "RECTS_USE_CLIENT", False):
            c = get_bluestacks_client_rect(title_hint)
            if not c:
                return None
            x, y, w, h = rect
            return (c.left + x, c.top + y, w, h)
        return rect
    return None

def sample_rect(rect) -> np.ndarray:
    try:
        resolved = _resolve_rect(rect)
        if not resolved:
            return np.array([])
        x, y, w, h = resolved
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
    band = max(1, min(int(getattr(config, "CYAN_BORDER_WIDTH", 4)), width // 3, height // 3))
    border_parts = [
        pixels[:, :band, :],
        pixels[:, width - band :, :],
        pixels[:band, :, :],
        pixels[height - band :, :, :],
    ]
    border = np.concatenate([part.reshape(-1, 3) for part in border_parts], axis=0)
    red = border[:, 0].astype(np.int16)
    green = border[:, 1].astype(np.int16)
    blue = border[:, 2].astype(np.int16)

    cyan_mask = (
        (blue > 145)
        & (green > 115)
        & ((blue - red) > 55)
        & ((green - red) > 25)
        & ((blue - green) > -35)
    )
    cyan_ratio = float(cyan_mask.mean()) if cyan_mask.size else 0.0
    affordable = cyan_ratio >= float(getattr(config, "CYAN_MIN_PIXEL_RATIO", 0.06))
    if getattr(config, "CYAN_DEBUG", False):
        sample_points = []
        for sy in (height // 4, height // 2, (3 * height) // 4):
            sx = min(width - 1, max(0, band // 2))
            r, g, b = pixels[sy, sx, 0], pixels[sy, sx, 1], pixels[sy, sx, 2]
            sample_points.append((int(r), int(g), int(b)))
        print(
            f"[CYAN] rect={rect} samples={sample_points} "
            f"ratio={cyan_ratio:.3f} band={band} -> affordable={affordable}"
        )
    return affordable

def mining_available() -> bool:
    return is_cyan_present(RECT_MINING)

def speed_available() -> bool:
    return is_cyan_present(RECT_SPEED)

def cargo_available() -> bool:
    return is_cyan_present(RECT_CARGO)
