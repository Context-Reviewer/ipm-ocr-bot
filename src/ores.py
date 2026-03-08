import time
import re
import cv2
import numpy as np
import pytesseract
import config
import policy
import ocr
import perception
import statistics
from config import KEY_DELAY, MENU_DELAY, SCROLL_DELAY
from input_utils import tap, reset_ui, click_screen_point
from signals import sample_rect


_LAST_CONFIRMED_QTY_BY_ORE: dict[str, int] = {}

def mean_abs_diff(rect, prev_pixels):
    curr = sample_rect(rect)
    if prev_pixels is None or curr.size == 0:
        return (1e9, curr)
    diff = float(np.mean(np.abs(curr.astype(np.int16) - prev_pixels.astype(np.int16))))
    return (diff, curr)


def wait_for_rect_settle(rect) -> bool:
    stable = 0
    prev = None
    for _ in range(config.ORES_ROW_SETTLE_MAX_STEPS):
        diff, prev = mean_abs_diff(rect, prev)
        if diff <= config.ORES_ROW_SETTLE_DIFF_THRESHOLD:
            stable += 1
            if stable >= config.ORES_ROW_SETTLE_STABLE_READS:
                return True
        else:
            stable = 0
        time.sleep(config.ORES_ROW_SETTLE_DELAY)
    return False

def select_ore(ore_name: str) -> bool:
    key = config.ORE_SELECT_KEYS.get(ore_name)
    if not key:
        print(f"[ORES] select ore={ore_name} missing key mapping; skipping fail-closed")
        return False
    print(f"[ORES] select ore={ore_name} key={key}")
    tap(key, KEY_DELAY)
    return True

def scroll_to_top() -> bool:
    stable = 0
    prev = None
    for _ in range(config.ORES_TOP_LATCH_MAX_STEPS):
        tap(config.ORES_SCROLL_UP_KEY, KEY_DELAY)
        time.sleep(config.ORES_TOP_LATCH_SETTLE_DELAY)
        diff, prev = mean_abs_diff(config.RECT_ORES_TOP_ANCHOR, prev)
        if diff <= config.ORES_TOP_LATCH_DIFF_THRESHOLD:
            stable += 1
            if stable >= config.ORES_TOP_LATCH_STABLE_READS:
                return True
        else:
            stable = 0
    return False

def qty_bbox_for_row(row_index: int):
    key = f"ORE_ROW{row_index}_QTY"
    if ocr.resolve_bbox(key):
        return key
    strip_x, strip_y, strip_w, strip_h = config.RECT_ORE_QTY_STRIP
    row_count = max(1, int(getattr(config, "VISIBLE_ORE_ROWS", len(config.ORE_ROW_MAP)) or len(config.ORE_ROW_MAP)))
    if row_count <= 0:
        return (strip_x, strip_y, config.ORE_QTY_BOX_W, config.ORE_QTY_BOX_H)
    row_h = strip_h // row_count
    row_top = strip_y + (row_index - 1) * row_h
    bbox_y = row_top + (row_h - config.ORE_QTY_BOX_H) // 2
    bbox_h = config.ORE_QTY_BOX_H
    pad_x = config.ORE_QTY_BBOX_PAD_X
    bbox_w = min(config.ORE_QTY_BOX_W, strip_w - 2 * pad_x)
    bbox_x = strip_x + pad_x
    return (bbox_x, bbox_y, bbox_w, bbox_h)


def row_bbox_for_row(row_index: int):
    explicit_key = f"ORE_ROW{row_index}_READ"
    explicit_rect = ocr.resolve_bbox(explicit_key)
    if isinstance(explicit_rect, (tuple, list)) and len(explicit_rect) == 4:
        return explicit_rect

    qty_bbox = qty_bbox_for_row(row_index)
    resolved = ocr.resolve_bbox(qty_bbox)
    if not resolved:
        return qty_bbox
    x, y, w, h = resolved
    left = max(0, x - int(getattr(config, "ORE_ROW_READ_PAD_LEFT", 170)))
    top = max(0, y - int(getattr(config, "ORE_ROW_READ_PAD_TOP", 10)))
    right = x + w + int(getattr(config, "ORE_ROW_READ_PAD_RIGHT", 12))
    bottom = y + h + int(getattr(config, "ORE_ROW_READ_PAD_BOTTOM", 10))
    return (left, top, max(1, right - left), max(1, bottom - top))


def row_has_visible_content(row_index: int) -> bool:
    pixels = sample_rect(row_bbox_for_row(row_index))
    if getattr(pixels, "size", 0) == 0 or getattr(pixels, "ndim", 0) != 3:
        return False
    try:
        gray = pixels[:, :, :3].mean(axis=2)
    except Exception:
        return False
    mean_val = float(gray.mean()) if gray.size else 0.0
    bright_ratio = float((gray >= 40).mean()) if gray.size else 0.0
    return (
        mean_val >= float(getattr(config, "ORES_ROW_VISIBLE_MIN_MEAN", 18.0))
        or bright_ratio >= float(getattr(config, "ORES_ROW_VISIBLE_MIN_BRIGHT_RATIO", 0.02))
    )


_ROW_QTY_TOKEN_RE = re.compile(r"[0-9OQDSIl|!SBs.,]+[KMBTqQkmbt]?")


def _parse_qty_from_text(text: str | None):
    if not text:
        return None
    candidates = _ROW_QTY_TOKEN_RE.findall(text)
    for token in reversed(candidates):
        if not any(ch.isdigit() for ch in token):
            continue
        value = ocr.parse_compact_number_for_mode(token, mode="ore_qty")
        if value is not None:
            return value
    return None


def _parse_plain_digits(text: str | None):
    if not text:
        return None
    matches = re.findall(r"\d+", text)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except Exception:
        return None


def _shift_bbox_y(bbox, dy: int):
    try:
        x, y, w, h = bbox
        return (int(x), int(y) + int(dy), int(w), int(h))
    except Exception:
        return bbox


def read_qty_from_row_text(row_index: int, *, debug_tag: str | None = None):
    bbox = row_bbox_for_row(row_index)
    offsets = [0, -6, 6]
    last_text = None
    last_bbox = bbox
    for dy in offsets:
        shifted_bbox = _shift_bbox_y(bbox, dy)
        text = ocr.ocr_read_text(shifted_bbox, mode="generic")
        qty = _parse_qty_from_text(text)
        if qty is not None:
            return shifted_bbox, qty
        last_text = text
        last_bbox = shifted_bbox

    dbg = ocr.ocr_read_debug(bbox, mode="generic")
    dbg_text = dbg.get("text")
    qty = _parse_qty_from_text(dbg_text)
    if qty is not None:
        return bbox, qty
    last_text = dbg_text if dbg_text is not None else last_text

    for dy in offsets:
        shifted_bbox = _shift_bbox_y(bbox, dy)
        img, meta = ocr.capture_bbox(shifted_bbox)
        if img is None:
            continue
        arr = np.array(img)
        if arr.size == 0:
            continue
        try:
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr
        except Exception:
            continue
        height, width = gray.shape[:2]
        if height <= 0 or width <= 0:
            continue
        row_crops = [
            gray[:, int(width * 0.55) :],
            gray,
            gray[:, int(width * 0.45) : int(width * 0.85)],
        ]
        for crop in row_crops:
            if crop.size == 0:
                continue
            upscaled = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            text = pytesseract.image_to_string(
                upscaled,
                config="--psm 7 -c tessedit_char_whitelist=0123456789",
            ).strip()
            qty = _parse_plain_digits(text)
            if qty is not None:
                return shifted_bbox, qty
            last_text = text or last_text
            last_bbox = shifted_bbox

    try:
        perceived_qty, perceived = perception.read_number_value(
            bbox,
            mode="ore_qty",
            prompt=str(getattr(config, "PERCEPTION_ORE_QTY_PROMPT", "")),
        )
    except Exception:
        perceived_qty, perceived = (None, None)
    if perceived_qty is not None:
        return bbox, perceived_qty
    if perceived is not None and debug_tag:
        last_text = perceived.text or last_text
    if debug_tag:
        print(f"[ORES] row={row_index} row_text parse failed bbox={last_bbox} text={last_text!r}")
    return last_bbox, None


def read_qty_from_row_text_stable(row_index: int, *, debug_tag: str | None = None):
    values: list[int] = []
    last_bbox = row_bbox_for_row(row_index)
    attempts = max(1, int(getattr(config, "ORE_ROW_TEXT_SAMPLES", 3) or 1))
    min_valid = max(1, int(getattr(config, "ORE_ROW_TEXT_MIN_VALID_SAMPLES", 2) or 1))
    max_rel_spread = float(getattr(config, "ORE_ROW_TEXT_MAX_REL_SPREAD", 0.25) or 0.25)
    delay = float(getattr(config, "ORE_ROW_TEXT_SAMPLE_DELAY", 0.05) or 0.0)
    for attempt in range(attempts):
        sample_tag = f"{debug_tag}_txt{attempt}" if debug_tag else None
        bbox, qty = read_qty_from_row_text(row_index, debug_tag=sample_tag)
        last_bbox = bbox
        if qty is not None:
            values.append(int(qty))
        if attempt < attempts - 1 and delay > 0:
            time.sleep(delay)
    if len(values) < min_valid:
        return last_bbox, None
    median_val = statistics.median(values)
    if median_val <= 0:
        return last_bbox, None
    rel_spread = (max(values) - min(values)) / median_val
    if rel_spread > max_rel_spread:
        if debug_tag:
            print(f"[ORES] row={row_index} row_text spread={rel_spread:.3f} values={values} -> None")
        return last_bbox, None
    return last_bbox, int(median_val)

def debug_read_qty_for_row(row_index: int):
    bbox, qty = read_qty_for_row(row_index, debug_tag=f"ore_row{row_index}", settle=False)
    return (bbox, qty)


def read_qty_for_row(row_index: int, *, debug_tag: str | None = None, settle: bool = True):
    bbox = qty_bbox_for_row(row_index)
    settle_bbox = row_bbox_for_row(row_index)
    if settle:
        wait_for_rect_settle(settle_bbox)

    row_bbox, row_qty = read_qty_from_row_text_stable(row_index, debug_tag=debug_tag)
    if row_qty is not None:
        return row_bbox, row_qty

    for attempt in range(config.ORE_QTY_READ_RETRIES):
        tag = debug_tag
        if tag and config.ORE_QTY_READ_RETRIES > 1:
            tag = f"{tag}_r{attempt}"
        qty = ocr.ocr_read_number(bbox, mode="ore_qty", debug_tag=tag)
        if qty is not None:
            return bbox, qty
        if attempt < config.ORE_QTY_READ_RETRIES - 1:
            time.sleep(config.ORE_QTY_RETRY_DELAY)
            wait_for_rect_settle(settle_bbox)
    return bbox, None


def choose_sell_preset(desired_fraction: float):
    preset = None
    for key_name, fraction in config.ORE_SLIDER_PRESETS:
        if fraction <= desired_fraction:
            preset = key_name
    return preset


def _read_selected_sell_qty() -> int | None:
    rect_key = getattr(config, "SELL_SELECTED_QTY_RECT", "SELL_SELECTED_QTY")
    if not ocr.resolve_bbox(rect_key):
        return None
    value = ocr.ocr_read_number(rect_key, mode="ore_qty", debug_tag="sell_selected_qty")
    if value is not None and value >= 0:
        return int(value)
    text = ocr.ocr_read_text(rect_key, mode="generic")
    value = _parse_qty_from_text(text) or _parse_plain_digits(text)
    if value is not None and value >= 0:
        return int(value)
    try:
        value, _result = perception.read_number_value(
            rect_key,
            mode="ore_qty",
            prompt=str(getattr(config, "PERCEPTION_ORE_QTY_PROMPT", "")),
        )
    except Exception:
        value = None
    if value is not None and value >= 0:
        return int(value)
    return None


def _slider_track_centered_point(fraction: float):
    rect_key = getattr(config, "SELL_SLIDER_TRACK_RECT", "SELL_SLIDER_TRACK")
    rect = ocr.resolve_bbox(rect_key)
    if not isinstance(rect, (tuple, list)) or len(rect) != 4:
        return None
    x, y, w, h = rect
    fraction = max(0.0, min(1.0, float(fraction)))
    px = int(x + round((w - 1) * fraction))
    py = int(y + (h // 2))
    return (px, py)


def try_apply_precise_sell_fraction(desired_fraction: float, before_qty: int) -> bool:
    if not bool(getattr(config, "SELL_PRECISE_SLIDER_ENABLED", False)):
        return False
    if before_qty <= 0:
        return False
    if _slider_track_centered_point(0.0) is None or ocr.resolve_bbox(getattr(config, "SELL_SELECTED_QTY_RECT", "SELL_SELECTED_QTY")) is None:
        return False

    target_qty = max(0, int(round(before_qty * desired_fraction)))
    qty_tol_abs = int(getattr(config, "SELL_PRECISE_QTY_TOLERANCE_ABS", 250) or 250)
    qty_tol_ratio = float(getattr(config, "SELL_PRECISE_QTY_TOLERANCE_RATIO", 0.05) or 0.05)
    tolerance = max(qty_tol_abs, int(round(target_qty * qty_tol_ratio)))
    settle_delay = float(getattr(config, "SELL_PRECISE_SLIDER_SETTLE_DELAY", 0.12) or 0.12)
    max_steps = max(1, int(getattr(config, "SELL_PRECISE_SLIDER_MAX_STEPS", 5) or 5))

    low = 0.0
    high = 1.0
    best_error = None
    best_fraction = None

    zero_point = _slider_track_centered_point(0.0)
    if zero_point is None or not click_screen_point(zero_point, delay=KEY_DELAY):
        return False
    time.sleep(settle_delay)

    for _ in range(max_steps):
        fraction = desired_fraction if best_fraction is None else (low + high) / 2.0
        point = _slider_track_centered_point(fraction)
        if point is None or not click_screen_point(point, delay=KEY_DELAY):
            return False
        time.sleep(settle_delay)
        selected_qty = _read_selected_sell_qty()
        if selected_qty is None:
            continue
        error = abs(selected_qty - target_qty)
        if best_error is None or error < best_error:
            best_error = error
            best_fraction = fraction
        if error <= tolerance:
            return True
        if selected_qty < target_qty:
            low = max(low, fraction)
        else:
            high = min(high, fraction)

    if best_fraction is None:
        return False
    point = _slider_track_centered_point(best_fraction)
    if point is None or not click_screen_point(point, delay=KEY_DELAY):
        return False
    time.sleep(settle_delay)
    selected_qty = _read_selected_sell_qty()
    return selected_qty is not None and abs(selected_qty - target_qty) <= tolerance


def confirm_qty_for_sale(row_index: int, ore_name: str, initial_qty: int):
    values = [int(initial_qty)] if initial_qty > 0 else []
    attempts = max(1, int(getattr(config, "ORE_SELL_CONFIRM_SAMPLES", 3) or 1))
    min_valid = max(1, int(getattr(config, "ORE_SELL_CONFIRM_MIN_VALID_SAMPLES", 2) or 1))
    max_rel_spread = float(getattr(config, "ORE_SELL_CONFIRM_MAX_REL_SPREAD", 0.20) or 0.20)
    delay = float(getattr(config, "ORE_SELL_CONFIRM_SAMPLE_DELAY", 0.08) or 0.0)
    for attempt in range(max(0, attempts - 1)):
        _bbox, qty = read_qty_for_row(row_index, debug_tag=f"ore_row{row_index}_confirm{attempt}", settle=True)
        if qty is not None and qty > 0:
            values.append(int(qty))
        if attempt < attempts - 2 and delay > 0:
            time.sleep(delay)
    if len(values) < min_valid:
        return None, values, "insufficient_samples"
    median_val = int(statistics.median(values))
    if median_val <= 0:
        return None, values, "median_nonpositive"
    rel_spread = (max(values) - min(values)) / median_val
    if rel_spread > max_rel_spread:
        return None, values, f"spread={rel_spread:.3f}"

    last_qty = _LAST_CONFIRMED_QTY_BY_ORE.get(ore_name)
    if last_qty and last_qty > 0:
        jump_ratio = float(getattr(config, "ORE_SELL_MAX_SUSPICIOUS_JUMP_RATIO", 6.0) or 6.0)
        jump_abs = int(getattr(config, "ORE_SELL_MAX_SUSPICIOUS_JUMP_ABS", 50000) or 50000)
        if median_val > int(last_qty * jump_ratio) and (median_val - last_qty) > jump_abs:
            return None, values, f"suspicious_jump last={last_qty} now={median_val}"
    return median_val, values, "ok"


def verify_sale_for_row(row_index: int, before_qty: int):
    last_qty = None
    for _ in range(config.ORE_SELL_VERIFY_READS):
        time.sleep(config.ORE_SELL_VERIFY_DELAY)
        bbox, after_qty = read_qty_for_row(row_index, debug_tag=f"ore_row{row_index}_post", settle=True)
        if after_qty is None:
            continue
        last_qty = after_qty
        if after_qty < before_qty:
            return True, bbox, after_qty
    return False, qty_bbox_for_row(row_index), last_qty

def ore_module(pages: int = 2):
    # Open resources
    tap("shift+1", MENU_DELAY)
    tap("f1", MENU_DELAY)  # ores tab
    time.sleep(float(getattr(config, "ORES_MENU_OPEN_DELAY", 0.60)))

    sold_any = False
    sold_rows = 0
    verified_sales = 0

    ore_row_map = getattr(config, "unlocked_ore_row_map", lambda: config.ORE_ROW_MAP)()
    visible_rows = getattr(config, "visible_ore_rows", lambda: min(len(ore_row_map), config.VISIBLE_ORE_ROWS))()
    rows = min(len(ore_row_map), visible_rows)

    if len(ore_row_map) > visible_rows or pages > 1:
        # Only force the list to the top when scrolling is actually possible/needed.
        if not scroll_to_top():
            print("[ORES] top latch failed")
        time.sleep(config.ORES_RESET_SETTLE_DELAY)

    reservations = policy.compute_reservations(None, config)

    if len(ore_row_map) > visible_rows:
        print(f"[ORES] ores_unlocked={len(ore_row_map)} > visible={visible_rows}; skipping rows beyond visible (fail-closed)")
    row_keys = [str(i) for i in range(1, rows + 1)]
    scroll_key = config.ORES_SCROLL_DOWN_KEY  # BlueStacks swipe down

    for page in range(pages):
        missing_row_hit = False
        for key in row_keys:
            row_index = int(key)
            ore_name = ore_row_map.get(row_index)
            if ore_name is None:
                print(f"[ORES] row={row_index} no mapping; skipping")
                continue
            if not row_has_visible_content(row_index):
                print(f"[ORES] row={row_index} not visibly present; stopping visible-row scan")
                missing_row_hit = True
                break
            tap(key, KEY_DELAY)
            if config.ENABLE_ORE_OCR:
                time.sleep(float(getattr(config, "ORES_ROW_SELECT_DELAY", MENU_DELAY)))
                sell_start = config.ORE_SELL_START_BY_ROW.get(row_index, config.ORE_SELL_START_DEFAULT)
                bbox, qty = read_qty_for_row(row_index, debug_tag=f"ore_row{row_index}", settle=True)
                if qty is None:
                    if bool(getattr(config, "ORES_STOP_AT_FIRST_MISSING_ROW", True)) and row_index > 1 and not row_has_visible_content(row_index):
                        print(f"[ORES] row={row_index} absent after OCR check; stopping visible-row scan")
                        missing_row_hit = True
                        break
                    resolved = ocr.resolve_bbox(bbox)
                    y_val = resolved[1] if resolved else "n/a"
                    print(f"[ORES] row={row_index} qty OCR failed after y-scan y={y_val} bbox={bbox} offsets={config.OCR_QTY_Y_OFFSETS}; skipping row")
                    continue
                if qty < sell_start:
                    _LAST_CONFIRMED_QTY_BY_ORE[ore_name] = int(qty)
                    print(f"[ORES] row={row_index} qty={qty} < sell_start={sell_start}; skipping")
                    continue
                confirmed_qty, confirm_values, confirm_reason = confirm_qty_for_sale(row_index, ore_name, qty)
                if confirmed_qty is None:
                    print(
                        f"[ORES] row={row_index} ore={ore_name} qty={qty} "
                        f"sell blocked confidence={confirm_reason} samples={confirm_values}"
                    )
                    continue
                qty = confirmed_qty
                _LAST_CONFIRMED_QTY_BY_ORE[ore_name] = int(qty)
                actions = policy.decide_ore_sales({ore_name: qty}, reservations, config)
                action = actions[0] if actions else None
                if not action:
                    print(f"[ORES] row={row_index} ore={ore_name} qty={qty} reserved={reservations.get(ore_name, 0)}; skipping")
                    continue
                target = config.ORE_SELL_TARGET_DEFAULT
                reserved = reservations.get(ore_name, 0)
                allowed_to_sell = max(0, qty - max(target, reserved))
                if allowed_to_sell <= 0:
                    print(f"[ORES] row={row_index} ore={ore_name} qty={qty} allowed={allowed_to_sell}; skipping")
                    continue
                desired_fraction = allowed_to_sell / qty
                preset = choose_sell_preset(desired_fraction)
                precise_applied = try_apply_precise_sell_fraction(desired_fraction, qty)
                if not precise_applied and preset is None:
                    print(f"[ORES] row={row_index} ore={ore_name} qty={qty} allowed={allowed_to_sell} desired={desired_fraction:.2f} preset=None; skipping")
                    continue
                print(
                    f"[ORES] row={row_index} ore={ore_name} qty={qty} allowed={allowed_to_sell} "
                    f"desired={desired_fraction:.2f} mode={'precise' if precise_applied else 'preset'} "
                    f"preset={preset}"
                )
                if not precise_applied:
                    tap(config.SELL_PRESET_25_KEY, KEY_DELAY)
                    time.sleep(config.SELL_PRESET_APPLY_DELAY)
                    tap(preset, KEY_DELAY)  # set slider
                    time.sleep(config.SELL_PRESET_APPLY_DELAY)
                tap(config.SELL_CONFIRM_KEY, KEY_DELAY)  # execute sell
                sold_any = True
                sold_rows += 1
                verified, _post_bbox, after_qty = verify_sale_for_row(row_index, qty)
                if verified:
                    verified_sales += 1
                    print(f"[ORES] row={row_index} ore={ore_name} sell verified {qty}->{after_qty}")
                else:
                    print(f"[ORES] row={row_index} ore={ore_name} sell unverified before={qty} after={after_qty}")
                continue
            tap("\\", KEY_DELAY)  # open sell
            tap("'", KEY_DELAY)  # slider to ~100%
            tap("\\", KEY_DELAY)  # execute sell
            sold_any = True
            sold_rows += 1

        if missing_row_hit:
            break

        if page < pages - 1:
            tap(scroll_key, SCROLL_DELAY)
            time.sleep(config.ORES_PAGE_SCROLL_SETTLE_DELAY)

    # Close resources
    tap("shift+1", MENU_DELAY)
    tap("shift+1", MENU_DELAY)
    reset_ui()
    return {
        "sold_any": sold_any,
        "sold_rows": sold_rows,
        "verified_sales": verified_sales,
        "unverified_sales": max(0, sold_rows - verified_sales),
    }
