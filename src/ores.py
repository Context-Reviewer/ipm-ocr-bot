import time
import config
from config import KEY_DELAY, MENU_DELAY, SCROLL_DELAY
from input_utils import tap
from signals import mean_abs_diff
from ocr_utils import ocr_qty_median

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
    strip_x, strip_y, strip_w, strip_h = config.RECT_ORE_QTY_STRIP
    row_count = len(config.ORE_ROW_MAP)
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

def debug_read_qty_for_row(row_index: int):
    bbox = qty_bbox_for_row(row_index)
    qty = ocr_qty_median(
        bbox,
        samples=config.ORE_QTY_SAMPLES,
        delay=config.ORE_QTY_SAMPLE_DELAY,
        min_valid=config.ORE_QTY_MIN_VALID_SAMPLES,
        max_rel_spread=config.ORE_QTY_MAX_REL_SPREAD,
    )
    return (bbox, qty)

def ore_module(pages: int = 2):
    # Open resources
    tap("shift+1", MENU_DELAY)
    tap("f1", MENU_DELAY)  # ores tab

    # Reset to top (safe even if already at top)
    if not scroll_to_top():
        print("[ORES] top latch failed")
    time.sleep(config.ORES_RESET_SETTLE_DELAY)

    rows = min(len(config.ORE_ROW_MAP), config.VISIBLE_ORE_ROWS)
    if len(config.ORE_ROW_MAP) > config.VISIBLE_ORE_ROWS:
        print(f"[ORES] ores_unlocked={len(config.ORE_ROW_MAP)} > visible={config.VISIBLE_ORE_ROWS}; skipping rows beyond visible (fail-closed)")
    row_keys = [str(i) for i in range(1, rows + 1)]
    scroll_key = config.ORES_SCROLL_DOWN_KEY  # BlueStacks swipe down

    for page in range(pages):
        for key in row_keys:
            row_index = int(key)
            ore_name = config.ORE_ROW_MAP.get(row_index)
            if ore_name is None:
                print(f"[ORES] row={row_index} no mapping; skipping")
                continue
            tap(key, KEY_DELAY)
            if config.ENABLE_ORE_OCR:
                time.sleep(MENU_DELAY)
                reserve = config.ORE_RESERVE_BY_ROW.get(row_index, config.ORE_RESERVE_DEFAULT)
                sell_start = config.ORE_SELL_START_BY_ROW.get(row_index, config.ORE_SELL_START_DEFAULT)
                bbox = qty_bbox_for_row(row_index)
                qty = ocr_qty_median(
                    bbox,
                    samples=config.ORE_QTY_SAMPLES,
                    delay=config.ORE_QTY_SAMPLE_DELAY,
                    min_valid=config.ORE_QTY_MIN_VALID_SAMPLES,
                    max_rel_spread=config.ORE_QTY_MAX_REL_SPREAD,
                )
                if qty is None:
                    print(f"[ORES] row={row_index} qty OCR failed after y-scan y={bbox[1]} bbox={bbox} offsets={config.OCR_QTY_Y_OFFSETS}; skipping row")
                    continue
                if qty < sell_start:
                    print(f"[ORES] row={row_index} qty={qty} < sell_start={sell_start}; skipping")
                    continue
                target = config.ORE_SELL_TARGET_DEFAULT
                allowed_to_sell = max(0, qty - target)
                if allowed_to_sell <= 0:
                    print(f"[ORES] row={row_index} qty={qty} <= target={target}; skipping")
                    continue
                desired_fraction = allowed_to_sell / qty
                preset = None
                for key_name, fraction in config.ORE_SLIDER_PRESETS:
                    if fraction <= desired_fraction:
                        preset = key_name
                if preset is None:
                    print(f"[ORES] row={row_index} qty={qty} target={target} allowed={allowed_to_sell} desired={desired_fraction:.2f} preset=None; skipping")
                    continue
                print(f"[ORES] row={row_index} qty={qty} target={target} allowed={allowed_to_sell} desired={desired_fraction:.2f} preset={preset}")
                tap(config.SELL_PRESET_25_KEY, KEY_DELAY)
                time.sleep(config.SELL_PRESET_APPLY_DELAY)
                tap(preset, KEY_DELAY)  # set slider
                time.sleep(config.SELL_PRESET_APPLY_DELAY)
                tap(config.SELL_CONFIRM_KEY, KEY_DELAY)  # execute sell
                continue
            tap("\\", KEY_DELAY)  # open sell
            tap("'", KEY_DELAY)  # slider to ~100%
            tap("\\", KEY_DELAY)  # execute sell

        if page < pages - 1:
            tap(scroll_key, SCROLL_DELAY)
            time.sleep(config.ORES_PAGE_SCROLL_SETTLE_DELAY)

    # Close resources
    tap("shift+1", MENU_DELAY)
    tap("shift+1", MENU_DELAY)
