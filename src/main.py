"""
PROJECT CONTRACT (summary)
- Keyboard-only automation via BlueStacks key mappings.
- No OCR / no heavy computer vision. Avoid OpenCV/template matching.
- Keep modules small and deterministic. Fail-closed.
- F9 toggles AFK; F10 emergency stop.
See PROJECT_CONTRACT.md for full rules and keybindings.
"""
# AGENT OPERATING RULES:
# 1) Make minimal changes. Prefer adding new small modules over refactors.
# 2) Never add OCR/CV unless explicitly requested by the user.
# 3) Preserve existing keybindings and the F9/F10 control pattern.

import time
import os
import sys
import ctypes
import keyboard
import config
import input_utils
from input_utils import reset_ui, tap
from planets import planet_module
from ores import ore_module, debug_read_qty_for_row, qty_bbox_for_row
from ocr_utils import ocr_qty_median, grab_bbox, preprocess

RUNNING = False
DEBUG_KEYS = "--debug-keys" in sys.argv
TASK_ORDER = ["planets", "ores"]
next_at = {name: 0.0 for name in TASK_ORDER}
last_heartbeat = 0.0
task_ctx = {"last_ores": None}

if DEBUG_KEYS:
    input_utils.DEBUG = True

def get_active_window_title() -> str:
    try:
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        if not hwnd:
            return ""
        length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return ""
        buf = ctypes.create_unicode_buffer(length + 1)
        ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
        return buf.value
    except Exception:
        return ""

def is_focused_ok() -> bool:
    if not config.REQUIRE_FOCUS:
        return True
    title = get_active_window_title()
    return config.FOCUS_WINDOW_SUBSTR.lower() in title.lower()

def toggle():
    global RUNNING, next_at, last_heartbeat
    RUNNING = not RUNNING
    if RUNNING:
        now = time.monotonic()
        for name in TASK_ORDER:
            if config.TASKS.get(name, {}).get("enabled", False):
                next_at[name] = now
        last_heartbeat = 0.0
        print("[AFK] ON - scheduling tasks")
    print(f"AFK: {'ON' if RUNNING else 'OFF'}")

def selftest():
    print("SELFTEST")
    reset_ui()
    tap("shift+1")
    tap("shift+1")
    tap("p")
    tap("shift+1")
    tap("shift+1")
    print("SELFTEST DONE")

def ocr_calibration():
    qty = ocr_qty_median(
        config.ORE_QTY_BBOX,
        samples=config.ORE_QTY_SAMPLES,
        delay=config.ORE_QTY_SAMPLE_DELAY,
        min_valid=config.ORE_QTY_MIN_VALID_SAMPLES,
        max_rel_spread=config.ORE_QTY_MAX_REL_SPREAD,
    )
    print(f"[OCR] ore qty median = {qty}")

def debug_ocr_rows():
    for i in range(1, config.ORES_ROWS_TO_PROCESS + 1):
        bbox, qty = debug_read_qty_for_row(i)
        print(f"[DBG OCR] row={i} bbox={bbox} qty={qty}")

def debug_save_crops():
    os.makedirs("out", exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    for i in range(1, config.ORES_ROWS_TO_PROCESS + 1):
        bbox = qty_bbox_for_row(i)
        raw = grab_bbox(bbox)
        bw = preprocess(raw)
        raw_path = os.path.join("out", f"qty_row{i}_raw_{ts}.png")
        bw_path = os.path.join("out", f"qty_row{i}_bw_{ts}.png")
        raw.save(raw_path)
        bw.save(bw_path)
    print("[DBG OCR] crops saved")

keyboard.add_hotkey("f9", toggle)
keyboard.add_hotkey("f8", selftest)
keyboard.add_hotkey("f7", debug_save_crops)
keyboard.add_hotkey("f6", debug_ocr_rows)
keyboard.add_hotkey("f10", lambda: os._exit(0))

print("Ready. F9 toggles AFK. F10 exits. Ctrl+C quits.")

try:
    while True:
        if RUNNING:
            now = time.monotonic()

            if not is_focused_ok():
                if now - last_heartbeat >= config.HEARTBEAT_EVERY:
                    title = get_active_window_title()
                    print(f"[AFK] waiting for focus: {title}")
                    last_heartbeat = now
                time.sleep(config.MODULE_IDLE)
                continue

            if now - last_heartbeat >= config.HEARTBEAT_EVERY:
                title = get_active_window_title()
                print(f"[AFK] heartbeat | active='{title}'")
                last_heartbeat = now

            for name in TASK_ORDER:
                task_cfg = config.TASKS.get(name, {})
                if not task_cfg.get("enabled", False):
                    continue
                if now >= next_at.get(name, 0.0):
                    print(f"[TASK] {name} start")
                    reset_ui()
                    if name == "planets":
                        planet_module(planets=config.PLANETS_PER_TICK)
                    elif name == "ores":
                        result = ore_module(pages=config.ORE_PAGES_PER_TICK)
                        task_ctx["last_ores"] = result
                        if result and result.get("sold_any") and config.TASKS.get("planets", {}).get("enabled", False):
                            soon = time.monotonic() + 5.0
                            next_at["planets"] = min(next_at.get("planets", soon), soon)
                            print(f"[TASK] ores sold; planets scheduled sooner at +5s")
                    print(f"[TASK] {name} done")
                    next_at[name] = time.monotonic() + task_cfg.get("every", 0)

            time.sleep(config.MODULE_IDLE)
        else:
            time.sleep(config.MODULE_IDLE)
except KeyboardInterrupt:
    pass
