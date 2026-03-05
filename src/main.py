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
import keyboard
import config
from input_utils import reset_ui
from planets import planet_module
from ores import ore_module

RUNNING = False
NEXT_PLANETS_AT = 0.0
NEXT_ORES_AT = 0.0

def toggle():
    global RUNNING, NEXT_PLANETS_AT, NEXT_ORES_AT
    RUNNING = not RUNNING
    if RUNNING:
        now = time.monotonic()
        NEXT_PLANETS_AT = now
        NEXT_ORES_AT = now
    print(f"AFK: {'ON' if RUNNING else 'OFF'}")

keyboard.add_hotkey("f9", toggle)
keyboard.add_hotkey("f10", lambda: os._exit(0))

print("Ready. F9 toggles AFK. F10 exits. Ctrl+C quits.")

try:
    while True:
        if RUNNING:
            now = time.monotonic()

            if now >= NEXT_PLANETS_AT:
                reset_ui()
                planet_module(planets=config.PLANETS_PER_TICK)
                NEXT_PLANETS_AT = time.monotonic() + config.RUN_PLANETS_EVERY

            if now >= NEXT_ORES_AT:
                reset_ui()
                ore_module(pages=config.ORE_PAGES_PER_TICK)
                NEXT_ORES_AT = time.monotonic() + config.RUN_ORES_EVERY

            time.sleep(config.MODULE_IDLE)
        else:
            time.sleep(config.MODULE_IDLE)
except KeyboardInterrupt:
    pass
