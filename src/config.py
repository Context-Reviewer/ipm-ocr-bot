# Timing configuration

KEY_DELAY = 0.08
MENU_DELAY = 0.12
SCROLL_DELAY = 0.18
MODULE_IDLE = 0.8

# Work chunk sizes
PLANETS_PER_TICK = 8
ORE_PAGES_PER_TICK = 1

# Scheduler intervals (seconds)
RUN_PLANETS_EVERY = 60
RUN_ORES_EVERY = 20

# Runtime observability
HEARTBEAT_EVERY = 5.0   # seconds between heartbeat prints while AFK is ON

# Safety
REQUIRE_FOCUS = True
FOCUS_WINDOW_SUBSTR = "BlueStacks"   # substring match against the active window title

# Task configuration
TASKS = {
    "planets": {"every": RUN_PLANETS_EVERY, "enabled": True},
    "ores": {"every": RUN_ORES_EVERY, "enabled": True},
}

PLANET_INITIAL_LEVELS = {
    1: {"m": 45, "s": 41, "c": 38},
    2: {"m": 31, "s": 29, "c": 26},
    3: {"m": 30, "s": 24, "c": 22},
    4: {"m": 26, "s": 20, "c": 18},
    5: {"m": 17, "s": 12, "c": 14},
    6: {"m": 14, "s": 9,  "c": 8},
    7: {"m": 12, "s": 8,  "c": 10},
}

# Planet unlock prices
PLANET_UNLOCK_PRICE = {
    1: 100,
    2: 200,
    3: 500,
    4: 1250,
    5: 5000,
    6: 9000,
    7: 15000,
    8: 25000,
    9: 40000,
    10: 75000,
}

# Planet yields (percentage mix)
PLANET_YIELDS = {
    1: {"Copper": 100},
    2: {"Copper": 80, "Iron": 20},
    3: {"Copper": 50, "Iron": 50},
    4: {"Iron": 80, "Lead": 20},
    5: {"Lead": 50, "Iron": 30, "Copper": 20},
    6: {"Lead": 100},
    7: {"Iron": 40, "Copper": 40, "Silica": 20},
    8: {"Silica": 60, "Copper": 40},
    9: {"Silica": 80, "Aluminium": 20},
    10: {"Aluminium": 50, "Silica": 30, "Lead": 20},
}

# Ore base values
ORE_VALUE = {
    "Copper": 1,
    "Iron": 2,
    "Lead": 4,
    "Silica": 8,
    "Aluminium": 17,
}

# Ore selection hotkeys
ORE_SELECT_KEYS = {
    "Copper": "1",
    "Iron": "2",
    "Lead": "3",
    "Silica": "4",
    "Aluminium": "5",
}

# ROI optimizer limits
MAX_UPGRADES_PER_PLANET_TASK = 3
MIN_ROI_TO_SPEND = 0.0

# Cycle time model (seconds per round trip)
DEFAULT_CYCLE_SECONDS = 0.5
PLANET_CYCLE_SECONDS = {
    1: 0.5,
    2: 0.5,
    3: 0.5,
    4: 0.5,
    5: 0.5,
    6: 0.5,
    7: 0.5,
    8: 0.5,
    9: 0.5,
    10: 0.5,
}

# Fill-ratio governor
FILL_TARGET = 0.95
FILL_BAND = 0.05
USE_SPEED_IN_CYCLE_MODEL = False

# Ores top latch via anchor patch stability (no scrollbar needed)
RECT_ORES_TOP_ANCHOR = (1330, 547, 30, 30)  # calibrated anchor patch (x,y,w,h)
ORES_SCROLL_UP_KEY = "["
ORES_SCROLL_DOWN_KEY = "]"
ORES_TOP_LATCH_MAX_STEPS = 25
ORES_TOP_LATCH_STABLE_READS = 3
ORES_TOP_LATCH_DIFF_THRESHOLD = 6.0
ORES_TOP_LATCH_SETTLE_DELAY = 0.12
ORES_RESET_SETTLE_DELAY = 0.35
ORES_PAGE_SCROLL_SETTLE_DELAY = 0.25

# OCR (optional)
ENABLE_ORE_OCR = True
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # leave empty to use PATH; optional override if needed
OCR_SNAP_DEBUG = True  # set True to save OCR debug crops
# Cyan border sampling debug logs
CYAN_DEBUG = False
# HUD cash bbox: (x, y, w, h) - TODO calibrate for your screen
RECT_HUD_CASH = (0, 0, 0, 0)
# Planet stats panel bbox: (x, y, w, h)
PLANET_STATS_PANEL = (1443, 697, 149, 263)
PLANET_SWITCH_DELAY = 0.20
PLANET_OCR_RETRY_DELAY = 0.20

ORE_QTY_SAMPLES = 7
ORE_QTY_SAMPLE_DELAY = 0.08

# Selected-row quantity bbox in ABSOLUTE SCREEN coords (x, y, w, h)
ORE_QTY_BBOX = (1565, 572, 100, 30)  # placeholder; will calibrate

# Quantity column strip covering visible ore rows (absolute screen coords)
# Derived from TL=(1563,557), BR=(1670,841) then trimmed
ORE_QTY_STRIP = (1563, 563, 107, 272)   # (x,y,w,h) covering rows 1-5
RECT_ORE_QTY_STRIP = ORE_QTY_STRIP

# Per-row bbox layout
ORE_QTY_BOX_W = 98
ORE_QTY_BOX_H = 44

# Padding applied to each derived row bbox
ORE_QTY_BBOX_PAD_X = 2
ORE_QTY_BBOX_PAD_Y = 2

# OCR gating: reject if samples disagree too much
ORE_QTY_MIN_VALID_SAMPLES = 4
ORE_QTY_MAX_REL_SPREAD = 0.25   # max((max-min)/median) allowed

# OCR Y-offset scan (deterministic)
OCR_QTY_Y_OFFSETS = [0, -6, 6, -12, 12]

# Selling policy
ORE_RESERVE_DEFAULT = 2500
ORE_SELL_START_DEFAULT = 25000
ORE_SELL_TARGET_DEFAULT = 25000

# Economy / Policy Engine (v0)
ECON_ENABLED = True
ECON_SAVING_MODE = False
ECON_MIN_ROI_WHEN_SAVING = 1e-4

ORE_KEEP_FLOOR_DEFAULT = ORE_RESERVE_DEFAULT
ORE_KEEP_OVERRIDES = {}

BAR_KEEP_FLOOR_DEFAULT = 0
BAR_KEEP_OVERRIDES = {}

ITEM_KEEP_FLOOR_DEFAULT = 0
ITEM_KEEP_OVERRIDES = {}

TECH_RESERVES = {}
CRAFT_RESERVES = {}
SMELTER_FEED_RESERVES = {}

# Per-row overrides (top to bottom at current stage)
# 1=copper, 2=iron, 3=lead, 4=silica
ORE_RESERVE_BY_ROW = {
    1: 2500,
    2: 2500,
    3: 2500,
    4: 2500,
}
ORE_SELL_START_BY_ROW = {
    1: 25000,
    2: 25000,
    3: 25000,
    4: 25000,
}

ORE_ROW_MAP = {
    1: "Copper",
    2: "Iron",
    3: "Lead",
    4: "Silica",
    5: "Aluminium",
}

# Ores processing
ORE_QTY_STRIP_ROWS = len(ORE_ROW_MAP)
ORES_ROWS_TO_PROCESS = len(ORE_ROW_MAP)
VISIBLE_ORE_ROWS = 5  # current UI shows 5 ores without scrolling

# Slider preset fractions (must match your mappings)
ORE_SLIDER_PRESETS = [
    ("shift+;", 0.25),
    (";", 0.50),
    ("shift+'", 0.75),
    ("'", 1.00),
]

# Hysteresis: only sell if qty >= reserve + buffer
ORE_SELL_BUFFER_DEFAULT = 0

# Optional per-row override (row 1..4)
ORE_SELL_BUFFER = {
    1: 5000,   # copper buffer
    2: 2000,   # iron
    3: 1000,   # lead
    4: 500,    # silica
}

# Sell slider preset keys (BlueStacks)
SELL_PRESET_25_KEY = "shift+;"
SELL_PRESET_50_KEY = ";"
SELL_PRESET_75_KEY = "shift+'"
SELL_PRESET_100_KEY = "'"
SELL_CONFIRM_KEY = "\\"

# Excess thresholds (qty - reserve) to choose sell presets
# Tune later if needed
SELL_EXCESS_T1 = 10_000     # under this, sell 25%
SELL_EXCESS_T2 = 50_000     # under this, sell 50%
SELL_EXCESS_T3 = 200_000    # under this, sell 75%
# >= T3 sell 100%

# Sell timing
SELL_PRESET_APPLY_DELAY = 0.12   # wait after setting slider preset before pressing sell

# Safety floor: if qty < reserve * SELL_MIN_RESERVE_FRACTION, never sell
SELL_MIN_RESERVE_FRACTION = 0.10  # 10% of reserve
