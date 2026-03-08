from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict


class PlanetEntryPoint(TypedDict):
    point: tuple[int, int]
    planet_id: int
    verify: str


class PlanetEntryPointFile(TypedDict):
    point: list[int]
    planet_id: int
    verify: str

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
FOCUS_WINDOW_SUBSTR = "BlueStacks App Player"
FOCUS_WINDOW_EXCLUDE_SUBSTRINGS = ["Keymap Overlay"]
AUTO_ACTIVATE_FOCUS = True
FOCUS_ACTIVATE_RETRY_DELAY = 0.25

# Task configuration
TASKS: dict[str, dict[str, int | bool]] = {
    "planets": {"every": RUN_PLANETS_EVERY, "enabled": True},
    "ores": {"every": RUN_ORES_EVERY, "enabled": True},
}

# Confirmed project / tech progression for the current run.
TECH_STATE: dict[str, bool | int] = {
    "management": True,
    "asteroid_miner": True,
    "beacon": True,
    "resource_details": True,
    "asteroid_refined_drilling": True,
    "rover": True,
    "smelter": True,
    "crafter": True,
    "colonization": True,
    "colonization_scouting": True,
    "telescope_level": 1,
}


def configured_visible_planet_count() -> int:
    telescope_level = int(TECH_STATE.get("telescope_level", 0) or 0)
    # Base game exposes planet 1, and each telescope adds 3 more planets.
    return max(1, 1 + (3 * max(0, telescope_level)))


def normalize_visible_planet_count(count: int | None) -> int:
    try:
        count_int = int(count or 0)
    except Exception:
        count_int = 0
    if count_int <= 1:
        return 1

    try:
        max_planets = max(int(k) for k in PLANET_UNLOCK_PRICE.keys())
    except Exception:
        max_planets = 13

    remainder = (count_int - 1) % 3
    if remainder != 0:
        count_int += 3 - remainder
    return max(1, min(max_planets, count_int))


def visible_planet_count() -> int:
    override = int(TECH_STATE.get("visible_planet_count_override", 0) or 0)
    if override > 0:
        return normalize_visible_planet_count(override)
    return configured_visible_planet_count()


def set_visible_planet_count(count: int | None) -> None:
    try:
        count_int = int(count or 0)
    except Exception:
        return
    if count_int <= 0:
        return
    count_int = normalize_visible_planet_count(count_int)
    TECH_STATE["visible_planet_count_override"] = count_int
    TECH_STATE["telescope_level"] = max(0, (count_int - 1) // 3)


def clear_visible_planet_count_override() -> None:
    TECH_STATE.pop("visible_planet_count_override", None)


def calibrated_visible_planet_count() -> int:
    try:
        entries = get_planet_entry_points()
    except Exception:
        return 0
    calibrated = 0
    for entry in entries:
        try:
            calibrated = max(calibrated, int(entry.get("planet_id", 0) or 0))
        except Exception:
            continue
    return calibrated


def effective_visible_planet_count() -> int:
    return max(visible_planet_count(), calibrated_visible_planet_count())


def automation_visible_planet_count() -> int:
    return effective_visible_planet_count()


def unlocked_ore_names(visible_planets: int | None = None) -> list[str]:
    try:
        count = int(visible_planets or 0)
    except Exception:
        count = 0
    if count <= 0:
        count = automation_visible_planet_count()

    unlocked: list[str] = []
    seen: set[str] = set()
    for planet_id in range(1, count + 1):
        yields = PLANET_YIELDS.get(planet_id, {})
        if not isinstance(yields, dict):
            continue
        for ore_name in yields.keys():
            if not isinstance(ore_name, str) or not ore_name or ore_name in seen:
                continue
            unlocked.append(ore_name)
            seen.add(ore_name)
    return unlocked


def unlocked_ore_row_map(visible_planets: int | None = None) -> dict[int, str]:
    unlocked = set(unlocked_ore_names(visible_planets))
    active: dict[int, str] = {}
    row_index = 1
    for ore_name in ORE_ROW_MAP.values():
        if ore_name not in unlocked:
            continue
        active[row_index] = ore_name
        row_index += 1
    return active


def visible_ore_rows(visible_planets: int | None = None) -> int:
    return min(int(VISIBLE_ORE_ROWS), len(unlocked_ore_row_map(visible_planets)))

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
    11: 150000,
    12: 250000,
    13: 400000,
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
    11: {"Aluminium": 100},
    12: {"Lead": 45, "Silica": 35, "Silver": 20},
    13: {"Silver": 80, "Aluminium": 20},
}

# Ore base values
ORE_VALUE = {
    "Copper": 1,
    "Iron": 2,
    "Lead": 4,
    "Silica": 8,
    "Aluminium": 17,
    "Silver": 36,
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
MAX_UPGRADES_PER_PLANET_TASK = 8
MIN_ROI_TO_SPEND = 0.0
OPT_PLAN_WITH_BUDGET = True

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
RECT_ORES_TOP_ANCHOR = "ORES_TOP_ANCHOR"
ORES_SCROLL_UP_KEY = "["
ORES_SCROLL_DOWN_KEY = "]"
ORES_TOP_LATCH_MAX_STEPS = 25
ORES_TOP_LATCH_STABLE_READS = 3
ORES_TOP_LATCH_DIFF_THRESHOLD = 6.0
ORES_TOP_LATCH_SETTLE_DELAY = 0.12
ORES_RESET_SETTLE_DELAY = 0.35
ORES_PAGE_SCROLL_SETTLE_DELAY = 0.25
ORES_MENU_OPEN_DELAY = 0.60
ORES_ROW_SELECT_DELAY = 0.45

# OCR (optional)
ENABLE_ORE_OCR = True
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # leave empty to use PATH; optional override if needed
OCR_SNAP_DEBUG = False  # set True to save OCR debug crops
PERCEPTION_BACKEND = "hybrid"  # legacy | openai | hybrid
PERCEPTION_HYBRID_ORDER = "openai_first"  # legacy_first | openai_first
PERCEPTION_OPENAI_ENABLED = True
PERCEPTION_OPENAI_MODEL = "gpt-4.1-mini"
PERCEPTION_OPENAI_MAX_OUTPUT_TOKENS = 64
PERCEPTION_CACHE_SIZE = 64
PERCEPTION_DEFAULT_TEXT_PROMPT = ""
PERCEPTION_PLANET_TITLE_PROMPT = (
    "Read the planet title exactly as displayed in this game UI crop. "
    "Return only the visible title text, preserving the leading number and punctuation. "
    "Example: 2. DRASTA. If unreadable, return UNREADABLE."
)
PERCEPTION_ORE_QTY_PROMPT = (
    "Read the ore quantity shown in this game UI crop. "
    "Return only the quantity text exactly as displayed, including suffixes like K, M, or B. "
    "If unreadable, return UNREADABLE."
)
PERCEPTION_HUD_PRICE_PROMPT = (
    "Read the price or cash amount shown in this game UI crop. "
    "Return only the amount text exactly as displayed. If unreadable, return UNREADABLE."
)
# Cyan border sampling debug logs
CYAN_DEBUG = False
CYAN_BORDER_WIDTH = 4
CYAN_MIN_PIXEL_RATIO = 0.06
# Rects storage + BlueStacks anchor
RECTS_JSON_PATH = "rects.json"
RECTS_USE_CLIENT = False  # use rects.json keys for client-relative rects
BLUESTACKS_TITLE_HINT = "BlueStacks App Player"
HOME_NAV_CLICK = None
HOME_NAV_SETTLE_DELAY = 0.8
PLANET_ENTRY_SETTLE_DELAY = 1.0
PLANET_ENTRY_CLICK_OFFSETS = [
    (0, 0),
    (8, 0),
    (-8, 0),
    (0, 8),
    (0, -8),
    (12, 12),
    (-12, 12),
    (12, -12),
    (-12, -12),
]
PLANET_ENTRY_POINTS: list[PlanetEntryPoint] = [
    {"point": (250, 520), "planet_id": 1, "verify": "full"},
    {"point": (320, 490), "planet_id": 2, "verify": "full"},
    {"point": (385, 705), "planet_id": 3, "verify": "full"},
    {"point": (150, 740), "planet_id": 4, "verify": "full"},
]
PLANET_ENTRY_POINTS_PATH = "out/planet_entry_points.json"
STARFIELD_SHIP_CALIBRATION: dict[str, tuple[int, int] | int] = {
    "center": (88, 415),
    "width": 118,
    "height": 72,
}


def _normalize_planet_entry(entry: object) -> PlanetEntryPoint | None:
    if not isinstance(entry, dict):
        return None
    point = entry.get("point")
    planet_id = entry.get("planet_id")
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        return None
    if not isinstance(planet_id, (int, str)):
        return None
    if isinstance(planet_id, str) and not planet_id.strip():
        return None
    try:
        x, y = int(point[0]), int(point[1])
        pid = int(planet_id)
    except Exception:
        return None
    return {
        "point": (x, y),
        "planet_id": pid,
        "verify": str(entry.get("verify", "full")),
    }


def get_planet_entry_points() -> list[PlanetEntryPoint]:
    try:
        raw = Path(PLANET_ENTRY_POINTS_PATH)
        if raw.exists():
            data = json.loads(raw.read_text(encoding="utf-8"))
            if isinstance(data, list):
                entries: list[PlanetEntryPoint] = []
                for entry in data:
                    normalized = _normalize_planet_entry(entry)
                    if normalized is not None:
                        entries.append(normalized)
                if entries:
                    return entries
    except Exception:
        pass
    fallback_entries: list[PlanetEntryPoint] = []
    for entry in PLANET_ENTRY_POINTS:
        normalized = _normalize_planet_entry(entry)
        if normalized is not None:
            fallback_entries.append(normalized)
    return fallback_entries


def save_planet_entry_points(entries: list[PlanetEntryPoint]) -> bool:
    normalized: list[PlanetEntryPointFile] = []
    for entry in entries:
        normalized_entry = _normalize_planet_entry(entry)
        if normalized_entry is None:
            continue
        x, y = normalized_entry["point"]
        pid = normalized_entry["planet_id"]
        normalized.append(
            {
                "point": [x, y],
                "planet_id": pid,
                "verify": str(normalized_entry.get("verify", "full")),
            }
        )
    if not normalized:
        return False
    try:
        path = Path(PLANET_ENTRY_POINTS_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    except Exception:
        return False
    return True
# Rect editor settings
RECT_EDITOR_GRID_SIZE = 5
RECT_EDITOR_OCR_FPS = 4
RECT_EDITOR_DEFAULT_MODE = "generic"
# HUD cash bbox: (x, y, w, h) - TODO calibrate for your screen
RECT_HUD_CASH = "HUD_CASH"
# Planet stats panel bbox: (x, y, w, h)
PLANET_STATS_PANEL = "PLANET_STATS_PANEL"
PLANET_ROW_PAD_LEFT = 40
PLANET_ROW_PAD_RIGHT = 14
PLANET_ROW_PAD_TOP = 18
PLANET_ROW_PAD_BOTTOM = 10
PLANET_SWITCH_DELAY = 0.20
PLANET_OCR_RETRY_DELAY = 0.20
PLANET_MENU_OPEN_DELAY = 0.45
PLANET_MENU_OPEN_RETRIES = 3
PLANET_MENU_OPEN_RETRY_DELAY = 0.25
PLANET_PANEL_MIN_CYAN_PIXELS = 40
PLANET_PANEL_CHANGE_MAX_STEPS = 8
PLANET_PANEL_CHANGE_DIFF_THRESHOLD = 8.0
PLANET_PANEL_CHANGE_SETTLE_DELAY = 0.06
PLANET_DISCOVERY_SAMPLES = 3
PLANET_DISCOVERY_MAX_TOTAL_DELTA = 18
PLANET_DISCOVERY_MIN_MARGIN = 3
PLANET_SEED_DEBUG = False
PLANET_AUTO_DISCOVER_VISIBLE = True
PLANET_VERIFY_VISIBLE_EVERY_RUN = False
PLANET_CYCLE_DISCOVERY_MIN_STEPS = 2
PLANET_CYCLE_RETURN_DIFF_THRESHOLD = 4.0
PLANET_TITLE_READ_SAMPLES = 3
PLANET_TITLE_READ_DELAY = 0.04
PLANET_TITLE_NUMBER_WIDTH_RATIO = 0.36
PLANET_TITLE_NUMBER_PAD_X = 4
PLANET_TITLE_NUMBER_PAD_Y = 4
PLANET_NAV_CLICK_OFFSETS = [
    (0, 0),
    (8, 0),
    (-8, 0),
    (0, 8),
    (0, -8),
]
PLANET_NAV_VERIFY_RECHECKS = 2
PLANET_NAV_VERIFY_DELAY = 0.10
PLANET_PREFER_FORWARD_NAV = False
PLANET_RESET_CLEAR_ON_VISIBLE_DROP = True
PLANET_RESET_CLEAR_ON_SINGLE_VISIBLE = True
RUNTIME_STATE_PATH = "out/runtime_state.json"
WINDOW_RECT_CACHE_TTL = 0.5

ORE_QTY_SAMPLES = 7
ORE_QTY_SAMPLE_DELAY = 0.08
OCR_STABLE_SAMPLES = 3
OCR_STABLE_SAMPLE_DELAY = 0.05
OCR_STABLE_MIN_VALID_SAMPLES = 2
OCR_STABLE_MAX_REL_SPREAD = 0.15
OCR_TEXT_STABLE_SAMPLES = 3
OCR_TEXT_STABLE_SAMPLE_DELAY = 0.04
OCR_TEXT_STABLE_MIN_VALID_SAMPLES = 2

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
ORE_ROW_READ_PAD_LEFT = 170
ORE_ROW_READ_PAD_RIGHT = 12
ORE_ROW_READ_PAD_TOP = 10
ORE_ROW_READ_PAD_BOTTOM = 10

# OCR gating: reject if samples disagree too much
ORE_QTY_MIN_VALID_SAMPLES = 4
ORE_QTY_MAX_REL_SPREAD = 0.25   # max((max-min)/median) allowed

# OCR Y-offset scan (deterministic)
OCR_QTY_Y_OFFSETS = [0, -6, 6, -12, 12]
ORE_QTY_READ_RETRIES = 3
ORE_QTY_RETRY_DELAY = 0.12
ORES_ROW_SETTLE_MAX_STEPS = 6
ORES_ROW_SETTLE_STABLE_READS = 2
ORES_ROW_SETTLE_DIFF_THRESHOLD = 4.0
ORES_ROW_SETTLE_DELAY = 0.06
ORE_SELL_VERIFY_READS = 3
ORE_SELL_VERIFY_DELAY = 0.18
ORE_ROW_TEXT_SAMPLES = 3
ORE_ROW_TEXT_SAMPLE_DELAY = 0.05
ORE_ROW_TEXT_MIN_VALID_SAMPLES = 2
ORE_ROW_TEXT_MAX_REL_SPREAD = 0.25
ORE_SELL_CONFIRM_SAMPLES = 3
ORE_SELL_CONFIRM_SAMPLE_DELAY = 0.08
ORE_SELL_CONFIRM_MIN_VALID_SAMPLES = 2
ORE_SELL_CONFIRM_MAX_REL_SPREAD = 0.20
ORE_SELL_MAX_SUSPICIOUS_JUMP_RATIO = 6.0
ORE_SELL_MAX_SUSPICIOUS_JUMP_ABS = 50000
SELL_PRECISE_SLIDER_ENABLED = True
SELL_PRECISE_SLIDER_MAX_STEPS = 5
SELL_PRECISE_SLIDER_SETTLE_DELAY = 0.12
SELL_PRECISE_QTY_TOLERANCE_RATIO = 0.05
SELL_PRECISE_QTY_TOLERANCE_ABS = 250

# Selling policy
ORE_RESERVE_DEFAULT = 2500
ORE_SELL_START_DEFAULT = 25000
ORE_SELL_TARGET_DEFAULT = 25000

# Economy / Policy Engine (v0)
ECON_ENABLED = True
ECON_SAVING_MODE = False
ECON_MIN_ROI_WHEN_SAVING = 1e-4

# Optimizer modifier hooks (leave at 1.0 until you wire bonuses)
OPT_MINING_MULT = 1.0
OPT_SPEED_MULT = 1.0
OPT_CARGO_MULT = 1.0
OPT_VALUE_MULT = 1.0
OPT_LOOKAHEAD_DEPTH = 2
OPT_LOOKAHEAD_DISCOUNT = 0.85
OPT_BOTTLENECK_BONUS = 0.20
OPT_BOTTLENECK_BALANCE_TOLERANCE = 0.05

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
ORES_STOP_AT_FIRST_MISSING_ROW = True
ORES_ROW_VISIBLE_MIN_BRIGHT_RATIO = 0.02
ORES_ROW_VISIBLE_MIN_MEAN = 18.0

# Slider preset fractions (must match your mappings)
ORE_SLIDER_PRESETS = [
    ("shift+;", 0.25),
    (";", 0.50),
    ("shift+'", 0.75),
    ("'", 1.00),
]
SELL_SLIDER_TRACK_RECT = "SELL_SLIDER_TRACK"
SELL_SELECTED_QTY_RECT = "SELL_SELECTED_QTY"

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
