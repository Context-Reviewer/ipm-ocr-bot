from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from .domain_data import ORE_NAMES


@dataclass(slots=True)
class FocusConfig:
    required: bool = True
    window_substring: str = "BlueStacks App Player"
    excluded_substrings: tuple[str, ...] = ("Keymap Overlay",)
    auto_activate: bool = True
    activate_retry_delay: float = 0.25


@dataclass(slots=True)
class CaptureConfig:
    backend: str = "desktop"
    adb_path: str = "adb"
    adb_serial: str = ""
    cache_ttl_seconds: float = 0.2
    target_resolution: tuple[int, int] | None = None
    rects_path: str = "rects.json"
    window_title: str = "BlueStacks App Player"


@dataclass(slots=True)
class PerceptionConfig:
    backend: str = "hybrid"
    hybrid_order: str = "windows_first"
    openai_enabled: bool = True
    openai_model: str = "gpt-4.1-mini"
    openai_max_output_tokens: int = 96
    semantic_level_min: int = 1
    semantic_level_max: int = 999
    semantic_upgrade_cost_min: int = 1
    semantic_upgrade_cost_max: int = 1_000_000_000_000
    prompt_planet_title: str = "Read the planet title exactly as shown. Return only the visible title."
    prompt_planet_panel: str = (
        "Read all visible text from this planet menu crop in top-to-bottom order. "
        "Keep line breaks where possible."
    )
    prompt_numeric: str = "Read only the visible numeric value from this image. Return only the value."
    prompt_ore_name: str = "Read only the ore or resource name visible in this row."
    prompt_ore_panel: str = (
        "Read all visible text from this resources ores panel crop in top-to-bottom order. "
        "Keep line breaks where possible."
    )
    prompt_ore_quantity: str = "Read only the ore quantity visible in this row. Keep suffixes like K or M if present."


@dataclass(slots=True)
class SchedulerConfig:
    module_idle_seconds: float = 0.8
    heartbeat_seconds: float = 5.0
    tasks: Dict[str, float] = field(
        default_factory=lambda: {
            "planets": 60.0,
            "ores": 20.0,
        }
    )


@dataclass(slots=True)
class ActionConfig:
    key_delay_seconds: float = 0.08
    menu_delay_seconds: float = 0.12
    scroll_delay_seconds: float = 0.18
    open_planet_menu_key: str = "p"
    open_resources_key: str = "shift+1"
    open_production_key: str = "shift+2"
    ores_tab_key: str = "f1"
    increase_mining_key: str = "ctrl+1"
    increase_speed_key: str = "ctrl+2"
    increase_cargo_key: str = "ctrl+3"
    sell_open_key: str = "num5"
    sell_confirm_key: str = "\\"
    ore_select_keys: Dict[int, str] = field(
        default_factory=lambda: {
            1: "1",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
        }
    )
    sell_fraction_keys: Dict[float, str] = field(
        default_factory=lambda: {
            0.25: "shift+;",
            0.50: ";",
            0.75: "shift+'",
            1.00: "'",
        }
    )


@dataclass(slots=True)
class StarfieldConfig:
    enable_click_probe: bool = True
    click_probe_settle_seconds: float = 0.35
    save_probe_annotation: bool = True
    probe_annotation_dir: str = "out/starfield"
    ship_template_enabled: bool = True
    ship_template_path: str = "src/assets/ship_template.png"
    ship_template_threshold: float = 0.55
    ship_template_scales: tuple[float, ...] = (1.0, 0.75, 0.5, 0.35, 0.25, 0.18, 0.12, 0.08)
    ship_template_use_edges: bool = True
    ship_template_allow_fallback: bool = True
    scene_viewport: tuple[float, float, float, float] = (0.08, 0.12, 0.88, 0.94)
    scene_exclusion_zones: tuple[tuple[float, float, float, float], ...] = (
        (0.0, 0.0, 1.0, 0.08),
        (0.90, 0.0, 1.0, 1.0),
        (0.0, 0.88, 1.0, 1.0),
        (0.0, 0.0, 0.12, 0.18),
    )
    ship_exclusion_margin: int = 14
    ship_cluster_exclusion_x_margin: int = 60
    ship_cluster_exclusion_y_margin: int = 110
    candidate_min_radius: int = 6
    candidate_min_area: int = 80
    min_ship_bbox_width: int = 20
    min_ship_bbox_height: int = 8
    min_ship_area: int = 150
    max_ship_radius: int = 72
    max_ship_bbox_width: int = 140
    max_ship_bbox_height: int = 90
    max_ship_area_ratio: float = 0.08


@dataclass(slots=True)
class PolicyConfig:
    ore_sell_start_quantity: int = 25_000
    ore_keep_quantity: int = 25_000
    max_planet_upgrades_per_task: int = 5
    planet_panel_open_attempts: int = 3
    ore_sell_confirm_reads: int = 3
    ore_sell_max_relative_spread: float = 0.35
    known_ore_names: tuple[str, ...] = ORE_NAMES


@dataclass(slots=True)
class RuntimeConfig:
    focus: FocusConfig = field(default_factory=FocusConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    actions: ActionConfig = field(default_factory=ActionConfig)
    starfield: StarfieldConfig = field(default_factory=StarfieldConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    emergency_stop_hotkey: str = "f10"
    toggle_hotkey: str = "f9"
    visible_ore_rows: int = 6


def load_runtime_config() -> RuntimeConfig:
    return RuntimeConfig()
