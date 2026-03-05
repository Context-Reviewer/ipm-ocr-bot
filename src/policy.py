from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import bars_data
import items_data
import config


@dataclass(frozen=True)
class StateSnapshot:
    ores: Optional[dict] = None
    planet_levels: Optional[dict] = None
    bars: Optional[dict] = None
    items: Optional[dict] = None
    smelter_queue: Optional[dict] = None
    crafter_queue: Optional[dict] = None
    cash: Optional[float] = None


def _cfg_value(cfg, name, fallback=None):
    if hasattr(cfg, name):
        return getattr(cfg, name)
    return fallback


def compute_reservations(state: Optional[StateSnapshot], cfg=config) -> dict:
    reservations: dict[str, int] = {}

    ore_floor = int(_cfg_value(cfg, "ORE_KEEP_FLOOR_DEFAULT", 0) or 0)
    ore_overrides = _cfg_value(cfg, "ORE_KEEP_OVERRIDES", {})
    ore_map = _cfg_value(cfg, "ORE_ROW_MAP", {})
    ore_reserve_by_row = _cfg_value(cfg, "ORE_RESERVE_BY_ROW", {})

    for row_index, ore_name in ore_map.items():
        if not isinstance(ore_name, str) or not ore_name:
            continue
        floor = ore_floor
        if isinstance(ore_overrides, dict) and ore_name in ore_overrides:
            val = ore_overrides.get(ore_name)
            if isinstance(val, int) and val >= 0:
                floor = val
        elif isinstance(ore_reserve_by_row, dict) and row_index in ore_reserve_by_row:
            val = ore_reserve_by_row.get(row_index)
            if isinstance(val, int) and val >= 0:
                floor = max(floor, val)
        reservations[ore_name] = max(reservations.get(ore_name, 0), floor)

    bar_floor = int(_cfg_value(cfg, "BAR_KEEP_FLOOR_DEFAULT", 0) or 0)
    bar_overrides = _cfg_value(cfg, "BAR_KEEP_OVERRIDES", {})
    for bar_name in bars_data.list_bars():
        floor = bar_floor
        if isinstance(bar_overrides, dict) and bar_name in bar_overrides:
            val = bar_overrides.get(bar_name)
            if isinstance(val, int) and val >= 0:
                floor = val
        reservations[bar_name] = max(reservations.get(bar_name, 0), floor)

    item_floor = int(_cfg_value(cfg, "ITEM_KEEP_FLOOR_DEFAULT", 0) or 0)
    item_overrides = _cfg_value(cfg, "ITEM_KEEP_OVERRIDES", {})
    for item_name in items_data.list_items():
        floor = item_floor
        if isinstance(item_overrides, dict) and item_name in item_overrides:
            val = item_overrides.get(item_name)
            if isinstance(val, int) and val >= 0:
                floor = val
        reservations[item_name] = max(reservations.get(item_name, 0), floor)

    tech = _cfg_value(cfg, "TECH_RESERVES", {})
    if isinstance(tech, dict):
        for name, qty in tech.items():
            if isinstance(name, str) and isinstance(qty, int) and qty > 0:
                reservations[name] = max(reservations.get(name, 0), qty)

    craft = _cfg_value(cfg, "CRAFT_RESERVES", {})
    if isinstance(craft, dict):
        for name, qty in craft.items():
            if isinstance(name, str) and isinstance(qty, int) and qty > 0:
                reservations[name] = max(reservations.get(name, 0), qty)

    smelter = _cfg_value(cfg, "SMELTER_FEED_RESERVES", {})
    if isinstance(smelter, dict):
        for name, qty in smelter.items():
            if isinstance(name, str) and isinstance(qty, int) and qty > 0:
                reservations[name] = max(reservations.get(name, 0), qty)

    return reservations


def decide_ore_sales(ore_qty_by_name_or_row: dict, reservations: dict, cfg=config) -> list[dict]:
    actions: list[dict] = []
    if not isinstance(ore_qty_by_name_or_row, dict):
        return actions

    ore_map = _cfg_value(cfg, "ORE_ROW_MAP", {})

    for key, qty in ore_qty_by_name_or_row.items():
        if isinstance(key, int):
            ore_name = ore_map.get(key)
        else:
            ore_name = key

        if not isinstance(ore_name, str) or not ore_name:
            continue
        if not isinstance(qty, int) or qty <= 0:
            continue
        reserved = int(reservations.get(ore_name, 0) or 0)
        sell_qty = max(0, qty - reserved)
        if sell_qty <= 0:
            continue
        actions.append({"name": ore_name, "qty_to_sell": sell_qty})

    return actions


def allow_upgrade(candidate: dict, cfg=config, state: Optional[StateSnapshot] = None) -> bool:
    if not isinstance(candidate, dict):
        return False
    roi = candidate.get("roi")
    if not isinstance(roi, (int, float)):
        return False

    enabled = bool(_cfg_value(cfg, "ECON_ENABLED", False))
    if not enabled:
        return True

    saving_mode = bool(_cfg_value(cfg, "ECON_SAVING_MODE", False))
    if not saving_mode:
        return True

    min_roi = float(_cfg_value(cfg, "ECON_MIN_ROI_WHEN_SAVING", 0.0) or 0.0)
    return roi >= min_roi
