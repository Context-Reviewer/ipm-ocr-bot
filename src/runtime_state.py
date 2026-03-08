from __future__ import annotations

import json
from pathlib import Path

import config


def _state_path() -> Path:
    return Path(getattr(config, "RUNTIME_STATE_PATH", "out/runtime_state.json"))


def load_runtime_state() -> dict:
    path = _state_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def save_runtime_state(
    *,
    planet_levels: dict | None = None,
    current_planet_id: int | None = None,
    visible_planet_count: int | None = None,
    preferred_planet_entry_point: tuple[int, int] | None = None,
) -> bool:
    path = _state_path()
    payload: dict = {}

    if isinstance(planet_levels, dict):
        saved_levels = {}
        for planet_id, levels in planet_levels.items():
            try:
                saved_levels[str(int(planet_id))] = {
                    "m": int(levels["m"]),
                    "s": int(levels["s"]),
                    "c": int(levels["c"]),
                }
            except Exception:
                continue
        payload["planet_levels"] = saved_levels

    if current_planet_id is not None:
        try:
            payload["current_planet_id"] = int(current_planet_id)
        except Exception:
            pass

    if visible_planet_count is not None:
        try:
            payload["visible_planet_count"] = int(visible_planet_count)
        except Exception:
            pass

    if preferred_planet_entry_point is not None:
        try:
            x, y = preferred_planet_entry_point
            payload["preferred_planet_entry_point"] = [int(x), int(y)]
        except Exception:
            pass

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
    except Exception:
        return False
    return True


def clear_runtime_state() -> bool:
    path = _state_path()
    try:
        if path.exists():
            path.unlink()
    except Exception:
        return False
    return True
