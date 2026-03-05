import json
import logging
from pathlib import Path

_LOG = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def load_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        _LOG.warning("Failed to load JSON from %s: %s", path, exc)
        return None

def validate_ores(ores):
    if not isinstance(ores, dict):
        return False
    for k, v in ores.items():
        if not isinstance(k, str):
            return False
        if not isinstance(v, dict):
            return False
        base = v.get("base_value")
        if not isinstance(base, (int, float)) or base <= 0:
            return False
    return True

def validate_planets(planets):
    if not isinstance(planets, dict):
        return False
    for k, v in planets.items():
        if not isinstance(k, str) or not k.isdigit():
            return False
        if not isinstance(v, dict):
            return False
        name = v.get("name")
        if not isinstance(name, str) or not name:
            return False
        unlock_price = v.get("unlock_price")
        distance = v.get("distance")
        yields = v.get("yields")
        if not isinstance(unlock_price, (int, float)):
            return False
        if not isinstance(distance, (int, float)):
            return False
        if not isinstance(yields, dict):
            return False
        for ore, pct in yields.items():
            if not isinstance(ore, str):
                return False
            if not isinstance(pct, (int, float)):
                return False
    return True

def load_ores():
    path = _DATA_DIR / "ores.json"
    data = load_json(path)
    if data is None or not validate_ores(data):
        _LOG.warning("Invalid ores data; using empty set")
        return {}
    return data

def load_planets():
    path = _DATA_DIR / "planets.json"
    data = load_json(path)
    if data is None or not validate_planets(data):
        _LOG.warning("Invalid planets data; using empty set")
        return {}
    return data

ORES = load_ores()
PLANETS = load_planets()
