import config
from analytics import mining_rate, ship_speed, cargo_cap
from data_store import ORES

ORE_VALUE = {k: v["base_value"] for k, v in ORES.items()}

def weighted_value(planet_index: int):
    yields = config.PLANET_YIELDS.get(planet_index)
    if not yields:
        return None
    total = 0.0
    for ore, pct in yields.items():
        value = ORE_VALUE.get(ore)
        if value is None:
            return None
        total += (pct / 100.0) * value
    if total <= 0:
        return None
    return total

def _effective_cycle_seconds(planet_index: int, levels: dict):
    base_cycle = config.PLANET_CYCLE_SECONDS.get(planet_index, config.DEFAULT_CYCLE_SECONDS)
    if base_cycle is None:
        return None
    if not config.USE_SPEED_IN_CYCLE_MODEL:
        return base_cycle
    base_speed_level = levels.get("s_base")
    cur_speed_level = levels.get("s")
    if base_speed_level is None or cur_speed_level is None:
        return None
    base_speed = ship_speed(base_speed_level)
    cur_speed = ship_speed(cur_speed_level)
    if cur_speed <= 0:
        return None
    return base_cycle * (base_speed / cur_speed)

def revenue_per_sec(planet_index: int, levels: dict):
    cycle = _effective_cycle_seconds(planet_index, levels)
    if cycle is None or cycle <= 0:
        return None
    value = weighted_value(planet_index)
    if value is None:
        return None
    prod_cycle = mining_rate(levels["m"]) * cycle
    cap = cargo_cap(levels["c"])
    delivered = prod_cycle if prod_cycle < cap else cap
    return (delivered * value) / cycle

def upgrade_cost(unlock_price: float, level: int):
    return (unlock_price / 20) * (1.3 ** (level - 1))

def planet_candidates(planet_index: int, levels: dict, cyan_flags: dict):
    unlock_price = config.PLANET_UNLOCK_PRICE.get(planet_index)
    if unlock_price is None:
        return []
    rev_now = revenue_per_sec(planet_index, levels)
    if rev_now is None:
        return []

    candidates = []
    level_sum = levels.get("m", 0) + levels.get("s", 0) + levels.get("c", 0)
    for stat, key in (("M", "m"), ("C", "c"), ("S", "s")):
        if not cyan_flags.get(stat, False):
            continue
        if stat == "S" and not config.USE_SPEED_IN_CYCLE_MODEL:
            continue
        levels_up = {"m": levels["m"], "s": levels["s"], "c": levels["c"], "s_base": levels.get("s_base")}
        levels_up[key] += 1
        rev_after = revenue_per_sec(planet_index, levels_up)
        if rev_after is None:
            continue
        delta = rev_after - rev_now
        cost = upgrade_cost(unlock_price, levels[key])
        if cost <= 0:
            continue
        roi = delta / cost
        boost = None
        if level_sum <= 6:
            roi *= 2.0
            boost = 2.0
        candidates.append(
            {"planet": planet_index, "stat": stat, "roi": roi, "delta_rev": delta, "cost": cost, "boost": boost}
        )
    return candidates
