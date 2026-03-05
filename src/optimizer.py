from __future__ import annotations

from typing import Optional


def mine_rate(level: int) -> float:
    n = max(0, level - 1)
    return 0.25 + 0.1 * n + 0.017 * n * n


def ship_speed(level: int) -> float:
    n = max(0, level - 1)
    return 1 + 0.2 * n + (1 / 75) * n * n


def cargo_cap(level: int) -> float:
    n = max(0, level - 1)
    return 5 + 2 * n + 0.1 * n * n


def upgrade_cost(unlock_price: float, level: int) -> float:
    return (unlock_price / 20) * (1.3 ** (level - 1))


def delivered_throughput(levels: dict, planet_params: dict) -> Optional[float]:
    try:
        m = int(levels.get("m"))
        s = int(levels.get("s"))
        c = int(levels.get("c"))
    except Exception:
        return None

    if m <= 0 or s <= 0 or c <= 0:
        return None

    mining = mine_rate(m)
    speed = ship_speed(s)
    cargo = cargo_cap(c)

    dist = planet_params.get("dist", planet_params.get("distance", 1.0))
    overhead = planet_params.get("overhead", 0.0)
    try:
        dist = float(dist)
        overhead = float(overhead)
    except Exception:
        return None

    if speed <= 0:
        return None

    denom = overhead + dist / speed
    if denom <= 0:
        return None

    ship_rate = cargo / denom
    return mining if mining < ship_rate else ship_rate


def choose_best_upgrades(levels_by_planet: dict, planets_cfg: dict, top_n: int = 3) -> list[dict]:
    candidates: list[dict] = []

    for planet_id, levels in levels_by_planet.items():
        key = str(planet_id)
        planet_cfg = planets_cfg.get(key) if isinstance(planets_cfg, dict) else None
        if not planet_cfg:
            planet_cfg = planets_cfg.get(planet_id) if isinstance(planets_cfg, dict) else None
        if not isinstance(planet_cfg, dict):
            continue

        unlock_price = planet_cfg.get("unlock_price")
        if not isinstance(unlock_price, (int, float)) or unlock_price <= 0:
            continue

        planet_params = {
            "dist": planet_cfg.get("dist", planet_cfg.get("distance", 1.0)),
            "overhead": planet_cfg.get("overhead", 0.0),
        }

        base = delivered_throughput(levels, planet_params)
        if base is None:
            continue

        for stat, key in (("M", "m"), ("S", "s"), ("C", "c")):
            level_now = levels.get(key)
            if not isinstance(level_now, int) or level_now <= 0:
                continue
            levels_after = {"m": levels["m"], "s": levels["s"], "c": levels["c"]}
            levels_after[key] = level_now + 1

            after = delivered_throughput(levels_after, planet_params)
            if after is None:
                continue

            delta = after - base
            if delta <= 0:
                continue

            cost = upgrade_cost(unlock_price, level_now)
            if cost <= 0:
                continue

            roi = delta / cost
            candidates.append(
                {
                    "planet_id": planet_id,
                    "stat": stat,
                    "cost": cost,
                    "roi": roi,
                    "delta": delta,
                    "levels_before": {"m": levels["m"], "s": levels["s"], "c": levels["c"]},
                    "levels_after": levels_after,
                }
            )

    stat_order = {"M": 0, "S": 1, "C": 2}
    candidates.sort(key=lambda c: (-c["roi"], c["planet_id"], stat_order.get(c["stat"], 9)))
    return candidates[: max(0, int(top_n))]
