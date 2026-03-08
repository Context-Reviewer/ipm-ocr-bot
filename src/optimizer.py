from __future__ import annotations

from typing import Optional

from data_store import ORES


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


def _weighted_ore_price(planet_cfg: dict) -> Optional[float]:
    yields = planet_cfg.get("yields")
    if not isinstance(yields, dict) or not yields:
        return None
    total = 0.0
    for ore, pct in yields.items():
        if not isinstance(ore, str) or not isinstance(pct, (int, float)):
            return None
        ore_cfg = ORES.get(ore)
        if not isinstance(ore_cfg, dict):
            return None
        base_value = ore_cfg.get("base_value")
        if not isinstance(base_value, (int, float)):
            return None
        total += (float(pct) / 100.0) * float(base_value)
    return total if total > 0 else None


def compute_mining_output(level: int, mining_multiplier_mods: float = 1.0) -> Optional[float]:
    if level <= 0:
        return None
    base = mine_rate(level)
    if base <= 0:
        return None
    return base * float(mining_multiplier_mods)


def compute_ship_speed(level: int, speed_multiplier_mods: float = 1.0) -> Optional[float]:
    if level <= 0:
        return None
    base = ship_speed(level)
    if base <= 0:
        return None
    return base * float(speed_multiplier_mods)


def compute_cargo_capacity(level: int, cargo_multiplier_mods: float = 1.0) -> Optional[int]:
    if level <= 0:
        return None
    base = cargo_cap(level)
    if base <= 0:
        return None
    return int(round(base * float(cargo_multiplier_mods)))


def compute_transport_output(cargo_capacity: int, ship_speed_val: float, distance: float) -> Optional[float]:
    try:
        dist = float(distance)
    except Exception:
        return None
    if cargo_capacity is None or ship_speed_val is None:
        return None
    if cargo_capacity <= 0 or ship_speed_val <= 0 or dist <= 0:
        return None
    return (float(cargo_capacity) * float(ship_speed_val)) / (2.0 * dist)


def compute_effective_output(mining_output: float, transport_output: float) -> Optional[float]:
    if mining_output is None or transport_output is None:
        return None
    return mining_output if mining_output < transport_output else transport_output


def classify_bottleneck(
    levels: dict,
    planet_params: dict,
    *,
    mining_multiplier_mods: float = 1.0,
    speed_multiplier_mods: float = 1.0,
    cargo_multiplier_mods: float = 1.0,
    balance_tolerance: float = 0.05,
) -> Optional[str]:
    try:
        m = int(levels.get("m"))
        s = int(levels.get("s"))
        c = int(levels.get("c"))
    except Exception:
        return None

    mining_output = compute_mining_output(m, mining_multiplier_mods)
    ship_speed_val = compute_ship_speed(s, speed_multiplier_mods)
    cargo_capacity = compute_cargo_capacity(c, cargo_multiplier_mods)
    if mining_output is None or ship_speed_val is None or cargo_capacity is None:
        return None
    dist = planet_params.get("distance", planet_params.get("dist", None))
    transport_output = compute_transport_output(cargo_capacity, ship_speed_val, dist)
    if transport_output is None:
        return None
    baseline = max(mining_output, transport_output, 1e-9)
    gap_ratio = abs(mining_output - transport_output) / baseline
    if gap_ratio <= max(0.0, float(balance_tolerance)):
        return "BAL"
    return "M" if mining_output < transport_output else "T"


def compute_profit_per_second(
    levels: dict,
    planet_params: dict,
    ore_price: float,
    mining_multiplier_mods: float = 1.0,
    speed_multiplier_mods: float = 1.0,
    cargo_multiplier_mods: float = 1.0,
    value_multiplier_mods: float = 1.0,
) -> Optional[float]:
    try:
        m = int(levels.get("m"))
        s = int(levels.get("s"))
        c = int(levels.get("c"))
    except Exception:
        return None

    mining_output = compute_mining_output(m, mining_multiplier_mods)
    ship_speed_val = compute_ship_speed(s, speed_multiplier_mods)
    cargo_capacity = compute_cargo_capacity(c, cargo_multiplier_mods)
    if mining_output is None or ship_speed_val is None or cargo_capacity is None:
        return None

    dist = planet_params.get("distance", planet_params.get("dist", None))
    transport_output = compute_transport_output(cargo_capacity, ship_speed_val, dist)
    effective = compute_effective_output(mining_output, transport_output)
    if effective is None:
        return None

    if not isinstance(ore_price, (int, float)) or ore_price <= 0:
        return None
    return effective * float(ore_price) * float(value_multiplier_mods)


def compute_upgrade_delta_profit(
    levels: dict,
    levels_after: dict,
    planet_params: dict,
    ore_price: float,
    mining_multiplier_mods: float = 1.0,
    speed_multiplier_mods: float = 1.0,
    cargo_multiplier_mods: float = 1.0,
    value_multiplier_mods: float = 1.0,
) -> Optional[float]:
    before = compute_profit_per_second(
        levels,
        planet_params,
        ore_price,
        mining_multiplier_mods=mining_multiplier_mods,
        speed_multiplier_mods=speed_multiplier_mods,
        cargo_multiplier_mods=cargo_multiplier_mods,
        value_multiplier_mods=value_multiplier_mods,
    )
    after = compute_profit_per_second(
        levels_after,
        planet_params,
        ore_price,
        mining_multiplier_mods=mining_multiplier_mods,
        speed_multiplier_mods=speed_multiplier_mods,
        cargo_multiplier_mods=cargo_multiplier_mods,
        value_multiplier_mods=value_multiplier_mods,
    )
    if before is None or after is None:
        return None
    return after - before


def _planet_cfg(planets_cfg: dict, planet_id: int) -> Optional[dict]:
    planet_cfg = planets_cfg.get(str(planet_id)) if isinstance(planets_cfg, dict) else None
    if not planet_cfg and isinstance(planets_cfg, dict):
        planet_cfg = planets_cfg.get(planet_id)
    return planet_cfg if isinstance(planet_cfg, dict) else None


def _immediate_candidates_for_planet(
    *,
    planet_id: int,
    levels: dict,
    planet_cfg: dict,
    mining_multiplier_mods: float,
    speed_multiplier_mods: float,
    cargo_multiplier_mods: float,
    value_multiplier_mods: float,
    bottleneck_bonus: float,
    balance_tolerance: float,
) -> list[dict]:
    unlock_price = planet_cfg.get("unlock_price")
    if not isinstance(unlock_price, (int, float)) or unlock_price <= 0:
        return []

    planet_params = {
        "distance": planet_cfg.get("distance", planet_cfg.get("dist", None)),
    }
    ore_price = _weighted_ore_price(planet_cfg)
    if ore_price is None:
        return []

    base_profit = compute_profit_per_second(
        levels,
        planet_params,
        ore_price,
        mining_multiplier_mods=mining_multiplier_mods,
        speed_multiplier_mods=speed_multiplier_mods,
        cargo_multiplier_mods=cargo_multiplier_mods,
        value_multiplier_mods=value_multiplier_mods,
    )
    if base_profit is None:
        return []

    bottleneck = classify_bottleneck(
        levels,
        planet_params,
        mining_multiplier_mods=mining_multiplier_mods,
        speed_multiplier_mods=speed_multiplier_mods,
        cargo_multiplier_mods=cargo_multiplier_mods,
        balance_tolerance=balance_tolerance,
    )

    candidates: list[dict] = []
    for stat, key in (("M", "m"), ("S", "s"), ("C", "c")):
        level_now = levels.get(key)
        if not isinstance(level_now, int) or level_now <= 0:
            continue
        levels_after = {"m": levels["m"], "s": levels["s"], "c": levels["c"]}
        levels_after[key] = level_now + 1

        delta = compute_upgrade_delta_profit(
            levels,
            levels_after,
            planet_params,
            ore_price,
            mining_multiplier_mods=mining_multiplier_mods,
            speed_multiplier_mods=speed_multiplier_mods,
            cargo_multiplier_mods=cargo_multiplier_mods,
            value_multiplier_mods=value_multiplier_mods,
        )
        if delta is None or delta <= 0:
            continue

        cost = upgrade_cost(unlock_price, level_now)
        if cost <= 0:
            continue

        roi = delta / cost
        payback_seconds = cost / delta if delta > 0 else None
        alignment_multiplier = 1.0
        if bottleneck == "M" and stat == "M":
            alignment_multiplier += float(bottleneck_bonus)
        elif bottleneck == "T" and stat in {"S", "C"}:
            alignment_multiplier += float(bottleneck_bonus)

        base_score = roi * alignment_multiplier
        candidates.append(
            {
                "planet_id": planet_id,
                "stat": stat,
                "cost": cost,
                "roi": roi,
                "delta": delta,
                "ore_price": ore_price,
                "levels_before": {"m": levels["m"], "s": levels["s"], "c": levels["c"]},
                "levels_after": levels_after,
                "payback_seconds": payback_seconds,
                "bottleneck": bottleneck,
                "alignment_multiplier": alignment_multiplier,
                "base_score": base_score,
            }
        )
    return candidates


def _best_future_score(
    *,
    levels: dict,
    planet_id: int,
    planet_cfg: dict,
    depth: int,
    lookahead_discount: float,
    mining_multiplier_mods: float,
    speed_multiplier_mods: float,
    cargo_multiplier_mods: float,
    value_multiplier_mods: float,
    bottleneck_bonus: float,
    balance_tolerance: float,
    memo: dict,
) -> float:
    if depth <= 0:
        return 0.0
    key = (planet_id, int(levels["m"]), int(levels["s"]), int(levels["c"]), int(depth))
    if key in memo:
        return memo[key]
    best = 0.0
    for candidate in _immediate_candidates_for_planet(
        planet_id=planet_id,
        levels=levels,
        planet_cfg=planet_cfg,
        mining_multiplier_mods=mining_multiplier_mods,
        speed_multiplier_mods=speed_multiplier_mods,
        cargo_multiplier_mods=cargo_multiplier_mods,
        value_multiplier_mods=value_multiplier_mods,
        bottleneck_bonus=bottleneck_bonus,
        balance_tolerance=balance_tolerance,
    ):
        future = _best_future_score(
            levels=candidate["levels_after"],
            planet_id=planet_id,
            planet_cfg=planet_cfg,
            depth=depth - 1,
            lookahead_discount=lookahead_discount,
            mining_multiplier_mods=mining_multiplier_mods,
            speed_multiplier_mods=speed_multiplier_mods,
            cargo_multiplier_mods=cargo_multiplier_mods,
            value_multiplier_mods=value_multiplier_mods,
            bottleneck_bonus=bottleneck_bonus,
            balance_tolerance=balance_tolerance,
            memo=memo,
        )
        score = candidate["base_score"] + (float(lookahead_discount) * future)
        if score > best:
            best = score
    memo[key] = best
    return best


def choose_best_upgrades(
    levels_by_planet: dict,
    planets_cfg: dict,
    top_n: int = 3,
    mining_multiplier_mods: float = 1.0,
    speed_multiplier_mods: float = 1.0,
    cargo_multiplier_mods: float = 1.0,
    value_multiplier_mods: float = 1.0,
    lookahead_depth: int = 2,
    lookahead_discount: float = 0.85,
    bottleneck_bonus: float = 0.20,
    balance_tolerance: float = 0.05,
) -> list[dict]:
    candidates: list[dict] = []
    depth = max(1, int(lookahead_depth))
    memo: dict = {}

    for planet_id, levels in levels_by_planet.items():
        planet_cfg = _planet_cfg(planets_cfg, planet_id)
        if planet_cfg is None:
            continue
        immediate_candidates = _immediate_candidates_for_planet(
            planet_id=planet_id,
            levels=levels,
            planet_cfg=planet_cfg,
            mining_multiplier_mods=mining_multiplier_mods,
            speed_multiplier_mods=speed_multiplier_mods,
            cargo_multiplier_mods=cargo_multiplier_mods,
            value_multiplier_mods=value_multiplier_mods,
            bottleneck_bonus=bottleneck_bonus,
            balance_tolerance=balance_tolerance,
        )
        for candidate in immediate_candidates:
            future_score = _best_future_score(
                levels=candidate["levels_after"],
                planet_id=planet_id,
                planet_cfg=planet_cfg,
                depth=depth - 1,
                lookahead_discount=lookahead_discount,
                mining_multiplier_mods=mining_multiplier_mods,
                speed_multiplier_mods=speed_multiplier_mods,
                cargo_multiplier_mods=cargo_multiplier_mods,
                value_multiplier_mods=value_multiplier_mods,
                bottleneck_bonus=bottleneck_bonus,
                balance_tolerance=balance_tolerance,
                memo=memo,
            )
            candidate["lookahead_score"] = future_score
            candidate["score"] = candidate["base_score"] + (float(lookahead_discount) * future_score)
            candidates.append(candidate)

    stat_order = {"M": 0, "S": 1, "C": 2}
    candidates.sort(
        key=lambda c: (
            -c.get("score", c["roi"]),
            -c["roi"],
            c["planet_id"],
            stat_order.get(c["stat"], 9),
        )
    )
    return candidates[: max(0, int(top_n))]


def choose_upgrade_plan(
    levels_by_planet: dict,
    planets_cfg: dict,
    *,
    available_cash: float | None,
    max_actions: int = 3,
    min_roi: float = 0.0,
    mining_multiplier_mods: float = 1.0,
    speed_multiplier_mods: float = 1.0,
    cargo_multiplier_mods: float = 1.0,
    value_multiplier_mods: float = 1.0,
    lookahead_depth: int = 2,
    lookahead_discount: float = 0.85,
    bottleneck_bonus: float = 0.20,
    balance_tolerance: float = 0.05,
) -> list[dict]:
    simulated_levels: dict[int, dict[str, int]] = {}
    for planet_id, levels in levels_by_planet.items():
        try:
            simulated_levels[int(planet_id)] = {
                "m": int(levels["m"]),
                "s": int(levels["s"]),
                "c": int(levels["c"]),
            }
        except Exception:
            continue

    remaining_cash = None if available_cash is None else float(available_cash)
    min_roi = float(min_roi)
    plan: list[dict] = []
    max_actions = max(0, int(max_actions))

    for step in range(max_actions):
        candidates = choose_best_upgrades(
            simulated_levels,
            planets_cfg,
            top_n=max(10, len(simulated_levels) * 3),
            mining_multiplier_mods=mining_multiplier_mods,
            speed_multiplier_mods=speed_multiplier_mods,
            cargo_multiplier_mods=cargo_multiplier_mods,
            value_multiplier_mods=value_multiplier_mods,
            lookahead_depth=lookahead_depth,
            lookahead_discount=lookahead_discount,
            bottleneck_bonus=bottleneck_bonus,
            balance_tolerance=balance_tolerance,
        )
        if not candidates:
            break

        selected = None
        for candidate in candidates:
            if candidate["roi"] < min_roi:
                continue
            if remaining_cash is not None and candidate["cost"] > remaining_cash:
                continue
            selected = dict(candidate)
            break

        if selected is None:
            break

        selected["plan_step"] = step + 1
        selected["cash_before"] = remaining_cash
        if remaining_cash is not None:
            remaining_cash -= float(selected["cost"])
            selected["cash_after"] = remaining_cash
        plan.append(selected)

        pid = int(selected["planet_id"])
        simulated_levels[pid] = {
            "m": int(selected["levels_after"]["m"]),
            "s": int(selected["levels_after"]["s"]),
            "c": int(selected["levels_after"]["c"]),
        }

    return plan
