def mining_rate(level: int) -> float:
    n = max(0, level - 1)
    return 0.25 + 0.1 * n + 0.017 * n * n

def ship_speed(level: int) -> float:
    n = max(0, level - 1)
    return 1 + 0.2 * n + (1 / 75) * n * n

def cargo_cap(level: int) -> float:
    n = max(0, level - 1)
    return 5 + 2 * n + 0.1 * n * n

def production_per_cycle(level_m: int, cycle_seconds: float) -> float:
    return mining_rate(level_m) * cycle_seconds

def fill_ratio(level_m: int, level_c: int, cycle_seconds: float):
    cap = cargo_cap(level_c)
    if cap <= 0:
        return None
    return production_per_cycle(level_m, cycle_seconds) / cap

def surplus_per_cycle(level_m: int, level_c: int, cycle_seconds: float):
    return production_per_cycle(level_m, cycle_seconds) - cargo_cap(level_c)

def mining_required_for_full(level_c: int, cycle_seconds: float):
    if cycle_seconds <= 0:
        return None
    return cargo_cap(level_c) / cycle_seconds

def simulate_upgrade(levels: dict, cycle_seconds: float):
    m = levels["m"]
    s = levels["s"]
    c = levels["c"]
    return {
        "M": fill_ratio(m + 1, c, cycle_seconds),
        "C": fill_ratio(m, c + 1, cycle_seconds),
        "S": fill_ratio(m, c, cycle_seconds),
    }
