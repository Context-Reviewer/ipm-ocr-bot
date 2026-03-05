import time

import analytics
import config
import optimizer
import policy
from config import KEY_DELAY, MENU_DELAY, SCROLL_DELAY
from data_store import PLANETS
from input_utils import tap, reset_ui
from ocr_snap import read_hud_cash, read_planet_levels
from signals import cargo_available, mining_available, speed_available

def mine_rate(level: int) -> float:
    n = max(0, level - 1)
    return 0.25 + 0.1 * n + 0.017 * n * n

def ship_speed(level: int) -> float:
    n = max(0, level - 1)
    return 1 + 0.2 * n + (1 / 75) * n * n

def cargo_cap(level: int) -> float:
    n = max(0, level - 1)
    return 5 + 2 * n + 0.1 * n * n

def get_cycle_seconds(planet_index: int, current_speed_level: int, base_speed_level: int):
    base_cycle = config.PLANET_CYCLE_SECONDS.get(planet_index, config.DEFAULT_CYCLE_SECONDS)
    if base_cycle is None:
        return None
    if not config.USE_SPEED_IN_CYCLE_MODEL:
        return base_cycle
    base_speed = ship_speed(base_speed_level)
    cur_speed = ship_speed(current_speed_level)
    if cur_speed <= 0:
        return None
    return base_cycle * (base_speed / cur_speed)

def compute_fill_ratio(planet_index: int, m_level: int, s_level: int, c_level: int, base_speed_level: int):
    cycle_seconds = get_cycle_seconds(planet_index, s_level, base_speed_level)
    if cycle_seconds is None:
        return None, None, None, None
    prod_per_cycle = mine_rate(m_level) * cycle_seconds
    cap_per_cycle = cargo_cap(c_level)
    if cap_per_cycle <= 0:
        return cycle_seconds, prod_per_cycle, cap_per_cycle, None
    fill_ratio = prod_per_cycle / cap_per_cycle
    return cycle_seconds, prod_per_cycle, cap_per_cycle, fill_ratio

def choose_upgrade_governor(planet_index: int, levels: dict, base_speed_level: int):
    m_level = levels["m"]
    s_level = levels["s"]
    c_level = levels["c"]

    cycle_seconds, prod_per_cycle, cap_per_cycle, fill_ratio = compute_fill_ratio(
        planet_index, m_level, s_level, c_level, base_speed_level
    )
    if fill_ratio is None:
        return None, cycle_seconds, prod_per_cycle, cap_per_cycle, fill_ratio

    fill_low = config.FILL_TARGET - config.FILL_BAND
    fill_high = config.FILL_TARGET + config.FILL_BAND

    if fill_ratio < fill_low:
        return "M", cycle_seconds, prod_per_cycle, cap_per_cycle, fill_ratio

    if fill_ratio > fill_high:
        _, prod_c, cap_c, fill_c = compute_fill_ratio(
            planet_index, m_level, s_level, c_level + 1, base_speed_level
        )
        _, prod_s, cap_s, fill_s = compute_fill_ratio(
            planet_index, m_level, s_level + 1, c_level, base_speed_level
        )
        if fill_c is None and fill_s is None:
            return None, cycle_seconds, prod_per_cycle, cap_per_cycle, fill_ratio
        if fill_s is None:
            return "C", cycle_seconds, prod_per_cycle, cap_per_cycle, fill_ratio
        if fill_c is None:
            return "S", cycle_seconds, prod_per_cycle, cap_per_cycle, fill_ratio

        diff_c = abs(fill_c - config.FILL_TARGET)
        diff_s = abs(fill_s - config.FILL_TARGET)
        if diff_s < diff_c and config.USE_SPEED_IN_CYCLE_MODEL:
            return "S", cycle_seconds, prod_per_cycle, cap_per_cycle, fill_ratio
        return "C", cycle_seconds, prod_per_cycle, cap_per_cycle, fill_ratio

    return None, cycle_seconds, prod_per_cycle, cap_per_cycle, fill_ratio

def get_unlock_price(planet_id: int):
    if isinstance(PLANETS, dict):
        cfg = PLANETS.get(str(planet_id)) or PLANETS.get(planet_id)
        if isinstance(cfg, dict):
            unlock_price = cfg.get("unlock_price")
            if isinstance(unlock_price, (int, float)) and unlock_price > 0:
                return float(unlock_price)
    fallback = config.PLANET_UNLOCK_PRICE.get(planet_id)
    if isinstance(fallback, (int, float)) and fallback > 0:
        return float(fallback)
    return None

_SEED_LOGGED = set()

def planet_module(planets: int = 15):
    reset_ui()
    cash_warned = False
    if PLANETS:
        planet_ids = sorted(int(k) for k in PLANETS.keys() if str(k).isdigit())
    else:
        planet_ids = list(range(1, planets + 1))

    for pid in planet_ids:
        if pid not in config.PLANET_INITIAL_LEVELS:
            config.PLANET_INITIAL_LEVELS[pid] = {"m": 1, "s": 1, "c": 1}
            if pid not in _SEED_LOGGED:
                print(f"[PLANET] seed created p={pid} m=1 s=1 c=1")
                _SEED_LOGGED.add(pid)

    planet_levels = {}
    planet_base_speed = {}
    for idx in planet_ids:
        seed = config.PLANET_INITIAL_LEVELS.get(idx)
        if seed:
            planet_levels[idx] = {"m": seed["m"], "s": seed["s"], "c": seed["c"]}
            planet_base_speed[idx] = seed["s"]
        else:
            seed = {"m": 1, "s": 1, "c": 1}
            config.PLANET_INITIAL_LEVELS[idx] = seed
            planet_levels[idx] = {"m": 1, "s": 1, "c": 1}
            planet_base_speed[idx] = 1
            if idx not in _SEED_LOGGED:
                print(f"[PLANET] seed created p={idx} m=1 s=1 c=1")
                _SEED_LOGGED.add(idx)

    def reset_to_first_planet():
        for _ in range(30):
            tap("-", KEY_DELAY)
        time.sleep(MENU_DELAY)

    def go_to_planet(target_id: int) -> bool:
        reset_to_first_planet()
        for pid in planet_ids:
            if pid == target_id:
                return True
            tap("=", SCROLL_DELAY)
            time.sleep(config.PLANET_SWITCH_DELAY)
        return False

    def read_levels_with_retry(planet_index: int):
        ui_levels = read_planet_levels("PLANET_STATS_PANEL")
        if not ui_levels:
            time.sleep(config.PLANET_OCR_RETRY_DELAY)
            ui_levels = read_planet_levels("PLANET_STATS_PANEL")
        if not ui_levels:
            print(f"[PLANET] p={planet_index} level OCR failed; skipping")
            return None
        return ui_levels

    def maybe_resync_levels_for_current_planet(levels: dict, planet_index: int) -> bool:
        ui_levels = read_levels_with_retry(planet_index)
        if not ui_levels:
            return False
        levels["m"] = ui_levels.mining
        levels["s"] = ui_levels.speed
        levels["c"] = ui_levels.cargo
        print(f"[PLANET] p={planet_index} levels m={levels['m']} s={levels['s']} c={levels['c']}")
        return True

    def run_governor_pass():
        for planet_index in planet_ids:
            levels = planet_levels[planet_index]
            if not maybe_resync_levels_for_current_planet(levels, planet_index):
                tap("=", SCROLL_DELAY)    # next planet
                continue
            base_speed = planet_base_speed[planet_index]
            decision, cycle_seconds, prod_per_cycle, cap_per_cycle, fill_ratio = choose_upgrade_governor(
                planet_index, levels, base_speed
            )

            cyan_m = mining_available()
            cyan_s = speed_available()
            cyan_c = cargo_available()

            if config.USE_SPEED_IN_CYCLE_MODEL:
                print(
                    f"[PLANET] p={planet_index} speed_model=ON base_s={base_speed} cur_s={levels['s']} cycle_eff={cycle_seconds:.2f}"
                )

            if cycle_seconds is None or fill_ratio is None:
                print(
                    f"[PLANET] p={planet_index} cycle=None m={levels['m']} s={levels['s']} c={levels['c']} "
                    f"prod_cycle=None cap=None fill=None decision=None cyan_m={cyan_m} cyan_s={cyan_s} cyan_c={cyan_c}"
                )
            else:
                print(
                    f"[PLANET] p={planet_index} cycle={cycle_seconds:.2f} m={levels['m']} s={levels['s']} c={levels['c']} "
                    f"prod_cycle={prod_per_cycle:.2f} cap={cap_per_cycle:.2f} fill={fill_ratio:.2f} decision={decision} "
                    f"cyan_m={cyan_m} cyan_s={cyan_s} cyan_c={cyan_c}"
                )

            upgrades_m = 0
            upgrades_s = 0
            upgrades_c = 0

            if decision == "M":
                if cyan_m:
                    tap("ctrl+1", KEY_DELAY)
                    upgrades_m += 1
                    levels["m"] += 1
                else:
                    print(f"[PLANET] p={planet_index} decision=M but mining not available (cyan={cyan_m}); skipping fail-closed")
            elif decision == "S":
                if cyan_s:
                    tap("ctrl+2", KEY_DELAY)
                    upgrades_s += 1
                    levels["s"] += 1
                else:
                    print(f"[PLANET] p={planet_index} decision=S but speed not available (cyan={cyan_s}); skipping fail-closed")
            elif decision == "C":
                if cyan_c:
                    tap("ctrl+3", KEY_DELAY)
                    upgrades_c += 1
                    levels["c"] += 1
                else:
                    print(f"[PLANET] p={planet_index} decision=C but cargo not available (cyan={cyan_c}); skipping fail-closed")

            print(f"[PLANET] upgrades mining={upgrades_m} speed={upgrades_s} cargo={upgrades_c}")
            tap("=", SCROLL_DELAY)    # next planet
            time.sleep(config.PLANET_SWITCH_DELAY)

    # Open planet menu
    tap("p", MENU_DELAY)

    # Reset left a bunch so we're near planet 1
    reset_to_first_planet()

    levels_by_planet = {}
    dashboard_rows = []

    # Pass 1: gather levels + analytics logging
    for planet_index in planet_ids:
        levels = planet_levels[planet_index]
        if not maybe_resync_levels_for_current_planet(levels, planet_index):
            tap("=", SCROLL_DELAY)    # next planet
            time.sleep(config.PLANET_SWITCH_DELAY)
            continue
        levels_by_planet[planet_index] = {"m": levels["m"], "s": levels["s"], "c": levels["c"]}
        cycle = config.PLANET_CYCLE_SECONDS.get(planet_index, config.DEFAULT_CYCLE_SECONDS)
        if cycle is not None:
            prod_cycle = analytics.production_per_cycle(levels["m"], cycle)
            cargo = analytics.cargo_cap(levels["c"])
            fill = analytics.fill_ratio(levels["m"], levels["c"], cycle)
            surplus = analytics.surplus_per_cycle(levels["m"], levels["c"], cycle)
            if fill is not None:
                dashboard_rows.append(
                    (planet_index, levels["m"], levels["s"], levels["c"], prod_cycle, cargo, fill)
                )
                print(
                    f"[ANALYTICS] p={planet_index} m={levels['m']} s={levels['s']} c={levels['c']} "
                    f"cycle={cycle:.2f} prod_cycle={prod_cycle:.2f} cargo={cargo:.2f} fill={fill:.2f} surplus={surplus:.2f}"
                )
                impact = analytics.simulate_upgrade(levels, cycle)
                if impact["M"] is not None and impact["C"] is not None and impact["S"] is not None:
                    d_m = impact["M"] - fill
                    d_c = impact["C"] - fill
                    d_s = impact["S"] - fill
                    print(f"[IMPACT] p={planet_index} ΔM={d_m:.4f} ΔC={d_c:.4f} ΔS={d_s:.4f}")
        tap("=", SCROLL_DELAY)    # next planet
        time.sleep(config.PLANET_SWITCH_DELAY)

    if dashboard_rows:
        print("PLANET | M  | S  | C  | PROD/CYCLE | CARGO | FILL")
        print("--------------------------------------------------")
        for p, m, s, c, prod_cycle, cargo, fill in dashboard_rows:
            print(f"{p:<6} | {m:<2} | {s:<2} | {c:<2} | {prod_cycle:>10.2f} | {cargo:>5.2f} | {fill:>4.2f}")

    candidates = optimizer.choose_best_upgrades(levels_by_planet, PLANETS, top_n=3)
    if not candidates:
        print("[OPT] no viable candidates; skipping upgrades")
    else:
        gated = []
        saving_mode = bool(getattr(config, "ECON_SAVING_MODE", False)) and bool(getattr(config, "ECON_ENABLED", True))
        if saving_mode:
            print("[POLICY] saving_mode=ON")
        for c in candidates:
            if policy.allow_upgrade(c, config):
                gated.append(c)
            else:
                print(f"[POLICY] skip p={c['planet_id']} stat={c['stat']} roi={c['roi']:.6g}")

        if not gated:
            if saving_mode:
                print("[POLICY] saving_mode blocked upgrades; skipping")
            else:
                print("[OPT] no candidates after policy gating; skipping upgrades")
        else:
            for c in gated:
                print(
                    f"[OPT] top: p={c['planet_id']} stat={c['stat']} roi={c['roi']:.6g} "
                    f"delta={c['delta']:.4f} cost={c['cost']:.2f}"
                )

            stat_key = {"M": "ctrl+1", "S": "ctrl+2", "C": "ctrl+3"}
            stat_attr = {"M": "mining", "S": "speed", "C": "cargo"}
            stat_afford = {"M": mining_available, "S": speed_available, "C": cargo_available}

            for cand in gated:
                planet_id = cand["planet_id"]
                if not go_to_planet(planet_id):
                    print(f"[OPT] p={planet_id} navigation failed; skipping")
                    continue
                time.sleep(config.PLANET_SWITCH_DELAY)

                before = read_levels_with_retry(planet_id)
                if not before:
                    continue

                key = stat_key.get(cand["stat"])
                attr = stat_attr.get(cand["stat"])
                afford_fn = stat_afford.get(cand["stat"])
                if not key or not attr or not afford_fn:
                    print(f"[OPT] p={planet_id} invalid stat={cand['stat']}; skipping")
                    continue

                cash = read_hud_cash()
                if cash is None:
                    if not cash_warned:
                        print("[PLANET] cash OCR unavailable; using cyan afford check only")
                        cash_warned = True
                else:
                    unlock_price = get_unlock_price(planet_id)
                    if unlock_price is None:
                        print(f"[OPT] p={planet_id} stat={cand['stat']} skip: missing unlock_price")
                        continue
                    current_level = getattr(before, attr)
                    cost = optimizer.upgrade_cost(unlock_price, current_level)
                    if cash < cost:
                        deficit = cost - cash
                        print(f"[OPT] p={planet_id} stat={cand['stat']} skip: cash={cash} < cost={cost:.2f} deficit={deficit:.2f}")
                        continue

                if not afford_fn():
                    if cash is None:
                        print(f"[OPT] p={planet_id} stat={cand['stat']} skip: cyan=False (cash OCR unavailable)")
                    else:
                        print(f"[OPT] p={planet_id} stat={cand['stat']} skip: cyan=False despite cash>=cost cash={cash} cost={cost:.2f}")
                    continue

                tap(key, KEY_DELAY)
                if cash is None:
                    print(f"[OPT] p={planet_id} stat={cand['stat']} click: cash=UNKNOWN")
                else:
                    print(f"[OPT] p={planet_id} stat={cand['stat']} click: cash={cash} cost={cost:.2f} lvl={current_level}")
                time.sleep(config.PLANET_OCR_RETRY_DELAY)

                after = read_levels_with_retry(planet_id)
                if not after:
                    continue

                before_val = getattr(before, attr)
                after_val = getattr(after, attr)
                if after_val == before_val + 1:
                    print(f"[OPT] exec p={planet_id} stat={cand['stat']} ok {before_val}->{after_val}")
                else:
                    print(
                        f"[OPT] exec p={planet_id} stat={cand['stat']} failed "
                        f"{before_val}->{after_val}"
                    )

    # Exit planet menu (your known escape)
    tap("shift+1", MENU_DELAY)
    tap("shift+1", MENU_DELAY)
