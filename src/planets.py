import time

import numpy as np

import analytics
import config
import ocr
import optimizer
import policy
import runtime_state
from config import KEY_DELAY, MENU_DELAY, SCROLL_DELAY
from data_store import PLANETS
from input_utils import click_client_point, click_screen_point, tap, reset_ui
from ocr_snap import (
    read_hud_cash,
    read_planet_levels,
    read_planet_levels_fast,
    read_planet_title_id_stable,
    read_upgrade_button_cost,
)
from signals import cargo_available, mining_available, sample_rect, speed_available


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
        _, _prod_c, _cap_c, fill_c = compute_fill_ratio(
            planet_index, m_level, s_level, c_level + 1, base_speed_level
        )
        _, _prod_s, _cap_s, fill_s = compute_fill_ratio(
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


def _levels_dict_from_ui(levels_obj):
    try:
        return {
            "m": int(levels_obj.mining),
            "s": int(levels_obj.speed),
            "c": int(levels_obj.cargo),
        }
    except Exception:
        return None


def _levels_signature(levels: dict) -> tuple[int, int, int] | None:
    try:
        return (int(levels["m"]), int(levels["s"]), int(levels["c"]))
    except Exception:
        return None


def _signature_delta(observed: tuple[int, int, int], expected: tuple[int, int, int]) -> int:
    return sum(abs(int(a) - int(b)) for a, b in zip(observed, expected))


def levels_match_expected(ui_levels, expected_levels: dict) -> bool:
    if ui_levels is None or not isinstance(expected_levels, dict):
        return False
    try:
        observed = (int(ui_levels.mining), int(ui_levels.speed), int(ui_levels.cargo))
        expected = (int(expected_levels["m"]), int(expected_levels["s"]), int(expected_levels["c"]))
    except Exception:
        return False
    return observed == expected


def match_planet_sequence(observed_levels: list[dict], known_levels: dict, planet_ids: list[int]) -> int | None:
    observations = [_levels_signature(levels) for levels in observed_levels]
    if not observations or any(sig is None for sig in observations):
        return None
    if not planet_ids:
        return None

    expected = {pid: _levels_signature(levels) for pid, levels in known_levels.items()}
    if any(expected.get(pid) is None for pid in planet_ids):
        return None

    scores: list[tuple[int, int]] = []
    for start_idx, planet_id in enumerate(planet_ids):
        total_delta = 0
        for offset, observed_sig in enumerate(observations):
            expected_pid = planet_ids[(start_idx + offset) % len(planet_ids)]
            total_delta += _signature_delta(observed_sig, expected[expected_pid])
        scores.append((planet_id, total_delta))

    scores.sort(key=lambda item: item[1])
    best_id, best_total = scores[0]
    if best_total > int(getattr(config, "PLANET_DISCOVERY_MAX_TOTAL_DELTA", 18)):
        return None
    if len(scores) > 1:
        next_total = scores[1][1]
        if next_total - best_total < int(getattr(config, "PLANET_DISCOVERY_MIN_MARGIN", 3)):
            return None
    return best_id


def match_unique_planet(levels: dict, known_levels: dict, planet_ids: list[int]) -> int | None:
    observed = _levels_signature(levels)
    if observed is None:
        return None
    matches = []
    for planet_id in planet_ids:
        expected = known_levels.get(planet_id)
        if _levels_signature(expected) == observed:
            matches.append(planet_id)
    return matches[0] if len(matches) == 1 else None


def resolve_planet_id_from_observation(
    observed_title_id: int | None,
    fallback_planet_id: int | None,
    planet_ids: list[int],
) -> int | None:
    valid_ids = set(int(pid) for pid in planet_ids)
    if observed_title_id in valid_ids:
        return int(observed_title_id)
    if fallback_planet_id in valid_ids:
        return int(fallback_planet_id)
    return None


def resolve_scan_planet_id(
    observed_title_id: int | None,
    expected_planet_id: int | None,
    previous_planet_id: int | None,
    seen_planet_ids: set[int],
    planet_ids: list[int],
) -> int | None:
    valid_ids = [int(pid) for pid in planet_ids]
    valid_set = set(valid_ids)

    def _is_forward_step(candidate_id: int, from_id: int | None) -> bool:
        if from_id not in valid_set:
            return True
        try:
            prev_idx = valid_ids.index(int(from_id))
            cand_idx = valid_ids.index(int(candidate_id))
        except Exception:
            return False
        return cand_idx == ((prev_idx + 1) % len(valid_ids))

    if observed_title_id in valid_set and observed_title_id not in seen_planet_ids:
        if observed_title_id == expected_planet_id or _is_forward_step(int(observed_title_id), previous_planet_id):
            return int(observed_title_id)

    if expected_planet_id in valid_set and expected_planet_id not in seen_planet_ids:
        return int(expected_planet_id)

    if observed_title_id in valid_set and observed_title_id not in seen_planet_ids:
        return int(observed_title_id)
    return None


_SEED_LOGGED = set()
_PLANET_NAV_RECT_KEYS = (
    "MINING_LVL",
    "SHIP_LVL",
    "CARGO_LVL",
    "MINING_RATE",
    "SHIP_SPEED",
    "CARGO_CAPACITY",
)


def planet_nav_rect_key(direction: int) -> str | None:
    if direction > 0:
        return "CYCLE_PLANETS_RIGHT"
    if direction < 0:
        return "CYCLE_PLANETS_LEFT"
    return None


def planet_nav_click_point(direction: int):
    rect_key = planet_nav_rect_key(direction)
    if not rect_key:
        return None
    rect = ocr.resolve_client_bbox(rect_key)
    if not isinstance(rect, (tuple, list)) or len(rect) != 4:
        return None
    x, y, w, h = rect
    return (int(x + (w // 2)), int(y + (h // 2)))


def _log_seed_created(pid: int, levels: dict) -> None:
    if not bool(getattr(config, "PLANET_SEED_DEBUG", False)):
        return
    if pid in _SEED_LOGGED:
        return
    print(f"[PLANET] seed created p={pid} m={levels['m']} s={levels['s']} c={levels['c']}")
    _SEED_LOGGED.add(pid)


def should_clear_learned_state(
    *,
    stored_visible_planets: int | None,
    discovered_visible_planets: int | None,
) -> bool:
    try:
        discovered = int(discovered_visible_planets or 0)
    except Exception:
        return False
    try:
        stored = int(stored_visible_planets or 0)
    except Exception:
        stored = 0
    if discovered <= 0:
        return False
    if bool(getattr(config, "PLANET_RESET_CLEAR_ON_SINGLE_VISIBLE", True)) and discovered == 1:
        return True
    if (
        bool(getattr(config, "PLANET_RESET_CLEAR_ON_VISIBLE_DROP", True))
        and stored > 0
        and discovered < stored
    ):
        return True
    return False


def build_planet_level_state(all_planet_ids: list[int], stored_levels: dict | None):
    planet_levels: dict[int, dict[str, int]] = {}
    planet_base_speed: dict[int, int] = {}
    stored_levels = stored_levels if isinstance(stored_levels, dict) else {}

    for idx in all_planet_ids:
        seed = None
        stored = stored_levels.get(str(idx)) or stored_levels.get(idx)
        if isinstance(stored, dict):
            sig = _levels_signature(stored)
            if sig is not None:
                seed = {"m": sig[0], "s": sig[1], "c": sig[2]}
        if seed is None:
            seed = config.PLANET_INITIAL_LEVELS.get(idx)
        if seed:
            planet_levels[idx] = {"m": int(seed["m"]), "s": int(seed["s"]), "c": int(seed["c"])}
            planet_base_speed[idx] = int(seed["s"])
        else:
            seed = {"m": 1, "s": 1, "c": 1}
            config.PLANET_INITIAL_LEVELS[idx] = seed
            planet_levels[idx] = {"m": 1, "s": 1, "c": 1}
            planet_base_speed[idx] = 1
            _log_seed_created(idx, seed)
    return planet_levels, planet_base_speed


def planet_module(planets: int = 15):
    reset_ui()
    cash_warned = False
    stored_state = runtime_state.load_runtime_state()
    stored_levels = stored_state.get("planet_levels") if isinstance(stored_state, dict) else {}
    stored_visible_planets = None
    preferred_entry_point = None
    if isinstance(stored_state, dict):
        try:
            stored_visible_planets = int(stored_state.get("visible_planet_count"))
        except Exception:
            stored_visible_planets = None
        try:
            raw_point = stored_state.get("preferred_planet_entry_point")
            if isinstance(raw_point, (list, tuple)) and len(raw_point) == 2:
                preferred_entry_point = (int(raw_point[0]), int(raw_point[1]))
        except Exception:
            preferred_entry_point = None
    configured_visible_planets = int(getattr(config, "automation_visible_planet_count", config.visible_planet_count)() or 1)
    if stored_visible_planets and stored_visible_planets > 0:
        config.set_visible_planet_count(max(stored_visible_planets, configured_visible_planets))
    nav_state = {"current_planet_id": None}
    if isinstance(stored_state, dict):
        try:
            nav_state["current_planet_id"] = int(stored_state.get("current_planet_id"))
        except Exception:
            nav_state["current_planet_id"] = None

    if PLANETS:
        all_planet_ids = sorted(int(k) for k in PLANETS.keys() if str(k).isdigit())
    else:
        all_planet_ids = list(range(1, planets + 1))
    visible_planets = int(getattr(config, "automation_visible_planet_count", config.visible_planet_count)() or len(all_planet_ids))
    planet_ids = all_planet_ids[: max(1, min(len(all_planet_ids), visible_planets))]
    if not planet_ids:
        return

    planet_id_to_index = {pid: idx for idx, pid in enumerate(planet_ids)}

    for pid in all_planet_ids:
        if pid not in config.PLANET_INITIAL_LEVELS:
            seed = {"m": 1, "s": 1, "c": 1}
            config.PLANET_INITIAL_LEVELS[pid] = seed
            _log_seed_created(pid, seed)

    planet_levels, planet_base_speed = build_planet_level_state(all_planet_ids, stored_levels)

    if nav_state["current_planet_id"] not in planet_id_to_index:
        nav_state["current_planet_id"] = None

    def persist_runtime() -> None:
        runtime_state.save_runtime_state(
            planet_levels=planet_levels,
            current_planet_id=nav_state.get("current_planet_id"),
            visible_planet_count=len(planet_ids),
            preferred_planet_entry_point=preferred_entry_point,
        )

    def planet_menu_is_readable() -> bool:
        ui_levels = read_planet_levels_fast()
        if not ui_levels:
            return False
        return all(
            isinstance(value, int) and value > 0
            for value in (ui_levels.mining, ui_levels.speed, ui_levels.cargo)
        )

    def planet_menu_is_readable_full() -> bool:
        ui_levels = read_planet_levels("PLANET_STATS_PANEL")
        if not ui_levels:
            return False
        return all(
            isinstance(value, int) and value > 0
            for value in (ui_levels.mining, ui_levels.speed, ui_levels.cargo)
        )

    def _rect_has_planet_cyan(rect_key: str) -> bool:
        pixels = sample_rect(rect_key)
        if getattr(pixels, "size", 0) == 0 or getattr(pixels, "ndim", 0) != 3:
            return False
        try:
            red = pixels[:, :, 0]
            green = pixels[:, :, 1]
            blue = pixels[:, :, 2]
        except Exception:
            return False
        mask = (red < 120) & (green > 120) & (blue > 150)
        return int(mask.sum()) >= int(getattr(config, "PLANET_PANEL_MIN_CYAN_PIXELS", 40))

    def planet_panel_is_open() -> bool:
        return _rect_has_planet_cyan("CYCLE_PLANETS_LEFT") or _rect_has_planet_cyan("CYCLE_PLANETS_RIGHT")

    def open_planet_menu() -> int | None:
        nonlocal preferred_entry_point
        entries = list(getattr(config, "get_planet_entry_points", lambda: getattr(config, "PLANET_ENTRY_POINTS", []))())
        if preferred_entry_point is not None:
            entries.sort(key=lambda entry: 0 if tuple(entry.get("point", ())) == preferred_entry_point else 1)
        for entry in entries:
            home_point = getattr(config, "HOME_NAV_CLICK", None)
            if home_point:
                click_client_point(home_point, delay=MENU_DELAY)
                time.sleep(getattr(config, "HOME_NAV_SETTLE_DELAY", config.PLANET_MENU_OPEN_DELAY))
            point = entry.get("point")
            planet_id = entry.get("planet_id")
            if point is None or planet_id not in planet_id_to_index:
                continue
            print(f"[PLANET] opening entry point={point} expect_id={planet_id}")
            verify_mode = str(entry.get("verify", "fast")).lower()
            offsets = list(getattr(config, "PLANET_ENTRY_CLICK_OFFSETS", [(0, 0)]))
            for attempt in range(config.PLANET_MENU_OPEN_RETRIES):
                opened = False
                for dx, dy in offsets:
                    probe_point = (int(point[0]) + int(dx), int(point[1]) + int(dy))
                    if not click_client_point(probe_point, delay=MENU_DELAY):
                        continue
                    time.sleep(getattr(config, "PLANET_ENTRY_SETTLE_DELAY", config.PLANET_MENU_OPEN_DELAY))
                    time.sleep(config.PLANET_MENU_OPEN_DELAY if attempt == 0 else config.PLANET_MENU_OPEN_RETRY_DELAY)
                    if not planet_panel_is_open():
                        continue
                    readable = planet_menu_is_readable()
                    if not readable and verify_mode == "full":
                        readable = planet_menu_is_readable_full()
                    if not readable:
                        continue
                    observed_planet_id = read_planet_title_id_stable()
                    preferred_entry_point = tuple(point)
                    if observed_planet_id in planet_id_to_index:
                        print(
                            f"[PLANET] entry point opened readable panel "
                            f"expect_id={planet_id} observed_id={observed_planet_id}"
                        )
                        return int(observed_planet_id)
                    print(
                        f"[PLANET] entry point opened readable panel "
                        f"expect_id={planet_id} observed_id=UNKNOWN"
                    )
                    return int(planet_id)
                if not opened:
                    print(f"[PLANET] entry point did not visibly open panel expect_id={planet_id} attempt={attempt + 1}")
        print("[PLANET] planet menu did not become readable; skipping")
        return None

    def capture_nav_signature():
        parts = []
        for key in _PLANET_NAV_RECT_KEYS:
            pixels = sample_rect(key)
            if getattr(pixels, "size", 0) == 0:
                continue
            parts.append(pixels.reshape(-1))
        if not parts:
            return np.array([], dtype=np.uint8)
        return np.concatenate(parts)

    def capture_level_signature():
        ui_levels = read_planet_levels("PLANET_STATS_PANEL")
        if not ui_levels:
            time.sleep(config.PLANET_OCR_RETRY_DELAY)
            ui_levels = read_planet_levels("PLANET_STATS_PANEL")
        levels_dict = _levels_dict_from_ui(ui_levels) if ui_levels else None
        return _levels_signature(levels_dict) if levels_dict else None

    def _nav_signatures_match(lhs, rhs) -> bool:
        if getattr(lhs, "size", 0) == 0 or getattr(rhs, "size", 0) == 0:
            return False
        if lhs.size != rhs.size:
            return False
        diff = float(np.mean(np.abs(lhs.astype(np.int16) - rhs.astype(np.int16))))
        return diff <= float(getattr(config, "PLANET_CYCLE_RETURN_DIFF_THRESHOLD", 4.0))

    def wait_for_planet_panel_change(prev_pixels) -> bool:
        if prev_pixels is None or getattr(prev_pixels, "size", 0) == 0:
            time.sleep(config.PLANET_SWITCH_DELAY)
            return True
        for _ in range(config.PLANET_PANEL_CHANGE_MAX_STEPS):
            curr = capture_nav_signature()
            if curr.size == 0:
                time.sleep(config.PLANET_PANEL_CHANGE_SETTLE_DELAY)
                continue
            if curr.size != prev_pixels.size:
                time.sleep(config.PLANET_PANEL_CHANGE_SETTLE_DELAY)
                return True
            diff = float(np.mean(np.abs(curr.astype(np.int16) - prev_pixels.astype(np.int16))))
            if diff >= config.PLANET_PANEL_CHANGE_DIFF_THRESHOLD:
                time.sleep(config.PLANET_PANEL_CHANGE_SETTLE_DELAY)
                return True
            time.sleep(config.PLANET_PANEL_CHANGE_SETTLE_DELAY)
        return False

    def discover_visible_planet_count() -> int | None:
        if not bool(getattr(config, "PLANET_AUTO_DISCOVER_VISIBLE", True)):
            return None
        max_count = len(all_planet_ids)
        if max_count <= 1:
            return max_count
        minimum_visible = int(getattr(config, "automation_visible_planet_count", config.visible_planet_count)() or 1)
        start_pixels = capture_nav_signature()
        start_levels = capture_level_signature()
        start_title_id = read_planet_title_id_stable()
        if getattr(start_pixels, "size", 0) == 0 or start_levels is None:
            return None
        prev_pixels = start_pixels
        min_steps = max(2, int(getattr(config, "PLANET_CYCLE_DISCOVERY_MIN_STEPS", 2)))
        for steps in range(1, max_count + 1):
            if not step_planet(1, update_nav=False):
                break
            curr_pixels = capture_nav_signature()
            curr_levels = capture_level_signature()
            curr_title_id = read_planet_title_id_stable()
            if start_title_id is not None and curr_title_id is not None:
                if steps == 1 and curr_title_id == start_title_id:
                    return 1
                if steps >= min_steps and curr_title_id == start_title_id:
                    if steps >= minimum_visible:
                        return steps
            if steps == 1 and curr_levels is not None and curr_levels == start_levels and _nav_signatures_match(curr_pixels, start_pixels):
                return 1
            if (
                steps >= min_steps
                and curr_levels is not None
                and curr_levels == start_levels
                and _nav_signatures_match(curr_pixels, start_pixels)
            ):
                if steps >= minimum_visible:
                    return steps
            prev_pixels = curr_pixels
        if stored_visible_planets and stored_visible_planets > 0:
            return int(stored_visible_planets)
        return None

    def step_planet(direction: int, *, update_nav: bool = True) -> bool:
        if direction not in (-1, 1):
            return False
        prev_levels = capture_level_signature()
        prev_title_id = read_planet_title_id_stable()
        prev_pixels = capture_nav_signature()
        nav_point = planet_nav_click_point(direction)
        if nav_point is None:
            print(f"[PLANET] navigation click unavailable direction={direction}")
            return False

        after_title_id = None
        changed = False
        offsets = list(getattr(config, "PLANET_NAV_CLICK_OFFSETS", [(0, 0)]))
        verify_rechecks = max(0, int(getattr(config, "PLANET_NAV_VERIFY_RECHECKS", 2)))
        verify_delay = max(0.0, float(getattr(config, "PLANET_NAV_VERIFY_DELAY", 0.10)))
        for dx, dy in offsets:
            probe_point = (int(nav_point[0]) + int(dx), int(nav_point[1]) + int(dy))
            if not click_client_point(probe_point, delay=SCROLL_DELAY):
                continue
            changed = wait_for_planet_panel_change(prev_pixels)
            for _ in range(verify_rechecks + 1):
                if not changed and prev_levels is not None:
                    after_levels = capture_level_signature()
                    changed = after_levels is not None and after_levels != prev_levels
                after_title_id = read_planet_title_id_stable()
                if not changed and prev_title_id is not None:
                    changed = after_title_id is not None and after_title_id != prev_title_id
                if changed:
                    break
                if verify_delay > 0:
                    time.sleep(verify_delay)
            if changed:
                break

        time.sleep(config.PLANET_SWITCH_DELAY)
        if not changed:
            print(f"[PLANET] navigation step direction={direction} did not visibly change panel")
            return False
        if update_nav and nav_state["current_planet_id"] in planet_id_to_index:
            if after_title_id in planet_id_to_index:
                nav_state["current_planet_id"] = int(after_title_id)
            else:
                current_idx = planet_id_to_index[nav_state["current_planet_id"]]
                nav_state["current_planet_id"] = planet_ids[(current_idx + direction) % len(planet_ids)]
        return True

    def read_levels_with_retry(planet_index: int):
        ui_levels = read_planet_levels("PLANET_STATS_PANEL")
        if not ui_levels:
            time.sleep(config.PLANET_OCR_RETRY_DELAY)
            ui_levels = read_planet_levels("PLANET_STATS_PANEL")
        if not ui_levels:
            print(f"[PLANET] p={planet_index} level OCR failed; skipping")
            return None
        return ui_levels

    def maybe_resync_levels_for_current_planet(expected_planet_id: int) -> int | None:
        ui_levels = read_levels_with_retry(expected_planet_id)
        if not ui_levels:
            return None
        observed_levels = {
            "m": int(ui_levels.mining),
            "s": int(ui_levels.speed),
            "c": int(ui_levels.cargo),
        }
        raw_observed_planet_id = read_planet_title_id_stable()
        observed_planet_id = resolve_scan_planet_id(
            raw_observed_planet_id,
            expected_planet_id,
            previous_scanned_planet_id,
            scanned_planet_ids,
            planet_ids,
        )
        if observed_planet_id is None:
            observed_planet_id = resolve_planet_id_from_observation(
                raw_observed_planet_id,
                expected_planet_id,
                planet_ids,
            )
        if observed_planet_id is None:
            return None
        if scanned_planet_ids:
            unique_seen_match = match_unique_planet(
                observed_levels,
                {pid: planet_levels[pid] for pid in scanned_planet_ids},
                list(scanned_planet_ids),
            )
            if (
                unique_seen_match is not None
                and unique_seen_match != observed_planet_id
                and observed_planet_id == expected_planet_id
            ):
                print(
                    f"[PLANET] scan rejected expected p={expected_planet_id}; "
                    f"levels uniquely match seen p={unique_seen_match}"
                )
                return None
        if raw_observed_planet_id in planet_id_to_index and raw_observed_planet_id != observed_planet_id:
            print(
                f"[PLANET] scan rejected title p={raw_observed_planet_id}; "
                f"using p={observed_planet_id}"
            )
        if observed_planet_id in scanned_planet_ids:
            print(f"[PLANET] scan duplicate p={observed_planet_id}; skipping")
            return None
        if observed_planet_id != expected_planet_id:
            print(
                f"[PLANET] scan relabeled expected p={expected_planet_id} "
                f"observed p={observed_planet_id}"
            )
        levels = planet_levels[observed_planet_id]
        levels["m"] = observed_levels["m"]
        levels["s"] = observed_levels["s"]
        levels["c"] = observed_levels["c"]
        print(f"[PLANET] p={observed_planet_id} levels m={levels['m']} s={levels['s']} c={levels['c']}")
        nav_state["current_planet_id"] = observed_planet_id
        scanned_planet_ids.add(observed_planet_id)
        return observed_planet_id

    def align_current_panel_to_expected(planet_id: int, expected_levels: dict):
        before = read_levels_with_retry(planet_id)
        if not before:
            return None
        observed_title_id = read_planet_title_id_stable()
        if observed_title_id in planet_id_to_index and observed_title_id != planet_id:
            nav_state["current_planet_id"] = int(observed_title_id)
            print(
                f"[OPT] p={planet_id} title mismatch: observed p={observed_title_id}; "
                "retrying navigation"
            )
            if go_to_planet(planet_id):
                before = read_levels_with_retry(planet_id)
                if not before:
                    return None
                observed_title_id = read_planet_title_id_stable()
        if levels_match_expected(before, expected_levels):
            return before

        print(
            f"[OPT] p={planet_id} panel mismatch: "
            f"expected m={expected_levels['m']} s={expected_levels['s']} c={expected_levels['c']} "
            f"got m={before.mining} s={before.speed} c={before.cargo}"
        )

        original_nav_id = nav_state.get("current_planet_id")

        if step_planet(1):
            shifted = read_levels_with_retry(planet_id)
            if shifted and levels_match_expected(shifted, expected_levels):
                print(f"[OPT] p={planet_id} corrected panel drift by +1 step")
                return shifted
            step_planet(-1)

        if step_planet(-1):
            shifted = read_levels_with_retry(planet_id)
            if shifted and levels_match_expected(shifted, expected_levels):
                print(f"[OPT] p={planet_id} corrected panel drift by -1 step")
                return shifted

        if original_nav_id in planet_id_to_index:
            go_to_planet(original_nav_id)
        return None

    def discover_current_planet_id() -> int | None:
        observed = []
        min_samples = min(len(planet_ids), max(1, int(getattr(config, "PLANET_DISCOVERY_SAMPLES", 3))))
        max_samples = len(planet_ids)
        for offset in range(max_samples):
            ui_levels = read_levels_with_retry(-1)
            if not ui_levels:
                return None
            levels_dict = _levels_dict_from_ui(ui_levels)
            if levels_dict is None:
                return None
            observed.append(levels_dict)
            if offset == 0:
                unique_now = match_unique_planet(levels_dict, planet_levels, planet_ids)
                if unique_now is not None:
                    print(f"[PLANET] discovered current planet id={unique_now} via exact level match")
                    return unique_now
            sample_count = len(observed)
            if sample_count >= min_samples:
                start_id = match_planet_sequence(observed, planet_levels, planet_ids)
                if start_id is not None:
                    start_idx = planet_id_to_index[start_id]
                    current_id = planet_ids[(start_idx + sample_count - 1) % len(planet_ids)]
                    print(f"[PLANET] discovered current planet id={current_id} via {sample_count}-planet sequence")
                    return current_id
            if offset < max_samples - 1 and not step_planet(1, update_nav=False):
                return None
        print("[PLANET] current planet discovery failed; aborting module fail-closed")
        return None

    def ensure_current_planet_known() -> bool:
        current_id = nav_state.get("current_planet_id")
        if current_id in planet_id_to_index:
            return True
        nav_state["current_planet_id"] = discover_current_planet_id()
        persist_runtime()
        return nav_state["current_planet_id"] in planet_id_to_index

    def go_to_planet(target_id: int) -> bool:
        if target_id not in planet_id_to_index:
            return False
        if not ensure_current_planet_known():
            return False
        observed_current_id = read_planet_title_id_stable()
        resolved_current_id = resolve_planet_id_from_observation(
            observed_current_id,
            nav_state.get("current_planet_id"),
            planet_ids,
        )
        if resolved_current_id in planet_id_to_index:
            nav_state["current_planet_id"] = int(resolved_current_id)
        current_id = nav_state["current_planet_id"]
        if current_id == target_id:
            return True
        current_idx = planet_id_to_index[current_id]
        target_idx = planet_id_to_index[target_id]
        forward = (target_idx - current_idx) % len(planet_ids)
        backward = (current_idx - target_idx) % len(planet_ids)
        prefer_forward = bool(getattr(config, "PLANET_PREFER_FORWARD_NAV", True))
        candidate_routes: list[tuple[int, int]] = []
        if prefer_forward:
            candidate_routes.append((1, forward))
            if backward != forward:
                candidate_routes.append((-1, backward))
        else:
            preferred_direction = 1 if forward <= backward else -1
            preferred_steps = forward if preferred_direction > 0 else backward
            fallback_direction = -preferred_direction
            fallback_steps = backward if fallback_direction < 0 else forward
            candidate_routes.append((preferred_direction, preferred_steps))
            if fallback_steps != preferred_steps or fallback_direction != preferred_direction:
                candidate_routes.append((fallback_direction, fallback_steps))

        original_current_id = current_id
        for route_index, (direction, steps) in enumerate(candidate_routes):
            nav_state["current_planet_id"] = original_current_id
            ok = True
            for _ in range(steps):
                if not step_planet(direction):
                    ok = False
                    break
            if ok:
                observed_target_id = read_planet_title_id_stable()
                resolved_target_id = resolve_planet_id_from_observation(
                    observed_target_id,
                    nav_state.get("current_planet_id"),
                    planet_ids,
                )
                if resolved_target_id in planet_id_to_index:
                    nav_state["current_planet_id"] = int(resolved_target_id)
                if nav_state["current_planet_id"] == target_id:
                    persist_runtime()
                    return True
                print(
                    f"[PLANET] route direction={direction} steps={steps} landed on "
                    f"p={nav_state['current_planet_id']} expected p={target_id}"
                )
            if route_index < len(candidate_routes) - 1:
                nav_state["current_planet_id"] = original_current_id
                print(
                    f"[PLANET] route direction={direction} steps={steps} failed for target={target_id}; "
                    "trying alternate route"
                )
        persist_runtime()
        return nav_state["current_planet_id"] == target_id

    def step_to_next_planet() -> bool:
        if len(planet_ids) <= 1:
            return True
        return step_planet(1)

    opened_planet_id = open_planet_menu()
    if opened_planet_id is None:
        return
    learned_visible_planets = discover_visible_planet_count()
    if learned_visible_planets:
        minimum_visible_planets = int(getattr(config, "automation_visible_planet_count", config.visible_planet_count)() or 1)
        if learned_visible_planets < minimum_visible_planets:
            print(
                f"[PLANET] discovered visible planet count={learned_visible_planets} "
                f"below configured minimum={minimum_visible_planets}; keeping minimum"
            )
        visible_planets = max(1, min(len(all_planet_ids), max(int(learned_visible_planets), minimum_visible_planets)))
        planet_ids = all_planet_ids[:visible_planets]
        planet_id_to_index = {pid: idx for idx, pid in enumerate(planet_ids)}
        if should_clear_learned_state(
            stored_visible_planets=stored_visible_planets,
            discovered_visible_planets=visible_planets,
        ):
            print(
                f"[PLANET] reset detected: visible_planets {stored_visible_planets or 'UNKNOWN'}->{visible_planets}; "
                "clearing learned runtime state"
            )
            runtime_state.clear_runtime_state()
            stored_levels = {}
            nav_state["current_planet_id"] = None
            config.clear_visible_planet_count_override()
            planet_levels, planet_base_speed = build_planet_level_state(all_planet_ids, stored_levels)
        config.set_visible_planet_count(visible_planets)
        if stored_visible_planets != visible_planets:
            print(f"[PLANET] discovered visible planet count={visible_planets}")
        print(f"[PLANET] context updated visible_planets={visible_planets}")
    nav_state["current_planet_id"] = opened_planet_id
    persist_runtime()

    levels_by_planet = {}
    dashboard_rows = []
    scanned_planet_ids: set[int] = set()
    previous_scanned_planet_id: int | None = None
    start_planet_id = nav_state.get("current_planet_id")
    if start_planet_id not in planet_id_to_index:
        tap("shift+1", MENU_DELAY)
        tap("shift+1", MENU_DELAY)
        return
    start_idx = planet_id_to_index[start_planet_id]
    scan_order = planet_ids[start_idx:] + planet_ids[:start_idx]

    for idx, planet_index in enumerate(scan_order):
        observed_planet_id = maybe_resync_levels_for_current_planet(planet_index)
        if observed_planet_id is None:
            if idx < len(scan_order) - 1 and not step_to_next_planet():
                break
            continue
        previous_scanned_planet_id = observed_planet_id
        levels = planet_levels[observed_planet_id]
        levels_by_planet[observed_planet_id] = {"m": levels["m"], "s": levels["s"], "c": levels["c"]}
        cycle = config.PLANET_CYCLE_SECONDS.get(observed_planet_id, config.DEFAULT_CYCLE_SECONDS)
        if cycle is not None:
            prod_cycle = analytics.production_per_cycle(levels["m"], cycle)
            cargo = analytics.cargo_cap(levels["c"])
            fill = analytics.fill_ratio(levels["m"], levels["c"], cycle)
            surplus = analytics.surplus_per_cycle(levels["m"], levels["c"], cycle)
            if fill is not None:
                dashboard_rows.append(
                    (observed_planet_id, levels["m"], levels["s"], levels["c"], prod_cycle, cargo, fill)
                )
                print(
                    f"[ANALYTICS] p={observed_planet_id} m={levels['m']} s={levels['s']} c={levels['c']} "
                    f"cycle={cycle:.2f} prod_cycle={prod_cycle:.2f} cargo={cargo:.2f} fill={fill:.2f} surplus={surplus:.2f}"
                )
                impact = analytics.simulate_upgrade(levels, cycle)
                if impact["M"] is not None and impact["C"] is not None and impact["S"] is not None:
                    d_m = impact["M"] - fill
                    d_c = impact["C"] - fill
                    d_s = impact["S"] - fill
                    print(f"[IMPACT] p={observed_planet_id} dM={d_m:.4f} dC={d_c:.4f} dS={d_s:.4f}")
        if idx < len(scan_order) - 1 and not step_to_next_planet():
            break

    if dashboard_rows:
        print("PLANET | M  | S  | C  | PROD/CYCLE | CARGO | FILL")
        print("--------------------------------------------------")
        for p, m, s, c, prod_cycle, cargo, fill in dashboard_rows:
            print(f"{p:<6} | {m:<2} | {s:<2} | {c:<2} | {prod_cycle:>10.2f} | {cargo:>5.2f} | {fill:>4.2f}")

    def choose_candidates(available_cash, remaining_actions: int):
        common_kwargs = dict(
            mining_multiplier_mods=float(getattr(config, "OPT_MINING_MULT", 1.0)),
            speed_multiplier_mods=float(getattr(config, "OPT_SPEED_MULT", 1.0)),
            cargo_multiplier_mods=float(getattr(config, "OPT_CARGO_MULT", 1.0)),
            value_multiplier_mods=float(getattr(config, "OPT_VALUE_MULT", 1.0)),
            lookahead_depth=int(getattr(config, "OPT_LOOKAHEAD_DEPTH", 2)),
            lookahead_discount=float(getattr(config, "OPT_LOOKAHEAD_DISCOUNT", 0.85)),
            bottleneck_bonus=float(getattr(config, "OPT_BOTTLENECK_BONUS", 0.20)),
            balance_tolerance=float(getattr(config, "OPT_BOTTLENECK_BALANCE_TOLERANCE", 0.05)),
        )
        if bool(getattr(config, "OPT_PLAN_WITH_BUDGET", True)):
            return optimizer.choose_upgrade_plan(
                levels_by_planet,
                PLANETS,
                available_cash=available_cash,
                max_actions=max(1, int(remaining_actions)),
                min_roi=float(getattr(config, "MIN_ROI_TO_SPEND", 0.0)),
                **common_kwargs,
            )
        return optimizer.choose_best_upgrades(
            levels_by_planet,
            PLANETS,
            top_n=3,
            **common_kwargs,
        )

    stat_key = {"M": "ctrl+1", "S": "ctrl+2", "C": "ctrl+3"}
    stat_attr = {"M": "mining", "S": "speed", "C": "cargo"}
    stat_afford = {"M": mining_available, "S": speed_available, "C": cargo_available}
    level_key = {"M": "m", "S": "s", "C": "c"}

    max_actions = max(0, int(getattr(config, "MAX_UPGRADES_PER_PLANET_TASK", 3)))
    saving_mode = bool(getattr(config, "ECON_SAVING_MODE", False)) and bool(getattr(config, "ECON_ENABLED", True))
    if saving_mode:
        print("[POLICY] saving_mode=ON")

    executed_actions = 0
    while executed_actions < max_actions:
        planning_cash = read_hud_cash()
        candidates = choose_candidates(planning_cash, max_actions - executed_actions)
        if not candidates:
            if executed_actions == 0:
                print("[OPT] no viable candidates; skipping upgrades")
            break

        gated = []
        for candidate in candidates:
            if policy.allow_upgrade(candidate, config):
                gated.append(candidate)
            else:
                print(f"[POLICY] skip p={candidate['planet_id']} stat={candidate['stat']} roi={candidate['roi']:.6g}")

        if not gated:
            if executed_actions == 0:
                if saving_mode:
                    print("[POLICY] saving_mode blocked upgrades; skipping")
                else:
                    print("[OPT] no candidates after policy gating; skipping upgrades")
            break

        preview_candidates = gated[: min(len(gated), max(1, 3 if executed_actions == 0 else 1))]
        for candidate in preview_candidates:
            print(
                f"[OPT] top: step={candidate.get('plan_step', '?')} p={candidate['planet_id']} stat={candidate['stat']} "
                f"score={candidate.get('score', candidate['roi']):.6g} roi={candidate['roi']:.6g} "
                f"delta={candidate['delta']:.4f} cost={candidate['cost']:.2f} "
                f"bottleneck={candidate.get('bottleneck')} "
                f"cash_before={candidate.get('cash_before')}"
            )

        executed_this_cycle = False
        for cand in gated:
            planet_id = cand["planet_id"]
            if not go_to_planet(planet_id):
                print(f"[OPT] p={planet_id} navigation failed; skipping")
                continue

            expected_levels = planet_levels.get(planet_id) or levels_by_planet.get(planet_id)
            if expected_levels:
                before = align_current_panel_to_expected(planet_id, expected_levels)
            else:
                before = read_levels_with_retry(planet_id)
            if not before:
                print(f"[OPT] p={planet_id} unable to verify target panel; skipping")
                continue

            key = stat_key.get(cand["stat"])
            attr = stat_attr.get(cand["stat"])
            afford_fn = stat_afford.get(cand["stat"])
            if not key or not attr or not afford_fn:
                print(f"[OPT] p={planet_id} invalid stat={cand['stat']}; skipping")
                continue

            current_level = getattr(before, attr)
            cash = read_hud_cash()
            ui_cost = read_upgrade_button_cost(cand["stat"])
            if cash is None and not cash_warned:
                print("[PLANET] cash OCR unavailable; using fallback afford checks")
                cash_warned = True

            if cash is not None and ui_cost is not None:
                if cash < ui_cost:
                    deficit = ui_cost - cash
                    print(
                        f"[OPT] p={planet_id} stat={cand['stat']} skip: "
                        f"cash={cash} < ui_cost={ui_cost} deficit={deficit}"
                    )
                    continue
            else:
                unlock_price = get_unlock_price(planet_id)
                if unlock_price is None:
                    print(f"[OPT] p={planet_id} stat={cand['stat']} skip: missing unlock_price")
                    continue
                cost = optimizer.upgrade_cost(unlock_price, current_level)
                if cash is not None and cash < cost:
                    deficit = cost - cash
                    print(
                        f"[OPT] p={planet_id} stat={cand['stat']} skip: "
                        f"cash={cash} < est_cost={cost:.2f} deficit={deficit:.2f}"
                    )
                    continue
                if not afford_fn():
                    if cash is None:
                        print(f"[OPT] p={planet_id} stat={cand['stat']} skip: cyan=False (cash OCR unavailable)")
                    else:
                        print(
                            f"[OPT] p={planet_id} stat={cand['stat']} skip: "
                            f"cyan=False and ui_cost unreadable cash={cash} est_cost={cost:.2f}"
                        )
                    continue

            tap(key, KEY_DELAY)
            if cash is None:
                print(f"[OPT] p={planet_id} stat={cand['stat']} click: cash=UNKNOWN")
            elif ui_cost is not None:
                print(
                    f"[OPT] p={planet_id} stat={cand['stat']} click: "
                    f"cash={cash} ui_cost={ui_cost} lvl={current_level}"
                )
            else:
                print(
                    f"[OPT] p={planet_id} stat={cand['stat']} click: "
                    f"cash={cash} est_cost={cost:.2f} lvl={current_level}"
                )
            time.sleep(config.PLANET_OCR_RETRY_DELAY)

            after = read_levels_with_retry(planet_id)
            if not after:
                continue

            before_val = getattr(before, attr)
            after_val = getattr(after, attr)
            if after_val == before_val + 1:
                print(f"[OPT] exec p={planet_id} stat={cand['stat']} ok {before_val}->{after_val}")
                level_name = level_key.get(cand["stat"])
                if level_name:
                    planet_levels[planet_id][level_name] = after_val
                    if planet_id in levels_by_planet:
                        levels_by_planet[planet_id][level_name] = after_val
                executed_actions += 1
                executed_this_cycle = True
                persist_runtime()
                break

            print(
                f"[OPT] exec p={planet_id} stat={cand['stat']} failed "
                f"{before_val}->{after_val}"
            )

        if not executed_this_cycle:
            break

    persist_runtime()

    tap("shift+1", MENU_DELAY)
    tap("shift+1", MENU_DELAY)
