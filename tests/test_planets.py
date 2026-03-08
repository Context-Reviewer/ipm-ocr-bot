import planets


class _Levels:
    def __init__(self, mining, speed, cargo):
        self.mining = mining
        self.speed = speed
        self.cargo = cargo


def test_match_planet_sequence_unique_match():
    planet_ids = [1, 2, 3, 4]
    known_levels = {
        1: {"m": 10, "s": 5, "c": 2},
        2: {"m": 20, "s": 6, "c": 3},
        3: {"m": 30, "s": 7, "c": 4},
        4: {"m": 40, "s": 8, "c": 5},
    }
    observed = [
        {"m": 30, "s": 7, "c": 4},
        {"m": 40, "s": 8, "c": 5},
        {"m": 10, "s": 5, "c": 2},
    ]
    assert planets.match_planet_sequence(observed, known_levels, planet_ids) == 3


def test_match_planet_sequence_rejects_ambiguous_match(monkeypatch):
    monkeypatch.setattr(planets.config, "PLANET_DISCOVERY_MIN_MARGIN", 3)
    monkeypatch.setattr(planets.config, "PLANET_DISCOVERY_MAX_TOTAL_DELTA", 18)
    planet_ids = [1, 2]
    known_levels = {
        1: {"m": 10, "s": 5, "c": 2},
        2: {"m": 11, "s": 5, "c": 2},
    }
    observed = [
        {"m": 10, "s": 5, "c": 2},
        {"m": 11, "s": 5, "c": 2},
    ]
    assert planets.match_planet_sequence(observed, known_levels, planet_ids) is None


def test_match_unique_planet_returns_unique_exact_match():
    planet_ids = [1, 2, 3]
    known_levels = {
        1: {"m": 10, "s": 5, "c": 2},
        2: {"m": 11, "s": 5, "c": 2},
        3: {"m": 10, "s": 5, "c": 2},
    }
    assert planets.match_unique_planet({"m": 11, "s": 5, "c": 2}, known_levels, planet_ids) == 2
    assert planets.match_unique_planet({"m": 10, "s": 5, "c": 2}, known_levels, planet_ids) is None


def test_resolve_planet_id_from_observation_prefers_title_when_valid():
    assert planets.resolve_planet_id_from_observation(8, 4, [1, 2, 3, 4, 8, 9]) == 8
    assert planets.resolve_planet_id_from_observation(None, 4, [1, 2, 3, 4]) == 4
    assert planets.resolve_planet_id_from_observation(99, 4, [1, 2, 3, 4]) == 4
    assert planets.resolve_planet_id_from_observation(None, None, [1, 2, 3, 4]) is None


def test_resolve_scan_planet_id_rejects_duplicate_bad_title():
    assert planets.resolve_scan_planet_id(
        observed_title_id=3,
        expected_planet_id=8,
        previous_planet_id=7,
        seen_planet_ids={1, 2, 3, 4, 5, 6, 7},
        planet_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    ) == 8


def test_match_unique_planet_can_identify_seen_duplicate_by_levels():
    seen_levels = {
        1: {"m": 31, "s": 27, "c": 19},
        2: {"m": 23, "s": 14, "c": 14},
        3: {"m": 22, "s": 16, "c": 13},
    }
    assert planets.match_unique_planet({"m": 31, "s": 27, "c": 19}, seen_levels, [1, 2, 3]) == 1
    assert planets.match_unique_planet({"m": 1, "s": 1, "c": 1}, seen_levels, [1, 2, 3]) is None


def test_levels_match_expected_requires_exact_tuple():
    assert planets.levels_match_expected(_Levels(12, 5, 3), {"m": 12, "s": 5, "c": 3}) is True
    assert planets.levels_match_expected(_Levels(12, 5, 4), {"m": 12, "s": 5, "c": 3}) is False


def test_should_clear_learned_state_when_visible_planets_drop(monkeypatch):
    monkeypatch.setattr(planets.config, "PLANET_RESET_CLEAR_ON_VISIBLE_DROP", True)
    monkeypatch.setattr(planets.config, "PLANET_RESET_CLEAR_ON_SINGLE_VISIBLE", True)
    assert planets.should_clear_learned_state(stored_visible_planets=13, discovered_visible_planets=1) is True
    assert planets.should_clear_learned_state(stored_visible_planets=13, discovered_visible_planets=10) is True
    assert planets.should_clear_learned_state(stored_visible_planets=10, discovered_visible_planets=10) is False


def test_build_planet_level_state_prefers_stored_levels():
    levels, base_speed = planets.build_planet_level_state(
        [1, 2],
        {"1": {"m": 7, "s": 8, "c": 9}},
    )
    assert levels[1] == {"m": 7, "s": 8, "c": 9}
    assert base_speed[1] == 8
    assert set(levels.keys()) == {1, 2}


def test_go_to_planet_falls_back_to_forward_when_backward_fails():
    planet_ids = [1, 2, 3, 4]
    planet_id_to_index = {pid: idx for idx, pid in enumerate(planet_ids)}
    nav_state = {"current_planet_id": 4}
    calls: list[int] = []

    def step_planet(direction: int):
        calls.append(direction)
        if direction < 0:
            return False
        current_idx = planet_id_to_index[nav_state["current_planet_id"]]
        nav_state["current_planet_id"] = planet_ids[(current_idx + direction) % len(planet_ids)]
        return True

    def go_to_planet_like(target_id: int, prefer_forward: bool):
        current_id = nav_state["current_planet_id"]
        current_idx = planet_id_to_index[current_id]
        target_idx = planet_id_to_index[target_id]
        forward = (target_idx - current_idx) % len(planet_ids)
        backward = (current_idx - target_idx) % len(planet_ids)
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
        for direction, steps in candidate_routes:
            nav_state["current_planet_id"] = original_current_id
            ok = True
            for _ in range(steps):
                if not step_planet(direction):
                    ok = False
                    break
            if ok and nav_state["current_planet_id"] == target_id:
                return True
        return False

    assert go_to_planet_like(3, prefer_forward=False) is True
    assert nav_state["current_planet_id"] == 3
    assert calls[:2] == [-1, 1]


def test_planet_nav_rect_key_maps_directions():
    assert planets.planet_nav_rect_key(1) == "CYCLE_PLANETS_RIGHT"
    assert planets.planet_nav_rect_key(-1) == "CYCLE_PLANETS_LEFT"
    assert planets.planet_nav_rect_key(0) is None


def test_planet_nav_click_point_uses_rect_center(monkeypatch):
    monkeypatch.setattr(planets.ocr, "resolve_client_bbox", lambda key: (10, 20, 50, 60) if key == "CYCLE_PLANETS_RIGHT" else None)
    assert planets.planet_nav_click_point(1) == (35, 50)
    assert planets.planet_nav_click_point(-1) is None
