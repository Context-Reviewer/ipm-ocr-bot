import optimizer


def test_classify_bottleneck_returns_transport_when_transport_limited():
    levels = {"m": 20, "s": 1, "c": 1}
    planet_params = {"distance": 16}
    assert optimizer.classify_bottleneck(levels, planet_params) == "T"


def test_classify_bottleneck_returns_mining_when_mining_limited():
    levels = {"m": 5, "s": 10, "c": 10}
    planet_params = {"distance": 16}
    assert optimizer.classify_bottleneck(levels, planet_params) == "M"


def test_choose_best_upgrades_uses_lookahead_to_prefer_better_sequence():
    levels_by_planet = {
        5: {"m": 1, "s": 1, "c": 2},
    }
    planets_cfg = {
        "5": {
            "unlock_price": 5000,
            "distance": 16,
            "yields": {"Lead": 50, "Iron": 30, "Copper": 20},
        }
    }
    candidates = optimizer.choose_best_upgrades(
        levels_by_planet,
        planets_cfg,
        top_n=3,
        lookahead_depth=2,
        lookahead_discount=0.85,
        bottleneck_bonus=0.20,
        balance_tolerance=0.05,
    )
    assert candidates
    assert candidates[0]["planet_id"] == 5
    assert candidates[0]["stat"] == "C"
    assert candidates[0]["score"] > candidates[0]["roi"]


def test_choose_upgrade_plan_respects_budget_and_updates_levels():
    levels_by_planet = {
        2: {"m": 17, "s": 11, "c": 10},
        5: {"m": 8, "s": 4, "c": 4},
    }
    planets_cfg = {
        "2": {
            "unlock_price": 200,
            "distance": 12,
            "yields": {"Copper": 80, "Iron": 20},
        },
        "5": {
            "unlock_price": 5000,
            "distance": 16,
            "yields": {"Lead": 50, "Iron": 30, "Copper": 20},
        },
    }
    plan = optimizer.choose_upgrade_plan(
        levels_by_planet,
        planets_cfg,
        available_cash=700,
        max_actions=3,
        min_roi=0.0,
        lookahead_depth=2,
        lookahead_discount=0.85,
        bottleneck_bonus=0.20,
        balance_tolerance=0.05,
    )
    assert plan
    assert len(plan) <= 3
    total_cost = sum(item["cost"] for item in plan)
    assert total_cost <= 700
    assert plan[0]["plan_step"] == 1
    if len(plan) > 1:
        assert plan[1]["cash_before"] <= plan[0]["cash_before"]


def test_choose_upgrade_plan_can_pick_sequence_on_same_planet():
    levels_by_planet = {
        5: {"m": 8, "s": 3, "c": 3},
    }
    planets_cfg = {
        "5": {
            "unlock_price": 5000,
            "distance": 16,
            "yields": {"Lead": 50, "Iron": 30, "Copper": 20},
        },
    }
    plan = optimizer.choose_upgrade_plan(
        levels_by_planet,
        planets_cfg,
        available_cash=2000,
        max_actions=3,
        min_roi=0.0,
        lookahead_depth=2,
        lookahead_discount=0.85,
        bottleneck_bonus=0.20,
        balance_tolerance=0.05,
    )
    assert plan
    assert all(item["planet_id"] == 5 for item in plan[:2])


def test_choose_upgrade_plan_can_extend_beyond_three_actions():
    levels_by_planet = {
        5: {"m": 8, "s": 3, "c": 3},
    }
    planets_cfg = {
        "5": {
            "unlock_price": 5000,
            "distance": 16,
            "yields": {"Lead": 50, "Iron": 30, "Copper": 20},
        },
    }
    plan = optimizer.choose_upgrade_plan(
        levels_by_planet,
        planets_cfg,
        available_cash=5000,
        max_actions=8,
        min_roi=0.0,
        lookahead_depth=2,
        lookahead_discount=0.85,
        bottleneck_bonus=0.20,
        balance_tolerance=0.05,
    )
    assert len(plan) >= 4
    assert len(plan) <= 8


def test_choose_upgrade_plan_can_replan_from_updated_cached_state():
    levels_by_planet = {
        2: {"m": 18, "s": 12, "c": 11},
        5: {"m": 8, "s": 5, "c": 6},
    }
    planets_cfg = {
        "2": {
            "unlock_price": 200,
            "distance": 12,
            "yields": {"Copper": 80, "Iron": 20},
        },
        "5": {
            "unlock_price": 5000,
            "distance": 16,
            "yields": {"Lead": 50, "Iron": 30, "Copper": 20},
        },
    }
    plan = optimizer.choose_upgrade_plan(
        levels_by_planet,
        planets_cfg,
        available_cash=2000,
        max_actions=4,
        min_roi=0.0,
        lookahead_depth=2,
        lookahead_discount=0.85,
        bottleneck_bonus=0.20,
        balance_tolerance=0.05,
    )
    assert len(plan) >= 2

    first = plan[0]
    replanned_levels = {
        pid: dict(levels) for pid, levels in levels_by_planet.items()
    }
    replanned_levels[first["planet_id"]] = dict(first["levels_after"])

    replanned = optimizer.choose_upgrade_plan(
        replanned_levels,
        planets_cfg,
        available_cash=first["cash_after"],
        max_actions=3,
        min_roi=0.0,
        lookahead_depth=2,
        lookahead_discount=0.85,
        bottleneck_bonus=0.20,
        balance_tolerance=0.05,
    )
    assert replanned
    assert replanned[0]["plan_step"] == 1
    assert replanned[0]["cash_before"] == first["cash_after"]
