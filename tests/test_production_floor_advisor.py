from production_floor_advisor import compute_production_floor_advice_from_mapping


def test_production_floor_advisor_computes_recursive_floors_for_hammer_chain():
    advice = compute_production_floor_advice_from_mapping(
        {
            "ores": {"Iron": 80000, "Lead": 30000},
            "bars": {"Iron Bar": 55, "Lead Bar": 26},
            "items": {"Iron Nails": 12},
            "crafter_queue": {"Hammer": 1},
        }
    )

    assert advice["active_production_assignments_detected"] == {
        "smelter": [],
        "crafter": [{"name": "Hammer", "active_count": 1}],
    }
    assert advice["expanded_requirements"] == {
        "Iron": 10000,
        "Iron Bar": 10,
        "Iron Nails": 2,
        "Lead": 5000,
        "Lead Bar": 5,
    }
    assert advice["protected_floors"] == {
        "Iron": 50000,
        "Iron Bar": 50,
        "Iron Nails": 10,
        "Lead": 25000,
        "Lead Bar": 25,
    }
    assert advice["sellable_surplus"] == {
        "Iron": 30000,
        "Iron Bar": 5,
        "Iron Nails": 2,
        "Lead": 5000,
        "Lead Bar": 1,
    }
    assert advice["materials_that_must_not_be_sold"] == []
    assert advice["protected_floor_reasons"]["Iron"][0]["dependency_path"] == "Hammer -> Iron Nails -> Iron Bar -> Iron"


def test_production_floor_advisor_normalizes_bar_aliases_and_ore_aliases():
    advice = compute_production_floor_advice_from_mapping(
        {
            "ores": {"Aluminum": 7000, "Silica": 9000, "Copper": 60000},
            "bars": {"Aluminum Bar": 30, "Silicon Bar": 30, "Copper Bar": 70},
            "items": {"Copper Wire": 15},
            "crafter_queue": {"Circuit": 1},
        }
    )

    assert advice["expanded_requirements"] == {
        "Aluminium Bar": 5,
        "Aluminum": 5000,
        "Copper": 50000,
        "Copper Bar": 50,
        "Copper Wire": 10,
        "Silica": 5000,
        "Silicon Bar": 5,
    }
    assert advice["protected_floors"]["Aluminium Bar"] == 25
    assert advice["current_stock"]["Aluminium Bar"] == 30
    assert advice["sellable_surplus"]["Aluminium Bar"] == 5
    assert advice["protected_floors"]["Silica"] == 25000


def test_production_floor_advisor_reports_missing_recipe_materials_for_unmodeled_alloys():
    advice = compute_production_floor_advice_from_mapping(
        {
            "crafter_queue": {"Collider": 1},
            "items": {"Collider": 0},
        }
    )

    assert advice["expanded_requirements"] == {
        "Inerton Alloy": 500,
        "Quadium Alloy": 100,
    }
    assert advice["protected_floors"] == {
        "Inerton Alloy": 2500,
        "Quadium Alloy": 500,
    }
    assert advice["materials_that_must_not_be_sold"] == ["Inerton Alloy", "Quadium Alloy"]
    assert advice["limitations"]["missing_recipe_materials"] == ["Inerton Alloy", "Quadium Alloy"]


def test_production_floor_advisor_reports_live_bar_and_item_reader_support():
    advice = compute_production_floor_advice_from_mapping({"items": {"Hammer": 0}})

    assert advice["live_reader_support"] == {
        "active_assignments": False,
        "ores": True,
        "bars": True,
        "items": True,
    }
    assert advice["limitations"]["current_inventory_live_readers"] == {
        "ores": True,
        "bars": True,
        "items": True,
    }
