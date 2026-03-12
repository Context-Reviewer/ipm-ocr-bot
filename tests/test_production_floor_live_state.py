from ipm.state import InventoryRowState
from production_floor_live_state import ProductionFloorLiveStateReader


class FakeRects:
    def __init__(self):
        self.rects = {
            "ORES_PANEL_TEXT": (0, 0, 1, 1),
            "ORES_TOP_ANCHOR": (0, 0, 1, 1),
        }

    def get(self, key):
        return self.rects.get(key)


class FakeCapture:
    def capture_client_bbox(self, _bbox):
        return None


class FakeActions:
    def __init__(self):
        self.current_tab = ""
        self.page_index = 0
        self.reset_calls = 0

    def open_alloys_panel(self):
        self.current_tab = "bars"
        self.page_index = 0
        return True

    def open_items_panel(self):
        self.current_tab = "items"
        self.page_index = 0
        return True

    def scroll_resource_list_down(self):
        self.page_index += 1
        return True

    def reset_ui(self):
        self.reset_calls += 1


class FakeInventoryReader:
    def __init__(self, actions):
        self.actions = actions
        self.pages = {
            ("bars", 0): {
                1: InventoryRowState(name="Copper Bar", quantity=12, backend="fake"),
                2: InventoryRowState(name="Iron Bar", quantity=7, backend="fake"),
            },
            ("bars", 1): {
                1: InventoryRowState(name="Inerton Alloy", quantity=3, backend="fake"),
                2: InventoryRowState(name="Iron Bar", quantity=7, backend="fake"),
            },
            ("bars", 2): {
                1: InventoryRowState(name="Inerton Alloy", quantity=3, backend="fake"),
                2: InventoryRowState(name="Iron Bar", quantity=7, backend="fake"),
            },
            ("items", 0): {
                1: InventoryRowState(name="Copper Wire", quantity=9, backend="fake"),
            },
            ("items", 1): {
                1: InventoryRowState(name="Copper Wire", quantity=9, backend="fake"),
            },
        }

    def read_visible_rows(self, *, known_names, aliases=None):
        _ = known_names, aliases
        return self.pages.get((self.actions.current_tab, self.actions.page_index), {})


class _TestableProductionFloorLiveStateReader(ProductionFloorLiveStateReader):
    def _scroll_to_top(self) -> None:
        return None


def test_production_floor_live_state_reads_bars_and_items_and_reports_assignment_blockers():
    actions = FakeActions()
    reader = _TestableProductionFloorLiveStateReader(
        config=type("Config", (), {"visible_ore_rows": 2})(),
        rects=FakeRects(),
        capture=FakeCapture(),
        actions=actions,
        inventory_reader=FakeInventoryReader(actions),
    )

    result = reader.read()

    assert result["bars"] == {
        "Copper Bar": 12,
        "Iron Bar": 7,
        "Inerton Alloy": 3,
    }
    assert result["items"] == {
        "Copper Wire": 9,
    }
    assert result["seam_status"] == {
        "active_smelter_assignments": {
            "feasible": False,
            "blocker": "blocked: no calibrated production list text/row rects and no verified production assignment parser exist",
        },
        "active_crafter_assignments": {
            "feasible": False,
            "blocker": "blocked: no calibrated production list text/row rects and no verified production assignment parser exist",
        },
        "current_bar_inventory": {
            "feasible": True,
            "blocker": "",
        },
        "current_item_inventory": {
            "feasible": True,
            "blocker": "",
        },
    }
    assert actions.reset_calls == 1
