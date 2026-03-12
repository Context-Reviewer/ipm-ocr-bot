from PIL import Image

from ipm.state import InventoryRowState, ProductionOverviewCardState
from production_floor_live_state import ProductionFloorLiveStateReader


class FakeRects:
    def __init__(self):
        self.rects = {
            "ORES_PANEL_TEXT": (0, 0, 1, 1),
            "ORES_TOP_ANCHOR": (0, 0, 1, 1),
            "ORE_ROW1_READ": (70, 545, 292, 50),
            "ORE_ROW2_READ": (70, 605, 292, 45),
            "PRODUCTION_CARD1": (0, 0, 1, 1),
            "PRODUCTION_CARD2": (0, 0, 1, 1),
            "PRODUCTION_CARD3": (0, 0, 1, 1),
            "PRODUCTION_CARD4": (0, 0, 1, 1),
        }

    def get(self, key):
        return self.rects.get(key)


class FakeCapture:
    def capture_client_bbox(self, _bbox):
        return Image.new("RGB", (48, 48), "white")


class FakeActions:
    def __init__(self):
        self.current_tab = ""
        self.page_index = 0
        self.reset_calls = 0

    def open_alloys_panel(self):
        self.current_tab = "bars"
        self.page_index = 0
        return True

    def open_ores_panel(self):
        self.current_tab = "ores"
        self.page_index = 0
        return True

    def open_items_panel(self):
        self.current_tab = "items"
        self.page_index = 0
        return True

    def open_smelter_panel(self):
        return True

    def open_crafter_panel(self):
        return True

    def scroll_resource_list_down(self):
        self.page_index += 1
        return True

    def scroll_resource_list_up(self):
        return True

    def reset_ui(self):
        self.reset_calls += 1


class FakeInventoryReader:
    def __init__(self, actions):
        self.actions = actions
        self.pages = {
            ("ores", 0): {
                1: InventoryRowState(name="Copper", quantity=500, backend="fake"),
                2: InventoryRowState(name="Iron", quantity=250, backend="fake"),
            },
            ("ores", 1): {
                1: InventoryRowState(name="Copper", quantity=500, backend="fake"),
                2: InventoryRowState(name="Iron", quantity=250, backend="fake"),
            },
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


class FakeProductionReader:
    def read_cards(self, *, tab, open_tab, templates, inventory_counts, input_templates=None, ore_inventory_counts=None):
        if not open_tab():
            raise ValueError("open_tab_failed")
        if not templates:
            raise ValueError("missing_templates")
        if not inventory_counts:
            raise ValueError("missing_inventory")
        _ = input_templates, ore_inventory_counts
        if tab == "smelt":
            return [
                ProductionOverviewCardState(slot_index=1, tab="smelt", output_name="Lead Bar", active=False, timer_text=None, backend="fake"),
                ProductionOverviewCardState(slot_index=2, tab="smelt", output_name="Iron Bar", active=True, timer_text="12s", backend="fake"),
                ProductionOverviewCardState(slot_index=3, tab="smelt", output_name="Copper Bar", active=True, timer_text="3s", backend="fake"),
                ProductionOverviewCardState(slot_index=4, tab="smelt", output_name="", active=False, timer_text=None, backend="locked"),
            ]
        return [
            ProductionOverviewCardState(slot_index=1, tab="craft", output_name="Copper Wire", active=True, timer_text="21s", backend="fake"),
            ProductionOverviewCardState(slot_index=2, tab="craft", output_name="Battery", active=True, timer_text="21s", backend="fake"),
            ProductionOverviewCardState(slot_index=3, tab="craft", output_name="", active=False, timer_text=None, backend="locked"),
            ProductionOverviewCardState(slot_index=4, tab="craft", output_name="", active=False, timer_text=None, backend="empty"),
        ]


class _TestableProductionFloorLiveStateReader(ProductionFloorLiveStateReader):
    def _scroll_to_top(self) -> None:
        return None


def test_production_floor_live_state_reads_bars_items_and_overview_assignment_queues():
    actions = FakeActions()
    reader = _TestableProductionFloorLiveStateReader(
        config=type("Config", (), {"visible_ore_rows": 2})(),
        rects=FakeRects(),
        capture=FakeCapture(),
        actions=actions,
        inventory_reader=FakeInventoryReader(actions),
        production_reader=FakeProductionReader(),
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
    assert result["smelter_queue"] == {
        "Copper Bar": 1,
        "Iron Bar": 1,
    }
    assert result["crafter_queue"] == {
        "Battery": 1,
        "Copper Wire": 1,
    }
    assert result["seam_status"] == {
        "active_smelter_assignments": {
            "feasible": True,
            "blocker": "",
            "reader_contract": (
                "read four production overview slots as ProductionOverviewCardState(slot_index=<int>, tab=<smelt|craft>, "
                "output_name=<canonical output>, active=<bool>, timer_text=<optional>, backend=<matcher>)"
            ),
            "parser_contract": "aggregate only cards with output_name and active=true into dict[output_name, active_count]",
            "required_rects": [
                "PRODUCTION_CARD1",
                "PRODUCTION_CARD2",
                "PRODUCTION_CARD3",
                "PRODUCTION_CARD4",
            ],
            "cards_read": [
                {"slot_index": 1, "tab": "smelt", "output_name": "Lead Bar", "active": False, "timer_text": None, "backend": "fake"},
                {"slot_index": 2, "tab": "smelt", "output_name": "Iron Bar", "active": True, "timer_text": "12s", "backend": "fake"},
                {"slot_index": 3, "tab": "smelt", "output_name": "Copper Bar", "active": True, "timer_text": "3s", "backend": "fake"},
                {"slot_index": 4, "tab": "smelt", "output_name": "", "active": False, "timer_text": None, "backend": "locked"},
            ],
        },
        "active_crafter_assignments": {
            "feasible": True,
            "blocker": "",
            "reader_contract": (
                "read four production overview slots as ProductionOverviewCardState(slot_index=<int>, tab=<smelt|craft>, "
                "output_name=<canonical output>, active=<bool>, timer_text=<optional>, backend=<matcher>)"
            ),
            "parser_contract": "aggregate only cards with output_name and active=true into dict[output_name, active_count]",
            "required_rects": [
                "PRODUCTION_CARD1",
                "PRODUCTION_CARD2",
                "PRODUCTION_CARD3",
                "PRODUCTION_CARD4",
            ],
            "cards_read": [
                {"slot_index": 1, "tab": "craft", "output_name": "Copper Wire", "active": True, "timer_text": "21s", "backend": "fake"},
                {"slot_index": 2, "tab": "craft", "output_name": "Battery", "active": True, "timer_text": "21s", "backend": "fake"},
                {"slot_index": 3, "tab": "craft", "output_name": "", "active": False, "timer_text": None, "backend": "locked"},
                {"slot_index": 4, "tab": "craft", "output_name": "", "active": False, "timer_text": None, "backend": "empty"},
            ],
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
