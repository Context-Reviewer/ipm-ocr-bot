from PIL import Image

from ipm.config import RuntimeConfig
from ipm.readers import HudReader, OrePanelReader, PlanetPanelReader, SellDialogReader
from ipm.rects import RectStore
from ipm.state_reader import GameStateReader


class FakeCapture:
    def capture_client_bbox(self, _bbox):
        return Image.new("RGB", (50, 20), "white")


class FakePerception:
    def __init__(self, mapping):
        self.mapping = mapping

    def read_text(self, _image, *, prompt="", mode="generic"):
        value = self.mapping.get((prompt, mode), "")

        class Result:
            backend = "fake"
            confidence = 1.0

            def __init__(self, value):
                self.value = value

        return Result(value)


def test_state_reader_aggregates_snapshot():
    cfg = RuntimeConfig(visible_ore_rows=1)
    rects = RectStore(
        path=None,
        rects={
            "HUD_CASH": (0, 0, 1, 1),
            "PLANET_TITLE": (0, 0, 1, 1),
            "MINING_LVL": (0, 0, 1, 1),
            "SHIP_LVL": (0, 0, 1, 1),
            "CARGO_LVL": (0, 0, 1, 1),
            "ORE_ROW1_READ": (0, 0, 1, 1),
            "ORE_ROW1_QTY": (0, 0, 1, 1),
            "SELL_SELECTED_QTY": (0, 0, 1, 1),
            "SELL_SLIDER_TRACK": (0, 0, 1, 1),
        },
    )
    perception = FakePerception(
        {
            (cfg.perception.prompt_numeric, "numeric"): "108.33K",
            (cfg.perception.prompt_planet_title, "planet_title"): "2. DRASTA",
            (cfg.perception.prompt_ore_name, "generic"): "Copper",
            (cfg.perception.prompt_ore_quantity, "ore_qty"): "50K",
        }
    )
    capture = FakeCapture()
    state_reader = GameStateReader(
        hud_reader=HudReader(cfg, rects, capture, perception),
        planet_reader=PlanetPanelReader(cfg, rects, capture, perception),
        ore_reader=OrePanelReader(cfg, rects, capture, perception),
        sell_reader=SellDialogReader(cfg, rects, capture, perception),
    )
    snapshot = state_reader.read()
    assert snapshot.cash == 108330
    assert snapshot.current_planet.planet_id == 2
    assert snapshot.ore_rows[1].quantity == 50000
    assert snapshot.sell_dialog.selected_quantity == 50000


def test_state_reader_planet_snapshot_skips_ore_and_sell_reads():
    cfg = RuntimeConfig(visible_ore_rows=1)
    rects = RectStore(
        path=None,
        rects={
            "HUD_CASH": (0, 0, 1, 1),
            "PLANET_TITLE": (0, 0, 1, 1),
            "MINING_LVL": (0, 0, 1, 1),
            "SHIP_LVL": (0, 0, 1, 1),
            "CARGO_LVL": (0, 0, 1, 1),
            "ORE_ROW1_READ": (0, 0, 1, 1),
            "ORE_ROW1_QTY": (0, 0, 1, 1),
            "SELL_SELECTED_QTY": (0, 0, 1, 1),
            "SELL_SLIDER_TRACK": (0, 0, 1, 1),
        },
    )
    perception = FakePerception(
        {
            (cfg.perception.prompt_numeric, "numeric"): "108.33K",
            (cfg.perception.prompt_planet_title, "planet_title"): "2. DRASTA",
            (cfg.perception.prompt_ore_name, "generic"): "Copper",
            (cfg.perception.prompt_ore_quantity, "ore_qty"): "50K",
        }
    )
    capture = FakeCapture()
    state_reader = GameStateReader(
        hud_reader=HudReader(cfg, rects, capture, perception),
        planet_reader=PlanetPanelReader(cfg, rects, capture, perception),
        ore_reader=OrePanelReader(cfg, rects, capture, perception),
        sell_reader=SellDialogReader(cfg, rects, capture, perception),
    )
    snapshot = state_reader.read_planet_snapshot()
    assert snapshot.cash == 108330
    assert snapshot.current_planet.planet_id == 2
    assert snapshot.ore_rows == {}
    assert snapshot.sell_dialog.selected_quantity is None


def test_state_reader_cash_snapshot_skips_planet_ore_and_sell_reads():
    cfg = RuntimeConfig(visible_ore_rows=1)
    rects = RectStore(
        path=None,
        rects={
            "HUD_CASH": (0, 0, 1, 1),
            "PLANET_TITLE": (0, 0, 1, 1),
            "ORE_ROW1_READ": (0, 0, 1, 1),
            "ORE_ROW1_QTY": (0, 0, 1, 1),
            "SELL_SELECTED_QTY": (0, 0, 1, 1),
            "SELL_SLIDER_TRACK": (0, 0, 1, 1),
        },
    )
    perception = FakePerception(
        {
            (cfg.perception.prompt_numeric, "numeric"): "108.33K",
            (cfg.perception.prompt_planet_title, "planet_title"): "2. DRASTA",
            (cfg.perception.prompt_ore_name, "generic"): "Copper",
            (cfg.perception.prompt_ore_quantity, "ore_qty"): "50K",
        }
    )
    capture = FakeCapture()
    state_reader = GameStateReader(
        hud_reader=HudReader(cfg, rects, capture, perception),
        planet_reader=PlanetPanelReader(cfg, rects, capture, perception),
        ore_reader=OrePanelReader(cfg, rects, capture, perception),
        sell_reader=SellDialogReader(cfg, rects, capture, perception),
    )
    snapshot = state_reader.read_cash_snapshot()
    assert snapshot.cash == 108330
    assert snapshot.current_planet.planet_id is None
    assert snapshot.ore_rows == {}
    assert snapshot.sell_dialog.selected_quantity is None
