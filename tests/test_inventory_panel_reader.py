from PIL import Image
from PIL import ImageDraw

from ipm.config import RuntimeConfig
from ipm.readers.inventory_panel import InventoryPanelReader
from ipm.rects import RectStore
from ipm.state import InventoryRowState


class FakeCapture:
    def capture_client_bbox(self, _bbox):
        return Image.new("RGB", (50, 20), "white")


class FakePerception:
    def __init__(self, mapping):
        self.mapping = mapping

    def read_text(self, _image, *, prompt="", mode="generic"):
        value = self.mapping.get((prompt, mode), "")

        class Result:
            confidence = 1.0

            def __init__(self, value):
                self.value = value
                self.backend = "fake"

        return Result(value)


def test_inventory_panel_reader_reads_bar_rows_from_panel_text():
    cfg = RuntimeConfig(visible_ore_rows=2)
    rects = RectStore(path=None, rects={"ORES_PANEL_TEXT": (0, 0, 1, 1)})
    perception = FakePerception(
        {
            (cfg.perception.prompt_resource_panel, "generic"): "\n".join(
                [
                    "Copper Bar",
                    "Iron Bar",
                    "12",
                    "7",
                ]
            )
        }
    )

    rows = InventoryPanelReader(cfg, rects, FakeCapture(), perception).read_visible_rows(
        known_names=["Copper Bar", "Iron Bar"],
    )

    assert rows == {
        1: InventoryRowState(name="Copper Bar", quantity=12, backend="fake"),
        2: InventoryRowState(name="Iron Bar", quantity=7, backend="fake"),
    }


def test_inventory_panel_reader_matches_multiword_item_name_from_row_text():
    cfg = RuntimeConfig(visible_ore_rows=1)
    rects = RectStore(
        path=None,
        rects={
            "ORE_ROW1_READ": (0, 0, 1, 1),
            "ORE_ROW1_QTY": (0, 0, 1, 1),
        },
    )
    perception = FakePerception(
        {
            (cfg.perception.prompt_resource_name, "generic"): "Copper Wire 82K",
            (cfg.perception.prompt_ore_quantity, "ore_qty"): "82K",
        }
    )

    rows = InventoryPanelReader(cfg, rects, FakeCapture(), perception).read_visible_rows(
        known_names=["Copper Wire"],
    )

    assert rows == {
        1: InventoryRowState(name="Copper Wire", quantity=82_000, backend="fake"),
    }


def test_inventory_panel_reader_detects_panel_structure_from_list_like_image():
    image = Image.new("RGB", (240, 180), "#171d29")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((8, 20, 228, 64), radius=18, outline="#3bdcff", width=2, fill="#27506d")
    draw.rounded_rectangle((8, 74, 228, 118), radius=18, outline="#3bdcff", width=2, fill="#27506d")

    assert InventoryPanelReader._panel_structure_present(image) is True


def test_inventory_panel_reader_rejects_blank_panel_structure():
    image = Image.new("RGB", (240, 180), "#111111")

    assert InventoryPanelReader._panel_structure_present(image) is False
