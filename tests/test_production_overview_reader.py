from PIL import Image

from ipm.readers.production_overview import ProductionOverviewReader


class _FakePerception:
    def read_text(self, image, *, prompt, mode):
        _ = image, prompt, mode
        return type("Result", (), {"value": "", "backend": "fake"})()


def test_production_overview_reader_uses_inventory_quantity_to_break_icon_tie(monkeypatch):
    reader = ProductionOverviewReader(
        rects=None,
        capture=None,
        actions=None,
        perception=_FakePerception(),
    )
    card = Image.new("RGB", (240, 295), "black")
    alpha = Image.new("RGB", (10, 10), "red")
    beta = Image.new("RGB", (10, 10), "blue")
    monkeypatch.setattr(ProductionOverviewReader, "_candidate_output_icons", classmethod(lambda cls, image: [image]))
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_icon_similarity",
        classmethod(lambda cls, template, target: 0.78 if template is alpha else 0.79),
    )

    name, backend = reader._resolve_output_name(
        card=card,
        templates={"Alpha": alpha, "Beta": beta},
        inventory_counts={"Alpha": 42, "Beta": 7},
        output_quantity=42,
    )

    assert name == "Alpha"
    assert backend == "icon_template_match"


class _FakeActions:
    def __init__(self):
        self.scrolls = []

    def scroll_client_wheel(self, point, delta, *, delay=None):
        self.scrolls.append((point, delta, delay))
        return True


class _FakeCapture:
    def __init__(self, images):
        self._images = list(images)
        self._index = 0

    def capture_client_bbox(self, rect):
        _ = rect
        image = self._images[min(self._index, len(self._images) - 1)]
        self._index += 1
        return image.copy()


class _FakeRects:
    def get(self, key):
        if key == "PRODUCTION_CARD1":
            return (0, 0, 240, 295)
        if key == "PRODUCTION_CARD3":
            return (0, 0, 240, 295)
        return None


def test_production_overview_reader_scrolls_to_top_without_icon_latch():
    frame_a = Image.new("RGB", (240, 295), "#123456")
    frame_b = Image.new("RGB", (240, 295), "#345678")
    reader = ProductionOverviewReader(
        rects=_FakeRects(),
        capture=_FakeCapture([frame_a, frame_b, frame_b, frame_b]),
        actions=_FakeActions(),
        perception=_FakePerception(),
    )

    top_anchor = reader._scroll_to_top_view()

    assert top_anchor is not None
    assert top_anchor.tobytes() == frame_b.tobytes()
