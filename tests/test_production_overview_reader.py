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
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_output_region_bonus",
        lambda self, **kwargs: (0.0, ""),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_smelt_input_quantity_bonus",
        lambda self, **kwargs: (0.0, ""),
    )

    name, backend = reader._resolve_output_name(
        tab="smelt",
        card=card,
        templates={"Alpha": alpha, "Beta": beta},
        inventory_counts={"Alpha": 42, "Beta": 7},
        output_quantity=42,
    )

    assert name == "Alpha"
    assert backend == "icon_template_match"


def test_production_overview_reader_uses_input_signal_to_break_icon_tie(monkeypatch):
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
        classmethod(lambda cls, template, target: 0.84 if template is alpha else 0.85),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_input_identity_bonus",
        lambda self, **kwargs: (0.05, "input_template_match") if kwargs["output_name"] == "Alpha" else (0.0, ""),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_output_region_bonus",
        lambda self, **kwargs: (0.0, ""),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_smelt_input_quantity_bonus",
        lambda self, **kwargs: (0.0, ""),
    )

    name, backend = reader._resolve_output_name(
        tab="smelt",
        card=card,
        templates={"Alpha": alpha, "Beta": beta},
        inventory_counts={},
        output_quantity=None,
        input_templates={"Lead": Image.new("RGB", (10, 10), "white")},
    )

    assert name == "Alpha"
    assert backend == "icon_template_match+input_template_match"


def test_production_overview_reader_uses_smelt_output_region_signal_to_break_tie(monkeypatch):
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
        classmethod(lambda cls, template, target: 0.84 if template is alpha else 0.85),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_output_region_bonus",
        lambda self, **kwargs: (0.03, "output_region_match") if kwargs["output_name"] == "Alpha" else (0.0, ""),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_smelt_input_quantity_bonus",
        lambda self, **kwargs: (0.0, ""),
    )

    name, backend = reader._resolve_output_name(
        tab="smelt",
        card=card,
        templates={"Alpha": alpha, "Beta": beta},
        inventory_counts={},
        output_quantity=None,
    )

    assert name == "Alpha"
    assert backend == "icon_template_match+output_region_match"


def test_production_overview_reader_uses_smelt_input_quantity_signal_to_break_tie(monkeypatch):
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
        classmethod(lambda cls, template, target: 0.84 if template is alpha else 0.85),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_output_region_bonus",
        lambda self, **kwargs: (0.0, ""),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_smelt_input_quantity_bonus",
        lambda self, **kwargs: (0.03, "input_quantity_match") if kwargs["output_name"] == "Alpha" else (0.0, ""),
    )

    name, backend = reader._resolve_output_name(
        tab="smelt",
        card=card,
        templates={"Alpha": alpha, "Beta": beta},
        inventory_counts={},
        output_quantity=None,
        ore_inventory_counts={"Lead": 404},
        input_available_quantity=404,
    )

    assert name == "Alpha"
    assert backend == "icon_template_match+input_quantity_match"


def test_production_overview_reader_parses_verified_smelt_recipe_names():
    verified = ProductionOverviewReader._verified_smelt_recipe_names(
        "SMELT RECIPES Copper Bar 15s 1.00K Lead Bar 30s Iron B 23s 1.00K Silicon 46s"
    )

    assert verified == {"Copper Bar", "Iron Bar", "Lead Bar", "Silicon Bar"}


def test_production_overview_reader_uses_craft_visual_active_fallback(monkeypatch):
    reader = ProductionOverviewReader(
        rects=None,
        capture=None,
        actions=None,
        perception=_FakePerception(),
    )
    card = Image.new("RGB", (240, 295), "black")
    monkeypatch.setattr(ProductionOverviewReader, "_read_timer_text", lambda self, image: ("797/5", "fake"))
    monkeypatch.setattr(ProductionOverviewReader, "_progress_fill_fraction", staticmethod(lambda image: 0.012))
    monkeypatch.setattr(ProductionOverviewReader, "_cancel_button_signal", classmethod(lambda cls, image: True))
    monkeypatch.setattr(ProductionOverviewReader, "_timer_region_signal", classmethod(lambda cls, image: True))

    active, timer_text, backend = reader._resolve_active_state(tab="craft", card=card)

    assert active is True
    assert timer_text == "797/5"
    assert backend == "cancel_button_signal+timer_region_signal"


def test_production_overview_reader_keeps_off_without_visual_signals_inactive(monkeypatch):
    reader = ProductionOverviewReader(
        rects=None,
        capture=None,
        actions=None,
        perception=_FakePerception(),
    )
    card = Image.new("RGB", (240, 295), "black")
    monkeypatch.setattr(ProductionOverviewReader, "_read_timer_text", lambda self, image: ("OFF", "fake"))
    monkeypatch.setattr(ProductionOverviewReader, "_progress_fill_fraction", staticmethod(lambda image: 0.0))
    monkeypatch.setattr(ProductionOverviewReader, "_cancel_button_signal", classmethod(lambda cls, image: False))
    monkeypatch.setattr(ProductionOverviewReader, "_timer_region_signal", classmethod(lambda cls, image: False))

    active, timer_text, backend = reader._resolve_active_state(tab="craft", card=card)

    assert active is False
    assert timer_text is None
    assert backend == "fake"


def test_production_overview_reader_falls_back_to_smelt_recipe_popup(monkeypatch):
    card = Image.new("RGB", (240, 295), "black")

    class _SingleRectCapture:
        def capture_client_bbox(self, rect):
            _ = rect
            return card.copy()

    class _SingleRectStore:
        def get(self, key):
            if key == "PRODUCTION_CARD1":
                return (0, 0, 240, 295)
            return None

    reader = ProductionOverviewReader(
        rects=_SingleRectStore(),
        capture=_SingleRectCapture(),
        actions=_FakeActions(),
        perception=_FakePerception(),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_card_status",
        staticmethod(lambda image: "card"),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_read_output_quantity",
        lambda self, image: (574, "qty"),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_read_input_available_quantity",
        lambda self, image: (969_870, "input_qty"),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_resolve_output_name",
        lambda self, **kwargs: (_ for _ in ()).throw(ValueError("ambiguous_output_match:Silicon Bar:1.01:1.00")),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_resolve_smelt_output_from_recipe_popup",
        lambda self, **kwargs: ("Silicon Bar", "smelt_recipe_popup_match"),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_resolve_active_state",
        lambda self, **kwargs: (True, "8s", "progress_fill_signal"),
    )

    state = reader._read_card(
        slot_index=1,
        tab="smelt",
        rect_key="PRODUCTION_CARD1",
        templates={"Silicon Bar": Image.new("RGB", (10, 10), "green")},
        inventory_counts={},
    )

    assert state.output_name == "Silicon Bar"
    assert state.active is True
    assert state.backend == "smelt_recipe_popup_match+qty+input_qty+progress_fill_signal"


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


def test_production_overview_reader_scrolls_back_to_top_with_best_anchor_fallback():
    top_frame = Image.new("RGB", (240, 295), "#345678")
    lower_frame = Image.new("RGB", (240, 295), "#111111")
    noisy_frame = Image.new("RGB", (240, 295), "#222222")
    near_top_frame = top_frame.copy()
    near_top_frame.paste(Image.new("RGB", (20, 20), "#355779"), (5, 5))
    reader = ProductionOverviewReader(
        rects=_FakeRects(),
        capture=_FakeCapture([lower_frame, noisy_frame, near_top_frame, noisy_frame]),
        actions=_FakeActions(),
        perception=_FakePerception(),
    )

    reader._scroll_back_to_top(top_anchor=top_frame)

    assert len(reader.actions.scrolls) >= 1
