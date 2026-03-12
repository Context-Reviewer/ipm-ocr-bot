from PIL import Image
import pytest

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


def test_production_overview_reader_accepts_scrolled_recipe_popup_with_tile_signal(monkeypatch):
    panel = Image.new("RGB", (350, 580), "#112244")

    class _PopupCapture:
        def capture_client_bbox(self, rect):
            _ = rect
            return panel.copy()

    class _PopupRects:
        def get(self, key):
            if key == "PRODUCTION_CARD1":
                return (0, 0, 240, 295)
            if key == "SMELT_RECIPES_PANEL":
                return (0, 0, 350, 580)
            return None

    reader = ProductionOverviewReader(
        rects=_PopupRects(),
        capture=_PopupCapture(),
        actions=_FakeActions(),
        perception=_FakePerception(),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_read_smelt_recipe_popup_text",
        lambda self, image: "SMELT RECIPES",
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_smelt_recipe_popup_tile_signal_count",
        classmethod(lambda cls, image: 4),
    )

    panel_image, verified_names = reader._open_smelt_recipe_popup(rect_key="PRODUCTION_CARD1")

    assert panel_image.size == (350, 580)
    assert verified_names == set()


def test_production_overview_reader_maps_scrolled_popup_pages_to_recipe_order():
    page0 = ProductionOverviewReader._smelt_recipe_page_candidates(page_index=0)
    page1 = ProductionOverviewReader._smelt_recipe_page_candidates(page_index=1)
    page2 = ProductionOverviewReader._smelt_recipe_page_candidates(page_index=2)

    assert [entry.output_name for entry in page0] == [
        "Copper Bar",
        "Iron Bar",
        "Lead Bar",
        "Silicon Bar",
        "Aluminium Bar",
        "Silver Bar",
    ]
    assert [entry.output_name for entry in page1] == [
        "Lead Bar",
        "Silicon Bar",
        "Aluminium Bar",
        "Silver Bar",
    ]
    assert [entry.output_name for entry in page2] == [
        "Aluminium Bar",
        "Silver Bar",
        "Gold Bar",
        "Bronze Bar",
    ]


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
    assert timer_text is None
    assert backend == "progress_fill_hint+cancel_button_signal+timer_region_signal"


def test_production_overview_reader_uses_smelt_visual_active_fallback(monkeypatch):
    reader = ProductionOverviewReader(
        rects=None,
        capture=None,
        actions=None,
        perception=_FakePerception(),
    )
    card = Image.new("RGB", (240, 295), "black")
    monkeypatch.setattr(ProductionOverviewReader, "_read_timer_text", lambda self, image: (",.44K/1.OOK", "fake"))
    monkeypatch.setattr(ProductionOverviewReader, "_progress_fill_fraction", staticmethod(lambda image: 0.019))
    monkeypatch.setattr(ProductionOverviewReader, "_cancel_button_signal", classmethod(lambda cls, image: False))
    monkeypatch.setattr(ProductionOverviewReader, "_timer_region_signal", classmethod(lambda cls, image: True))

    active, timer_text, backend = reader._resolve_active_state(tab="smelt", card=card)

    assert active is True
    assert timer_text is None
    assert backend == "progress_fill_hint+timer_region_signal"


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


def test_production_overview_reader_keeps_blank_without_visual_signals_inactive(monkeypatch):
    reader = ProductionOverviewReader(
        rects=None,
        capture=None,
        actions=None,
        perception=_FakePerception(),
    )
    card = Image.new("RGB", (240, 295), "black")
    monkeypatch.setattr(ProductionOverviewReader, "_read_timer_text", lambda self, image: ("", "fake"))
    monkeypatch.setattr(ProductionOverviewReader, "_progress_fill_fraction", staticmethod(lambda image: 0.0))
    monkeypatch.setattr(ProductionOverviewReader, "_cancel_button_signal", classmethod(lambda cls, image: False))
    monkeypatch.setattr(ProductionOverviewReader, "_timer_region_signal", classmethod(lambda cls, image: False))

    active, timer_text, backend = reader._resolve_active_state(tab="smelt", card=card)

    assert active is False
    assert timer_text is None
    assert backend == "visual_idle_signal"


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
        self.clicks = []

    def scroll_client_wheel(self, point, delta, *, delay=None):
        self.scrolls.append((point, delta, delay))
        return True

    def click_rect_center(self, rect_key, *, delay=None):
        self.clicks.append((rect_key, delay))
        return True

    def click_client_point(self, point, *, delay=None):
        self.clicks.append((point, delay))
        return True


class _FakeCapture:
    def __init__(self, images, *, screen_frames=None):
        self._images = list(images)
        self._index = 0
        self._screen_frames = list(screen_frames or images)
        self._screen_index = 0

    def capture_client_bbox(self, rect):
        _ = rect
        image = self._images[min(self._index, len(self._images) - 1)]
        self._index += 1
        return image.copy()

    def capture_screen(self):
        image = self._screen_frames[min(self._screen_index, len(self._screen_frames) - 1)]
        self._screen_index += 1
        return image.copy()


class _FakeRects:
    def get(self, key):
        if key == "PRODUCTION_CARD1":
            return (0, 0, 240, 295)
        if key == "PRODUCTION_CARD2":
            return (0, 0, 240, 295)
        if key == "PRODUCTION_CARD3":
            return (0, 0, 240, 295)
        if key == "PRODUCTION_CARD4":
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


def test_production_overview_reader_caps_top_scroll_attempts(monkeypatch):
    frame = Image.new("RGB", (240, 295), "#123456")
    frame.paste(Image.new("RGB", (40, 40), "#f0f0f0"), (20, 20))
    reader = ProductionOverviewReader(
        rects=_FakeRects(),
        capture=_FakeCapture([frame, frame, frame, frame, frame]),
        actions=_FakeActions(),
        perception=_FakePerception(),
    )
    monkeypatch.setattr(ProductionOverviewReader, "_extract_output_icon", classmethod(lambda cls, image: image))

    top_anchor = reader._scroll_to_top_view()

    assert top_anchor is not None
    assert len(reader.actions.scrolls) <= 3


def test_production_overview_reader_uses_upper_rects_for_lower_view(monkeypatch):
    reader = ProductionOverviewReader(
        rects=_FakeRects(),
        capture=_FakeCapture([Image.new("RGB", (240, 295), "#123456")]),
        actions=_FakeActions(),
        perception=_FakePerception(),
    )
    calls = []

    monkeypatch.setattr(ProductionOverviewReader, "_scroll_to_top_view", lambda self: Image.new("RGB", (240, 295), "#123456"))
    monkeypatch.setattr(ProductionOverviewReader, "_scroll_to_lower_cards", lambda self, top_anchor: None)
    monkeypatch.setattr(ProductionOverviewReader, "_scroll_back_to_top", lambda self, top_anchor: None)
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_read_card",
        lambda self, *, slot_index, rect_key, **kwargs: calls.append((slot_index, rect_key)) or object(),
    )

    reader.read_cards(
        tab="smelt",
        open_tab=lambda: True,
        templates={"Copper Bar": Image.new("RGB", (10, 10), "red")},
        inventory_counts={},
    )

    assert calls == [
        (1, "PRODUCTION_CARD1"),
        (2, "PRODUCTION_CARD2"),
        (3, "PRODUCTION_CARD1"),
        (4, "PRODUCTION_CARD2"),
    ]


def test_production_overview_reader_closes_smelt_recipe_popup_by_visual_diff():
    popup_panel = Image.new("RGB", (350, 580), "#112244")
    closed_panel = Image.new("RGB", (350, 580), "#001122")

    class _PopupCapture:
        def __init__(self):
            self._index = 0

        def capture_client_bbox(self, rect):
            _ = rect
            self._index += 1
            return popup_panel.copy() if self._index == 1 else closed_panel.copy()

    class _PopupRects:
        def get(self, key):
            if key == "SMELT_RECIPES_PANEL":
                return (0, 0, 350, 580)
            if key == "SMELT_RECIPES_CLOSE":
                return (0, 0, 10, 10)
            return None

    actions = _FakeActions()
    reader = ProductionOverviewReader(
        rects=_PopupRects(),
        capture=_PopupCapture(),
        actions=actions,
        perception=_FakePerception(),
    )

    reader._close_smelt_recipe_popup()

    assert actions.clicks[0][0] == "SMELT_RECIPES_CLOSE"


def test_production_overview_reader_generates_deterministic_tooltip_probe_points():
    points = ProductionOverviewReader._tooltip_probe_points((10, 20, 40, 60))

    assert points == ((19, 34), (28, 34), (19, 46), (28, 46))


def test_production_overview_reader_computes_local_tooltip_crop_box():
    crop_box = ProductionOverviewReader._tooltip_crop_box((60, 80, 95, 120), card_size=(240, 295))

    assert crop_box == (0, 44, 111, 94)


def test_production_overview_reader_stops_on_first_valid_tooltip_label(monkeypatch):
    card = Image.new("RGB", (240, 295), "#112244")
    frame = Image.new("RGB", (240, 295), "#223355")

    class _TooltipPerception:
        def __init__(self):
            self.calls = 0

        def read_text(self, image, *, prompt, mode):
            _ = image, prompt, mode
            self.calls += 1
            value = "noise" if self.calls == 1 else "Lead"
            return type("Result", (), {"value": value, "backend": "fake"})()

    reader = ProductionOverviewReader(
        rects=_FakeRects(),
        capture=_FakeCapture([card], screen_frames=[frame, frame, frame]),
        actions=_FakeActions(),
        perception=_TooltipPerception(),
    )

    output_name, backend = reader._probe_tooltip_identity(
        rect_key="PRODUCTION_CARD1",
        tab="smelt",
        templates={"Lead Bar": Image.new("RGB", (10, 10), "gray")},
    )

    assert output_name == "Lead Bar"
    assert backend == "tooltip_probe_bar_fake"
    assert len(reader.actions.clicks) == 2


def test_production_overview_reader_rejects_invalid_tooltip_text():
    card = Image.new("RGB", (240, 295), "#112244")
    frame = Image.new("RGB", (240, 295), "#223355")

    class _TooltipPerception:
        def read_text(self, image, *, prompt, mode):
            _ = image, prompt, mode
            return type("Result", (), {"value": "not a real name", "backend": "fake"})()

    reader = ProductionOverviewReader(
        rects=_FakeRects(),
        capture=_FakeCapture([card], screen_frames=[frame, frame, frame, frame]),
        actions=_FakeActions(),
        perception=_TooltipPerception(),
    )

    with pytest.raises(ValueError, match="tooltip_probe_no_valid_label"):
        reader._probe_tooltip_identity(
            rect_key="PRODUCTION_CARD1",
            tab="smelt",
            templates={"Lead Bar": Image.new("RGB", (10, 10), "gray")},
        )


def test_production_overview_reader_fails_closed_without_screen_capture():
    card = Image.new("RGB", (240, 295), "#112244")

    class _NoScreenCapture:
        def capture_client_bbox(self, rect):
            _ = rect
            return card.copy()

    reader = ProductionOverviewReader(
        rects=_FakeRects(),
        capture=_NoScreenCapture(),
        actions=_FakeActions(),
        perception=_FakePerception(),
    )

    with pytest.raises(ValueError, match="tooltip_probe_capture_unavailable"):
        reader._probe_tooltip_identity(
            rect_key="PRODUCTION_CARD1",
            tab="smelt",
            templates={"Lead Bar": Image.new("RGB", (10, 10), "gray")},
        )


def test_production_overview_reader_tries_tooltip_probe_before_recipe_popup(monkeypatch):
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
    fallback_calls = []
    monkeypatch.setattr(ProductionOverviewReader, "_card_status", staticmethod(lambda image: "card"))
    monkeypatch.setattr(ProductionOverviewReader, "_read_output_quantity", lambda self, image: (574, "qty"))
    monkeypatch.setattr(ProductionOverviewReader, "_read_input_available_quantity", lambda self, image: (404, "input_qty"))
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_resolve_output_name",
        lambda self, **kwargs: (_ for _ in ()).throw(ValueError("ambiguous_output_match:Lead Bar:1.00:0.99")),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_probe_tooltip_identity",
        lambda self, **kwargs: fallback_calls.append("tooltip") or ("Lead Bar", "tooltip_probe_bar_fake"),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_resolve_smelt_output_from_recipe_popup",
        lambda self, **kwargs: fallback_calls.append("popup") or ("Lead Bar", "smelt_recipe_popup_match"),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_resolve_active_state",
        lambda self, **kwargs: (False, None, "visual_idle_signal"),
    )

    state = reader._read_card(
        slot_index=1,
        tab="smelt",
        rect_key="PRODUCTION_CARD1",
        templates={"Lead Bar": Image.new("RGB", (10, 10), "gray")},
        inventory_counts={},
    )

    assert state.output_name == "Lead Bar"
    assert state.backend == "tooltip_probe_bar_fake+qty+input_qty+visual_idle_signal"
    assert fallback_calls == ["tooltip"]
