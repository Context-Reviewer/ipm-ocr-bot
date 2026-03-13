from pathlib import Path
from PIL import Image, ImageDraw
import pytest

from ipm.readers.production_overview import (
    _LocalizedTooltipTarget,
    ProductionOverviewReader,
    _ProductionCardLayout,
    _TooltipCropCandidate,
    _TooltipIconRegion,
    _TooltipProbeMarker,
    _TooltipProbePoint,
)


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


def test_production_overview_reader_derives_timer_text_box_from_card_geometry():
    assert ProductionOverviewReader._timer_text_box((240, 295)) == (65, 165, 139, 195)


def test_production_overview_reader_timer_text_box_stays_separate_from_quantity_regions():
    timer_box = ProductionOverviewReader._timer_text_box((240, 295))
    input_qty_box = (5, 85, 120, 125)
    output_qty_box = (140, 80, 230, 135)

    assert timer_box[1] >= input_qty_box[3]
    assert timer_box[1] >= output_qty_box[3]


def test_production_overview_reader_accepts_timer_text_with_visual_timer_presence(monkeypatch):
    reader = ProductionOverviewReader(
        rects=None,
        capture=None,
        actions=None,
        perception=_FakePerception(),
    )
    card = Image.new("RGB", (240, 295), "black")
    monkeypatch.setattr(ProductionOverviewReader, "_read_timer_text", lambda self, image: ("03:46", "fake"))
    monkeypatch.setattr(ProductionOverviewReader, "_progress_fill_fraction", staticmethod(lambda image: 0.0))
    monkeypatch.setattr(ProductionOverviewReader, "_cancel_button_signal", classmethod(lambda cls, image: False))
    monkeypatch.setattr(ProductionOverviewReader, "_timer_region_signal", classmethod(lambda cls, image: False))
    monkeypatch.setattr(ProductionOverviewReader, "_timer_text_presence_signal", classmethod(lambda cls, image: True))

    active, timer_text, backend = reader._resolve_active_state(tab="smelt", card=card)

    assert active is True
    assert timer_text == "03:46"
    assert backend == "timer_text_visual_signal+fake"


def test_production_overview_reader_normalizes_common_timer_ocr_substitutions(monkeypatch):
    reader = ProductionOverviewReader(
        rects=None,
        capture=None,
        actions=None,
        perception=_FakePerception(),
    )
    card = Image.new("RGB", (240, 295), "black")
    monkeypatch.setattr(ProductionOverviewReader, "_read_timer_text", lambda self, image: ("O3;46", "fake"))
    monkeypatch.setattr(ProductionOverviewReader, "_progress_fill_fraction", staticmethod(lambda image: 0.0))
    monkeypatch.setattr(ProductionOverviewReader, "_cancel_button_signal", classmethod(lambda cls, image: False))
    monkeypatch.setattr(ProductionOverviewReader, "_timer_region_signal", classmethod(lambda cls, image: False))
    monkeypatch.setattr(ProductionOverviewReader, "_timer_text_presence_signal", classmethod(lambda cls, image: True))

    active, timer_text, backend = reader._resolve_active_state(tab="smelt", card=card)

    assert active is True
    assert timer_text == "O3;46"
    assert backend == "timer_text_visual_signal+fake"


def test_production_overview_reader_accepts_hour_timer_text(monkeypatch):
    reader = ProductionOverviewReader(
        rects=None,
        capture=None,
        actions=None,
        perception=_FakePerception(),
    )
    card = Image.new("RGB", (240, 295), "black")
    monkeypatch.setattr(ProductionOverviewReader, "_read_timer_text", lambda self, image: ("1:22:43", "fake"))
    monkeypatch.setattr(ProductionOverviewReader, "_progress_fill_fraction", staticmethod(lambda image: 0.0))
    monkeypatch.setattr(ProductionOverviewReader, "_cancel_button_signal", classmethod(lambda cls, image: False))
    monkeypatch.setattr(ProductionOverviewReader, "_timer_region_signal", classmethod(lambda cls, image: False))
    monkeypatch.setattr(ProductionOverviewReader, "_timer_text_presence_signal", classmethod(lambda cls, image: True))

    active, timer_text, backend = reader._resolve_active_state(tab="smelt", card=card)

    assert active is True
    assert timer_text == "1:22:43"
    assert backend == "timer_text_visual_signal+fake"


def test_production_overview_reader_rejects_ampm_clock_text(monkeypatch):
    reader = ProductionOverviewReader(
        rects=None,
        capture=None,
        actions=None,
        perception=_FakePerception(),
    )
    card = Image.new("RGB", (240, 295), "black")
    monkeypatch.setattr(ProductionOverviewReader, "_read_timer_text", lambda self, image: ("4:44PM", "fake"))
    monkeypatch.setattr(ProductionOverviewReader, "_progress_fill_fraction", staticmethod(lambda image: 0.0))
    monkeypatch.setattr(ProductionOverviewReader, "_cancel_button_signal", classmethod(lambda cls, image: False))
    monkeypatch.setattr(ProductionOverviewReader, "_timer_region_signal", classmethod(lambda cls, image: False))
    monkeypatch.setattr(ProductionOverviewReader, "_timer_text_presence_signal", classmethod(lambda cls, image: True))

    with pytest.raises(ValueError, match="unreadable_active_state:4:44PM"):
        reader._resolve_active_state(tab="smelt", card=card)


def test_production_overview_reader_rejects_quantity_like_timer_text(monkeypatch):
    reader = ProductionOverviewReader(
        rects=None,
        capture=None,
        actions=None,
        perception=_FakePerception(),
    )
    card = Image.new("RGB", (240, 295), "black")
    monkeypatch.setattr(ProductionOverviewReader, "_read_timer_text", lambda self, image: ("77211.00K", "fake"))
    monkeypatch.setattr(ProductionOverviewReader, "_progress_fill_fraction", staticmethod(lambda image: 0.0))
    monkeypatch.setattr(ProductionOverviewReader, "_cancel_button_signal", classmethod(lambda cls, image: False))
    monkeypatch.setattr(ProductionOverviewReader, "_timer_region_signal", classmethod(lambda cls, image: False))
    monkeypatch.setattr(ProductionOverviewReader, "_timer_text_presence_signal", classmethod(lambda cls, image: True))

    with pytest.raises(ValueError, match="unreadable_active_state:77211.00K"):
        reader._resolve_active_state(tab="smelt", card=card)


def test_production_overview_reader_rejects_slash_quantity_like_timer_text(monkeypatch):
    reader = ProductionOverviewReader(
        rects=None,
        capture=None,
        actions=None,
        perception=_FakePerception(),
    )
    card = Image.new("RGB", (240, 295), "black")
    monkeypatch.setattr(ProductionOverviewReader, "_read_timer_text", lambda self, image: ("404/1.00K", "fake"))
    monkeypatch.setattr(ProductionOverviewReader, "_progress_fill_fraction", staticmethod(lambda image: 0.0))
    monkeypatch.setattr(ProductionOverviewReader, "_cancel_button_signal", classmethod(lambda cls, image: False))
    monkeypatch.setattr(ProductionOverviewReader, "_timer_region_signal", classmethod(lambda cls, image: False))
    monkeypatch.setattr(ProductionOverviewReader, "_timer_text_presence_signal", classmethod(lambda cls, image: True))

    with pytest.raises(ValueError, match="unreadable_active_state:404/1.00K"):
        reader._resolve_active_state(tab="smelt", card=card)


def test_production_overview_reader_rejects_malformed_timer_like_text_without_support(monkeypatch):
    reader = ProductionOverviewReader(
        rects=None,
        capture=None,
        actions=None,
        perception=_FakePerception(),
    )
    card = Image.new("RGB", (240, 295), "black")
    monkeypatch.setattr(ProductionOverviewReader, "_read_timer_text", lambda self, image: ("03:4G", "fake"))
    monkeypatch.setattr(ProductionOverviewReader, "_progress_fill_fraction", staticmethod(lambda image: 0.0))
    monkeypatch.setattr(ProductionOverviewReader, "_cancel_button_signal", classmethod(lambda cls, image: False))
    monkeypatch.setattr(ProductionOverviewReader, "_timer_region_signal", classmethod(lambda cls, image: False))
    monkeypatch.setattr(ProductionOverviewReader, "_timer_text_presence_signal", classmethod(lambda cls, image: True))

    with pytest.raises(ValueError, match="unreadable_active_state"):
        reader._resolve_active_state(tab="smelt", card=card)


def test_production_overview_reader_keeps_no_recipe_selected_inactive(monkeypatch):
    reader = ProductionOverviewReader(
        rects=None,
        capture=None,
        actions=None,
        perception=_FakePerception(),
    )
    card = Image.new("RGB", (240, 295), "black")
    monkeypatch.setattr(ProductionOverviewReader, "_read_timer_text", lambda self, image: ("No Recipe Selected", "fake"))
    monkeypatch.setattr(ProductionOverviewReader, "_progress_fill_fraction", staticmethod(lambda image: 0.0))
    monkeypatch.setattr(ProductionOverviewReader, "_cancel_button_signal", classmethod(lambda cls, image: False))
    monkeypatch.setattr(ProductionOverviewReader, "_timer_region_signal", classmethod(lambda cls, image: False))
    monkeypatch.setattr(ProductionOverviewReader, "_timer_text_presence_signal", classmethod(lambda cls, image: False))

    active, timer_text, backend = reader._resolve_active_state(tab="smelt", card=card)

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


class _RectRecordingCapture:
    def __init__(self, image):
        self.image = image
        self.rects = []

    def capture_client_bbox(self, rect):
        self.rects.append(rect)
        return self.image.copy()


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


def _make_structural_top_anchor(*, card_offset_y: int = 0) -> Image.Image:
    image = Image.new("RGB", (240, 150), "#161d2b")
    draw = ImageDraw.Draw(image)
    draw.rectangle((18, 10, 108, 30), fill="#f3f3f3")
    draw.rectangle((0, 40, 239, 64), fill="#2a3040")
    top = 65 + card_offset_y
    draw.rounded_rectangle((0, top, 205, top + 84), radius=20, fill="#2c5874", outline="#42e7ff", width=2)
    draw.rounded_rectangle((55, top + 48, 145, min(149, top + 76)), radius=10, outline="#18e3ff", width=3)
    draw.line((20, top + 40, 175, top + 40), fill="#090f18", width=8)
    draw.rectangle((15, top + 10, 55, top + 40), fill="#d6d6d6")
    draw.rectangle((122, top + 12, 162, top + 38), fill="#c2c2c2")
    return image


def _make_non_top_anchor() -> Image.Image:
    image = Image.new("RGB", (240, 150), "#161d2b")
    draw = ImageDraw.Draw(image)
    draw.rectangle((18, 10, 108, 30), fill="#f3f3f3")
    draw.rectangle((0, 40, 239, 90), fill="#2a3040")
    draw.rounded_rectangle((0, 95, 205, 149), radius=20, fill="#24384b", outline="#2b8aa1", width=2)
    draw.line((20, 126, 175, 126), fill="#090f18", width=8)
    return image


def test_production_overview_reader_derives_structural_top_anchor_rect_from_card():
    rect = ProductionOverviewReader._top_anchor_rect_from_card_rect((35, 540, 240, 295))

    assert rect == (35, 500, 240, 150)


def test_production_overview_reader_captures_structural_top_anchor_region():
    capture = _RectRecordingCapture(Image.new("RGB", (240, 150), "#123456"))
    reader = ProductionOverviewReader(
        rects=type("Rects", (), {"get": lambda self, key: (35, 540, 240, 295) if key == "PRODUCTION_CARD1" else None})(),
        capture=capture,
        actions=_FakeActions(),
        perception=_FakePerception(),
    )

    anchor = reader._capture_top_anchor()

    assert anchor is not None
    assert capture.rects == [(35, 500, 240, 150)]


def test_production_overview_reader_scrolls_to_top_without_icon_latch():
    frame_a = _make_non_top_anchor()
    frame_b = _make_structural_top_anchor()
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
    top_frame = _make_structural_top_anchor()
    lower_frame = _make_non_top_anchor()
    noisy_frame = _make_non_top_anchor()
    draw = ImageDraw.Draw(noisy_frame)
    draw.rectangle((12, 78, 42, 108), fill="#5fa6cf")
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


def test_production_overview_reader_scroll_back_stops_when_already_at_top():
    top_frame = _make_structural_top_anchor()
    reader = ProductionOverviewReader(
        rects=_FakeRects(),
        capture=_FakeCapture([top_frame, top_frame]),
        actions=_FakeActions(),
        perception=_FakePerception(),
    )

    reader._scroll_back_to_top(top_anchor=top_frame)

    assert reader.actions.scrolls == []


def test_production_overview_reader_scroll_back_stops_on_repeated_structural_top():
    top_frame = _make_structural_top_anchor()
    shifted_top = top_frame.copy()
    ImageDraw.Draw(shifted_top).rectangle((170, 112, 210, 138), fill="#345d78")
    reader = ProductionOverviewReader(
        rects=_FakeRects(),
        capture=_FakeCapture([_make_non_top_anchor(), shifted_top, shifted_top, shifted_top]),
        actions=_FakeActions(),
        perception=_FakePerception(),
    )

    reader._scroll_back_to_top(top_anchor=top_frame)

    assert len(reader.actions.scrolls) == 1


def test_production_overview_reader_caps_top_scroll_attempts(monkeypatch):
    frame = _make_structural_top_anchor()
    reader = ProductionOverviewReader(
        rects=_FakeRects(),
        capture=_FakeCapture([frame, frame, frame, frame, frame]),
        actions=_FakeActions(),
        perception=_FakePerception(),
    )

    top_anchor = reader._scroll_to_top_view()

    assert top_anchor is not None
    assert len(reader.actions.scrolls) <= 5


def test_production_overview_reader_fails_closed_when_top_never_stabilizes():
    frames = [
        Image.new("RGB", (240, 150), "#101010"),
        Image.new("RGB", (240, 150), "#202020"),
        Image.new("RGB", (240, 150), "#303030"),
        Image.new("RGB", (240, 150), "#404040"),
        Image.new("RGB", (240, 150), "#505050"),
        Image.new("RGB", (240, 150), "#606060"),
        Image.new("RGB", (240, 150), "#707070"),
    ]
    reader = ProductionOverviewReader(
        rects=_FakeRects(),
        capture=_FakeCapture(frames),
        actions=_FakeActions(),
        perception=_FakePerception(),
    )

    with pytest.raises(ValueError, match="production_top_latch_failed"):
        reader._scroll_to_top_view()


def test_production_overview_reader_structural_top_anchor_requires_layout():
    assert ProductionOverviewReader._is_structural_top_anchor(_make_structural_top_anchor()) is True
    assert ProductionOverviewReader._is_structural_top_anchor(_make_non_top_anchor()) is False


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


def test_production_overview_reader_reopens_tab_before_scroll_back(monkeypatch):
    reader = ProductionOverviewReader(
        rects=_FakeRects(),
        capture=_FakeCapture([Image.new("RGB", (240, 295), "#123456")]),
        actions=_FakeActions(),
        perception=_FakePerception(),
    )
    open_calls = []
    scroll_back_calls = []

    monkeypatch.setattr(ProductionOverviewReader, "_scroll_to_top_view", lambda self: Image.new("RGB", (240, 150), "#123456"))
    monkeypatch.setattr(ProductionOverviewReader, "_scroll_to_lower_cards", lambda self, top_anchor: None)
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_read_card",
        lambda self, **kwargs: object(),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_scroll_back_to_top",
        lambda self, *, top_anchor: scroll_back_calls.append(top_anchor),
    )

    def _open_tab():
        open_calls.append(True)
        return True

    reader.read_cards(
        tab="smelt",
        open_tab=_open_tab,
        templates={"Iron Bar": Image.new("RGB", (10, 10), "white")},
        inventory_counts={"Iron Bar": 1},
    )

    assert len(open_calls) == 2
    assert len(scroll_back_calls) == 1


def test_production_overview_reader_fails_closed_when_reopen_before_scroll_back_fails(monkeypatch):
    reader = ProductionOverviewReader(
        rects=_FakeRects(),
        capture=_FakeCapture([Image.new("RGB", (240, 295), "#123456")]),
        actions=_FakeActions(),
        perception=_FakePerception(),
    )

    monkeypatch.setattr(ProductionOverviewReader, "_scroll_to_top_view", lambda self: Image.new("RGB", (240, 150), "#123456"))
    monkeypatch.setattr(ProductionOverviewReader, "_scroll_to_lower_cards", lambda self, top_anchor: None)
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_read_card",
        lambda self, **kwargs: object(),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_scroll_back_to_top",
        lambda self, *, top_anchor: None,
    )

    open_results = iter([True, False])

    with pytest.raises(ValueError, match="open_tab_failed:smelt"):
        reader.read_cards(
            tab="smelt",
            open_tab=lambda: next(open_results),
            templates={"Iron Bar": Image.new("RGB", (10, 10), "white")},
            inventory_counts={"Iron Bar": 1},
        )


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

    assert points == ((25, 40), (20, 34), (30, 46))


def test_production_overview_reader_generates_named_tooltip_probe_points():
    points = ProductionOverviewReader._tooltip_probe_point_specs((10, 20, 40, 60))

    assert [(point.point_id, point.point) for point in points] == [
        ("center", (25, 40)),
        ("upper_left", (20, 34)),
        ("lower_right", (30, 46)),
    ]


def test_production_overview_reader_computes_local_tooltip_crop_box():
    crop_box = ProductionOverviewReader._tooltip_crop_box((60, 80, 95, 120), card_size=(240, 295))

    assert crop_box == (48, 56, 211, 140)


def test_production_overview_reader_derives_tooltip_search_region_from_icon_bounds():
    search_region = ProductionOverviewReader._tooltip_search_region((60, 80, 95, 120), card_size=(240, 295))

    assert search_region == (48, 56, 211, 140)


def test_production_overview_reader_wraps_search_region_as_single_candidate():
    candidates = ProductionOverviewReader._tooltip_crop_candidates((60, 80, 95, 120), card_size=(240, 295))

    assert [(candidate.crop_id, candidate.box) for candidate in candidates] == [
        ("search_region", (48, 56, 211, 140)),
    ]


def test_production_overview_reader_extracts_known_name_from_noisy_ocr_text():
    matches = ProductionOverviewReader._extract_tooltip_label_matches("Lead 40/1.00K 59")

    assert matches == ("Lead",)


def test_production_overview_reader_prefers_stronger_known_name_match_from_noisy_ocr_text():
    matches = ProductionOverviewReader._extract_tooltip_label_matches("Copper Bar Production")

    assert matches == ("Copper Bar",)


def test_production_overview_reader_extracts_item_name_from_broader_noisy_ocr_text():
    matches = ProductionOverviewReader._extract_tooltip_label_matches("Battery Build Crafter")

    assert matches == ("Battery",)


def test_production_overview_reader_rejects_conflicting_noisy_ocr_names():
    matches = ProductionOverviewReader._extract_tooltip_label_matches("Lead Iron")

    assert matches == ("Iron", "Lead")


def test_production_overview_reader_localizes_tighter_icon_box_inside_search_window():
    card = Image.new("RGB", (240, 295), "#173650")
    draw = ImageDraw.Draw(card)
    draw.rectangle((22, 82, 57, 118), fill="#f2c94c")
    search_box = (10, 70, 80, 140)

    localized = ProductionOverviewReader._localize_target_box(card, search_box=search_box, kind="ore")

    assert localized is not None
    assert localized != search_box
    assert localized[0] > search_box[0]
    assert localized[1] > search_box[1]
    assert localized[2] < search_box[2]
    assert localized[3] < search_box[3]
    center_x, center_y = ProductionOverviewReader._box_center(localized)
    assert 34.0 <= center_x <= 46.0
    assert 92.0 <= center_y <= 108.0


def test_production_overview_reader_localizes_small_smelt_arrow_contour():
    card = Image.new("RGB", (240, 295), "#173650")
    draw = ImageDraw.Draw(card)
    draw.line((94, 86, 101, 92), fill="#c7d0df", width=2)
    draw.line((101, 92, 94, 98), fill="#c7d0df", width=2)

    localized = ProductionOverviewReader._localize_arrow_box(card, search_box=(66, 71, 124, 103))

    assert localized is not None
    center_x, center_y = ProductionOverviewReader._box_center(localized)
    assert 92.0 <= center_x <= 103.0
    assert 86.0 <= center_y <= 98.0


def test_production_overview_reader_derives_card_regions_from_recipe_anchor(monkeypatch):
    reader = ProductionOverviewReader(
        rects=None,
        capture=None,
        actions=None,
        perception=_FakePerception(),
    )
    card = Image.new("RGB", (240, 295), "#112244")
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_detect_recipe_button_box",
        classmethod(lambda cls, image: (56, 226, 163, 271)),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_localize_arrow_box",
        classmethod(lambda cls, image, *, search_box: (90, 84, 102, 96)),
    )

    layout = reader._derive_card_layout(card=card, tab="smelt")

    assert layout.recipe_button_box == (56, 226, 163, 271)
    assert layout.input_icon_box == (14, 71, 70, 136)
    assert layout.output_icon_box == (120, 71, 182, 136)
    assert layout.arrow_search_box == (66, 71, 124, 103)
    assert layout.localized_arrow_box == (90, 84, 102, 96)
    assert layout.progress_bar_box == (31, 145, 204, 183)
    assert layout.cancel_box == (185, 145, 226, 186)


def test_production_overview_reader_generates_regions_from_card_layout(monkeypatch):
    reader = ProductionOverviewReader(
        rects=None,
        capture=None,
        actions=None,
        perception=_FakePerception(),
    )
    card = Image.new("RGB", (240, 295), "#112244")
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_detect_recipe_button_box",
        classmethod(lambda cls, image: (56, 226, 163, 271)),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_localize_arrow_box",
        classmethod(lambda cls, image, *, search_box: (103, 88, 115, 100)),
    )

    layout = reader._derive_card_layout(card=card, tab="craft")
    regions = reader._tooltip_icon_regions_from_layout(tab="craft", layout=layout)

    assert [(region.kind, region.box) for region in regions] == [
        ("output", (134, 74, 194, 145)),
        ("bar", (34, 91, 94, 165)),
        ("ore", (43, 18, 96, 71)),
    ]


def test_production_overview_reader_attempts_arrow_localization_in_bounded_center_region(monkeypatch):
    reader = ProductionOverviewReader(
        rects=None,
        capture=None,
        actions=None,
        perception=_FakePerception(),
    )
    card = Image.new("RGB", (240, 295), "#112244")
    seen = []
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_detect_recipe_button_box",
        classmethod(lambda cls, image: (56, 226, 163, 271)),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_localize_arrow_box",
        classmethod(lambda cls, image, *, search_box: seen.append(search_box) or (90, 84, 102, 96)),
    )

    layout = reader._derive_card_layout(card=card, tab="smelt")

    assert seen == [(66, 71, 124, 103)]
    assert layout.arrow_search_box == (66, 71, 124, 103)
    assert layout.localized_arrow_box == (90, 84, 102, 96)


def test_production_overview_reader_returns_search_window_and_localized_box_separately(monkeypatch):
    reader = ProductionOverviewReader(
        rects=None,
        capture=None,
        actions=None,
        perception=_FakePerception(),
    )
    card = Image.new("RGB", (240, 295), "#112244")
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_detect_recipe_button_box",
        classmethod(lambda cls, image: (56, 226, 163, 271)),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_localize_arrow_box",
        classmethod(lambda cls, image, *, search_box: (104, 84, 116, 96)),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_localize_target_box",
        classmethod(
            lambda cls, image, *, search_box, kind: (
                search_box[0] + 5,
                search_box[1] + 6,
                search_box[2] - 7,
                search_box[3] - 8,
            )
        ),
    )

    layout = reader._derive_card_layout(card=card, tab="smelt")
    targets = reader._localized_tooltip_targets_from_layout(card=card, tab="smelt", layout=layout)

    assert [(target.kind, target.search_box, target.localized_box) for target in targets] == [
        ("bar", (135, 72, 197, 137), (140, 78, 190, 129)),
        ("ore", (37, 69, 93, 116), (42, 75, 86, 108)),
    ]


def test_production_overview_reader_falls_back_to_layout_windows_when_arrow_missing(monkeypatch):
    reader = ProductionOverviewReader(
        rects=None,
        capture=None,
        actions=None,
        perception=_FakePerception(),
    )
    card = Image.new("RGB", (240, 295), "#112244")
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_detect_recipe_button_box",
        classmethod(lambda cls, image: (56, 226, 163, 271)),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_localize_arrow_box",
        classmethod(lambda cls, image, *, search_box: None),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_localize_target_box",
        classmethod(lambda cls, image, *, search_box, kind: search_box),
    )

    layout = reader._derive_card_layout(card=card, tab="smelt")
    targets = reader._localized_tooltip_targets_from_layout(card=card, tab="smelt", layout=layout)

    assert [(target.kind, target.search_box) for target in targets] == [
        ("bar", (120, 71, 182, 136)),
        ("ore", (22, 68, 78, 115)),
    ]


def test_production_overview_reader_estimates_smelt_ore_window_from_bar_when_ore_missing(monkeypatch):
    reader = ProductionOverviewReader(
        rects=None,
        capture=None,
        actions=None,
        perception=_FakePerception(),
    )
    card = Image.new("RGB", (240, 295), "#112244")
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_detect_recipe_button_box",
        classmethod(lambda cls, image: (56, 226, 163, 271)),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_localize_arrow_box",
        classmethod(lambda cls, image, *, search_box: None),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_localize_target_box",
        classmethod(
            lambda cls, image, *, search_box, kind: (156, 118, 182, 136) if kind == "bar" else None
        ),
    )

    layout = reader._derive_card_layout(card=card, tab="smelt")
    targets = reader._localized_tooltip_targets_from_layout(card=card, tab="smelt", layout=layout)

    assert [(target.kind, target.search_box, target.localized_box) for target in targets] == [
        ("bar", (120, 71, 182, 136), (156, 118, 182, 136)),
        ("ore", (26, 104, 81, 150), None),
    ]


def test_production_overview_reader_uses_localized_icon_center_after_arrow_shift(monkeypatch):
    card = Image.new("RGB", (240, 295), "#112244")
    frame = Image.new("RGB", (240, 295), "#223355")
    requested_probe_boxes = []

    class _TooltipPerception:
        def read_text(self, image, *, prompt, mode):
            _ = image, prompt, mode
            return type("Result", (), {"value": "Lead", "backend": "fake"})()

    reader = ProductionOverviewReader(
        rects=_FakeRects(),
        capture=_FakeCapture([card], screen_frames=[frame]),
        actions=_FakeActions(),
        perception=_TooltipPerception(),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_detect_recipe_button_box",
        classmethod(lambda cls, image: (56, 226, 163, 271)),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_localize_arrow_box",
        classmethod(lambda cls, image, *, search_box: (104, 84, 116, 96)),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_localize_target_box",
        classmethod(
            lambda cls, image, *, search_box, kind: (140, 76, 190, 127) if kind == "bar" else None
        ),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_tooltip_probe_point_specs",
        classmethod(
            lambda cls, icon_box: requested_probe_boxes.append(icon_box)
            or (_TooltipProbePoint(point_id="center", point=(165, 102)),)
        ),
    )

    output_name, backend = reader._probe_tooltip_identity(
        rect_key="PRODUCTION_CARD1",
        tab="smelt",
        templates={"Lead Bar": Image.new("RGB", (10, 10), "gray")},
    )

    assert output_name == "Lead Bar"
    assert backend == "tooltip_probe_bar_fake"
    assert requested_probe_boxes == [(140, 76, 190, 127)]


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
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_detect_recipe_button_box",
        classmethod(lambda cls, image: (56, 226, 163, 271)),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_localized_tooltip_targets_from_layout",
        classmethod(
            lambda cls, *, card, tab, layout: (
                _LocalizedTooltipTarget(kind="bar", search_box=(120, 71, 182, 136), localized_box=(130, 80, 170, 120)),
            )
        ),
    )

    output_name, backend = reader._probe_tooltip_identity(
        rect_key="PRODUCTION_CARD1",
        tab="smelt",
        templates={"Lead Bar": Image.new("RGB", (10, 10), "gray")},
    )

    assert output_name == "Lead Bar"
    assert backend == "tooltip_probe_bar_fake"
    assert len(reader.actions.clicks) == 2
    assert reader.perception.calls == 2


def test_production_overview_reader_stops_on_first_valid_tooltip_probe(monkeypatch):
    card = Image.new("RGB", (240, 295), "#112244")
    frame = Image.new("RGB", (240, 295), "#223355")
    crop_sequence = []

    class _TooltipPerception:
        def __init__(self):
            self.calls = 0

        def read_text(self, image, *, prompt, mode):
            _ = prompt, mode
            self.calls += 1
            crop_sequence.append(image.size)
            value = "Lead" if self.calls == 2 else "noise"
            return type("Result", (), {"value": value, "backend": "fake"})()

    reader = ProductionOverviewReader(
        rects=_FakeRects(),
        capture=_FakeCapture([card], screen_frames=[frame, frame, frame]),
        actions=_FakeActions(),
        perception=_TooltipPerception(),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_detect_recipe_button_box",
        classmethod(lambda cls, image: (56, 226, 163, 271)),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_localized_tooltip_targets_from_layout",
        classmethod(
            lambda cls, *, card, tab, layout: (
                _LocalizedTooltipTarget(kind="bar", search_box=(120, 71, 182, 136), localized_box=(130, 80, 170, 120)),
            )
        ),
    )

    output_name, backend = reader._probe_tooltip_identity(
        rect_key="PRODUCTION_CARD1",
        tab="smelt",
        templates={"Lead Bar": Image.new("RGB", (10, 10), "gray")},
    )

    assert output_name == "Lead Bar"
    assert backend == "tooltip_probe_bar_fake"
    assert crop_sequence == [(186, 84), (186, 84)]


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
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_detect_recipe_button_box",
        classmethod(lambda cls, image: (56, 226, 163, 271)),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_localized_tooltip_targets_from_layout",
        classmethod(
            lambda cls, *, card, tab, layout: (
                _LocalizedTooltipTarget(kind="bar", search_box=(120, 71, 182, 136), localized_box=(130, 80, 170, 120)),
            )
        ),
    )

    try:
        with pytest.raises(ValueError, match="tooltip_probe_no_valid_label"):
            reader._probe_tooltip_identity(
                rect_key="PRODUCTION_CARD1",
                tab="smelt",
                templates={"Lead Bar": Image.new("RGB", (10, 10), "gray")},
            )
    finally:
        monkeypatch.undo()


def test_production_overview_reader_rejects_invalid_tooltip_text_across_all_crops(monkeypatch):
    card = Image.new("RGB", (240, 295), "#112244")
    frame = Image.new("RGB", (240, 295), "#223355")

    class _TooltipPerception:
        def __init__(self):
            self.calls = 0

        def read_text(self, image, *, prompt, mode):
            _ = image, prompt, mode
            self.calls += 1
            return type("Result", (), {"value": "Inventory Editor", "backend": "fake"})()

    reader = ProductionOverviewReader(
        rects=_FakeRects(),
        capture=_FakeCapture([card], screen_frames=[frame, frame, frame]),
        actions=_FakeActions(),
        perception=_TooltipPerception(),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_detect_recipe_button_box",
        classmethod(lambda cls, image: (56, 226, 163, 271)),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_localized_tooltip_targets_from_layout",
        classmethod(
            lambda cls, *, card, tab, layout: (
                _LocalizedTooltipTarget(kind="bar", search_box=(120, 71, 182, 136), localized_box=(130, 80, 170, 120)),
            )
        ),
    )

    with pytest.raises(ValueError, match="tooltip_probe_no_valid_label"):
        reader._probe_tooltip_identity(
            rect_key="PRODUCTION_CARD1",
            tab="smelt",
            templates={"Lead Bar": Image.new("RGB", (10, 10), "gray")},
        )

    assert reader.perception.calls == 3


def test_production_overview_reader_fails_closed_without_screen_capture(monkeypatch):
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
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_detect_recipe_button_box",
        classmethod(lambda cls, image: (56, 226, 163, 271)),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_localized_tooltip_targets_from_layout",
        classmethod(
            lambda cls, *, card, tab, layout: (
                _LocalizedTooltipTarget(kind="bar", search_box=(120, 71, 182, 136), localized_box=(130, 80, 170, 120)),
            )
        ),
    )

    with pytest.raises(ValueError, match="tooltip_probe_capture_unavailable"):
        reader._probe_tooltip_identity(
            rect_key="PRODUCTION_CARD1",
            tab="smelt",
            templates={"Lead Bar": Image.new("RGB", (10, 10), "gray")},
        )


def test_production_overview_reader_fails_closed_without_card_anchor(monkeypatch):
    card = Image.new("RGB", (240, 295), "#112244")
    frame = Image.new("RGB", (240, 295), "#223355")
    reader = ProductionOverviewReader(
        rects=_FakeRects(),
        capture=_FakeCapture([card], screen_frames=[frame]),
        actions=_FakeActions(),
        perception=_FakePerception(),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_detect_recipe_button_box",
        classmethod(lambda cls, image: None),
    )

    with pytest.raises(ValueError, match="tooltip_card_anchor_unverified"):
        reader._probe_tooltip_identity(
            rect_key="PRODUCTION_CARD1",
            tab="smelt",
            templates={"Lead Bar": Image.new("RGB", (10, 10), "gray")},
        )


def test_production_overview_reader_fails_closed_without_localized_icon(monkeypatch):
    card = Image.new("RGB", (240, 295), "#112244")
    frame = Image.new("RGB", (240, 295), "#223355")
    reader = ProductionOverviewReader(
        rects=_FakeRects(),
        capture=_FakeCapture([card], screen_frames=[frame]),
        actions=_FakeActions(),
        perception=_FakePerception(),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_detect_recipe_button_box",
        classmethod(lambda cls, image: (56, 226, 163, 271)),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_localized_tooltip_targets_from_layout",
        classmethod(
            lambda cls, *, card, tab, layout: (
                _LocalizedTooltipTarget(kind="bar", search_box=(120, 71, 182, 136), localized_box=None),
            )
        ),
    )

    with pytest.raises(ValueError, match="tooltip_probe_no_valid_label"):
        reader._probe_tooltip_identity(
            rect_key="PRODUCTION_CARD1",
            tab="smelt",
            templates={"Lead Bar": Image.new("RGB", (10, 10), "gray")},
        )


def test_production_overview_reader_uses_smelt_ore_search_box_when_localization_missing(monkeypatch):
    card = Image.new("RGB", (240, 295), "#112244")
    frame = Image.new("RGB", (240, 295), "#223355")

    class _TooltipPerception:
        def read_text(self, image, *, prompt, mode):
            _ = image, prompt, mode
            return type("Result", (), {"value": "Lead 40/1.00K 59", "backend": "fake"})()

    reader = ProductionOverviewReader(
        rects=_FakeRects(),
        capture=_FakeCapture([card], screen_frames=[frame]),
        actions=_FakeActions(),
        perception=_TooltipPerception(),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_detect_recipe_button_box",
        classmethod(lambda cls, image: (56, 226, 163, 271)),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_localized_tooltip_targets_from_layout",
        classmethod(
            lambda cls, *, card, tab, layout: (
                _LocalizedTooltipTarget(kind="ore", search_box=(22, 68, 78, 115), localized_box=None),
            )
        ),
    )

    output_name, backend = reader._probe_tooltip_identity(
        rect_key="PRODUCTION_CARD1",
        tab="smelt",
        templates={"Lead Bar": Image.new("RGB", (10, 10), "gray")},
    )

    assert output_name == "Lead Bar"
    assert backend == "tooltip_probe_ore_fake"
    assert reader.actions.clicks[0] == ((50, 92), 0.35)


def test_production_overview_reader_writes_tooltip_probe_audit_artifacts(tmp_path):
    card = Image.new("RGB", (240, 295), "#112244")
    frame = Image.new("RGB", (600, 1000), "#223355")
    reader = ProductionOverviewReader(
        rects=_FakeRects(),
        capture=_FakeCapture([card], screen_frames=[frame] * 6),
        actions=_FakeActions(),
        perception=_FakePerception(),
    )
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_detect_recipe_button_box",
        classmethod(lambda cls, image: (56, 226, 163, 271)),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_localize_arrow_box",
        classmethod(lambda cls, image, *, search_box: (104, 84, 116, 96)),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_localized_tooltip_targets_from_layout",
        classmethod(
                lambda cls, *, card, tab, layout: (
                    _LocalizedTooltipTarget(kind="bar", search_box=(120, 71, 182, 136), localized_box=(128, 79, 172, 124)),
                    _LocalizedTooltipTarget(kind="ore", search_box=(22, 68, 78, 115), localized_box=(27, 73, 69, 105)),
                )
            ),
        )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_localize_target_box",
        classmethod(lambda cls, image, *, search_box, kind: (188, 147, 220, 182) if kind == "cancel" else None),
    )

    try:
        attempts = reader.audit_tooltip_probe_geometry(
            rect_key="PRODUCTION_CARD1",
            tab="smelt",
            output_dir=tmp_path,
        )
    finally:
        monkeypatch.undo()

    assert len(attempts) == 6
    assert attempts[0]["label"] == "smelt_PRODUCTION_CARD1_bar_center_search_region"
    assert attempts[0]["search_box"] == (120, 71, 182, 136)
    assert attempts[0]["localized_box"] == (128, 79, 172, 124)
    assert attempts[0]["icon_box"] == (120, 71, 182, 136)
    assert attempts[0]["tooltip_crop_box"] == (35, 52, 240, 146)
    assert attempts[0]["crop_id"] == "search_region"
    assert attempts[0]["recipe_button_box"] == (56, 226, 163, 271)
    assert attempts[0]["arrow_search_box"] == (66, 71, 124, 103)
    assert attempts[0]["localized_arrow_box"] == (104, 84, 116, 96)
    assert attempts[0]["progress_bar_box"] == (31, 145, 204, 183)
    assert attempts[0]["cancel_box"] == (185, 145, 226, 186)
    assert attempts[0]["localized_cancel_box"] == (188, 147, 220, 182)
    assert Path(str(attempts[0]["overlay_artifact"])).exists()
    assert Path(str(attempts[0]["tooltip_artifact"])).exists()


def test_production_overview_reader_renders_probe_markers_with_labels(monkeypatch):
    frame = Image.new("RGB", (600, 1000), "#223355")
    calls = []

    monkeypatch.setattr(
        ProductionOverviewReader,
        "_draw_probe_marker",
        classmethod(
            lambda cls, draw, point, *, color, marker_label: calls.append(
                {"point": point, "color": color, "marker_label": marker_label}
            )
        ),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_draw_crosshair",
        staticmethod(lambda draw, point, *, color: (_ for _ in ()).throw(AssertionError("crosshair fallback should not run"))),
    )

    overlay = ProductionOverviewReader._render_tooltip_probe_audit_overlay(
        frame=frame,
        card_rect=(35, 540, 240, 295),
        layout=_ProductionCardLayout(
            recipe_button_box=(56, 226, 163, 271),
            input_icon_box=(14, 71, 70, 136),
            output_icon_box=(120, 71, 182, 136),
            progress_bar_box=(31, 145, 204, 183),
            cancel_box=(185, 145, 226, 186),
            arrow_search_box=(66, 79, 124, 108),
            localized_arrow_box=(104, 84, 116, 96),
        ),
        search_box=(120, 71, 182, 136),
        localized_box=(128, 79, 172, 124),
        probe_point=(151, 104),
        tooltip_crop_box=(54, 64, 240, 130),
        label="smelt_PRODUCTION_CARD1_bar_center_search_region",
        probe_markers=(
            _TooltipProbeMarker(point=(151, 104), marker_label="bar_center_1", color="#ffd400"),
            _TooltipProbeMarker(point=(142, 95), marker_label="bar_upper_left_2", color="#ff9f0a"),
        ),
        localized_targets=(
            _LocalizedTooltipTarget(kind="bar", search_box=(120, 71, 182, 136), localized_box=(128, 79, 172, 124)),
        ),
        localized_cancel_box=(188, 147, 220, 182),
    )

    assert overlay.size == frame.size
    assert calls == [
        {"point": (186, 644), "color": "#ffd400", "marker_label": "bar_center_1"},
        {"point": (177, 635), "color": "#ff9f0a", "marker_label": "bar_upper_left_2"},
    ]


def test_production_overview_reader_overlay_falls_back_without_probe_markers(monkeypatch):
    frame = Image.new("RGB", (600, 1000), "#223355")
    crosshair_calls = []
    marker_calls = []

    monkeypatch.setattr(
        ProductionOverviewReader,
        "_draw_crosshair",
        staticmethod(lambda draw, point, *, color: crosshair_calls.append({"point": point, "color": color})),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_draw_probe_marker",
        classmethod(lambda cls, draw, point, *, color, marker_label: marker_calls.append(marker_label)),
    )

    overlay = ProductionOverviewReader._render_tooltip_probe_audit_overlay(
        frame=frame,
        card_rect=(35, 540, 240, 295),
        layout=_ProductionCardLayout(
            recipe_button_box=(56, 226, 163, 271),
            input_icon_box=(14, 71, 70, 136),
            output_icon_box=(120, 71, 182, 136),
            progress_bar_box=(31, 145, 204, 183),
            cancel_box=(185, 145, 226, 186),
            arrow_search_box=(66, 79, 124, 108),
            localized_arrow_box=(104, 84, 116, 96),
        ),
        search_box=(120, 71, 182, 136),
        localized_box=(128, 79, 172, 124),
        probe_point=(151, 104),
        tooltip_crop_box=(54, 64, 240, 130),
        label="smelt_PRODUCTION_CARD1_bar_center_search_region",
    )

    assert overlay.size == frame.size
    assert crosshair_calls == [{"point": (186, 644), "color": "#ff3b30"}]
    assert marker_calls == []


def test_production_overview_reader_overlay_distinguishes_search_and_localized_boxes():
    frame = Image.new("RGB", (400, 400), "#223355")

    overlay = ProductionOverviewReader._render_tooltip_probe_audit_overlay(
        frame=frame,
        card_rect=(20, 20, 200, 200),
        layout=_ProductionCardLayout(
            recipe_button_box=(60, 150, 140, 185),
            input_icon_box=(20, 40, 70, 95),
            output_icon_box=(120, 40, 180, 100),
            progress_bar_box=(30, 110, 170, 130),
            cancel_box=(160, 105, 190, 135),
            arrow_search_box=(80, 55, 122, 82),
            localized_arrow_box=(94, 60, 108, 74),
        ),
        search_box=(120, 40, 180, 100),
        localized_box=(132, 50, 166, 86),
        probe_point=(149, 68),
        tooltip_crop_box=(160, 45, 200, 95),
        label="overlay_test",
        localized_targets=(
            _LocalizedTooltipTarget(kind="output", search_box=(120, 40, 180, 100), localized_box=(132, 50, 166, 86)),
        ),
        localized_cancel_box=(165, 108, 184, 128),
    )

    assert overlay.getpixel((140, 60)) == (255, 224, 102)
    assert overlay.getpixel((152, 70)) == (61, 220, 255)
    assert overlay.getpixel((100, 75)) == (199, 125, 255)
    assert overlay.getpixel((114, 80)) == (255, 92, 255)


def test_production_overview_reader_audit_records_multiple_probe_markers(monkeypatch, tmp_path):
    card = Image.new("RGB", (240, 295), "#112244")
    frame = Image.new("RGB", (600, 1000), "#223355")
    reader = ProductionOverviewReader(
        rects=_FakeRects(),
        capture=_FakeCapture([card], screen_frames=[frame] * 4),
        actions=_FakeActions(),
        perception=_FakePerception(),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_detect_recipe_button_box",
        classmethod(lambda cls, image: (56, 226, 163, 271)),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_localize_arrow_box",
        classmethod(lambda cls, image, *, search_box: (104, 84, 116, 96)),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_localized_tooltip_targets_from_layout",
        classmethod(
            lambda cls, *, card, tab, layout: (
                _LocalizedTooltipTarget(kind="bar", search_box=(120, 71, 182, 136), localized_box=(128, 79, 172, 124)),
            )
        ),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_tooltip_probe_point_specs",
        classmethod(
            lambda cls, icon_box: (
                _TooltipProbePoint(point_id="center", point=(151, 104)),
                _TooltipProbePoint(point_id="lower_right", point=(160, 113)),
            )
        ),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_tooltip_crop_candidates",
        classmethod(
            lambda cls, icon_box, *, card_size: (
                _TooltipCropCandidate(crop_id="search_region", box=(54, 64, 240, 130)),
            )
        ),
    )
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_localize_target_box",
        classmethod(lambda cls, image, *, search_box, kind: (188, 147, 220, 182) if kind == "cancel" else None),
    )

    attempts = reader.audit_tooltip_probe_geometry(
        rect_key="PRODUCTION_CARD1",
        tab="smelt",
        output_dir=tmp_path,
    )

    assert len(attempts) == 2
    assert attempts[0]["probe_markers"] == (
        {"point": (151, 104), "marker_label": "bar_center_1", "color": "#ffd400"},
    )
    assert attempts[1]["probe_markers"] == (
        {"point": (151, 104), "marker_label": "bar_center_1", "color": "#ffd400"},
        {"point": (160, 113), "marker_label": "bar_lower_right_2", "color": "#ff9f0a"},
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
