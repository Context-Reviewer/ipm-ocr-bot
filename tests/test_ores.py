import ores
import numpy as np


def test_parse_qty_from_text_prefers_rightmost_compact_token():
    assert ores._parse_qty_from_text("Copper 120.00M") == 120_000_000
    assert ores._parse_qty_from_text("Lead 5T") == 5_000_000_000_000


def test_parse_qty_from_text_handles_safe_ocr_noise():
    assert ores._parse_qty_from_text("Silica 2OK") == 20_000
    assert ores._parse_qty_from_text("Iron 214.39K.") == 214_390


def test_parse_qty_from_text_returns_none_without_number():
    assert ores._parse_qty_from_text("Silica ss") is None


def test_parse_plain_digits_returns_rightmost_integer():
    assert ores._parse_plain_digits("Lead 62") == 62
    assert ores._parse_plain_digits("7 45") == 45
    assert ores._parse_plain_digits("ss") is None


def test_row_has_visible_content_false_for_dark_region(monkeypatch):
    monkeypatch.setattr(ores, "sample_rect", lambda _bbox: np.zeros((20, 40, 3), dtype=np.uint8))
    assert ores.row_has_visible_content(2) is False


def test_row_has_visible_content_true_for_bright_region(monkeypatch):
    monkeypatch.setattr(ores, "sample_rect", lambda _bbox: np.full((20, 40, 3), 90, dtype=np.uint8))
    assert ores.row_has_visible_content(2) is True


def test_read_qty_from_row_text_retries_y_offsets(monkeypatch):
    monkeypatch.setattr(ores, "row_bbox_for_row", lambda _row_index: (100, 200, 50, 20))

    def fake_read_text(bbox, *, mode):
        if bbox == (100, 200, 50, 20):
            return ""
        if bbox == (100, 194, 50, 20):
            return "Lead 48"
        return ""

    monkeypatch.setattr(ores.ocr, "ocr_read_text", fake_read_text)
    monkeypatch.setattr(ores.ocr, "ocr_read_debug", lambda _bbox, *, mode: {"text": ""})
    monkeypatch.setattr(ores.config, "OCR_QTY_Y_OFFSETS", [0, -6, 6])

    bbox, qty = ores.read_qty_from_row_text(3)
    assert bbox == (100, 194, 50, 20)
    assert qty == 48


def test_read_qty_from_row_text_falls_back_to_perception(monkeypatch):
    monkeypatch.setattr(ores, "row_bbox_for_row", lambda _row_index: (100, 200, 50, 20))
    monkeypatch.setattr(ores.ocr, "ocr_read_text", lambda *args, **kwargs: "")
    monkeypatch.setattr(ores.ocr, "ocr_read_debug", lambda *args, **kwargs: {"text": ""})
    monkeypatch.setattr(ores.ocr, "capture_bbox", lambda *_args, **_kwargs: (None, {}))
    monkeypatch.setattr(ores.perception, "read_number_value", lambda *args, **kwargs: (214_390, None))

    bbox, qty = ores.read_qty_from_row_text(3)
    assert bbox == (100, 200, 50, 20)
    assert qty == 214_390


def test_read_qty_from_row_text_stable_uses_median(monkeypatch):
    samples = iter([
        ((100, 200, 50, 20), 11120),
        ((100, 200, 50, 20), 11116),
        ((100, 200, 50, 20), 11120),
    ])
    monkeypatch.setattr(ores, "read_qty_from_row_text", lambda *args, **kwargs: next(samples))
    monkeypatch.setattr(ores.config, "ORE_ROW_TEXT_SAMPLES", 3)
    monkeypatch.setattr(ores.config, "ORE_ROW_TEXT_MIN_VALID_SAMPLES", 2)
    monkeypatch.setattr(ores.config, "ORE_ROW_TEXT_MAX_REL_SPREAD", 0.25)
    monkeypatch.setattr(ores.config, "ORE_ROW_TEXT_SAMPLE_DELAY", 0.0)
    bbox, qty = ores.read_qty_from_row_text_stable(3)
    assert bbox == (100, 200, 50, 20)
    assert qty == 11120


def test_read_qty_from_row_text_stable_rejects_wide_spread(monkeypatch):
    samples = iter([
        ((100, 200, 50, 20), 111200),
        ((100, 200, 50, 20), 11116),
        ((100, 200, 50, 20), 11120),
    ])
    monkeypatch.setattr(ores, "read_qty_from_row_text", lambda *args, **kwargs: next(samples))
    monkeypatch.setattr(ores.config, "ORE_ROW_TEXT_SAMPLES", 3)
    monkeypatch.setattr(ores.config, "ORE_ROW_TEXT_MIN_VALID_SAMPLES", 2)
    monkeypatch.setattr(ores.config, "ORE_ROW_TEXT_MAX_REL_SPREAD", 0.25)
    monkeypatch.setattr(ores.config, "ORE_ROW_TEXT_SAMPLE_DELAY", 0.0)
    _bbox, qty = ores.read_qty_from_row_text_stable(3, debug_tag="row3")
    assert qty is None


def test_choose_sell_preset_returns_largest_not_over_target():
    assert ores.choose_sell_preset(0.10) is None
    assert ores.choose_sell_preset(0.30) == "shift+;"
    assert ores.choose_sell_preset(0.80) == "shift+'"


def test_confirm_qty_for_sale_rejects_wide_spread(monkeypatch):
    samples = iter([
        ((0, 0, 1, 1), 111200),
        ((0, 0, 1, 1), 11116),
    ])
    monkeypatch.setattr(ores, "read_qty_for_row", lambda *args, **kwargs: next(samples))
    monkeypatch.setattr(ores.config, "ORE_SELL_CONFIRM_SAMPLES", 3)
    monkeypatch.setattr(ores.config, "ORE_SELL_CONFIRM_MIN_VALID_SAMPLES", 2)
    monkeypatch.setattr(ores.config, "ORE_SELL_CONFIRM_MAX_REL_SPREAD", 0.20)
    monkeypatch.setattr(ores.config, "ORE_SELL_CONFIRM_SAMPLE_DELAY", 0.0)
    qty, values, reason = ores.confirm_qty_for_sale(3, "Lead", 11120)
    assert qty is None
    assert values == [11120, 111200, 11116]
    assert reason.startswith("spread=")


def test_confirm_qty_for_sale_rejects_suspicious_jump(monkeypatch):
    samples = iter([
        ((0, 0, 1, 1), 700000),
        ((0, 0, 1, 1), 700100),
    ])
    monkeypatch.setattr(ores, "read_qty_for_row", lambda *args, **kwargs: next(samples))
    monkeypatch.setattr(ores.config, "ORE_SELL_CONFIRM_SAMPLES", 3)
    monkeypatch.setattr(ores.config, "ORE_SELL_CONFIRM_MIN_VALID_SAMPLES", 2)
    monkeypatch.setattr(ores.config, "ORE_SELL_CONFIRM_MAX_REL_SPREAD", 0.20)
    monkeypatch.setattr(ores.config, "ORE_SELL_CONFIRM_SAMPLE_DELAY", 0.0)
    monkeypatch.setattr(ores.config, "ORE_SELL_MAX_SUSPICIOUS_JUMP_RATIO", 6.0)
    monkeypatch.setattr(ores.config, "ORE_SELL_MAX_SUSPICIOUS_JUMP_ABS", 50000)
    ores._LAST_CONFIRMED_QTY_BY_ORE["Lead"] = 11120
    qty, values, reason = ores.confirm_qty_for_sale(3, "Lead", 700050)
    assert qty is None
    assert "suspicious_jump" in reason


def test_try_apply_precise_sell_fraction_returns_false_without_calibration(monkeypatch):
    monkeypatch.setattr(ores.config, "SELL_PRECISE_SLIDER_ENABLED", True)
    monkeypatch.setattr(ores.ocr, "resolve_bbox", lambda _key: None)
    assert ores.try_apply_precise_sell_fraction(0.61, 10000) is False


def test_read_selected_sell_qty_falls_back_to_perception(monkeypatch):
    monkeypatch.setattr(ores.ocr, "resolve_bbox", lambda _key: (1, 2, 3, 4))
    monkeypatch.setattr(ores.ocr, "ocr_read_number", lambda *args, **kwargs: None)
    monkeypatch.setattr(ores.ocr, "ocr_read_text", lambda *args, **kwargs: "")
    monkeypatch.setattr(ores.perception, "read_number_value", lambda *args, **kwargs: (50000, None))

    assert ores._read_selected_sell_qty() == 50000
