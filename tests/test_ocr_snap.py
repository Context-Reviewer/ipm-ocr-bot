import ocr_snap


def test_read_upgrade_button_cost_uses_expected_rect(monkeypatch):
    seen = []

    def fake_read_number(bbox, *, mode, debug_tag=None):
        seen.append((bbox, mode, debug_tag))
        return 12345

    monkeypatch.setattr(ocr_snap.ocr, "ocr_read_number", fake_read_number)

    assert ocr_snap.read_upgrade_button_cost("M") == 12345
    assert seen == [("UPGRADE_MINING", "hud_cash", "upgrade_cost_m")]


def test_read_upgrade_button_cost_rejects_unknown_stat(monkeypatch):
    monkeypatch.setattr(ocr_snap.ocr, "ocr_read_number", lambda *args, **kwargs: 999)
    assert ocr_snap.read_upgrade_button_cost("X") is None


def test_read_upgrade_button_cost_falls_back_to_perception(monkeypatch):
    monkeypatch.setattr(ocr_snap.ocr, "ocr_read_number", lambda *args, **kwargs: None)
    monkeypatch.setattr(ocr_snap.perception, "read_number_value", lambda *args, **kwargs: (43210, None))
    assert ocr_snap.read_upgrade_button_cost("C") == 43210


def test_read_hud_cash_falls_back_to_perception(monkeypatch):
    monkeypatch.setattr(ocr_snap.ocr, "ocr_read_number", lambda *args, **kwargs: None)
    monkeypatch.setattr(ocr_snap.perception, "read_number_value", lambda *args, **kwargs: (108330, None))
    assert ocr_snap.read_hud_cash() == 108330


def test_read_planet_title_id_parses_leading_title_number(monkeypatch):
    monkeypatch.setattr(ocr_snap.ocr, "ocr_read_text", lambda *args, **kwargs: "13. KRONOS")
    monkeypatch.setattr(ocr_snap.ocr, "resolve_bbox", lambda *_args, **_kwargs: None)
    assert ocr_snap.read_planet_title_id() == 13


def test_read_planet_title_id_prefers_numeric_title_crop(monkeypatch):
    monkeypatch.setattr(ocr_snap.ocr, "resolve_bbox", lambda *_args, **_kwargs: (100, 200, 150, 60))
    monkeypatch.setattr(ocr_snap.ocr, "capture_bbox", lambda *_args, **_kwargs: (None, {}))
    monkeypatch.setattr(ocr_snap.ocr, "validate_crop", lambda *_args, **_kwargs: (False, "empty"))
    monkeypatch.setattr(ocr_snap.ocr, "ocr_read_text", lambda bbox, **kwargs: "10. NOVA" if bbox == (104, 204, 46, 52) else "3. WRONG")
    monkeypatch.setattr(ocr_snap.ocr, "ocr_read_debug", lambda *args, **kwargs: {"text": ""})
    assert ocr_snap.read_planet_title_id() == 10


def test_read_planet_title_id_stable_prefers_fast_numeric_crop(monkeypatch):
    monkeypatch.setattr(ocr_snap.ocr, "resolve_bbox", lambda *_args, **_kwargs: (100, 200, 150, 60))
    monkeypatch.setattr(ocr_snap.ocr, "capture_bbox", lambda *_args, **_kwargs: ("img", {}))
    monkeypatch.setattr(ocr_snap.ocr, "validate_crop", lambda *_args, **_kwargs: (True, "ok"))
    monkeypatch.setattr(ocr_snap.template_number_reader, "read_text", lambda *_args, **_kwargs: ("10.", 0.99))

    called = {"slow": 0}

    def fake_read_planet_title_text_stable(**_kwargs):
        called["slow"] += 1
        return "9. WRONG"

    monkeypatch.setattr(ocr_snap, "read_planet_title_text_stable", fake_read_planet_title_text_stable)

    assert ocr_snap.read_planet_title_id_stable(samples=3, delay=0) == 10
    assert called["slow"] == 0


def test_normalize_planet_title_text_uppercases_and_cleans_noise():
    assert ocr_snap._normalize_planet_title_text("10. no|va\n") == "10. NOIVA"


def test_read_planet_title_id_stable_requires_repeat_when_multiple_samples(monkeypatch):
    monkeypatch.setattr(ocr_snap, "read_planet_title_text_stable", lambda **_kwargs: "")
    values = iter(["", "8. NOVA", "8. NOVA"])
    monkeypatch.setattr(ocr_snap, "_read_planet_title_number_text", lambda: next(values))
    assert ocr_snap.read_planet_title_id_stable(samples=3, delay=0) == 8


def test_read_planet_title_id_stable_rejects_conflicting_singletons(monkeypatch):
    monkeypatch.setattr(ocr_snap, "read_planet_title_text_stable", lambda **_kwargs: "")
    values = iter(["8. NOVA", "9. NOVA", "10. NOVA"])
    monkeypatch.setattr(ocr_snap, "_read_planet_title_number_text", lambda: next(values))
    assert ocr_snap.read_planet_title_id_stable(samples=3, delay=0) is None


def test_read_level_from_row_requires_lv_prefix(monkeypatch):
    monkeypatch.setattr(ocr_snap.ocr, "ocr_read_text", lambda *args, **kwargs: "1.00 > 1.00 kmph")
    monkeypatch.setattr(ocr_snap.ocr, "ocr_read_debug", lambda *args, **kwargs: {"text": "2.50 / sec"})
    assert ocr_snap._read_level_from_row((0, 0, 1, 1), "ship_row") is None


def test_read_level_from_row_parses_lv_prefix(monkeypatch):
    monkeypatch.setattr(ocr_snap.ocr, "ocr_read_text", lambda *args, **kwargs: "Lv. 27 1.00 kmph")
    assert ocr_snap._read_level_from_row((0, 0, 1, 1), "ship_row") == 27
