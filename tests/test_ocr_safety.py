import numpy as np
import ocr


def test_validate_crop_none():
    ok, reason = ocr.validate_crop(None, (0, 0, 1, 1), "hud_cash")
    assert ok is False
    assert reason in {"img_none", "img_empty", "arr_empty", "unsupported_type"}


def test_validate_crop_empty_array():
    arr = np.zeros((0, 0, 3), dtype=np.uint8)
    ok, reason = ocr.validate_crop(arr, (0, 0, 1, 1), "ore_qty")
    assert ok is False
    assert reason in {"arr_empty", "img_empty", "img_none", "unsupported_type"}


def test_read_number_stable_uses_median(monkeypatch):
    values = iter([100, 101, 100])

    def fake_read(_bbox, *, mode, debug_tag=None):
        return next(values)

    monkeypatch.setattr(ocr, "_read_number_once", fake_read)
    result = ocr._read_number_stable(
        (0, 0, 1, 1),
        mode="level",
        attempts=3,
        min_valid=2,
        max_rel_spread=0.05,
        delay=0.0,
    )
    assert result == 100


def test_read_number_stable_rejects_wide_spread(monkeypatch):
    values = iter([100, 140, 180])

    def fake_read(_bbox, *, mode, debug_tag=None):
        return next(values)

    monkeypatch.setattr(ocr, "_read_number_once", fake_read)
    result = ocr._read_number_stable(
        (0, 0, 1, 1),
        mode="hud_cash",
        attempts=3,
        min_valid=2,
        max_rel_spread=0.10,
        delay=0.0,
    )
    assert result is None


def test_ocr_read_text_stable_uses_consensus(monkeypatch):
    values = iter(["10. NOVA", "10. NOVA", "10. NOUA"])
    monkeypatch.setattr(ocr, "_read_text_once", lambda _bbox, *, mode: next(values))
    result = ocr.ocr_read_text_stable((0, 0, 1, 1), mode="planet_title", attempts=3, min_valid=2, delay=0.0)
    assert result == "10. NOVA"


def test_choose_best_text_candidate_prefers_title_with_id(monkeypatch):
    result = ocr._choose_best_text_candidate(["NOVA", "10. NOVA", "10."], mode="planet_title")
    assert result == "10. NOVA"
