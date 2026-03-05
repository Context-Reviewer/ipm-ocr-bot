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
