from pathlib import Path

from PIL import Image

import ocr
import template_number_reader


ASSET_DIR = Path(__file__).resolve().parent.parent / "src" / "assets" / "text_samples"


def test_level_templates_read_known_samples():
    cases = {
        "level_23.png": 23,
        "level_12.png": 12,
        "level_14.png": 14,
        "level_18.png": 18,
        "level_27.png": 27,
    }
    for name, expected in cases.items():
        img = Image.open(ASSET_DIR / name)
        text, score = template_number_reader.read_text(img, mode="level")
        assert text == str(expected)
        assert score >= 0.58
        assert ocr.parse_compact_number_for_mode(text, mode="level") == expected


def test_ore_templates_read_known_samples():
    cases = {
        "ore_67021K.png": ("67021K", 670_210),
        "ore_722K.png": ("722K", 722_000),
        "ore_4305K.png": ("4305K", 43_050),
    }
    for name, (expected_text, expected_value) in cases.items():
        img = Image.open(ASSET_DIR / name)
        text, score = template_number_reader.read_text(img, mode="ore_qty")
        assert text == expected_text
        assert score >= 0.58
        assert ocr.parse_compact_number_for_mode(text, mode="ore_qty") == expected_value


def test_planet_title_template_bank_contains_letters_and_dot():
    bank = template_number_reader._template_bank()
    assert "A" in bank
    assert "." in bank
    assert bank["A"]
    assert bank["."]
