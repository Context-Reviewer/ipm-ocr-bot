import ocr
import pytest

@pytest.mark.parametrize(
    "text,expected",
    [
        ("705.53K", 705_530),
        ("17.73M", 17_730_000),
        ("1.2B", 1_200_000_000),
        ("250T", 250_000_000_000_000),
        ("5q", 5_000_000_000_000_000),
        ("2.5Q", 2_500_000_000_000_000_000),
        ("14s", 14_000_000_000_000_000_000_000),
        ("3.2 O", 3_200_000_000_000_000_000_000_000_000),
        ("$1,234.5K", 1_234_500),
        ("  7.06K ", 7_060),
    ],
)
def test_parse_compact_number(text, expected):
    assert ocr.parse_compact_number(text) == expected


def test_parse_compact_number_invalid():
    assert ocr.parse_compact_number("O0DEFF") is None
