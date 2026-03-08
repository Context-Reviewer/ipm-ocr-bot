import json
from types import SimpleNamespace

from PIL import Image
import pytest

from ipm.perception import OpenAIPerceptionBackend, StructuredPerceptionError


class FakeResponses:
    def __init__(self, output_text: str):
        self.output_text = output_text

    def create(self, **_kwargs):
        return SimpleNamespace(output_text=self.output_text, output=[])


class FakeClient:
    def __init__(self, output_text: str):
        self.responses = FakeResponses(output_text)


def _image():
    return Image.new("RGB", (50, 20), "white")


def test_read_ore_panel_json_preserves_valid_structured_read():
    backend = OpenAIPerceptionBackend(enabled=True)
    backend._client = FakeClient(
        '{"panel_type":"ore_panel","planet_name":"8. ACHEAON","ores":[{"name":"Silica","quantity":"2.26K","price":"$1"}]}'
    )
    result = backend.read_ore_panel_json(_image())
    assert result.panel_type == "ore_panel"
    assert result.planet_name == "8. ACHEAON"
    assert result.ores[0].name == "Silica"
    assert result.ores[0].quantity == "2.26K"


def test_read_ore_panel_json_normalizes_alias_to_canonical_ore_name():
    backend = OpenAIPerceptionBackend(enabled=True)
    backend._client = FakeClient(
        '{"panel_type":"ore_panel","planet_name":"8. ACHEAON","ores":[{"name":"Silicon","quantity":"2.26K","price":"$1"}]}'
    )
    result = backend.read_ore_panel_json(_image())
    assert result.ores[0].name == "Silica"


def test_read_ore_panel_json_accepts_evidence_backed_resource_row_name():
    backend = OpenAIPerceptionBackend(enabled=True)
    backend._client = FakeClient(
        '{"panel_type":"ore_panel","planet_name":"8. ACHEAON","ores":[{"name":"Sulfur","quantity":"2.26K","price":"$1"}]}'
    )
    result = backend.read_ore_panel_json(_image())
    assert result.ores[0].name == "Sulfur"


def test_read_planet_panel_json_preserves_valid_structured_read():
    backend = OpenAIPerceptionBackend(enabled=True)
    backend._client = FakeClient(
        (
            '{"panel_type":"planet_panel","planet_name":"8. ACHEAON","level":"7",'
            '"upgrades":{"mining_cost":"$7.84K","speed_cost":"32.11K","cargo_cost":"31.62K"},'
            '"cash":"$235"}'
        )
    )
    result = backend.read_planet_panel_json(_image())
    assert result.panel_type == "planet_panel"
    assert result.planet_name == "8. ACHEAON"
    assert result.upgrades.mining_cost == "$7.84K"
    assert result.cash == "$235"


def test_read_planet_panel_json_rejects_prose_wrapped_title():
    backend = OpenAIPerceptionBackend(enabled=True)
    backend._client = FakeClient(
        (
            '{"panel_type":"planet_panel","planet_name":"The visible planet title is Water Planet.",'
            '"level":"7","upgrades":{"mining_cost":"$7.84K","speed_cost":"32.11K","cargo_cost":"31.62K"},'
            '"cash":"$235"}'
        )
    )
    with pytest.raises(StructuredPerceptionError) as exc:
        backend.read_planet_panel_json(_image())
    assert "invalid_title_prose" in str(exc.value)


def test_read_planet_panel_json_preserves_usable_numbered_title_format():
    backend = OpenAIPerceptionBackend(enabled=True)
    backend._client = FakeClient(
        (
            '{"panel_type":"planet_panel","planet_name":"8. ACHEAON","level":"7",'
            '"upgrades":{"mining_cost":"$7.84K","speed_cost":"32.11K","cargo_cost":"31.62K"},'
            '"cash":"$235"}'
        )
    )
    result = backend.read_planet_panel_json(_image())
    assert result.planet_name == "8. ACHEAON"


def test_read_ore_panel_json_rejects_junk_ore_name():
    backend = OpenAIPerceptionBackend(enabled=True)
    backend._client = FakeClient(
        '{"panel_type":"ore_panel","planet_name":"8. ACHEAON","ores":[{"name":"The","quantity":"2.26K","price":"$1"}]}'
    )
    with pytest.raises(StructuredPerceptionError) as exc:
        backend.read_ore_panel_json(_image())
    assert "invalid_ore_name" in str(exc.value)


@pytest.mark.parametrize(
    ("raw_name", "expected_reason"),
    [
        ('The ore or resource name visible in the row is "Sulfur."', "prose_wrapper"),
        ('The resource name visible in the row is "Iron".', "prose_wrapper"),
        ("Ship Speed", "ui_text"),
        ("8.92 mkph", "digit_text"),
        ("v. 24", "digit_text"),
        ("100%", "digit_text"),
        ("Cooper 390", "digit_text"),
    ],
)
def test_read_ore_panel_json_rejects_live_resource_name_failures(raw_name: str, expected_reason: str):
    backend = OpenAIPerceptionBackend(enabled=True)
    backend._client = FakeClient(
        json.dumps(
            {
                "panel_type": "ore_panel",
                "planet_name": "8. ACHEAON",
                "ores": [{"name": raw_name, "quantity": "2.26K", "price": "$1"}],
            }
        )
    )
    with pytest.raises(StructuredPerceptionError) as exc:
        backend.read_ore_panel_json(_image())
    assert expected_reason in str(exc.value)


def test_read_ore_panel_json_rejects_empty_ore_name():
    backend = OpenAIPerceptionBackend(enabled=True)
    backend._client = FakeClient(
        '{"panel_type":"ore_panel","planet_name":"8. ACHEAON","ores":[{"name":"","quantity":"2.26K","price":"$1"}]}'
    )
    with pytest.raises(StructuredPerceptionError) as exc:
        backend.read_ore_panel_json(_image())
    assert "invalid_ore_name" in str(exc.value) or "invalid_schema" in str(exc.value)


def test_read_ore_panel_json_rejects_invalid_quantity():
    backend = OpenAIPerceptionBackend(enabled=True)
    backend._client = FakeClient(
        '{"panel_type":"ore_panel","planet_name":"8. ACHEAON","ores":[{"name":"Copper","quantity":"about 2K","price":"$1"}]}'
    )
    with pytest.raises(StructuredPerceptionError) as exc:
        backend.read_ore_panel_json(_image())
    assert "invalid_quantity" in str(exc.value)


def test_read_planet_panel_json_rejects_implausible_panel_numbers():
    backend = OpenAIPerceptionBackend(enabled=True)
    backend._client = FakeClient(
        (
            '{"panel_type":"planet_panel","planet_name":"8. ACHEAON","level":"7470",'
            '"upgrades":{"mining_cost":"47","speed_cost":"32.11K","cargo_cost":"31.62K"},'
            '"cash":"$235"}'
        )
    )
    with pytest.raises(StructuredPerceptionError) as exc:
        backend.read_planet_panel_json(_image())
    assert "implausible_level" in str(exc.value) or "implausible_cost" in str(exc.value)


def test_read_ore_panel_json_raises_on_malformed_json():
    backend = OpenAIPerceptionBackend(enabled=True)
    backend._client = FakeClient("not-json")
    with pytest.raises(StructuredPerceptionError) as exc:
        backend.read_ore_panel_json(_image())
    assert "invalid_json" in str(exc.value)
