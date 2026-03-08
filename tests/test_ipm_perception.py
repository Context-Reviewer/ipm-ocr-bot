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


def test_read_ore_panel_json_parses_valid_openai_payload():
    backend = OpenAIPerceptionBackend(enabled=True)
    backend._client = FakeClient(
        '{"panel_type":"ore_panel","planet_name":"8. ACHEAON","ores":[{"name":"Copper","quantity":"2.26K","price":"$1"}]}'
    )
    result = backend.read_ore_panel_json(_image())
    assert result.panel_type == "ore_panel"
    assert result.planet_name == "8. ACHEAON"
    assert result.ores[0].name == "Copper"
    assert result.ores[0].quantity == "2.26K"


def test_read_planet_panel_json_parses_valid_openai_payload():
    backend = OpenAIPerceptionBackend(enabled=True)
    backend._client = FakeClient(
        (
            '{"panel_type":"planet_panel","planet_name":"8. ACHEAON","level":"7",'
            '"upgrades":{"mining_cost":"$7.84K","speed_cost":"32.11K","cargo_cost":"31.62K"},'
            '"cash":"$ 235"}'
        )
    )
    result = backend.read_planet_panel_json(_image())
    assert result.panel_type == "planet_panel"
    assert result.planet_name == "8. ACHEAON"
    assert result.upgrades.mining_cost == "$7.84K"
    assert result.cash == "$ 235"


def test_read_ore_panel_json_raises_on_malformed_json():
    backend = OpenAIPerceptionBackend(enabled=True)
    backend._client = FakeClient("not-json")
    with pytest.raises(StructuredPerceptionError) as exc:
        backend.read_ore_panel_json(_image())
    assert "invalid_json" in str(exc.value)


def test_read_planet_panel_json_raises_on_schema_failure():
    backend = OpenAIPerceptionBackend(enabled=True)
    backend._client = FakeClient('{"panel_type":"planet_panel","planet_name":"8. ACHEAON"}')
    with pytest.raises(StructuredPerceptionError) as exc:
        backend.read_planet_panel_json(_image())
    assert "schema_validation_failed" in str(exc.value)
