import config
import perception


def test_hybrid_backend_falls_back_to_legacy(monkeypatch):
    hybrid = perception.HybridPerceptionBackend()
    monkeypatch.setattr(hybrid._openai, "available", lambda: False)
    monkeypatch.setattr(hybrid._legacy, "available", lambda: True)
    monkeypatch.setattr(hybrid._legacy, "read_text", lambda *args, **kwargs: "2. DRASTA")
    monkeypatch.setattr(config, "PERCEPTION_HYBRID_ORDER", "openai_first")

    assert hybrid.read_text("PLANET_TITLE", mode="planet_title") == "2. DRASTA"


def test_read_planet_title_text_uses_configured_prompt(monkeypatch):
    monkeypatch.setattr(config, "PERCEPTION_PLANET_TITLE_PROMPT", "Read the title.")

    class FakeBackend:
        name = "fake"

        def read_text(self, bbox, *, mode="generic", prompt=None):
            assert bbox == "PLANET_TITLE"
            assert mode == "planet_title"
            assert prompt == "Read the title."
            return "7. HELIOS"

    monkeypatch.setattr(perception, "_BACKEND", FakeBackend())
    result = perception.read_planet_title_text("PLANET_TITLE")
    assert result.text == "7. HELIOS"
    assert result.backend == "fake"


def test_read_number_value_parses_compact_result(monkeypatch):
    class FakeBackend:
        name = "fake"

        def read_text(self, bbox, *, mode="generic", prompt=None):
            assert bbox == "ORE_ROW1_QTY"
            assert mode == "ore_qty"
            return "214.39K"

    monkeypatch.setattr(perception, "_BACKEND", FakeBackend())
    value, result = perception.read_number_value("ORE_ROW1_QTY", mode="ore_qty")
    assert value == 214_390
    assert result.text == "214.39K"
