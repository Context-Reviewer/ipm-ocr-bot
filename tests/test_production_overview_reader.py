from PIL import Image

from ipm.readers.production_overview import ProductionOverviewReader


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
    monkeypatch.setattr(ProductionOverviewReader, "_extract_output_icon", classmethod(lambda cls, image: image))
    monkeypatch.setattr(
        ProductionOverviewReader,
        "_icon_similarity",
        classmethod(lambda cls, template, target: 0.78 if template is alpha else 0.79),
    )

    name, backend = reader._resolve_output_name(
        card=card,
        templates={"Alpha": alpha, "Beta": beta},
        inventory_counts={"Alpha": 42, "Beta": 7},
        output_quantity=42,
    )

    assert name == "Alpha"
    assert backend == "icon_template_match"
