from PIL import Image

from ipm.config import RuntimeConfig
from ipm.perception import (
    HybridPerceptionBackend,
    OrePanelJSON,
    OrePanelRowJSON,
    PlanetPanelJSON,
    PlanetPanelUpgradesJSON,
    StructuredPerceptionError,
)
from ipm.readers.common import parse_alpha_label, parse_compact_number
from ipm.readers import OrePanelReader, PlanetPanelReader, SellDialogReader
from ipm.rects import RectStore


class FakeCapture:
    def capture_client_bbox(self, _bbox):
        return Image.new("RGB", (50, 20), "white")


class FakeBackend:
    def __init__(
        self,
        *,
        name,
        mapping=None,
        ore_json=None,
        planet_json=None,
        fail_panel_text=False,
    ):
        self.name = name
        self.mapping = mapping or {}
        self.ore_json = ore_json
        self.planet_json = planet_json
        self.fail_panel_text = fail_panel_text

    def available(self):
        return True

    def read_text(self, _image, *, prompt="", mode="generic"):
        if self.fail_panel_text and mode in {"ore_panel", "planet_panel"}:
            raise AssertionError(f"{self.name} plain-text panel path should not be called")
        value = self.mapping.get((prompt, mode), "")

        class Result:
            confidence = 1.0

            def __init__(self, value, backend):
                self.value = value
                self.backend = backend

        return Result(value, self.name)

    def read_ore_panel_json(self, _image):
        if self.ore_json is None:
            raise AssertionError("unexpected ore panel structured call")
        return self.ore_json

    def read_planet_panel_json(self, _image):
        if self.planet_json is None:
            raise AssertionError("unexpected planet panel structured call")
        return self.planet_json


class SequentialNumericBackend(FakeBackend):
    def __init__(self, *, name, numeric_values, title_value="", panel_text=""):
        super().__init__(name=name)
        self.numeric_values = list(numeric_values)
        self.title_value = title_value
        self.panel_text = panel_text

    def read_text(self, _image, *, prompt="", mode="generic"):
        if mode == "planet_panel":
            value = self.panel_text
        elif mode == "planet_title":
            value = self.title_value
        elif mode == "numeric":
            value = self.numeric_values.pop(0) if self.numeric_values else ""
        else:
            value = ""

        class Result:
            confidence = 1.0

            def __init__(self, value, backend):
                self.value = value
                self.backend = backend

        return Result(value, self.name)


class TrackingNumericBackend(FakeBackend):
    def __init__(self, *, name, mapping=None):
        super().__init__(name=name, mapping=mapping or {})
        self.calls = []

    def read_text(self, _image, *, prompt="", mode="generic"):
        self.calls.append((prompt, mode))
        return super().read_text(_image, prompt=prompt, mode=mode)


def _openai_fallback(backend):
    return HybridPerceptionBackend(
        primary=backend,
        fallback=None,
    )


def test_planet_panel_reader_windows_path_parses_basic_fields():
    cfg = RuntimeConfig()
    rects = RectStore(
        path=None,
        rects={
            "PLANET_PANEL_TEXT": (0, 0, 1, 1),
            "MINING_LVL": (0, 0, 1, 1),
            "SHIP_LVL": (0, 0, 1, 1),
            "CARGO_LVL": (0, 0, 1, 1),
        },
    )
    perception = FakeBackend(
        name="windows",
        mapping={
            (
                cfg.perception.prompt_planet_panel,
                "planet_panel",
            ): "\n".join(
                [
                    "$ 235",
                    "7.74M",
                    "8. ACHEAON",
                    "Resource Yield",
                    "Rate",
                    "Special!",
                    "Mining Rate",
                    "2.76 / sec",
                    "Ship Speed",
                    "1.45 mkph",
                    "Cargo",
                    "7",
                    "1.66/ sec 1.94K",
                    "1.11 /sec 1.78K",
                    "$7.84K",
                    "32.11K",
                    "31.62K",
                ]
            ),
            (cfg.perception.prompt_numeric, "numeric"): "14",
        },
    )
    state = PlanetPanelReader(cfg, rects, FakeCapture(), perception).read()
    assert state.planet_id == 8
    assert state.title == "8. ACHEAON"
    assert state.mining_level == 14
    assert state.speed_level == 14
    assert state.cargo_level == 7
    assert state.mining_cost == 7_840
    assert state.speed_cost == 32_110
    assert state.cargo_cost == 31_620


def test_ore_panel_reader_windows_path_reads_visible_rows():
    cfg = RuntimeConfig(visible_ore_rows=5)
    rects = RectStore(
        path=None,
        rects={
            "ORES_PANEL_TEXT": (0, 0, 1, 1),
        },
    )
    perception = FakeBackend(
        name="windows",
        mapping={
            (
                cfg.perception.prompt_ore_panel,
                "ore_panel",
            ): "\n".join(
                [
                    "Copper",
                    "Iron",
                    "Lead",
                    "Silica",
                    "Aluminum",
                    "2.26K",
                    "44.51K",
                    "28.77K",
                    "725",
                    "556",
                    "$1",
                    "$2",
                    "$8",
                    "$17",
                ]
            ),
        },
    )
    rows = OrePanelReader(cfg, rects, FakeCapture(), perception).read_visible_rows()
    assert rows[1].ore_name == "Copper"
    assert rows[1].quantity == 2_260
    assert rows[2].ore_name == "Iron"
    assert rows[2].quantity == 44_510
    assert rows[5].ore_name == "Aluminum"
    assert rows[5].quantity == 556


def test_ore_panel_reader_uses_openai_json_without_plain_text_panel_parse():
    cfg = RuntimeConfig(visible_ore_rows=2)
    rects = RectStore(
        path=None,
        rects={
            "ORES_PANEL_TEXT": (0, 0, 1, 1),
        },
    )
    perception = HybridPerceptionBackend(
        primary=FakeBackend(name="windows", mapping={(cfg.perception.prompt_ore_panel, "ore_panel"): ""}),
        fallback=_openai_fallback(
            FakeBackend(
                name="openai",
                ore_json=OrePanelJSON(
                    panel_type="ore_panel",
                    planet_name="8. ACHEAON",
                    ores=(
                        OrePanelRowJSON(name="Copper", quantity="2.26K", price="$1"),
                        OrePanelRowJSON(name="Iron", quantity="44.51K", price="$2"),
                    ),
                ),
                fail_panel_text=True,
            )
        ),
    )
    rows = OrePanelReader(cfg, rects, FakeCapture(), perception).read_visible_rows()
    assert rows[1].ore_name == "Copper"
    assert rows[1].quantity == 2_260
    assert rows[2].ore_name == "Iron"
    assert rows[2].quantity == 44_510
    assert rows[1].backend == "openai"


def test_ore_panel_reader_preserves_expanded_resource_row_name():
    cfg = RuntimeConfig(visible_ore_rows=1)
    rects = RectStore(
        path=None,
        rects={
            "ORES_PANEL_TEXT": (0, 0, 1, 1),
        },
    )
    perception = HybridPerceptionBackend(
        primary=FakeBackend(name="windows", mapping={(cfg.perception.prompt_ore_panel, "ore_panel"): ""}),
        fallback=_openai_fallback(
            FakeBackend(
                name="openai",
                ore_json=OrePanelJSON(
                    panel_type="ore_panel",
                    planet_name="8. ACHEAON",
                    ores=(
                        OrePanelRowJSON(name="Sulfur", quantity="2.26K", price="$1"),
                    ),
                ),
                fail_panel_text=True,
            )
        ),
    )
    rows = OrePanelReader(cfg, rects, FakeCapture(), perception).read_visible_rows()
    assert rows[1].ore_name == "Sulfur"
    assert rows[1].quantity == 2_260
    assert rows[1].backend == "openai"


def test_planet_panel_reader_uses_openai_json_without_plain_text_panel_parse():
    cfg = RuntimeConfig()
    rects = RectStore(
        path=None,
        rects={
            "PLANET_PANEL_TEXT": (0, 0, 1, 1),
        },
    )
    perception = HybridPerceptionBackend(
        primary=FakeBackend(name="windows", mapping={(cfg.perception.prompt_planet_panel, "planet_panel"): ""}),
        fallback=_openai_fallback(
            FakeBackend(
                name="openai",
                planet_json=PlanetPanelJSON(
                    panel_type="planet_panel",
                    planet_name="8. ACHEAON",
                    level=None,
                    upgrades=PlanetPanelUpgradesJSON(
                        mining_cost="$7.84K",
                        speed_cost="32.11K",
                        cargo_cost="31.62K",
                    ),
                    cash="$ 235",
                ),
                fail_panel_text=True,
            )
        ),
    )
    state = PlanetPanelReader(cfg, rects, FakeCapture(), perception).read()
    assert state.planet_id == 8
    assert state.title == "8. ACHEAON"
    assert state.mining_cost == 7_840
    assert state.speed_cost == 32_110
    assert state.cargo_cost == 31_620
    assert state.title_backend == "openai"


def test_planet_panel_reader_accepts_early_planet_costs_from_openai_json():
    cfg = RuntimeConfig()
    rects = RectStore(
        path=None,
        rects={
            "PLANET_PANEL_TEXT": (0, 0, 1, 1),
        },
    )
    perception = HybridPerceptionBackend(
        primary=FakeBackend(name="windows", mapping={(cfg.perception.prompt_planet_panel, "planet_panel"): ""}),
        fallback=_openai_fallback(
            FakeBackend(
                name="openai",
                planet_json=PlanetPanelJSON(
                    panel_type="planet_panel",
                    planet_name="DRAŠTA",
                    level="2",
                    upgrades=PlanetPanelUpgradesJSON(
                        mining_cost="233",
                        speed_cost="106",
                        cargo_cost="29",
                    ),
                    cash="$147",
                ),
                fail_panel_text=True,
            )
        ),
    )
    state = PlanetPanelReader(cfg, rects, FakeCapture(), perception).read()
    assert state.title == "DRAŠTA"
    assert state.mining_cost == 233
    assert state.speed_cost == 106
    assert state.cargo_cost == 29
    assert state.title_backend == "openai"


def test_planet_panel_reader_scan_mode_skips_level_reads_when_costs_are_unaffordable():
    cfg = RuntimeConfig()
    rects = RectStore(
        path=None,
        rects={
            "PLANET_PANEL_TEXT": (0, 0, 1, 1),
            "PLANET_TITLE": (0, 0, 1, 1),
            "MINING_LVL": (0, 0, 1, 1),
            "SHIP_LVL": (0, 0, 1, 1),
            "CARGO_LVL": (0, 0, 1, 1),
            "UPGRADE_MINING": (0, 0, 1, 1),
            "UPGRADE_SPEED": (0, 0, 1, 1),
            "UPGRADE_CARGO": (0, 0, 1, 1),
        },
    )
    perception = TrackingNumericBackend(
        name="windows",
        mapping={
            (cfg.perception.prompt_planet_panel, "planet_panel"): "",
            (cfg.perception.prompt_planet_title, "planet_title"): "8. ACHERON",
            (cfg.perception.prompt_numeric, "numeric"): "999999",
        },
    )
    state = PlanetPanelReader(cfg, rects, FakeCapture(), perception).read_for_scan(cash=100)
    assert state.title == "8. ACHERON"
    assert state.mining_cost == 999999
    assert state.speed_cost == 999999
    assert state.cargo_cost == 999999
    assert state.mining_level is None
    assert state.speed_level is None
    assert state.cargo_level is None
    numeric_prompts = [mode for prompt, mode in perception.calls if mode == "numeric"]
    assert numeric_prompts == ["numeric", "numeric", "numeric"]


def test_planet_panel_reader_scan_mode_prefers_direct_title_and_cost_reads():
    cfg = RuntimeConfig()
    rects = RectStore(
        path=None,
        rects={
            "PLANET_PANEL_TEXT": (0, 0, 1, 1),
            "PLANET_TITLE": (0, 0, 1, 1),
            "UPGRADE_MINING": (0, 0, 1, 1),
            "UPGRADE_SPEED": (0, 0, 1, 1),
            "UPGRADE_CARGO": (0, 0, 1, 1),
        },
    )
    perception = TrackingNumericBackend(
        name="windows",
        mapping={
            (cfg.perception.prompt_planet_title, "planet_title"): "8. ACHERON",
            (cfg.perception.prompt_numeric, "numeric"): "999999",
            (cfg.perception.prompt_planet_panel, "planet_panel"): "should not be used",
        },
    )
    state = PlanetPanelReader(cfg, rects, FakeCapture(), perception).read_for_scan(cash=100)
    assert state.planet_id == 8
    assert state.title == "8. ACHERON"
    assert state.mining_cost == 999999
    assert state.speed_cost == 999999
    assert state.cargo_cost == 999999
    assert state.mining_level is None
    assert state.speed_level is None
    assert state.cargo_level is None
    assert ("planet_panel" not in [mode for _prompt, mode in perception.calls])


def test_planet_panel_reader_scan_mode_uses_legacy_title_retry_before_panel_parse():
    cfg = RuntimeConfig()
    rects = RectStore(
        path=None,
        rects={
            "PLANET_PANEL_TEXT": (0, 0, 1, 1),
            "PLANET_TITLE": (0, 0, 1, 1),
            "UPGRADE_MINING": (0, 0, 1, 1),
            "UPGRADE_SPEED": (0, 0, 1, 1),
            "UPGRADE_CARGO": (0, 0, 1, 1),
        },
    )
    perception = HybridPerceptionBackend(
        primary=FakeBackend(
            name="windows",
            mapping={
                (cfg.perception.prompt_planet_title, "planet_title"): "5. NEWTON",
                (cfg.perception.prompt_numeric, "numeric"): "999999",
            },
            fail_panel_text=True,
        ),
        fallback=FakeBackend(
            name="legacy",
            mapping={
                (cfg.perception.prompt_planet_title, "planet_title"): "6. NEWTON",
            },
        ),
    )
    state = PlanetPanelReader(cfg, rects, FakeCapture(), perception).read_for_scan(cash=100)
    assert state.planet_id == 6
    assert state.title == "6. NEWTON"
    assert state.mining_cost == 999999
    assert state.speed_cost == 999999
    assert state.cargo_cost == 999999


def test_planet_panel_reader_scan_mode_retries_untrusted_affordable_cost_before_panel_parse():
    cfg = RuntimeConfig()
    rects = RectStore(
        path=None,
        rects={
            "PLANET_PANEL_TEXT": (0, 0, 1, 1),
            "PLANET_TITLE": (0, 0, 1, 1),
            "UPGRADE_MINING": (0, 0, 1, 1),
            "UPGRADE_SPEED": (0, 0, 1, 1),
            "UPGRADE_CARGO": (0, 0, 1, 1),
        },
    )
    perception = HybridPerceptionBackend(
        primary=FakeBackend(
            name="windows",
            mapping={
                (cfg.perception.prompt_planet_title, "planet_title"): "5. NEWTON",
                (cfg.perception.prompt_numeric, "numeric"): "9",
            },
            fail_panel_text=True,
        ),
        fallback=FakeBackend(
            name="legacy",
            mapping={
                (cfg.perception.prompt_planet_title, "planet_title"): "6. NEWTON",
                (cfg.perception.prompt_numeric, "numeric"): "9440",
            },
        ),
    )
    state = PlanetPanelReader(cfg, rects, FakeCapture(), perception).read_for_scan(cash=187)
    assert state.planet_id == 6
    assert state.title == "6. NEWTON"
    assert state.mining_cost == 9440
    assert state.speed_cost == 9440
    assert state.cargo_cost == 9440


def test_planet_panel_reader_scan_mode_accepts_colony_level_title_without_panel_parse():
    cfg = RuntimeConfig()
    rects = RectStore(
        path=None,
        rects={
            "PLANET_PANEL_TEXT": (0, 0, 1, 1),
            "PLANET_TITLE": (0, 0, 1, 1),
            "UPGRADE_MINING": (0, 0, 1, 1),
            "UPGRADE_SPEED": (0, 0, 1, 1),
            "UPGRADE_CARGO": (0, 0, 1, 1),
        },
    )
    perception = TrackingNumericBackend(
        name="windows",
        mapping={
            (cfg.perception.prompt_planet_title, "planet_title"): "1. BALOR Colony Lv 1",
            (cfg.perception.prompt_numeric, "numeric"): "999999",
            (cfg.perception.prompt_planet_panel, "planet_panel"): "should not be used",
        },
    )
    state = PlanetPanelReader(cfg, rects, FakeCapture(), perception).read_for_scan(cash=100)
    assert state.planet_id == 1
    assert state.title == "1. BALOR Colony Lv 1"
    assert state.mining_cost == 999999
    assert state.speed_cost == 999999
    assert state.cargo_cost == 999999
    assert ("planet_panel" not in [mode for _prompt, mode in perception.calls])


def test_planet_panel_reader_probe_mode_prefers_direct_title_and_cost_reads():
    cfg = RuntimeConfig()
    rects = RectStore(
        path=None,
        rects={
            "PLANET_PANEL_TEXT": (0, 0, 1, 1),
            "PLANET_TITLE": (0, 0, 1, 1),
            "UPGRADE_MINING": (0, 0, 1, 1),
            "UPGRADE_SPEED": (0, 0, 1, 1),
            "UPGRADE_CARGO": (0, 0, 1, 1),
        },
    )
    perception = TrackingNumericBackend(
        name="windows",
        mapping={
            (cfg.perception.prompt_planet_title, "planet_title"): "8. ACHERON",
            (cfg.perception.prompt_numeric, "numeric"): "999999",
            (cfg.perception.prompt_planet_panel, "planet_panel"): "should not be used",
        },
    )
    state = PlanetPanelReader(cfg, rects, FakeCapture(), perception).read_for_probe()
    assert state.planet_id == 8
    assert state.title == "8. ACHERON"
    assert state.mining_cost == 999999
    assert state.speed_cost == 999999
    assert state.cargo_cost == 999999
    assert state.mining_level is None
    assert state.speed_level is None
    assert state.cargo_level is None
    assert ("planet_panel" not in [mode for _prompt, mode in perception.calls])


def test_planet_panel_reader_scan_mode_falls_back_to_panel_parse_when_direct_seed_is_incomplete():
    cfg = RuntimeConfig()
    rects = RectStore(
        path=None,
        rects={
            "PLANET_PANEL_TEXT": (0, 0, 1, 1),
            "PLANET_TITLE": (0, 0, 1, 1),
            "UPGRADE_MINING": (0, 0, 1, 1),
            "UPGRADE_SPEED": (0, 0, 1, 1),
            "UPGRADE_CARGO": (0, 0, 1, 1),
        },
    )
    perception = TrackingNumericBackend(
        name="windows",
        mapping={
            (cfg.perception.prompt_planet_title, "planet_title"): "",
            (cfg.perception.prompt_numeric, "numeric"): "",
            (
                cfg.perception.prompt_planet_panel,
                "planet_panel",
            ): "\n".join(
                [
                    "$ 174",
                    "2. BALOR",
                    "Mining Rate",
                    "1.11 / sec",
                    "Ship Speed",
                    "1.45 mkph",
                    "Cargo",
                    "7",
                    "$461",
                    "$210",
                    "$210",
                ]
            ),
        },
    )
    state = PlanetPanelReader(cfg, rects, FakeCapture(), perception).read_for_scan(cash=100)
    assert state.planet_id == 2
    assert state.title == "2. BALOR"
    assert state.mining_cost == 461
    assert state.speed_cost == 210
    assert state.cargo_cost == 210
    assert ("planet_panel" in [mode for _prompt, mode in perception.calls])


def test_planet_panel_reader_scan_mode_falls_back_when_direct_title_id_is_inconsistent():
    cfg = RuntimeConfig()
    rects = RectStore(
        path=None,
        rects={
            "PLANET_PANEL_TEXT": (0, 0, 1, 1),
            "PLANET_TITLE": (0, 0, 1, 1),
            "UPGRADE_MINING": (0, 0, 1, 1),
            "UPGRADE_SPEED": (0, 0, 1, 1),
            "UPGRADE_CARGO": (0, 0, 1, 1),
        },
    )
    perception = TrackingNumericBackend(
        name="windows",
        mapping={
            (cfg.perception.prompt_planet_title, "planet_title"): "5. NEWTON",
            (cfg.perception.prompt_numeric, "numeric"): "999999",
            (
                cfg.perception.prompt_planet_panel,
                "planet_panel",
            ): "\n".join(
                [
                    "$ 174",
                    "6. NEWTON",
                    "Mining Rate",
                    "1.11 / sec",
                    "Ship Speed",
                    "1.45 mkph",
                    "Cargo",
                    "7",
                    "$9440",
                    "$1500",
                    "$1950",
                ]
            ),
        },
    )
    state = PlanetPanelReader(cfg, rects, FakeCapture(), perception).read_for_scan(cash=100)
    assert state.planet_id == 6
    assert state.title == "6. NEWTON"
    assert state.mining_cost == 9440
    assert state.speed_cost == 1500
    assert state.cargo_cost == 1950
    assert ("planet_panel" in [mode for _prompt, mode in perception.calls])


def test_ore_panel_reader_falls_through_to_legacy_after_openai_semantic_failure():
    cfg = RuntimeConfig(visible_ore_rows=1)
    rects = RectStore(
        path=None,
        rects={
            "ORES_PANEL_TEXT": (0, 0, 1, 1),
        },
    )

    class RejectingOpenAI(FakeBackend):
        def read_ore_panel_json(self, _image):
            raise StructuredPerceptionError(
                backend="openai",
                panel_type="ore_panel",
                reason="invalid_ore_name:'The'",
            )

    perception = HybridPerceptionBackend(
        primary=FakeBackend(name="windows", mapping={(cfg.perception.prompt_ore_panel, "ore_panel"): ""}),
        fallback=HybridPerceptionBackend(
            primary=RejectingOpenAI(name="openai", fail_panel_text=True),
            fallback=FakeBackend(
                name="legacy",
                mapping={
                    (cfg.perception.prompt_ore_panel, "ore_panel"): "\n".join(["Copper", "2.26K", "$1"]),
                },
            ),
        ),
    )
    rows = OrePanelReader(cfg, rects, FakeCapture(), perception).read_visible_rows()
    assert rows[1].ore_name == "Copper"
    assert rows[1].quantity == 2_260
    assert rows[1].backend == "legacy"


def test_planet_panel_reader_rejects_implausible_field_numbers():
    cfg = RuntimeConfig()
    rects = RectStore(
        path=None,
        rects={
            "PLANET_TITLE": (0, 0, 1, 1),
            "MINING_LVL": (0, 0, 1, 1),
            "SHIP_LVL": (0, 0, 1, 1),
            "CARGO_LVL": (0, 0, 1, 1),
            "UPGRADE_MINING": (0, 0, 1, 1),
            "UPGRADE_SPEED": (0, 0, 1, 1),
            "UPGRADE_CARGO": (0, 0, 1, 1),
        },
    )
    perception = SequentialNumericBackend(
        name="windows",
        title_value="8. ACHEAON",
        numeric_values=["7470", "14", "0", "0", "32.11K", "31.62K"],
    )
    state = PlanetPanelReader(cfg, rects, FakeCapture(), perception).read()
    assert state.planet_id == 8
    assert state.title == "8. ACHEAON"
    assert state.mining_level is None
    assert state.speed_level == 14
    assert state.cargo_level is None
    assert state.mining_cost is None
    assert state.speed_cost == 32_110
    assert state.cargo_cost == 31_620


def test_sell_dialog_reader_reads_selected_quantity():
    cfg = RuntimeConfig()
    rects = RectStore(
        path=None,
        rects={
            "SELL_SELECTED_QTY": (0, 0, 1, 1),
            "SELL_SLIDER_TRACK": (0, 0, 1, 1),
        },
    )
    perception = FakeBackend(
        name="windows",
        mapping={
            (cfg.perception.prompt_ore_quantity, "ore_qty"): "50K",
        },
    )
    state = SellDialogReader(cfg, rects, FakeCapture(), perception).read()
    assert state.selected_quantity == 50_000
    assert state.slider_visible is True


def test_ore_row_parsers_extract_name_and_last_quantity():
    assert parse_alpha_label("Silica 20.8 82K") == "Silica"
    assert parse_compact_number("Silica 20.8 82K") == 82_000


def test_ore_panel_reader_prefers_row_quantity_when_qty_box_is_wildly_off():
    cfg = RuntimeConfig(visible_ore_rows=1)
    rects = RectStore(
        path=None,
        rects={
            "ORE_ROW1_READ": (0, 0, 1, 1),
            "ORE_ROW1_QTY": (0, 0, 1, 1),
        },
    )
    perception = FakeBackend(
        name="windows",
        mapping={
            (cfg.perception.prompt_ore_name, "generic"): "Silica 20.8 82K",
            (cfg.perception.prompt_ore_quantity, "ore_qty"): "7.5M",
        },
    )
    rows = OrePanelReader(cfg, rects, FakeCapture(), perception).read_visible_rows()
    assert rows[1].ore_name == "Silica"
    assert rows[1].quantity == 82_000
