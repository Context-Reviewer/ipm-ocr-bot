from __future__ import annotations

from dataclasses import dataclass
import re

from .. import perception as perception_backend
from ..config import RuntimeConfig
from ..rects import RectStore
from ..state import PlanetPanelState
from .common import parse_compact_number, parse_int
from .panel_text import ParsedPlanetPanel, parse_planet_panel_text


@dataclass(slots=True)
class PlanetPanelReader:
    config: RuntimeConfig
    rects: RectStore
    capture: object
    perception: object

    def _capture_key(self, key: str):
        rect = self.rects.get(key)
        if rect is None:
            return None
        return self.capture.capture_client_bbox(rect)

    def _read_key_text(self, key: str, *, prompt: str, mode: str) -> tuple[str, str]:
        image = self._capture_key(key)
        if image is None:
            return "", ""
        result = self.perception.read_text(image, prompt=prompt, mode=mode)
        return result.value.strip(), result.backend

    @staticmethod
    def _panel_has_enough_data(panel: ParsedPlanetPanel) -> bool:
        cost_count = sum(
            1
            for value in (panel.mining_cost, panel.speed_cost, panel.cargo_cost)
            if value is not None
        )
        return bool(panel.title and cost_count >= 2)

    @staticmethod
    def _merge_panels(base: ParsedPlanetPanel, override: ParsedPlanetPanel) -> ParsedPlanetPanel:
        return ParsedPlanetPanel(
            title=override.title or base.title,
            planet_id=override.planet_id if override.planet_id is not None else base.planet_id,
            mining_level=override.mining_level if override.mining_level is not None else base.mining_level,
            speed_level=override.speed_level if override.speed_level is not None else base.speed_level,
            cargo_level=override.cargo_level if override.cargo_level is not None else base.cargo_level,
            mining_cost=override.mining_cost if override.mining_cost is not None else base.mining_cost,
            speed_cost=override.speed_cost if override.speed_cost is not None else base.speed_cost,
            cargo_cost=override.cargo_cost if override.cargo_cost is not None else base.cargo_cost,
        )

    def _panel_from_openai(self, image) -> tuple[ParsedPlanetPanel, str]:
        structured = perception_backend.read_planet_panel_json(self.perception, image)
        if structured is None:
            return ParsedPlanetPanel(), ""
        title = structured.planet_name or ""
        match = re.search(r"^\s*(\d+)", title)
        return (
            ParsedPlanetPanel(
                title=title,
                planet_id=int(match.group(1)) if match else None,
                mining_cost=parse_compact_number(structured.upgrades.mining_cost),
                speed_cost=parse_compact_number(structured.upgrades.speed_cost),
                cargo_cost=parse_compact_number(structured.upgrades.cargo_cost),
            ),
            structured.backend,
        )

    def _read_panel(self) -> tuple[ParsedPlanetPanel, str]:
        image = self._capture_key("PLANET_PANEL_TEXT")
        if image is None:
            return ParsedPlanetPanel(), ""

        title_backend = ""
        windows_result = perception_backend.read_text_from_backends(
            self.perception,
            image,
            prompt=self.config.perception.prompt_planet_panel,
            mode="planet_panel",
            allowed_backend_names=("windows",),
        )
        panel = parse_planet_panel_text(windows_result.value) if windows_result.value else ParsedPlanetPanel()
        if panel.title:
            title_backend = windows_result.backend
        if self._panel_has_enough_data(panel):
            return panel, title_backend

        try:
            openai_panel, openai_backend = self._panel_from_openai(image)
        except perception_backend.StructuredPerceptionError:
            openai_panel, openai_backend = (ParsedPlanetPanel(), "")
        if openai_panel.title or any(
            value is not None for value in (openai_panel.mining_cost, openai_panel.speed_cost, openai_panel.cargo_cost)
        ):
            panel = self._merge_panels(panel, openai_panel)
            title_backend = openai_backend or title_backend
        if self._panel_has_enough_data(panel):
            return panel, title_backend

        legacy_result = perception_backend.read_text_from_backends(
            self.perception,
            image,
            prompt=self.config.perception.prompt_planet_panel,
            mode="planet_panel",
            allowed_backend_names=("legacy",),
        )
        legacy_panel = parse_planet_panel_text(legacy_result.value) if legacy_result.value else ParsedPlanetPanel()
        if legacy_panel.title or any(
            value is not None for value in (legacy_panel.mining_cost, legacy_panel.speed_cost, legacy_panel.cargo_cost)
        ):
            panel = self._merge_panels(panel, legacy_panel)
            title_backend = title_backend or legacy_result.backend
        return panel, title_backend

    def read(self) -> PlanetPanelState:
        parsed_panel, title_backend = self._read_panel()

        title_text = parsed_panel.title
        if not title_text:
            title_text, title_backend = self._read_key_text(
                "PLANET_TITLE",
                prompt=self.config.perception.prompt_planet_title,
                mode="planet_title",
            )

        mining_level = parsed_panel.mining_level
        if mining_level is None:
            mining_text, _ = self._read_key_text(
                "MINING_LVL",
                prompt=self.config.perception.prompt_numeric,
                mode="numeric",
            )
            mining_level = parse_int(mining_text)

        speed_level = parsed_panel.speed_level
        if speed_level is None:
            speed_text, _ = self._read_key_text(
                "SHIP_LVL",
                prompt=self.config.perception.prompt_numeric,
                mode="numeric",
            )
            speed_level = parse_int(speed_text)

        cargo_level = parsed_panel.cargo_level
        if cargo_level is None:
            cargo_text, _ = self._read_key_text(
                "CARGO_LVL",
                prompt=self.config.perception.prompt_numeric,
                mode="numeric",
            )
            cargo_level = parse_int(cargo_text)

        mining_cost = parsed_panel.mining_cost
        if mining_cost is None:
            mining_cost_text, _ = self._read_key_text(
                "UPGRADE_MINING",
                prompt=self.config.perception.prompt_numeric,
                mode="numeric",
            )
            mining_cost = parse_compact_number(mining_cost_text)

        speed_cost = parsed_panel.speed_cost
        if speed_cost is None:
            speed_cost_text, _ = self._read_key_text(
                "UPGRADE_SPEED",
                prompt=self.config.perception.prompt_numeric,
                mode="numeric",
            )
            speed_cost = parse_compact_number(speed_cost_text)

        cargo_cost = parsed_panel.cargo_cost
        if cargo_cost is None:
            cargo_cost_text, _ = self._read_key_text(
                "UPGRADE_CARGO",
                prompt=self.config.perception.prompt_numeric,
                mode="numeric",
            )
            cargo_cost = parse_compact_number(cargo_cost_text)

        match = re.search(r"^\s*(\d+)", title_text)
        planet_id = parsed_panel.planet_id or (int(match.group(1)) if match else None)
        return PlanetPanelState(
            planet_id=planet_id,
            title=title_text,
            mining_level=mining_level,
            speed_level=speed_level,
            cargo_level=cargo_level,
            mining_cost=mining_cost,
            speed_cost=speed_cost,
            cargo_cost=cargo_cost,
            title_backend=title_backend,
        )
