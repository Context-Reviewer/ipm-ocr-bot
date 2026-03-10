from __future__ import annotations

from dataclasses import dataclass
import re

from .. import perception as perception_backend
from ..config import RuntimeConfig
from ..domain_data import PLANET_NAMES, normalize_planet_name
from ..rects import RectStore
from ..state import PlanetPanelState
from .common import parse_compact_number, parse_int
from .panel_text import ParsedPlanetPanel, parse_planet_panel_text

_PLANET_ID_BY_NAME = {name: index + 1 for index, name in enumerate(PLANET_NAMES)}


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

    def _sanitize_title(self, value: str | None) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            return ""
        normalized = " ".join(cleaned.lower().split())
        if any(
            token in normalized
            for token in (
                "the visible planet title is",
                "visible planet title is",
                "planet title is",
                "title is",
            )
        ):
            print(f"[PERCEPTION] planet_panel semantic_reject reason=invalid_title_prose value={cleaned!r}")
            return ""
        return cleaned

    def _sanitize_level(self, value: int | None, *, field_name: str) -> int | None:
        if value is None:
            return None
        minimum = int(self.config.perception.semantic_level_min)
        maximum = int(self.config.perception.semantic_level_max)
        if value < minimum or value > maximum:
            print(f"[PERCEPTION] planet_panel semantic_reject reason=implausible_level field={field_name} value={value}")
            return None
        return value

    def _sanitize_cost(self, value: int | None, *, field_name: str) -> int | None:
        if value is None:
            return None
        minimum = int(self.config.perception.semantic_upgrade_cost_min)
        maximum = int(self.config.perception.semantic_upgrade_cost_max)
        if value < minimum or value > maximum:
            print(f"[PERCEPTION] planet_panel semantic_reject reason=implausible_cost field={field_name} value={value}")
            return None
        return value

    @staticmethod
    def _scan_seed_title_is_trustworthy(panel: ParsedPlanetPanel) -> bool:
        canonical_name = normalize_planet_name(panel.title)
        if not canonical_name:
            return False
        if panel.planet_id is None:
            return True
        expected_id = _PLANET_ID_BY_NAME.get(canonical_name)
        return expected_id is None or int(panel.planet_id) == int(expected_id)

    def _panel_from_openai(self, image) -> tuple[ParsedPlanetPanel, str]:
        structured = perception_backend.read_planet_panel_json(self.perception, image)
        if structured is None:
            return ParsedPlanetPanel(), ""
        title = self._sanitize_title(structured.planet_name)
        match = re.search(r"^\s*(\d+)", title)
        return (
            ParsedPlanetPanel(
                title=title,
                planet_id=int(match.group(1)) if match else None,
                mining_cost=self._sanitize_cost(parse_compact_number(structured.upgrades.mining_cost), field_name="mining_cost"),
                speed_cost=self._sanitize_cost(parse_compact_number(structured.upgrades.speed_cost), field_name="speed_cost"),
                cargo_cost=self._sanitize_cost(parse_compact_number(structured.upgrades.cargo_cost), field_name="cargo_cost"),
            ),
            structured.backend,
        )

    def _read_scan_seed(self) -> tuple[ParsedPlanetPanel, str]:
        title_text, title_backend = self._read_key_text(
            "PLANET_TITLE",
            prompt=self.config.perception.prompt_planet_title,
            mode="planet_title",
        )
        title_text = self._sanitize_title(title_text)
        match = re.search(r"^\s*(\d+)", title_text)

        mining_cost_text, _ = self._read_key_text(
            "UPGRADE_MINING",
            prompt=self.config.perception.prompt_numeric,
            mode="numeric",
        )
        speed_cost_text, _ = self._read_key_text(
            "UPGRADE_SPEED",
            prompt=self.config.perception.prompt_numeric,
            mode="numeric",
        )
        cargo_cost_text, _ = self._read_key_text(
            "UPGRADE_CARGO",
            prompt=self.config.perception.prompt_numeric,
            mode="numeric",
        )
        return (
            ParsedPlanetPanel(
                title=title_text,
                planet_id=int(match.group(1)) if match else None,
                mining_cost=self._sanitize_cost(parse_compact_number(mining_cost_text), field_name="mining_cost"),
                speed_cost=self._sanitize_cost(parse_compact_number(speed_cost_text), field_name="speed_cost"),
                cargo_cost=self._sanitize_cost(parse_compact_number(cargo_cost_text), field_name="cargo_cost"),
            ),
            title_backend,
        )

    def _read_scan_title_retry(self) -> tuple[str, str]:
        image = self._capture_key("PLANET_TITLE")
        if image is None:
            return "", ""
        result = perception_backend.read_text_from_backends(
            self.perception,
            image,
            prompt=self.config.perception.prompt_planet_title,
            mode="planet_title",
            allowed_backend_names=("legacy",),
        )
        return self._sanitize_title(result.value), result.backend

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
        panel = ParsedPlanetPanel(
            title=self._sanitize_title(panel.title),
            planet_id=panel.planet_id,
            mining_level=self._sanitize_level(panel.mining_level, field_name="mining_level"),
            speed_level=self._sanitize_level(panel.speed_level, field_name="speed_level"),
            cargo_level=self._sanitize_level(panel.cargo_level, field_name="cargo_level"),
            mining_cost=self._sanitize_cost(panel.mining_cost, field_name="mining_cost"),
            speed_cost=self._sanitize_cost(panel.speed_cost, field_name="speed_cost"),
            cargo_cost=self._sanitize_cost(panel.cargo_cost, field_name="cargo_cost"),
        )
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
        legacy_panel = ParsedPlanetPanel(
            title=self._sanitize_title(legacy_panel.title),
            planet_id=legacy_panel.planet_id,
            mining_level=self._sanitize_level(legacy_panel.mining_level, field_name="mining_level"),
            speed_level=self._sanitize_level(legacy_panel.speed_level, field_name="speed_level"),
            cargo_level=self._sanitize_level(legacy_panel.cargo_level, field_name="cargo_level"),
            mining_cost=self._sanitize_cost(legacy_panel.mining_cost, field_name="mining_cost"),
            speed_cost=self._sanitize_cost(legacy_panel.speed_cost, field_name="speed_cost"),
            cargo_cost=self._sanitize_cost(legacy_panel.cargo_cost, field_name="cargo_cost"),
        )
        if legacy_panel.title or any(
            value is not None for value in (legacy_panel.mining_cost, legacy_panel.speed_cost, legacy_panel.cargo_cost)
        ):
            panel = self._merge_panels(panel, legacy_panel)
            title_backend = title_backend or legacy_result.backend
        return panel, title_backend

    def _read_state(self, *, scan_cash: int | None = None) -> PlanetPanelState:
        if scan_cash is None:
            parsed_panel, title_backend = self._read_panel()
        else:
            parsed_panel, title_backend = self._read_scan_seed()
            scan_costs = tuple(
                value
                for value in (parsed_panel.mining_cost, parsed_panel.speed_cost, parsed_panel.cargo_cost)
                if value is not None
            )
            seed_is_enough = self._panel_has_enough_data(parsed_panel)
            clearly_unaffordable = bool(scan_costs) and all(int(cost) > int(scan_cash) for cost in scan_costs)
            title_is_trustworthy = self._scan_seed_title_is_trustworthy(parsed_panel)
            if seed_is_enough and clearly_unaffordable and not title_is_trustworthy:
                retry_title, retry_backend = self._read_scan_title_retry()
                if retry_title:
                    retry_match = re.search(r"^\s*(\d+)", retry_title)
                    parsed_panel = ParsedPlanetPanel(
                        title=retry_title,
                        planet_id=int(retry_match.group(1)) if retry_match else None,
                        mining_cost=parsed_panel.mining_cost,
                        speed_cost=parsed_panel.speed_cost,
                        cargo_cost=parsed_panel.cargo_cost,
                    )
                    title_backend = retry_backend or title_backend
                    title_is_trustworthy = self._scan_seed_title_is_trustworthy(parsed_panel)
            if not (seed_is_enough and title_is_trustworthy and clearly_unaffordable):
                panel_parse, panel_title_backend = self._read_panel()
                parsed_panel = panel_parse
                title_backend = panel_title_backend

        title_text = parsed_panel.title
        if not title_text:
            title_text, title_backend = self._read_key_text(
                "PLANET_TITLE",
                prompt=self.config.perception.prompt_planet_title,
                mode="planet_title",
            )
            title_text = self._sanitize_title(title_text)

        mining_level = parsed_panel.mining_level
        if scan_cash is None and mining_level is None:
            mining_text, _ = self._read_key_text(
                "MINING_LVL",
                prompt=self.config.perception.prompt_numeric,
                mode="numeric",
            )
            mining_level = self._sanitize_level(parse_int(mining_text), field_name="mining_level")

        speed_level = parsed_panel.speed_level
        if scan_cash is None and speed_level is None:
            speed_text, _ = self._read_key_text(
                "SHIP_LVL",
                prompt=self.config.perception.prompt_numeric,
                mode="numeric",
            )
            speed_level = self._sanitize_level(parse_int(speed_text), field_name="speed_level")

        cargo_level = parsed_panel.cargo_level
        if scan_cash is None and cargo_level is None:
            cargo_text, _ = self._read_key_text(
                "CARGO_LVL",
                prompt=self.config.perception.prompt_numeric,
                mode="numeric",
            )
            cargo_level = self._sanitize_level(parse_int(cargo_text), field_name="cargo_level")

        mining_cost = parsed_panel.mining_cost
        if mining_cost is None:
            mining_cost_text, _ = self._read_key_text(
                "UPGRADE_MINING",
                prompt=self.config.perception.prompt_numeric,
                mode="numeric",
            )
            mining_cost = self._sanitize_cost(parse_compact_number(mining_cost_text), field_name="mining_cost")

        speed_cost = parsed_panel.speed_cost
        if speed_cost is None:
            speed_cost_text, _ = self._read_key_text(
                "UPGRADE_SPEED",
                prompt=self.config.perception.prompt_numeric,
                mode="numeric",
            )
            speed_cost = self._sanitize_cost(parse_compact_number(speed_cost_text), field_name="speed_cost")

        cargo_cost = parsed_panel.cargo_cost
        if cargo_cost is None:
            cargo_cost_text, _ = self._read_key_text(
                "UPGRADE_CARGO",
                prompt=self.config.perception.prompt_numeric,
                mode="numeric",
            )
            cargo_cost = self._sanitize_cost(parse_compact_number(cargo_cost_text), field_name="cargo_cost")

        costs = tuple(value for value in (mining_cost, speed_cost, cargo_cost) if value is not None)
        needs_level_reads = scan_cash is None or any(int(cost) <= int(scan_cash) for cost in costs)

        if scan_cash is not None and needs_level_reads and mining_level is None:
            mining_text, _ = self._read_key_text(
                "MINING_LVL",
                prompt=self.config.perception.prompt_numeric,
                mode="numeric",
            )
            mining_level = self._sanitize_level(parse_int(mining_text), field_name="mining_level")

        if scan_cash is not None and needs_level_reads and speed_level is None:
            speed_text, _ = self._read_key_text(
                "SHIP_LVL",
                prompt=self.config.perception.prompt_numeric,
                mode="numeric",
            )
            speed_level = self._sanitize_level(parse_int(speed_text), field_name="speed_level")

        if scan_cash is not None and needs_level_reads and cargo_level is None:
            cargo_text, _ = self._read_key_text(
                "CARGO_LVL",
                prompt=self.config.perception.prompt_numeric,
                mode="numeric",
            )
            cargo_level = self._sanitize_level(parse_int(cargo_text), field_name="cargo_level")

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

    def read(self) -> PlanetPanelState:
        return self._read_state()

    def read_for_scan(self, *, cash: int | None = None) -> PlanetPanelState:
        return self._read_state(scan_cash=cash)
