from __future__ import annotations

from dataclasses import dataclass

from .. import perception as perception_backend
from ..config import RuntimeConfig
from ..rects import RectStore
from ..state import OreRowState
from .common import parse_alpha_label, parse_compact_number
from .panel_text import normalize_ore_name, parse_ore_panel_text


@dataclass(slots=True)
class OrePanelReader:
    config: RuntimeConfig
    rects: RectStore
    capture: object
    perception: object

    def _capture_key(self, key: str):
        rect = self.rects.get(key)
        if rect is None:
            return None
        return self.capture.capture_client_bbox(rect)

    def _read_text(self, key: str, *, prompt: str, mode: str) -> tuple[str, str]:
        image = self._capture_key(key)
        if image is None:
            return "", ""
        result = self.perception.read_text(image, prompt=prompt, mode=mode)
        return result.value.strip(), result.backend

    def _usable_panel_rows(self, rows: dict[int, OreRowState]) -> bool:
        populated = sum(1 for row in rows.values() if row.ore_name and row.quantity is not None)
        required = 1 if self.config.visible_ore_rows <= 1 else 2
        return populated >= required

    def _rows_from_text(self, text: str, backend: str) -> dict[int, OreRowState]:
        parsed_rows = parse_ore_panel_text(
            text,
            visible_rows=self.config.visible_ore_rows,
            known_names=self.config.policy.known_ore_names,
        )
        rows: dict[int, OreRowState] = {}
        for row_index, parsed in enumerate(parsed_rows, start=1):
            rows[row_index] = OreRowState(
                ore_name=parsed.ore_name,
                quantity=parsed.quantity,
                selected=False,
                backend=backend,
            )
        return rows

    def _rows_from_openai(self, image) -> dict[int, OreRowState]:
        structured = perception_backend.read_ore_panel_json(self.perception, image)
        if structured is None:
            return {}
        rows: dict[int, OreRowState] = {}
        for row_index, entry in enumerate(structured.ores[: self.config.visible_ore_rows], start=1):
            ore_name = normalize_ore_name(entry.name) or entry.name.strip()
            quantity = parse_compact_number(entry.quantity)
            rows[row_index] = OreRowState(
                ore_name=ore_name,
                quantity=quantity,
                selected=False,
                backend=structured.backend,
            )
        return rows

    def _read_panel_rows(self) -> dict[int, OreRowState]:
        image = self._capture_key("ORES_PANEL_TEXT")
        if image is None:
            return {}

        windows_result = perception_backend.read_text_from_backends(
            self.perception,
            image,
            prompt=self.config.perception.prompt_ore_panel,
            mode="ore_panel",
            allowed_backend_names=("windows",),
        )
        rows = self._rows_from_text(windows_result.value, windows_result.backend) if windows_result.value else {}
        if self._usable_panel_rows(rows):
            return rows

        try:
            openai_rows = self._rows_from_openai(image)
        except perception_backend.StructuredPerceptionError:
            openai_rows = {}
        if self._usable_panel_rows(openai_rows):
            return openai_rows
        if openai_rows:
            rows = openai_rows

        legacy_result = perception_backend.read_text_from_backends(
            self.perception,
            image,
            prompt=self.config.perception.prompt_ore_panel,
            mode="ore_panel",
            allowed_backend_names=("legacy",),
        )
        legacy_rows = self._rows_from_text(legacy_result.value, legacy_result.backend) if legacy_result.value else {}
        if self._usable_panel_rows(legacy_rows):
            return legacy_rows
        return rows or legacy_rows

    def read_visible_rows(self) -> dict[int, OreRowState]:
        rows = self._read_panel_rows()

        for row_index in range(1, self.config.visible_ore_rows + 1):
            read_key = f"ORE_ROW{row_index}_READ"
            qty_key = f"ORE_ROW{row_index}_QTY"
            if self.rects.get(read_key) is None and self.rects.get(qty_key) is None:
                continue
            row_text, row_backend = self._read_text(
                read_key,
                prompt=self.config.perception.prompt_ore_name,
                mode="generic",
            )
            qty_text, qty_backend = self._read_text(
                qty_key,
                prompt=self.config.perception.prompt_ore_quantity,
                mode="ore_qty",
            )
            ore_name = parse_alpha_label(row_text)
            row_quantity = parse_compact_number(row_text)
            qty_quantity = parse_compact_number(qty_text)
            quantity = qty_quantity
            if row_quantity is not None:
                if quantity is None:
                    quantity = row_quantity
                else:
                    larger = max(quantity, row_quantity)
                    smaller = max(1, min(quantity, row_quantity))
                    if (larger / smaller) > 3.0:
                        quantity = row_quantity
            existing = rows.get(row_index)
            if existing is not None and existing.ore_name and existing.quantity is not None:
                continue
            rows[row_index] = OreRowState(
                ore_name=existing.ore_name if existing and existing.ore_name else ore_name,
                quantity=existing.quantity if existing and existing.quantity is not None else quantity,
                selected=False,
                backend=(existing.backend if existing and existing.backend else (qty_backend or row_backend)),
            )
        return rows
