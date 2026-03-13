from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

import cv2
import numpy as np

from ..config import RuntimeConfig
from ..rects import RectStore
from ..state import InventoryRowState
from .common import parse_compact_number

_NAME_TOKEN_RE = re.compile(r"[A-Z]+")


def _ascii_upper(text: str | None) -> str:
    normalized = unicodedata.normalize("NFKD", str(text or ""))
    return normalized.encode("ascii", "ignore").decode("ascii").upper()


def _inventory_name_key(text: str | None) -> str:
    return "".join(_NAME_TOKEN_RE.findall(_ascii_upper(text)))


@dataclass(slots=True, frozen=True)
class _CatalogMatcher:
    lookup: dict[str, str]

    @classmethod
    def from_names(
        cls,
        known_names: tuple[str, ...] | list[str],
        *,
        aliases: dict[str, str] | None = None,
    ) -> _CatalogMatcher:
        lookup: dict[str, str] = {}
        for name in known_names:
            key = _inventory_name_key(name)
            if key:
                lookup[key] = str(name)
        for alias, canonical_name in (aliases or {}).items():
            key = _inventory_name_key(alias)
            if key and canonical_name:
                lookup[key] = str(canonical_name)
        return cls(lookup=lookup)

    def match(self, text: str | None) -> str:
        tokens = _NAME_TOKEN_RE.findall(_ascii_upper(text))
        best_match = ""
        for end in range(1, len(tokens) + 1):
            candidate = self.lookup.get("".join(tokens[:end]), "")
            if candidate:
                best_match = candidate
        return best_match


@dataclass(slots=True)
class InventoryPanelReader:
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

    def _usable_panel_rows(self, rows: dict[int, InventoryRowState]) -> bool:
        populated = sum(1 for row in rows.values() if row.name and row.quantity is not None)
        required = 1 if self.config.visible_ore_rows <= 1 else 2
        return populated >= required

    @staticmethod
    def _panel_structure_present(image) -> bool:
        if image is None:
            return False
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
        if gray.size == 0:
            return False
        edge = cv2.Canny(gray, 40, 120)
        dynamic_range = float(int(gray.max()) - int(gray.min()))
        dark_fraction = float(np.mean(gray <= 90))
        edge_fraction = float(np.mean(edge > 0)) if edge.size else 0.0
        return bool(
            dynamic_range >= 35.0
            and dark_fraction >= 0.45
            and edge_fraction >= 0.006
        )

    @staticmethod
    def _rows_from_text(
        text: str,
        *,
        backend: str,
        matcher: _CatalogMatcher,
        visible_rows: int,
    ) -> dict[int, InventoryRowState]:
        lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
        if not lines:
            return {}
        names: list[str] = []
        quantities: list[int] = []
        seen_names: set[str] = set()
        for line in lines:
            canonical_name = matcher.match(line)
            if canonical_name:
                if canonical_name not in seen_names:
                    names.append(canonical_name)
                    seen_names.add(canonical_name)
                continue
            if "$" in line:
                continue
            quantity = parse_compact_number(line)
            if quantity is not None:
                quantities.append(quantity)
        row_count = min(max(0, int(visible_rows)), len(names), len(quantities))
        return {
            row_index: InventoryRowState(
                name=names[row_index - 1],
                quantity=quantities[row_index - 1],
                backend=backend,
            )
            for row_index in range(1, row_count + 1)
        }

    def read_visible_rows(
        self,
        *,
        known_names: tuple[str, ...] | list[str],
        aliases: dict[str, str] | None = None,
    ) -> dict[int, InventoryRowState]:
        matcher = _CatalogMatcher.from_names(known_names, aliases=aliases)
        rows: dict[int, InventoryRowState] = {}
        panel_image = self._capture_key("ORES_PANEL_TEXT")
        if panel_image is not None:
            panel_result = self.perception.read_text(
                panel_image,
                prompt=self.config.perception.prompt_resource_panel,
                mode="generic",
            )
            rows = self._rows_from_text(
                panel_result.value,
                backend=panel_result.backend,
                matcher=matcher,
                visible_rows=self.config.visible_ore_rows,
            )

        if self._usable_panel_rows(rows):
            return rows

        for row_index in range(1, self.config.visible_ore_rows + 1):
            read_key = f"ORE_ROW{row_index}_READ"
            qty_key = f"ORE_ROW{row_index}_QTY"
            if self.rects.get(read_key) is None and self.rects.get(qty_key) is None:
                continue
            row_text, row_backend = self._read_text(
                read_key,
                prompt=self.config.perception.prompt_resource_name,
                mode="generic",
            )
            qty_text, qty_backend = self._read_text(
                qty_key,
                prompt=self.config.perception.prompt_ore_quantity,
                mode="ore_qty",
            )
            name = matcher.match(row_text)
            row_quantity = parse_compact_number(row_text)
            qty_quantity = parse_compact_number(qty_text)
            quantity = qty_quantity if qty_quantity is not None else row_quantity
            existing = rows.get(row_index)
            if existing is not None and existing.name and existing.quantity is not None:
                continue
            rows[row_index] = InventoryRowState(
                name=existing.name if existing and existing.name else name,
                quantity=existing.quantity if existing and existing.quantity is not None else quantity,
                backend=existing.backend if existing and existing.backend else (qty_backend or row_backend),
            )
        return rows
