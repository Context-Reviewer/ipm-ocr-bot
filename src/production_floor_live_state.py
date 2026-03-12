from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Callable

from PIL import ImageChops, ImageStat

import bars_data
import items_data
from ipm.domain_data import RESOURCE_ROW_NAMES, normalize_resource_row_name
from ipm.readers.inventory_panel import InventoryPanelReader

_PRODUCTION_ASSIGNMENT_BLOCKER = (
    "blocked: no calibrated production list text/row rects and no verified production assignment parser exist"
)


def _alloy_inventory_names() -> list[str]:
    names = set(bars_data.list_bars())
    known_materials = names | set(items_data.list_items()) | set(RESOURCE_ROW_NAMES)
    for item_name in items_data.list_items():
        item = items_data.get_item(item_name) or {}
        for input_name in (item.get("inputs") or {}).keys():
            if str(input_name) not in known_materials:
                names.add(str(input_name))
    return sorted(names)


def _inventory_aliases(known_names: list[str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for name in known_names:
        normalized_name = str(name).strip()
        for suffix in (" Bar", " Alloy"):
            if not normalized_name.endswith(suffix):
                continue
            base_name = normalized_name[: -len(suffix)].strip()
            normalized_base = normalize_resource_row_name(base_name)
            if normalized_base:
                alias_name = f"{normalized_base}{suffix}"
                if alias_name != normalized_name:
                    aliases[alias_name] = normalized_name
    return aliases


def _merge_inventory_rows(
    current: dict[str, int],
    page_rows: dict[int, Any],
) -> tuple[dict[str, int], int]:
    merged = dict(current)
    new_names = 0
    for row in page_rows.values():
        name = str(getattr(row, "name", "") or "").strip()
        quantity = getattr(row, "quantity", None)
        if not name or quantity is None:
            continue
        quantity_value = int(quantity)
        if name in merged and int(merged[name]) != quantity_value:
            raise ValueError(f"conflicting_quantity:{name}:{merged[name]}:{quantity_value}")
        if name not in merged:
            merged[name] = quantity_value
            new_names += 1
    return merged, new_names


@dataclass(slots=True)
class ProductionFloorLiveStateReader:
    config: object
    rects: object
    capture: object
    actions: object
    inventory_reader: InventoryPanelReader

    def _capture_anchor(self):
        rect = self.rects.get("ORES_TOP_ANCHOR")
        if rect is None:
            return None
        return self.capture.capture_client_bbox(rect)

    @staticmethod
    def _image_mean_abs_diff(previous, current) -> float:
        if previous is None or current is None:
            return 1e9
        diff = ImageChops.difference(previous.convert("L"), current.convert("L"))
        return float(ImageStat.Stat(diff).mean[0])

    def _scroll_to_top(self) -> None:
        previous = None
        stable_reads = 0
        max_attempts = 12
        for _ in range(max_attempts):
            if not self.actions.scroll_resource_list_up():
                raise ValueError("scroll_up_failed")
            current = self._capture_anchor()
            if current is None:
                raise ValueError("missing_rect:ORES_TOP_ANCHOR")
            if self._image_mean_abs_diff(previous, current) <= 1.0:
                stable_reads += 1
                if stable_reads >= 2:
                    return
            else:
                stable_reads = 0
            previous = current
        raise ValueError("top_latch_failed")

    def _read_inventory_tab(
        self,
        *,
        tab_name: str,
        open_tab: Callable[[], bool],
        known_names: list[str],
        aliases: dict[str, str] | None = None,
    ) -> dict[str, int]:
        if self.rects.get("ORES_PANEL_TEXT") is None:
            raise ValueError("missing_rect:ORES_PANEL_TEXT")
        if not open_tab():
            raise ValueError(f"open_tab_failed:{tab_name}")
        self._scroll_to_top()
        visible_rows = max(1, int(getattr(self.config, "visible_ore_rows", 1) or 1))
        max_pages = max(1, int(ceil(len(known_names) / visible_rows)) + 2)
        merged: dict[str, int] = {}
        seen_pages: set[tuple[tuple[str, int], ...]] = set()
        stagnant_pages = 0
        for page_index in range(max_pages):
            rows = self.inventory_reader.read_visible_rows(known_names=known_names, aliases=aliases)
            fingerprint = tuple(
                (str(row.name), int(row.quantity))
                for row in rows.values()
                if str(getattr(row, "name", "") or "").strip() and getattr(row, "quantity", None) is not None
            )
            if not fingerprint:
                if page_index == 0:
                    raise ValueError(f"unreadable_page:{tab_name}:1")
                break
            if fingerprint in seen_pages:
                break
            seen_pages.add(fingerprint)
            merged, new_names = _merge_inventory_rows(merged, rows)
            if len(merged) >= len(set(known_names)):
                break
            stagnant_pages = stagnant_pages + 1 if new_names == 0 else 0
            if stagnant_pages >= 2 or len(fingerprint) < visible_rows:
                break
            if not self.actions.scroll_resource_list_down():
                raise ValueError("scroll_down_failed")
        if not merged:
            raise ValueError(f"empty_inventory:{tab_name}")
        return merged

    def read(self) -> dict[str, Any]:
        alloy_names = _alloy_inventory_names()
        seam_status = {
            "active_smelter_assignments": {
                "feasible": False,
                "blocker": _PRODUCTION_ASSIGNMENT_BLOCKER,
            },
            "active_crafter_assignments": {
                "feasible": False,
                "blocker": _PRODUCTION_ASSIGNMENT_BLOCKER,
            },
            "current_bar_inventory": {
                "feasible": True,
                "blocker": "",
            },
            "current_item_inventory": {
                "feasible": True,
                "blocker": "",
            },
        }
        try:
            bars = self._read_inventory_tab(
                tab_name="bars",
                open_tab=self.actions.open_alloys_panel,
                known_names=alloy_names,
                aliases=_inventory_aliases(alloy_names),
            )
            items = self._read_inventory_tab(
                tab_name="items",
                open_tab=self.actions.open_items_panel,
                known_names=list(items_data.list_items()),
            )
            return {
                "bars": bars,
                "items": items,
                "seam_status": seam_status,
            }
        finally:
            reset_ui = getattr(self.actions, "reset_ui", None)
            if callable(reset_ui):
                reset_ui()
