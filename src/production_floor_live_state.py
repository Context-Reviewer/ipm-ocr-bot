from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
import time
from typing import Any, Callable

from PIL import ImageChops, ImageStat

import bars_data
import items_data
from ipm.domain_data import RESOURCE_ROW_NAMES, normalize_resource_row_name
from ipm.readers.inventory_panel import InventoryPanelReader
from ipm.readers.production_overview import ProductionOverviewReader
from production_overview_seams import (
    allowed_overview_outputs,
    parse_active_overview_cards,
    required_production_overview_rects,
    seam_contract_summary,
)

_INVENTORY_FIRST_PAGE_READ_ATTEMPTS = 3
_INVENTORY_PAGE_REREAD_DELAY_SECONDS = 0.12


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


def _template_bank_with_aliases(templates: dict[str, object]) -> dict[str, object]:
    augmented = dict(templates)
    for name, image in list(templates.items()):
        normalized_name = str(name or "").strip()
        if not normalized_name:
            continue
        normalized_resource = normalize_resource_row_name(normalized_name)
        if normalized_resource and normalized_resource not in augmented:
            augmented[normalized_resource] = image
        explicit_aliases = {
            "Silicon": "Silica",
            "Silica": "Silicon",
            "Aluminum": "Aluminium",
            "Aluminium": "Aluminum",
            "Aluminum Bar": "Aluminium Bar",
            "Aluminium Bar": "Aluminum Bar",
        }
        alias_name = explicit_aliases.get(normalized_name)
        if alias_name and alias_name not in augmented:
            augmented[alias_name] = image
    return augmented


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
    perception: object | None = None
    production_reader: ProductionOverviewReader | None = None

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
        template_row_names: list[str] | None = None,
    ) -> tuple[dict[str, int], dict[str, object]]:
        if self.rects.get("ORES_PANEL_TEXT") is None:
            raise ValueError("missing_rect:ORES_PANEL_TEXT")
        if not open_tab():
            raise ValueError(f"open_tab_failed:{tab_name}")
        self._scroll_to_top()
        visible_rows = max(1, int(getattr(self.config, "visible_ore_rows", 1) or 1))
        max_pages = max(1, int(ceil(len(known_names) / visible_rows)) + 2)
        merged: dict[str, int] = {}
        templates: dict[str, object] = {}
        seen_pages: set[tuple[tuple[str, int], ...]] = set()
        stagnant_pages = 0
        for page_index in range(max_pages):
            rows: dict[int, Any] = {}
            structure_present = False
            attempts = _INVENTORY_FIRST_PAGE_READ_ATTEMPTS if page_index == 0 else 1
            for attempt in range(attempts):
                rows = self.inventory_reader.read_visible_rows(known_names=known_names, aliases=aliases)
                fingerprint = tuple(
                    (str(row.name), int(row.quantity))
                    for row in rows.values()
                    if str(getattr(row, "name", "") or "").strip() and getattr(row, "quantity", None) is not None
                )
                if fingerprint:
                    break
                panel_image = self.inventory_reader._capture_key("ORES_PANEL_TEXT")
                structure_present = self.inventory_reader._panel_structure_present(panel_image)
                if not structure_present or attempt + 1 >= attempts:
                    break
                time.sleep(_INVENTORY_PAGE_REREAD_DELAY_SECONDS)
            for row_index, row in rows.items():
                row_name = str(getattr(row, "name", "") or "").strip()
                if not row_name or row_name in templates:
                    continue
                read_rect = self.rects.get(f"ORE_ROW{int(row_index)}_READ")
                if read_rect is None:
                    continue
                icon_rect = (int(read_rect[0]) - 62, int(read_rect[1]) + 10, 47, 47)
                icon_image = self.capture.capture_client_bbox(icon_rect)
                if icon_image is not None:
                    templates[row_name] = icon_image
            if page_index == 0:
                for row_index, template_name in enumerate(template_row_names or [], start=1):
                    if template_name in templates:
                        continue
                    read_rect = self.rects.get(f"ORE_ROW{int(row_index)}_READ")
                    if read_rect is None:
                        continue
                    icon_rect = (int(read_rect[0]) - 62, int(read_rect[1]) + 10, 47, 47)
                    icon_image = self.capture.capture_client_bbox(icon_rect)
                    if icon_image is not None:
                        templates[str(template_name)] = icon_image
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
        return merged, templates

    def _overview_seam_blocker(self, *, kind: str, blocker: str) -> dict[str, Any]:
        return {
            "feasible": False,
            "blocker": str(blocker),
            **seam_contract_summary(),
        }

    def _overview_seam_ok(self) -> dict[str, Any]:
        return {
            "feasible": True,
            "blocker": "",
            **seam_contract_summary(),
        }

    def read(self) -> dict[str, Any]:
        alloy_names = _alloy_inventory_names()
        overview_rects = required_production_overview_rects()
        production_reader = self.production_reader
        if production_reader is None and self.perception is not None:
            production_reader = ProductionOverviewReader(
                rects=self.rects,
                capture=self.capture,
                actions=self.actions,
                perception=self.perception,
            )
        seam_status = {
            "active_smelter_assignments": self._overview_seam_blocker(
                kind="smelt",
                blocker="production overview reader unavailable",
            ),
            "active_crafter_assignments": self._overview_seam_blocker(
                kind="craft",
                blocker="production overview reader unavailable",
            ),
            "current_bar_inventory": {
                "feasible": True,
                "blocker": "",
            },
            "current_item_inventory": {
                "feasible": True,
                "blocker": "",
            },
        }
        missing_overview_rects = [rect_key for rect_key in overview_rects if self.rects.get(rect_key) is None]
        if missing_overview_rects:
            blocker = "missing calibrated rects: " + ", ".join(missing_overview_rects)
            seam_status["active_smelter_assignments"] = self._overview_seam_blocker(kind="smelt", blocker=blocker)
            seam_status["active_crafter_assignments"] = self._overview_seam_blocker(kind="craft", blocker=blocker)
        try:
            ore_templates: dict[str, object] = {}
            ore_counts: dict[str, int] = {}
            try:
                ore_counts, ore_templates = self._read_inventory_tab(
                    tab_name="ores",
                    open_tab=self.actions.open_ores_panel,
                    known_names=list(RESOURCE_ROW_NAMES),
                    template_row_names=list(RESOURCE_ROW_NAMES)[:5],
                )
            except Exception:
                ore_templates = {}
                ore_counts = {}
            bars, bar_templates = self._read_inventory_tab(
                tab_name="bars",
                open_tab=self.actions.open_alloys_panel,
                known_names=alloy_names,
                aliases=_inventory_aliases(alloy_names),
                template_row_names=list(bars_data.list_bars())[:5],
            )
            items, item_templates = self._read_inventory_tab(
                tab_name="items",
                open_tab=self.actions.open_items_panel,
                known_names=list(items_data.list_items()),
                template_row_names=list(items_data.list_items())[:4],
            )
            smelter_queue: dict[str, int] = {}
            crafter_queue: dict[str, int] = {}
            ore_templates = _template_bank_with_aliases(ore_templates)
            ore_counts = _template_bank_with_aliases({name: value for name, value in ore_counts.items()})
            bar_templates = _template_bank_with_aliases(bar_templates)
            item_templates = _template_bank_with_aliases(item_templates)
            if production_reader is not None and not missing_overview_rects:
                try:
                    smelter_cards = production_reader.read_cards(
                        tab="smelt",
                        open_tab=self.actions.open_smelter_panel,
                        templates={name: image for name, image in bar_templates.items() if name in allowed_overview_outputs("smelt")},
                        inventory_counts=bars,
                        input_templates=ore_templates,
                        ore_inventory_counts={name: int(value) for name, value in ore_counts.items()},
                    )
                    smelter_queue = parse_active_overview_cards(
                        smelter_cards,
                        allowed_outputs=allowed_overview_outputs("smelt"),
                    )
                    seam_status["active_smelter_assignments"] = {
                        **self._overview_seam_ok(),
                        "cards_read": [asdict(card) for card in smelter_cards],
                    }
                except Exception as exc:
                    seam_status["active_smelter_assignments"] = self._overview_seam_blocker(
                        kind="smelt",
                        blocker=str(exc),
                    )
                try:
                    craft_input_templates = {}
                    craft_input_templates.update(bar_templates)
                    craft_input_templates.update(item_templates)
                    craft_input_templates.update(ore_templates)
                    crafter_cards = production_reader.read_cards(
                        tab="craft",
                        open_tab=self.actions.open_crafter_panel,
                        templates={name: image for name, image in item_templates.items() if name in allowed_overview_outputs("craft")},
                        inventory_counts=items,
                        input_templates=craft_input_templates,
                    )
                    crafter_queue = parse_active_overview_cards(
                        crafter_cards,
                        allowed_outputs=allowed_overview_outputs("craft"),
                    )
                    seam_status["active_crafter_assignments"] = {
                        **self._overview_seam_ok(),
                        "cards_read": [asdict(card) for card in crafter_cards],
                    }
                except Exception as exc:
                    seam_status["active_crafter_assignments"] = self._overview_seam_blocker(
                        kind="craft",
                        blocker=str(exc),
                    )
            return {
                "bars": bars,
                "items": items,
                "smelter_queue": smelter_queue,
                "crafter_queue": crafter_queue,
                "seam_status": seam_status,
            }
        finally:
            reset_ui = getattr(self.actions, "reset_ui", None)
            if callable(reset_ui):
                reset_ui()
