from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

import bars_data
import items_data
from ipm.domain_data import normalize_resource_row_name

_READER_CONTRACT = (
    "read each visible production row as ProductionAssignmentRowState(name=<canonical output>, active=<bool>, backend=<reader>)"
)
_PARSER_CONTRACT = "aggregate only active rows into dict[output_name, active_count]"


def _build_output_aliases() -> tuple[dict[str, str], set[str], set[str]]:
    aliases: dict[str, str] = {}
    smelter_outputs = set(bars_data.list_bars())
    crafter_outputs = set(items_data.list_items())
    for name in sorted(smelter_outputs | crafter_outputs):
        aliases[name] = name
    for bar_name in smelter_outputs:
        if not bar_name.endswith(" Bar"):
            continue
        base_name = bar_name[: -len(" Bar")].strip()
        normalized_base = normalize_resource_row_name(base_name)
        if normalized_base and f"{normalized_base} Bar" != bar_name:
            aliases[f"{normalized_base} Bar"] = bar_name
    return aliases, smelter_outputs, crafter_outputs


_OUTPUT_ALIASES, _SMELTER_OUTPUTS, _CRAFTER_OUTPUTS = _build_output_aliases()


@dataclass(slots=True, frozen=True)
class ProductionAssignmentRowState:
    name: str = ""
    active: bool | None = None
    backend: str = ""


def canonical_output_name(name: str | None) -> str:
    cleaned = str(name or "").strip()
    if not cleaned:
        return ""
    if cleaned in _OUTPUT_ALIASES:
        return _OUTPUT_ALIASES[cleaned]
    normalized_ore = normalize_resource_row_name(cleaned)
    if normalized_ore:
        candidate = f"{normalized_ore} Bar"
        if candidate in _OUTPUT_ALIASES:
            return _OUTPUT_ALIASES[candidate]
    return cleaned


def parse_active_assignment_rows(
    rows: Iterable[ProductionAssignmentRowState],
    *,
    allowed_outputs: set[str],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        canonical_name = canonical_output_name(getattr(row, "name", ""))
        if not canonical_name:
            continue
        if canonical_name not in allowed_outputs:
            raise ValueError(f"unsupported_assignment_row:{canonical_name}")
        active = getattr(row, "active", None)
        if active is None:
            raise ValueError(f"missing_assignment_state:{canonical_name}")
        if bool(active):
            counts[canonical_name] += 1
    return dict(counts)


def required_production_assignment_rect_groups(*, visible_rows: int) -> dict[str, list[str]]:
    row_count = max(1, int(visible_rows or 1))
    return {
        "panel_text": ["PRODUCTION_PANEL_TEXT"],
        "top_anchor": ["PRODUCTION_TOP_ANCHOR"],
        "row_labels": [f"PRODUCTION_ROW{row_index}_READ" for row_index in range(1, row_count + 1)],
        "row_active_indicators": [f"PRODUCTION_ROW{row_index}_ACTIVE" for row_index in range(1, row_count + 1)],
    }


def _missing_rects(rects: object, rect_names: Iterable[str]) -> list[str]:
    missing: list[str] = []
    getter = getattr(rects, "get", None)
    for rect_name in rect_names:
        if not callable(getter) or getter(rect_name) is None:
            missing.append(rect_name)
    return missing


def inspect_production_assignment_seams(
    *,
    rects: object,
    actions: object,
    visible_rows: int,
) -> dict[str, Any]:
    required_rect_groups = required_production_assignment_rect_groups(visible_rows=visible_rows)
    required_rects = [
        rect_name
        for group in required_rect_groups.values()
        for rect_name in group
    ]
    missing_rects = _missing_rects(rects, required_rects)
    navigation = {
        "open_smelter_panel": callable(getattr(actions, "open_smelter_panel", None)),
        "open_crafter_panel": callable(getattr(actions, "open_crafter_panel", None)),
    }
    shared = {
        "reader_contract": _READER_CONTRACT,
        "parser_contract": _PARSER_CONTRACT,
        "parser_helper_available": True,
        "required_rect_groups": required_rect_groups,
        "missing_rects": missing_rects,
        "navigation": navigation,
    }

    def _seam_payload(kind: str) -> dict[str, Any]:
        blocker_parts: list[str] = []
        nav_key = "open_smelter_panel" if kind == "smelter" else "open_crafter_panel"
        if not navigation[nav_key]:
            blocker_parts.append(f"missing read-only navigation helper: {nav_key}")
        if missing_rects:
            blocker_parts.append("missing calibrated rects: " + ", ".join(missing_rects))
        else:
            blocker_parts.append("no verified live production row reader implemented")
        return {
            "feasible": False,
            "blocker": "; ".join(blocker_parts),
            **shared,
        }

    return {
        "smelter": _seam_payload("smelter"),
        "crafter": _seam_payload("crafter"),
    }


def allowed_assignment_outputs(kind: str) -> set[str]:
    if str(kind).lower() == "smelter":
        return set(_SMELTER_OUTPUTS)
    if str(kind).lower() == "crafter":
        return set(_CRAFTER_OUTPUTS)
    raise ValueError(f"unsupported_assignment_kind:{kind}")
