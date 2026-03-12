from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from typing import Any

import bars_data
import items_data
from ipm.domain_data import normalize_resource_row_name

from policy import StateSnapshot

_FLOOR_MULTIPLIER = 5


def _sorted_dict(values: dict[str, int]) -> dict[str, int]:
    return {name: int(values[name]) for name in sorted(values)}


def _canonical_ore_name(name: str) -> str:
    normalized = normalize_resource_row_name(name)
    if normalized:
        return normalized
    return str(name or "").strip()


def _build_recipe_aliases() -> tuple[dict[str, str], set[str], set[str]]:
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


_RECIPE_ALIASES, _SMELTER_OUTPUTS, _CRAFTER_OUTPUTS = _build_recipe_aliases()


def _canonical_material_name(name: str) -> str:
    cleaned = str(name or "").strip()
    if not cleaned:
        return ""
    if cleaned in _RECIPE_ALIASES:
        return _RECIPE_ALIASES[cleaned]
    normalized_ore = normalize_resource_row_name(cleaned)
    if normalized_ore:
        return normalized_ore
    return cleaned


def _recipe_book() -> dict[str, dict[str, int]]:
    recipes: dict[str, dict[str, int]] = {}
    for bar_name in bars_data.list_bars():
        bar = bars_data.get_bar(bar_name) or {}
        inputs = {
            _canonical_material_name(input_name): int(input_qty)
            for input_name, input_qty in (bar.get("inputs") or {}).items()
            if int(input_qty) > 0
        }
        recipes[bar_name] = inputs
    for item_name in items_data.list_items():
        item = items_data.get_item(item_name) or {}
        inputs = {
            _canonical_material_name(input_name): int(input_qty)
            for input_name, input_qty in (item.get("inputs") or {}).items()
            if int(input_qty) > 0
        }
        recipes[item_name] = inputs
    return recipes


_RECIPE_BOOK = _recipe_book()


def _material_kind(name: str) -> str:
    if name in _SMELTER_OUTPUTS:
        return "smelter_output"
    if name in _CRAFTER_OUTPUTS:
        return "crafter_output"
    if normalize_resource_row_name(name):
        return "ore"
    return "unknown"


def _normalize_inventory_map(raw: Any) -> dict[str, int]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("inventory fields must be dict[str, int]")
    normalized: dict[str, int] = {}
    for name, qty in raw.items():
        canonical_name = _canonical_material_name(str(name or ""))
        if not canonical_name:
            continue
        if not isinstance(qty, int) or qty < 0:
            raise ValueError(f"inventory quantity for {canonical_name!r} must be a non-negative int")
        normalized[canonical_name] = normalized.get(canonical_name, 0) + int(qty)
    return normalized


def _normalize_assignment_counts(raw: Any, *, allowed_outputs: set[str], field_name: str) -> dict[str, int]:
    if raw is None:
        return {}
    counts: Counter[str] = Counter()
    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, list):
        items = ((name, 1) for name in raw)
    else:
        raise ValueError(f"{field_name} must be dict[str, int] or list[str]")
    for name, count in items:
        canonical_name = _canonical_material_name(str(name or ""))
        if canonical_name not in allowed_outputs:
            raise ValueError(f"{field_name} contains unsupported assignment {name!r}")
        if not isinstance(count, int) or count <= 0:
            raise ValueError(f"{field_name} count for {canonical_name!r} must be a positive int")
        counts[canonical_name] += int(count)
    return dict(counts)


def snapshot_from_mapping(payload: dict[str, Any]) -> StateSnapshot:
    if not isinstance(payload, dict):
        raise ValueError("state payload must be a dict")
    return StateSnapshot(
        ores=_normalize_inventory_map(payload.get("ores")),
        bars=_normalize_inventory_map(payload.get("bars")),
        items=_normalize_inventory_map(payload.get("items")),
        smelter_queue=_normalize_assignment_counts(
            payload.get("smelter_queue"),
            allowed_outputs=_SMELTER_OUTPUTS,
            field_name="smelter_queue",
        ),
        crafter_queue=_normalize_assignment_counts(
            payload.get("crafter_queue"),
            allowed_outputs=_CRAFTER_OUTPUTS,
            field_name="crafter_queue",
        ),
        cash=payload.get("cash"),
    )


def _merge_current_stock(state: StateSnapshot) -> dict[str, int]:
    merged: Counter[str] = Counter()
    for values in (state.ores or {}, state.bars or {}, state.items or {}):
        for name, qty in values.items():
            merged[_canonical_material_name(name)] += int(qty)
    return dict(merged)


def _make_reason(
    *,
    assignment_name: str,
    assignment_kind: str,
    active_count: int,
    material_name: str,
    required_amount: int,
    dependency_path: list[str],
) -> dict[str, Any]:
    return {
        "assignment": assignment_name,
        "assignment_kind": assignment_kind,
        "active_count": active_count,
        "material": material_name,
        "required_amount": required_amount,
        "dependency_path": " -> ".join(dependency_path),
    }


def _expand_requirement(
    *,
    material_name: str,
    required_amount: int,
    assignment_name: str,
    assignment_kind: str,
    active_count: int,
    dependency_path: list[str],
) -> tuple[dict[str, Any], Counter[str], dict[str, list[dict[str, Any]]], set[str]]:
    canonical_name = _canonical_material_name(material_name)
    recipe_inputs = _RECIPE_BOOK.get(canonical_name)
    kind = _material_kind(canonical_name)
    reasons: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reasons[canonical_name].append(
        _make_reason(
            assignment_name=assignment_name,
            assignment_kind=assignment_kind,
            active_count=active_count,
            material_name=canonical_name,
            required_amount=required_amount,
            dependency_path=[*dependency_path, canonical_name],
        )
    )
    aggregate = Counter({canonical_name: int(required_amount)})
    missing_recipe_materials: set[str] = set()
    node = {
        "name": canonical_name,
        "material_kind": kind,
        "required_amount": int(required_amount),
        "recipe_available": bool(recipe_inputs),
        "dependency_path": " -> ".join([*dependency_path, canonical_name]),
        "children": [],
    }
    if recipe_inputs:
        for child_name, child_qty in recipe_inputs.items():
            child_required_amount = int(required_amount) * int(child_qty)
            child_node, child_aggregate, child_reasons, child_missing = _expand_requirement(
                material_name=child_name,
                required_amount=child_required_amount,
                assignment_name=assignment_name,
                assignment_kind=assignment_kind,
                active_count=active_count,
                dependency_path=[*dependency_path, canonical_name],
            )
            node["children"].append(child_node)
            aggregate.update(child_aggregate)
            for reason_name, entries in child_reasons.items():
                reasons[reason_name].extend(entries)
            missing_recipe_materials.update(child_missing)
    elif kind == "unknown":
        missing_recipe_materials.add(canonical_name)
    return node, aggregate, reasons, missing_recipe_materials


def compute_production_floor_advice(state: StateSnapshot) -> dict[str, Any]:
    current_stock = _merge_current_stock(state)
    smelter_assignments = dict(state.smelter_queue or {})
    crafter_assignments = dict(state.crafter_queue or {})
    active_assignments = {
        "smelter": [
            {"name": name, "active_count": int(count)}
            for name, count in sorted(smelter_assignments.items())
        ],
        "crafter": [
            {"name": name, "active_count": int(count)}
            for name, count in sorted(crafter_assignments.items())
        ],
    }

    expanded_requirements: Counter[str] = Counter()
    floor_reasons: dict[str, list[dict[str, Any]]] = defaultdict(list)
    dependency_trees: list[dict[str, Any]] = []
    missing_recipe_materials: set[str] = set()

    for assignment_kind, assignments in (("smelter", smelter_assignments), ("crafter", crafter_assignments)):
        for assignment_name, active_count in sorted(assignments.items()):
            direct_inputs = _RECIPE_BOOK.get(assignment_name)
            if direct_inputs is None:
                raise ValueError(f"no recipe data available for active {assignment_kind} assignment {assignment_name!r}")
            assignment_requirements: Counter[str] = Counter()
            assignment_reasons: dict[str, list[dict[str, Any]]] = defaultdict(list)
            children: list[dict[str, Any]] = []
            for input_name, input_qty in direct_inputs.items():
                child_required_amount = int(input_qty) * int(active_count)
                child_node, child_requirements, child_reasons, child_missing = _expand_requirement(
                    material_name=input_name,
                    required_amount=child_required_amount,
                    assignment_name=assignment_name,
                    assignment_kind=assignment_kind,
                    active_count=int(active_count),
                    dependency_path=[assignment_name],
                )
                children.append(child_node)
                assignment_requirements.update(child_requirements)
                missing_recipe_materials.update(child_missing)
                for reason_name, entries in child_reasons.items():
                    assignment_reasons[reason_name].extend(entries)
                    floor_reasons[reason_name].extend(entries)
            dependency_trees.append(
                {
                    "assignment_kind": assignment_kind,
                    "assignment_name": assignment_name,
                    "active_count": int(active_count),
                    "direct_inputs": _sorted_dict(
                        {
                            _canonical_material_name(name): int(qty) * int(active_count)
                            for name, qty in direct_inputs.items()
                        }
                    ),
                    "expanded_requirements": _sorted_dict(dict(assignment_requirements)),
                    "dependency_tree": children,
                }
            )
            expanded_requirements.update(assignment_requirements)

    protected_floors = {name: int(required_qty) * _FLOOR_MULTIPLIER for name, required_qty in expanded_requirements.items()}
    material_names = sorted(set(current_stock) | set(protected_floors))
    sellable_surplus = {
        name: max(0, int(current_stock.get(name, 0)) - int(protected_floors.get(name, 0)))
        for name in material_names
    }
    protected_materials = [
        {
            "name": name,
            "material_kind": _material_kind(name),
            "expanded_requirement": int(expanded_requirements.get(name, 0)),
            "protected_floor": int(protected_floors.get(name, 0)),
            "current_stock": int(current_stock.get(name, 0)),
            "sellable_surplus": int(sellable_surplus.get(name, 0)),
        }
        for name in material_names
        if int(protected_floors.get(name, 0)) > 0
    ]
    materials_that_must_not_be_sold = [
        name
        for name in material_names
        if int(protected_floors.get(name, 0)) > 0 and int(sellable_surplus.get(name, 0)) == 0
    ]
    floor_reason_map = {
        name: sorted(
            entries,
            key=lambda item: (
                str(item.get("assignment_kind") or ""),
                str(item.get("assignment") or ""),
                str(item.get("dependency_path") or ""),
            ),
        )
        for name, entries in sorted(floor_reasons.items())
    }
    return {
        "input_mode": {
            "active_assignments": "manual_state_snapshot",
            "inventory": "manual_state_snapshot",
        },
        "live_reader_support": {
            "active_assignments": False,
            "ores": True,
            "bars": True,
            "items": True,
        },
        "active_production_assignments_detected": active_assignments,
        "expanded_dependency_trees": dependency_trees,
        "expanded_requirements": _sorted_dict(dict(expanded_requirements)),
        "protected_floors": _sorted_dict(protected_floors),
        "current_stock": _sorted_dict({name: int(current_stock.get(name, 0)) for name in material_names}),
        "sellable_surplus": _sorted_dict(sellable_surplus),
        "materials_that_must_not_be_sold": materials_that_must_not_be_sold,
        "protected_materials": protected_materials,
        "protected_floor_reasons": floor_reason_map,
        "limitations": {
            "active_assignment_reader_available": False,
            "current_inventory_live_readers": {
                "ores": True,
                "bars": True,
                "items": True,
            },
            "missing_recipe_materials": sorted(missing_recipe_materials),
        },
    }


def compute_production_floor_advice_from_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    snapshot = snapshot_from_mapping(payload)
    advice = compute_production_floor_advice(snapshot)
    advice["input_snapshot"] = asdict(snapshot)
    return advice
