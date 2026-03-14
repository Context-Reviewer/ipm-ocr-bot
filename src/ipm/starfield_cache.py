from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .domain_data import normalize_planet_name


@dataclass(slots=True, frozen=True)
class CachedStarfieldPlanetNode:
    target_rank: int
    point: tuple[int, int]
    image_size: tuple[int, int]
    orientation: str
    radius: int | None = None
    planet_id: int | None = None
    title: str | None = None
    canonical_title: str | None = None
    ship_center: tuple[int, int] | None = None
    anchor_offset: tuple[int, int] | None = None


def image_orientation(image_size: tuple[int, int]) -> str:
    width, height = (max(0, int(image_size[0])), max(0, int(image_size[1])))
    if width > height:
        return "landscape"
    if height > width:
        return "portrait"
    return "square"


def _normalize_pair(value: object) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return (int(value[0]), int(value[1]))
    except Exception:
        return None


def _normalize_entry(payload: object) -> CachedStarfieldPlanetNode | None:
    if not isinstance(payload, dict):
        return None
    point = _normalize_pair(payload.get("point"))
    image_size = _normalize_pair(payload.get("image_size"))
    if point is None or image_size is None:
        return None
    try:
        target_rank = max(1, int(payload.get("target_rank", 0)))
    except Exception:
        return None
    orientation = str(payload.get("orientation", "") or "").strip().lower()
    if not orientation:
        orientation = image_orientation(image_size)
    radius = payload.get("radius")
    planet_id = payload.get("planet_id")
    try:
        radius_value = int(radius) if radius is not None else None
    except Exception:
        radius_value = None
    try:
        planet_id_value = int(planet_id) if planet_id is not None else None
    except Exception:
        planet_id_value = None
    title = str(payload.get("title", "") or "").strip() or None
    canonical_title = str(payload.get("canonical_title", "") or "").strip() or None
    ship_center = _normalize_pair(payload.get("ship_center"))
    anchor_offset = _normalize_pair(payload.get("anchor_offset"))
    return CachedStarfieldPlanetNode(
        target_rank=target_rank,
        point=point,
        image_size=image_size,
        orientation=orientation,
        radius=radius_value,
        planet_id=planet_id_value,
        title=title,
        canonical_title=canonical_title,
        ship_center=ship_center,
        anchor_offset=anchor_offset,
    )


def load_starfield_planet_nodes(path: str | None) -> dict[int, CachedStarfieldPlanetNode]:
    if not path:
        return {}
    target = Path(path)
    if not target.exists():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list):
        return {}
    entries: dict[int, CachedStarfieldPlanetNode] = {}
    for raw_entry in raw_entries:
        normalized = _normalize_entry(raw_entry)
        if normalized is None:
            continue
        entries[normalized.target_rank] = normalized
    return entries


def save_starfield_planet_nodes(path: str | None, entries: dict[int, CachedStarfieldPlanetNode]) -> bool:
    if not path:
        return False
    payload = {
        "entries": [
            asdict(entry)
            for entry in sorted(entries.values(), key=lambda item: (item.target_rank, item.point[1], item.point[0]))
        ]
    }
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        return False
    return True


def upsert_starfield_planet_node(path: str | None, entry: CachedStarfieldPlanetNode) -> bool:
    entries = load_starfield_planet_nodes(path)
    entries[entry.target_rank] = entry
    return save_starfield_planet_nodes(path, entries)


def cached_node_validation_reason(
    entry: CachedStarfieldPlanetNode,
    *,
    image_size: tuple[int, int],
    expected_orientation: str | None = None,
) -> str | None:
    current_orientation = image_orientation(image_size)
    if expected_orientation:
        normalized_expected = str(expected_orientation).strip().lower()
        if normalized_expected and current_orientation != normalized_expected:
            return f"expected_{normalized_expected}_got_{current_orientation}"
    if tuple(int(value) for value in entry.image_size) != tuple(int(value) for value in image_size):
        return "image_size_mismatch"
    if str(entry.orientation or "").strip().lower() != current_orientation:
        return "orientation_mismatch"
    x, y = entry.point
    width, height = (max(0, int(image_size[0])), max(0, int(image_size[1])))
    if x < 0 or y < 0 or x >= width or y >= height:
        return "point_out_of_bounds"
    return None


def remap_cached_node_point(
    entry: CachedStarfieldPlanetNode,
    *,
    image_size: tuple[int, int],
    ship_center: tuple[int, int] | None,
) -> tuple[tuple[int, int] | None, str | None]:
    current_orientation = image_orientation(image_size)
    if str(entry.orientation or "").strip().lower() != current_orientation:
        return None, "orientation_mismatch"
    if ship_center is None:
        return None, "ship_anchor_missing"
    if entry.anchor_offset is None:
        return None, "anchor_offset_missing"
    normalized_point = (int(entry.point[0]), int(entry.point[1]))
    normalized_offset = (int(entry.anchor_offset[0]), int(entry.anchor_offset[1]))
    if entry.ship_center is not None:
        normalized_cached_ship = (int(entry.ship_center[0]), int(entry.ship_center[1]))
        expected_point = (
            normalized_cached_ship[0] + normalized_offset[0],
            normalized_cached_ship[1] + normalized_offset[1],
        )
        if expected_point != normalized_point:
            return None, "anchor_offset_inconsistent"
    remapped_point = (
        int(ship_center[0]) + normalized_offset[0],
        int(ship_center[1]) + normalized_offset[1],
    )
    width, height = (max(0, int(image_size[0])), max(0, int(image_size[1])))
    x, y = remapped_point
    if x < 0 or y < 0 or x >= width or y >= height:
        return None, "remapped_point_out_of_bounds"
    radius = max(0, int(entry.radius)) if entry.radius is not None else 0
    if radius > 0 and (x - radius < 0 or y - radius < 0 or x + radius >= width or y + radius >= height):
        return None, "remapped_radius_out_of_bounds"
    return remapped_point, None


def panel_matches_cached_identity(panel: object, entry: CachedStarfieldPlanetNode) -> bool:
    cached_planet_id = entry.planet_id
    cached_title = entry.canonical_title or normalize_planet_name(entry.title or "")
    panel_planet_id = getattr(panel, "planet_id", None)
    panel_title = normalize_planet_name(getattr(panel, "title", ""))
    if cached_planet_id is None and not cached_title:
        return True
    if cached_planet_id is not None and panel_planet_id is not None and int(panel_planet_id) != int(cached_planet_id):
        return False
    if cached_title and panel_title and panel_title != cached_title:
        return False
    if cached_planet_id is not None and panel_planet_id is not None:
        return True
    if cached_title and panel_title:
        return True
    return False
