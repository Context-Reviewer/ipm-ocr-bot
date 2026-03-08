from __future__ import annotations

from dataclasses import dataclass
from math import hypot, inf

import numpy as np
from PIL import Image, ImageDraw


@dataclass(slots=True, frozen=True)
class StarfieldObject:
    center_x: int
    center_y: int
    radius: int | None
    area: int | None = None
    distance_from_ship: float | None = None


@dataclass(slots=True, frozen=True)
class StarfieldScene:
    ship_center_x: int | None
    ship_center_y: int | None
    ship_radius: int | None
    ship_bbox: tuple[int, int, int, int] | None
    ship_area: int | None
    ship_reject_reason: str | None
    objects: tuple[StarfieldObject, ...]
    viewport: tuple[int, int, int, int] | None = None
    exclusion_zones: tuple[tuple[int, int, int, int], ...] = ()


@dataclass(slots=True, frozen=True)
class _Component:
    center_x: int
    center_y: int
    area: int
    width: int
    height: int

    @property
    def aspect_ratio(self) -> float:
        smaller = max(1, min(self.width, self.height))
        larger = max(self.width, self.height)
        return float(larger) / float(smaller)

    @property
    def fill_ratio(self) -> float:
        return float(self.area) / float(max(1, self.width * self.height))

    @property
    def radius(self) -> int:
        return max(1, int(round(max(self.width, self.height) / 2.0)))

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        half_width = self.width / 2.0
        half_height = self.height / 2.0
        left = int(round(self.center_x - half_width))
        top = int(round(self.center_y - half_height))
        right = left + self.width
        bottom = top + self.height
        return (left, top, right, bottom)


def _connected_components(mask: np.ndarray) -> list[_Component]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[_Component] = []
    for y in range(height):
        for x in range(width):
            if not bool(mask[y, x]) or bool(visited[y, x]):
                continue
            stack = [(x, y)]
            visited[y, x] = True
            points: list[tuple[int, int]] = []
            while stack:
                px, py = stack.pop()
                points.append((px, py))
                for ny in range(max(0, py - 1), min(height, py + 2)):
                    for nx in range(max(0, px - 1), min(width, px + 2)):
                        if not bool(mask[ny, nx]) or bool(visited[ny, nx]):
                            continue
                        visited[ny, nx] = True
                        stack.append((nx, ny))
            if not points:
                continue
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            min_x = min(xs)
            max_x = max(xs)
            min_y = min(ys)
            max_y = max(ys)
            area = len(points)
            if area < 12:
                continue
            components.append(
                _Component(
                    center_x=int(round(sum(xs) / area)),
                    center_y=int(round(sum(ys) / area)),
                    area=area,
                    width=max_x - min_x + 1,
                    height=max_y - min_y + 1,
                )
            )
    return components


def _bright_mask(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    intensity = rgb.max(axis=2)
    threshold = max(148, int(np.percentile(intensity, 96)))
    return intensity >= threshold


def _normalize_viewport(
    viewport: tuple[int, int, int, int] | None,
    *,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    if viewport is None:
        return None
    width, height = image_size
    left, top, right, bottom = (int(value) for value in viewport)
    left = max(0, min(left, width - 1))
    top = max(0, min(top, height - 1))
    right = max(left + 1, min(right, width))
    bottom = max(top + 1, min(bottom, height))
    if right - left < 2 or bottom - top < 2:
        return None
    return (left, top, right, bottom)


def _ship_score(component: _Component, *, image_size: tuple[int, int]) -> float:
    width, height = image_size
    center_x = width / 2.0
    center_y = height / 2.0
    distance = hypot(component.center_x - center_x, component.center_y - center_y)
    max_distance = max(1.0, hypot(center_x, center_y))
    center_bonus = 1.0 - min(1.0, distance / max_distance)
    return (component.aspect_ratio * 2.0) + center_bonus + min(component.area / 120.0, 2.0)


def _normalize_exclusion_zones(
    exclusion_zones: tuple[tuple[int, int, int, int], ...] | None,
    *,
    image_size: tuple[int, int],
) -> tuple[tuple[int, int, int, int], ...]:
    if not exclusion_zones:
        return ()
    normalized: list[tuple[int, int, int, int]] = []
    width, height = image_size
    for zone in exclusion_zones:
        left, top, right, bottom = (int(value) for value in zone)
        left = max(0, min(left, width - 1))
        top = max(0, min(top, height - 1))
        right = max(left + 1, min(right, width))
        bottom = max(top + 1, min(bottom, height))
        if right - left < 2 or bottom - top < 2:
            continue
        normalized.append((left, top, right, bottom))
    return tuple(normalized)


def _apply_exclusion_zones(
    mask: np.ndarray,
    exclusion_zones: tuple[tuple[int, int, int, int], ...],
) -> np.ndarray:
    if not exclusion_zones:
        return mask
    filtered = mask.copy()
    for left, top, right, bottom in exclusion_zones:
        filtered[top:bottom, left:right] = False
    return filtered


def _detect_ship(components: list[_Component], *, image_size: tuple[int, int]) -> _Component | None:
    candidates = [
        component
        for component in components
        if component.area >= 20 and (component.aspect_ratio >= 1.35 or component.fill_ratio <= 0.62)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda component: _ship_score(component, image_size=image_size))


def _ship_reject_reason(
    ship: _Component | None,
    *,
    image_size: tuple[int, int],
    max_radius: int,
    max_bbox_width: int,
    max_bbox_height: int,
    max_area_ratio: float,
) -> str | None:
    if ship is None:
        return None
    if ship.radius > max_radius:
        return "max_radius"
    if ship.width > max_bbox_width:
        return "max_bbox_width"
    if ship.height > max_bbox_height:
        return "max_bbox_height"
    viewport_area = max(1, image_size[0] * image_size[1])
    if (float(ship.area) / float(viewport_area)) > max_area_ratio:
        return "max_area_ratio"
    return None


def _component_inside_expanded_bbox(
    component: _Component,
    ship: _Component,
    *,
    margin: int,
) -> bool:
    ship_left, ship_top, ship_right, ship_bottom = ship.bbox
    expanded_left = ship_left - margin
    expanded_top = ship_top - margin
    expanded_right = ship_right + margin
    expanded_bottom = ship_bottom + margin
    return expanded_left <= component.center_x <= expanded_right and expanded_top <= component.center_y <= expanded_bottom


def _component_is_ship_proximal(
    component: _Component,
    ship: _Component | None,
    *,
    margin: int,
) -> bool:
    if ship is None:
        return False
    return _component_inside_expanded_bbox(component, ship, margin=margin)


def _detect_objects(
    components: list[_Component],
    *,
    ship: _Component | None,
    minimum_area: int,
    minimum_radius: int,
    ship_exclusion_margin: int,
) -> list[_Component]:
    objects: list[_Component] = []
    for component in components:
        if ship is not None and component == ship:
            continue
        if component.area < minimum_area:
            continue
        if component.radius < minimum_radius:
            continue
        if component.aspect_ratio > 1.6:
            continue
        if component.fill_ratio < 0.32:
            continue
        if _component_is_ship_proximal(component, ship, margin=ship_exclusion_margin):
            continue
        objects.append(component)
    return objects


def detect_starfield_scene(
    image: Image.Image,
    *,
    viewport: tuple[int, int, int, int] | None = None,
    exclusion_zones: tuple[tuple[int, int, int, int], ...] | None = None,
    candidate_min_area: int = 80,
    candidate_min_radius: int = 6,
    ship_exclusion_margin: int = 14,
    max_ship_radius: int = 72,
    max_ship_bbox_width: int = 140,
    max_ship_bbox_height: int = 90,
    max_ship_area_ratio: float = 0.08,
) -> StarfieldScene:
    normalized_viewport = _normalize_viewport(viewport, image_size=image.size)
    if normalized_viewport is not None:
        left, top, right, bottom = normalized_viewport
        working_image = image.crop((left, top, right, bottom))
    else:
        left = top = 0
        working_image = image
    normalized_exclusion_zones = _normalize_exclusion_zones(exclusion_zones, image_size=working_image.size)
    components = _connected_components(_apply_exclusion_zones(_bright_mask(working_image), normalized_exclusion_zones))
    detected_ship = _detect_ship(components, image_size=working_image.size)
    ship_reject_reason = _ship_reject_reason(
        detected_ship,
        image_size=working_image.size,
        max_radius=max(1, int(max_ship_radius)),
        max_bbox_width=max(1, int(max_ship_bbox_width)),
        max_bbox_height=max(1, int(max_ship_bbox_height)),
        max_area_ratio=max(0.0, float(max_ship_area_ratio)),
    )
    ship = detected_ship if ship_reject_reason is None else None
    objects = _detect_objects(
        components,
        ship=ship,
        minimum_area=max(12, int(candidate_min_area)),
        minimum_radius=max(1, int(candidate_min_radius)),
        ship_exclusion_margin=max(0, int(ship_exclusion_margin)),
    )
    ship_x = (detected_ship.center_x + left) if detected_ship is not None else None
    ship_y = (detected_ship.center_y + top) if detected_ship is not None else None
    ship_radius = detected_ship.radius if detected_ship is not None else None
    ship_bbox = (
        (
            detected_ship.bbox[0] + left,
            detected_ship.bbox[1] + top,
            detected_ship.bbox[2] + left,
            detected_ship.bbox[3] + top,
        )
        if detected_ship is not None
        else None
    )
    ship_area = detected_ship.area if detected_ship is not None else None
    scene_objects = tuple(
        StarfieldObject(
            center_x=component.center_x + left,
            center_y=component.center_y + top,
            radius=component.radius,
            area=component.area,
            distance_from_ship=(
                hypot((component.center_x + left) - ship_x, (component.center_y + top) - ship_y)
                if ship_reject_reason is None and ship_x is not None and ship_y is not None
                else None
            ),
        )
        for component in objects
    )
    return StarfieldScene(
        ship_center_x=ship_x,
        ship_center_y=ship_y,
        ship_radius=ship_radius,
        ship_bbox=ship_bbox,
        ship_area=ship_area,
        ship_reject_reason=ship_reject_reason,
        objects=scene_objects,
        viewport=normalized_viewport,
        exclusion_zones=tuple(
            (zone_left + left, zone_top + top, zone_right + left, zone_bottom + top)
            for zone_left, zone_top, zone_right, zone_bottom in normalized_exclusion_zones
        ),
    )


def get_ranked_planet_candidates(scene: StarfieldScene) -> list[StarfieldObject]:
    if scene.ship_center_x is None or scene.ship_center_y is None or scene.ship_reject_reason is not None:
        return []
    return sorted(
        scene.objects,
        key=lambda obj: (obj.distance_from_ship if obj.distance_from_ship is not None else inf, obj.center_y, obj.center_x),
    )


def select_nearest_candidate(scene: StarfieldScene) -> StarfieldObject | None:
    ranked = get_ranked_planet_candidates(scene)
    return ranked[0] if ranked else None


def format_starfield_scene_debug(scene: StarfieldScene) -> str:
    ship = (
        f"ship=({scene.ship_center_x},{scene.ship_center_y}) r={scene.ship_radius}"
        + (
            f" bbox={scene.ship_bbox[2] - scene.ship_bbox[0]}x{scene.ship_bbox[3] - scene.ship_bbox[1]}"
            if scene.ship_bbox is not None
            else ""
        )
        + (f" a={scene.ship_area}" if scene.ship_area is not None else "")
        + (f" invalid={scene.ship_reject_reason}" if scene.ship_reject_reason is not None else "")
        if scene.ship_center_x is not None and scene.ship_center_y is not None and scene.ship_radius is not None
        else "ship=missing"
    )
    ranked = get_ranked_planet_candidates(scene)
    candidates = ", ".join(
        f"#{index + 1}@({obj.center_x},{obj.center_y}) d={obj.distance_from_ship:.1f} r={obj.radius}"
        + (f" a={obj.area}" if obj.area is not None else "")
        for index, obj in enumerate(ranked[:6])
        if obj.distance_from_ship is not None and obj.radius is not None
    )
    viewport = (
        f" viewport=({scene.viewport[0]},{scene.viewport[1]})-({scene.viewport[2]},{scene.viewport[3]})"
        if scene.viewport is not None
        else ""
    )
    exclusions = f" exclusions={len(scene.exclusion_zones)}" if scene.exclusion_zones else ""
    return f"[STARFIELD] {ship}{viewport}{exclusions} candidates={len(scene.objects)} ranked=[{candidates}]"


def annotate_starfield_scene(image: Image.Image, scene: StarfieldScene) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    if scene.viewport is not None:
        left, top, right, bottom = scene.viewport
        draw.rectangle((left, top, right, bottom), outline=(255, 180, 48), width=2)
    for left, top, right, bottom in scene.exclusion_zones:
        draw.rectangle((left, top, right, bottom), outline=(255, 96, 96), width=2)
    if scene.ship_bbox is not None:
        draw.rectangle(scene.ship_bbox, outline=(255, 220, 32), width=2)
    if scene.ship_center_x is not None and scene.ship_center_y is not None:
        cx = scene.ship_center_x
        cy = scene.ship_center_y
        draw.line((cx - 8, cy, cx + 8, cy), fill=(255, 220, 32), width=2)
        draw.line((cx, cy - 8, cx, cy + 8), fill=(255, 220, 32), width=2)
    for index, obj in enumerate(get_ranked_planet_candidates(scene), start=1):
        radius = obj.radius or 8
        draw.ellipse(
            (obj.center_x - radius, obj.center_y - radius, obj.center_x + radius, obj.center_y + radius),
            outline=(64, 220, 255),
            width=2,
        )
        draw.text((obj.center_x + radius + 2, obj.center_y - radius), str(index), fill=(255, 255, 255))
    return annotated
