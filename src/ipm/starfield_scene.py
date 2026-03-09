from __future__ import annotations

from dataclasses import dataclass
from math import hypot, inf

import numpy as np
from PIL import Image, ImageDraw

from .ship_template import ShipTemplateDetection, detect_ship_template


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
    ship_detection_mode: str | None = None
    ship_template_status: str | None = None
    ship_match_score: float | None = None
    ship_match_scale: float | None = None
    ship_template_reject_reason: str | None = None
    ship_template_raw_bbox: tuple[int, int, int, int] | None = None
    heuristic_detection_status: str | None = None
    heuristic_reject_reason: str | None = None
    heuristic_raw_bbox: tuple[int, int, int, int] | None = None
    heuristic_raw_area: int | None = None
    viewport: tuple[int, int, int, int] | None = None
    exclusion_zones: tuple[tuple[int, int, int, int], ...] = ()
    rejected_candidate_debug: tuple[str, ...] = ()


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


def _normalize_search_region_margins(
    *,
    image_size: tuple[int, int],
    left_margin: int,
    top_margin: int,
    right_margin: int,
    bottom_margin: int,
) -> tuple[int, int, int, int] | None:
    width, height = image_size
    left = max(0, int(left_margin))
    top = max(0, int(top_margin))
    right = max(left + 1, width - max(0, int(right_margin)))
    bottom = max(top + 1, height - max(0, int(bottom_margin)))
    if right - left < 2 or bottom - top < 2:
        return None
    return (left, top, right, bottom)


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


def _build_allowed_template_mask(
    *,
    image_size: tuple[int, int],
    exclusion_zones: tuple[tuple[int, int, int, int], ...],
) -> np.ndarray:
    width, height = image_size
    mask = np.ones((height, width), dtype=bool)
    return _apply_exclusion_zones(mask, exclusion_zones)


def _detect_ship(components: list[_Component], *, image_size: tuple[int, int]) -> _Component | None:
    candidates = [
        component
        for component in components
        if component.area >= 20 and (component.aspect_ratio >= 1.35 or component.fill_ratio <= 0.62)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda component: _ship_score(component, image_size=image_size))


def _component_from_template_detection(detection: ShipTemplateDetection) -> _Component | None:
    match = detection.match
    if match is None:
        return None
    return _Component(
        center_x=int(match.center_x),
        center_y=int(match.center_y),
        area=max(1, int(match.area)),
        width=max(1, int(match.width)),
        height=max(1, int(match.height)),
    )


def _ship_reject_reason(
    ship: _Component | None,
    *,
    image_size: tuple[int, int],
    min_bbox_width: int,
    min_bbox_height: int,
    min_area: int,
    max_radius: int,
    max_bbox_width: int,
    max_bbox_height: int,
    max_area_ratio: float,
) -> str | None:
    if ship is None:
        return None
    if ship.width < min_bbox_width:
        return "min_bbox_width"
    if ship.height < min_bbox_height:
        return "min_bbox_height"
    if ship.area < min_area:
        return "min_area"
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
    x_margin: int,
    y_margin: int,
) -> bool:
    ship_left, ship_top, ship_right, ship_bottom = ship.bbox
    expanded_left = ship_left - x_margin
    expanded_top = ship_top - y_margin
    expanded_right = ship_right + x_margin
    expanded_bottom = ship_bottom + y_margin
    return expanded_left <= component.center_x <= expanded_right and expanded_top <= component.center_y <= expanded_bottom


def _component_is_ship_proximal(
    component: _Component,
    ship: _Component | None,
    *,
    margin: int,
) -> bool:
    if ship is None:
        return False
    return _component_inside_expanded_bbox(component, ship, x_margin=margin, y_margin=margin)


def _component_is_ship_cluster_proximal(
    component: _Component,
    ship: _Component | None,
    *,
    x_margin: int,
    y_margin: int,
) -> bool:
    if ship is None:
        return False
    return _component_inside_expanded_bbox(component, ship, x_margin=x_margin, y_margin=y_margin)


def _component_ship_distance(
    component: _Component,
    ship: _Component | None,
) -> float | None:
    if ship is None:
        return None
    return hypot(component.center_x - ship.center_x, component.center_y - ship.center_y)


def _detect_objects(
    components: list[_Component],
    *,
    ship: _Component | None,
    minimum_area: int,
    minimum_radius: int,
    ship_exclusion_margin: int,
    ship_candidate_exclusion_radius: float,
    ship_cluster_exclusion_x_margin: int,
    ship_cluster_exclusion_y_margin: int,
) -> tuple[list[_Component], tuple[str, ...]]:
    objects: list[_Component] = []
    rejected_debug: list[str] = []
    candidate_exclusion_radius = max(0.0, float(ship_candidate_exclusion_radius))
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
        distance_from_ship = _component_ship_distance(component, ship)
        if distance_from_ship is not None and distance_from_ship < candidate_exclusion_radius:
            rejected_debug.append(f"[STARFIELD] reject reason=near_ship_cluster d={distance_from_ship:.1f}")
            continue
        if _component_is_ship_cluster_proximal(
            component,
            ship,
            x_margin=ship_cluster_exclusion_x_margin,
            y_margin=ship_cluster_exclusion_y_margin,
        ):
            continue
        objects.append(component)
    return objects, tuple(rejected_debug)


def detect_starfield_scene(
    image: Image.Image,
    *,
    viewport: tuple[int, int, int, int] | None = None,
    exclusion_zones: tuple[tuple[int, int, int, int], ...] | None = None,
    ship_template_enabled: bool = False,
    ship_template_path: str | None = None,
    ship_template_scales: tuple[float, ...] = (0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25),
    ship_template_threshold: float = 0.55,
    ship_template_use_edges: bool = True,
    ship_template_allow_fallback: bool = True,
    ship_template_image: Image.Image | None = None,
    ship_template_search_left_margin: int = 0,
    ship_template_search_top_margin: int = 0,
    ship_template_search_right_margin: int = 0,
    ship_template_search_bottom_margin: int = 0,
    ship_template_min_scale: float = 0.0,
    ship_template_min_width: int = 0,
    ship_template_min_height: int = 0,
    ship_template_min_area: int = 0,
    candidate_min_area: int = 80,
    candidate_min_radius: int = 6,
    ship_exclusion_margin: int = 14,
    ship_candidate_exclusion_radius: int = 0,
    ship_cluster_exclusion_x_margin: int = 0,
    ship_cluster_exclusion_y_margin: int = 0,
    min_ship_bbox_width: int = 20,
    min_ship_bbox_height: int = 8,
    min_ship_area: int = 150,
    heuristic_fallback_min_bbox_width: int = 20,
    heuristic_fallback_min_bbox_height: int = 12,
    heuristic_fallback_min_area: int = 180,
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
    template_detection: ShipTemplateDetection | None = None
    detected_ship: _Component | None = None
    ship_detection_mode: str | None = None
    heuristic_detection_status: str | None = None
    heuristic_reject_reason: str | None = None
    heuristic_raw_ship: _Component | None = None
    if ship_template_enabled:
        template_allowed_mask = _build_allowed_template_mask(
            image_size=working_image.size,
            exclusion_zones=normalized_exclusion_zones,
        )
        template_search_region = _normalize_search_region_margins(
            image_size=working_image.size,
            left_margin=ship_template_search_left_margin,
            top_margin=ship_template_search_top_margin,
            right_margin=ship_template_search_right_margin,
            bottom_margin=ship_template_search_bottom_margin,
        )
        template_detection = detect_ship_template(
            working_image,
            template_path=ship_template_path,
            template_image=ship_template_image,
            search_region=template_search_region,
            allowed_mask=template_allowed_mask,
            scales=ship_template_scales,
            threshold=ship_template_threshold,
            use_edges=ship_template_use_edges,
            min_scale=ship_template_min_scale,
            min_width=ship_template_min_width,
            min_height=ship_template_min_height,
            min_area=ship_template_min_area,
        )
        detected_ship = _component_from_template_detection(template_detection)
        if detected_ship is not None:
            ship_detection_mode = "template"
    if detected_ship is None and (not ship_template_enabled or ship_template_allow_fallback):
        heuristic_raw_ship = _detect_ship(components, image_size=working_image.size)
        if heuristic_raw_ship is None:
            heuristic_detection_status = "not_found"
        elif not ship_template_enabled:
            detected_ship = heuristic_raw_ship
            ship_detection_mode = "heuristic"
            heuristic_detection_status = "accepted"
        else:
            heuristic_reject_reason = _ship_reject_reason(
                heuristic_raw_ship,
                image_size=working_image.size,
                min_bbox_width=max(1, int(heuristic_fallback_min_bbox_width)),
                min_bbox_height=max(1, int(heuristic_fallback_min_bbox_height)),
                min_area=max(1, int(heuristic_fallback_min_area)),
                max_radius=max(1, int(max_ship_radius)),
                max_bbox_width=max(1, int(max_ship_bbox_width)),
                max_bbox_height=max(1, int(max_ship_bbox_height)),
                max_area_ratio=max(0.0, float(max_ship_area_ratio)),
            )
            if heuristic_reject_reason is None:
                detected_ship = heuristic_raw_ship
                ship_detection_mode = "heuristic"
                heuristic_detection_status = "accepted"
            else:
                heuristic_detection_status = "rejected"
    ship_reject_reason = _ship_reject_reason(
        detected_ship,
        image_size=working_image.size,
        min_bbox_width=max(1, int(min_ship_bbox_width)),
        min_bbox_height=max(1, int(min_ship_bbox_height)),
        min_area=max(1, int(min_ship_area)),
        max_radius=max(1, int(max_ship_radius)),
        max_bbox_width=max(1, int(max_ship_bbox_width)),
        max_bbox_height=max(1, int(max_ship_bbox_height)),
        max_area_ratio=max(0.0, float(max_ship_area_ratio)),
    )
    ship = detected_ship if ship_reject_reason is None else None
    effective_ship_candidate_exclusion_radius = max(0.0, float(ship_candidate_exclusion_radius))
    if ship is not None and effective_ship_candidate_exclusion_radius <= 0.0:
        effective_ship_candidate_exclusion_radius = float(ship.width) * 2.5
    objects, rejected_candidate_debug = _detect_objects(
        components,
        ship=ship,
        minimum_area=max(12, int(candidate_min_area)),
        minimum_radius=max(1, int(candidate_min_radius)),
        ship_exclusion_margin=max(0, int(ship_exclusion_margin)),
        ship_candidate_exclusion_radius=effective_ship_candidate_exclusion_radius,
        ship_cluster_exclusion_x_margin=max(0, int(ship_cluster_exclusion_x_margin)),
        ship_cluster_exclusion_y_margin=max(0, int(ship_cluster_exclusion_y_margin)),
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
        ship_detection_mode=ship_detection_mode,
        ship_template_status=template_detection.status if template_detection is not None else None,
        ship_match_score=template_detection.best_score if template_detection is not None else None,
        ship_match_scale=template_detection.best_scale if template_detection is not None else None,
        ship_template_reject_reason=template_detection.reject_reason if template_detection is not None else None,
        ship_template_raw_bbox=template_detection.raw_match.bbox if template_detection is not None and template_detection.raw_match is not None else None,
        heuristic_detection_status=heuristic_detection_status,
        heuristic_reject_reason=heuristic_reject_reason,
        heuristic_raw_bbox=(
            (
                heuristic_raw_ship.bbox[0] + left,
                heuristic_raw_ship.bbox[1] + top,
                heuristic_raw_ship.bbox[2] + left,
                heuristic_raw_ship.bbox[3] + top,
            )
            if heuristic_raw_ship is not None
            else None
        ),
        heuristic_raw_area=heuristic_raw_ship.area if heuristic_raw_ship is not None else None,
        viewport=normalized_viewport,
        exclusion_zones=tuple(
            (zone_left + left, zone_top + top, zone_right + left, zone_bottom + top)
            for zone_left, zone_top, zone_right, zone_bottom in normalized_exclusion_zones
        ),
        rejected_candidate_debug=rejected_candidate_debug,
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


def format_ship_detection_debug(scene: StarfieldScene) -> str | None:
    status = scene.ship_template_status
    if status is None:
        return None
    bbox = (
        f" bbox={scene.ship_template_raw_bbox[2] - scene.ship_template_raw_bbox[0]}x{scene.ship_template_raw_bbox[3] - scene.ship_template_raw_bbox[1]}"
        if scene.ship_template_raw_bbox is not None
        else ""
    )
    if status == "match" and scene.ship_center_x is not None and scene.ship_center_y is not None:
        score = f"{scene.ship_match_score:.2f}" if scene.ship_match_score is not None else "?"
        scale = f"{scene.ship_match_scale:.2f}" if scene.ship_match_scale is not None else "?"
        suffix = "" if scene.ship_detection_mode == "template" else " fallback=heuristic"
        return f"[SHIP_DETECT] raw={score} scale={scale}{bbox} accepted center=({scene.ship_center_x},{scene.ship_center_y}){suffix}"
    if status == "below_threshold":
        score = f"{scene.ship_match_score:.2f}" if scene.ship_match_score is not None else "?"
        suffix = " fallback=heuristic" if scene.ship_detection_mode == "heuristic" else ""
        return f"[SHIP_DETECT] result=below_threshold score={score}{suffix}"
    if status == "rejected":
        score = f"{scene.ship_match_score:.2f}" if scene.ship_match_score is not None else "?"
        scale = f"{scene.ship_match_scale:.2f}" if scene.ship_match_scale is not None else "?"
        suffix = " fallback=heuristic" if scene.ship_detection_mode == "heuristic" else ""
        reason = scene.ship_template_reject_reason or "rejected"
        return f"[SHIP_DETECT] raw={score} scale={scale}{bbox} rejected={reason}{suffix}"
    if status == "not_found":
        suffix = " fallback=heuristic" if scene.ship_detection_mode == "heuristic" else ""
        return f"[SHIP_DETECT] result=not_found{suffix}"
    if status == "template_missing":
        suffix = " fallback=heuristic" if scene.ship_detection_mode == "heuristic" else ""
        return f"[SHIP_DETECT] result=template_missing{suffix}"
    return f"[SHIP_DETECT] result={status}"


def format_ship_detection_followup_debug(scene: StarfieldScene) -> tuple[str, ...]:
    lines: list[str] = []
    if scene.heuristic_detection_status == "accepted":
        bbox = (
            f" bbox={scene.heuristic_raw_bbox[2] - scene.heuristic_raw_bbox[0]}x"
            f"{scene.heuristic_raw_bbox[3] - scene.heuristic_raw_bbox[1]}"
            if scene.heuristic_raw_bbox is not None
            else ""
        )
        area = f" a={scene.heuristic_raw_area}" if scene.heuristic_raw_area is not None else ""
        if scene.ship_center_x is not None and scene.ship_center_y is not None:
            lines.append(
                "[SHIP_DETECT] "
                f"heuristic=accepted{bbox}{area} center=({scene.ship_center_x},{scene.ship_center_y})"
            )
    elif scene.heuristic_detection_status == "rejected":
        bbox = (
            f" bbox={scene.heuristic_raw_bbox[2] - scene.heuristic_raw_bbox[0]}x"
            f"{scene.heuristic_raw_bbox[3] - scene.heuristic_raw_bbox[1]}"
            if scene.heuristic_raw_bbox is not None
            else ""
        )
        area = f" a={scene.heuristic_raw_area}" if scene.heuristic_raw_area is not None else ""
        lines.append(
            "[SHIP_DETECT] "
            f"heuristic=rejected reason={scene.heuristic_reject_reason or 'rejected'}{bbox}{area}"
        )
    elif scene.heuristic_detection_status == "not_found":
        lines.append("[SHIP_DETECT] heuristic=result=not_found")
    if scene.ship_center_x is None and scene.ship_center_y is None:
        if scene.ship_template_status is not None or scene.heuristic_detection_status is not None:
            lines.append("[SHIP_DETECT] result=no_accepted_ship")
    return tuple(lines)


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
