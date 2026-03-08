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
    distance_from_ship: float | None = None


@dataclass(slots=True, frozen=True)
class StarfieldScene:
    ship_center_x: int | None
    ship_center_y: int | None
    objects: tuple[StarfieldObject, ...]


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


def _ship_score(component: _Component, *, image_size: tuple[int, int]) -> float:
    width, height = image_size
    center_x = width / 2.0
    center_y = height / 2.0
    distance = hypot(component.center_x - center_x, component.center_y - center_y)
    max_distance = max(1.0, hypot(center_x, center_y))
    center_bonus = 1.0 - min(1.0, distance / max_distance)
    return (component.aspect_ratio * 2.0) + center_bonus + min(component.area / 120.0, 2.0)


def _detect_ship(components: list[_Component], *, image_size: tuple[int, int]) -> _Component | None:
    candidates = [
        component
        for component in components
        if component.area >= 20 and (component.aspect_ratio >= 1.35 or component.fill_ratio <= 0.62)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda component: _ship_score(component, image_size=image_size))


def _detect_objects(components: list[_Component], *, ship: _Component | None) -> list[_Component]:
    objects: list[_Component] = []
    for component in components:
        if ship is not None and component == ship:
            continue
        if component.area < 20:
            continue
        if component.aspect_ratio > 1.6:
            continue
        if component.fill_ratio < 0.32:
            continue
        objects.append(component)
    return objects


def detect_starfield_scene(image: Image.Image) -> StarfieldScene:
    components = _connected_components(_bright_mask(image))
    ship = _detect_ship(components, image_size=image.size)
    objects = _detect_objects(components, ship=ship)
    ship_x = ship.center_x if ship is not None else None
    ship_y = ship.center_y if ship is not None else None
    scene_objects = tuple(
        StarfieldObject(
            center_x=component.center_x,
            center_y=component.center_y,
            radius=component.radius,
            distance_from_ship=(
                hypot(component.center_x - ship_x, component.center_y - ship_y)
                if ship_x is not None and ship_y is not None
                else None
            ),
        )
        for component in objects
    )
    return StarfieldScene(
        ship_center_x=ship_x,
        ship_center_y=ship_y,
        objects=scene_objects,
    )


def get_ranked_planet_candidates(scene: StarfieldScene) -> list[StarfieldObject]:
    if scene.ship_center_x is None or scene.ship_center_y is None:
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
        f"ship=({scene.ship_center_x},{scene.ship_center_y})"
        if scene.ship_center_x is not None and scene.ship_center_y is not None
        else "ship=missing"
    )
    ranked = get_ranked_planet_candidates(scene)
    candidates = ", ".join(
        f"#{index + 1}@({obj.center_x},{obj.center_y}) d={obj.distance_from_ship:.1f}"
        for index, obj in enumerate(ranked[:6])
        if obj.distance_from_ship is not None
    )
    return f"[STARFIELD] {ship} candidates={len(scene.objects)} ranked=[{candidates}]"


def annotate_starfield_scene(image: Image.Image, scene: StarfieldScene) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
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
