from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .domain_data import normalize_planet_name
from .starfield_scene import (
    StarfieldScene,
    annotate_starfield_scene,
    detect_starfield_scene,
    format_ship_detection_debug,
    format_starfield_scene_debug,
    get_ranked_planet_candidates,
)


@dataclass(slots=True, frozen=True)
class StarfieldProbeResult:
    ok: bool
    reason: str
    scene: StarfieldScene | None = None
    target_point: tuple[int, int] | None = None
    rank: int | None = None
    panel: object | None = None


@dataclass(slots=True, frozen=True)
class PlanetDiscoveryResult:
    ok: bool
    reason: str
    target_rank: int | None = None
    target_point: tuple[int, int] | None = None
    ship_center: tuple[int, int] | None = None
    planet_title_raw: str | None = None
    planet_title_canonical: str | None = None
    planet_id: int | None = None
    returned_to_starfield: bool = False
    panel: object | None = None
    scene: StarfieldScene | None = None


def resolve_starfield_viewport(
    image_size: tuple[int, int],
    viewport: tuple[float, float, float, float] | None,
) -> tuple[int, int, int, int] | None:
    if viewport is None:
        return None
    width, height = image_size
    left, top, right, bottom = viewport
    resolved = (
        int(round(width * float(left))),
        int(round(height * float(top))),
        int(round(width * float(right))),
        int(round(height * float(bottom))),
    )
    if resolved[2] <= resolved[0] or resolved[3] <= resolved[1]:
        return None
    return resolved


def resolve_starfield_exclusion_zones(
    image_size: tuple[int, int],
    viewport: tuple[int, int, int, int] | None,
    exclusion_zones: tuple[tuple[float, float, float, float], ...] | None,
) -> tuple[tuple[int, int, int, int], ...]:
    if viewport is None or not exclusion_zones:
        return ()
    left, top, right, bottom = viewport
    width = max(1, right - left)
    height = max(1, bottom - top)
    resolved: list[tuple[int, int, int, int]] = []
    for zone_left, zone_top, zone_right, zone_bottom in exclusion_zones:
        zone = (
            int(round(width * float(zone_left))),
            int(round(height * float(zone_top))),
            int(round(width * float(zone_right))),
            int(round(height * float(zone_bottom))),
        )
        if zone[2] <= zone[0] or zone[3] <= zone[1]:
            continue
        resolved.append(zone)
    return tuple(resolved)


def maybe_save_starfield_annotation(
    image: Image.Image,
    scene: StarfieldScene,
    *,
    output_dir: str,
) -> str | None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    target = path / "starfield_probe.png"
    annotate_starfield_scene(image, scene).save(target)
    return str(target)


def try_open_nearest_starfield_candidate(
    **kwargs,
) -> StarfieldProbeResult:
    return try_open_starfield_candidate_by_rank(target_rank=1, **kwargs)


def try_open_starfield_candidate_by_rank(
    *,
    capture: object,
    actions: object,
    reader: object,
    target_rank: int = 1,
    panel_is_readable,
    starfield_ready_check=None,
    panel_is_confirmed=None,
    settle_seconds: float,
    save_annotation: bool = False,
    annotation_dir: str = "out/starfield",
    scene_viewport: tuple[float, float, float, float] | None = None,
    scene_exclusion_zones: tuple[tuple[float, float, float, float], ...] | None = None,
    ship_template_enabled: bool = False,
    ship_template_path: str | None = None,
    ship_template_scales: tuple[float, ...] = (1.0, 0.75, 0.5, 0.35, 0.25, 0.18, 0.12, 0.08),
    ship_template_threshold: float = 0.55,
    ship_template_use_edges: bool = True,
    ship_template_allow_fallback: bool = True,
    ship_template_image: Image.Image | None = None,
    ship_template_search_left_margin: int = 0,
    ship_template_search_top_margin: int = 0,
    ship_template_search_right_margin: int = 0,
    ship_template_search_bottom_margin: int = 0,
    ship_exclusion_margin: int = 14,
    ship_cluster_exclusion_x_margin: int = 0,
    ship_cluster_exclusion_y_margin: int = 0,
    candidate_min_radius: int = 6,
    candidate_min_area: int = 80,
    min_ship_bbox_width: int = 20,
    min_ship_bbox_height: int = 8,
    min_ship_area: int = 150,
    max_ship_radius: int = 72,
    max_ship_bbox_width: int = 140,
    max_ship_bbox_height: int = 90,
    max_ship_area_ratio: float = 0.08,
) -> StarfieldProbeResult:
    if callable(starfield_ready_check):
        precheck = starfield_ready_check()
        if precheck:
            if isinstance(precheck, tuple):
                reason, panel = precheck
            else:
                reason, panel = str(precheck), None
            print(f"[PLANET_NAV] open_failed reason={reason}")
            return StarfieldProbeResult(ok=False, reason=str(reason), rank=max(1, int(target_rank)), panel=panel)
    target_rank = max(1, int(target_rank))
    capture_screen = getattr(capture, "capture_screen", None)
    if not callable(capture_screen):
        print("[PLANET_NAV] open_failed reason=capture_unavailable")
        return StarfieldProbeResult(ok=False, reason="capture_unavailable", rank=target_rank)
    image = capture_screen()
    if image is None:
        print("[PLANET_NAV] open_failed reason=capture_unavailable")
        return StarfieldProbeResult(ok=False, reason="capture_unavailable", rank=target_rank)
    resolved_viewport = resolve_starfield_viewport(image.size, scene_viewport)
    scene = detect_starfield_scene(
        image,
        viewport=resolved_viewport,
        exclusion_zones=resolve_starfield_exclusion_zones(image.size, resolved_viewport, scene_exclusion_zones),
        ship_template_enabled=ship_template_enabled,
        ship_template_path=ship_template_path,
        ship_template_scales=ship_template_scales,
        ship_template_threshold=ship_template_threshold,
        ship_template_use_edges=ship_template_use_edges,
        ship_template_allow_fallback=ship_template_allow_fallback,
        ship_template_image=ship_template_image,
        ship_template_search_left_margin=ship_template_search_left_margin,
        ship_template_search_top_margin=ship_template_search_top_margin,
        ship_template_search_right_margin=ship_template_search_right_margin,
        ship_template_search_bottom_margin=ship_template_search_bottom_margin,
        ship_exclusion_margin=ship_exclusion_margin,
        ship_cluster_exclusion_x_margin=ship_cluster_exclusion_x_margin,
        ship_cluster_exclusion_y_margin=ship_cluster_exclusion_y_margin,
        candidate_min_radius=candidate_min_radius,
        candidate_min_area=candidate_min_area,
        min_ship_bbox_width=min_ship_bbox_width,
        min_ship_bbox_height=min_ship_bbox_height,
        min_ship_area=min_ship_area,
        max_ship_radius=max_ship_radius,
        max_ship_bbox_width=max_ship_bbox_width,
        max_ship_bbox_height=max_ship_bbox_height,
        max_ship_area_ratio=max_ship_area_ratio,
    )
    ship_debug = format_ship_detection_debug(scene)
    if ship_debug:
        print(ship_debug)
    print(format_starfield_scene_debug(scene))
    if scene.ship_reject_reason is not None:
        print(f"[PLANET_NAV] open_failed reason=ship_implausible detail={scene.ship_reject_reason}")
        return StarfieldProbeResult(ok=False, reason="ship_implausible", scene=scene, rank=target_rank)
    if scene.ship_center_x is None or scene.ship_center_y is None:
        print("[PLANET_NAV] open_failed reason=ship_missing")
        return StarfieldProbeResult(ok=False, reason="ship_missing", scene=scene, rank=target_rank)
    ranked = get_ranked_planet_candidates(scene)
    if not ranked:
        print("[PLANET_NAV] open_failed reason=no_candidate")
        return StarfieldProbeResult(ok=False, reason="no_candidate", scene=scene, rank=target_rank)
    if target_rank > len(ranked):
        print(f"[PLANET_NAV] open_failed reason=candidate_rank_unavailable rank={target_rank}")
        return StarfieldProbeResult(ok=False, reason="candidate_rank_unavailable", scene=scene, rank=target_rank)
    target = ranked[target_rank - 1]
    if save_annotation:
        saved_path = maybe_save_starfield_annotation(image, scene, output_dir=annotation_dir)
        if saved_path:
            print(f"[STARFIELD] saved_annotation={saved_path}")
    target_point = (target.center_x, target.center_y)
    print(f"[PLANET_NAV] target=({target.center_x},{target.center_y}) rank={target_rank}")
    click_point = getattr(actions, "click_client_point", None)
    if not callable(click_point) or not click_point(target_point, delay=settle_seconds):
        print("[PLANET_NAV] open_failed reason=click_failed")
        return StarfieldProbeResult(
            ok=False,
            reason="click_failed",
            scene=scene,
            target_point=target_point,
            rank=target_rank,
        )
    panel = reader.read()
    if not panel_is_readable(panel):
        print("[PLANET_NAV] open_failed reason=panel_not_visible")
        return StarfieldProbeResult(
            ok=False,
            reason="panel_not_visible",
            scene=scene,
            target_point=target_point,
            rank=target_rank,
            panel=panel,
        )
    if callable(panel_is_confirmed) and not panel_is_confirmed(panel):
        print("[PLANET_NAV] open_failed reason=panel_not_confirmed")
        return StarfieldProbeResult(
            ok=False,
            reason="panel_not_confirmed",
            scene=scene,
            target_point=target_point,
            rank=target_rank,
            panel=panel,
        )
    print("[PLANET_NAV] open_confirmed")
    return StarfieldProbeResult(
        ok=True,
        reason="open_confirmed",
        scene=scene,
        target_point=target_point,
        rank=target_rank,
        panel=panel,
    )


def discover_nearest_starfield_planet(
    *,
    return_to_starfield=None,
    **kwargs,
) -> PlanetDiscoveryResult:
    return discover_starfield_planet_by_rank(target_rank=1, return_to_starfield=return_to_starfield, **kwargs)


def discover_starfield_planet_by_rank(
    *,
    target_rank: int = 1,
    return_to_starfield=None,
    **kwargs,
) -> PlanetDiscoveryResult:
    probe = try_open_starfield_candidate_by_rank(target_rank=target_rank, **kwargs)
    ship_center = None
    if probe.scene is not None and probe.scene.ship_center_x is not None and probe.scene.ship_center_y is not None:
        ship_center = (probe.scene.ship_center_x, probe.scene.ship_center_y)
    if not probe.ok:
        return PlanetDiscoveryResult(
            ok=False,
            reason=probe.reason,
            target_rank=probe.rank,
            target_point=probe.target_point,
            ship_center=ship_center,
            panel=probe.panel,
            scene=probe.scene,
        )
    raw_title = str(getattr(probe.panel, "title", "")).strip() or None
    canonical_title = normalize_planet_name(raw_title)
    if not canonical_title:
        return PlanetDiscoveryResult(
            ok=False,
            reason="planet_identity_unresolved",
            target_rank=probe.rank,
            target_point=probe.target_point,
            ship_center=ship_center,
            planet_title_raw=raw_title,
            panel=probe.panel,
            scene=probe.scene,
        )
    planet_id = getattr(probe.panel, "planet_id", None)
    returned_to_starfield = bool(callable(return_to_starfield) and return_to_starfield())
    return PlanetDiscoveryResult(
        ok=True,
        reason="ok" if returned_to_starfield else "return_to_starfield_failed",
        target_rank=probe.rank,
        target_point=probe.target_point,
        ship_center=ship_center,
        planet_title_raw=raw_title,
        planet_title_canonical=canonical_title,
        planet_id=int(planet_id) if planet_id is not None else None,
        returned_to_starfield=returned_to_starfield,
        panel=probe.panel,
        scene=probe.scene,
    )
