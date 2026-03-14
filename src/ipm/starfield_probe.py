from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .domain_data import normalize_planet_name
from .starfield_cache import (
    CachedStarfieldPlanetNode,
    cached_node_validation_reason,
    image_orientation,
    load_starfield_planet_nodes,
    panel_matches_cached_identity,
    upsert_starfield_planet_node,
)
from .starfield_scene import (
    StarfieldScene,
    annotate_starfield_scene,
    detect_starfield_scene,
    format_ship_detection_debug,
    format_ship_detection_followup_debug,
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
    image: Image.Image | None = None,
    actions: object,
    reader: object,
    target_rank: int = 1,
    panel_is_readable,
    starfield_ready_check=None,
    panel_is_confirmed=None,
    settle_seconds: float,
    save_annotation: bool = False,
    annotation_dir: str = "out/starfield",
    expected_orientation: str | None = None,
    planet_node_cache_path: str | None = None,
    scene_viewport: tuple[float, float, float, float] | None = None,
    scene_exclusion_zones: tuple[tuple[float, float, float, float], ...] | None = None,
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
    ship_exclusion_margin: int = 14,
    ship_cluster_exclusion_x_margin: int = 0,
    ship_cluster_exclusion_y_margin: int = 0,
    candidate_min_radius: int = 6,
    candidate_min_area: int = 80,
    ship_candidate_exclusion_radius: int = 0,
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
) -> StarfieldProbeResult:
    probe_read = getattr(reader, "read_for_probe", None)
    click_point = getattr(actions, "click_client_point", None)
    recoverable_confirmation_failures = frozenset(
        {"panel_not_visible", "panel_not_confirmed", "panel_identity_mismatch"}
    )
    immediate_click_failures = frozenset({"click_failed"})

    def _attempt_confirmation(
        point: tuple[int, int],
        *,
        attempt_label: str,
        radius: int | None,
        expected_cache_entry: CachedStarfieldPlanetNode | None = None,
    ) -> tuple[bool, str, object | None]:
        print(
            "[PLANET_NAV] "
            f"click_policy attempt={attempt_label} point=({point[0]},{point[1]}) "
            f"radius={radius}"
        )
        if not callable(click_point) or not click_point(point, delay=settle_seconds):
            return False, "click_failed", None
        first_panel = probe_read() if callable(probe_read) else reader.read()
        first_panel_confirmed = panel_is_readable(first_panel) and (
            not callable(panel_is_confirmed) or panel_is_confirmed(first_panel)
        )
        if first_panel_confirmed and expected_cache_entry is not None and not panel_matches_cached_identity(
            first_panel,
            expected_cache_entry,
        ):
            return False, "panel_identity_mismatch", first_panel
        if first_panel_confirmed:
            return True, "open_confirmed", first_panel
        if callable(probe_read):
            panel = reader.read()
            if not panel_is_readable(panel):
                return False, "panel_not_visible", panel
            if callable(panel_is_confirmed) and not panel_is_confirmed(panel):
                return False, "panel_not_confirmed", panel
            if expected_cache_entry is not None and not panel_matches_cached_identity(panel, expected_cache_entry):
                return False, "panel_identity_mismatch", panel
            return True, "open_confirmed", panel
        if not panel_is_readable(first_panel):
            return False, "panel_not_visible", first_panel
        return False, "panel_not_confirmed", first_panel

    def _cache_result(
        *,
        scene: StarfieldScene,
        panel: object,
        target_point: tuple[int, int],
        radius: int,
        scene_image: Image.Image,
    ) -> None:
        if not planet_node_cache_path:
            return
        raw_title = str(getattr(panel, "title", "")).strip() or None
        ship_center = (
            (scene.ship_center_x, scene.ship_center_y)
            if scene.ship_center_x is not None and scene.ship_center_y is not None
            else None
        )
        anchor_offset = (
            (target_point[0] - ship_center[0], target_point[1] - ship_center[1]) if ship_center is not None else None
        )
        cache_saved = upsert_starfield_planet_node(
            planet_node_cache_path,
            CachedStarfieldPlanetNode(
                target_rank=target_rank,
                point=target_point,
                image_size=(int(scene_image.size[0]), int(scene_image.size[1])),
                orientation=image_orientation(scene_image.size),
                radius=radius,
                planet_id=(
                    int(getattr(panel, "planet_id"))
                    if getattr(panel, "planet_id", None) is not None
                    else None
                ),
                title=raw_title,
                canonical_title=normalize_planet_name(raw_title),
                ship_center=ship_center,
                anchor_offset=anchor_offset,
            ),
        )
        print(f"[PLANET_NAV] cache_update rank={target_rank} saved={'true' if cache_saved else 'false'}")

    def _detect_and_open(scene_image: Image.Image, *, attempt_label: str) -> StarfieldProbeResult:
        attempt_orientation = image_orientation(scene_image.size)
        if normalized_expected_orientation is not None and attempt_orientation != normalized_expected_orientation:
            print(
                "[PLANET_NAV] "
                f"open_failed reason=geometry_mismatch expected={normalized_expected_orientation} actual={attempt_orientation}"
            )
            return StarfieldProbeResult(ok=False, reason="geometry_mismatch", rank=target_rank)
        resolved_viewport = resolve_starfield_viewport(scene_image.size, scene_viewport)
        scene = detect_starfield_scene(
            scene_image,
            viewport=resolved_viewport,
            exclusion_zones=resolve_starfield_exclusion_zones(
                scene_image.size, resolved_viewport, scene_exclusion_zones
            ),
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
            ship_template_min_scale=ship_template_min_scale,
            ship_template_min_width=ship_template_min_width,
            ship_template_min_height=ship_template_min_height,
            ship_template_min_area=ship_template_min_area,
            ship_exclusion_margin=ship_exclusion_margin,
            ship_candidate_exclusion_radius=ship_candidate_exclusion_radius,
            ship_cluster_exclusion_x_margin=ship_cluster_exclusion_x_margin,
            ship_cluster_exclusion_y_margin=ship_cluster_exclusion_y_margin,
            candidate_min_radius=candidate_min_radius,
            candidate_min_area=candidate_min_area,
            min_ship_bbox_width=min_ship_bbox_width,
            min_ship_bbox_height=min_ship_bbox_height,
            min_ship_area=min_ship_area,
            heuristic_fallback_min_bbox_width=heuristic_fallback_min_bbox_width,
            heuristic_fallback_min_bbox_height=heuristic_fallback_min_bbox_height,
            heuristic_fallback_min_area=heuristic_fallback_min_area,
            max_ship_radius=max_ship_radius,
            max_ship_bbox_width=max_ship_bbox_width,
            max_ship_bbox_height=max_ship_bbox_height,
            max_ship_area_ratio=max_ship_area_ratio,
        )
        ship_debug = format_ship_detection_debug(scene)
        if ship_debug:
            print(ship_debug)
        for ship_followup in format_ship_detection_followup_debug(scene):
            print(ship_followup)
        print(format_starfield_scene_debug(scene))
        for rejected in scene.rejected_candidate_debug:
            print(rejected)
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
            saved_path = maybe_save_starfield_annotation(scene_image, scene, output_dir=annotation_dir)
            if saved_path:
                print(f"[STARFIELD] saved_annotation={saved_path}")
        target_point = (target.center_x, target.center_y)
        print(
            f"[PLANET_NAV] target=({target.center_x},{target.center_y}) "
            f"rank={target_rank} radius={target.radius}"
        )
        ok, reason, panel = _attempt_confirmation(target_point, attempt_label=attempt_label, radius=target.radius)
        if not ok:
            return StarfieldProbeResult(
                ok=False,
                reason=reason,
                scene=scene,
                target_point=target_point,
                rank=target_rank,
                panel=panel,
            )
        _cache_result(
            scene=scene,
            panel=panel,
            target_point=target_point,
            radius=target.radius,
            scene_image=scene_image,
        )
        print(f"[PLANET_NAV] open_confirmed via={attempt_label}")
        return StarfieldProbeResult(
            ok=True,
            reason="open_confirmed",
            scene=scene,
            target_point=target_point,
            rank=target_rank,
            panel=panel,
        )

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
    working_image = image
    if working_image is None:
        capture_screen = getattr(capture, "capture_screen", None)
        if not callable(capture_screen):
            print("[PLANET_NAV] open_failed reason=capture_unavailable")
            return StarfieldProbeResult(ok=False, reason="capture_unavailable", rank=target_rank)
        working_image = capture_screen()
        if working_image is None:
            print("[PLANET_NAV] open_failed reason=capture_unavailable")
            return StarfieldProbeResult(ok=False, reason="capture_unavailable", rank=target_rank)
    current_orientation = image_orientation(working_image.size)
    normalized_expected_orientation = str(expected_orientation or "").strip().lower() or None
    if normalized_expected_orientation is not None and current_orientation != normalized_expected_orientation:
        print(
            "[PLANET_NAV] "
            f"open_failed reason=geometry_mismatch expected={normalized_expected_orientation} actual={current_orientation}"
        )
        return StarfieldProbeResult(ok=False, reason="geometry_mismatch", rank=target_rank)
    cached_entry = load_starfield_planet_nodes(planet_node_cache_path).get(target_rank)
    if cached_entry is not None:
        cache_reason = cached_node_validation_reason(
            cached_entry,
            image_size=working_image.size,
            expected_orientation=normalized_expected_orientation,
        )
        if cache_reason is None:
            print(
                "[PLANET_NAV] "
                f"cache_hit rank={target_rank} point=({cached_entry.point[0]},{cached_entry.point[1]})"
            )
            cached_ok, cached_reason_text, cached_panel = _attempt_confirmation(
                cached_entry.point,
                attempt_label="cached",
                radius=cached_entry.radius,
                expected_cache_entry=cached_entry,
            )
            if cached_ok:
                print("[PLANET_NAV] open_confirmed via=cache")
                return StarfieldProbeResult(
                    ok=True,
                    reason="open_confirmed",
                    target_point=cached_entry.point,
                    rank=target_rank,
                    panel=cached_panel,
                )
            print(f"[PLANET_NAV] cache_recover rank={target_rank} reason={cached_reason_text}")
        else:
            print(f"[PLANET_NAV] cache_skip rank={target_rank} reason={cache_reason}")
    primary_result = _detect_and_open(working_image, attempt_label="primary")
    if primary_result.ok:
        return primary_result
    if primary_result.reason in immediate_click_failures:
        print(f"[PLANET_NAV] open_failed reason={primary_result.reason}")
        return primary_result
    if primary_result.reason not in recoverable_confirmation_failures:
        return primary_result
    capture_screen = getattr(capture, "capture_screen", None)
    if not callable(capture_screen):
        print(
            "[PLANET_NAV] "
            f"open_failed reason={primary_result.reason} recovery_unavailable=capture_unavailable"
        )
        return primary_result
    recovery_image = capture_screen()
    if recovery_image is None:
        print(
            "[PLANET_NAV] "
            f"open_failed reason={primary_result.reason} recovery_unavailable=capture_unavailable"
        )
        return primary_result
    print(f"[PLANET_NAV] rediscover attempt=1 prior_reason={primary_result.reason}")
    recovery_result = _detect_and_open(recovery_image, attempt_label="rediscovery")
    if recovery_result.ok:
        return recovery_result
    if recovery_result.reason in recoverable_confirmation_failures or recovery_result.reason in immediate_click_failures:
        print(f"[PLANET_NAV] open_failed reason={recovery_result.reason}")
    return recovery_result


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
