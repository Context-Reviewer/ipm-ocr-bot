from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .starfield_scene import (
    StarfieldScene,
    annotate_starfield_scene,
    detect_starfield_scene,
    format_starfield_scene_debug,
    select_nearest_candidate,
)


@dataclass(slots=True, frozen=True)
class StarfieldProbeResult:
    ok: bool
    reason: str
    scene: StarfieldScene | None = None
    target_point: tuple[int, int] | None = None
    rank: int | None = None
    panel: object | None = None


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
    *,
    capture: object,
    actions: object,
    reader: object,
    panel_is_readable,
    starfield_ready_check=None,
    panel_is_confirmed=None,
    settle_seconds: float,
    save_annotation: bool = False,
    annotation_dir: str = "out/starfield",
    scene_viewport: tuple[float, float, float, float] | None = None,
    scene_exclusion_zones: tuple[tuple[float, float, float, float], ...] | None = None,
    ship_exclusion_margin: int = 14,
    candidate_min_radius: int = 6,
    candidate_min_area: int = 80,
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
            return StarfieldProbeResult(ok=False, reason=str(reason), panel=panel)
    capture_screen = getattr(capture, "capture_screen", None)
    if not callable(capture_screen):
        print("[PLANET_NAV] open_failed reason=capture_unavailable")
        return StarfieldProbeResult(ok=False, reason="capture_unavailable")
    image = capture_screen()
    if image is None:
        print("[PLANET_NAV] open_failed reason=capture_unavailable")
        return StarfieldProbeResult(ok=False, reason="capture_unavailable")
    resolved_viewport = resolve_starfield_viewport(image.size, scene_viewport)
    scene = detect_starfield_scene(
        image,
        viewport=resolved_viewport,
        exclusion_zones=resolve_starfield_exclusion_zones(image.size, resolved_viewport, scene_exclusion_zones),
        ship_exclusion_margin=ship_exclusion_margin,
        candidate_min_radius=candidate_min_radius,
        candidate_min_area=candidate_min_area,
        max_ship_radius=max_ship_radius,
        max_ship_bbox_width=max_ship_bbox_width,
        max_ship_bbox_height=max_ship_bbox_height,
        max_ship_area_ratio=max_ship_area_ratio,
    )
    print(format_starfield_scene_debug(scene))
    if scene.ship_reject_reason is not None:
        print(f"[PLANET_NAV] open_failed reason=ship_implausible detail={scene.ship_reject_reason}")
        return StarfieldProbeResult(ok=False, reason="ship_implausible", scene=scene)
    if scene.ship_center_x is None or scene.ship_center_y is None:
        print("[PLANET_NAV] open_failed reason=ship_missing")
        return StarfieldProbeResult(ok=False, reason="ship_missing", scene=scene)
    target = select_nearest_candidate(scene)
    if target is None:
        print("[PLANET_NAV] open_failed reason=no_candidate")
        return StarfieldProbeResult(ok=False, reason="no_candidate", scene=scene)
    if save_annotation:
        saved_path = maybe_save_starfield_annotation(image, scene, output_dir=annotation_dir)
        if saved_path:
            print(f"[STARFIELD] saved_annotation={saved_path}")
    target_point = (target.center_x, target.center_y)
    print(f"[PLANET_NAV] target=({target.center_x},{target.center_y}) rank=1")
    click_point = getattr(actions, "click_client_point", None)
    if not callable(click_point) or not click_point(target_point, delay=settle_seconds):
        print("[PLANET_NAV] open_failed reason=click_failed")
        return StarfieldProbeResult(
            ok=False,
            reason="click_failed",
            scene=scene,
            target_point=target_point,
            rank=1,
        )
    panel = reader.read()
    if not panel_is_readable(panel):
        print("[PLANET_NAV] open_failed reason=panel_not_visible")
        return StarfieldProbeResult(
            ok=False,
            reason="panel_not_visible",
            scene=scene,
            target_point=target_point,
            rank=1,
            panel=panel,
        )
    if callable(panel_is_confirmed) and not panel_is_confirmed(panel):
        print("[PLANET_NAV] open_failed reason=panel_not_confirmed")
        return StarfieldProbeResult(
            ok=False,
            reason="panel_not_confirmed",
            scene=scene,
            target_point=target_point,
            rank=1,
            panel=panel,
        )
    print("[PLANET_NAV] open_confirmed")
    return StarfieldProbeResult(
        ok=True,
        reason="open_confirmed",
        scene=scene,
        target_point=target_point,
        rank=1,
        panel=panel,
    )
