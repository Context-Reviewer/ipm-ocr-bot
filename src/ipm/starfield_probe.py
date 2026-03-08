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
    settle_seconds: float,
    save_annotation: bool = False,
    annotation_dir: str = "out/starfield",
) -> StarfieldProbeResult:
    capture_screen = getattr(capture, "capture_screen", None)
    if not callable(capture_screen):
        print("[PLANET_NAV] open_failed reason=capture_unavailable")
        return StarfieldProbeResult(ok=False, reason="capture_unavailable")
    image = capture_screen()
    if image is None:
        print("[PLANET_NAV] open_failed reason=capture_unavailable")
        return StarfieldProbeResult(ok=False, reason="capture_unavailable")
    scene = detect_starfield_scene(image)
    print(format_starfield_scene_debug(scene))
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
    print("[PLANET_NAV] open_confirmed")
    return StarfieldProbeResult(
        ok=True,
        reason="open_confirmed",
        scene=scene,
        target_point=target_point,
        rank=1,
        panel=panel,
    )
