from PIL import Image, ImageDraw

from ipm.ship_template import detect_ship_template
from ipm.starfield_scene import (
    StarfieldObject,
    StarfieldScene,
    detect_starfield_scene,
    format_ship_detection_debug,
    format_ship_detection_followup_debug,
    format_starfield_scene_debug,
    get_ranked_planet_candidates,
    select_nearest_candidate,
)


def _draw_ship(draw, *, cx, cy, size, fill=(245, 245, 245), glow=None):
    ship_w, ship_h = size
    body_w = max(10, int(round(ship_w * 0.46)))
    body_h = max(8, int(round(ship_h * 0.62)))
    pod_w = max(6, int(round(ship_w * 0.18)))
    pod_h = max(8, int(round(ship_h * 0.56)))
    draw.ellipse(
        (cx - body_w // 2, cy - body_h // 2, cx + body_w // 2, cy + body_h // 2),
        fill=fill,
    )
    draw.rectangle(
        (cx - ship_w // 2, cy - pod_h // 2, cx - ship_w // 2 + pod_w, cy + pod_h // 2),
        fill=fill,
    )
    draw.rectangle(
        (cx + ship_w // 2 - pod_w, cy - pod_h // 2, cx + ship_w // 2, cy + pod_h // 2),
        fill=fill,
    )
    draw.polygon(
        (
            (cx, cy + ship_h // 2),
            (cx - max(4, ship_w // 8), cy + max(4, ship_h // 7)),
            (cx + max(3, ship_w // 10), cy + max(1, ship_h // 10)),
        ),
        fill=fill,
    )
    if glow is not None:
        draw.ellipse(
            (cx - body_w // 2 - 2, cy - body_h // 2 - 2, cx + body_w // 2 + 2, cy + body_h // 2 + 2),
            outline=glow,
            width=2,
        )


def _scene_image(*, ship_center=None, ship_size=(36, 18), ship_style="ellipse", objects=()):
    image = Image.new("RGB", (320, 240), (4, 8, 16))
    draw = ImageDraw.Draw(image)
    if ship_center is not None:
        cx, cy = ship_center
        if ship_style == "sprite":
            _draw_ship(draw, cx=cx, cy=cy, size=ship_size)
        else:
            ship_w, ship_h = ship_size
            draw.ellipse(
                (cx - ship_w // 2, cy - ship_h // 2, cx + ship_w // 2, cy + ship_h // 2),
                fill=(245, 245, 245),
            )
    for cx, cy, radius in objects:
        draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            fill=(220, 240, 255),
        )
    return image


def _ship_template_image(size=(36, 18)):
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    _draw_ship(
        draw,
        cx=size[0] // 2,
        cy=size[1] // 2,
        size=size,
        fill=(245, 245, 245, 255),
        glow=(180, 255, 255, 255),
    )
    return image


def test_detect_ship_template_finds_synthetic_ship():
    scene = _scene_image(ship_center=(160, 120), ship_size=(44, 16), ship_style="sprite")
    detection = detect_ship_template(
        scene,
        template_image=_ship_template_image((44, 16)),
        scales=(1.0,),
        threshold=0.5,
        use_edges=True,
    )
    assert detection.status == "match"
    assert detection.match is not None
    assert abs(detection.match.center_x - 160) <= 3
    assert abs(detection.match.center_y - 120) <= 3
    assert detection.best_scale in {1.0, 0.75, 0.5, 0.35}


def test_detect_ship_template_supports_multi_scale_matching():
    scene = _scene_image(ship_center=(160, 120), ship_size=(22, 8), ship_style="sprite")
    detection = detect_ship_template(
        scene,
        template_image=_ship_template_image((44, 16)),
        scales=(1.0, 0.5),
        threshold=0.5,
        use_edges=True,
    )
    assert detection.status == "match"
    assert detection.match is not None
    assert detection.best_scale in {0.5, 0.35, 0.25}


def test_detect_ship_template_rejects_tiny_match_by_min_scale():
    scene = _scene_image(ship_center=(160, 120), ship_size=(22, 8), ship_style="sprite")
    detection = detect_ship_template(
        scene,
        template_image=_ship_template_image((44, 16)),
        scales=(0.5,),
        threshold=0.5,
        use_edges=True,
        min_scale=0.6,
    )
    assert detection.status == "rejected"
    assert detection.reject_reason == "min_scale"
    assert detection.raw_match is not None
    assert detection.match is None


def test_detect_ship_template_rejects_tiny_match_by_min_bbox():
    scene = _scene_image(ship_center=(160, 120), ship_size=(22, 8), ship_style="sprite")
    detection = detect_ship_template(
        scene,
        template_image=_ship_template_image((44, 16)),
        scales=(0.5,),
        threshold=0.5,
        use_edges=True,
        min_width=24,
        min_height=10,
    )
    assert detection.status == "rejected"
    assert detection.reject_reason in {"min_width", "min_height"}
    assert detection.raw_match is not None
    assert detection.match is None


def test_detect_ship_template_translates_coordinates_from_search_region():
    scene = _scene_image(ship_center=(160, 120), ship_size=(44, 16), ship_style="sprite")
    detection = detect_ship_template(
        scene,
        template_image=_ship_template_image((44, 16)),
        search_region=(120, 80, 220, 180),
        scales=(1.0,),
        threshold=0.2,
        use_edges=True,
    )
    assert detection.status == "match"
    assert detection.match is not None
    assert abs(detection.match.center_x - 160) <= 3
    assert abs(detection.match.center_y - 120) <= 3


def test_detect_ship_template_fails_cleanly_below_threshold():
    scene = _scene_image()
    detection = detect_ship_template(
        scene,
        template_image=_ship_template_image((44, 16)),
        scales=(1.0,),
        threshold=0.80,
        use_edges=True,
    )
    assert detection.status in {"below_threshold", "not_found"}
    assert detection.match is None


def test_detect_ship_template_ignores_false_positive_outside_search_region():
    scene = _scene_image(ship_center=(280, 120), ship_size=(44, 16), ship_style="sprite")
    detection = detect_ship_template(
        scene,
        template_image=_ship_template_image((44, 16)),
        search_region=(0, 0, 200, 240),
        scales=(1.0,),
        threshold=0.5,
        use_edges=True,
    )
    assert detection.status in {"below_threshold", "not_found"}
    assert detection.match is None


def test_detect_starfield_scene_can_fallback_to_heuristic_when_template_misses():
    scene = detect_starfield_scene(
        _scene_image(ship_center=(160, 120), ship_size=(44, 16), ship_style="sprite", objects=((220, 120, 12),)),
        ship_template_enabled=True,
        ship_template_image=Image.new("RGBA", (24, 24), (0, 0, 0, 0)),
        ship_template_scales=(1.0,),
        ship_template_threshold=0.95,
        ship_template_use_edges=True,
        ship_template_allow_fallback=True,
    )
    assert scene.ship_center_x is not None
    assert scene.ship_detection_mode == "heuristic"
    assert scene.ship_template_status in {"below_threshold", "not_found"}


def test_detect_starfield_scene_can_fallback_after_template_rejection():
    scene = detect_starfield_scene(
        _scene_image(ship_center=(160, 120), ship_size=(22, 8), ship_style="sprite", objects=((220, 120, 12),)),
        ship_template_enabled=True,
        ship_template_image=_ship_template_image((44, 16)),
        ship_template_scales=(0.5,),
        ship_template_threshold=0.5,
        ship_template_use_edges=True,
        ship_template_allow_fallback=True,
        ship_template_min_scale=0.6,
        heuristic_fallback_min_bbox_width=20,
        heuristic_fallback_min_bbox_height=8,
        heuristic_fallback_min_area=150,
    )
    assert scene.ship_center_x is not None
    assert scene.ship_detection_mode == "heuristic"
    assert scene.ship_template_status == "rejected"
    assert scene.ship_template_reject_reason == "min_scale"


def test_detect_starfield_scene_rejects_implausible_heuristic_fallback_after_template_rejection():
    scene = detect_starfield_scene(
        _scene_image(ship_center=(160, 120), ship_size=(22, 8), ship_style="sprite"),
        ship_template_enabled=True,
        ship_template_image=_ship_template_image((44, 16)),
        ship_template_scales=(0.5,),
        ship_template_threshold=0.5,
        ship_template_use_edges=True,
        ship_template_allow_fallback=True,
        ship_template_min_scale=0.6,
        heuristic_fallback_min_bbox_width=24,
        heuristic_fallback_min_bbox_height=12,
        heuristic_fallback_min_area=180,
    )
    assert scene.ship_center_x is None
    assert scene.ship_center_y is None
    assert scene.ship_detection_mode is None
    assert scene.ship_reject_reason is None
    assert scene.ship_template_status == "rejected"
    assert scene.ship_template_reject_reason == "min_scale"
    assert scene.heuristic_detection_status == "rejected"
    assert scene.heuristic_reject_reason in {"min_bbox_width", "min_bbox_height", "min_area"}


def test_detect_starfield_scene_returns_no_anchor_when_template_rejects_and_fallback_fails():
    scene = detect_starfield_scene(
        _scene_image(ship_center=(160, 120), ship_size=(22, 8), ship_style="sprite"),
        ship_template_enabled=True,
        ship_template_image=_ship_template_image((44, 16)),
        ship_template_scales=(0.5,),
        ship_template_threshold=0.5,
        ship_template_use_edges=True,
        ship_template_allow_fallback=False,
        ship_template_min_scale=0.6,
    )
    assert scene.ship_center_x is None
    assert scene.ship_detection_mode is None
    assert scene.ship_template_status == "rejected"
    assert scene.ship_template_reject_reason == "min_scale"


def test_detect_starfield_scene_search_margins_exclude_edge_false_anchor():
    image = _scene_image(ship_center=(280, 120), ship_size=(44, 16), ship_style="sprite")
    scene = detect_starfield_scene(
        image,
        ship_template_enabled=True,
        ship_template_image=_ship_template_image((44, 16)),
        ship_template_scales=(1.0,),
        ship_template_use_edges=True,
        ship_template_allow_fallback=False,
        ship_template_search_right_margin=140,
    )
    assert scene.ship_center_x is None
    assert scene.ship_detection_mode is None
    assert scene.ship_template_status in {"below_threshold", "not_found"}


def test_detect_starfield_scene_rejects_template_match_inside_excluded_zone():
    image = _scene_image(ship_center=(280, 120), ship_size=(44, 16), ship_style="sprite")
    scene = detect_starfield_scene(
        image,
        viewport=(0, 0, 320, 240),
        exclusion_zones=((240, 0, 320, 240),),
        ship_template_enabled=True,
        ship_template_image=_ship_template_image((44, 16)),
        ship_template_scales=(1.0,),
        ship_template_use_edges=True,
        ship_template_allow_fallback=False,
    )
    assert scene.ship_center_x is None
    assert scene.ship_center_y is None
    assert scene.ship_detection_mode is None
    assert scene.ship_template_status in {"below_threshold", "not_found"}


def test_detect_starfield_scene_template_match_uses_allowed_roi_when_excluded_false_anchor_exists():
    image = _scene_image(
        ship_center=(160, 120),
        ship_size=(44, 16),
        ship_style="sprite",
    )
    draw = ImageDraw.Draw(image)
    _draw_ship(draw, cx=280, cy=120, size=(44, 16))
    scene = detect_starfield_scene(
        image,
        viewport=(0, 0, 320, 240),
        exclusion_zones=((240, 0, 320, 240),),
        ship_template_enabled=True,
        ship_template_image=_ship_template_image((44, 16)),
        ship_template_scales=(1.0,),
        ship_template_use_edges=True,
        ship_template_allow_fallback=False,
    )
    assert scene.ship_detection_mode == "template"
    assert scene.ship_template_status == "match"
    assert scene.ship_bbox is not None
    assert abs(scene.ship_center_x - 160) <= 3
    assert abs(scene.ship_center_y - 120) <= 3
    assert scene.ship_bbox[2] <= 240


def test_detect_starfield_scene_fails_closed_when_allowed_template_roi_is_empty():
    image = _scene_image(ship_center=(160, 120), ship_size=(44, 16), ship_style="sprite")
    scene = detect_starfield_scene(
        image,
        viewport=(0, 0, 320, 240),
        exclusion_zones=((0, 0, 320, 240),),
        ship_template_enabled=True,
        ship_template_image=_ship_template_image((44, 16)),
        ship_template_scales=(1.0,),
        ship_template_use_edges=True,
        ship_template_allow_fallback=False,
    )
    assert scene.ship_center_x is None
    assert scene.ship_center_y is None
    assert scene.ship_detection_mode is None
    assert scene.ship_template_status == "allowed_region_invalid"


def test_detect_starfield_scene_finds_ship_and_ranks_objects_nearest_first():
    image = _scene_image(
        ship_center=(160, 120),
        ship_size=(44, 16),
        ship_style="sprite",
        objects=((160, 225, 10),),
    )
    scene = detect_starfield_scene(
        image,
        ship_template_enabled=True,
        ship_template_image=_ship_template_image((44, 16)),
        ship_template_scales=(1.0,),
        ship_template_use_edges=True,
        ship_template_allow_fallback=False,
        ship_template_search_right_margin=120,
        ship_candidate_exclusion_radius=80,
    )
    assert scene.ship_center_x is not None
    assert scene.ship_center_y is not None
    assert scene.ship_radius is not None
    assert scene.ship_bbox is not None
    assert scene.ship_area is not None
    assert scene.ship_reject_reason is None
    assert scene.ship_detection_mode == "template"
    assert scene.ship_template_status == "match"
    assert abs(scene.ship_center_x - 160) <= 3
    assert abs(scene.ship_center_y - 120) <= 3
    ranked = get_ranked_planet_candidates(scene)
    assert len(ranked) == 1
    assert ranked[0].center_x == 160
    assert ranked[0].center_y == 225
    assert ranked[0].radius is not None and ranked[0].radius >= 10
    assert ranked[0].area is not None and ranked[0].area >= 80
    assert select_nearest_candidate(scene) == ranked[0]


def test_detect_starfield_scene_handles_missing_ship_cleanly():
    image = _scene_image()
    scene = detect_starfield_scene(
        image,
        ship_template_enabled=True,
        ship_template_image=_ship_template_image((44, 16)),
        ship_template_scales=(1.0,),
        ship_template_use_edges=True,
        ship_template_allow_fallback=False,
    )
    assert scene.ship_center_x is None
    assert scene.ship_center_y is None
    assert scene.ship_radius is None
    assert scene.ship_bbox is None
    assert scene.ship_area is None
    assert scene.ship_reject_reason is None
    assert scene.ship_template_status in {"below_threshold", "not_found"}
    assert len(scene.objects) == 0
    assert get_ranked_planet_candidates(scene) == []
    assert select_nearest_candidate(scene) is None


def test_detect_starfield_scene_handles_ship_with_no_objects():
    image = _scene_image(ship_center=(160, 120), ship_size=(44, 16))
    scene = detect_starfield_scene(image)
    assert scene.ship_center_x is not None
    assert scene.ship_center_y is not None
    assert scene.ship_radius is not None
    assert scene.ship_reject_reason is None
    assert scene.objects == ()
    assert get_ranked_planet_candidates(scene) == []


def test_detect_starfield_scene_respects_viewport_and_ignores_top_hud_blobs():
    image = _scene_image(
        ship_center=(160, 160),
        ship_size=(44, 16),
        objects=((285, 160, 12),),
    )
    draw = ImageDraw.Draw(image)
    draw.rectangle((120, 12, 200, 30), fill=(255, 255, 255))
    scene = detect_starfield_scene(image, viewport=(0, 60, 320, 240))
    assert scene.viewport == (0, 60, 320, 240)
    assert scene.ship_center_x is not None
    assert scene.ship_center_y is not None
    assert abs(scene.ship_center_x - 160) <= 3
    assert abs(scene.ship_center_y - 160) <= 3
    ranked = get_ranked_planet_candidates(scene)
    assert len(ranked) == 1
    assert (ranked[0].center_x, ranked[0].center_y) == (285, 160)


def test_detect_starfield_scene_ignores_excluded_peripheral_blobs_inside_viewport():
    image = _scene_image(
        ship_center=(160, 160),
        ship_size=(44, 16),
        objects=((200, 160, 12),),
    )
    draw = ImageDraw.Draw(image)
    draw.ellipse((278, 140, 306, 168), fill=(255, 255, 255))
    draw.rectangle((40, 210, 260, 236), fill=(255, 255, 255))
    scene = detect_starfield_scene(
        image,
        viewport=(0, 60, 320, 240),
        exclusion_zones=((275, 0, 320, 180), (0, 148, 320, 180)),
    )
    assert len(scene.exclusion_zones) == 2
    assert len(scene.objects) == 1
    assert (scene.objects[0].center_x, scene.objects[0].center_y) == (200, 160)


def test_detect_starfield_scene_excludes_ship_adjacent_artifact_blob():
    image = _scene_image(
        ship_center=(160, 160),
        ship_size=(44, 16),
        objects=((195, 160, 7), (290, 160, 12)),
    )
    scene = detect_starfield_scene(image, ship_exclusion_margin=14)
    ranked = get_ranked_planet_candidates(scene)
    assert len(ranked) == 1
    assert (ranked[0].center_x, ranked[0].center_y) == (290, 160)


def test_detect_starfield_scene_excludes_broader_ship_cluster_blob():
    image = _scene_image(
        ship_center=(160, 160),
        ship_size=(44, 16),
        objects=((205, 94, 12), (50, 216, 12)),
    )
    scene = detect_starfield_scene(
        image,
        ship_exclusion_margin=14,
        ship_cluster_exclusion_x_margin=60,
        ship_cluster_exclusion_y_margin=90,
    )
    ranked = get_ranked_planet_candidates(scene)
    assert len(ranked) == 1
    assert (ranked[0].center_x, ranked[0].center_y) == (50, 216)


def test_detect_starfield_scene_excludes_near_ship_cluster_by_radius():
    image = _scene_image(
        ship_center=(160, 160),
        ship_size=(44, 16),
        objects=((205, 160, 12), (285, 160, 12)),
    )
    scene = detect_starfield_scene(
        image,
        ship_candidate_exclusion_radius=70,
    )
    ranked = get_ranked_planet_candidates(scene)
    assert len(ranked) == 1
    assert (ranked[0].center_x, ranked[0].center_y) == (285, 160)
    assert scene.rejected_candidate_debug == ("[STARFIELD] reject reason=near_ship_cluster d=45.0",)


def test_detect_starfield_scene_auto_ship_candidate_exclusion_uses_ship_width_scale():
    image = _scene_image(
        ship_center=(160, 160),
        ship_size=(44, 16),
        objects=((240, 160, 12), (280, 160, 12)),
    )
    scene = detect_starfield_scene(image, ship_candidate_exclusion_radius=0)
    ranked = get_ranked_planet_candidates(scene)
    assert len(ranked) == 1
    assert (ranked[0].center_x, ranked[0].center_y) == (280, 160)
    assert scene.rejected_candidate_debug


def test_detect_starfield_scene_cluster_exclusion_can_leave_no_candidates():
    image = _scene_image(
        ship_center=(160, 160),
        ship_size=(44, 16),
        objects=((205, 94, 12),),
    )
    scene = detect_starfield_scene(
        image,
        ship_exclusion_margin=14,
        ship_cluster_exclusion_x_margin=60,
        ship_cluster_exclusion_y_margin=90,
    )
    assert get_ranked_planet_candidates(scene) == []


def test_detect_starfield_scene_filters_tiny_background_blob_by_size():
    image = _scene_image(
        ship_center=(160, 160),
        ship_size=(44, 16),
        objects=((210, 160, 4), (280, 160, 12)),
    )
    scene = detect_starfield_scene(image, candidate_min_radius=6, candidate_min_area=80)
    ranked = get_ranked_planet_candidates(scene)
    assert len(ranked) == 1
    assert (ranked[0].center_x, ranked[0].center_y) == (280, 160)


def test_detect_starfield_scene_rejects_implausibly_large_ship_anchor():
    image = _scene_image(ship_center=(160, 120), ship_size=(220, 80), objects=((260, 120, 14),))
    scene = detect_starfield_scene(
        image,
        max_ship_radius=72,
        max_ship_bbox_width=140,
        max_ship_bbox_height=90,
        max_ship_area_ratio=0.08,
    )
    assert scene.ship_center_x is not None
    assert scene.ship_radius is not None
    assert scene.ship_area is not None
    assert scene.ship_reject_reason is not None
    assert get_ranked_planet_candidates(scene) == []


def test_detect_starfield_scene_rejects_too_thin_ship_anchor():
    image = _scene_image(ship_center=(160, 120), ship_size=(31, 2), objects=((220, 120, 14),))
    scene = detect_starfield_scene(
        image,
        min_ship_bbox_width=20,
        min_ship_bbox_height=8,
        min_ship_area=150,
    )
    assert scene.ship_center_x is not None
    assert scene.ship_reject_reason == "min_bbox_height"
    assert get_ranked_planet_candidates(scene) == []


def test_detect_starfield_scene_rejects_too_small_ship_area():
    image = _scene_image(ship_center=(160, 120), ship_size=(8, 12), objects=((220, 120, 14),))
    scene = detect_starfield_scene(
        image,
        min_ship_bbox_width=6,
        min_ship_bbox_height=8,
        min_ship_area=150,
    )
    assert scene.ship_center_x is not None
    assert scene.ship_reject_reason == "min_area"
    assert get_ranked_planet_candidates(scene) == []


def test_ranked_candidates_sort_deterministically():
    scene = StarfieldScene(
        ship_center_x=100,
        ship_center_y=100,
        ship_radius=12,
        ship_bbox=(88, 92, 112, 108),
        ship_area=200,
        ship_reject_reason=None,
        objects=(
            StarfieldObject(center_x=140, center_y=100, radius=10, area=180, distance_from_ship=40.0),
            StarfieldObject(center_x=100, center_y=130, radius=10, area=180, distance_from_ship=30.0),
            StarfieldObject(center_x=120, center_y=100, radius=10, area=180, distance_from_ship=20.0),
        ),
    )
    ranked = get_ranked_planet_candidates(scene)
    assert [(obj.center_x, obj.center_y) for obj in ranked] == [(120, 100), (100, 130), (140, 100)]


def test_format_starfield_scene_debug_is_concise_and_includes_rank_order():
    scene = StarfieldScene(
        ship_center_x=100,
        ship_center_y=100,
        ship_radius=12,
        ship_bbox=(88, 92, 112, 108),
        ship_area=200,
        ship_reject_reason=None,
        objects=(
            StarfieldObject(center_x=120, center_y=100, radius=10, area=180, distance_from_ship=20.0),
            StarfieldObject(center_x=140, center_y=100, radius=10, area=220, distance_from_ship=40.0),
        ),
        viewport=(10, 20, 300, 220),
        exclusion_zones=((250, 20, 300, 220),),
    )
    text = format_starfield_scene_debug(scene)
    assert "ship=(100,100) r=12 bbox=24x16 a=200" in text
    assert "viewport=(10,20)-(300,220)" in text
    assert "exclusions=1" in text
    assert "candidates=2" in text
    assert "#1@(120,100) d=20.0 r=10 a=180" in text


def test_format_starfield_scene_debug_includes_ship_reject_reason():
    scene = StarfieldScene(
        ship_center_x=141,
        ship_center_y=543,
        ship_radius=16,
        ship_bbox=(126, 542, 157, 544),
        ship_area=42,
        ship_reject_reason="min_bbox_height",
        objects=(),
    )
    text = format_starfield_scene_debug(scene)
    assert "ship=(141,543) r=16 bbox=31x2 a=42 invalid=min_bbox_height" in text


def test_format_ship_detection_debug_reports_template_match():
    scene = StarfieldScene(
        ship_center_x=160,
        ship_center_y=120,
        ship_radius=22,
        ship_bbox=(138, 112, 182, 128),
        ship_area=500,
        ship_reject_reason=None,
        objects=(),
        ship_detection_mode="template",
        ship_template_status="match",
        ship_match_score=0.91,
        ship_match_scale=0.5,
        ship_template_raw_bbox=(138, 112, 182, 128),
    )
    text = format_ship_detection_debug(scene)
    assert text == "[SHIP_DETECT] raw=0.91 scale=0.50 bbox=44x16 accepted center=(160,120)"


def test_format_ship_detection_debug_reports_template_rejection():
    scene = StarfieldScene(
        ship_center_x=160,
        ship_center_y=120,
        ship_radius=22,
        ship_bbox=(138, 112, 182, 128),
        ship_area=500,
        ship_reject_reason=None,
        objects=(),
        ship_detection_mode="heuristic",
        ship_template_status="rejected",
        ship_match_score=0.93,
        ship_match_scale=0.08,
        ship_template_reject_reason="min_width",
        ship_template_raw_bbox=(334, 479, 352, 502),
    )
    text = format_ship_detection_debug(scene)
    assert text == "[SHIP_DETECT] raw=0.93 scale=0.08 bbox=18x23 rejected=min_width fallback=heuristic"


def test_format_ship_detection_followup_debug_reports_heuristic_acceptance():
    scene = StarfieldScene(
        ship_center_x=160,
        ship_center_y=120,
        ship_radius=22,
        ship_bbox=(138, 112, 182, 128),
        ship_area=500,
        ship_reject_reason=None,
        objects=(),
        ship_detection_mode="heuristic",
        ship_template_status="rejected",
        ship_match_score=0.93,
        ship_match_scale=0.08,
        ship_template_reject_reason="min_width",
        ship_template_raw_bbox=(334, 479, 352, 502),
        heuristic_detection_status="accepted",
        heuristic_raw_bbox=(138, 112, 182, 128),
        heuristic_raw_area=500,
    )
    lines = format_ship_detection_followup_debug(scene)
    assert lines == ("[SHIP_DETECT] heuristic=accepted bbox=44x16 a=500 center=(160,120)",)


def test_format_ship_detection_followup_debug_reports_heuristic_rejection_and_no_accepted_ship():
    scene = StarfieldScene(
        ship_center_x=None,
        ship_center_y=None,
        ship_radius=None,
        ship_bbox=None,
        ship_area=None,
        ship_reject_reason=None,
        objects=(),
        ship_detection_mode=None,
        ship_template_status="rejected",
        ship_match_score=0.92,
        ship_match_scale=0.08,
        ship_template_reject_reason="min_scale",
        ship_template_raw_bbox=(334, 479, 352, 502),
        heuristic_detection_status="rejected",
        heuristic_reject_reason="min_bbox_width",
        heuristic_raw_bbox=(474, 318, 476, 334),
        heuristic_raw_area=31,
    )
    lines = format_ship_detection_followup_debug(scene)
    assert lines == (
        "[SHIP_DETECT] heuristic=rejected reason=min_bbox_width bbox=2x16 a=31",
        "[SHIP_DETECT] result=no_accepted_ship",
    )
