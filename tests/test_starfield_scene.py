from PIL import Image, ImageDraw

from ipm.starfield_scene import (
    StarfieldObject,
    StarfieldScene,
    detect_starfield_scene,
    format_starfield_scene_debug,
    get_ranked_planet_candidates,
    select_nearest_candidate,
)


def _scene_image(*, ship_center=None, ship_size=(36, 18), objects=()):
    image = Image.new("RGB", (320, 240), (4, 8, 16))
    draw = ImageDraw.Draw(image)
    if ship_center is not None:
        cx, cy = ship_center
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


def test_detect_starfield_scene_finds_ship_and_ranks_objects_nearest_first():
    image = _scene_image(
        ship_center=(160, 120),
        ship_size=(44, 16),
        objects=((200, 120, 12), (250, 120, 14), (160, 200, 10)),
    )
    scene = detect_starfield_scene(image)
    assert scene.ship_center_x is not None
    assert scene.ship_center_y is not None
    assert scene.ship_radius is not None
    assert scene.ship_bbox is not None
    assert scene.ship_area is not None
    assert scene.ship_reject_reason is None
    assert abs(scene.ship_center_x - 160) <= 3
    assert abs(scene.ship_center_y - 120) <= 3
    ranked = get_ranked_planet_candidates(scene)
    assert len(ranked) == 3
    assert ranked[0].center_x == 200
    assert ranked[0].center_y == 120
    assert ranked[0].radius is not None and ranked[0].radius >= 10
    assert ranked[0].area is not None and ranked[0].area >= 80
    assert select_nearest_candidate(scene) == ranked[0]


def test_detect_starfield_scene_handles_missing_ship_cleanly():
    image = _scene_image(objects=((120, 120, 12), (180, 120, 12)))
    scene = detect_starfield_scene(image)
    assert scene.ship_center_x is None
    assert scene.ship_center_y is None
    assert scene.ship_radius is None
    assert scene.ship_bbox is None
    assert scene.ship_area is None
    assert scene.ship_reject_reason is None
    assert len(scene.objects) == 2
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
        objects=((200, 160, 12),),
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
    assert (ranked[0].center_x, ranked[0].center_y) == (200, 160)


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
        objects=((195, 160, 7), (238, 160, 12)),
    )
    scene = detect_starfield_scene(image, ship_exclusion_margin=14)
    ranked = get_ranked_planet_candidates(scene)
    assert len(ranked) == 1
    assert (ranked[0].center_x, ranked[0].center_y) == (238, 160)


def test_detect_starfield_scene_filters_tiny_background_blob_by_size():
    image = _scene_image(
        ship_center=(160, 160),
        ship_size=(44, 16),
        objects=((210, 160, 4), (250, 160, 12)),
    )
    scene = detect_starfield_scene(image, candidate_min_radius=6, candidate_min_area=80)
    ranked = get_ranked_planet_candidates(scene)
    assert len(ranked) == 1
    assert (ranked[0].center_x, ranked[0].center_y) == (250, 160)


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
