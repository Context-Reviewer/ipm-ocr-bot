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
        objects=((200, 120, 12), (250, 120, 14), (160, 200, 10)),
    )
    scene = detect_starfield_scene(image)
    assert scene.ship_center_x is not None
    assert scene.ship_center_y is not None
    assert abs(scene.ship_center_x - 160) <= 3
    assert abs(scene.ship_center_y - 120) <= 3
    ranked = get_ranked_planet_candidates(scene)
    assert len(ranked) == 3
    assert ranked[0].center_x == 200
    assert ranked[0].center_y == 120
    assert select_nearest_candidate(scene) == ranked[0]


def test_detect_starfield_scene_handles_missing_ship_cleanly():
    image = _scene_image(objects=((120, 120, 12), (180, 120, 12)))
    scene = detect_starfield_scene(image)
    assert scene.ship_center_x is None
    assert scene.ship_center_y is None
    assert len(scene.objects) == 2
    assert get_ranked_planet_candidates(scene) == []
    assert select_nearest_candidate(scene) is None


def test_detect_starfield_scene_handles_ship_with_no_objects():
    image = _scene_image(ship_center=(160, 120))
    scene = detect_starfield_scene(image)
    assert scene.ship_center_x is not None
    assert scene.ship_center_y is not None
    assert scene.objects == ()
    assert get_ranked_planet_candidates(scene) == []


def test_ranked_candidates_sort_deterministically():
    scene = StarfieldScene(
        ship_center_x=100,
        ship_center_y=100,
        objects=(
            StarfieldObject(center_x=140, center_y=100, radius=10, distance_from_ship=40.0),
            StarfieldObject(center_x=100, center_y=130, radius=10, distance_from_ship=30.0),
            StarfieldObject(center_x=120, center_y=100, radius=10, distance_from_ship=20.0),
        ),
    )
    ranked = get_ranked_planet_candidates(scene)
    assert [(obj.center_x, obj.center_y) for obj in ranked] == [(120, 100), (100, 130), (140, 100)]


def test_format_starfield_scene_debug_is_concise_and_includes_rank_order():
    scene = StarfieldScene(
        ship_center_x=100,
        ship_center_y=100,
        objects=(
            StarfieldObject(center_x=120, center_y=100, radius=10, distance_from_ship=20.0),
            StarfieldObject(center_x=140, center_y=100, radius=10, distance_from_ship=40.0),
        ),
    )
    text = format_starfield_scene_debug(scene)
    assert "ship=(100,100)" in text
    assert "candidates=2" in text
    assert "#1@(120,100)" in text
