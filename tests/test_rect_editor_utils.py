from tools.rect_editor_utils import color_for_name, snap_value, snap_rect


def test_color_deterministic():
    c1 = color_for_name("HUD_CASH")
    c2 = color_for_name("HUD_CASH")
    c3 = color_for_name("PLANET_STATS_PANEL")
    assert c1 == c2
    assert c1 != c3
    assert all(0 <= v <= 255 for v in c1)


def test_snap_value():
    assert snap_value(12, 5) == 10
    assert snap_value(13, 5) == 15
    assert snap_value(0, 5) == 0
    assert snap_value(7, 1) == 7


def test_snap_rect():
    assert snap_rect(12, 13, 27, 29, 5) == (10, 15, 25, 30)
