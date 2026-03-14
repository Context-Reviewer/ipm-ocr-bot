import inspect

from PIL import Image

from ipm.starfield_cache import CachedStarfieldPlanetNode, load_starfield_planet_nodes, upsert_starfield_planet_node
from ipm.starfield_probe import (
    PlanetDiscoveryResult,
    discover_nearest_starfield_planet,
    discover_starfield_planet_by_rank,
    StarfieldProbeResult,
    resolve_starfield_exclusion_zones,
    resolve_starfield_viewport,
    try_open_starfield_candidate_by_rank,
    try_open_nearest_starfield_candidate,
)
from ipm.state import PlanetPanelState
from ipm.tasks import PlanetsTask
from ipm.config import RuntimeConfig


class FakeCapture:
    def __init__(self, image):
        if isinstance(image, list):
            self.images = list(image)
            self.image = self.images[-1] if self.images else None
        else:
            self.images = None
            self.image = image
        self.calls = 0

    def capture_screen(self):
        self.calls += 1
        if self.images is None:
            return self.image
        if not self.images:
            return None
        self.image = self.images.pop(0)
        return self.image


class FakeActions:
    def __init__(self, click_ok=True, close_ok=True):
        self.calls = []
        self.click_ok = click_ok
        self.close_ok = close_ok

    def click_client_point(self, point, *, delay=None):
        self.calls.append(("click_client_point", point, delay))
        return self.click_ok

    def open_planet_menu(self):
        self.calls.append(("open_planet_menu",))
        return self.close_ok

    def close_planet_panel(self):
        self.calls.append(("close_planet_panel",))
        return self.close_ok


class FakeReader:
    def __init__(self, panel):
        if isinstance(panel, list):
            self.panels = list(panel)
            self.panel = self.panels[-1] if self.panels else None
        else:
            self.panels = None
            self.panel = panel
        self.calls = 0

    def read(self):
        self.calls += 1
        if self.panels is None:
            return self.panel
        if not self.panels:
            return None
        value = self.panels.pop(0)
        self.panel = value
        return value


class ProbeAwareReader(FakeReader):
    def __init__(self, probe_panel, full_panel):
        super().__init__(full_panel)
        self.probe_panel = probe_panel
        self.probe_calls = 0

    def read_for_probe(self):
        self.probe_calls += 1
        return self.probe_panel


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


def _scene_image(*, ship_center=None, ship_size=(36, 18), ship_style="ellipse", objects=(), size=(320, 240)):
    image = Image.new("RGB", size, (4, 8, 16))
    from PIL import ImageDraw

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
    from PIL import ImageDraw

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


def _panel_is_readable(panel):
    return bool(panel and (panel.planet_id is not None or panel.title))


def test_starfield_probe_fails_closed_when_ship_missing():
    result = try_open_nearest_starfield_candidate(
        capture=FakeCapture(_scene_image(objects=((120, 120, 12),))),
        actions=FakeActions(),
        reader=FakeReader(PlanetPanelState(planet_id=1, title="1. BALOR")),
        panel_is_readable=_panel_is_readable,
        settle_seconds=0.0,
    )
    assert result.ok is False
    assert result.reason == "ship_missing"


def test_starfield_probe_uses_provided_image_without_recapturing():
    class BrokenCapture:
        def capture_screen(self):
            raise AssertionError("capture_screen should not be called when image is provided")

    result = try_open_nearest_starfield_candidate(
        capture=BrokenCapture(),
        image=_scene_image(ship_center=(160, 120), objects=((270, 120, 12),)),
        actions=FakeActions(),
        reader=FakeReader(PlanetPanelState(planet_id=1, title="1. BALOR")),
        panel_is_readable=_panel_is_readable,
        settle_seconds=0.0,
    )
    assert result.ok is True
    assert result.reason == "open_confirmed"
    assert result.target_point == (270, 120)


def test_starfield_probe_fails_closed_when_template_detection_misses_without_fallback():
    actions = FakeActions()
    result = try_open_nearest_starfield_candidate(
        capture=FakeCapture(_scene_image(ship_center=(160, 120), objects=((270, 120, 12),))),
        actions=actions,
        reader=FakeReader(PlanetPanelState(planet_id=1, title="1. BALOR")),
        panel_is_readable=_panel_is_readable,
        settle_seconds=0.0,
        ship_template_enabled=True,
        ship_template_image=Image.new("RGBA", (24, 24), (0, 0, 0, 0)),
        ship_template_threshold=0.95,
        ship_template_allow_fallback=False,
    )
    assert result.ok is False
    assert result.reason == "ship_missing"
    assert actions.calls == []


def test_starfield_probe_fails_closed_when_template_rejected_and_heuristic_is_implausible():
    actions = FakeActions()
    result = try_open_nearest_starfield_candidate(
        capture=FakeCapture(_scene_image(ship_center=(160, 120), ship_size=(22, 8), objects=((270, 120, 12),))),
        actions=actions,
        reader=FakeReader(PlanetPanelState(planet_id=1, title="1. BALOR")),
        panel_is_readable=_panel_is_readable,
        settle_seconds=0.0,
        ship_template_enabled=True,
        ship_template_scales=(0.5,),
        ship_template_threshold=0.5,
        ship_template_use_edges=True,
        ship_template_allow_fallback=True,
        ship_template_min_scale=0.6,
        heuristic_fallback_min_bbox_width=24,
        heuristic_fallback_min_bbox_height=12,
        heuristic_fallback_min_area=180,
    )
    assert result.ok is False
    assert result.reason == "ship_missing"
    assert result.scene is not None
    assert result.scene.ship_center_x is None
    assert result.scene.ship_detection_mode is None
    assert result.scene.heuristic_detection_status == "rejected"
    assert actions.calls == []


def test_starfield_probe_can_fallback_when_template_search_region_misses():
    actions = FakeActions()
    result = try_open_nearest_starfield_candidate(
        capture=FakeCapture(_scene_image(ship_center=(160, 120), objects=((270, 120, 12),))),
        actions=actions,
        reader=FakeReader(PlanetPanelState(planet_id=1, title="1. BALOR")),
        panel_is_readable=_panel_is_readable,
        settle_seconds=0.0,
        ship_template_enabled=True,
        ship_template_search_right_margin=220,
        ship_template_allow_fallback=True,
    )
    assert result.ok is True
    assert result.reason == "open_confirmed"
    assert result.target_point == (270, 120)


def test_starfield_probe_excluded_template_false_anchor_fails_closed():
    actions = FakeActions()
    result = try_open_nearest_starfield_candidate(
        capture=FakeCapture(
            _scene_image(
                ship_center=(280, 120),
                ship_size=(44, 16),
                ship_style="sprite",
            )
        ),
        actions=actions,
        reader=FakeReader(PlanetPanelState(planet_id=1, title="1. BALOR")),
        panel_is_readable=_panel_is_readable,
        settle_seconds=0.0,
        scene_viewport=(0.0, 0.0, 1.0, 1.0),
        scene_exclusion_zones=((0.75, 0.0, 1.0, 1.0),),
        ship_template_enabled=True,
        ship_template_image=_ship_template_image((44, 16)),
        ship_template_scales=(1.0,),
        ship_template_use_edges=True,
        ship_template_allow_fallback=False,
    )
    assert result.ok is False
    assert result.reason == "ship_missing"
    assert result.scene is not None
    assert result.scene.ship_center_x is None
    assert actions.calls == []


def test_starfield_probe_uses_allowed_template_roi_when_excluded_false_anchor_exists():
    actions = FakeActions()
    image = _scene_image(
        ship_center=(160, 120),
        ship_size=(44, 16),
        ship_style="sprite",
        objects=((160, 235, 12),),
    )
    from PIL import ImageDraw

    _draw_ship(ImageDraw.Draw(image), cx=280, cy=120, size=(44, 16))
    result = try_open_nearest_starfield_candidate(
        capture=FakeCapture(image),
        actions=actions,
        reader=FakeReader(PlanetPanelState(planet_id=1, title="1. BALOR")),
        panel_is_readable=_panel_is_readable,
        settle_seconds=0.0,
        scene_viewport=(0.0, 0.0, 1.0, 1.0),
        scene_exclusion_zones=((0.75, 0.0, 1.0, 1.0),),
        ship_template_enabled=True,
        ship_template_image=_ship_template_image((44, 16)),
        ship_template_scales=(1.0,),
        ship_template_use_edges=True,
        ship_template_allow_fallback=False,
    )
    assert result.ok is True
    assert result.reason == "open_confirmed"
    assert result.scene is not None
    assert result.scene.ship_detection_mode == "template"
    assert abs(result.scene.ship_center_x - 160) <= 3
    assert abs(result.scene.ship_center_y - 120) <= 3
    assert result.target_point == (160, 231)
    assert actions.calls == [("click_client_point", (160, 231), 0.0)]


def test_starfield_probe_fails_closed_when_not_starfield_ready():
    actions = FakeActions()
    open_panel = PlanetPanelState(planet_id=2, title="DRAŠTA", mining_level=2, mining_cost=233)
    result = try_open_nearest_starfield_candidate(
        capture=FakeCapture(_scene_image(ship_center=(160, 120), objects=((270, 120, 12),))),
        actions=actions,
        reader=FakeReader(open_panel),
        panel_is_readable=_panel_is_readable,
        starfield_ready_check=lambda: ("not_starfield_ready", open_panel),
        settle_seconds=0.0,
    )
    assert result.ok is False
    assert result.reason == "not_starfield_ready"
    assert result.panel == open_panel
    assert actions.calls == []


def test_starfield_probe_fails_closed_when_no_candidates():
    result = try_open_nearest_starfield_candidate(
        capture=FakeCapture(_scene_image(ship_center=(160, 120))),
        actions=FakeActions(),
        reader=FakeReader(PlanetPanelState(planet_id=1, title="1. BALOR")),
        panel_is_readable=_panel_is_readable,
        settle_seconds=0.0,
    )
    assert result.ok is False
    assert result.reason == "no_candidate"


def test_starfield_probe_fails_closed_when_requested_rank_is_unavailable():
    actions = FakeActions()
    result = try_open_starfield_candidate_by_rank(
        capture=FakeCapture(_scene_image(ship_center=(160, 120), objects=((270, 120, 12),))),
        actions=actions,
        reader=FakeReader(PlanetPanelState(planet_id=1, title="1. BALOR")),
        target_rank=2,
        panel_is_readable=_panel_is_readable,
        settle_seconds=0.0,
    )
    assert result.ok is False
    assert result.reason == "candidate_rank_unavailable"
    assert result.rank == 2
    assert actions.calls == []


def test_starfield_probe_rank_becomes_unavailable_after_ship_cluster_exclusion():
    actions = FakeActions()
    result = try_open_starfield_candidate_by_rank(
        capture=FakeCapture(_scene_image(ship_center=(160, 160), objects=((205, 94, 12),))),
        actions=actions,
        reader=FakeReader(PlanetPanelState(planet_id=1, title="1. BALOR")),
        target_rank=1,
        panel_is_readable=_panel_is_readable,
        ship_cluster_exclusion_x_margin=60,
        ship_cluster_exclusion_y_margin=90,
        settle_seconds=0.0,
    )
    assert result.ok is False
    assert result.reason == "no_candidate"
    assert result.rank == 1
    assert actions.calls == []


def test_starfield_probe_selects_nearest_candidate_and_confirms_panel():
    actions = FakeActions()
    result = try_open_nearest_starfield_candidate(
        capture=FakeCapture(_scene_image(ship_center=(160, 120), objects=((270, 120, 12), (300, 120, 14)))),
        actions=actions,
        reader=FakeReader(PlanetPanelState(planet_id=1, title="1. BALOR")),
        panel_is_readable=_panel_is_readable,
        settle_seconds=0.0,
    )
    assert result.ok is True
    assert result.reason == "open_confirmed"
    assert result.target_point == (270, 120)
    assert actions.calls[0][0] == "click_client_point"
    assert actions.calls == [("click_client_point", (270, 120), 0.0)]


def test_starfield_probe_uses_cached_center_point_before_scene_detection(tmp_path):
    cache_path = tmp_path / "starfield_nodes.json"
    assert (
        upsert_starfield_planet_node(
            str(cache_path),
            CachedStarfieldPlanetNode(
                target_rank=1,
                point=(123, 77),
                image_size=(320, 240),
                orientation="landscape",
                radius=12,
                planet_id=1,
                title="1. BALOR",
                canonical_title="Balor",
            ),
        )
        is True
    )
    actions = FakeActions()
    result = try_open_nearest_starfield_candidate(
        capture=FakeCapture(Image.new("RGB", (320, 240), (4, 8, 16))),
        actions=actions,
        reader=FakeReader(PlanetPanelState(planet_id=1, title="1. BALOR", mining_cost=100, speed_cost=200)),
        panel_is_readable=_panel_is_readable,
        panel_is_confirmed=lambda panel: bool(panel and panel.title == "1. BALOR"),
        settle_seconds=0.0,
        expected_orientation="landscape",
        planet_node_cache_path=str(cache_path),
    )
    assert result.ok is True
    assert result.reason == "open_confirmed"
    assert result.target_point == (123, 77)
    assert actions.calls == [("click_client_point", (123, 77), 0.0)]


def test_starfield_probe_logs_exact_cache_hit_acceptance(tmp_path, capsys):
    cache_path = tmp_path / "starfield_nodes.json"
    assert (
        upsert_starfield_planet_node(
            str(cache_path),
            CachedStarfieldPlanetNode(
                target_rank=1,
                point=(123, 77),
                image_size=(320, 240),
                orientation="landscape",
                radius=12,
                planet_id=1,
                title="1. BALOR",
                canonical_title="Balor",
            ),
        )
        is True
    )
    result = try_open_nearest_starfield_candidate(
        capture=FakeCapture(Image.new("RGB", (320, 240), (4, 8, 16))),
        actions=FakeActions(),
        reader=FakeReader(PlanetPanelState(planet_id=1, title="1. BALOR", mining_cost=100, speed_cost=200)),
        panel_is_readable=_panel_is_readable,
        panel_is_confirmed=lambda panel: bool(panel and panel.title == "1. BALOR"),
        settle_seconds=0.0,
        expected_orientation="landscape",
        planet_node_cache_path=str(cache_path),
    )
    assert result.ok is True
    output = capsys.readouterr().out
    assert "event=exact_hit_validation_passed" in output
    assert "event=exact_hit_accepted" in output
    assert "event=fallback_to_rediscovery" not in output
    assert "[STARFIELD_CACHE_SUMMARY]" in output
    assert "exact_hit_accepted=1" in output
    assert "fallback_to_rediscovery=0" in output


def test_starfield_probe_remaps_cached_point_from_anchor_offset_on_same_orientation_drift(tmp_path):
    cache_path = tmp_path / "starfield_nodes.json"
    assert (
        upsert_starfield_planet_node(
            str(cache_path),
            CachedStarfieldPlanetNode(
                target_rank=1,
                point=(270, 120),
                image_size=(320, 240),
                orientation="landscape",
                radius=12,
                planet_id=1,
                title="1. BALOR",
                canonical_title="Balor",
                ship_center=(160, 120),
                anchor_offset=(110, 0),
            ),
        )
        is True
    )
    actions = FakeActions()
    result = try_open_nearest_starfield_candidate(
        capture=FakeCapture(_scene_image(ship_center=(180, 120), objects=((290, 120, 12),), size=(360, 240))),
        actions=actions,
        reader=FakeReader(PlanetPanelState(planet_id=1, title="1. BALOR", mining_cost=100, speed_cost=200)),
        panel_is_readable=_panel_is_readable,
        panel_is_confirmed=lambda panel: bool(panel and panel.title == "1. BALOR"),
        settle_seconds=0.0,
        expected_orientation="landscape",
        planet_node_cache_path=str(cache_path),
    )
    assert result.ok is True
    assert result.reason == "open_confirmed"
    assert result.target_point == (290, 120)
    assert actions.calls == [("click_client_point", (290, 120), 0.0)]
    cached_nodes = load_starfield_planet_nodes(str(cache_path))
    assert cached_nodes[1].point == (290, 120)
    assert cached_nodes[1].image_size == (360, 240)
    assert cached_nodes[1].ship_center == (180, 120)
    assert cached_nodes[1].anchor_offset == (110, 0)


def test_starfield_probe_logs_remap_acceptance_and_refresh(tmp_path, capsys):
    cache_path = tmp_path / "starfield_nodes.json"
    assert (
        upsert_starfield_planet_node(
            str(cache_path),
            CachedStarfieldPlanetNode(
                target_rank=1,
                point=(270, 120),
                image_size=(320, 240),
                orientation="landscape",
                radius=12,
                planet_id=1,
                title="1. BALOR",
                canonical_title="Balor",
                ship_center=(160, 120),
                anchor_offset=(110, 0),
            ),
        )
        is True
    )
    result = try_open_nearest_starfield_candidate(
        capture=FakeCapture(_scene_image(ship_center=(180, 120), objects=((290, 120, 12),), size=(360, 240))),
        actions=FakeActions(),
        reader=FakeReader(PlanetPanelState(planet_id=1, title="1. BALOR", mining_cost=100, speed_cost=200)),
        panel_is_readable=_panel_is_readable,
        panel_is_confirmed=lambda panel: bool(panel and panel.title == "1. BALOR"),
        settle_seconds=0.0,
        expected_orientation="landscape",
        planet_node_cache_path=str(cache_path),
    )
    assert result.ok is True
    output = capsys.readouterr().out
    assert "event=remap_attempted" in output
    assert "event=remap_accepted" in output
    assert "event=cache_refresh_saved" in output
    assert "[STARFIELD_CACHE_SUMMARY]" in output
    assert "remap_attempted=1" in output
    assert "remap_accepted=1" in output
    assert "cache_refresh_saved=1" in output


def test_starfield_probe_rejects_cached_remap_when_orientation_changes(tmp_path, capsys):
    cache_path = tmp_path / "starfield_nodes.json"
    assert (
        upsert_starfield_planet_node(
            str(cache_path),
            CachedStarfieldPlanetNode(
                target_rank=1,
                point=(270, 120),
                image_size=(320, 240),
                orientation="landscape",
                radius=12,
                planet_id=1,
                title="1. BALOR",
                canonical_title="Balor",
                ship_center=(160, 120),
                anchor_offset=(110, 0),
            ),
        )
        is True
    )
    actions = FakeActions()
    result = try_open_nearest_starfield_candidate(
        capture=FakeCapture(Image.new("RGB", (240, 320), (4, 8, 16))),
        actions=actions,
        reader=FakeReader(PlanetPanelState(planet_id=1, title="1. BALOR", mining_cost=100, speed_cost=200)),
        panel_is_readable=_panel_is_readable,
        panel_is_confirmed=lambda panel: bool(panel and panel.title == "1. BALOR"),
        settle_seconds=0.0,
        planet_node_cache_path=str(cache_path),
    )
    assert result.ok is False
    assert result.reason == "ship_missing"
    assert actions.calls == []
    output = capsys.readouterr().out
    assert "event=remap_skipped" in output
    assert "reason=orientation_mismatch" in output
    assert "event=fallback_to_rediscovery" in output


def test_starfield_probe_logs_remap_skip_and_fallback_to_rediscovery(tmp_path, capsys):
    cache_path = tmp_path / "starfield_nodes.json"
    assert (
        upsert_starfield_planet_node(
            str(cache_path),
            CachedStarfieldPlanetNode(
                target_rank=1,
                point=(270, 120),
                image_size=(320, 240),
                orientation="landscape",
                radius=12,
                planet_id=1,
                title="1. BALOR",
                canonical_title="Balor",
                ship_center=(160, 120),
                anchor_offset=(300, 0),
            ),
        )
        is True
    )
    result = try_open_nearest_starfield_candidate(
        capture=FakeCapture(_scene_image(ship_center=(180, 120), objects=((320, 120, 12),), size=(360, 240))),
        actions=FakeActions(),
        reader=FakeReader(PlanetPanelState(planet_id=1, title="1. BALOR", mining_cost=100, speed_cost=200)),
        panel_is_readable=_panel_is_readable,
        panel_is_confirmed=lambda panel: bool(panel and panel.title == "1. BALOR"),
        settle_seconds=0.0,
        expected_orientation="landscape",
        planet_node_cache_path=str(cache_path),
    )
    assert result.ok is True
    output = capsys.readouterr().out
    assert "event=remap_attempted" in output
    assert "event=remap_skipped" in output
    assert "reason=anchor_offset_inconsistent" in output
    assert "event=fallback_to_rediscovery" in output
    assert "[STARFIELD_CACHE_SUMMARY]" in output
    assert "remap_skipped=1" in output
    assert "fallback_to_rediscovery=1" in output
    assert "remap_skipped_anchor_offset_inconsistent=1" in output


def test_starfield_probe_rejects_invalid_anchor_offset_remap_and_falls_back_to_detection(tmp_path):
    cache_path = tmp_path / "starfield_nodes.json"
    assert (
        upsert_starfield_planet_node(
            str(cache_path),
            CachedStarfieldPlanetNode(
                target_rank=1,
                point=(270, 120),
                image_size=(320, 240),
                orientation="landscape",
                radius=12,
                planet_id=1,
                title="1. BALOR",
                canonical_title="Balor",
                ship_center=(160, 120),
                anchor_offset=(300, 0),
            ),
        )
        is True
    )
    actions = FakeActions()
    result = try_open_nearest_starfield_candidate(
        capture=FakeCapture(_scene_image(ship_center=(180, 120), objects=((320, 120, 12),), size=(360, 240))),
        actions=actions,
        reader=FakeReader(PlanetPanelState(planet_id=1, title="1. BALOR", mining_cost=100, speed_cost=200)),
        panel_is_readable=_panel_is_readable,
        panel_is_confirmed=lambda panel: bool(panel and panel.title == "1. BALOR"),
        settle_seconds=0.0,
        expected_orientation="landscape",
        planet_node_cache_path=str(cache_path),
    )
    assert result.ok is True
    assert result.reason == "open_confirmed"
    assert result.target_point == (320, 120)
    assert actions.calls == [("click_client_point", (320, 120), 0.0)]


def test_starfield_probe_refreshes_cache_after_cached_identity_mismatch(tmp_path):
    cache_path = tmp_path / "starfield_nodes.json"
    assert (
        upsert_starfield_planet_node(
            str(cache_path),
            CachedStarfieldPlanetNode(
                target_rank=1,
                point=(123, 77),
                image_size=(320, 240),
                orientation="landscape",
                radius=12,
                planet_id=1,
                title="1. BALOR",
                canonical_title="Balor",
            ),
        )
        is True
    )
    actions = FakeActions()
    reader = FakeReader(
        [
            PlanetPanelState(planet_id=2, title="2. DRASTA", mining_level=2, mining_cost=233, speed_cost=106),
            PlanetPanelState(planet_id=1, title="1. BALOR", mining_level=2, mining_cost=233, speed_cost=106),
        ]
    )
    result = try_open_nearest_starfield_candidate(
        capture=FakeCapture(_scene_image(ship_center=(160, 120), objects=((270, 120, 12),))),
        actions=actions,
        reader=reader,
        panel_is_readable=_panel_is_readable,
        panel_is_confirmed=PlanetsTask._probe_panel_confirmed,
        settle_seconds=0.0,
        expected_orientation="landscape",
        planet_node_cache_path=str(cache_path),
    )
    assert result.ok is True
    assert result.reason == "open_confirmed"
    assert result.target_point == (270, 120)
    assert actions.calls == [
        ("click_client_point", (123, 77), 0.0),
        ("click_client_point", (270, 120), 0.0),
    ]
    cached_nodes = load_starfield_planet_nodes(str(cache_path))
    assert cached_nodes[1].point == (270, 120)
    assert cached_nodes[1].planet_id == 1


def test_starfield_probe_logs_exact_hit_rejection_and_fallback(tmp_path, capsys):
    cache_path = tmp_path / "starfield_nodes.json"
    assert (
        upsert_starfield_planet_node(
            str(cache_path),
            CachedStarfieldPlanetNode(
                target_rank=1,
                point=(123, 77),
                image_size=(320, 240),
                orientation="landscape",
                radius=12,
                planet_id=1,
                title="1. BALOR",
                canonical_title="Balor",
            ),
        )
        is True
    )
    result = try_open_nearest_starfield_candidate(
        capture=FakeCapture(_scene_image(ship_center=(160, 120), objects=((270, 120, 12),))),
        actions=FakeActions(),
        reader=FakeReader(
            [
                PlanetPanelState(planet_id=2, title="2. DRASTA", mining_level=2, mining_cost=233, speed_cost=106),
                PlanetPanelState(planet_id=1, title="1. BALOR", mining_level=2, mining_cost=233, speed_cost=106),
            ]
        ),
        panel_is_readable=_panel_is_readable,
        panel_is_confirmed=PlanetsTask._probe_panel_confirmed,
        settle_seconds=0.0,
        expected_orientation="landscape",
        planet_node_cache_path=str(cache_path),
    )
    assert result.ok is True
    output = capsys.readouterr().out
    assert "event=exact_hit_rejected" in output
    assert "event=fallback_to_rediscovery" in output
    assert "source=exact_hit_rejected" in output


def test_starfield_probe_uses_probe_read_fast_path_without_full_reread_on_confirmed_success():
    actions = FakeActions()
    reader = ProbeAwareReader(
        PlanetPanelState(planet_id=1, title="1. BALOR", mining_cost=100, speed_cost=200),
        PlanetPanelState(title="should not be used"),
    )
    result = try_open_nearest_starfield_candidate(
        capture=FakeCapture(_scene_image(ship_center=(160, 120), objects=((270, 120, 12),))),
        actions=actions,
        reader=reader,
        panel_is_readable=_panel_is_readable,
        panel_is_confirmed=lambda panel: bool(panel and panel.title == "1. BALOR"),
        settle_seconds=0.0,
    )
    assert result.ok is True
    assert result.reason == "open_confirmed"
    assert reader.probe_calls == 1
    assert reader.calls == 0


def test_starfield_probe_falls_back_to_full_read_when_probe_read_is_not_confirmed():
    actions = FakeActions()
    reader = ProbeAwareReader(
        PlanetPanelState(title="Ship Speed", mining_cost=100, speed_cost=200),
        PlanetPanelState(planet_id=1, title="1. BALOR", mining_cost=100, speed_cost=200),
    )
    result = try_open_nearest_starfield_candidate(
        capture=FakeCapture(_scene_image(ship_center=(160, 120), objects=((270, 120, 12),))),
        actions=actions,
        reader=reader,
        panel_is_readable=_panel_is_readable,
        panel_is_confirmed=lambda panel: bool(panel and panel.title == "1. BALOR"),
        settle_seconds=0.0,
    )
    assert result.ok is True
    assert result.reason == "open_confirmed"
    assert reader.probe_calls == 1
    assert reader.calls == 1


def test_starfield_probe_retries_once_after_primary_confirmation_failure():
    capture = FakeCapture(
        [
            _scene_image(ship_center=(160, 120), objects=((270, 120, 12),)),
            _scene_image(ship_center=(160, 120), objects=((300, 120, 12),)),
        ]
    )
    actions = FakeActions()
    reader = FakeReader(
        [
            PlanetPanelState(title="Ship Speed", mining_cost=300, speed_cost=400),
            PlanetPanelState(planet_id=1, title="1. BALOR", mining_level=2, mining_cost=233, speed_cost=106),
        ]
    )
    result = try_open_nearest_starfield_candidate(
        capture=capture,
        actions=actions,
        reader=reader,
        panel_is_readable=_panel_is_readable,
        panel_is_confirmed=PlanetsTask._probe_panel_confirmed,
        settle_seconds=0.0,
    )
    assert result.ok is True
    assert result.reason == "open_confirmed"
    assert result.target_point == (300, 120)
    assert capture.calls == 2
    assert actions.calls == [
        ("click_client_point", (270, 120), 0.0),
        ("click_client_point", (300, 120), 0.0),
    ]


def test_starfield_probe_second_confirmation_failure_fails_loud_after_single_rediscovery():
    capture = FakeCapture(
        [
            _scene_image(ship_center=(160, 120), objects=((270, 120, 12),)),
            _scene_image(ship_center=(160, 120), objects=((300, 120, 12),)),
        ]
    )
    actions = FakeActions()
    reader = FakeReader(
        [
            PlanetPanelState(title="Ship Speed", mining_cost=300, speed_cost=400),
            PlanetPanelState(title="Ship Speed", mining_cost=301, speed_cost=401),
        ]
    )
    result = try_open_nearest_starfield_candidate(
        capture=capture,
        actions=actions,
        reader=reader,
        panel_is_readable=_panel_is_readable,
        panel_is_confirmed=lambda panel: bool(panel and panel.title == "1. BALOR"),
        settle_seconds=0.0,
    )
    assert result.ok is False
    assert result.reason == "panel_not_confirmed"
    assert result.target_point == (300, 120)
    assert capture.calls == 2
    assert actions.calls == [
        ("click_client_point", (270, 120), 0.0),
        ("click_client_point", (300, 120), 0.0),
    ]


def test_starfield_probe_active_surface_removes_small_candidate_fallback_config():
    starfield_config = RuntimeConfig().starfield
    assert not hasattr(starfield_config, "small_candidate_fallback_max_radius")
    assert not hasattr(starfield_config, "small_candidate_fallback_offset_x")
    assert not hasattr(starfield_config, "small_candidate_fallback_offset_y")
    parameters = inspect.signature(try_open_starfield_candidate_by_rank).parameters
    assert "small_candidate_fallback_max_radius" not in parameters
    assert "small_candidate_fallback_offset_x" not in parameters
    assert "small_candidate_fallback_offset_y" not in parameters


def test_starfield_probe_fails_closed_on_portrait_geometry_mismatch():
    actions = FakeActions()
    result = try_open_nearest_starfield_candidate(
        capture=FakeCapture(Image.new("RGB", (240, 320), (4, 8, 16))),
        actions=actions,
        reader=FakeReader(PlanetPanelState(planet_id=1, title="1. BALOR")),
        panel_is_readable=_panel_is_readable,
        settle_seconds=0.0,
        expected_orientation="landscape",
    )
    assert result.ok is False
    assert result.reason == "geometry_mismatch"
    assert actions.calls == []


def test_starfield_probe_selects_requested_rank_deterministically():
    actions = FakeActions()
    result = try_open_starfield_candidate_by_rank(
        capture=FakeCapture(_scene_image(ship_center=(160, 120), objects=((270, 120, 12), (300, 120, 14)))),
        actions=actions,
        reader=FakeReader(PlanetPanelState(planet_id=1, title="1. BALOR")),
        target_rank=2,
        panel_is_readable=_panel_is_readable,
        settle_seconds=0.0,
    )
    assert result.ok is True
    assert result.reason == "open_confirmed"
    assert result.rank == 2
    assert result.target_point == (300, 120)
    assert actions.calls[0] == ("click_client_point", (300, 120), 0.0)


def test_starfield_probe_failed_confirmation_is_not_success():
    result = try_open_nearest_starfield_candidate(
        capture=FakeCapture(_scene_image(ship_center=(160, 120), objects=((270, 120, 12),))),
        actions=FakeActions(),
        reader=FakeReader(PlanetPanelState()),
        panel_is_readable=_panel_is_readable,
        settle_seconds=0.0,
    )
    assert result.ok is False
    assert result.reason == "panel_not_visible"


def test_starfield_probe_requires_stricter_confirmation_when_provided():
    result = try_open_nearest_starfield_candidate(
        capture=FakeCapture(_scene_image(ship_center=(160, 120), objects=((270, 120, 12),))),
        actions=FakeActions(),
        reader=FakeReader(PlanetPanelState(title="Ship Speed", mining_cost=300, speed_cost=400)),
        panel_is_readable=_panel_is_readable,
        panel_is_confirmed=lambda panel: bool(panel and panel.title == "1. BALOR"),
        settle_seconds=0.0,
    )
    assert result.ok is False
    assert result.reason == "panel_not_confirmed"


def test_starfield_probe_fails_closed_when_ship_is_implausible():
    result = try_open_nearest_starfield_candidate(
        capture=FakeCapture(_scene_image(ship_center=(160, 120), ship_size=(220, 80), objects=((260, 120, 14),))),
        actions=FakeActions(),
        reader=FakeReader(PlanetPanelState(planet_id=1, title="1. BALOR")),
        panel_is_readable=_panel_is_readable,
        settle_seconds=0.0,
        max_ship_radius=72,
        max_ship_bbox_width=140,
        max_ship_bbox_height=90,
        max_ship_area_ratio=0.08,
    )
    assert result.ok is False
    assert result.reason == "ship_implausible"


def test_starfield_probe_fails_closed_when_ship_is_too_thin():
    actions = FakeActions()
    result = try_open_nearest_starfield_candidate(
        capture=FakeCapture(_scene_image(ship_center=(160, 120), ship_size=(31, 2), objects=((260, 120, 14),))),
        actions=actions,
        reader=FakeReader(PlanetPanelState(planet_id=1, title="1. BALOR")),
        panel_is_readable=_panel_is_readable,
        settle_seconds=0.0,
        min_ship_bbox_width=20,
        min_ship_bbox_height=8,
        min_ship_area=150,
    )
    assert result.ok is False
    assert result.reason == "ship_implausible"
    assert result.scene is not None
    assert result.scene.ship_reject_reason == "min_bbox_height"
    assert actions.calls == []


def test_discover_nearest_starfield_planet_resolves_canonical_identity():
    actions = FakeActions()
    result = discover_nearest_starfield_planet(
        capture=FakeCapture(_scene_image(ship_center=(160, 120), objects=((270, 120, 12), (300, 120, 14)))),
        actions=actions,
        reader=FakeReader(
            PlanetPanelState(
                planet_id=2,
                title="DRAŠTA",
                mining_level=2,
                mining_cost=233,
                speed_cost=106,
                cargo_cost=29,
            )
        ),
        panel_is_readable=_panel_is_readable,
        panel_is_confirmed=PlanetsTask._probe_panel_confirmed,
        return_to_starfield=lambda: True,
        settle_seconds=0.0,
    )
    assert result.ok is True
    assert result.reason == "ok"
    assert result.target_rank == 1
    assert result.target_point == (270, 120)
    assert result.planet_title_raw == "DRAŠTA"
    assert result.planet_title_canonical == "Drasta"
    assert result.planet_id == 2
    assert result.returned_to_starfield is True


def test_discover_starfield_planet_by_rank_selects_second_candidate():
    result = discover_starfield_planet_by_rank(
        capture=FakeCapture(_scene_image(ship_center=(160, 120), objects=((270, 120, 12), (300, 120, 14)))),
        actions=FakeActions(),
        reader=FakeReader(
            PlanetPanelState(
                planet_id=4,
                title="DHOLEN",
                mining_level=2,
                mining_cost=233,
                speed_cost=106,
                cargo_cost=29,
            )
        ),
        target_rank=2,
        panel_is_readable=_panel_is_readable,
        panel_is_confirmed=PlanetsTask._probe_panel_confirmed,
        return_to_starfield=lambda: True,
        settle_seconds=0.0,
    )
    assert result.ok is True
    assert result.reason == "ok"
    assert result.target_rank == 2
    assert result.target_point == (300, 120)
    assert result.planet_title_canonical == "Dholen"


def test_discover_starfield_planet_by_rank_fails_closed_when_rank_is_unavailable():
    actions = FakeActions()
    result = discover_starfield_planet_by_rank(
        capture=FakeCapture(_scene_image(ship_center=(160, 120), objects=((270, 120, 12),))),
        actions=actions,
        reader=FakeReader(PlanetPanelState(planet_id=4, title="DHOLEN", mining_level=2, mining_cost=233)),
        target_rank=3,
        panel_is_readable=_panel_is_readable,
        panel_is_confirmed=PlanetsTask._probe_panel_confirmed,
        return_to_starfield=lambda: True,
        settle_seconds=0.0,
    )
    assert result.ok is False
    assert result.reason == "candidate_rank_unavailable"
    assert result.target_rank == 3
    assert result.target_point is None
    assert actions.calls == []


def test_discover_nearest_starfield_planet_fails_when_identity_is_unresolved():
    result = discover_nearest_starfield_planet(
        capture=FakeCapture(_scene_image(ship_center=(160, 120), objects=((270, 120, 12),))),
        actions=FakeActions(),
        reader=FakeReader(PlanetPanelState(title="MYSTERY", mining_level=2, mining_cost=233)),
        panel_is_readable=_panel_is_readable,
        panel_is_confirmed=lambda panel: True,
        settle_seconds=0.0,
    )
    assert result.ok is False
    assert result.reason == "planet_identity_unresolved"
    assert result.planet_title_raw == "MYSTERY"
    assert result.planet_title_canonical is None


def test_discover_nearest_starfield_planet_preserves_fail_closed_reason():
    open_panel = PlanetPanelState(planet_id=2, title="DRAŠTA", mining_level=2, mining_cost=233)
    result = discover_nearest_starfield_planet(
        capture=FakeCapture(_scene_image(ship_center=(160, 120), objects=((270, 120, 12),))),
        actions=FakeActions(),
        reader=FakeReader(open_panel),
        panel_is_readable=_panel_is_readable,
        starfield_ready_check=lambda: ("not_starfield_ready", open_panel),
        panel_is_confirmed=PlanetsTask._probe_panel_confirmed,
        settle_seconds=0.0,
    )
    assert result.ok is False
    assert result.reason == "not_starfield_ready"
    assert result.target_point is None


def test_discover_nearest_starfield_planet_reports_return_to_starfield_failure():
    actions = FakeActions(close_ok=False)

    def _return_to_starfield() -> bool:
        return actions.close_planet_panel()

    result = discover_nearest_starfield_planet(
        capture=FakeCapture(_scene_image(ship_center=(160, 120), objects=((270, 120, 12),))),
        actions=actions,
        reader=FakeReader(
            PlanetPanelState(
                planet_id=4,
                title="DHOLEN",
                mining_level=2,
                mining_cost=233,
                speed_cost=106,
                cargo_cost=29,
            )
        ),
        panel_is_readable=_panel_is_readable,
        panel_is_confirmed=PlanetsTask._probe_panel_confirmed,
        return_to_starfield=_return_to_starfield,
        settle_seconds=0.0,
    )
    assert result.ok is True
    assert result.reason == "return_to_starfield_failed"
    assert result.planet_title_canonical == "Dholen"
    assert result.returned_to_starfield is False
    assert ("close_planet_panel",) in actions.calls


def test_discover_nearest_starfield_planet_returns_to_starfield_after_success():
    actions = FakeActions(close_ok=True)
    reader = FakeReader(
        [
            PlanetPanelState(
                planet_id=4,
                title="DHOLEN",
                mining_level=2,
                mining_cost=233,
                speed_cost=106,
                cargo_cost=29,
            ),
            PlanetPanelState(),
        ]
    )

    def _return_to_starfield() -> bool:
        if not actions.close_planet_panel():
            return False
        return not _panel_is_readable(reader.read())

    result = discover_nearest_starfield_planet(
        capture=FakeCapture(_scene_image(ship_center=(160, 120), objects=((270, 120, 12),))),
        actions=actions,
        reader=reader,
        panel_is_readable=_panel_is_readable,
        panel_is_confirmed=PlanetsTask._probe_panel_confirmed,
        return_to_starfield=_return_to_starfield,
        settle_seconds=0.0,
    )
    assert result.ok is True
    assert result.reason == "ok"
    assert result.planet_title_canonical == "Dholen"
    assert result.returned_to_starfield is True
    assert ("close_planet_panel",) in actions.calls


def test_resolve_starfield_viewport_converts_normalized_bounds():
    assert resolve_starfield_viewport((400, 300), (0.1, 0.2, 0.9, 0.8)) == (40, 60, 360, 240)


def test_resolve_starfield_exclusion_zones_converts_bounds_inside_viewport():
    viewport = (40, 60, 360, 240)
    assert resolve_starfield_exclusion_zones((400, 300), viewport, ((0.9, 0.0, 1.0, 1.0),)) == ((288, 0, 320, 180),)


def test_planets_task_debug_flag_disabled_keeps_existing_path():
    class Actions:
        def __init__(self):
            self.calls = []

        def reset_ui(self):
            self.calls.append(("reset_ui",))

        def open_planet_menu(self):
            self.calls.append(("open_planet_menu",))
            return True

    class Reader:
        def read(self):
            return PlanetPanelState(planet_id=1, title="1. BALOR")

    class StateReader:
        def read(self):
            from ipm.state import GameSnapshot

            return GameSnapshot(cash=0, current_planet=PlanetPanelState(planet_id=1, title="1. BALOR"))

    class Navigator:
        def __init__(self, reader, actions, *, max_planets=16):
            self.reader = reader
            self.actions = actions

        def scan_visible_planets(self, initial_panel=None):
            return type("Scan", (), {"planets": {1: PlanetPanelState(planet_id=1, title="1. BALOR")}, "order": [1]})()

        def go_to_planet(self, target_id, order, known_planets=None):
            return target_id in order

    import ipm.tasks.planets as planets_task_module

    original = planets_task_module.PlanetNavigator
    planets_task_module.PlanetNavigator = Navigator
    try:
        actions = Actions()
        runtime_config = RuntimeConfig()
        runtime_config.starfield.enable_click_probe = False
        result = PlanetsTask(
            reader=Reader(),
            state_reader=StateReader(),
            actions=actions,
            capture=FakeCapture(_scene_image(ship_center=(160, 120), objects=((270, 120, 12),))),
            config=runtime_config,
        ).run()
    finally:
        planets_task_module.PlanetNavigator = original
    assert ("open_planet_menu",) in actions.calls
    assert result.ok is True


def test_planets_task_probe_confirmation_requires_plausible_title_and_costs():
    assert PlanetsTask._probe_panel_confirmed(PlanetPanelState(title="1. BALOR", mining_cost=100, speed_cost=200)) is True
    assert PlanetsTask._probe_panel_confirmed(PlanetPanelState(title="", mining_cost=100, speed_cost=200)) is False
    assert PlanetsTask._probe_panel_confirmed(PlanetPanelState(title="Ship Speed", mining_cost=100, speed_cost=200)) is False


def test_planets_task_probe_confirmation_accepts_partial_upgrade_data_with_plausible_level():
    assert (
        PlanetsTask._probe_panel_confirmed(
            PlanetPanelState(
                title="DRAŠTA",
                mining_level=2,
                mining_cost=233,
                speed_cost=106,
                cargo_cost=None,
            )
        )
        is True
    )


def test_planets_task_probe_precondition_rejects_open_planet_panel():
    assert (
        PlanetsTask._probe_precondition_failure_reason(
            PlanetPanelState(
                planet_id=2,
                title="DRAŠTA",
                mining_level=2,
                mining_cost=233,
            )
        )
        == "not_starfield_ready"
    )
    assert PlanetsTask._probe_precondition_failure_reason(PlanetPanelState(title="Ship Speed", mining_cost=100)) is None
