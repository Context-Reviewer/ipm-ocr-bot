from PIL import Image

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
        self.image = image

    def capture_screen(self):
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


def _scene_image(*, ship_center=None, ship_size=(36, 18), objects=()):
    image = Image.new("RGB", (320, 240), (4, 8, 16))
    from PIL import ImageDraw

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


def test_starfield_probe_fails_closed_when_template_detection_misses_without_fallback():
    actions = FakeActions()
    result = try_open_nearest_starfield_candidate(
        capture=FakeCapture(_scene_image(ship_center=(160, 120), objects=((200, 120, 12),))),
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


def test_starfield_probe_can_fallback_when_template_search_region_misses():
    actions = FakeActions()
    result = try_open_nearest_starfield_candidate(
        capture=FakeCapture(_scene_image(ship_center=(160, 120), objects=((200, 120, 12),))),
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
    assert result.target_point == (200, 120)


def test_starfield_probe_fails_closed_when_not_starfield_ready():
    actions = FakeActions()
    open_panel = PlanetPanelState(planet_id=2, title="DRAŠTA", mining_level=2, mining_cost=233)
    result = try_open_nearest_starfield_candidate(
        capture=FakeCapture(_scene_image(ship_center=(160, 120), objects=((200, 120, 12),))),
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
        capture=FakeCapture(_scene_image(ship_center=(160, 120), objects=((200, 120, 12),))),
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
        capture=FakeCapture(_scene_image(ship_center=(160, 120), objects=((200, 120, 12), (250, 120, 14)))),
        actions=actions,
        reader=FakeReader(PlanetPanelState(planet_id=1, title="1. BALOR")),
        panel_is_readable=_panel_is_readable,
        settle_seconds=0.0,
    )
    assert result.ok is True
    assert result.reason == "open_confirmed"
    assert result.target_point == (200, 120)
    assert actions.calls[0][0] == "click_client_point"


def test_starfield_probe_selects_requested_rank_deterministically():
    actions = FakeActions()
    result = try_open_starfield_candidate_by_rank(
        capture=FakeCapture(_scene_image(ship_center=(160, 120), objects=((200, 120, 12), (250, 120, 14)))),
        actions=actions,
        reader=FakeReader(PlanetPanelState(planet_id=1, title="1. BALOR")),
        target_rank=2,
        panel_is_readable=_panel_is_readable,
        settle_seconds=0.0,
    )
    assert result.ok is True
    assert result.reason == "open_confirmed"
    assert result.rank == 2
    assert result.target_point == (250, 120)
    assert actions.calls[0] == ("click_client_point", (250, 120), 0.0)


def test_starfield_probe_failed_confirmation_is_not_success():
    result = try_open_nearest_starfield_candidate(
        capture=FakeCapture(_scene_image(ship_center=(160, 120), objects=((200, 120, 12),))),
        actions=FakeActions(),
        reader=FakeReader(PlanetPanelState()),
        panel_is_readable=_panel_is_readable,
        settle_seconds=0.0,
    )
    assert result.ok is False
    assert result.reason == "panel_not_visible"


def test_starfield_probe_requires_stricter_confirmation_when_provided():
    result = try_open_nearest_starfield_candidate(
        capture=FakeCapture(_scene_image(ship_center=(160, 120), objects=((200, 120, 12),))),
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
        capture=FakeCapture(_scene_image(ship_center=(160, 120), objects=((200, 120, 12), (250, 120, 14)))),
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
    assert result.target_point == (200, 120)
    assert result.planet_title_raw == "DRAŠTA"
    assert result.planet_title_canonical == "Drasta"
    assert result.planet_id == 2
    assert result.returned_to_starfield is True


def test_discover_starfield_planet_by_rank_selects_second_candidate():
    result = discover_starfield_planet_by_rank(
        capture=FakeCapture(_scene_image(ship_center=(160, 120), objects=((200, 120, 12), (250, 120, 14)))),
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
    assert result.target_point == (250, 120)
    assert result.planet_title_canonical == "Dholen"


def test_discover_starfield_planet_by_rank_fails_closed_when_rank_is_unavailable():
    actions = FakeActions()
    result = discover_starfield_planet_by_rank(
        capture=FakeCapture(_scene_image(ship_center=(160, 120), objects=((200, 120, 12),))),
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
        capture=FakeCapture(_scene_image(ship_center=(160, 120), objects=((200, 120, 12),))),
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
        capture=FakeCapture(_scene_image(ship_center=(160, 120), objects=((200, 120, 12),))),
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
        capture=FakeCapture(_scene_image(ship_center=(160, 120), objects=((200, 120, 12),))),
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
        capture=FakeCapture(_scene_image(ship_center=(160, 120), objects=((200, 120, 12),))),
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

        def scan_visible_planets(self):
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
            capture=FakeCapture(_scene_image(ship_center=(160, 120), objects=((200, 120, 12),))),
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
