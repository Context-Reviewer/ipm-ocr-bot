from PIL import Image

from ipm.starfield_probe import StarfieldProbeResult, try_open_nearest_starfield_candidate
from ipm.state import PlanetPanelState
from ipm.tasks import PlanetsTask
from ipm.config import RuntimeConfig


class FakeCapture:
    def __init__(self, image):
        self.image = image

    def capture_screen(self):
        return self.image


class FakeActions:
    def __init__(self, click_ok=True):
        self.calls = []
        self.click_ok = click_ok

    def click_client_point(self, point, *, delay=None):
        self.calls.append(("click_client_point", point, delay))
        return self.click_ok


class FakeReader:
    def __init__(self, panel):
        self.panel = panel
        self.calls = 0

    def read(self):
        self.calls += 1
        return self.panel


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
        result = PlanetsTask(
            reader=Reader(),
            state_reader=StateReader(),
            actions=actions,
            capture=FakeCapture(_scene_image(ship_center=(160, 120), objects=((200, 120, 12),))),
            config=RuntimeConfig(),
        ).run()
    finally:
        planets_task_module.PlanetNavigator = original
    assert ("open_planet_menu",) in actions.calls
    assert result.ok is True
