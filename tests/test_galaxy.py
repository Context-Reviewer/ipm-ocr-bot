from ipm.galaxy import PlanetNavigator
from ipm.state import PlanetPanelState


class FakeActions:
    def next_planet(self):
        return True

    def previous_planet(self):
        return True


class FakeReader:
    def __init__(self, panels):
        self.panels = list(panels)
        self.index = 0
        self.read_calls = 0

    def read(self):
        self.read_calls += 1
        value = self.panels[min(self.index, len(self.panels) - 1)]
        self.index += 1
        return value


class CycleActions:
    def __init__(self, world):
        self.world = world

    def next_planet(self):
        self.world.index = (self.world.index + 1) % len(self.world.panels)
        return True

    def previous_planet(self):
        self.world.index = (self.world.index - 1) % len(self.world.panels)
        return True


class CycleReader:
    def __init__(self, world):
        self.world = world

    def read(self):
        return self.world.panels[self.world.index]


def test_scan_infers_next_planet_when_title_name_disagrees_with_duplicate_id():
    panels = [
        PlanetPanelState(planet_id=1, title="1. BALOR"),
        PlanetPanelState(planet_id=2, title="2. DRASTA"),
        PlanetPanelState(planet_id=3, title="3. ANADIUS"),
        PlanetPanelState(planet_id=3, title="3. ACHERON"),
    ]
    scan = PlanetNavigator(FakeReader(panels), FakeActions(), max_planets=4).scan_visible_planets()
    assert scan.order == [1, 2, 3, 4]
    assert scan.planets[4].title == "4. ACHERON"


def test_go_to_planet_uses_title_name_when_numeric_id_is_wrong():
    known = {
        1: PlanetPanelState(planet_id=1, title="1. BALOR"),
        2: PlanetPanelState(planet_id=2, title="2. DRASTA"),
        3: PlanetPanelState(planet_id=3, title="3. ANADIUS"),
        4: PlanetPanelState(planet_id=4, title="4. ACHERON"),
    }
    world = type(
        "World",
        (),
        {
            "panels": [
                PlanetPanelState(planet_id=3, title="3. ACHERON"),
                PlanetPanelState(planet_id=1, title="1. BALOR"),
                PlanetPanelState(planet_id=2, title="2. DRASTA"),
                PlanetPanelState(planet_id=3, title="3. ANADIUS"),
            ],
            "index": 0,
        },
    )()
    navigator = PlanetNavigator(CycleReader(world), CycleActions(world), max_planets=4)
    assert navigator.go_to_planet(1, [1, 2, 3, 4], known) is True


def test_scan_normalizes_zero_id_to_expected_next_planet():
    panels = [
        PlanetPanelState(planet_id=8, title="8. ACHERON"),
        PlanetPanelState(planet_id=9, title="9. YANGTZE"),
        PlanetPanelState(planet_id=0, title="0. SOLVEIG"),
    ]
    scan = PlanetNavigator(FakeReader(panels), FakeActions(), max_planets=3).scan_visible_planets()
    assert scan.order == [8, 9, 10]
    assert scan.planets[10].title == "10. SOLVEIG"


def test_scan_stops_at_known_planet_wrap_instead_of_inventing_high_ids():
    panels = [
        PlanetPanelState(planet_id=3, title="3. ANADIUS"),
        PlanetPanelState(planet_id=4, title="4. DHOLEN"),
        PlanetPanelState(planet_id=1, title="1. BALOR"),
    ]
    scan = PlanetNavigator(FakeReader(panels), FakeActions(), max_planets=4).scan_visible_planets()
    assert scan.complete_loop is True
    assert scan.order == [3, 4]
    assert 5 not in scan.planets
    assert scan.final_panel.planet_id == 4


def test_scan_restores_last_valid_panel_before_go_to_planet_after_wrap():
    world = type(
        "World",
        (),
        {
            "panels": [
                PlanetPanelState(planet_id=4, title="4. DHOLEN"),
                PlanetPanelState(planet_id=5, title="5. VERR"),
                PlanetPanelState(planet_id=6, title="6. NEWTON"),
                PlanetPanelState(planet_id=1, title="1. BALOR"),
            ],
            "index": 0,
        },
    )()
    navigator = PlanetNavigator(CycleReader(world), CycleActions(world), max_planets=4)
    scan = navigator.scan_visible_planets()
    assert scan.order == [4, 5, 6]
    assert navigator.current().planet_id == 6
    assert navigator.go_to_planet(5, scan.order, scan.planets) is True


def test_go_to_planet_uses_alternate_direction_for_truncated_live_order():
    world = type(
        "World",
        (),
        {
            "panels": [
                PlanetPanelState(planet_id=4, title="DHOLEN"),
                PlanetPanelState(planet_id=5, title="VERR"),
                PlanetPanelState(planet_id=6, title="NEWTON"),
                PlanetPanelState(planet_id=7, title="WIDOW"),
                PlanetPanelState(planet_id=8, title="ACHERON"),
                PlanetPanelState(planet_id=9, title="YANGTZE"),
                PlanetPanelState(planet_id=10, title="SOLVEIG"),
                PlanetPanelState(planet_id=11, title="IMIR"),
                PlanetPanelState(planet_id=12, title="RELIC"),
                PlanetPanelState(planet_id=13, title="NITH"),
                PlanetPanelState(planet_id=1, title="BALOR"),
                PlanetPanelState(planet_id=2, title="DRASTA"),
                PlanetPanelState(planet_id=3, title="ANADIUS"),
            ],
            "index": 9,
        },
    )()
    navigator = PlanetNavigator(CycleReader(world), CycleActions(world), max_planets=16)
    order = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    known = {panel.planet_id: panel for panel in world.panels if panel.planet_id in order}
    assert navigator.current().title == "NITH"
    assert navigator.go_to_planet(5, order, known) is True
    assert navigator.current().planet_id == 5
    assert navigator.current().title == "VERR"


def test_scan_visible_planets_can_reuse_initial_panel_without_extra_current_read():
    first = PlanetPanelState(planet_id=1, title="1. BALOR")
    reader = FakeReader(
        [
            PlanetPanelState(planet_id=2, title="2. DRASTA"),
            PlanetPanelState(planet_id=3, title="3. ANADIUS"),
            PlanetPanelState(planet_id=1, title="1. BALOR"),
        ]
    )
    scan = PlanetNavigator(reader, FakeActions(), max_planets=4).scan_visible_planets(initial_panel=first)
    assert scan.order == [1, 2, 3]
    assert scan.final_panel.planet_id == 3
    assert reader.read_calls == 3


def test_go_to_planet_can_reuse_current_panel_without_initial_read():
    known = {
        1: PlanetPanelState(planet_id=1, title="1. BALOR"),
        2: PlanetPanelState(planet_id=2, title="2. DRASTA"),
        3: PlanetPanelState(planet_id=3, title="3. ANADIUS"),
        4: PlanetPanelState(planet_id=4, title="4. ACHERON"),
    }
    world = type(
        "World",
        (),
        {
            "panels": [
                PlanetPanelState(planet_id=1, title="1. BALOR"),
                PlanetPanelState(planet_id=2, title="2. DRASTA"),
                PlanetPanelState(planet_id=3, title="3. ANADIUS"),
                PlanetPanelState(planet_id=4, title="4. ACHERON"),
            ],
            "index": 2,
        },
    )()
    reader = CycleReader(world)
    navigator = PlanetNavigator(reader, CycleActions(world), max_planets=4)
    assert navigator.go_to_planet(
        1,
        [1, 2, 3, 4],
        known,
        current_panel=PlanetPanelState(planet_id=3, title="3. ANADIUS"),
    ) is True
    assert reader.read().planet_id == 1


def test_go_to_planet_restores_with_single_read_after_failed_primary_route():
    known = {
        1: PlanetPanelState(planet_id=1, title="1. BALOR"),
        2: PlanetPanelState(planet_id=2, title="2. DRASTA"),
        3: PlanetPanelState(planet_id=3, title="3. ANADIUS"),
        4: PlanetPanelState(planet_id=4, title="4. ACHERON"),
    }

    class World:
        def __init__(self):
            self.panels = [
                PlanetPanelState(planet_id=3, title="3. ANADIUS"),
                PlanetPanelState(planet_id=4, title="4. ACHERON"),
                PlanetPanelState(planet_id=1, title="1. BALOR"),
                PlanetPanelState(planet_id=2, title="2. DRASTA"),
            ]
            self.index = 0

    class FlakyActions(CycleActions):
        def __init__(self, world):
            super().__init__(world)
            self.next_calls = 0

        def next_planet(self):
            self.next_calls += 1
            self.world.index = (self.world.index + 2) % len(self.world.panels)
            return True

        def previous_planet(self):
            self.world.index = (self.world.index - 2) % len(self.world.panels)
            return True

    world = World()
    reader = CycleReader(world)
    actions = FlakyActions(world)
    navigator = PlanetNavigator(reader, actions, max_planets=4)
    assert navigator.go_to_planet(1, [1, 2, 3, 4], known, current_panel=world.panels[world.index]) is False
    assert world.index == 0
