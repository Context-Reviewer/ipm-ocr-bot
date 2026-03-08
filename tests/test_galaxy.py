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

    def read(self):
        value = self.panels[min(self.index, len(self.panels) - 1)]
        self.index += 1
        return value


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
    reader = FakeReader(
        [
            PlanetPanelState(planet_id=3, title="3. ACHERON"),
            PlanetPanelState(planet_id=2, title="2. DRASTA"),
            PlanetPanelState(planet_id=1, title="1. BALOR"),
        ]
    )
    assert PlanetNavigator(reader, FakeActions(), max_planets=4).go_to_planet(1, [1, 2, 3, 4], known) is True


def test_scan_normalizes_zero_id_to_expected_next_planet():
    panels = [
        PlanetPanelState(planet_id=8, title="8. ACHERON"),
        PlanetPanelState(planet_id=9, title="9. YANGTZE"),
        PlanetPanelState(planet_id=0, title="0. SOLVEIG"),
    ]
    scan = PlanetNavigator(FakeReader(panels), FakeActions(), max_planets=3).scan_visible_planets()
    assert scan.order == [8, 9, 10]
    assert scan.planets[10].title == "10. SOLVEIG"
