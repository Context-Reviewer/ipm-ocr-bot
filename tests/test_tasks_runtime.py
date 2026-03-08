from ipm.config import PolicyConfig, RuntimeConfig
from ipm.state import GameSnapshot, OreRowState, PlanetPanelState, SellDialogState
from ipm.tasks import OresTask, PlanetsTask


class FakeStateReader:
    def __init__(self, snapshots):
        self.snapshots = list(snapshots)
        self.index = 0

    def read(self):
        value = self.snapshots[min(self.index, len(self.snapshots) - 1)]
        self.index += 1
        return value


class FakeActions:
    def __init__(self):
        self.calls = []

    def reset_ui(self):
        self.calls.append(("reset_ui",))

    def open_planet_menu(self):
        self.calls.append(("open_planet_menu",))
        return True

    def increase_planet_stat(self, stat):
        self.calls.append(("increase_planet_stat", stat))
        return True

    def open_ores_panel(self):
        self.calls.append(("open_ores_panel",))
        return True

    def select_ore_row(self, row_index):
        self.calls.append(("select_ore_row", row_index))
        return True

    def open_sell_dialog(self):
        self.calls.append(("open_sell_dialog",))
        return True

    def choose_sell_fraction(self, fraction):
        self.calls.append(("choose_sell_fraction", round(fraction, 2)))
        return True

    def execute_sell(self):
        self.calls.append(("execute_sell",))
        return True

    def close_ores_panel(self):
        self.calls.append(("close_ores_panel",))


class FakeNavigator:
    def __init__(self, reader, actions, *, max_planets=16):
        self.reader = reader
        self.actions = actions
        self.max_planets = max_planets

    def scan_visible_planets(self):
        panel = self.reader.read()
        return type(
            "Scan",
            (),
            {
                "planets": {panel.planet_id: panel},
                "order": [panel.planet_id],
                "complete_loop": True,
            },
        )()

    def go_to_planet(self, target_id, order, known_planets=None):
        self.actions.calls.append(("go_to_planet", target_id, tuple(order)))
        return target_id in order


class FakePlanetReader:
    def __init__(self, panels):
        self.panels = list(panels)
        self.index = 0

    def read(self):
        value = self.panels[min(self.index, len(self.panels) - 1)]
        self.index += 1
        return value


def test_planets_task_executes_upgrade_decision(monkeypatch):
    before = GameSnapshot(
        cash=500,
        current_planet=PlanetPanelState(
            planet_id=1,
            mining_level=8,
            speed_level=6,
            cargo_level=7,
            mining_cost=400,
            speed_cost=200,
            cargo_cost=250,
        ),
    )
    after = GameSnapshot(
        cash=300,
        current_planet=PlanetPanelState(
            planet_id=1,
            mining_level=8,
            speed_level=7,
            cargo_level=7,
            mining_cost=400,
            speed_cost=240,
            cargo_cost=250,
        ),
    )
    monkeypatch.setattr("ipm.tasks.planets.PlanetNavigator", FakeNavigator)
    actions = FakeActions()
    reader = FakePlanetReader([before.current_planet, before.current_planet, after.current_planet])
    task = PlanetsTask(
        reader=reader,
        state_reader=FakeStateReader([before, after]),
        actions=actions,
        config=RuntimeConfig(policy=PolicyConfig(max_planet_upgrades_per_task=1)),
    )
    result = task.run()
    assert result.details["decision"]["stat"] == "S"
    assert result.details["executed"] is True
    assert result.details["verified"] is True
    assert ("increase_planet_stat", "S") in actions.calls
    assert ("go_to_planet", 1, (1,)) in actions.calls
    assert len(result.details["steps"]) == 1


def test_planets_task_fails_fast_when_planet_panel_never_becomes_readable(monkeypatch):
    unreadable = PlanetPanelState()
    monkeypatch.setattr("ipm.tasks.planets.PlanetNavigator", FakeNavigator)
    actions = FakeActions()
    reader = FakePlanetReader([unreadable, unreadable, unreadable])
    task = PlanetsTask(
        reader=reader,
        state_reader=FakeStateReader([GameSnapshot(cash=500, current_planet=unreadable)]),
        actions=actions,
        config=RuntimeConfig(policy=PolicyConfig(planet_panel_open_attempts=3)),
    )
    result = task.run()
    assert result.ok is False
    assert result.details["error"] == "planet_panel_unreadable"


def test_planets_task_replans_for_multiple_verified_upgrades(monkeypatch):
    start = PlanetPanelState(
        planet_id=2,
        mining_level=8,
        speed_level=6,
        cargo_level=7,
        mining_cost=400,
        speed_cost=200,
        cargo_cost=250,
    )
    after_first = PlanetPanelState(
        planet_id=2,
        mining_level=8,
        speed_level=7,
        cargo_level=7,
        mining_cost=400,
        speed_cost=240,
        cargo_cost=250,
    )
    after_second = PlanetPanelState(
        planet_id=2,
        mining_level=8,
        speed_level=8,
        cargo_level=7,
        mining_cost=400,
        speed_cost=280,
        cargo_cost=250,
    )
    monkeypatch.setattr("ipm.tasks.planets.PlanetNavigator", FakeNavigator)
    actions = FakeActions()
    reader = FakePlanetReader([start, start, after_first, after_second])
    task = PlanetsTask(
        reader=reader,
        state_reader=FakeStateReader(
            [
                GameSnapshot(cash=500, current_planet=start),
                GameSnapshot(cash=300, current_planet=after_first),
                GameSnapshot(cash=60, current_planet=after_second),
            ]
        ),
        actions=actions,
        config=RuntimeConfig(),
    )
    result = task.run()
    assert result.ok is True
    assert [step["decision"]["stat"] for step in result.details["steps"]] == ["S", "S"]
    assert result.details["steps"][-1]["verified"] is True
    assert actions.calls.count(("increase_planet_stat", "S")) == 2


def test_planets_task_verifies_upgrade_from_cost_increase_when_level_ocr_misses(monkeypatch):
    before = GameSnapshot(
        cash=1000,
        current_planet=PlanetPanelState(
            planet_id=1,
            mining_level=40,
            speed_level=32,
            cargo_level=19,
            mining_cost=138920,
            speed_cost=17030,
            cargo_cost=562,
        ),
    )
    after = GameSnapshot(
        cash=400,
        current_planet=PlanetPanelState(
            planet_id=1,
            mining_level=40,
            speed_level=32,
            cargo_level=None,
            mining_cost=138920,
            speed_cost=17030,
            cargo_cost=731,
        ),
    )
    monkeypatch.setattr("ipm.tasks.planets.PlanetNavigator", FakeNavigator)
    actions = FakeActions()
    reader = FakePlanetReader([before.current_planet, before.current_planet, after.current_planet])
    task = PlanetsTask(
        reader=reader,
        state_reader=FakeStateReader([before, after]),
        actions=actions,
        config=RuntimeConfig(policy=PolicyConfig(max_planet_upgrades_per_task=1)),
    )
    result = task.run()
    assert result.details["decision"]["stat"] == "C"
    assert result.details["executed"] is True
    assert result.details["verified"] is True


def test_ores_task_executes_sell_decision():
    before = GameSnapshot(
        ore_rows={
            2: OreRowState(ore_name="Iron", quantity=70000),
        },
        sell_dialog=SellDialogState(),
    )
    after = GameSnapshot(
        ore_rows={
            2: OreRowState(ore_name="Iron", quantity=25000),
        },
        sell_dialog=SellDialogState(),
    )
    actions = FakeActions()
    task = OresTask(
        ore_reader=object(),
        sell_reader=None,
        state_reader=FakeStateReader([before, after]),
        actions=actions,
        config=RuntimeConfig(policy=PolicyConfig(ore_sell_confirm_reads=1)),
    )
    result = task.run()
    assert result.details["decision"]["row_index"] == 2
    assert result.details["executed"] is True
    assert result.details["verified"] is True
    assert ("select_ore_row", 2) in actions.calls


def test_ores_task_blocks_sell_when_confirmation_reads_disagree():
    first = GameSnapshot(
        ore_rows={
            1: OreRowState(ore_name="Copper", quantity=30000),
        },
        sell_dialog=SellDialogState(),
    )
    second = GameSnapshot(
        ore_rows={
            1: OreRowState(ore_name="Copper", quantity=90000),
        },
        sell_dialog=SellDialogState(),
    )
    third = GameSnapshot(
        ore_rows={
            1: OreRowState(ore_name="Copper", quantity=31000),
        },
        sell_dialog=SellDialogState(),
    )
    actions = FakeActions()
    task = OresTask(
        ore_reader=object(),
        sell_reader=None,
        state_reader=FakeStateReader([first, second, third]),
        actions=actions,
        config=RuntimeConfig(policy=PolicyConfig(ore_sell_confirm_reads=3, ore_sell_max_relative_spread=0.20)),
    )
    result = task.run()
    assert result.details["decision"] is None
    assert result.details["action_results"]["confirmed"] is False
    assert result.details["executed"] is False
