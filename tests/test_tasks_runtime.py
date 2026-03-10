from dataclasses import replace

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

    def read_planet_snapshot(self):
        return self.read()


class CountingStateReader(FakeStateReader):
    def __init__(self, snapshots):
        super().__init__(snapshots)
        self.read_planet_snapshot_calls = 0

    def read_planet_snapshot(self):
        self.read_planet_snapshot_calls += 1
        return super().read_planet_snapshot()


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

    def scan_visible_planets(self, initial_panel=None):
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

    def go_to_planet(self, target_id, order, known_planets=None, current_panel=None):
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


def _runtime_config_without_starfield_probe(*, policy=None):
    runtime_config = RuntimeConfig(policy=policy or PolicyConfig())
    runtime_config.starfield.enable_click_probe = False
    return runtime_config


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
    reader = FakePlanetReader([before.current_planet, before.current_planet, before.current_planet])
    task = PlanetsTask(
        reader=reader,
        state_reader=FakeStateReader([before, after]),
        actions=actions,
        config=_runtime_config_without_starfield_probe(policy=PolicyConfig(max_planet_upgrades_per_task=1)),
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
        config=_runtime_config_without_starfield_probe(policy=PolicyConfig(planet_panel_open_attempts=3)),
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
    reader = FakePlanetReader([start, start, start, after_first, after_second])
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
        config=_runtime_config_without_starfield_probe(),
    )
    result = task.run()
    assert result.ok is True
    assert [step["decision"]["stat"] for step in result.details["steps"]] == ["S", "S"]
    assert result.details["steps"][-1]["verified"] is True
    assert actions.calls.count(("increase_planet_stat", "S")) == 2


def test_planets_task_retries_after_reads_for_later_same_planet_upgrade(monkeypatch):
    before = PlanetPanelState(
        planet_id=5,
        title="5. VERR",
        mining_level=12,
        speed_level=9,
        cargo_level=9,
        mining_cost=4480,
        speed_cost=2040,
        cargo_cost=2040,
    )
    stale_after = PlanetPanelState(
        planet_id=5,
        title="5. VERR",
        mining_level=12,
        speed_level=9,
        cargo_level=9,
        mining_cost=4480,
        speed_cost=2040,
        cargo_cost=2040,
    )
    verified_after = PlanetPanelState(
        planet_id=5,
        title="5. VERR",
        mining_level=12,
        speed_level=9,
        cargo_level=10,
        mining_cost=4480,
        speed_cost=2040,
        cargo_cost=2650,
    )
    monkeypatch.setattr("ipm.tasks.planets.PlanetNavigator", FakeNavigator)
    actions = FakeActions()
    reader = FakePlanetReader([before, before, before])
    task = PlanetsTask(
        reader=reader,
        state_reader=FakeStateReader(
            [
                GameSnapshot(cash=5000, current_planet=before),
                GameSnapshot(cash=5000, current_planet=stale_after),
                GameSnapshot(cash=2960, current_planet=verified_after),
            ]
        ),
        actions=actions,
        config=_runtime_config_without_starfield_probe(
            policy=PolicyConfig(max_planet_upgrades_per_task=1, planet_upgrade_confirm_reads=2)
        ),
    )
    result = task.run()
    assert result.ok is True
    assert result.details["executed"] is True
    assert result.details["verified"] is True
    assert result.details["decision"]["planet_id"] == 5
    assert result.details["decision"]["stat"] == "C"
    assert result.details["steps"][0]["after_planet"]["cargo_level"] == 10


def test_planets_task_stops_confirmation_reads_after_first_verified_snapshot(monkeypatch):
    before = GameSnapshot(
        cash=500,
        current_planet=PlanetPanelState(
            planet_id=1,
            title="1. BALOR",
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
            title="1. BALOR",
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
    reader = FakePlanetReader([before.current_planet, before.current_planet, before.current_planet])
    state_reader = CountingStateReader([before, after, after, after])
    task = PlanetsTask(
        reader=reader,
        state_reader=state_reader,
        actions=actions,
        config=_runtime_config_without_starfield_probe(policy=PolicyConfig(max_planet_upgrades_per_task=1)),
    )
    result = task.run()
    assert result.ok is True
    assert result.details["verified"] is True
    assert state_reader.read_planet_snapshot_calls == 2


def test_planets_task_stops_confirmation_reads_after_second_verified_snapshot(monkeypatch):
    before = GameSnapshot(
        cash=5000,
        current_planet=PlanetPanelState(
            planet_id=5,
            title="5. VERR",
            mining_level=12,
            speed_level=9,
            cargo_level=9,
            mining_cost=4480,
            speed_cost=2040,
            cargo_cost=2040,
        ),
    )
    stale_after = GameSnapshot(
        cash=5000,
        current_planet=PlanetPanelState(
            planet_id=5,
            title="5. VERR",
            mining_level=12,
            speed_level=9,
            cargo_level=9,
            mining_cost=4480,
            speed_cost=2040,
            cargo_cost=2040,
        ),
    )
    verified_after = GameSnapshot(
        cash=2960,
        current_planet=PlanetPanelState(
            planet_id=5,
            title="5. VERR",
            mining_level=12,
            speed_level=9,
            cargo_level=10,
            mining_cost=4480,
            speed_cost=2040,
            cargo_cost=2650,
        ),
    )
    monkeypatch.setattr("ipm.tasks.planets.PlanetNavigator", FakeNavigator)
    actions = FakeActions()
    reader = FakePlanetReader([before.current_planet, before.current_planet, before.current_planet])
    state_reader = CountingStateReader([before, stale_after, verified_after, verified_after])
    task = PlanetsTask(
        reader=reader,
        state_reader=state_reader,
        actions=actions,
        config=_runtime_config_without_starfield_probe(
            policy=PolicyConfig(max_planet_upgrades_per_task=1, planet_upgrade_confirm_reads=3)
        ),
    )
    result = task.run()
    assert result.ok is True
    assert result.details["verified"] is True
    assert state_reader.read_planet_snapshot_calls == 3


def test_planets_task_uses_full_confirmation_budget_when_step_never_verifies(monkeypatch):
    before = GameSnapshot(
        cash=500,
        current_planet=PlanetPanelState(
            planet_id=1,
            title="1. BALOR",
            mining_level=8,
            speed_level=6,
            cargo_level=7,
            mining_cost=400,
            speed_cost=200,
            cargo_cost=250,
        ),
    )
    stale_after = GameSnapshot(
        cash=300,
        current_planet=PlanetPanelState(
            planet_id=1,
            title="1. BALOR",
            mining_level=8,
            speed_level=6,
            cargo_level=7,
            mining_cost=400,
            speed_cost=200,
            cargo_cost=250,
        ),
    )
    monkeypatch.setattr("ipm.tasks.planets.PlanetNavigator", FakeNavigator)
    actions = FakeActions()
    reader = FakePlanetReader([before.current_planet, before.current_planet, before.current_planet])
    state_reader = CountingStateReader([before, stale_after, stale_after, stale_after])
    task = PlanetsTask(
        reader=reader,
        state_reader=state_reader,
        actions=actions,
        config=_runtime_config_without_starfield_probe(
            policy=PolicyConfig(max_planet_upgrades_per_task=1, planet_upgrade_confirm_reads=3)
        ),
    )
    result = task.run()
    assert result.ok is False
    assert result.details["executed"] is True
    assert result.details["verified"] is False
    assert state_reader.read_planet_snapshot_calls == 4


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
    reader = FakePlanetReader([before.current_planet, before.current_planet, before.current_planet])
    task = PlanetsTask(
        reader=reader,
        state_reader=FakeStateReader([before, after]),
        actions=actions,
        config=_runtime_config_without_starfield_probe(policy=PolicyConfig(max_planet_upgrades_per_task=1)),
    )
    result = task.run()
    assert result.details["decision"]["stat"] == "C"
    assert result.details["executed"] is True
    assert result.details["verified"] is True


def test_planets_task_preserves_wrapped_planet_identity_during_scan():
    class CycleWorld:
        def __init__(self):
            self.order = [3, 4, 1]
            self.index = 0
            self.upgraded = False
            self.panels = {
                1: PlanetPanelState(
                    planet_id=1,
                    title="1. BALOR",
                    mining_level=43,
                    speed_level=40,
                    cargo_level=25,
                    mining_cost=305200,
                    speed_cost=138920,
                    cargo_cost=2710,
                ),
                3: PlanetPanelState(
                    planet_id=3,
                    title="3. ANADIUS",
                    mining_level=33,
                    speed_level=23,
                    cargo_level=18,
                    mining_cost=110690,
                    speed_cost=8030,
                    cargo_cost=2160,
                ),
                4: PlanetPanelState(
                    planet_id=4,
                    title="4. DHOLEN",
                    mining_level=31,
                    speed_level=26,
                    cargo_level=18,
                    mining_cost=163750,
                    speed_cost=44100,
                    cargo_cost=5410,
                ),
            }

        def current_panel(self):
            panel = self.panels[self.order[self.index]]
            if self.upgraded and panel.planet_id == 4:
                return replace(panel, cargo_level=19, cargo_cost=6000)
            return panel

        def step(self, delta):
            self.index = (self.index + delta) % len(self.order)

    class CycleReader:
        def __init__(self, world):
            self.world = world

        def read(self):
            return self.world.current_panel()

    class CycleStateReader:
        def __init__(self, world):
            self.world = world

        def read(self):
            return GameSnapshot(cash=7660, current_planet=self.world.current_panel())

    class CycleActions(FakeActions):
        def __init__(self, world):
            super().__init__()
            self.world = world

        def next_planet(self):
            self.calls.append(("next_planet",))
            self.world.step(1)
            return True

        def previous_planet(self):
            self.calls.append(("previous_planet",))
            self.world.step(-1)
            return True

        def increase_planet_stat(self, stat):
            self.calls.append(("increase_planet_stat", stat))
            if self.world.current_panel().planet_id == 4 and stat == "C":
                self.world.upgraded = True
                return True
            return False

    world = CycleWorld()
    actions = CycleActions(world)
    task = PlanetsTask(
        reader=CycleReader(world),
        state_reader=CycleStateReader(world),
        actions=actions,
        config=_runtime_config_without_starfield_probe(policy=PolicyConfig(max_planet_upgrades_per_task=1)),
    )
    result = task.run()
    assert result.ok is True
    assert result.details["decision"]["planet_id"] == 4
    assert result.details["decision"]["stat"] == "C"
    assert result.details["verified"] is True
    assert result.details["planet_order"] == [3, 4]
    assert result.details["steps"][0]["before_planet"]["title"] == "4. DHOLEN"
    assert result.details["steps"][0]["after_planet"]["planet_id"] == 4
    assert result.details["steps"][0]["after_planet"]["cargo_level"] == 19


def test_planets_task_revalidates_live_target_cost_before_upgrade(monkeypatch):
    stale_scan_panel = PlanetPanelState(
        planet_id=4,
        title="4. DHOLEN",
        mining_level=31,
        speed_level=26,
        cargo_level=18,
        mining_cost=164,
        speed_cost=44,
        cargo_cost=5,
    )
    live_panel = PlanetPanelState(
        planet_id=4,
        title="4. DHOLEN",
        mining_level=31,
        speed_level=26,
        cargo_level=18,
        mining_cost=163750,
        speed_cost=44100,
        cargo_cost=5410,
    )

    class Navigator:
        def __init__(self, reader, actions, *, max_planets=16):
            self.reader = reader
            self.actions = actions

        def scan_visible_planets(self, initial_panel=None):
            return type(
                "Scan",
                (),
                {"planets": {4: stale_scan_panel}, "order": [4], "complete_loop": True},
            )()

        def go_to_planet(self, target_id, order, known_planets=None, current_panel=None):
            self.actions.calls.append(("go_to_planet", target_id, tuple(order)))
            return target_id == 4

    monkeypatch.setattr("ipm.tasks.planets.PlanetNavigator", Navigator)
    actions = FakeActions()
    reader = FakePlanetReader([stale_scan_panel, live_panel])
    task = PlanetsTask(
        reader=reader,
        state_reader=FakeStateReader([GameSnapshot(cash=100, current_planet=live_panel)]),
        actions=actions,
        config=_runtime_config_without_starfield_probe(policy=PolicyConfig(max_planet_upgrades_per_task=1)),
    )
    result = task.run()
    assert result.ok is False
    assert result.details["decision"]["planet_id"] == 4
    assert result.details["executed"] is False
    assert result.details["verified"] is False
    assert ("increase_planet_stat", "C") not in actions.calls


def test_planets_task_blocks_tiny_live_cost_when_scan_shows_same_level_is_more_expensive(monkeypatch):
    scanned_panel = PlanetPanelState(
        planet_id=4,
        title="4. DHOLEN",
        mining_level=31,
        speed_level=26,
        cargo_level=18,
        mining_cost=163750,
        speed_cost=44100,
        cargo_cost=5410,
    )
    live_panel = PlanetPanelState(
        planet_id=4,
        title="4. DHOLEN",
        mining_level=31,
        speed_level=26,
        cargo_level=18,
        mining_cost=164,
        speed_cost=44,
        cargo_cost=5,
    )

    class Navigator:
        def __init__(self, reader, actions, *, max_planets=16):
            self.reader = reader
            self.actions = actions

        def scan_visible_planets(self, initial_panel=None):
            return type(
                "Scan",
                (),
                {"planets": {4: scanned_panel}, "order": [4], "complete_loop": True},
            )()

        def go_to_planet(self, target_id, order, known_planets=None, current_panel=None):
            self.actions.calls.append(("go_to_planet", target_id, tuple(order)))
            return target_id == 4

    monkeypatch.setattr("ipm.tasks.planets.PlanetNavigator", Navigator)
    actions = FakeActions()
    reader = FakePlanetReader([scanned_panel, live_panel])
    task = PlanetsTask(
        reader=reader,
        state_reader=FakeStateReader([GameSnapshot(cash=6000, current_planet=live_panel)]),
        actions=actions,
        config=_runtime_config_without_starfield_probe(policy=PolicyConfig(max_planet_upgrades_per_task=1)),
    )
    result = task.run()
    assert result.ok is False
    assert result.details["decision"]["planet_id"] == 4
    assert result.details["steps"][0]["navigated"] is True
    assert result.details["executed"] is False
    assert result.details["verified"] is False
    assert ("increase_planet_stat", "C") not in actions.calls


def test_planets_task_prefers_planet_only_snapshot_reader(monkeypatch):
    before = GameSnapshot(
        cash=500,
        current_planet=PlanetPanelState(
            planet_id=1,
            title="1. BALOR",
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
            title="1. BALOR",
            mining_level=8,
            speed_level=7,
            cargo_level=7,
            mining_cost=400,
            speed_cost=240,
            cargo_cost=250,
        ),
    )

    class SelectiveStateReader:
        def __init__(self, snapshots):
            self.snapshots = list(snapshots)
            self.index = 0
            self.read_calls = 0
            self.read_planet_snapshot_calls = 0

        def read(self):
            self.read_calls += 1
            raise AssertionError("PlanetsTask should use read_planet_snapshot when available")

        def read_planet_snapshot(self):
            self.read_planet_snapshot_calls += 1
            value = self.snapshots[min(self.index, len(self.snapshots) - 1)]
            self.index += 1
            return value

    monkeypatch.setattr("ipm.tasks.planets.PlanetNavigator", FakeNavigator)
    actions = FakeActions()
    reader = FakePlanetReader([before.current_planet, before.current_planet, before.current_planet])
    state_reader = SelectiveStateReader([before, after])
    task = PlanetsTask(
        reader=reader,
        state_reader=state_reader,
        actions=actions,
        config=_runtime_config_without_starfield_probe(policy=PolicyConfig(max_planet_upgrades_per_task=1)),
    )
    result = task.run()
    assert result.ok is True
    assert result.details["executed"] is True
    assert result.details["verified"] is True
    assert state_reader.read_calls == 0
    assert state_reader.read_planet_snapshot_calls == 2


def test_planets_task_uses_current_planet_snapshot_for_confirmation_and_refreshes_cash_once(monkeypatch):
    before = GameSnapshot(
        cash=500,
        current_planet=PlanetPanelState(
            planet_id=1,
            title="1. BALOR",
            mining_level=8,
            speed_level=6,
            cargo_level=7,
            mining_cost=400,
            speed_cost=200,
            cargo_cost=250,
        ),
    )
    after_panel_only = GameSnapshot(
        current_planet=PlanetPanelState(
            planet_id=1,
            title="1. BALOR",
            mining_level=8,
            speed_level=7,
            cargo_level=7,
            mining_cost=400,
            speed_cost=240,
            cargo_cost=250,
        ),
    )

    class SelectiveStateReader:
        def __init__(self):
            self.read_planet_snapshot_calls = 0
            self.read_current_planet_snapshot_calls = 0
            self.read_cash_snapshot_calls = 0

        def read_planet_snapshot(self):
            self.read_planet_snapshot_calls += 1
            raise AssertionError("confirmation should not need full planet snapshots when current-planet snapshots are available")

        def read_current_planet_snapshot(self):
            self.read_current_planet_snapshot_calls += 1
            return after_panel_only

        def read_cash_snapshot(self):
            self.read_cash_snapshot_calls += 1
            return GameSnapshot(cash=500 if self.read_cash_snapshot_calls == 1 else 300)

    monkeypatch.setattr("ipm.tasks.planets.PlanetNavigator", FakeNavigator)
    actions = FakeActions()
    reader = FakePlanetReader([before.current_planet, before.current_planet, before.current_planet])
    state_reader = SelectiveStateReader()
    task = PlanetsTask(
        reader=reader,
        state_reader=state_reader,
        actions=actions,
        config=_runtime_config_without_starfield_probe(policy=PolicyConfig(max_planet_upgrades_per_task=1)),
    )
    result = task.run()
    assert result.ok is True
    assert result.details["verified"] is True
    assert result.details["cash"] == 300
    assert state_reader.read_planet_snapshot_calls == 0
    assert state_reader.read_current_planet_snapshot_calls == 1
    assert state_reader.read_cash_snapshot_calls == 2


def test_planets_task_reuses_scan_final_panel_as_current_panel_when_cash_snapshot_has_no_panel(monkeypatch):
    before = GameSnapshot(
        cash=500,
        current_planet=PlanetPanelState(
            planet_id=1,
            title="1. BALOR",
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
            title="1. BALOR",
            mining_level=8,
            speed_level=7,
            cargo_level=7,
            mining_cost=400,
            speed_cost=240,
            cargo_cost=250,
        ),
    )

    class CashOnlyStateReader:
        def __init__(self):
            self.read_cash_snapshot_calls = 0
            self.read_planet_snapshot_calls = 0

        def read_cash_snapshot(self):
            self.read_cash_snapshot_calls += 1
            return GameSnapshot(cash=500)

        def read_planet_snapshot(self):
            self.read_planet_snapshot_calls += 1
            return after

    class Navigator(FakeNavigator):
        def scan_visible_planets(self, initial_panel=None):
            panel = self.reader.read()
            return type(
                "Scan",
                (),
                {
                    "planets": {panel.planet_id: panel},
                    "order": [panel.planet_id],
                    "complete_loop": True,
                    "final_panel": panel,
                },
            )()

    monkeypatch.setattr("ipm.tasks.planets.PlanetNavigator", Navigator)
    actions = FakeActions()
    reader = FakePlanetReader([before.current_planet, before.current_planet])
    state_reader = CashOnlyStateReader()
    task = PlanetsTask(
        reader=reader,
        state_reader=state_reader,
        actions=actions,
        config=_runtime_config_without_starfield_probe(policy=PolicyConfig(max_planet_upgrades_per_task=1)),
    )
    result = task.run()
    assert result.ok is True
    assert result.details["verified"] is True
    assert state_reader.read_cash_snapshot_calls == 1
    assert state_reader.read_planet_snapshot_calls == 1


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
