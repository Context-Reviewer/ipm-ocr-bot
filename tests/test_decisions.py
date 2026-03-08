from ipm.config import RuntimeConfig
from ipm.decisions import choose_ore_sale, choose_planet_upgrade
from ipm.state import GameSnapshot, OreRowState, PlanetPanelState


def test_choose_planet_upgrade_prefers_lowest_level_affordable_stat():
    snapshot = GameSnapshot(
        cash=500,
        current_planet=PlanetPanelState(
            planet_id=2,
            mining_level=10,
            speed_level=8,
            cargo_level=9,
            mining_cost=300,
            speed_cost=200,
            cargo_cost=250,
        ),
    )
    decision = choose_planet_upgrade(snapshot, RuntimeConfig())
    assert decision is not None
    assert decision.planet_id == 2
    assert decision.stat == "S"
    assert decision.cost == 200


def test_choose_planet_upgrade_prefers_scanned_planet_candidates():
    snapshot = GameSnapshot(
        cash=500,
        current_planet=PlanetPanelState(
            planet_id=1,
            mining_level=10,
            speed_level=10,
            cargo_level=10,
            mining_cost=600,
            speed_cost=600,
            cargo_cost=600,
        ),
        scanned_planets={
            1: PlanetPanelState(
                planet_id=1,
                mining_level=10,
                speed_level=10,
                cargo_level=10,
                mining_cost=600,
                speed_cost=600,
                cargo_cost=600,
            ),
            3: PlanetPanelState(
                planet_id=3,
                mining_level=8,
                speed_level=6,
                cargo_level=7,
                mining_cost=400,
                speed_cost=200,
                cargo_cost=250,
            ),
        },
    )
    decision = choose_planet_upgrade(snapshot, RuntimeConfig())
    assert decision is not None
    assert decision.planet_id == 3
    assert decision.stat == "S"


def test_choose_ore_sale_prefers_largest_sellable_stack():
    cfg = RuntimeConfig()
    snapshot = GameSnapshot(
        ore_rows={
            1: OreRowState(ore_name="Copper", quantity=30000),
            2: OreRowState(ore_name="Iron", quantity=70000),
        }
    )
    decision = choose_ore_sale(snapshot, cfg)
    assert decision is not None
    assert decision.row_index == 2
    assert decision.ore_name == "Iron"


def test_choose_ore_sale_rejects_unknown_ore_names():
    cfg = RuntimeConfig()
    snapshot = GameSnapshot(
        ore_rows={
            1: OreRowState(ore_name="Ror", quantity=99000),
            2: OreRowState(ore_name="Silica", quantity=56000),
        }
    )
    decision = choose_ore_sale(snapshot, cfg)
    assert decision is not None
    assert decision.row_index == 2
    assert decision.ore_name == "Silica"


def test_choose_ore_sale_accepts_aluminum_alias():
    cfg = RuntimeConfig()
    snapshot = GameSnapshot(
        ore_rows={
            1: OreRowState(ore_name="Aluminum", quantity=56000),
        }
    )
    decision = choose_ore_sale(snapshot, cfg)
    assert decision is not None
    assert decision.row_index == 1
    assert decision.ore_name == "Aluminum"
