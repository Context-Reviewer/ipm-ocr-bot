from __future__ import annotations

from dataclasses import dataclass

from .config import RuntimeConfig
from .state import GameSnapshot, OreRowState, PlanetPanelState


@dataclass(slots=True, frozen=True)
class PlanetUpgradeDecision:
    planet_id: int
    stat: str
    cost: int
    reason: str


@dataclass(slots=True, frozen=True)
class OreSellDecision:
    row_index: int
    ore_name: str
    quantity: int
    fraction: float
    reason: str


def choose_planet_upgrade(snapshot: GameSnapshot, cfg: RuntimeConfig) -> PlanetUpgradeDecision | None:
    panels = snapshot.scanned_planets or (
        {snapshot.current_planet.planet_id: snapshot.current_planet}
        if snapshot.current_planet.planet_id is not None
        else {}
    )
    cash = snapshot.cash
    if cash is None or cash <= 0:
        return None
    candidates: list[tuple[int, str, int, int, str]] = []
    for planet_id, panel in panels.items():
        if panel.mining_level is not None and panel.mining_cost is not None and panel.mining_cost <= cash:
            target = min(
                level for level in (panel.speed_level, panel.cargo_level) if level is not None
            ) if any(level is not None for level in (panel.speed_level, panel.cargo_level)) else panel.mining_level
            deficit = max(0, target - panel.mining_level)
            candidates.append((planet_id, "M", panel.mining_cost, deficit, "balance levels"))
        if panel.speed_level is not None and panel.speed_cost is not None and panel.speed_cost <= cash:
            target = min(
                level for level in (panel.mining_level, panel.cargo_level) if level is not None
            ) if any(level is not None for level in (panel.mining_level, panel.cargo_level)) else panel.speed_level
            deficit = max(0, target - panel.speed_level)
            candidates.append((planet_id, "S", panel.speed_cost, deficit, "balance levels"))
        if panel.cargo_level is not None and panel.cargo_cost is not None and panel.cargo_cost <= cash:
            target = min(
                level for level in (panel.mining_level, panel.speed_level) if level is not None
            ) if any(level is not None for level in (panel.mining_level, panel.speed_level)) else panel.cargo_level
            deficit = max(0, target - panel.cargo_level)
            candidates.append((planet_id, "C", panel.cargo_cost, deficit, "balance levels"))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[3], item[2], item[1], item[0]))
    planet_id, stat, cost, _deficit, reason = candidates[0]
    return PlanetUpgradeDecision(planet_id=planet_id, stat=stat, cost=cost, reason=reason)


def choose_ore_sale(snapshot: GameSnapshot, cfg: RuntimeConfig) -> OreSellDecision | None:
    sell_start = int(cfg.policy.ore_sell_start_quantity)
    keep_qty = int(cfg.policy.ore_keep_quantity)
    allowed_names = {name.lower() for name in cfg.policy.known_ore_names}
    best: OreSellDecision | None = None
    for row_index, row in snapshot.ore_rows.items():
        state: OreRowState = row
        if not state.ore_name or state.quantity is None:
            continue
        if state.ore_name.lower() not in allowed_names:
            continue
        if state.quantity < sell_start:
            continue
        sellable = max(0, state.quantity - keep_qty)
        if sellable <= 0:
            continue
        fraction = max(0.25, min(1.0, sellable / max(1, state.quantity)))
        decision = OreSellDecision(
            row_index=row_index,
            ore_name=state.ore_name,
            quantity=state.quantity,
            fraction=fraction,
            reason="above reserve",
        )
        if best is None or decision.quantity > best.quantity:
            best = decision
    return best
