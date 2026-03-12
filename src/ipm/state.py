from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(slots=True)
class PlanetPanelState:
    planet_id: Optional[int] = None
    title: str = ""
    mining_level: Optional[int] = None
    speed_level: Optional[int] = None
    cargo_level: Optional[int] = None
    mining_cost: Optional[int] = None
    speed_cost: Optional[int] = None
    cargo_cost: Optional[int] = None
    title_backend: str = ""


@dataclass(slots=True)
class OreRowState:
    ore_name: str = ""
    quantity: Optional[int] = None
    selected: bool = False
    backend: str = ""


@dataclass(slots=True)
class InventoryRowState:
    name: str = ""
    quantity: Optional[int] = None
    backend: str = ""


@dataclass(slots=True)
class SellDialogState:
    selected_quantity: Optional[int] = None
    slider_visible: bool = False
    backend: str = ""


@dataclass(slots=True)
class GameSnapshot:
    cash: Optional[int] = None
    current_planet: PlanetPanelState = field(default_factory=PlanetPanelState)
    scanned_planets: Dict[int, PlanetPanelState] = field(default_factory=dict)
    planet_order: list[int] = field(default_factory=list)
    ore_rows: Dict[int, OreRowState] = field(default_factory=dict)
    sell_dialog: SellDialogState = field(default_factory=SellDialogState)
