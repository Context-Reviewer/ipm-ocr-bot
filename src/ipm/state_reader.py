from __future__ import annotations

from dataclasses import dataclass

from .state import GameSnapshot


@dataclass(slots=True)
class GameStateReader:
    hud_reader: object | None = None
    planet_reader: object | None = None
    ore_reader: object | None = None
    sell_reader: object | None = None

    def read(self) -> GameSnapshot:
        snapshot = GameSnapshot()
        if self.hud_reader is not None:
            cash, _backend = self.hud_reader.read_cash()
            snapshot.cash = cash
        if self.planet_reader is not None:
            snapshot.current_planet = self.planet_reader.read()
        if self.ore_reader is not None:
            snapshot.ore_rows = self.ore_reader.read_visible_rows()
        if self.sell_reader is not None:
            snapshot.sell_dialog = self.sell_reader.read()
        return snapshot
