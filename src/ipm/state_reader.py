from __future__ import annotations

from dataclasses import dataclass

from .state import GameSnapshot


@dataclass(slots=True)
class GameStateReader:
    hud_reader: object | None = None
    planet_reader: object | None = None
    ore_reader: object | None = None
    sell_reader: object | None = None

    def _read_snapshot(
        self,
        *,
        include_current_planet: bool = True,
        include_ore_rows: bool = True,
        include_sell_dialog: bool = True,
    ) -> GameSnapshot:
        snapshot = GameSnapshot()
        if self.hud_reader is not None:
            cash, _backend = self.hud_reader.read_cash()
            snapshot.cash = cash
        if include_current_planet and self.planet_reader is not None:
            snapshot.current_planet = self.planet_reader.read()
        if include_ore_rows and self.ore_reader is not None:
            snapshot.ore_rows = self.ore_reader.read_visible_rows()
        if include_sell_dialog and self.sell_reader is not None:
            snapshot.sell_dialog = self.sell_reader.read()
        return snapshot

    def read(self) -> GameSnapshot:
        return self._read_snapshot()

    def read_planet_snapshot(self) -> GameSnapshot:
        return self._read_snapshot(include_ore_rows=False, include_sell_dialog=False)

    def read_cash_snapshot(self) -> GameSnapshot:
        return self._read_snapshot(
            include_current_planet=False,
            include_ore_rows=False,
            include_sell_dialog=False,
        )
