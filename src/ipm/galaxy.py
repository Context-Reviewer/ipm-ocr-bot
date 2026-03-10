from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from .domain_data import PLANET_NAMES, normalize_planet_name
from .state import PlanetPanelState


def _title_name(title: str) -> str:
    cleaned = re.sub(r"^\s*\d+\s*[\.\-:]?\s*", "", str(title or "")).strip().upper()
    cleaned = re.sub(r"[^A-Z]+", "", cleaned)
    return cleaned


def _title_suffix(title: str) -> str:
    suffix = re.sub(r"^\s*\d+\s*[\.\-:]?\s*", "", str(title or "")).strip()
    return suffix


def _normalize_panel_id(panel: PlanetPanelState, planet_id: int) -> PlanetPanelState:
    suffix = _title_suffix(panel.title)
    title = f"{planet_id}. {suffix}".strip() if suffix else str(planet_id)
    return replace(panel, planet_id=planet_id, title=title)


_PLANET_ID_BY_NAME = {name: index + 1 for index, name in enumerate(PLANET_NAMES)}


def _known_title_planet_id(title: str) -> int | None:
    canonical = normalize_planet_name(title)
    if not canonical:
        return None
    return _PLANET_ID_BY_NAME.get(canonical)


@dataclass(slots=True)
class GalaxyScan:
    planets: dict[int, PlanetPanelState] = field(default_factory=dict)
    order: list[int] = field(default_factory=list)
    complete_loop: bool = False


class PlanetNavigator:
    def __init__(self, reader: object, actions: object, *, max_planets: int = 16) -> None:
        self.reader = reader
        self.actions = actions
        self.max_planets = max(1, int(max_planets))

    def current(self) -> PlanetPanelState:
        return self.reader.read()

    def _resolve_current_id(
        self,
        panel: PlanetPanelState,
        order: list[int],
        known_planets: dict[int, PlanetPanelState] | None = None,
    ) -> int | None:
        current_name = _title_name(panel.title)
        if panel.planet_id in order:
            if not known_planets:
                return panel.planet_id
            known = known_planets.get(panel.planet_id)
            if known is None:
                return panel.planet_id
            known_name = _title_name(known.title)
            if not current_name or not known_name or current_name == known_name:
                return panel.planet_id
        if known_planets and current_name:
            for planet_id, known in known_planets.items():
                if planet_id in order and _title_name(known.title) == current_name:
                    return planet_id
        return None

    def step(self, direction: int) -> PlanetPanelState | None:
        ok = self.actions.next_planet() if direction >= 0 else self.actions.previous_planet()
        if not ok:
            return None
        return self.reader.read()

    def scan_visible_planets(self) -> GalaxyScan:
        scan = GalaxyScan()
        first = self.current()
        if first.planet_id is None:
            return scan
        scan.planets[first.planet_id] = first
        scan.order.append(first.planet_id)
        for _ in range(self.max_planets - 1):
            nxt = self.step(1)
            if nxt is None:
                break
            known_title_id = _known_title_planet_id(nxt.title)
            if known_title_id is not None and scan.order:
                # Crossing from the tail of the local cycle back to an earlier known
                # planet is a loop boundary, not a new synthetic sequential id.
                if known_title_id == scan.order[0] or known_title_id in scan.planets or known_title_id < scan.order[-1]:
                    scan.complete_loop = True
                    break
            expected_id = scan.order[-1] + 1
            if nxt.planet_id is None:
                next_name = _title_name(nxt.title)
                if next_name:
                    matched_id = next(
                        (
                            known_id
                            for known_id, known in scan.planets.items()
                            if _title_name(known.title) == next_name
                        ),
                        None,
                    )
                    if matched_id == scan.order[0]:
                        scan.complete_loop = True
                        break
                    if matched_id is not None:
                        break
                inferred_id = expected_id
                if inferred_id in scan.planets:
                    break
                nxt = _normalize_panel_id(nxt, inferred_id)
            elif nxt.planet_id in scan.planets:
                next_name = _title_name(nxt.title)
                known_name = _title_name(scan.planets[nxt.planet_id].title)
                if next_name and known_name and next_name != known_name:
                    inferred_id = expected_id
                    if inferred_id in scan.planets:
                        break
                    nxt = _normalize_panel_id(nxt, inferred_id)
            elif nxt.planet_id <= 0 or nxt.planet_id < expected_id:
                inferred_id = expected_id
                if inferred_id in scan.planets:
                    break
                nxt = _normalize_panel_id(nxt, inferred_id)
            elif nxt.planet_id > expected_id + 1:
                inferred_id = expected_id
                if inferred_id in scan.planets:
                    break
                nxt = _normalize_panel_id(nxt, inferred_id)
            if nxt.planet_id == scan.order[0]:
                scan.complete_loop = True
                break
            if nxt.planet_id in scan.planets:
                break
            scan.planets[nxt.planet_id] = nxt
            scan.order.append(nxt.planet_id)
        return scan

    def go_to_planet(
        self,
        target_id: int,
        order: list[int],
        known_planets: dict[int, PlanetPanelState] | None = None,
    ) -> bool:
        if target_id not in order:
            return False
        current = self.current()
        current_id = self._resolve_current_id(current, order, known_planets)
        if current_id is None:
            for _ in range(len(order)):
                recovered = self.step(1)
                if recovered is None:
                    return False
                current_id = self._resolve_current_id(recovered, order, known_planets)
                if current_id is not None:
                    break
        if current_id is None:
            return False
        current_idx = order.index(current_id)
        target_idx = order.index(target_id)
        forward = (target_idx - current_idx) % len(order)
        backward = (current_idx - target_idx) % len(order)
        if forward <= backward:
            direction = 1
            steps = forward
        else:
            direction = -1
            steps = backward
        for _ in range(steps):
            moved = self.step(direction)
            if moved is None:
                return False
        final = self.current()
        final_id = self._resolve_current_id(final, order, known_planets)
        return final_id == target_id
