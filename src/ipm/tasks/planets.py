from __future__ import annotations

from dataclasses import asdict, dataclass

from ..decisions import choose_planet_upgrade
from ..domain_data import normalize_planet_name
from ..galaxy import PlanetNavigator
from ..starfield_probe import try_open_nearest_starfield_candidate
from .base import TaskResult


@dataclass(slots=True)
class PlanetsTask:
    reader: object | None = None
    state_reader: object | None = None
    actions: object | None = None
    capture: object | None = None
    config: object | None = None
    name: str = "planets"

    @staticmethod
    def _verified(before_panel, after_snapshot, target_planet_id: int, stat: str) -> bool:
        if before_panel is None or after_snapshot is None:
            return False
        after = after_snapshot.current_planet
        if after.planet_id is not None and int(after.planet_id) != int(target_planet_id):
            return False
        attr = {
            "M": "mining_level",
            "S": "speed_level",
            "C": "cargo_level",
        }.get(str(stat).upper())
        cost_attr = {
            "M": "mining_cost",
            "S": "speed_cost",
            "C": "cargo_cost",
        }.get(str(stat).upper())
        if not attr:
            return False
        before_value = getattr(before_panel, attr, None)
        after_value = getattr(after, attr, None)
        if before_value is not None and after_value is not None:
            return int(after_value) >= int(before_value) + 1
        if cost_attr:
            before_cost = getattr(before_panel, cost_attr, None)
            after_cost = getattr(after, cost_attr, None)
            if before_cost is not None and after_cost is not None:
                return int(after_cost) > int(before_cost)
        return False

    @staticmethod
    def _panel_readable(panel) -> bool:
        if panel is None:
            return False
        if getattr(panel, "planet_id", None) is not None:
            return True
        if str(getattr(panel, "title", "")).strip():
            return True
        for attr in (
            "mining_level",
            "speed_level",
            "cargo_level",
            "mining_cost",
            "speed_cost",
            "cargo_cost",
        ):
            if getattr(panel, attr, None) is not None:
                return True
        return False

    @staticmethod
    def _probe_panel_confirmed(panel) -> bool:
        if panel is None:
            return False
        if not normalize_planet_name(getattr(panel, "title", "")):
            return False
        cost_count = sum(
            1
            for attr in ("mining_cost", "speed_cost", "cargo_cost")
            if getattr(panel, attr, None) is not None
        )
        level_count = sum(
            1
            for attr in ("mining_level", "speed_level", "cargo_level")
            if getattr(panel, attr, None) is not None
        )
        return cost_count >= 2 or (cost_count >= 1 and level_count >= 1)

    @staticmethod
    def _probe_precondition_failure_reason(panel) -> str | None:
        if panel is None:
            return None
        if not normalize_planet_name(getattr(panel, "title", "")):
            return None
        if getattr(panel, "planet_id", None) is not None:
            return "not_starfield_ready"
        for attr in (
            "mining_level",
            "speed_level",
            "cargo_level",
            "mining_cost",
            "speed_cost",
            "cargo_cost",
        ):
            if getattr(panel, attr, None) is not None:
                return "not_starfield_ready"
        return None

    def run(self) -> TaskResult:
        if self.reader is None or self.state_reader is None or self.actions is None or self.config is None:
            return TaskResult(
                ok=True,
                details={
                    "implemented": False,
                    "message": "Planet task dependencies not configured.",
                },
            )
        self.actions.reset_ui()
        initial_panel = None
        probe_enabled = bool(getattr(getattr(self.config, "starfield", None), "enable_click_probe", False))
        if probe_enabled:
            probe = try_open_nearest_starfield_candidate(
                capture=self.capture,
                actions=self.actions,
                reader=self.reader,
                panel_is_readable=self._panel_readable,
                panel_is_confirmed=self._probe_panel_confirmed,
                settle_seconds=float(getattr(self.config.starfield, "click_probe_settle_seconds", 0.35)),
                save_annotation=bool(getattr(self.config.starfield, "save_probe_annotation", False)),
                annotation_dir=str(getattr(self.config.starfield, "probe_annotation_dir", "out/starfield")),
                scene_viewport=getattr(self.config.starfield, "scene_viewport", None),
                scene_exclusion_zones=getattr(self.config.starfield, "scene_exclusion_zones", None),
                ship_template_enabled=bool(getattr(self.config.starfield, "ship_template_enabled", True)),
                ship_template_path=str(getattr(self.config.starfield, "ship_template_path", "src/assets/ship_template.png")),
                ship_template_scales=tuple(getattr(self.config.starfield, "ship_template_scales", (1.0, 0.75, 0.5, 0.35, 0.25, 0.18, 0.12, 0.08))),
                ship_template_threshold=float(getattr(self.config.starfield, "ship_template_threshold", 0.55)),
                ship_template_use_edges=bool(getattr(self.config.starfield, "ship_template_use_edges", True)),
                ship_template_allow_fallback=bool(getattr(self.config.starfield, "ship_template_allow_fallback", True)),
                ship_template_search_left_margin=int(getattr(self.config.starfield, "ship_template_search_left_margin", 0)),
                ship_template_search_top_margin=int(getattr(self.config.starfield, "ship_template_search_top_margin", 0)),
                ship_template_search_right_margin=int(getattr(self.config.starfield, "ship_template_search_right_margin", 0)),
                ship_template_search_bottom_margin=int(getattr(self.config.starfield, "ship_template_search_bottom_margin", 0)),
                ship_template_min_scale=float(getattr(self.config.starfield, "ship_template_min_scale", 0.0)),
                ship_template_min_width=int(getattr(self.config.starfield, "ship_template_min_width", 0)),
                ship_template_min_height=int(getattr(self.config.starfield, "ship_template_min_height", 0)),
                ship_template_min_area=int(getattr(self.config.starfield, "ship_template_min_area", 0)),
                ship_exclusion_margin=int(getattr(self.config.starfield, "ship_exclusion_margin", 14)),
                ship_candidate_exclusion_radius=int(
                    getattr(self.config.starfield, "ship_candidate_exclusion_radius", 0)
                ),
                ship_cluster_exclusion_x_margin=int(
                    getattr(self.config.starfield, "ship_cluster_exclusion_x_margin", 60)
                ),
                ship_cluster_exclusion_y_margin=int(
                    getattr(self.config.starfield, "ship_cluster_exclusion_y_margin", 110)
                ),
                candidate_min_radius=int(getattr(self.config.starfield, "candidate_min_radius", 6)),
                candidate_min_area=int(getattr(self.config.starfield, "candidate_min_area", 80)),
                small_candidate_fallback_max_radius=int(
                    getattr(self.config.starfield, "small_candidate_fallback_max_radius", 20)
                ),
                small_candidate_fallback_offset_x=int(
                    getattr(self.config.starfield, "small_candidate_fallback_offset_x", 10)
                ),
                small_candidate_fallback_offset_y=int(
                    getattr(self.config.starfield, "small_candidate_fallback_offset_y", 0)
                ),
                min_ship_bbox_width=int(getattr(self.config.starfield, "min_ship_bbox_width", 20)),
                min_ship_bbox_height=int(getattr(self.config.starfield, "min_ship_bbox_height", 8)),
                min_ship_area=int(getattr(self.config.starfield, "min_ship_area", 150)),
                max_ship_radius=int(getattr(self.config.starfield, "max_ship_radius", 72)),
                max_ship_bbox_width=int(getattr(self.config.starfield, "max_ship_bbox_width", 140)),
                max_ship_bbox_height=int(getattr(self.config.starfield, "max_ship_bbox_height", 90)),
                max_ship_area_ratio=float(getattr(self.config.starfield, "max_ship_area_ratio", 0.08)),
            )
            if not probe.ok:
                self.actions.reset_ui()
                return TaskResult(
                    ok=False,
                    details={
                        "implemented": True,
                        "planet_order": [],
                        "scanned_planets": {},
                        "planet_panel": asdict(probe.panel) if probe.panel is not None else None,
                        "cash": None,
                        "decision": None,
                        "executed": False,
                        "verified": False,
                        "steps": [],
                        "after_planet_panel": None,
                        "error": f"starfield_probe_{probe.reason}",
                    },
                )
            initial_panel = probe.panel
        else:
            open_attempts = max(1, int(getattr(self.config.policy, "planet_panel_open_attempts", 1)))
            for _ in range(open_attempts):
                self.actions.open_planet_menu()
                initial_panel = self.reader.read()
                if self._panel_readable(initial_panel):
                    break
        if not self._panel_readable(initial_panel):
            self.actions.reset_ui()
            return TaskResult(
                ok=False,
                details={
                    "implemented": True,
                    "planet_order": [],
                    "scanned_planets": {},
                    "planet_panel": asdict(initial_panel) if initial_panel is not None else None,
                    "cash": None,
                    "decision": None,
                    "executed": False,
                    "verified": False,
                    "steps": [],
                    "after_planet_panel": None,
                    "error": "planet_panel_unreadable",
                },
            )
        navigator = PlanetNavigator(self.reader, self.actions)
        scan = navigator.scan_visible_planets()
        snapshot = self.state_reader.read()
        snapshot.scanned_planets = dict(scan.planets)
        snapshot.planet_order = list(scan.order)
        max_steps = max(1, int(getattr(self.config.policy, "max_planet_upgrades_per_task", 1)))
        steps: list[dict[str, object]] = []
        decision = None
        executed = False
        verified = False
        after_snapshot = None
        failure = False
        for _ in range(max_steps):
            decision = choose_planet_upgrade(snapshot, self.config)
            if decision is None:
                break
            target_before = snapshot.scanned_planets.get(decision.planet_id, snapshot.current_planet)
            navigated = True
            if scan.order:
                navigated = navigator.go_to_planet(decision.planet_id, scan.order, scan.planets)
            executed = bool(navigated and self.actions.increase_planet_stat(decision.stat))
            step_after = None
            verified = False
            if executed:
                step_after = self.state_reader.read()
                step_after.scanned_planets = dict(snapshot.scanned_planets)
                step_after.planet_order = list(snapshot.planet_order)
                verified = self._verified(target_before, step_after, decision.planet_id, decision.stat)
            steps.append(
                {
                    "decision": asdict(decision),
                    "navigated": navigated,
                    "executed": executed,
                    "verified": verified,
                    "before_planet": asdict(target_before) if target_before is not None else None,
                    "after_planet": asdict(step_after.current_planet) if step_after is not None else None,
                }
            )
            if not navigated or not executed or not verified:
                after_snapshot = step_after
                failure = True
                break
            after_snapshot = step_after
            snapshot.cash = after_snapshot.cash
            snapshot.current_planet = after_snapshot.current_planet
            snapshot.scanned_planets[decision.planet_id] = after_snapshot.current_planet
        self.actions.reset_ui()
        return TaskResult(
            ok=not failure,
            details={
                "implemented": True,
                "planet_order": list(snapshot.planet_order),
                "scanned_planets": {pid: asdict(panel) for pid, panel in snapshot.scanned_planets.items()},
                "planet_panel": asdict(snapshot.current_planet),
                "cash": snapshot.cash,
                "decision": asdict(decision) if decision is not None else None,
                "executed": executed,
                "verified": verified,
                "steps": steps,
                "after_planet_panel": asdict(after_snapshot.current_planet) if after_snapshot is not None else None,
            },
        )
