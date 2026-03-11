from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..decisions import choose_planet_upgrade
from ..domain_data import normalize_planet_name
from ..galaxy import PlanetNavigator
from ..starfield_probe import try_open_nearest_starfield_candidate
from .base import TaskResult


@dataclass(slots=True)
class _PlanetScanReader:
    reader: object
    cash: int | None = None

    def read(self):
        read_for_scan = getattr(self.reader, "read_for_scan", None)
        if callable(read_for_scan):
            return read_for_scan(cash=self.cash)
        return self.reader.read()


@dataclass(slots=True)
class PlanetsTask:
    reader: object | None = None
    state_reader: object | None = None
    actions: object | None = None
    capture: object | None = None
    config: object | None = None
    name: str = "planets"
    _run_observability: dict[str, object] | None = field(default=None, init=False, repr=False)

    @staticmethod
    def _new_run_observability(*, probe_enabled: bool) -> dict[str, object]:
        return {
            "classification": {
                "run_kind": "unknown",
                "decision_present": False,
                "steps_count": 0,
                "action_outcome": "none",
            },
            "probe": {
                "enabled": probe_enabled,
                "ok": None,
                "reason": None,
                "light_reads": 0,
                "full_fallback_reads": 0,
            },
            "scan": {
                "visible_planet_count": 0,
                "complete_loop": False,
                "panel_reads": 0,
                "direct_title_reads": 0,
                "fast_seed_reads": 0,
                "legacy_title_retries": 0,
                "suspicious_cost_retry_reads": 0,
                "suspicious_cost_retry_fields": 0,
                "full_panel_parses": 0,
                "full_parse_reasons": {},
                "escalation_reason_counts": {},
                "panel_class_counts": {},
                "slow_panel_threshold_seconds": 1.0,
                "slow_panel_count": 0,
                "slow_panel_class_counts": {},
                "slow_panels": [],
            },
            "confirmation": {
                "reads_used": 0,
                "verified_on_read": None,
                "current_planet_snapshot_reads": 0,
                "full_planet_snapshot_reads": 0,
                "cash_snapshot_reads": 0,
            },
        }

    @staticmethod
    def _increment_bucket_count(bucket: dict[str, object], key: str, amount: int = 1) -> None:
        bucket[key] = int(bucket.get(key, 0)) + int(amount)

    def _observe_reader_event(self, event: str, **payload) -> None:
        if self._run_observability is None:
            return
        if event == "planet_panel_probe_read":
            probe = self._run_observability["probe"]
            if payload.get("used_fast_path"):
                self._increment_bucket_count(probe, "light_reads")
            if payload.get("used_full_panel_fallback"):
                self._increment_bucket_count(probe, "full_fallback_reads")
            return
        if event != "planet_panel_scan_read":
            return
        scan = self._run_observability["scan"]
        self._increment_bucket_count(scan, "panel_reads")
        if payload.get("used_direct_title_read"):
            self._increment_bucket_count(scan, "direct_title_reads")
        if payload.get("seed_enough") and payload.get("title_trustworthy") and not payload.get("used_full_panel_parse"):
            self._increment_bucket_count(scan, "fast_seed_reads")
        if payload.get("used_legacy_title_retry"):
            self._increment_bucket_count(scan, "legacy_title_retries")
        retry_fields = tuple(payload.get("suspicious_cost_retry_fields", ()))
        if retry_fields:
            self._increment_bucket_count(scan, "suspicious_cost_retry_reads")
            self._increment_bucket_count(scan, "suspicious_cost_retry_fields", len(retry_fields))
        if payload.get("used_full_panel_parse"):
            self._increment_bucket_count(scan, "full_panel_parses")
            reasons = scan["full_parse_reasons"]
            reason = str(payload.get("full_panel_parse_reason") or "unknown")
            reasons[reason] = int(reasons.get(reason, 0)) + 1
        escalation_reasons = scan["escalation_reason_counts"]
        for reason in tuple(payload.get("escalation_reasons", ())):
            reason_text = str(reason or "unknown")
            escalation_reasons[reason_text] = int(escalation_reasons.get(reason_text, 0)) + 1
        panel_class = str(payload.get("panel_class") or "unknown")
        panel_classes = scan["panel_class_counts"]
        panel_classes[panel_class] = int(panel_classes.get(panel_class, 0)) + 1
        elapsed_seconds = float(payload.get("elapsed_seconds") or 0.0)
        slow_threshold = float(scan.get("slow_panel_threshold_seconds") or 1.0)
        if elapsed_seconds >= slow_threshold:
            scan["slow_panels"].append(
                {
                    "title": payload.get("title"),
                    "planet_id": payload.get("planet_id"),
                    "elapsed_seconds": elapsed_seconds,
                    "used_full_panel_parse": bool(payload.get("used_full_panel_parse")),
                    "full_panel_parse_reason": payload.get("full_panel_parse_reason") or "",
                    "suspicious_cost_retry_fields": list(retry_fields),
                    "escalation_reasons": list(payload.get("escalation_reasons", ())),
                    "title_trust": payload.get("title_trust"),
                    "panel_class": panel_class,
                }
            )

    def _finalize_observability(self, details: dict[str, object]) -> dict[str, object]:
        if self._run_observability is None:
            return {}
        observability = self._run_observability
        classification = observability["classification"]
        steps = details.get("steps") or []
        decision_present = bool(details.get("decision") is not None or steps)
        classification["decision_present"] = decision_present
        classification["steps_count"] = len(steps)
        if decision_present:
            classification["run_kind"] = "action_bearing"
            if details.get("verified"):
                classification["action_outcome"] = "verified"
            elif details.get("executed"):
                classification["action_outcome"] = "verification_failed"
            else:
                last_step = steps[-1] if steps else {}
                classification["action_outcome"] = str(last_step.get("block_reason") or "blocked_pre_action")
        else:
            error = str(details.get("error") or "")
            if error.startswith("starfield_probe_"):
                classification["run_kind"] = "probe_failed"
            elif error == "planet_panel_unreadable":
                classification["run_kind"] = "panel_unreadable"
            else:
                classification["run_kind"] = "no_op"
        observability["scan"]["slow_panels"] = sorted(
            observability["scan"]["slow_panels"],
            key=lambda item: float(item.get("elapsed_seconds") or 0.0),
            reverse=True,
        )
        slow_panel_classes: dict[str, int] = {}
        for item in observability["scan"]["slow_panels"]:
            panel_class = str(item.get("panel_class") or "unknown")
            slow_panel_classes[panel_class] = slow_panel_classes.get(panel_class, 0) + 1
        observability["scan"]["slow_panel_count"] = len(observability["scan"]["slow_panels"])
        observability["scan"]["slow_panel_class_counts"] = slow_panel_classes
        return observability

    def _task_result(self, *, ok: bool, details: dict[str, object]) -> TaskResult:
        details = dict(details)
        details["observability"] = self._finalize_observability(details)
        return TaskResult(ok=ok, details=details)

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

    @staticmethod
    def _panel_matches_expected(panel, expected_planet_id: int, expected_panel) -> bool:
        if panel is None:
            return False
        panel_id = getattr(panel, "planet_id", None)
        if panel_id is not None and int(panel_id) == int(expected_planet_id):
            return True
        expected_title = normalize_planet_name(getattr(expected_panel, "title", "")) if expected_panel is not None else ""
        current_title = normalize_planet_name(getattr(panel, "title", ""))
        return bool(expected_title and current_title and expected_title == current_title)

    @staticmethod
    def _panel_cost_for_stat(panel, stat: str) -> int | None:
        if panel is None:
            return None
        cost_attr = {
            "M": "mining_cost",
            "S": "speed_cost",
            "C": "cargo_cost",
        }.get(str(stat).upper())
        if not cost_attr:
            return None
        value = getattr(panel, cost_attr, None)
        return int(value) if value is not None else None

    @staticmethod
    def _panel_level_for_stat(panel, stat: str) -> int | None:
        if panel is None:
            return None
        level_attr = {
            "M": "mining_level",
            "S": "speed_level",
            "C": "cargo_level",
        }.get(str(stat).upper())
        if not level_attr:
            return None
        value = getattr(panel, level_attr, None)
        return int(value) if value is not None else None

    @classmethod
    def _live_cost_is_plausible_for_action(cls, scanned_panel, live_panel, stat: str) -> bool:
        scanned_cost = cls._panel_cost_for_stat(scanned_panel, stat)
        live_cost = cls._panel_cost_for_stat(live_panel, stat)
        scanned_level = cls._panel_level_for_stat(scanned_panel, stat)
        live_level = cls._panel_level_for_stat(live_panel, stat)
        if scanned_cost is None or live_cost is None:
            return True
        if scanned_level is None or live_level is None:
            return True
        if int(scanned_level) != int(live_level):
            return True
        return int(live_cost) >= int(scanned_cost)

    def _read_planet_snapshot(self):
        if self._run_observability is not None:
            self._increment_bucket_count(self._run_observability["confirmation"], "full_planet_snapshot_reads")
        read_planet_snapshot = getattr(self.state_reader, "read_planet_snapshot", None)
        if callable(read_planet_snapshot):
            return read_planet_snapshot()
        return self.state_reader.read()

    def _read_cash_snapshot(self):
        if self._run_observability is not None:
            self._increment_bucket_count(self._run_observability["confirmation"], "cash_snapshot_reads")
        read_cash_snapshot = getattr(self.state_reader, "read_cash_snapshot", None)
        if callable(read_cash_snapshot):
            return read_cash_snapshot()
        return None

    def _read_current_planet_snapshot(self):
        if self._run_observability is not None:
            self._increment_bucket_count(self._run_observability["confirmation"], "current_planet_snapshot_reads")
        read_current_planet_snapshot = getattr(self.state_reader, "read_current_planet_snapshot", None)
        if callable(read_current_planet_snapshot):
            return read_current_planet_snapshot()
        return self._read_planet_snapshot()

    def _collect_snapshots(self, count: int):
        snapshots = []
        for _ in range(max(1, count)):
            snapshots.append(self._read_planet_snapshot())
        return snapshots

    def _collect_confirmation_snapshots(self, count: int, before_panel, target_planet_id: int, stat: str):
        snapshots = []
        verified_candidate = None
        for _ in range(max(1, count)):
            snapshot = self._read_current_planet_snapshot()
            snapshots.append(snapshot)
            if self._verified(before_panel, snapshot, target_planet_id, stat):
                verified_candidate = snapshot
                if self._run_observability is not None:
                    self._run_observability["confirmation"]["verified_on_read"] = len(snapshots)
                break
        if self._run_observability is not None:
            self._run_observability["confirmation"]["reads_used"] = len(snapshots)
        if verified_candidate is not None and verified_candidate.cash is None:
            cash_snapshot = self._read_cash_snapshot()
            if cash_snapshot is None or cash_snapshot.cash is None:
                cash_snapshot = self._read_planet_snapshot()
            if cash_snapshot is not None:
                verified_candidate.cash = cash_snapshot.cash
        return snapshots, verified_candidate

    def run(self) -> TaskResult:
        if self.reader is None or self.state_reader is None or self.actions is None or self.config is None:
            self._run_observability = self._new_run_observability(probe_enabled=False)
            try:
                return self._task_result(
                    ok=True,
                    details={
                        "implemented": False,
                        "message": "Planet task dependencies not configured.",
                    },
                )
            finally:
                self._run_observability = None
        observer_supported = hasattr(self.reader, "observer")
        self.actions.reset_ui()
        initial_panel = None
        probe_enabled = bool(getattr(getattr(self.config, "starfield", None), "enable_click_probe", False))
        self._run_observability = self._new_run_observability(probe_enabled=probe_enabled)
        previous_observer = getattr(self.reader, "observer", None) if observer_supported else None
        if observer_supported:
            self.reader.observer = self._observe_reader_event
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
                ship_template_scales=tuple(getattr(self.config.starfield, "ship_template_scales", (0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25))),
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
                heuristic_fallback_min_bbox_width=int(
                    getattr(self.config.starfield, "heuristic_fallback_min_bbox_width", 20)
                ),
                heuristic_fallback_min_bbox_height=int(
                    getattr(self.config.starfield, "heuristic_fallback_min_bbox_height", 12)
                ),
                heuristic_fallback_min_area=int(
                    getattr(self.config.starfield, "heuristic_fallback_min_area", 180)
                ),
                max_ship_radius=int(getattr(self.config.starfield, "max_ship_radius", 72)),
                max_ship_bbox_width=int(getattr(self.config.starfield, "max_ship_bbox_width", 140)),
                max_ship_bbox_height=int(getattr(self.config.starfield, "max_ship_bbox_height", 90)),
                max_ship_area_ratio=float(getattr(self.config.starfield, "max_ship_area_ratio", 0.08)),
            )
            self._run_observability["probe"]["ok"] = bool(probe.ok)
            self._run_observability["probe"]["reason"] = probe.reason
            if not probe.ok:
                self.actions.reset_ui()
                try:
                    return self._task_result(
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
                finally:
                    if observer_supported:
                        self.reader.observer = previous_observer
                    self._run_observability = None
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
            try:
                return self._task_result(
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
            finally:
                if observer_supported:
                    self.reader.observer = previous_observer
                self._run_observability = None
        cash_snapshot = self._read_cash_snapshot()
        navigator = PlanetNavigator(
            _PlanetScanReader(self.reader, cash=getattr(cash_snapshot, "cash", None)),
            self.actions,
        )
        scan = navigator.scan_visible_planets(initial_panel=initial_panel)
        self._run_observability["scan"]["visible_planet_count"] = len(scan.order)
        self._run_observability["scan"]["complete_loop"] = bool(getattr(scan, "complete_loop", False))
        snapshot = cash_snapshot if cash_snapshot is not None else self._read_planet_snapshot()
        if getattr(cash_snapshot, "cash", None) is not None:
            snapshot.cash = cash_snapshot.cash
        scan_final_panel = getattr(scan, "final_panel", None)
        if not self._panel_readable(getattr(snapshot, "current_planet", None)) and scan_final_panel is not None:
            snapshot.current_planet = scan_final_panel
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
            scanned_target = snapshot.scanned_planets.get(decision.planet_id)
            target_before = scanned_target or snapshot.current_planet
            navigated = True
            if scan.order:
                navigated = navigator.go_to_planet(
                    decision.planet_id,
                    scan.order,
                    scan.planets,
                    current_panel=snapshot.current_planet,
                )
            live_target_panel = None
            live_cost_plausible = True
            if navigated:
                live_target_panel = self.reader.read()
                if not self._panel_matches_expected(live_target_panel, decision.planet_id, target_before):
                    navigated = False
                else:
                    live_cost_plausible = self._live_cost_is_plausible_for_action(scanned_target, live_target_panel, decision.stat)
                    target_before = live_target_panel
                    snapshot.current_planet = live_target_panel
                    snapshot.scanned_planets[decision.planet_id] = live_target_panel
            live_cost = self._panel_cost_for_stat(target_before, decision.stat)
            affordable = live_cost is not None and snapshot.cash is not None and live_cost <= int(snapshot.cash)
            executed = bool(navigated and live_cost_plausible and affordable and self.actions.increase_planet_stat(decision.stat))
            step_after = None
            verified = False
            if executed:
                confirm_reads = max(1, int(getattr(self.config.policy, "planet_upgrade_confirm_reads", 3)))
                after_reads, verified_candidate = self._collect_confirmation_snapshots(
                    confirm_reads,
                    target_before,
                    decision.planet_id,
                    decision.stat,
                )
                step_after = verified_candidate or after_reads[-1]
                step_after.scanned_planets = dict(snapshot.scanned_planets)
                step_after.planet_order = list(snapshot.planet_order)
                verified = verified_candidate is not None
            steps.append(
                {
                    "decision": asdict(decision),
                    "navigated": navigated,
                    "affordable": affordable,
                    "live_cost_guard_blocked": bool(navigated and not live_cost_plausible),
                    "block_reason": (
                        None
                        if verified
                        else "navigation_failed"
                        if not navigated
                        else "live_cost_guard_blocked"
                        if not live_cost_plausible
                        else "unaffordable"
                        if not affordable
                        else "verification_failed"
                        if executed
                        else "action_rejected"
                    ),
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
        try:
            return self._task_result(
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
        finally:
            if observer_supported:
                self.reader.observer = previous_observer
            self._run_observability = None
