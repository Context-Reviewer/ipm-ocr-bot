from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import keyboard
from PIL import Image

from .actions import ActionDriver
from .capture import create_capture_backend
from .config import RuntimeConfig, load_runtime_config
from .focus import ensure_focus, get_active_window_title
from .perception import create_perception_backend
from .readers import HudReader, OrePanelReader, PlanetPanelReader, SellDialogReader
from .rects import RectStore
from .runtime import RuntimeState
from .scheduler import ScheduledTask, Scheduler
from .starfield_probe import (
    discover_starfield_planet_by_rank,
    try_open_nearest_starfield_candidate,
)
from .state_reader import GameStateReader
from .tasks import OresTask, PlanetsTask


def prepare_run_artifact_dir(*, base_dir: str = "out/runs", now: datetime | None = None) -> Path:
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    path = Path(base_dir) / timestamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_run_frame(image: Image.Image, *, output_dir: Path, filename: str = "frame.png") -> str:
    target = output_dir / filename
    image.save(target)
    return str(target)


@dataclass(slots=True)
class Application:
    config: RuntimeConfig
    scheduler: Scheduler = field(init=False)
    runtime: RuntimeState = field(init=False)
    capture_backend: object = field(init=False)
    perception_backend: object = field(init=False)
    rects: RectStore = field(init=False)
    actions: ActionDriver = field(init=False)
    state_reader: GameStateReader = field(init=False)
    tasks: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        tasks = [
            ScheduledTask(name="planets", interval_seconds=self.config.scheduler.tasks["planets"]),
            ScheduledTask(name="ores", interval_seconds=self.config.scheduler.tasks["ores"]),
        ]
        self.scheduler = Scheduler(tasks)
        self.runtime = RuntimeState(next_run_at=self.scheduler.seed(time.monotonic()))
        self.capture_backend = create_capture_backend(
            self.config.capture.backend,
            serial=self.config.capture.adb_serial,
            adb_path=self.config.capture.adb_path,
            target_resolution=self.config.capture.target_resolution,
            window_title=self.config.capture.window_title,
            cache_ttl_seconds=self.config.capture.cache_ttl_seconds,
        )
        self.perception_backend = create_perception_backend(
            self.config.perception.backend,
            model=self.config.perception.openai_model,
            hybrid_order=self.config.perception.hybrid_order,
            openai_enabled=self.config.perception.openai_enabled,
            openai_max_output_tokens=self.config.perception.openai_max_output_tokens,
            known_ore_names=self.config.policy.known_ore_names,
            planet_level_min=self.config.perception.semantic_level_min,
            planet_level_max=self.config.perception.semantic_level_max,
            upgrade_cost_min=self.config.perception.semantic_upgrade_cost_min,
            upgrade_cost_max=self.config.perception.semantic_upgrade_cost_max,
        )
        self.rects = RectStore.load(self.config.capture.rects_path)
        self.actions = ActionDriver(self.config, rects=self.rects, capture_backend=self.capture_backend)
        hud_reader = HudReader(self.config, self.rects, self.capture_backend, self.perception_backend)
        planet_reader = PlanetPanelReader(self.config, self.rects, self.capture_backend, self.perception_backend)
        ore_reader = OrePanelReader(self.config, self.rects, self.capture_backend, self.perception_backend)
        sell_reader = SellDialogReader(self.config, self.rects, self.capture_backend, self.perception_backend)
        self.state_reader = GameStateReader(
            hud_reader=hud_reader,
            planet_reader=planet_reader,
            ore_reader=ore_reader,
            sell_reader=sell_reader,
        )
        self.tasks = {
            "planets": PlanetsTask(
                reader=planet_reader,
                state_reader=self.state_reader,
                actions=self.actions,
                capture=self.capture_backend,
                config=self.config,
            ),
            "ores": OresTask(
                ore_reader=ore_reader,
                sell_reader=sell_reader,
                state_reader=self.state_reader,
                actions=self.actions,
                config=self.config,
            ),
        }

    def install_hotkeys(self) -> None:
        keyboard.add_hotkey(self.config.toggle_hotkey, self.toggle)
        keyboard.add_hotkey(self.config.emergency_stop_hotkey, self.stop_now)

    def toggle(self) -> None:
        self.runtime.running = not self.runtime.running
        if self.runtime.running:
            now = time.monotonic()
            self.runtime.next_run_at = self.scheduler.seed(now)
            self.runtime.last_heartbeat_at = 0.0
            print("[AFK] ON - scheduling tasks")
        print(f"AFK: {'ON' if self.runtime.running else 'OFF'}")

    def stop_now(self) -> None:
        self.runtime.stop_requested = True
        os._exit(0)

    def heartbeat(self, now: float) -> None:
        if now - self.runtime.last_heartbeat_at < self.config.scheduler.heartbeat_seconds:
            return
        print(
            "[AFK] heartbeat | "
            f"active='{get_active_window_title()}' "
            f"capture={self.capture_backend.name} perception={self.perception_backend.name}"
        )
        self.runtime.last_heartbeat_at = now

    def tick(self) -> None:
        now = time.monotonic()
        if not ensure_focus(self.config.focus):
            print(f"[AFK] waiting for focus: {get_active_window_title()}")
            return
        self.heartbeat(now)
        for scheduled in self.scheduler.due(now, self.runtime.next_run_at):
            print(f"[TASK] {scheduled.name} start")
            result = self.tasks[scheduled.name].run()
            if result.details:
                print(f"[TASK] {scheduled.name} details={result.details}")
            print(f"[TASK] {scheduled.name} done")
            self.scheduler.mark_complete(scheduled, time.monotonic(), self.runtime.next_run_at)

    def run_forever(self) -> None:
        self.install_hotkeys()
        print("Ready. F9 toggles AFK. F10 exits. Ctrl+C quits.")
        try:
            while True:
                if self.runtime.running:
                    self.tick()
                time.sleep(self.config.scheduler.module_idle_seconds)
        except KeyboardInterrupt:
            pass

    def run_starfield_probe_once(self) -> int:
        if not bool(getattr(self.config.starfield, "enable_click_probe", False)):
            print("[STARFIELD_PROBE] result=probe_disabled")
            return 1
        if not ensure_focus(self.config.focus):
            print("[STARFIELD_PROBE] result=focus_unavailable")
            return 1
        planet_task = self.tasks.get("planets")
        if planet_task is None or getattr(planet_task, "reader", None) is None:
            print("[STARFIELD_PROBE] result=probe_unavailable")
            return 1

        def _starfield_ready_check():
            panel = planet_task.reader.read()
            reason = planet_task._probe_precondition_failure_reason(panel)
            if reason is None:
                return None
            return (reason, panel)

        probe = try_open_nearest_starfield_candidate(
            capture=self.capture_backend,
            actions=self.actions,
            reader=planet_task.reader,
            panel_is_readable=planet_task._panel_readable,
            starfield_ready_check=_starfield_ready_check,
            panel_is_confirmed=planet_task._probe_panel_confirmed,
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
            ship_cluster_exclusion_x_margin=int(
                getattr(self.config.starfield, "ship_cluster_exclusion_x_margin", 60)
            ),
            ship_cluster_exclusion_y_margin=int(
                getattr(self.config.starfield, "ship_cluster_exclusion_y_margin", 110)
            ),
            candidate_min_radius=int(getattr(self.config.starfield, "candidate_min_radius", 6)),
            candidate_min_area=int(getattr(self.config.starfield, "candidate_min_area", 80)),
            min_ship_bbox_width=int(getattr(self.config.starfield, "min_ship_bbox_width", 20)),
            min_ship_bbox_height=int(getattr(self.config.starfield, "min_ship_bbox_height", 8)),
            min_ship_area=int(getattr(self.config.starfield, "min_ship_area", 150)),
            max_ship_radius=int(getattr(self.config.starfield, "max_ship_radius", 72)),
            max_ship_bbox_width=int(getattr(self.config.starfield, "max_ship_bbox_width", 140)),
            max_ship_bbox_height=int(getattr(self.config.starfield, "max_ship_bbox_height", 90)),
            max_ship_area_ratio=float(getattr(self.config.starfield, "max_ship_area_ratio", 0.08)),
        )
        target = (
            f" target=({probe.target_point[0]},{probe.target_point[1]})"
            if probe.target_point is not None
            else ""
        )
        print(f"[STARFIELD_PROBE] result={probe.reason}{target}")
        return 0 if probe.ok else 1

    def run_discover_nearest_planet_once(self) -> int:
        return self.run_discover_planet_rank_once(1)

    def run_discover_planet_rank_once(self, target_rank: int) -> int:
        if not bool(getattr(self.config.starfield, "enable_click_probe", False)):
            print("[PLANET_DISCOVERY] result=probe_disabled")
            return 1
        if not ensure_focus(self.config.focus):
            print("[PLANET_DISCOVERY] result=focus_unavailable")
            return 1
        planet_task = self.tasks.get("planets")
        if planet_task is None or getattr(planet_task, "reader", None) is None:
            print("[PLANET_DISCOVERY] result=probe_unavailable")
            return 1
        capture_screen = getattr(self.capture_backend, "capture_screen", None)
        if not callable(capture_screen):
            print("[PLANET_DISCOVERY] result=capture_unavailable")
            return 1
        frame = capture_screen()
        if frame is None:
            print("[PLANET_DISCOVERY] result=capture_unavailable")
            return 1
        run_dir = prepare_run_artifact_dir()
        frame_path = save_run_frame(frame, output_dir=run_dir)
        print(f"[PLANET_DISCOVERY] frame={frame_path}")

        def _starfield_ready_check():
            panel = planet_task.reader.read()
            reason = planet_task._probe_precondition_failure_reason(panel)
            if reason is None:
                return None
            return (reason, panel)

        def _return_to_starfield() -> bool:
            if not self.actions.close_planet_panel():
                return False
            panel = planet_task.reader.read()
            return not planet_task._panel_readable(panel)

        discovery = discover_starfield_planet_by_rank(
            capture=self.capture_backend,
            image=frame,
            actions=self.actions,
            reader=planet_task.reader,
            target_rank=target_rank,
            panel_is_readable=planet_task._panel_readable,
            starfield_ready_check=_starfield_ready_check,
            panel_is_confirmed=planet_task._probe_panel_confirmed,
            return_to_starfield=_return_to_starfield,
            settle_seconds=float(getattr(self.config.starfield, "click_probe_settle_seconds", 0.35)),
            save_annotation=bool(getattr(self.config.starfield, "save_probe_annotation", False)),
            annotation_dir=str(run_dir),
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
            ship_cluster_exclusion_x_margin=int(
                getattr(self.config.starfield, "ship_cluster_exclusion_x_margin", 60)
            ),
            ship_cluster_exclusion_y_margin=int(
                getattr(self.config.starfield, "ship_cluster_exclusion_y_margin", 110)
            ),
            candidate_min_radius=int(getattr(self.config.starfield, "candidate_min_radius", 6)),
            candidate_min_area=int(getattr(self.config.starfield, "candidate_min_area", 80)),
            min_ship_bbox_width=int(getattr(self.config.starfield, "min_ship_bbox_width", 20)),
            min_ship_bbox_height=int(getattr(self.config.starfield, "min_ship_bbox_height", 8)),
            min_ship_area=int(getattr(self.config.starfield, "min_ship_area", 150)),
            max_ship_radius=int(getattr(self.config.starfield, "max_ship_radius", 72)),
            max_ship_bbox_width=int(getattr(self.config.starfield, "max_ship_bbox_width", 140)),
            max_ship_bbox_height=int(getattr(self.config.starfield, "max_ship_bbox_height", 90)),
            max_ship_area_ratio=float(getattr(self.config.starfield, "max_ship_area_ratio", 0.08)),
        )
        target = (
            f" target=({discovery.target_point[0]},{discovery.target_point[1]})"
            if discovery.target_point is not None
            else ""
        )
        rank = f" rank={discovery.target_rank}" if discovery.target_rank is not None else ""
        title_raw = f' title_raw="{discovery.planet_title_raw}"' if discovery.planet_title_raw else ""
        title = f' title="{discovery.planet_title_canonical}"' if discovery.planet_title_canonical else ""
        returned = f" returned_to_starfield={'true' if discovery.returned_to_starfield else 'false'}"
        print(f"[PLANET_DISCOVERY] result={discovery.reason}{rank}{target}{title_raw}{title}{returned}")
        return 0 if discovery.ok else 1


def build_application() -> Application:
    return Application(load_runtime_config())
