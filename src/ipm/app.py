from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import keyboard
from PIL import Image
from PIL import ImageStat

from .actions import ActionDriver
from .capture import create_capture_backend
from .config import RuntimeConfig, load_runtime_config
from .focus import ensure_focus_result, get_active_window_title
from .perception import create_perception_backend
from .readers import HudReader, OrePanelReader, PlanetPanelReader, SellDialogReader
from .rects import RectStore
from .runtime import RuntimeState
from .scheduler import ScheduledTask, Scheduler
from .starfield_probe import (
    discover_starfield_planet_by_rank,
    format_starfield_cache_run_rollup,
    merge_starfield_cache_summaries,
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


@dataclass(slots=True, frozen=True)
class UIStateCheck:
    state: str
    detail: str = ""
    panel_visible: bool = False
    starfield_ready: bool = False


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

    @staticmethod
    def _format_focus_failure(result) -> str:
        details = [
            f"reason={result.reason}",
            f"active_title_before={result.active_title_before!r}",
            f"active_title_after={result.active_title_after!r}",
        ]
        if result.activation_status:
            details.append(f"activation_status={result.activation_status}")
        if result.activation_error:
            details.append(f"activation_error={result.activation_error!r}")
        if result.target_title:
            details.append(f"target_title={result.target_title!r}")
        return " ".join(details)

    def tick(self) -> None:
        now = time.monotonic()
        focus_result = ensure_focus_result(self.config.focus)
        if not focus_result.ok:
            print(f"[AFK] waiting for focus: {self._format_focus_failure(focus_result)}")
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
        focus_result = ensure_focus_result(self.config.focus)
        if not focus_result.ok:
            print(f"[STARFIELD_PROBE] result=focus_unavailable {self._format_focus_failure(focus_result)}")
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
            expected_orientation=str(getattr(self.config.starfield, "expected_orientation", "landscape") or ""),
            planet_node_cache_path=str(
                getattr(self.config.starfield, "planet_node_cache_path", "out/starfield_planet_nodes.json") or ""
            ),
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
        summary = merge_starfield_cache_summaries(probe.cache_summary)
        if summary is not None:
            print(format_starfield_cache_run_rollup(summary, boundary="starfield_probe_command"))
        target = (
            f" target=({probe.target_point[0]},{probe.target_point[1]})"
            if probe.target_point is not None
            else ""
        )
        print(f"[STARFIELD_PROBE] result={probe.reason}{target}")
        return 0 if probe.ok else 1

    def run_discover_nearest_planet_once(self) -> int:
        return self.run_discover_planet_rank_once(1)

    def _capture_frame(self) -> Image.Image | None:
        capture_screen = getattr(self.capture_backend, "capture_screen", None)
        if not callable(capture_screen):
            return None
        return capture_screen()

    def _crop_rect_image(self, image: Image.Image, rect_key: str) -> Image.Image | None:
        rect = self.rects.get(rect_key)
        if rect is None:
            return None
        x, y, w, h = rect
        left = max(0, int(x))
        top = max(0, int(y))
        right = min(image.width, left + max(0, int(w)))
        bottom = min(image.height, top + max(0, int(h)))
        if right <= left or bottom <= top:
            return None
        return image.crop((left, top, right, bottom))

    @staticmethod
    def _normalized_crop_image(
        image: Image.Image,
        region: tuple[float, float, float, float],
    ) -> Image.Image | None:
        left, top, right, bottom = region
        crop = image.crop(
            (
                max(0, int(round(image.width * float(left)))),
                max(0, int(round(image.height * float(top)))),
                min(image.width, int(round(image.width * float(right)))),
                min(image.height, int(round(image.height * float(bottom)))),
            )
        )
        if crop.width < 2 or crop.height < 2:
            return None
        return crop

    @staticmethod
    def _crop_signal_stats(image: Image.Image | None, *, bright_threshold: int = 150) -> dict[str, float] | None:
        if image is None:
            return None
        gray = image.convert("L")
        stat = ImageStat.Stat(gray)
        extrema = stat.extrema[0]
        histogram = gray.histogram()
        total = float(sum(int(count) for count in histogram))
        if total <= 0.0:
            return None
        return {
            "mean": float(stat.mean[0]),
            "dynamic_range": float(int(extrema[1]) - int(extrema[0])),
            "bright_fraction": sum(float(histogram[index]) for index in range(int(bright_threshold), 256)) / total,
            "dark_fraction": sum(float(histogram[index]) for index in range(0, 81)) / total,
        }

    @classmethod
    def _crop_has_panel_signal(
        cls,
        image: Image.Image | None,
        *,
        min_dynamic_range: int,
        min_bright_fraction: float,
        bright_threshold: int = 150,
    ) -> bool:
        stats = cls._crop_signal_stats(image, bright_threshold=bright_threshold)
        if stats is None:
            return False
        return (
            stats["dynamic_range"] >= int(min_dynamic_range)
            and stats["bright_fraction"] >= float(min_bright_fraction)
        )

    def _planet_panel_likely_visible(self, image: Image.Image) -> bool:
        close_crop = self._crop_rect_image(image, "PLANET_PANEL_CLOSE")
        title_crop = self._crop_rect_image(image, "PLANET_TITLE")
        close_signal = self._crop_has_panel_signal(
            close_crop,
            min_dynamic_range=28,
            min_bright_fraction=0.08,
        )
        title_signal = self._crop_has_panel_signal(
            title_crop,
            min_dynamic_range=20,
            min_bright_fraction=0.03,
            bright_threshold=135,
        )
        return close_signal and title_signal

    def _central_overlay_likely_visible(self, image: Image.Image) -> bool:
        panel_crop = self._crop_rect_image(image, "PLANET_PANEL_TEXT")
        if panel_crop is None:
            panel_crop = self._normalized_crop_image(image, (0.08, 0.14, 0.88, 0.86))
        stats = self._crop_signal_stats(panel_crop, bright_threshold=150)
        if stats is None:
            return False
        return stats["mean"] <= 45.0 and stats["dark_fraction"] >= 0.90

    def _evaluate_ui_state(self, image: Image.Image) -> UIStateCheck:
        if self._planet_panel_likely_visible(image):
            return UIStateCheck(
                state="planet_panel_present",
                detail="panel_controls_visible",
                panel_visible=True,
                starfield_ready=False,
            )
        if self._central_overlay_likely_visible(image):
            return UIStateCheck(
                state="overlay_present",
                detail="central_modal_overlay",
                panel_visible=False,
                starfield_ready=False,
            )
        return UIStateCheck(
            state="starfield_ready",
            detail="panel_absent",
            panel_visible=False,
            starfield_ready=True,
        )

    @staticmethod
    def _log_ui_state(stage: str, state_check: UIStateCheck) -> None:
        print(
            "[UI_STATE] "
            f"stage={stage} state={state_check.state} "
            f"panel_visible={'true' if state_check.panel_visible else 'false'} "
            f"starfield_ready={'true' if state_check.starfield_ready else 'false'} "
            f"detail={state_check.detail}"
        )

    def _click_expanded_planet_close(self) -> bool:
        rect = self.rects.get("PLANET_PANEL_CLOSE")
        click_point = getattr(self.actions, "click_client_point", None)
        if rect is None or not callable(click_point):
            return False
        x, y, w, h = rect
        center_x = int(x + (w // 2))
        center_y = int(y + (h // 2))
        offsets = ((10, -10), (10, 10), (-10, -10), (-10, 10))
        clicked = False
        for offset_x, offset_y in offsets:
            clicked = click_point(
                (center_x + offset_x, center_y + offset_y),
                delay=self.config.actions.menu_delay_seconds,
            ) or clicked
        return clicked

    def _click_safe_dismiss(self, image: Image.Image) -> bool:
        click_point = getattr(self.actions, "click_client_point", None)
        if not callable(click_point):
            return False
        point = (32, min(96, max(0, image.height - 24)))
        return click_point(point, delay=self.config.actions.menu_delay_seconds)

    def _reset_ui_recovery(self) -> bool:
        reset_ui = getattr(self.actions, "reset_ui", None)
        if not callable(reset_ui):
            return False
        reset_ui()
        return True

    def _recover_starfield(self, *, stage: str, image: Image.Image) -> tuple[bool, Image.Image | None]:
        state_check = self._evaluate_ui_state(image)
        self._log_ui_state(f"{stage}_check", state_check)
        if state_check.starfield_ready:
            return True, image

        attempts: list[tuple[str, object]] = []
        if state_check.panel_visible:
            attempts.extend(
                [
                    (
                        "close_rect_primary",
                        lambda: self.actions.click_rect_center(
                            "PLANET_PANEL_CLOSE",
                            delay=self.config.actions.menu_delay_seconds,
                        ),
                    ),
                    ("close_rect_expanded", self._click_expanded_planet_close),
                ]
            )
        attempts.append(("safe_dismiss", lambda: self._click_safe_dismiss(image)))
        if state_check.state == "overlay_present":
            attempts.append(("reset_ui", self._reset_ui_recovery))

        for attempt_name, attempt_action in attempts:
            clicked = bool(attempt_action())
            print(
                "[UI_STATE] "
                f"stage={stage}_recovery action={attempt_name} "
                f"clicked={'true' if clicked else 'false'}"
            )
            verify_frame = self._capture_frame()
            if verify_frame is None:
                print(f"[UI_STATE] stage={stage}_verify state=capture_unavailable")
                continue
            state_check = self._evaluate_ui_state(verify_frame)
            self._log_ui_state(f"{stage}_verify_{attempt_name}", state_check)
            if state_check.starfield_ready:
                return True, verify_frame
        return False, None

    def run_discover_planet_rank_once(self, target_rank: int) -> int:
        if not bool(getattr(self.config.starfield, "enable_click_probe", False)):
            print("[PLANET_DISCOVERY] result=probe_disabled")
            return 1
        focus_result = ensure_focus_result(self.config.focus)
        if not focus_result.ok:
            print(f"[PLANET_DISCOVERY] result=focus_unavailable {self._format_focus_failure(focus_result)}")
            return 1
        planet_task = self.tasks.get("planets")
        if planet_task is None or getattr(planet_task, "reader", None) is None:
            print("[PLANET_DISCOVERY] result=probe_unavailable")
            return 1
        precheck_frame = self._capture_frame()
        if precheck_frame is None:
            print("[PLANET_DISCOVERY] result=capture_unavailable")
            return 1
        ready, frame = self._recover_starfield(stage="pre_run", image=precheck_frame)
        if not ready or frame is None:
            print("[PLANET_DISCOVERY] result=not_starfield_ready returned_to_starfield=false")
            return 1
        if frame is None:
            print("[PLANET_DISCOVERY] result=capture_unavailable")
            return 1
        run_dir = prepare_run_artifact_dir()
        frame_path = save_run_frame(frame, output_dir=run_dir)
        print(f"[PLANET_DISCOVERY] frame={frame_path}")

        def _return_to_starfield() -> bool:
            post_open_frame = self._capture_frame()
            if post_open_frame is None:
                print("[UI_STATE] stage=post_open_check state=capture_unavailable")
                return False
            recovered, _ = self._recover_starfield(stage="post_open", image=post_open_frame)
            return recovered

        discovery = discover_starfield_planet_by_rank(
            capture=self.capture_backend,
            image=frame,
            actions=self.actions,
            reader=planet_task.reader,
            target_rank=target_rank,
            panel_is_readable=planet_task._panel_readable,
            panel_is_confirmed=planet_task._probe_panel_confirmed,
            return_to_starfield=_return_to_starfield,
            settle_seconds=float(getattr(self.config.starfield, "click_probe_settle_seconds", 0.35)),
            save_annotation=bool(getattr(self.config.starfield, "save_probe_annotation", False)),
            annotation_dir=str(run_dir),
            expected_orientation=str(getattr(self.config.starfield, "expected_orientation", "landscape") or ""),
            planet_node_cache_path=str(
                getattr(self.config.starfield, "planet_node_cache_path", "out/starfield_planet_nodes.json") or ""
            ),
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
        summary = merge_starfield_cache_summaries(discovery.cache_summary)
        if summary is not None:
            print(format_starfield_cache_run_rollup(summary, boundary="planet_discovery_command"))
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
        return 0 if discovery.ok and discovery.reason == "ok" else 1


def build_application() -> Application:
    return Application(load_runtime_config())
