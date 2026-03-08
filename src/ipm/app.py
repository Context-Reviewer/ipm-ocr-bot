from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import keyboard

from .actions import ActionDriver
from .capture import create_capture_backend
from .config import RuntimeConfig, load_runtime_config
from .focus import ensure_focus, get_active_window_title
from .perception import create_perception_backend
from .readers import HudReader, OrePanelReader, PlanetPanelReader, SellDialogReader
from .rects import RectStore
from .runtime import RuntimeState
from .scheduler import ScheduledTask, Scheduler
from .state_reader import GameStateReader
from .tasks import OresTask, PlanetsTask


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


def build_application() -> Application:
    return Application(load_runtime_config())
