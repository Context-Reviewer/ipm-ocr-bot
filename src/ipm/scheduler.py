from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(slots=True, frozen=True)
class ScheduledTask:
    name: str
    interval_seconds: float


class Scheduler:
    def __init__(self, tasks: Iterable[ScheduledTask]) -> None:
        self._tasks = list(tasks)

    @property
    def tasks(self) -> list[ScheduledTask]:
        return list(self._tasks)

    def seed(self, now: float) -> dict[str, float]:
        return {task.name: now for task in self._tasks}

    def due(self, now: float, next_run_at: dict[str, float]) -> list[ScheduledTask]:
        ready: list[ScheduledTask] = []
        for task in self._tasks:
            if now >= float(next_run_at.get(task.name, 0.0)):
                ready.append(task)
        return ready

    def mark_complete(self, task: ScheduledTask, now: float, next_run_at: dict[str, float]) -> None:
        next_run_at[task.name] = now + task.interval_seconds

