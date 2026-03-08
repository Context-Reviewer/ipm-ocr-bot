from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class TaskResult:
    ok: bool = True
    details: dict[str, Any] = field(default_factory=dict)


class Task(Protocol):
    name: str

    def run(self) -> TaskResult:
        ...

