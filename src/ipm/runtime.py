from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass(slots=True)
class RuntimeState:
    running: bool = False
    stop_requested: bool = False
    last_heartbeat_at: float = 0.0
    next_run_at: Dict[str, float] = field(default_factory=dict)

