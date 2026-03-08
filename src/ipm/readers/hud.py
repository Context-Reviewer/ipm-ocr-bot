from __future__ import annotations

from dataclasses import dataclass

from ..config import RuntimeConfig
from ..rects import RectStore
from .common import parse_compact_number


@dataclass(slots=True)
class HudReader:
    config: RuntimeConfig
    rects: RectStore
    capture: object
    perception: object

    def read_cash(self) -> tuple[int | None, str]:
        rect = self.rects.get("HUD_CASH")
        if rect is None:
            return None, ""
        image = self.capture.capture_client_bbox(rect)
        if image is None:
            return None, ""
        result = self.perception.read_text(
            image,
            prompt=self.config.perception.prompt_numeric,
            mode="numeric",
        )
        return parse_compact_number(result.value), result.backend
