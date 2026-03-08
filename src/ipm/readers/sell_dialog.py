from __future__ import annotations

from dataclasses import dataclass

from ..config import RuntimeConfig
from ..rects import RectStore
from ..state import SellDialogState
from .common import parse_compact_number


@dataclass(slots=True)
class SellDialogReader:
    config: RuntimeConfig
    rects: RectStore
    capture: object
    perception: object

    def read(self) -> SellDialogState:
        qty_rect = self.rects.get("SELL_SELECTED_QTY")
        slider_rect = self.rects.get("SELL_SLIDER_TRACK")
        if qty_rect is None and slider_rect is None:
            return SellDialogState()

        qty_text = ""
        backend = ""
        if qty_rect is not None:
            image = self.capture.capture_client_bbox(qty_rect)
            if image is not None:
                result = self.perception.read_text(
                    image,
                    prompt=self.config.perception.prompt_ore_quantity,
                    mode="ore_qty",
                )
                qty_text = result.value.strip()
                backend = result.backend
        return SellDialogState(
            selected_quantity=parse_compact_number(qty_text),
            slider_visible=slider_rect is not None,
            backend=backend,
        )
