from __future__ import annotations

from dataclasses import dataclass
import re

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageStat

from ..state import ProductionOverviewCardState
from .common import parse_compact_number

_TIMER_TEXT_RE = re.compile(r"[0-9].*[SMH]|[0-9]+:[0-9]+", re.IGNORECASE)
_TIMER_PROMPT = "Read only the visible timer text or OFF. Return only the timer text."
_COUNT_PROMPT = "Read only the visible output quantity. Keep suffixes like K or M if present."


@dataclass(slots=True)
class ProductionOverviewReader:
    rects: object
    capture: object
    actions: object
    perception: object

    def _capture_rect(self, rect_key: str) -> Image.Image | None:
        rect = getattr(self.rects, "get", lambda _key: None)(rect_key)
        if rect is None:
            return None
        return self.capture.capture_client_bbox(rect)

    @staticmethod
    def _image_mean_abs_diff(previous: Image.Image | None, current: Image.Image | None) -> float:
        if previous is None or current is None:
            return 1e9
        diff = ImageChops.difference(previous.convert("L"), current.convert("L"))
        return float(ImageStat.Stat(diff).mean[0])

    @staticmethod
    def _normalize_timer_text(text: str | None) -> str:
        return re.sub(r"\s+", "", str(text or "").upper())

    @staticmethod
    def _card_status(card: Image.Image) -> str:
        lock_crop = card.crop((75, 45, 205, 220))
        arr = np.asarray(lock_crop.convert("RGB"), dtype=np.uint8)
        if arr.size:
            cyan_fraction = float(np.mean((arr[..., 1] >= 145) & (arr[..., 2] >= 145) & (arr[..., 0] <= 90)))
            if cyan_fraction >= 0.12:
                return "locked"
        grayscale = card.convert("L")
        extrema = grayscale.getextrema()
        dynamic_range = int(extrema[1]) - int(extrema[0])
        mean_value = float(ImageStat.Stat(grayscale).mean[0])
        if dynamic_range < 40 and mean_value < 55.0:
            return "empty"
        return "card"

    @staticmethod
    def _progress_fill_fraction(card: Image.Image) -> float:
        bar_crop = card.crop((32, 145, 205, 183))
        arr = np.asarray(bar_crop.convert("RGB"), dtype=np.uint8)
        if arr.size == 0:
            return 0.0
        return float(np.mean((arr[..., 1] >= 140) & (arr[..., 2] >= 140) & (arr[..., 0] <= 120)))

    def _read_timer_text(self, card: Image.Image) -> tuple[str | None, str]:
        crop = card.crop((70, 160, 185, 210))
        result = self.perception.read_text(crop, prompt=_TIMER_PROMPT, mode="generic")
        value = str(getattr(result, "value", "") or "").strip()
        return (value or None), str(getattr(result, "backend", "") or "")

    def _read_output_quantity(self, card: Image.Image) -> tuple[int | None, str]:
        crop = card.crop((140, 80, 230, 135))
        result = self.perception.read_text(crop, prompt=_COUNT_PROMPT, mode="ore_qty")
        return parse_compact_number(getattr(result, "value", "")), str(getattr(result, "backend", "") or "")

    @staticmethod
    def _extract_template_icon(image: Image.Image) -> Image.Image:
        arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        edge = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 40, 120)
        num, _labels, stats, _centroids = cv2.connectedComponentsWithStats((edge > 0).astype(np.uint8), 8)
        best_area = 0
        best_bbox: tuple[int, int, int, int] | None = None
        for index in range(1, num):
            x, y, w, h, area = (int(value) for value in stats[index])
            if area < 40 or w < 12 or h < 12:
                continue
            if area > best_area:
                best_area = area
                best_bbox = (x, y, w, h)
        if best_bbox is None:
            return image
        x, y, w, h = best_bbox
        pad = 6
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(arr.shape[1], x + w + pad)
        y2 = min(arr.shape[0], y + h + pad)
        return Image.fromarray(arr[y1:y2, x1:x2])

    @classmethod
    def _extract_output_icon(cls, card: Image.Image) -> Image.Image | None:
        arr = np.asarray(card.convert("RGB"), dtype=np.uint8)
        region = arr[50:180, 120:235]
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        edge = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 40, 120)
        num, _labels, stats, _centroids = cv2.connectedComponentsWithStats((edge > 0).astype(np.uint8), 8)
        best: tuple[int, tuple[int, int, int, int]] | None = None
        for index in range(1, num):
            x, y, w, h, area = (int(value) for value in stats[index])
            if area < 60 or w < 18 or h < 18:
                continue
            candidate = (area, (x, y, w, h))
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best is None:
            return None
        x, y, w, h = best[1]
        pad = 8
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(region.shape[1], x + w + pad)
        y2 = min(region.shape[0], y + h + pad)
        return Image.fromarray(region[y1:y2, x1:x2])

    @classmethod
    def _icon_similarity(cls, template: Image.Image, target: Image.Image) -> float:
        template_work = cls._extract_template_icon(template).resize((56, 56)).convert("RGB")
        target_work = target.resize((56, 56)).convert("RGB")
        template_rgb = np.asarray(template_work, dtype=np.float32)
        target_rgb = np.asarray(target_work, dtype=np.float32)
        template_gray = cv2.cvtColor(template_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        target_gray = cv2.cvtColor(target_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        template_edge = cv2.Canny(template_gray, 40, 120).astype(np.float32) / 255.0
        target_edge = cv2.Canny(target_gray, 40, 120).astype(np.float32) / 255.0
        edge_score = 1.0 - float(np.mean(np.abs(template_edge - target_edge)))
        color_score = 1.0 - float(np.mean(np.abs(template_rgb - target_rgb)) / 255.0)
        return (0.6 * edge_score) + (0.4 * color_score)

    @staticmethod
    def _quantity_match_bonus(name: str, *, quantity: int | None, inventory_counts: dict[str, int]) -> float:
        if quantity is None:
            return 0.0
        known_quantity = int(inventory_counts.get(name, -1))
        if known_quantity < 0:
            return 0.0
        tolerance = max(3, int(round(max(1, quantity) * 0.02)))
        return 0.08 if abs(known_quantity - quantity) <= tolerance else 0.0

    def _resolve_output_name(
        self,
        *,
        card: Image.Image,
        templates: dict[str, Image.Image],
        inventory_counts: dict[str, int],
        output_quantity: int | None,
    ) -> tuple[str, str]:
        target_icon = self._extract_output_icon(card)
        if target_icon is None:
            raise ValueError("output_icon_not_found")
        scored = sorted(
            (
                (
                    name,
                    self._icon_similarity(template, target_icon)
                    + self._quantity_match_bonus(name, quantity=output_quantity, inventory_counts=inventory_counts),
                )
                for name, template in templates.items()
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        if not scored:
            raise ValueError("icon_templates_missing")
        top_name, top_score = scored[0]
        next_score = scored[1][1] if len(scored) > 1 else 0.0
        if top_score < 0.76 or (top_score - next_score) < 0.02:
            raise ValueError(f"ambiguous_output_match:{top_name}:{top_score:.4f}:{next_score:.4f}")
        return top_name, "icon_template_match"

    def _resolve_active_state(self, card: Image.Image) -> tuple[bool, str | None, str]:
        timer_text, timer_backend = self._read_timer_text(card)
        normalized = self._normalize_timer_text(timer_text)
        fill_fraction = self._progress_fill_fraction(card)
        if normalized and _TIMER_TEXT_RE.search(normalized):
            return True, timer_text, timer_backend or "timer_text"
        if fill_fraction >= 0.05:
            if normalized and normalized != "OFF" and (_TIMER_TEXT_RE.search(normalized) or normalized.endswith("S")):
                return True, timer_text, timer_backend
            return True, timer_text, "progress_fill_signal"
        if normalized == "OFF":
            return False, None, timer_backend or "timer_text"
        raise ValueError(f"unreadable_active_state:{normalized or 'blank'}")

    def _read_card(
        self,
        *,
        slot_index: int,
        tab: str,
        rect_key: str,
        templates: dict[str, Image.Image],
        inventory_counts: dict[str, int],
    ) -> ProductionOverviewCardState:
        card = self._capture_rect(rect_key)
        if card is None:
            raise ValueError(f"missing_rect:{rect_key}")
        status = self._card_status(card)
        if status == "empty":
            return ProductionOverviewCardState(
                slot_index=slot_index,
                tab=tab,
                output_name="",
                active=False,
                timer_text=None,
                backend="empty_slot",
            )
        if status == "locked":
            return ProductionOverviewCardState(
                slot_index=slot_index,
                tab=tab,
                output_name="",
                active=False,
                timer_text=None,
                backend="locked_slot",
            )
        output_quantity, quantity_backend = self._read_output_quantity(card)
        output_name, match_backend = self._resolve_output_name(
            card=card,
            templates=templates,
            inventory_counts=inventory_counts,
            output_quantity=output_quantity,
        )
        active, timer_text, state_backend = self._resolve_active_state(card)
        backend_parts = [part for part in (match_backend, quantity_backend, state_backend) if part]
        return ProductionOverviewCardState(
            slot_index=slot_index,
            tab=tab,
            output_name=output_name,
            active=active,
            timer_text=timer_text,
            backend="+".join(backend_parts),
        )

    def _wheel_point(self) -> tuple[int, int]:
        rect = getattr(self.rects, "get", lambda _key: None)("PRODUCTION_CARD3")
        if rect is None:
            raise ValueError("missing_rect:PRODUCTION_CARD3")
        x, y, w, h = rect
        return (int(x + (w // 2)), int(y + (h // 2)))

    def _scroll_to_lower_cards(self, *, top_anchor: Image.Image | None) -> None:
        point = self._wheel_point()
        if not self.actions.scroll_client_wheel(point, -120):
            raise ValueError("production_scroll_down_failed")
        current = self._capture_rect("PRODUCTION_CARD1")
        if self._image_mean_abs_diff(top_anchor, current) <= 1.0:
            raise ValueError("production_scroll_down_not_observed")

    def _scroll_to_top_view(self) -> Image.Image | None:
        point = self._wheel_point()
        for _ in range(4):
            current = self._capture_rect("PRODUCTION_CARD1")
            if current is not None and self._extract_output_icon(current) is not None:
                return current
            if not self.actions.scroll_client_wheel(point, 120):
                raise ValueError("production_scroll_up_failed")
        raise ValueError("production_top_latch_failed")

    def _scroll_back_to_top(self, *, top_anchor: Image.Image | None) -> None:
        if top_anchor is None:
            return
        point = self._wheel_point()
        for _ in range(3):
            current = self._capture_rect("PRODUCTION_CARD1")
            if self._image_mean_abs_diff(top_anchor, current) <= 1.0:
                return
            if not self.actions.scroll_client_wheel(point, 120):
                break
        current = self._capture_rect("PRODUCTION_CARD1")
        if self._image_mean_abs_diff(top_anchor, current) > 1.0:
            raise ValueError("production_scroll_up_failed")

    def read_cards(
        self,
        *,
        tab: str,
        open_tab,
        templates: dict[str, Image.Image],
        inventory_counts: dict[str, int],
    ) -> list[ProductionOverviewCardState]:
        for rect_key in ("PRODUCTION_CARD1", "PRODUCTION_CARD2", "PRODUCTION_CARD3", "PRODUCTION_CARD4"):
            if getattr(self.rects, "get", lambda _key: None)(rect_key) is None:
                raise ValueError(f"missing_rect:{rect_key}")
        if not templates:
            raise ValueError(f"empty_template_bank:{tab}")
        if not open_tab():
            raise ValueError(f"open_tab_failed:{tab}")
        top_anchor = self._scroll_to_top_view()
        cards = [
            self._read_card(slot_index=1, tab=tab, rect_key="PRODUCTION_CARD1", templates=templates, inventory_counts=inventory_counts),
            self._read_card(slot_index=2, tab=tab, rect_key="PRODUCTION_CARD2", templates=templates, inventory_counts=inventory_counts),
        ]
        self._scroll_to_lower_cards(top_anchor=top_anchor)
        try:
            cards.extend(
                [
                    self._read_card(slot_index=3, tab=tab, rect_key="PRODUCTION_CARD3", templates=templates, inventory_counts=inventory_counts),
                    self._read_card(slot_index=4, tab=tab, rect_key="PRODUCTION_CARD4", templates=templates, inventory_counts=inventory_counts),
                ]
            )
        finally:
            self._scroll_back_to_top(top_anchor=top_anchor)
        return cards
