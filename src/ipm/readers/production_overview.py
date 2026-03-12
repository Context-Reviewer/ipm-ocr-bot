from __future__ import annotations

from dataclasses import dataclass
import re

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageStat

import bars_data
import items_data
from ..domain_data import normalize_resource_row_name
from ..state import ProductionOverviewCardState
from .common import parse_compact_number

_TIMER_TEXT_RE = re.compile(r"[0-9].*[SMH]|[0-9]+:[0-9]+", re.IGNORECASE)
_TIMER_PROMPT = "Read only the visible timer text or OFF. Return only the timer text."
_COUNT_PROMPT = "Read only the visible output quantity. Keep suffixes like K or M if present."
_PRODUCTION_SCROLL_DELAY_SECONDS = 0.35
_INPUT_SIGNAL_WEIGHT = 0.35
_SMELT_OUTPUT_REGION_WEIGHT = 0.20


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

    @staticmethod
    def _region_signal_stats(image: Image.Image) -> dict[str, float]:
        grayscale = image.convert("L")
        histogram = grayscale.histogram()
        total = max(1.0, float(sum(histogram)))
        gray_arr = np.asarray(grayscale, dtype=np.uint8)
        rgb_arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
        edge = cv2.Canny(gray_arr, 40, 120) if gray_arr.size else np.zeros((0, 0), dtype=np.uint8)
        extrema = grayscale.getextrema()
        return {
            "dynamic_range": float(int(extrema[1]) - int(extrema[0])),
            "bright_fraction": sum(float(histogram[index]) for index in range(170, 256)) / total,
            "dark_fraction": sum(float(histogram[index]) for index in range(0, 80)) / total,
            "edge_fraction": float(np.mean(edge > 0)) if edge.size else 0.0,
            "cyan_fraction": float(
                np.mean((rgb_arr[..., 1] >= 130) & (rgb_arr[..., 2] >= 130) & (rgb_arr[..., 0] <= 120))
            )
            if rgb_arr.size
            else 0.0,
        }

    @classmethod
    def _cancel_button_signal(cls, card: Image.Image) -> bool:
        stats = cls._region_signal_stats(card.crop((185, 145, 225, 185)))
        return (
            stats["dynamic_range"] >= 160.0
            and stats["bright_fraction"] >= 0.09
            and stats["dark_fraction"] <= 0.20
            and stats["edge_fraction"] >= 0.10
        )

    @classmethod
    def _timer_region_signal(cls, card: Image.Image) -> bool:
        stats = cls._region_signal_stats(card.crop((55, 145, 185, 205)))
        return stats["dynamic_range"] >= 240.0 and (
            stats["cyan_fraction"] >= 0.10
            or (stats["edge_fraction"] >= 0.068 and stats["bright_fraction"] >= 0.03)
        )

    def _read_timer_text(self, card: Image.Image) -> tuple[str | None, str]:
        crop = card.crop((55, 145, 185, 205))
        result = self.perception.read_text(crop, prompt=_TIMER_PROMPT, mode="generic")
        value = str(getattr(result, "value", "") or "").strip()
        return (value or None), str(getattr(result, "backend", "") or "")

    def _read_output_quantity(self, card: Image.Image) -> tuple[int | None, str]:
        crop = card.crop((140, 80, 230, 135))
        result = self.perception.read_text(crop, prompt=_COUNT_PROMPT, mode="ore_qty")
        return parse_compact_number(getattr(result, "value", "")), str(getattr(result, "backend", "") or "")

    @staticmethod
    def _border_median(arr: np.ndarray) -> np.ndarray:
        border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]], axis=0)
        return np.median(border, axis=0)

    @classmethod
    def _foreground_mask(cls, arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return np.zeros((0, 0), dtype=np.uint8)
        bg_rgb = cls._border_median(arr)
        diff = np.abs(arr.astype(np.int16) - bg_rgb.astype(np.int16)).sum(axis=2)
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        bg_hsv = cv2.cvtColor(np.uint8([[bg_rgb.astype(np.uint8)]]), cv2.COLOR_RGB2HSV)[0, 0]
        saturation_delta = hsv[..., 1].astype(np.int16) - int(bg_hsv[1])
        value_delta = hsv[..., 2].astype(np.int16) - int(bg_hsv[2])
        mask = (
            (diff > 40)
            | (saturation_delta > 24)
            | (value_delta > 18)
            | ((hsv[..., 2] < 75) & (diff > 28))
        ).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
        return mask

    @classmethod
    def _trim_foreground_icon(
        cls,
        image: Image.Image,
        *,
        prefer_center_x: float = 0.55,
        prefer_center_y: float = 0.45,
    ) -> Image.Image | None:
        arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if arr.size == 0:
            return None
        mask = cls._foreground_mask(arr)
        num, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
        best: tuple[float, tuple[int, int, int, int]] | None = None
        for index in range(1, num):
            x, y, w, h, area = (int(value) for value in stats[index])
            if area < 35 or w < 8 or h < 8:
                continue
            coverage = float(area) / float(max(1, arr.shape[0] * arr.shape[1]))
            if coverage > 0.80:
                continue
            center_x = x + (w / 2.0)
            center_y = y + (h / 2.0)
            distance = (
                abs(center_x - (arr.shape[1] * prefer_center_x))
                + abs(center_y - (arr.shape[0] * prefer_center_y))
            )
            score = float(area) - (distance * 1.5)
            candidate = (score, (x, y, w, h))
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best is None:
            return None
        x, y, w, h = best[1]
        pad = 4
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(arr.shape[1], x + w + pad)
        y2 = min(arr.shape[0], y + h + pad)
        return Image.fromarray(arr[y1:y2, x1:x2])

    @staticmethod
    def _extract_template_icon(image: Image.Image) -> Image.Image:
        trimmed = ProductionOverviewReader._trim_foreground_icon(image, prefer_center_x=0.5, prefer_center_y=0.5)
        return trimmed if trimmed is not None else image

    @staticmethod
    def _output_candidate_boxes(card: Image.Image) -> list[tuple[int, int, int, int]]:
        width, height = card.size
        fractions = [
            (0.48, 0.25, 0.86, 0.53),
            (0.42, 0.25, 0.80, 0.53),
            (0.38, 0.30, 0.82, 0.58),
        ]
        boxes: list[tuple[int, int, int, int]] = []
        for left, top, right, bottom in fractions:
            x1 = max(0, min(width - 1, int(round(width * left))))
            y1 = max(0, min(height - 1, int(round(height * top))))
            x2 = max(x1 + 1, min(width, int(round(width * right))))
            y2 = max(y1 + 1, min(height, int(round(height * bottom))))
            boxes.append((x1, y1, x2, y2))
        return boxes

    @classmethod
    def _candidate_output_icons(cls, card: Image.Image) -> list[Image.Image]:
        icons: list[Image.Image] = []
        for box in cls._output_candidate_boxes(card):
            crop = card.crop(box)
            trimmed = cls._trim_foreground_icon(crop)
            candidate = trimmed if trimmed is not None else crop
            grayscale = candidate.convert("L")
            extrema = grayscale.getextrema()
            if int(extrema[1]) - int(extrema[0]) < 18:
                continue
            icons.append(candidate)
        return icons

    @classmethod
    def _extract_output_icon(cls, card: Image.Image) -> Image.Image | None:
        candidates = cls._candidate_output_icons(card)
        if not candidates:
            return None
        return max(candidates, key=lambda image: image.size[0] * image.size[1])

    @staticmethod
    def _material_aliases(name: str) -> list[str]:
        normalized_name = str(name or "").strip()
        aliases = [normalized_name]
        if not normalized_name:
            return aliases
        normalized_resource = normalize_resource_row_name(normalized_name)
        if normalized_resource and normalized_resource not in aliases:
            aliases.append(normalized_resource)
        explicit_aliases = {
            "Silicon": "Silica",
            "Silica": "Silicon",
            "Aluminum": "Aluminium",
            "Aluminium": "Aluminum",
            "Aluminum Bar": "Aluminium Bar",
            "Aluminium Bar": "Aluminum Bar",
        }
        alias_name = explicit_aliases.get(normalized_name)
        if alias_name and alias_name not in aliases:
            aliases.append(alias_name)
        for suffix in (" Bar", " Alloy"):
            if not normalized_name.endswith(suffix):
                continue
            base_name = normalized_name[: -len(suffix)].strip()
            normalized_base = normalize_resource_row_name(base_name)
            if normalized_base:
                candidate = f"{normalized_base}{suffix}"
                if candidate not in aliases:
                    aliases.append(candidate)
        return aliases

    @classmethod
    def _lookup_template(cls, templates: dict[str, Image.Image], name: str) -> Image.Image | None:
        for alias in cls._material_aliases(name):
            template = templates.get(alias)
            if template is not None:
                return template
        return None

    @staticmethod
    def _input_search_boxes(card: Image.Image, *, tab: str) -> list[tuple[int, int, int, int]]:
        width, height = card.size
        if tab == "smelt":
            fractions = [
                (0.00, 0.19, 0.38, 0.53),
                (0.00, 0.24, 0.26, 0.50),
                (0.06, 0.22, 0.34, 0.50),
            ]
        else:
            fractions = [
                (0.19, 0.10, 0.52, 0.53),
            ]
        boxes: list[tuple[int, int, int, int]] = []
        for left, top, right, bottom in fractions:
            boxes.append(
                (
                    int(round(width * left)),
                    int(round(height * top)),
                    int(round(width * right)),
                    int(round(height * bottom)),
                )
            )
        return boxes

    @classmethod
    def _template_presence_score(cls, *, template: Image.Image, search: Image.Image) -> float:
        search_rgb = np.asarray(search.convert("RGB"), dtype=np.uint8)
        if search_rgb.size == 0:
            return 0.0
        search_gray = cv2.cvtColor(search_rgb, cv2.COLOR_RGB2GRAY)
        search_edge = cv2.Canny(cv2.GaussianBlur(search_gray, (3, 3), 0), 30, 100)
        template_rgb = np.asarray(cls._extract_template_icon(template).convert("RGB"), dtype=np.uint8)
        best_score = -1.0
        for scale in (0.70, 0.85, 1.00, 1.15, 1.30, 1.45):
            width = max(12, int(round(template_rgb.shape[1] * scale)))
            height = max(12, int(round(template_rgb.shape[0] * scale)))
            if width >= search_rgb.shape[1] or height >= search_rgb.shape[0]:
                continue
            resized = cv2.resize(template_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
            template_gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
            template_edge = cv2.Canny(cv2.GaussianBlur(template_gray, (3, 3), 0), 30, 100)
            raw_result = cv2.matchTemplate(search_gray, template_gray, cv2.TM_CCOEFF_NORMED)
            edge_result = cv2.matchTemplate(search_edge, template_edge, cv2.TM_CCOEFF_NORMED)
            _, raw_score, _, _ = cv2.minMaxLoc(raw_result)
            _, edge_score, _, _ = cv2.minMaxLoc(edge_result)
            best_score = max(best_score, (0.45 * float(raw_score)) + (0.55 * float(edge_score)))
        return max(0.0, best_score)

    def _output_region_bonus(
        self,
        *,
        tab: str,
        output_name: str,
        card: Image.Image,
        output_templates: dict[str, Image.Image],
    ) -> tuple[float, str]:
        if tab != "smelt":
            return 0.0, ""
        template = self._lookup_template(output_templates, output_name)
        if template is None:
            return 0.0, ""
        box_scores = [
            self._template_presence_score(template=template, search=card.crop(search_box))
            for search_box in self._output_candidate_boxes(card)
        ]
        if not box_scores:
            return 0.0, ""
        return _SMELT_OUTPUT_REGION_WEIGHT * max(box_scores), "output_region_match"

    @staticmethod
    def _expected_input_names(*, tab: str, output_name: str) -> tuple[str, ...]:
        if tab == "smelt":
            data = bars_data.get_bar(output_name) or {}
        else:
            data = items_data.get_item(output_name) or {}
        inputs = tuple(str(name).strip() for name in (data.get("inputs") or {}).keys() if str(name).strip())
        return inputs

    def _input_identity_bonus(
        self,
        *,
        tab: str,
        output_name: str,
        card: Image.Image,
        input_templates: dict[str, Image.Image],
    ) -> tuple[float, str]:
        if not input_templates:
            return 0.0, ""
        expected_inputs = self._expected_input_names(tab=tab, output_name=output_name)
        if tab == "craft" and len(expected_inputs) < 2:
            return 0.0, ""
        if tab == "craft":
            expected_inputs = expected_inputs[:2]
        if tab == "smelt":
            expected_inputs = expected_inputs[:1]
        resolved_templates: list[Image.Image] = []
        for input_name in expected_inputs:
            template = self._lookup_template(input_templates, input_name)
            if template is None:
                return 0.0, ""
            resolved_templates.append(template)
        if not resolved_templates:
            return 0.0, ""
        box_scores: list[float] = []
        for search_box in self._input_search_boxes(card, tab=tab):
            search = card.crop(search_box)
            scores = [self._template_presence_score(template=template, search=search) for template in resolved_templates]
            if tab == "craft" and len(scores) < 2:
                continue
            box_scores.append(float(sum(scores) / len(scores)))
        if not box_scores:
            return 0.0, ""
        return _INPUT_SIGNAL_WEIGHT * max(box_scores), "input_template_match"

    @classmethod
    def _icon_similarity(cls, template: Image.Image, target: Image.Image) -> float:
        template_work = cls._extract_template_icon(template).resize((56, 56)).convert("RGB")
        target_icon = cls._trim_foreground_icon(target, prefer_center_x=0.5, prefer_center_y=0.5)
        target_work = (target_icon if target_icon is not None else target).resize((56, 56)).convert("RGB")
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
        tab: str,
        card: Image.Image,
        templates: dict[str, Image.Image],
        inventory_counts: dict[str, int],
        output_quantity: int | None,
        input_templates: dict[str, Image.Image] | None = None,
    ) -> tuple[str, str]:
        target_icons = self._candidate_output_icons(card)
        if not target_icons:
            raise ValueError("output_icon_not_found")
        scored = []
        for name, template in templates.items():
            input_bonus, input_backend = self._input_identity_bonus(
                tab=tab,
                output_name=name,
                card=card,
                input_templates=input_templates or {},
            )
            output_region_bonus, output_region_backend = self._output_region_bonus(
                tab=tab,
                output_name=name,
                card=card,
                output_templates=templates,
            )
            score = (
                max(self._icon_similarity(template, target_icon) for target_icon in target_icons)
                + self._quantity_match_bonus(name, quantity=output_quantity, inventory_counts=inventory_counts)
                + input_bonus
                + output_region_bonus
            )
            backend_parts = [part for part in (input_backend, output_region_backend) if part]
            scored.append((name, score, "+".join(backend_parts)))
        scored.sort(key=lambda item: item[1], reverse=True)
        if not scored:
            raise ValueError("icon_templates_missing")
        top_name, top_score, top_input_backend = scored[0]
        next_score = scored[1][1] if len(scored) > 1 else 0.0
        if top_score < 0.80 or (top_score - next_score) < 0.018:
            raise ValueError(f"ambiguous_output_match:{top_name}:{top_score:.4f}:{next_score:.4f}")
        backend = "icon_template_match"
        if top_input_backend:
            backend = f"{backend}+{top_input_backend}"
        return top_name, backend

    def _resolve_active_state(self, *, tab: str, card: Image.Image) -> tuple[bool, str | None, str]:
        timer_text, timer_backend = self._read_timer_text(card)
        normalized = self._normalize_timer_text(timer_text)
        fill_fraction = self._progress_fill_fraction(card)
        cancel_signal = self._cancel_button_signal(card)
        timer_signal = self._timer_region_signal(card)
        if normalized and _TIMER_TEXT_RE.search(normalized):
            return True, timer_text, timer_backend or "timer_text"
        if fill_fraction >= 0.05:
            if normalized and normalized != "OFF" and (_TIMER_TEXT_RE.search(normalized) or normalized.endswith("S")):
                return True, timer_text, timer_backend
            return True, timer_text, "progress_fill_signal"
        if tab == "craft" and cancel_signal and (timer_signal or fill_fraction >= 0.008):
            return True, timer_text if normalized and normalized != "OFF" else None, "cancel_button_signal+timer_region_signal"
        if normalized == "OFF":
            if not cancel_signal and not timer_signal and fill_fraction < 0.01:
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
        input_templates: dict[str, Image.Image] | None = None,
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
            tab=tab,
            card=card,
            templates=templates,
            inventory_counts=inventory_counts,
            output_quantity=output_quantity,
            input_templates=input_templates,
        )
        active, timer_text, state_backend = self._resolve_active_state(tab=tab, card=card)
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

    def _scroll_delay_seconds(self) -> float:
        configured_delay = getattr(getattr(getattr(self.actions, "config", None), "actions", None), "scroll_delay_seconds", None)
        try:
            return max(_PRODUCTION_SCROLL_DELAY_SECONDS, float(configured_delay))
        except (TypeError, ValueError):
            return _PRODUCTION_SCROLL_DELAY_SECONDS

    @staticmethod
    def _top_anchor_frame(card: Image.Image | None) -> Image.Image | None:
        if card is None:
            return None
        width, height = card.size
        return card.crop(
            (
                int(round(width * 0.06)),
                int(round(height * 0.05)),
                int(round(width * 0.92)),
                int(round(height * 0.47)),
            )
        )

    def _scroll_to_lower_cards(self, *, top_anchor: Image.Image | None) -> None:
        point = self._wheel_point()
        if not self.actions.scroll_client_wheel(point, -120, delay=self._scroll_delay_seconds()):
            raise ValueError("production_scroll_down_failed")
        current = self._capture_rect("PRODUCTION_CARD1")
        if self._image_mean_abs_diff(self._top_anchor_frame(top_anchor), self._top_anchor_frame(current)) <= 1.0:
            raise ValueError("production_scroll_down_not_observed")

    def _scroll_to_top_view(self) -> Image.Image | None:
        point = self._wheel_point()
        previous = None
        best_candidate = None
        best_diff = 1e9
        for _ in range(8):
            if not self.actions.scroll_client_wheel(point, 120, delay=self._scroll_delay_seconds()):
                raise ValueError("production_scroll_up_failed")
            current = self._capture_rect("PRODUCTION_CARD1")
            if current is None:
                raise ValueError("missing_rect:PRODUCTION_CARD1")
            current_anchor = self._top_anchor_frame(current)
            status = self._card_status(current)
            if previous is not None:
                diff = self._image_mean_abs_diff(previous, current_anchor)
                if status == "card" and diff < best_diff:
                    best_diff = diff
                    best_candidate = current
                if diff <= 1.0 and status == "card":
                    return current
            previous = current_anchor
        if best_candidate is not None and best_diff <= 1.3 and self._extract_output_icon(best_candidate) is not None:
            return best_candidate
        raise ValueError("production_top_latch_failed")

    def _scroll_back_to_top(self, *, top_anchor: Image.Image | None) -> None:
        if top_anchor is None:
            return
        point = self._wheel_point()
        top_anchor_frame = self._top_anchor_frame(top_anchor)
        best_diff = 1e9
        best_current = None
        for _ in range(8):
            current = self._capture_rect("PRODUCTION_CARD1")
            current_frame = self._top_anchor_frame(current)
            diff = self._image_mean_abs_diff(top_anchor_frame, current_frame)
            if current is not None and diff < best_diff:
                best_diff = diff
                best_current = current
            if diff <= 1.0:
                return
            if not self.actions.scroll_client_wheel(point, 120, delay=self._scroll_delay_seconds()):
                break
        current = self._capture_rect("PRODUCTION_CARD1")
        current_diff = self._image_mean_abs_diff(top_anchor_frame, self._top_anchor_frame(current))
        if current_diff <= 1.0:
            return
        if best_current is not None and best_diff <= 1.2 and self._card_status(best_current) == "card":
            return
        if current_diff > 1.0:
            raise ValueError("production_scroll_up_failed")

    def read_cards(
        self,
        *,
        tab: str,
        open_tab,
        templates: dict[str, Image.Image],
        inventory_counts: dict[str, int],
        input_templates: dict[str, Image.Image] | None = None,
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
            self._read_card(slot_index=1, tab=tab, rect_key="PRODUCTION_CARD1", templates=templates, inventory_counts=inventory_counts, input_templates=input_templates),
            self._read_card(slot_index=2, tab=tab, rect_key="PRODUCTION_CARD2", templates=templates, inventory_counts=inventory_counts, input_templates=input_templates),
        ]
        self._scroll_to_lower_cards(top_anchor=top_anchor)
        try:
            cards.extend(
                [
                    self._read_card(slot_index=3, tab=tab, rect_key="PRODUCTION_CARD3", templates=templates, inventory_counts=inventory_counts, input_templates=input_templates),
                    self._read_card(slot_index=4, tab=tab, rect_key="PRODUCTION_CARD4", templates=templates, inventory_counts=inventory_counts, input_templates=input_templates),
                ]
            )
        finally:
            self._scroll_back_to_top(top_anchor=top_anchor)
        return cards
