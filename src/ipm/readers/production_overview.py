from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageStat

import bars_data
import items_data
from .. import perception as perception_backend
from ..domain_data import RESOURCE_ROW_NAMES, normalize_resource_row_name
from ..state import ProductionOverviewCardState
from .common import parse_compact_number

_TIMER_TEXT_RE = re.compile(r"[0-9].*[SMH]|[0-9]+:[0-9]+", re.IGNORECASE)
_CLOCK_TEXT_RE = re.compile(r"\d{1,2}:\d{2}(?::\d{2})?(AM|PM)\b", re.IGNORECASE)
_QUANTITY_TEXT_RE = re.compile(
    r"(\d[\d.,]*[KMBT]\b)|(\d[\d.,]*/\d[\d.,]*(?:[KMBT])?)",
    re.IGNORECASE,
)
_TIMER_PROMPT = "Read only the visible timer text or OFF. Return only the timer text."
_COUNT_PROMPT = "Read only the visible output quantity. Keep suffixes like K or M if present."
_INPUT_COUNT_PROMPT = "Read only the visible left input quantity text like 404/1.00K. Return only that quantity text."
_TOOLTIP_PROMPT = "Read only the visible tooltip/item name near the probed icon. Return only the name."
_SMELT_RECIPE_PANEL_PROMPT = "Read visible text from the SMELT RECIPES popup. Return only the visible recipe names and times."
_PRODUCTION_SCROLL_DELAY_SECONDS = 0.35
_INPUT_SIGNAL_WEIGHT = 0.35
_SMELT_OUTPUT_REGION_WEIGHT = 0.20
_SMELT_INPUT_QTY_WEIGHT = 0.18
_ACTIVE_FILL_HINT_MIN = 0.008
_ACTIVE_FILL_CONFIDENT_MIN = 0.05
_PRODUCTION_TOP_SCROLL_MAX_ATTEMPTS = 5
_TOP_ANCHOR_STABLE_DIFF_MAX = 1.0
_TOP_ANCHOR_BEST_DIFF_MAX = 1.2
_TOP_ANCHOR_HEADER_EDGE_MIN = 0.020
_TOP_ANCHOR_DYNAMIC_RANGE_MIN = 90.0
_TOP_ANCHOR_CARD_BAND_EDGE_MIN = 0.030
_TOP_ANCHOR_CARD_BAND_MEAN_MIN = 40.0
_LOCALIZED_TARGET_DIFF_MIN = 26.0
_LOCALIZED_TARGET_MIN_AREA = 18
_LOCALIZED_TARGET_EXPAND_PX = 2
_CANCEL_TARGET_DARK_DELTA = 18
_ARROW_TARGET_DIFF_MIN = 12.0
_ARROW_TARGET_MIN_AREA = 4
_ARROW_SEARCH_Y_TOP = 0.12
_ARROW_SEARCH_Y_BOTTOM = 0.58
_SMELT_ARROW_SEARCH_Y_TOP = 0.00
_SMELT_ARROW_SEARCH_Y_BOTTOM = 0.50
_ARROW_SHIFT_Y_SCALE = 0.35
_RECIPE_BUTTON_REL_X = 0.375
_RECIPE_BUTTON_REL_Y = 0.807
_SMELT_ORE_SEARCH_SHIFT_X = 0.14
_SMELT_ORE_SEARCH_SHIFT_Y = -0.04
_SMELT_ORE_SEARCH_BOTTOM_TRIM = 0.28
_SMELT_ICON_CENTER_X_SEPARATION = 0.48
_SMELT_ICON_ESTIMATED_BOX_WIDTH = 0.23
_SMELT_ICON_ESTIMATED_BOX_HEIGHT = 0.16
_TOOLTIP_PROBE_OFFSETS = (
    ("center", (0.00, 0.00)),
    ("upper_left", (-0.15, -0.15)),
    ("lower_right", (0.15, 0.15)),
)
_TOOLTIP_SEARCH_LEFT_FROM_ICON = 0.35
_TOOLTIP_SEARCH_RIGHT_EXTENT = 4.30
_TOOLTIP_SEARCH_TOP_FROM_CENTER = 1.10
_TOOLTIP_SEARCH_BOTTOM_FROM_CENTER = 1.00
_SMELT_RECIPE_SCROLL_DELTA = -120
_SMELT_RECIPE_SLOT_LAYOUT_PAGE0 = (
    ((49, 207, 96, 260), (105, 209, 165, 262)),
    ((184, 207, 231, 260), (240, 209, 300, 262)),
    ((49, 392, 96, 445), (105, 394, 165, 447)),
    ((184, 392, 231, 445), (240, 394, 300, 447)),
    ((49, 577, 96, 630), (105, 579, 165, 632)),
    ((184, 577, 231, 630), (240, 579, 300, 632)),
)
_SMELT_RECIPE_SLOT_LAYOUT_SCROLLED = (
    ((49, 207, 96, 260), (105, 209, 165, 262)),
    ((184, 207, 231, 260), (240, 209, 300, 262)),
    ((49, 392, 96, 445), (105, 394, 165, 447)),
    ((184, 392, 231, 445), (240, 394, 300, 447)),
)


@dataclass(slots=True, frozen=True)
class _SmeltRecipePopupEntry:
    output_name: str
    ore_icon_box: tuple[int, int, int, int]
    output_icon_box: tuple[int, int, int, int]


@dataclass(slots=True, frozen=True)
class _TooltipIconRegion:
    kind: str
    box: tuple[int, int, int, int]


@dataclass(slots=True, frozen=True)
class _TooltipProbePoint:
    point_id: str
    point: tuple[int, int]


@dataclass(slots=True, frozen=True)
class _TooltipCropCandidate:
    crop_id: str
    box: tuple[int, int, int, int]


@dataclass(slots=True, frozen=True)
class _ProductionCardLayout:
    recipe_button_box: tuple[int, int, int, int]
    input_icon_box: tuple[int, int, int, int]
    output_icon_box: tuple[int, int, int, int]
    progress_bar_box: tuple[int, int, int, int]
    cancel_box: tuple[int, int, int, int]
    extra_icon_regions: tuple[_TooltipIconRegion, ...] = ()
    arrow_search_box: tuple[int, int, int, int] | None = None
    localized_arrow_box: tuple[int, int, int, int] | None = None


@dataclass(slots=True, frozen=True)
class _TooltipProbeMarker:
    point: tuple[int, int]
    marker_label: str
    color: str = "#ffd400"


@dataclass(slots=True, frozen=True)
class _LocalizedTooltipTarget:
    kind: str
    search_box: tuple[int, int, int, int]
    localized_box: tuple[int, int, int, int] | None


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

    def _capture_screen(self) -> Image.Image | None:
        capture_screen = getattr(self.capture, "capture_screen", None)
        if not callable(capture_screen):
            return None
        return capture_screen()

    @staticmethod
    def _image_mean_abs_diff(previous: Image.Image | None, current: Image.Image | None) -> float:
        if previous is None or current is None:
            return 1e9
        diff = ImageChops.difference(previous.convert("L"), current.convert("L"))
        return float(ImageStat.Stat(diff).mean[0])

    @staticmethod
    def _normalize_timer_text(text: str | None) -> str:
        normalized = re.sub(r"\s+", "", str(text or "").upper())
        if not normalized:
            return ""
        if "NORECIPESELECTED" in normalized:
            return "NORECIPESELECTED"
        if "OFF" in normalized and len(normalized) <= 6:
            return "OFF"
        translated = (
            normalized.replace("|", "1")
            .replace("I", "1")
            .replace("L", "1")
            .replace("O", "0")
            .replace(";", ":")
        )
        return translated

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
        stats = cls._region_signal_stats(card.crop(cls._timer_text_box(card.size)))
        return stats["dynamic_range"] >= 240.0 and (
            stats["cyan_fraction"] >= 0.10
            or (stats["edge_fraction"] >= 0.068 and stats["bright_fraction"] >= 0.03)
        )

    @classmethod
    def _timer_text_presence_signal(cls, card: Image.Image) -> bool:
        stats = cls._region_signal_stats(card.crop(cls._timer_text_box(card.size)))
        return (
            stats["dynamic_range"] >= 120.0
            and stats["edge_fraction"] >= 0.028
            and stats["bright_fraction"] >= 0.01
        )

    @staticmethod
    def _timer_text_box(card_size: tuple[int, int]) -> tuple[int, int, int, int]:
        width, height = card_size
        return (
            int(round(width * 0.27)),
            int(round(height * 0.56)),
            int(round(width * 0.58)),
            int(round(height * 0.66)),
        )

    def _read_timer_text(self, card: Image.Image) -> tuple[str | None, str]:
        crop = card.crop(self._timer_text_box(card.size))
        result = self.perception.read_text(crop, prompt=_TIMER_PROMPT, mode="generic")
        value = str(getattr(result, "value", "") or "").strip()
        return (value or None), str(getattr(result, "backend", "") or "")

    @staticmethod
    def _valid_timer_like_text(normalized: str) -> bool:
        if not normalized:
            return False
        if _CLOCK_TEXT_RE.fullmatch(normalized):
            return False
        if _QUANTITY_TEXT_RE.search(normalized):
            return False
        return bool(
            re.fullmatch(r"\d{1,2}:\d{2}", normalized)
            or re.fullmatch(r"\d{1,2}:\d{2}:\d{2}", normalized)
            or re.fullmatch(r"\d+[SMH]", normalized)
        )

    def _read_output_quantity(self, card: Image.Image) -> tuple[int | None, str]:
        crop = card.crop((140, 80, 230, 135))
        result = self.perception.read_text(crop, prompt=_COUNT_PROMPT, mode="ore_qty")
        return parse_compact_number(getattr(result, "value", "")), str(getattr(result, "backend", "") or "")

    def _read_input_available_quantity(self, card: Image.Image) -> tuple[int | None, str]:
        crop = card.crop((5, 85, 120, 125))
        result = self.perception.read_text(crop, prompt=_INPUT_COUNT_PROMPT, mode="generic")
        raw_value = str(getattr(result, "value", "") or "").strip()
        left_value = raw_value.split("/", 1)[0].strip()
        return parse_compact_number(left_value), str(getattr(result, "backend", "") or "")

    @classmethod
    def _popup_recipe_entries(cls) -> tuple[_SmeltRecipePopupEntry, ...]:
        names = tuple(bars_data.list_bars())
        entries: list[_SmeltRecipePopupEntry] = []
        for output_name, (ore_icon_box, output_icon_box) in zip(names, _SMELT_RECIPE_SLOT_LAYOUT_PAGE0):
            entries.append(
                _SmeltRecipePopupEntry(
                    output_name=output_name,
                    ore_icon_box=ore_icon_box,
                    output_icon_box=output_icon_box,
                )
            )
        return tuple(entries)

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
    def _tooltip_icon_regions(*, tab: str, card_size: tuple[int, int]) -> tuple[_TooltipIconRegion, ...]:
        width, height = card_size

        def _box(left: float, top: float, right: float, bottom: float) -> tuple[int, int, int, int]:
            x1 = max(0, min(width - 1, int(round(width * left))))
            y1 = max(0, min(height - 1, int(round(height * top))))
            x2 = max(x1 + 1, min(width, int(round(width * right))))
            y2 = max(y1 + 1, min(height, int(round(height * bottom))))
            return (x1, y1, x2, y2)

        if tab == "smelt":
            return (
                _TooltipIconRegion(kind="bar", box=_box(0.50, 0.24, 0.76, 0.46)),
                _TooltipIconRegion(kind="ore", box=_box(0.06, 0.24, 0.29, 0.46)),
            )
        return (
            _TooltipIconRegion(kind="output", box=_box(0.56, 0.25, 0.81, 0.49)),
            _TooltipIconRegion(kind="bar", box=_box(0.14, 0.31, 0.39, 0.56)),
            _TooltipIconRegion(kind="ore", box=_box(0.18, 0.06, 0.40, 0.24)),
        )

    @staticmethod
    def _clip_box(
        box: tuple[int, int, int, int],
        *,
        card_size: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        card_width, card_height = card_size
        left, top, right, bottom = box
        clipped_left = max(0, min(card_width - 1, int(left)))
        clipped_top = max(0, min(card_height - 1, int(top)))
        clipped_right = max(clipped_left + 1, min(card_width, int(right)))
        clipped_bottom = max(clipped_top + 1, min(card_height, int(bottom)))
        return (clipped_left, clipped_top, clipped_right, clipped_bottom)

    @classmethod
    def _detect_recipe_button_box(cls, card: Image.Image) -> tuple[int, int, int, int] | None:
        width, height = card.size
        search_box = (
            int(round(width * 0.10)),
            int(round(height * 0.68)),
            int(round(width * 0.90)),
            int(round(height * 0.98)),
        )
        crop = card.crop(search_box)
        arr = np.asarray(crop.convert("RGB"), dtype=np.uint8)
        if arr.size == 0:
            return None
        mask = (
            (arr[..., 1] >= 150)
            & (arr[..., 2] >= 150)
            & (arr[..., 0] <= 120)
        ).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
        num, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
        best: tuple[int, tuple[int, int, int, int]] | None = None
        for index in range(1, num):
            x, y, w, h, area = (int(value) for value in stats[index])
            if area < 80:
                continue
            if w < int(round(width * 0.18)) or h < int(round(height * 0.06)):
                continue
            if w > int(round(width * 0.55)) or h > int(round(height * 0.22)):
                continue
            candidate = (area, (search_box[0] + x, search_box[1] + y, search_box[0] + x + w, search_box[1] + y + h))
            if best is None or candidate[0] > best[0]:
                best = candidate
        return best[1] if best is not None else None

    @classmethod
    def _expected_arrow_search_box(
        cls,
        *,
        tab: str,
        input_box: tuple[int, int, int, int],
        output_box: tuple[int, int, int, int],
        card_size: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        card_width, card_height = card_size
        input_height = max(1, input_box[3] - input_box[1])
        output_height = max(1, output_box[3] - output_box[1])
        avg_height = (input_height + output_height) / 2.0
        if tab == "smelt":
            top_scale = _SMELT_ARROW_SEARCH_Y_TOP
            bottom_scale = _SMELT_ARROW_SEARCH_Y_BOTTOM
        else:
            top_scale = _ARROW_SEARCH_Y_TOP
            bottom_scale = _ARROW_SEARCH_Y_BOTTOM
        top = min(input_box[1], output_box[1]) + int(round(avg_height * top_scale))
        bottom = min(input_box[1], output_box[1]) + int(round(avg_height * bottom_scale))
        return cls._clip_box(
            (
                int(round(input_box[2] - max(4, (input_box[2] - input_box[0]) * 0.06))),
                top,
                int(round(output_box[0] + max(4, (output_box[2] - output_box[0]) * 0.06))),
                bottom,
            ),
            card_size=(card_width, card_height),
        )

    @classmethod
    def _localize_arrow_box(
        cls,
        card: Image.Image,
        *,
        search_box: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int] | None:
        crop = card.crop(search_box).convert("RGB")
        arr = np.asarray(crop, dtype=np.uint8)
        if arr.size == 0:
            return None
        height, width = arr.shape[:2]
        if width < 2 or height < 2:
            return None

        top = arr[0, :, :]
        bottom = arr[-1, :, :]
        left = arr[:, 0, :]
        right = arr[:, -1, :]
        border = np.concatenate((top, bottom, left, right), axis=0).astype(np.float32)
        bg_rgb = np.median(border, axis=0)
        diff = np.linalg.norm(arr.astype(np.float32) - bg_rgb[None, None, :], axis=2)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

        mask = (
            (diff >= _ARROW_TARGET_DIFF_MIN)
            & (gray >= 60)
            & (gray <= 230)
            & (hsv[..., 1] <= 160)
        ).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
        count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

        expected_center = (width / 2.0, height * 0.42)
        best: tuple[float, tuple[int, int, int, int]] | None = None
        for index in range(1, count):
            x, y, comp_w, comp_h, area = (int(value) for value in stats[index])
            if area < _ARROW_TARGET_MIN_AREA:
                continue
            if comp_w < 3 or comp_h < 3:
                continue
            if comp_w > int(round(width * 0.55)) or comp_h > int(round(height * 0.75)):
                continue
            fill_ratio = float(area) / float(max(1, comp_w * comp_h))
            if fill_ratio < 0.08:
                continue
            aspect_ratio = float(comp_w) / float(max(1, comp_h))
            if aspect_ratio < 0.22 or aspect_ratio > 2.6:
                continue
            cx, cy = (float(value) for value in centroids[index])
            center_distance = float(np.hypot(cx - expected_center[0], cy - expected_center[1]))
            score = float(area) + (fill_ratio * 24.0) - (center_distance * 1.1)
            candidate = (
                search_box[0] + max(0, x - 1),
                search_box[1] + max(0, y - 1),
                search_box[0] + min(width, x + comp_w + 1),
                search_box[1] + min(height, y + comp_h + 1),
            )
            if best is None or score > best[0]:
                best = (score, candidate)
        if best is None:
            return None
        return cls._clip_box(best[1], card_size=card.size)

    @classmethod
    def _shift_box(
        cls,
        box: tuple[int, int, int, int],
        *,
        dx: float,
        dy: float,
        card_size: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        return cls._clip_box(
            (
                int(round(box[0] + dx)),
                int(round(box[1] + dy)),
                int(round(box[2] + dx)),
                int(round(box[3] + dy)),
            ),
            card_size=card_size,
        )

    @classmethod
    def _derive_card_layout(
        cls,
        *,
        card: Image.Image,
        tab: str,
    ) -> _ProductionCardLayout:
        recipe_button_box = cls._detect_recipe_button_box(card)
        if recipe_button_box is None:
            raise ValueError("tooltip_card_anchor_unverified")
        width, height = card.size

        def _box(left: float, top: float, right: float, bottom: float) -> tuple[int, int, int, int]:
            return cls._clip_box(
                (
                    int(round(width * left)),
                    int(round(height * top)),
                    int(round(width * right)),
                    int(round(height * bottom)),
                ),
                card_size=(width, height),
            )

        progress_bar_box = _box(0.13, 0.49, 0.85, 0.62)
        cancel_box = _box(0.77, 0.49, 0.94, 0.63)
        if tab == "smelt":
            input_icon_box = _box(0.06, 0.24, 0.29, 0.46)
            output_icon_box = _box(0.50, 0.24, 0.76, 0.46)
            extra_regions: tuple[_TooltipIconRegion, ...] = ()
        else:
            input_icon_box = _box(0.14, 0.31, 0.39, 0.56)
            output_icon_box = _box(0.56, 0.25, 0.81, 0.49)
            extra_regions = (_TooltipIconRegion(kind="ore", box=_box(0.18, 0.06, 0.40, 0.24)),)
        arrow_search_box = cls._expected_arrow_search_box(
            tab=tab,
            input_box=input_icon_box,
            output_box=output_icon_box,
            card_size=(width, height),
        )
        localized_arrow_box = cls._localize_arrow_box(card, search_box=arrow_search_box)
        return _ProductionCardLayout(
            recipe_button_box=recipe_button_box,
            input_icon_box=input_icon_box,
            output_icon_box=output_icon_box,
            progress_bar_box=progress_bar_box,
            cancel_box=cancel_box,
            extra_icon_regions=extra_regions,
            arrow_search_box=arrow_search_box,
            localized_arrow_box=localized_arrow_box,
        )

    @classmethod
    def _tooltip_icon_regions_from_layout(
        cls,
        *,
        tab: str,
        layout: _ProductionCardLayout,
    ) -> tuple[_TooltipIconRegion, ...]:
        if tab == "smelt":
            return (
                _TooltipIconRegion(kind="bar", box=layout.output_icon_box),
                _TooltipIconRegion(kind="ore", box=layout.input_icon_box),
            )
        return (
            _TooltipIconRegion(kind="output", box=layout.output_icon_box),
            _TooltipIconRegion(kind="bar", box=layout.input_icon_box),
            *layout.extra_icon_regions,
        )

    @staticmethod
    def _box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @classmethod
    def _localize_target_box(
        cls,
        card: Image.Image,
        *,
        search_box: tuple[int, int, int, int],
        kind: str,
    ) -> tuple[int, int, int, int] | None:
        crop = card.crop(search_box).convert("RGB")
        arr = np.asarray(crop, dtype=np.uint8)
        if arr.size == 0:
            return None
        height, width = arr.shape[:2]
        if width < 2 or height < 2:
            return None

        top = arr[0, :, :]
        bottom = arr[-1, :, :]
        left = arr[:, 0, :]
        right = arr[:, -1, :]
        border = np.concatenate((top, bottom, left, right), axis=0).astype(np.float32)
        bg_rgb = np.median(border, axis=0)
        diff = np.linalg.norm(arr.astype(np.float32) - bg_rgb[None, None, :], axis=2)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

        if kind == "cancel":
            bg_gray = float(np.median(cv2.cvtColor(border.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2GRAY)))
            mask = ((bg_gray - gray.astype(np.float32)) >= _CANCEL_TARGET_DARK_DELTA) | (diff >= (_LOCALIZED_TARGET_DIFF_MIN + 8.0))
            min_area = max(_LOCALIZED_TARGET_MIN_AREA, int(round(width * height * 0.03)))
            min_dim = max(6, int(round(min(width, height) * 0.18)))
        else:
            hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
            mask = (diff >= _LOCALIZED_TARGET_DIFF_MIN) & ((gray >= 40) | (hsv[..., 1] >= 25))
            min_area = max(_LOCALIZED_TARGET_MIN_AREA, int(round(width * height * 0.045)))
            min_dim = max(8, int(round(min(width, height) * 0.20)))

        mask_u8 = mask.astype(np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
        count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, 8)

        if kind == "cancel":
            expected_center = (width / 2.0, height / 2.0)
        else:
            expected_center = (width / 2.0, height * 0.42)
        best: tuple[float, tuple[int, int, int, int]] | None = None
        for index in range(1, count):
            x, y, comp_w, comp_h, area = (int(value) for value in stats[index])
            if area < min_area or comp_w < min_dim or comp_h < min_dim:
                continue
            fill_ratio = float(area) / float(max(1, comp_w * comp_h))
            if fill_ratio < 0.18:
                continue
            cx, cy = (float(value) for value in centroids[index])
            center_distance = float(np.hypot(cx - expected_center[0], cy - expected_center[1]))
            score = float(area) + (fill_ratio * 30.0) - (center_distance * 1.2)
            candidate = (
                search_box[0] + max(0, x - _LOCALIZED_TARGET_EXPAND_PX),
                search_box[1] + max(0, y - _LOCALIZED_TARGET_EXPAND_PX),
                search_box[0] + min(width, x + comp_w + _LOCALIZED_TARGET_EXPAND_PX),
                search_box[1] + min(height, y + comp_h + _LOCALIZED_TARGET_EXPAND_PX),
            )
            if best is None or score > best[0]:
                best = (score, candidate)
        if best is None:
            return None
        return cls._clip_box(best[1], card_size=card.size)

    @classmethod
    def _localized_tooltip_targets_from_layout(
        cls,
        *,
        card: Image.Image,
        tab: str,
        layout: _ProductionCardLayout,
    ) -> tuple[_LocalizedTooltipTarget, ...]:
        regions = cls._tooltip_icon_regions_from_layout(tab=tab, layout=layout)
        if layout.localized_arrow_box is not None and layout.arrow_search_box is not None:
            rough_arrow_center = cls._box_center(layout.arrow_search_box)
            localized_arrow_center = cls._box_center(layout.localized_arrow_box)
            dx = localized_arrow_center[0] - rough_arrow_center[0]
            dy = (localized_arrow_center[1] - rough_arrow_center[1]) * _ARROW_SHIFT_Y_SCALE
            shifted_regions: list[_TooltipIconRegion] = []
            for region in regions:
                if region.kind == "ore" and tab != "smelt":
                    shifted_regions.append(region)
                    continue
                shifted_regions.append(
                    _TooltipIconRegion(
                        kind=region.kind,
                        box=cls._shift_box(region.box, dx=dx, dy=dy, card_size=card.size),
                    )
                )
            regions = tuple(shifted_regions)

        if tab == "smelt":
            refined_regions: list[_TooltipIconRegion] = []
            for region in regions:
                if region.kind != "ore":
                    refined_regions.append(region)
                    continue
                x1, y1, x2, y2 = region.box
                box_width = max(1, x2 - x1)
                box_height = max(1, y2 - y1)
                shifted = cls._shift_box(
                    region.box,
                    dx=box_width * _SMELT_ORE_SEARCH_SHIFT_X,
                    dy=box_height * _SMELT_ORE_SEARCH_SHIFT_Y,
                    card_size=card.size,
                )
                sx1, sy1, sx2, sy2 = shifted
                trimmed = cls._clip_box(
                    (
                        sx1,
                        sy1,
                        sx2,
                        sy2 - int(round((sy2 - sy1) * _SMELT_ORE_SEARCH_BOTTOM_TRIM)),
                    ),
                    card_size=card.size,
                )
                refined_regions.append(_TooltipIconRegion(kind=region.kind, box=trimmed))
            regions = tuple(refined_regions)

        targets = [
            _LocalizedTooltipTarget(
                kind=region.kind,
                search_box=region.box,
                localized_box=cls._localize_target_box(card, search_box=region.box, kind=region.kind),
            )
            for region in regions
        ]

        if tab == "smelt":
            bar_target = next((target for target in targets if target.kind == "bar" and target.localized_box is not None), None)
            if bar_target is not None:
                estimated_ore_search_box = cls._estimate_smelt_ore_search_box_from_bar(
                    bar_box=bar_target.localized_box,
                    card_size=card.size,
                )
                targets = [
                    _LocalizedTooltipTarget(
                        kind=target.kind,
                        search_box=estimated_ore_search_box,
                        localized_box=target.localized_box,
                    )
                    if target.kind == "ore" and target.localized_box is None
                    else target
                    for target in targets
                ]

        return tuple(targets)

    @classmethod
    def _estimate_smelt_ore_search_box_from_bar(
        cls,
        *,
        bar_box: tuple[int, int, int, int],
        card_size: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        card_width, card_height = card_size
        bar_center_x, bar_center_y = cls._box_center(bar_box)
        box_width = max(1, int(round(card_width * _SMELT_ICON_ESTIMATED_BOX_WIDTH)))
        box_height = max(1, int(round(card_height * _SMELT_ICON_ESTIMATED_BOX_HEIGHT)))
        ore_center_x = bar_center_x - (card_width * _SMELT_ICON_CENTER_X_SEPARATION)
        ore_center_y = bar_center_y
        return cls._clip_box(
            (
                int(round(ore_center_x - (box_width / 2.0))),
                int(round(ore_center_y - (box_height / 2.0))),
                int(round(ore_center_x + (box_width / 2.0))),
                int(round(ore_center_y + (box_height / 2.0))),
            ),
            card_size=(card_width, card_height),
        )

    @staticmethod
    def _tooltip_probe_points(icon_box: tuple[int, int, int, int]) -> tuple[tuple[int, int], ...]:
        return tuple(spec.point for spec in ProductionOverviewReader._tooltip_probe_point_specs(icon_box))

    @staticmethod
    def _tooltip_probe_box_for_target(
        *,
        tab: str,
        target: _LocalizedTooltipTarget,
    ) -> tuple[int, int, int, int] | None:
        if target.localized_box is not None:
            return target.localized_box
        if tab == "smelt" and target.kind == "ore":
            return target.search_box
        return None

    @staticmethod
    def _tooltip_probe_point_specs(icon_box: tuple[int, int, int, int]) -> tuple[_TooltipProbePoint, ...]:
        x1, y1, x2, y2 = icon_box
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        center_x = x1 + (width / 2.0)
        center_y = y1 + (height / 2.0)
        return tuple(
            _TooltipProbePoint(
                point_id=point_id,
                point=(
                    int(round(center_x + (width * x_offset))),
                    int(round(center_y + (height * y_offset))),
                ),
            )
            for point_id, (x_offset, y_offset) in _TOOLTIP_PROBE_OFFSETS
        )

    @staticmethod
    def _tooltip_crop_box(icon_box: tuple[int, int, int, int], *, card_size: tuple[int, int]) -> tuple[int, int, int, int]:
        return ProductionOverviewReader._tooltip_search_region(icon_box, card_size=card_size)

    @staticmethod
    def _tooltip_search_region(
        icon_box: tuple[int, int, int, int],
        *,
        card_size: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        card_width, card_height = card_size
        x1, y1, x2, y2 = icon_box
        icon_width = max(1, x2 - x1)
        icon_height = max(1, y2 - y1)
        center_y = y1 + (icon_height / 2.0)

        def _clip(left: int, top: int, right: int, bottom: int) -> tuple[int, int, int, int]:
            clipped_left = max(0, min(card_width - 1, int(left)))
            clipped_top = max(0, min(card_height - 1, int(top)))
            clipped_right = max(clipped_left + 1, min(card_width, int(right)))
            clipped_bottom = max(clipped_top + 1, min(card_height, int(bottom)))
            return (clipped_left, clipped_top, clipped_right, clipped_bottom)

        region_width = max(icon_width + 1, int(round(icon_width * (_TOOLTIP_SEARCH_LEFT_FROM_ICON + _TOOLTIP_SEARCH_RIGHT_EXTENT))))
        region_height = max(
            icon_height + 1,
            int(round(icon_height * (_TOOLTIP_SEARCH_TOP_FROM_CENTER + _TOOLTIP_SEARCH_BOTTOM_FROM_CENTER))),
        )
        left = int(round(x1 - (icon_width * _TOOLTIP_SEARCH_LEFT_FROM_ICON)))
        top = int(round(center_y - (icon_height * _TOOLTIP_SEARCH_TOP_FROM_CENTER)))
        right = left + region_width
        bottom = top + region_height

        if right > card_width:
            overflow = right - card_width
            left -= overflow
            right -= overflow
        if bottom > card_height:
            overflow = bottom - card_height
            top -= overflow
            bottom -= overflow

        return _clip(left, top, right, bottom)

    @staticmethod
    def _tooltip_crop_candidates(
        icon_box: tuple[int, int, int, int],
        *,
        card_size: tuple[int, int],
    ) -> tuple[_TooltipCropCandidate, ...]:
        return (
            _TooltipCropCandidate(
                crop_id="search_region",
                box=ProductionOverviewReader._tooltip_search_region(icon_box, card_size=card_size),
            ),
        )

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
    def _normalize_tooltip_text(text: str | None) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9 ]+", " ", str(text or "").upper())).strip()

    @classmethod
    def _known_tooltip_labels(cls) -> dict[str, str]:
        lookup: dict[str, str] = {}
        known_names = list(RESOURCE_ROW_NAMES) + list(bars_data.list_bars()) + list(items_data.list_items())
        for name in known_names:
            for alias in cls._material_aliases(name):
                normalized = cls._normalize_tooltip_text(alias)
                if normalized and normalized not in lookup:
                    lookup[normalized] = name
                compact = normalized.replace(" ", "")
                if compact and compact not in lookup:
                    lookup[compact] = name
        return lookup

    @classmethod
    def _extract_tooltip_label_matches(cls, text: str | None) -> tuple[str, ...]:
        normalized = cls._normalize_tooltip_text(text)
        if not normalized:
            return ()
        compact = normalized.replace(" ", "")
        lookup = cls._known_tooltip_labels()
        exact_match = lookup.get(normalized) or lookup.get(compact)
        if exact_match is not None:
            return (exact_match,)

        matched_lengths: dict[str, int] = {}
        for alias, canonical in lookup.items():
            if not alias:
                continue
            alias_compact = alias.replace(" ", "")
            contains = alias in normalized if " " in alias else alias_compact in compact
            if not contains:
                continue
            matched_lengths[canonical] = max(matched_lengths.get(canonical, 0), len(alias_compact))

        if not matched_lengths:
            return ()
        strongest = max(matched_lengths.values())
        matches = sorted(canonical for canonical, length in matched_lengths.items() if length == strongest)
        return tuple(matches)

    @classmethod
    def _match_tooltip_label(cls, text: str | None) -> str | None:
        matches = cls._extract_tooltip_label_matches(text)
        if len(matches) != 1:
            return None
        return matches[0]

    def _resolve_smelt_tooltip_output(self, *, tooltip_label: str, templates: dict[str, Image.Image]) -> str | None:
        if self._lookup_template(templates, tooltip_label) is not None:
            for alias in self._material_aliases(tooltip_label):
                if alias in templates:
                    return alias
        normalized_label = self._match_tooltip_label(tooltip_label)
        if normalized_label is None:
            return None
        for output_name in templates:
            expected_inputs = self._expected_input_names(tab="smelt", output_name=output_name)
            if len(expected_inputs) != 1:
                continue
            if normalized_label in self._material_aliases(expected_inputs[0]):
                return output_name
        return None

    def _probe_tooltip_identity(
        self,
        *,
        rect_key: str,
        tab: str,
        templates: dict[str, Image.Image],
    ) -> tuple[str, str]:
        rect = getattr(self.rects, "get", lambda _key: None)(rect_key)
        if rect is None:
            raise ValueError(f"missing_rect:{rect_key}")
        card_x, card_y, card_w, card_h = rect
        card = self._capture_rect(rect_key)
        if card is None:
            raise ValueError(f"missing_rect:{rect_key}")
        layout = self._derive_card_layout(card=card, tab=tab)
        for target in self._localized_tooltip_targets_from_layout(card=card, tab=tab, layout=layout):
            probe_box = self._tooltip_probe_box_for_target(tab=tab, target=target)
            if probe_box is None:
                continue
            for probe in self._tooltip_probe_point_specs(probe_box):
                point = probe.point
                global_point = (int(card_x + point[0]), int(card_y + point[1]))
                if not self.actions.click_client_point(global_point, delay=self._scroll_delay_seconds()):
                    raise ValueError("tooltip_probe_click_failed")
                frame = self._capture_screen()
                if frame is None:
                    raise ValueError("tooltip_probe_capture_unavailable")
                for crop_candidate in self._tooltip_crop_candidates(probe_box, card_size=(card_w, card_h)):
                    crop_box = crop_candidate.box
                    tooltip = frame.crop(
                        (
                            int(card_x + crop_box[0]),
                            int(card_y + crop_box[1]),
                            int(card_x + crop_box[2]),
                            int(card_y + crop_box[3]),
                        )
                    )
                    result = self.perception.read_text(tooltip, prompt=_TOOLTIP_PROMPT, mode="generic")
                    matched_labels = self._extract_tooltip_label_matches(getattr(result, "value", ""))
                    if len(matched_labels) != 1:
                        continue
                    matched_label = matched_labels[0]
                    if tab != "smelt":
                        return matched_label, f"tooltip_probe_{target.kind}_{getattr(result, 'backend', '') or 'generic'}"
                    output_name = self._resolve_smelt_tooltip_output(tooltip_label=matched_label, templates=templates)
                    if output_name is not None:
                        return output_name, f"tooltip_probe_{target.kind}_{getattr(result, 'backend', '') or 'generic'}"
        raise ValueError("tooltip_probe_no_valid_label")

    @staticmethod
    def _tooltip_probe_artifact_label(*, tab: str, rect_key: str, icon_kind: str, point_id: str, crop_id: str) -> str:
        return f"{tab}_{rect_key}_{icon_kind}_{point_id}_{crop_id}"

    @staticmethod
    def _draw_crosshair(draw: ImageDraw.ImageDraw, point: tuple[int, int], *, color: str) -> None:
        x, y = point
        draw.line((x - 8, y, x + 8, y), fill=color, width=2)
        draw.line((x, y - 8, x, y + 8), fill=color, width=2)

    @classmethod
    def _draw_probe_marker(
        cls,
        draw: ImageDraw.ImageDraw,
        point: tuple[int, int],
        *,
        color: str,
        marker_label: str,
    ) -> None:
        x, y = point
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline="#08121f", width=1)
        cls._draw_crosshair(draw, point, color=color)
        text_origin = (x + 10, max(4, y - 10))
        text_width = max(26, (len(marker_label) * 7) + 4)
        draw.rectangle(
            (text_origin[0] - 2, text_origin[1] - 2, text_origin[0] + text_width, text_origin[1] + 14),
            fill="#08121f",
        )
        draw.text(text_origin, marker_label, fill=color)

    @classmethod
    def _render_tooltip_probe_audit_overlay(
        cls,
        *,
        frame: Image.Image,
        card_rect: tuple[int, int, int, int],
        layout: _ProductionCardLayout,
        search_box: tuple[int, int, int, int],
        localized_box: tuple[int, int, int, int] | None,
        probe_point: tuple[int, int],
        tooltip_crop_box: tuple[int, int, int, int],
        label: str,
        probe_markers: tuple[_TooltipProbeMarker, ...] = (),
        localized_targets: tuple[_LocalizedTooltipTarget, ...] = (),
        localized_cancel_box: tuple[int, int, int, int] | None = None,
    ) -> Image.Image:
        annotated = frame.convert("RGB").copy()
        draw = ImageDraw.Draw(annotated)
        card_x, card_y, card_w, card_h = card_rect
        card_box = (card_x, card_y, card_x + card_w, card_y + card_h)
        search_rect = (
            card_x + search_box[0],
            card_y + search_box[1],
            card_x + search_box[2],
            card_y + search_box[3],
        )
        crop_rect = (
            card_x + tooltip_crop_box[0],
            card_y + tooltip_crop_box[1],
            card_x + tooltip_crop_box[2],
            card_y + tooltip_crop_box[3],
        )
        recipe_rect = (
            card_x + layout.recipe_button_box[0],
            card_y + layout.recipe_button_box[1],
            card_x + layout.recipe_button_box[2],
            card_y + layout.recipe_button_box[3],
        )
        progress_rect = (
            card_x + layout.progress_bar_box[0],
            card_y + layout.progress_bar_box[1],
            card_x + layout.progress_bar_box[2],
            card_y + layout.progress_bar_box[3],
        )
        cancel_rect = (
            card_x + layout.cancel_box[0],
            card_y + layout.cancel_box[1],
            card_x + layout.cancel_box[2],
            card_y + layout.cancel_box[3],
        )
        draw.rectangle(card_box, outline="#00e5ff", width=3)
        draw.rectangle(recipe_rect, outline="#00ffcc", width=3)
        if layout.arrow_search_box is not None:
            draw.rectangle(
                (
                    card_x + layout.arrow_search_box[0],
                    card_y + layout.arrow_search_box[1],
                    card_x + layout.arrow_search_box[2],
                    card_y + layout.arrow_search_box[3],
                ),
                outline="#c77dff",
                width=2,
            )
        if layout.localized_arrow_box is not None:
            draw.rectangle(
                (
                    card_x + layout.localized_arrow_box[0],
                    card_y + layout.localized_arrow_box[1],
                    card_x + layout.localized_arrow_box[2],
                    card_y + layout.localized_arrow_box[3],
                ),
                outline="#ff5cff",
                width=3,
            )
        draw.rectangle(progress_rect, outline="#7dff6f", width=2)
        draw.rectangle(cancel_rect, outline="#ff3b30", width=2)
        if localized_targets:
            for target in localized_targets:
                draw.rectangle(
                    (
                        card_x + target.search_box[0],
                        card_y + target.search_box[1],
                        card_x + target.search_box[2],
                        card_y + target.search_box[3],
                    ),
                    outline="#ffe066",
                    width=2,
                )
        else:
            draw.rectangle(
                (
                    card_x + layout.input_icon_box[0],
                    card_y + layout.input_icon_box[1],
                    card_x + layout.input_icon_box[2],
                    card_y + layout.input_icon_box[3],
                ),
                outline="#ffd400",
                width=2,
            )
            draw.rectangle(
                (
                    card_x + layout.output_icon_box[0],
                    card_y + layout.output_icon_box[1],
                    card_x + layout.output_icon_box[2],
                    card_y + layout.output_icon_box[3],
                ),
                outline="#ff9f0a",
                width=2,
            )
            for extra_region in layout.extra_icon_regions:
                draw.rectangle(
                    (
                        card_x + extra_region.box[0],
                        card_y + extra_region.box[1],
                        card_x + extra_region.box[2],
                        card_y + extra_region.box[3],
                    ),
                    outline="#c77dff",
                    width=2,
                )
        if localized_cancel_box is not None:
            draw.rectangle(
                (
                    card_x + localized_cancel_box[0],
                    card_y + localized_cancel_box[1],
                    card_x + localized_cancel_box[2],
                    card_y + localized_cancel_box[3],
                ),
                outline="#ff66b3",
                width=3,
            )
        for target in localized_targets:
            if target.localized_box is None:
                continue
            draw.rectangle(
                (
                    card_x + target.localized_box[0],
                    card_y + target.localized_box[1],
                    card_x + target.localized_box[2],
                    card_y + target.localized_box[3],
                ),
                outline="#6aff6a",
                width=2,
            )
        draw.rectangle(search_rect, outline="#ffe066", width=3)
        if localized_box is not None:
            draw.rectangle(
                (
                    card_x + localized_box[0],
                    card_y + localized_box[1],
                    card_x + localized_box[2],
                    card_y + localized_box[3],
                ),
                outline="#3ddcff",
                width=3,
            )
        draw.rectangle(crop_rect, outline="#00ff66", width=2)
        if probe_markers:
            for marker in probe_markers:
                cls._draw_probe_marker(
                    draw,
                    (card_x + marker.point[0], card_y + marker.point[1]),
                    color=marker.color,
                    marker_label=marker.marker_label,
                )
        else:
            cls._draw_crosshair(draw, (card_x + probe_point[0], card_y + probe_point[1]), color="#ff3b30")
        text_origin = (max(4, card_x), max(4, card_y - 18))
        draw.rectangle((text_origin[0] - 2, text_origin[1] - 2, text_origin[0] + (len(label) * 7), text_origin[1] + 14), fill="#08121f")
        draw.text(text_origin, label, fill="#ffffff")
        return annotated

    def audit_tooltip_probe_geometry(
        self,
        *,
        rect_key: str,
        tab: str,
        output_dir: str | Path,
    ) -> list[dict[str, str | tuple[int, int] | tuple[int, int, int, int] | bool | None]]:
        rect = getattr(self.rects, "get", lambda _key: None)(rect_key)
        if rect is None:
            raise ValueError(f"missing_rect:{rect_key}")
        card_x, card_y, card_w, card_h = rect
        card = self._capture_rect(rect_key)
        if card is None:
            raise ValueError(f"missing_rect:{rect_key}")
        layout = self._derive_card_layout(card=card, tab=tab)
        localized_targets = self._localized_tooltip_targets_from_layout(card=card, tab=tab, layout=layout)
        localized_cancel_box = self._localize_target_box(card, search_box=layout.cancel_box, kind="cancel")
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        attempts: list[dict[str, str | tuple[int, int] | tuple[int, int, int, int] | bool | None]] = []
        for target in localized_targets:
            probe_box = self._tooltip_probe_box_for_target(tab=tab, target=target)
            if probe_box is None:
                continue
            region_markers: list[_TooltipProbeMarker] = []
            for probe in self._tooltip_probe_point_specs(probe_box):
                global_point = (int(card_x + probe.point[0]), int(card_y + probe.point[1]))
                click_ok = bool(self.actions.click_client_point(global_point, delay=self._scroll_delay_seconds()))
                frame = self._capture_screen()
                if frame is None:
                    raise ValueError("tooltip_probe_capture_unavailable")
                marker_label = f"{target.kind}_{probe.point_id}_{len(region_markers) + 1}"
                region_markers.append(
                    _TooltipProbeMarker(
                        point=probe.point,
                        marker_label=marker_label,
                        color="#ffd400" if len(region_markers) == 0 else "#ff9f0a",
                    )
                )
                for crop_candidate in self._tooltip_crop_candidates(probe_box, card_size=(card_w, card_h)):
                    crop_box = crop_candidate.box
                    label = self._tooltip_probe_artifact_label(
                        tab=tab,
                        rect_key=rect_key,
                        icon_kind=target.kind,
                        point_id=probe.point_id,
                        crop_id=crop_candidate.crop_id,
                    )
                    overlay = self._render_tooltip_probe_audit_overlay(
                        frame=frame,
                        card_rect=rect,
                        layout=layout,
                        search_box=target.search_box,
                        localized_box=target.localized_box,
                        probe_point=probe.point,
                        tooltip_crop_box=crop_box,
                        label=label,
                        probe_markers=tuple(region_markers),
                        localized_targets=localized_targets,
                        localized_cancel_box=localized_cancel_box,
                    )
                    overlay_path = output_root / f"{label}_overlay.png"
                    tooltip_path = output_root / f"{label}_tooltip.png"
                    overlay.save(overlay_path)
                    tooltip = frame.crop(
                        (
                            int(card_x + crop_box[0]),
                            int(card_y + crop_box[1]),
                            int(card_x + crop_box[2]),
                            int(card_y + crop_box[3]),
                        )
                    )
                    tooltip.save(tooltip_path)
                    attempts.append(
                        {
                            "label": label,
                            "tab": tab,
                            "rect_key": rect_key,
                            "icon_kind": target.kind,
                            "point_id": probe.point_id,
                            "crop_id": crop_candidate.crop_id,
                            "probe_point": probe.point,
                            "global_point": global_point,
                            "recipe_button_box": layout.recipe_button_box,
                            "input_icon_box": layout.input_icon_box,
                            "output_icon_box": layout.output_icon_box,
                            "arrow_search_box": layout.arrow_search_box,
                            "localized_arrow_box": layout.localized_arrow_box,
                            "progress_bar_box": layout.progress_bar_box,
                            "cancel_box": layout.cancel_box,
                            "search_box": target.search_box,
                            "localized_box": target.localized_box,
                            "localized_cancel_box": localized_cancel_box,
                            "icon_box": target.search_box,
                            "tooltip_crop_box": crop_box,
                            "click_ok": click_ok,
                            "probe_markers": tuple(
                                {
                                    "point": marker.point,
                                    "marker_label": marker.marker_label,
                                    "color": marker.color,
                                }
                                for marker in region_markers
                            ),
                            "overlay_artifact": str(overlay_path),
                            "tooltip_artifact": str(tooltip_path),
                        }
                    )
        return attempts

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

    def _smelt_input_quantity_bonus(
        self,
        *,
        output_name: str,
        input_available_quantity: int | None,
        ore_inventory_counts: dict[str, int] | None,
    ) -> tuple[float, str]:
        if input_available_quantity is None or not ore_inventory_counts:
            return 0.0, ""
        expected_inputs = self._expected_input_names(tab="smelt", output_name=output_name)
        if not expected_inputs:
            return 0.0, ""
        expected_ore_name = expected_inputs[0]
        known_quantity = None
        for alias in self._material_aliases(expected_ore_name):
            if alias in ore_inventory_counts:
                known_quantity = int(ore_inventory_counts[alias])
                break
        if known_quantity is None:
            return 0.0, ""
        tolerance = max(3, int(round(max(1, input_available_quantity) * 0.02)))
        if abs(known_quantity - input_available_quantity) <= tolerance:
            return _SMELT_INPUT_QTY_WEIGHT, "input_quantity_match"
        return 0.0, ""

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
    def _foreground_hsv_signature(cls, image: Image.Image) -> tuple[float, float, float]:
        trimmed = cls._trim_foreground_icon(image, prefer_center_x=0.45, prefer_center_y=0.5)
        work = trimmed if trimmed is not None else image
        arr = np.asarray(work.convert("RGB").resize((48, 48)), dtype=np.uint8)
        if arr.size == 0:
            return (0.0, 0.0, 0.0)
        mask = cls._foreground_mask(arr) > 0
        if not np.any(mask):
            mask = np.ones(arr.shape[:2], dtype=bool)
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
        values = hsv[mask]
        if values.size == 0:
            return (0.0, 0.0, 0.0)
        mean = values.mean(axis=0)
        return float(mean[0]), float(mean[1]), float(mean[2])

    @classmethod
    def _color_signature_similarity(cls, template: Image.Image, target: Image.Image) -> float:
        template_h, template_s, template_v = cls._foreground_hsv_signature(template)
        target_h, target_s, target_v = cls._foreground_hsv_signature(target)
        hue_delta = min(abs(template_h - target_h), 180.0 - abs(template_h - target_h)) / 90.0
        sat_delta = abs(template_s - target_s) / 255.0
        val_delta = abs(template_v - target_v) / 255.0
        return max(0.0, 1.0 - ((0.55 * hue_delta) + (0.30 * sat_delta) + (0.15 * val_delta)))

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
        ore_inventory_counts: dict[str, int] | None = None,
        input_available_quantity: int | None = None,
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
            input_quantity_bonus, input_quantity_backend = self._smelt_input_quantity_bonus(
                output_name=name,
                input_available_quantity=input_available_quantity,
                ore_inventory_counts=ore_inventory_counts,
            )
            score = (
                max(self._icon_similarity(template, target_icon) for target_icon in target_icons)
                + self._quantity_match_bonus(name, quantity=output_quantity, inventory_counts=inventory_counts)
                + input_bonus
                + output_region_bonus
                + input_quantity_bonus
            )
            backend_parts = [part for part in (input_backend, output_region_backend, input_quantity_backend) if part]
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

    def _recipe_button_point(self, rect_key: str) -> tuple[int, int]:
        rect = getattr(self.rects, "get", lambda _key: None)(rect_key)
        if rect is None:
            raise ValueError(f"missing_rect:{rect_key}")
        x, y, w, h = rect
        return (
            int(x + round(w * _RECIPE_BUTTON_REL_X)),
            int(y + round(h * _RECIPE_BUTTON_REL_Y)),
        )

    def _read_smelt_recipe_popup_text(self, panel: Image.Image) -> str:
        result = perception_backend.read_text_from_backends(
            self.perception,
            panel,
            prompt=_SMELT_RECIPE_PANEL_PROMPT,
            mode="generic",
            allowed_backend_names=("windows", "legacy"),
        )
        return str(getattr(result, "value", "") or "").strip()

    @staticmethod
    def _normalize_popup_text(text: str | None) -> str:
        return re.sub(r"\s+", " ", str(text or "").upper()).strip()

    @classmethod
    def _verified_smelt_recipe_names(cls, text: str | None) -> set[str]:
        normalized = cls._normalize_popup_text(text)
        patterns = {
            "Copper Bar": ("COPPER BAR",),
            "Iron Bar": ("IRON BAR", "IRON B"),
            "Lead Bar": ("LEAD BAR",),
            "Silicon Bar": ("SILICON BAR", "SILICON"),
            "Aluminum Bar": ("ALUMINUM BAR", "ALUMINIUM BAR", "ALUM"),
        }
        return {
            name
            for name, variants in patterns.items()
            if any(variant in normalized for variant in variants)
        }

    @classmethod
    def _smelt_recipe_popup_tile_signal_count(cls, panel: Image.Image) -> int:
        count = 0
        for ore_icon_box, output_icon_box in _SMELT_RECIPE_SLOT_LAYOUT_SCROLLED:
            x1 = max(0, int(ore_icon_box[0]) - 28)
            y1 = max(0, int(ore_icon_box[1]) - 120)
            x2 = min(panel.size[0], int(output_icon_box[2]) + 22)
            y2 = min(panel.size[1], int(output_icon_box[3]) + 18)
            crop = panel.crop((x1, y1, x2, y2))
            stats = cls._region_signal_stats(crop)
            if stats["dynamic_range"] >= 110.0 and stats["edge_fraction"] >= 0.025:
                count += 1
        return count

    @staticmethod
    def _popup_panel_signature(panel: Image.Image) -> Image.Image:
        width, height = panel.size
        return panel.crop(
            (
                int(round(width * 0.12)),
                int(round(height * 0.18)),
                int(round(width * 0.88)),
                int(round(height * 0.88)),
            )
        )

    def _smelt_recipe_scroll_point(self) -> tuple[int, int]:
        rect = getattr(self.rects, "get", lambda _key: None)("SMELT_RECIPES_PANEL")
        if rect is None:
            raise ValueError("missing_rect:SMELT_RECIPES_PANEL")
        x, y, w, h = rect
        return (int(x + (w // 2)), int(y + (h // 2)))

    def _scroll_smelt_recipe_popup_down(self) -> Image.Image:
        point = self._smelt_recipe_scroll_point()
        before = self._capture_rect("SMELT_RECIPES_PANEL")
        if before is None:
            raise ValueError("missing_rect:SMELT_RECIPES_PANEL")
        before_sig = self._popup_panel_signature(before)
        if not self.actions.scroll_client_wheel(point, _SMELT_RECIPE_SCROLL_DELTA, delay=self._scroll_delay_seconds()):
            raise ValueError("smelt_recipe_popup_scroll_failed")
        after = self._capture_rect("SMELT_RECIPES_PANEL")
        if after is None:
            raise ValueError("missing_rect:SMELT_RECIPES_PANEL")
        after_sig = self._popup_panel_signature(after)
        if self._image_mean_abs_diff(before_sig, after_sig) <= 1.0:
            raise ValueError("smelt_recipe_popup_scroll_not_observed")
        return after

    @staticmethod
    def _smelt_recipe_page_candidates(*, page_index: int) -> tuple[_SmeltRecipePopupEntry, ...]:
        bar_names = tuple(bars_data.list_bars())
        if page_index <= 0:
            visible_names = bar_names[: len(_SMELT_RECIPE_SLOT_LAYOUT_PAGE0)]
            slot_layout = _SMELT_RECIPE_SLOT_LAYOUT_PAGE0
        else:
            start_index = int(page_index) * 2
            visible_names = bar_names[start_index : start_index + len(_SMELT_RECIPE_SLOT_LAYOUT_SCROLLED)]
            slot_layout = _SMELT_RECIPE_SLOT_LAYOUT_SCROLLED[: len(visible_names)]
        return tuple(
            _SmeltRecipePopupEntry(
                output_name=output_name,
                ore_icon_box=ore_icon_box,
                output_icon_box=output_icon_box,
            )
            for output_name, (ore_icon_box, output_icon_box) in zip(visible_names, slot_layout)
        )

    def _iter_smelt_recipe_popup_pages(self) -> list[tuple[int, Image.Image]]:
        panel = self._capture_rect("SMELT_RECIPES_PANEL")
        if panel is None:
            raise ValueError("missing_rect:SMELT_RECIPES_PANEL")
        pages: list[tuple[int, Image.Image]] = [(0, panel)]
        previous_sig = self._popup_panel_signature(panel)
        max_pages = max(1, int((len(bars_data.list_bars()) + 1) // 2))
        for page_index in range(1, max_pages):
            panel = self._scroll_smelt_recipe_popup_down()
            current_sig = self._popup_panel_signature(panel)
            if self._image_mean_abs_diff(previous_sig, current_sig) <= 1.0:
                break
            pages.append((page_index, panel))
            previous_sig = current_sig
        return pages

    def _open_smelt_recipe_popup(self, *, rect_key: str) -> tuple[Image.Image, set[str]]:
        point = self._recipe_button_point(rect_key)
        if not self.actions.click_client_point(point, delay=self._scroll_delay_seconds()):
            raise ValueError("smelt_recipe_popup_open_failed")
        panel = self._capture_rect("SMELT_RECIPES_PANEL")
        if panel is None:
            raise ValueError("missing_rect:SMELT_RECIPES_PANEL")
        panel_text = self._read_smelt_recipe_popup_text(panel)
        normalized = self._normalize_popup_text(panel_text)
        if "SMELT RECIPES" not in normalized:
            raise ValueError("smelt_recipe_popup_not_visible")
        verified_names = self._verified_smelt_recipe_names(panel_text)
        tile_signal_count = self._smelt_recipe_popup_tile_signal_count(panel)
        if not verified_names and tile_signal_count < 3:
            raise ValueError(f"smelt_recipe_popup_unverified:{panel_text or 'blank'}")
        return panel, verified_names

    def _close_smelt_recipe_popup(self) -> None:
        if getattr(self.rects, "get", lambda _key: None)("SMELT_RECIPES_CLOSE") is None:
            raise ValueError("missing_rect:SMELT_RECIPES_CLOSE")
        before = self._capture_rect("SMELT_RECIPES_PANEL")
        if before is None:
            raise ValueError("missing_rect:SMELT_RECIPES_PANEL")
        before_sig = self._popup_panel_signature(before)
        if not self.actions.click_rect_center("SMELT_RECIPES_CLOSE", delay=self._scroll_delay_seconds()):
            raise ValueError("smelt_recipe_popup_close_failed")
        after = self._capture_rect("SMELT_RECIPES_PANEL")
        if after is None:
            raise ValueError("missing_rect:SMELT_RECIPES_PANEL")
        after_sig = self._popup_panel_signature(after)
        if self._image_mean_abs_diff(before_sig, after_sig) <= 6.0:
            raise ValueError("smelt_recipe_popup_close_unverified")

    def _resolve_smelt_output_from_recipe_popup(
        self,
        *,
        card: Image.Image,
        rect_key: str,
    ) -> tuple[str, str]:
        opened = False
        try:
            _panel, _verified_names = self._open_smelt_recipe_popup(rect_key=rect_key)
            opened = True
            target_ore = card.crop((18, 72, 78, 136))
            target_output = card.crop((132, 72, 212, 136))
            popup_pages = self._iter_smelt_recipe_popup_pages()
            scored: list[tuple[str, float]] = []
            for page_index, panel in popup_pages:
                for entry in self._smelt_recipe_page_candidates(page_index=page_index):
                    ore_icon = panel.crop(entry.ore_icon_box)
                    output_icon = panel.crop(entry.output_icon_box)
                    ore_icon_score = self._icon_similarity(ore_icon, target_ore)
                    ore_color_score = self._color_signature_similarity(ore_icon, target_ore)
                    output_icon_score = self._icon_similarity(output_icon, target_output)
                    score = (0.62 * ore_icon_score) + (0.20 * ore_color_score) + (0.18 * output_icon_score)
                    scored.append((entry.output_name, score))
            scored.sort(key=lambda item: item[1], reverse=True)
            if not scored:
                raise ValueError("smelt_recipe_popup_no_candidates")
            top_name, top_score = scored[0]
            next_score = scored[1][1] if len(scored) > 1 else 0.0
            if top_score < 0.80 or (top_score - next_score) < 0.018:
                raise ValueError(f"ambiguous_recipe_popup_match:{top_name}:{top_score:.4f}:{next_score:.4f}")
            return top_name, "smelt_recipe_popup_match"
        finally:
            if opened:
                self._close_smelt_recipe_popup()

    @staticmethod
    def _signal_parts(*, fill_fraction: float, cancel_signal: bool, timer_signal: bool) -> list[str]:
        parts: list[str] = []
        if fill_fraction >= _ACTIVE_FILL_CONFIDENT_MIN:
            parts.append("progress_fill_signal")
        elif fill_fraction >= _ACTIVE_FILL_HINT_MIN:
            parts.append("progress_fill_hint")
        if cancel_signal:
            parts.append("cancel_button_signal")
        if timer_signal:
            parts.append("timer_region_signal")
        return parts

    @classmethod
    def _visual_active_backend(cls, *, fill_fraction: float, cancel_signal: bool, timer_signal: bool) -> str | None:
        evidence_count = int(fill_fraction >= _ACTIVE_FILL_HINT_MIN) + int(cancel_signal) + int(timer_signal)
        if fill_fraction >= _ACTIVE_FILL_CONFIDENT_MIN:
            return "+".join(cls._signal_parts(fill_fraction=fill_fraction, cancel_signal=cancel_signal, timer_signal=timer_signal))
        if evidence_count >= 2:
            return "+".join(cls._signal_parts(fill_fraction=fill_fraction, cancel_signal=cancel_signal, timer_signal=timer_signal))
        return None

    def _resolve_active_state(self, *, tab: str, card: Image.Image) -> tuple[bool, str | None, str]:
        timer_text, timer_backend = self._read_timer_text(card)
        normalized = self._normalize_timer_text(timer_text)
        fill_fraction = self._progress_fill_fraction(card)
        cancel_signal = self._cancel_button_signal(card)
        timer_signal = self._timer_region_signal(card)
        timer_text_signal = self._timer_text_presence_signal(card)
        valid_timer_text = self._valid_timer_like_text(normalized)
        visual_backend = self._visual_active_backend(
            fill_fraction=fill_fraction,
            cancel_signal=cancel_signal,
            timer_signal=timer_signal,
        )
        if visual_backend is not None:
            if valid_timer_text:
                return True, timer_text, f"{visual_backend}+{timer_backend or 'timer_text'}"
            return True, None, visual_backend
        if valid_timer_text:
            support_parts = []
            if cancel_signal:
                support_parts.append("cancel_button_signal")
            if timer_signal:
                support_parts.append("timer_region_signal")
            if timer_text_signal:
                support_parts.append("timer_text_visual_signal")
            if fill_fraction >= _ACTIVE_FILL_HINT_MIN:
                support_parts.append("progress_fill_hint")
            if support_parts:
                return True, timer_text, "+".join([*support_parts, timer_backend or "timer_text"])
        if normalized == "OFF":
            if not cancel_signal and not timer_signal and fill_fraction < _ACTIVE_FILL_HINT_MIN:
                return False, None, timer_backend or "timer_text"
        if normalized == "NORECIPESELECTED":
            if not cancel_signal and not timer_signal and fill_fraction < _ACTIVE_FILL_HINT_MIN:
                return False, None, timer_backend or "timer_text"
        if not normalized and not cancel_signal and not timer_signal and fill_fraction < _ACTIVE_FILL_HINT_MIN:
            return False, None, "visual_idle_signal"
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
        ore_inventory_counts: dict[str, int] | None = None,
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
        input_available_quantity, input_quantity_backend = self._read_input_available_quantity(card)
        try:
            output_name, match_backend = self._resolve_output_name(
                tab=tab,
                card=card,
                templates=templates,
                inventory_counts=inventory_counts,
                output_quantity=output_quantity,
                input_templates=input_templates,
                ore_inventory_counts=ore_inventory_counts,
                input_available_quantity=input_available_quantity,
            )
        except ValueError as exc:
            if tab != "smelt" or not str(exc).startswith("ambiguous_output_match:"):
                raise
            try:
                output_name, match_backend = self._probe_tooltip_identity(
                    rect_key=rect_key,
                    tab=tab,
                    templates=templates,
                )
            except Exception as tooltip_exc:
                try:
                    output_name, match_backend = self._resolve_smelt_output_from_recipe_popup(
                        card=card,
                        rect_key=rect_key,
                    )
                except Exception as popup_exc:
                    raise ValueError(
                        f"smelt_tooltip_fallback_failed:{tooltip_exc};smelt_recipe_popup_fallback_failed:{popup_exc}"
                    ) from popup_exc
        active, timer_text, state_backend = self._resolve_active_state(tab=tab, card=card)
        backend_parts = [part for part in (match_backend, quantity_backend, input_quantity_backend, state_backend) if part]
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
    def _top_anchor_rect_from_card_rect(card_rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x, y, width, _height = card_rect
        return (int(x), int(y - 40), int(width), 150)

    def _capture_top_anchor(self) -> Image.Image | None:
        rect = getattr(self.rects, "get", lambda _key: None)("PRODUCTION_CARD1")
        if rect is None:
            return None
        capture_rect = getattr(self.capture, "capture_client_bbox", None)
        if not callable(capture_rect):
            return None
        return capture_rect(self._top_anchor_rect_from_card_rect(rect))

    @classmethod
    def _top_anchor_signature(cls, anchor: Image.Image | None) -> Image.Image | None:
        if anchor is None:
            return None
        gray = np.asarray(anchor.convert("L"), dtype=np.uint8)
        if gray.size == 0:
            return anchor.convert("L")
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        resized = cv2.resize(blurred, (64, 40), interpolation=cv2.INTER_AREA)
        return Image.fromarray(resized, mode="L")

    @classmethod
    def _is_structural_top_anchor(cls, anchor: Image.Image | None) -> bool:
        if anchor is None:
            return False
        gray = anchor.convert("L")
        extrema = gray.getextrema()
        dynamic_range = float(int(extrema[1]) - int(extrema[0]))
        if dynamic_range < _TOP_ANCHOR_DYNAMIC_RANGE_MIN:
            return False
        width, height = gray.size
        if width < 32 or height < 60:
            return False

        def _band(top: float, bottom: float) -> Image.Image:
            return gray.crop(
                (
                    0,
                    int(round(height * top)),
                    width,
                    int(round(height * bottom)),
                )
            )

        header_band = _band(0.05, 0.28)
        card_upper_band = _band(0.43, 0.65)
        card_lower_band = _band(0.65, 0.95)

        header_stats = cls._region_signal_stats(header_band.convert("RGB"))
        candidate_card_stats = (
            cls._region_signal_stats(card_upper_band.convert("RGB")),
            cls._region_signal_stats(card_lower_band.convert("RGB")),
        )
        strongest_card_band = max(
            candidate_card_stats,
            key=lambda stats: (stats["dynamic_range"], stats["edge_fraction"]),
        )

        return bool(
            header_stats["edge_fraction"] >= _TOP_ANCHOR_HEADER_EDGE_MIN
            and strongest_card_band["edge_fraction"] >= _TOP_ANCHOR_CARD_BAND_EDGE_MIN
            and strongest_card_band["dynamic_range"] >= _TOP_ANCHOR_DYNAMIC_RANGE_MIN
            and strongest_card_band["bright_fraction"] >= 0.001
            and max(
                float(ImageStat.Stat(card_upper_band).mean[0]),
                float(ImageStat.Stat(card_lower_band).mean[0]),
            )
            >= _TOP_ANCHOR_CARD_BAND_MEAN_MIN
        )

    def _scroll_to_lower_cards(self, *, top_anchor: Image.Image | None) -> None:
        point = self._wheel_point()
        if not self.actions.scroll_client_wheel(point, -120, delay=self._scroll_delay_seconds()):
            raise ValueError("production_scroll_down_failed")
        current = self._capture_top_anchor()
        diff = self._image_mean_abs_diff(self._top_anchor_signature(top_anchor), self._top_anchor_signature(current))
        if diff <= _TOP_ANCHOR_STABLE_DIFF_MAX:
            raise ValueError(f"production_scroll_down_not_observed:{diff:.3f}")

    def _scroll_to_top_view(self) -> Image.Image | None:
        point = self._wheel_point()
        previous = self._top_anchor_signature(self._capture_top_anchor())
        best_candidate = None
        best_diff = 1e9
        for _ in range(_PRODUCTION_TOP_SCROLL_MAX_ATTEMPTS):
            if not self.actions.scroll_client_wheel(point, 120, delay=self._scroll_delay_seconds()):
                raise ValueError("production_scroll_up_failed")
            current_anchor_image = self._capture_top_anchor()
            if current_anchor_image is None:
                raise ValueError("missing_rect:PRODUCTION_CARD1")
            current_anchor = self._top_anchor_signature(current_anchor_image)
            is_structural_top = self._is_structural_top_anchor(current_anchor_image)
            if previous is not None:
                diff = self._image_mean_abs_diff(previous, current_anchor)
                if is_structural_top and diff < best_diff:
                    best_diff = diff
                    best_candidate = current_anchor_image
                if is_structural_top and diff <= _TOP_ANCHOR_STABLE_DIFF_MAX:
                    return current_anchor_image
            previous = current_anchor
        if best_candidate is not None and best_diff <= _TOP_ANCHOR_BEST_DIFF_MAX:
            return best_candidate
        raise ValueError(f"production_top_latch_failed:{best_diff:.3f}")

    def _scroll_back_to_top(self, *, top_anchor: Image.Image | None) -> None:
        if top_anchor is None:
            return
        point = self._wheel_point()
        top_anchor_frame = self._top_anchor_signature(top_anchor)
        best_diff = 1e9
        best_current = None
        current = self._capture_top_anchor()
        current_frame = self._top_anchor_signature(current)
        current_diff = self._image_mean_abs_diff(top_anchor_frame, current_frame)
        last_structural_frame = current_frame if self._is_structural_top_anchor(current) else None
        if current_diff <= _TOP_ANCHOR_STABLE_DIFF_MAX and last_structural_frame is not None:
            return
        if current is not None and last_structural_frame is not None:
            best_diff = current_diff
            best_current = current
        for _ in range(_PRODUCTION_TOP_SCROLL_MAX_ATTEMPTS):
            if not self.actions.scroll_client_wheel(point, 120, delay=self._scroll_delay_seconds()):
                break
            current = self._capture_top_anchor()
            current_frame = self._top_anchor_signature(current)
            diff = self._image_mean_abs_diff(top_anchor_frame, current_frame)
            is_structural_top = self._is_structural_top_anchor(current)
            if current is not None and is_structural_top and diff < best_diff:
                best_diff = diff
                best_current = current
            if is_structural_top and diff <= _TOP_ANCHOR_STABLE_DIFF_MAX:
                return
            if (
                is_structural_top
                and last_structural_frame is not None
                and self._image_mean_abs_diff(last_structural_frame, current_frame) <= _TOP_ANCHOR_STABLE_DIFF_MAX
            ):
                return
            last_structural_frame = current_frame if is_structural_top else None
        current = self._capture_top_anchor()
        current_diff = self._image_mean_abs_diff(top_anchor_frame, self._top_anchor_signature(current))
        if current_diff <= _TOP_ANCHOR_STABLE_DIFF_MAX and self._is_structural_top_anchor(current):
            return
        if best_current is not None and best_diff <= _TOP_ANCHOR_BEST_DIFF_MAX:
            return
        if current_diff > _TOP_ANCHOR_STABLE_DIFF_MAX:
            raise ValueError(f"production_scroll_up_failed:{current_diff:.3f}:{best_diff:.3f}")

    def read_cards(
        self,
        *,
        tab: str,
        open_tab,
        templates: dict[str, Image.Image],
        inventory_counts: dict[str, int],
        input_templates: dict[str, Image.Image] | None = None,
        ore_inventory_counts: dict[str, int] | None = None,
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
            self._read_card(slot_index=1, tab=tab, rect_key="PRODUCTION_CARD1", templates=templates, inventory_counts=inventory_counts, input_templates=input_templates, ore_inventory_counts=ore_inventory_counts),
            self._read_card(slot_index=2, tab=tab, rect_key="PRODUCTION_CARD2", templates=templates, inventory_counts=inventory_counts, input_templates=input_templates, ore_inventory_counts=ore_inventory_counts),
        ]
        self._scroll_to_lower_cards(top_anchor=top_anchor)
        try:
            cards.extend(
                [
                    # After one wheel-down, the next visible card pair occupies the upper view slots.
                    self._read_card(slot_index=3, tab=tab, rect_key="PRODUCTION_CARD1", templates=templates, inventory_counts=inventory_counts, input_templates=input_templates, ore_inventory_counts=ore_inventory_counts),
                    self._read_card(slot_index=4, tab=tab, rect_key="PRODUCTION_CARD2", templates=templates, inventory_counts=inventory_counts, input_templates=input_templates, ore_inventory_counts=ore_inventory_counts),
                ]
            )
        finally:
            if not open_tab():
                raise ValueError(f"open_tab_failed:{tab}")
            self._scroll_back_to_top(top_anchor=top_anchor)
        return cards
