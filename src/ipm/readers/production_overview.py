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
_PRODUCTION_TOP_SCROLL_MAX_ATTEMPTS = 3
_RECIPE_BUTTON_REL_X = 0.375
_RECIPE_BUTTON_REL_Y = 0.807
_TOOLTIP_PROBE_POINTS = (
    ("center", (0.50, 0.50)),
    ("upper_left", (0.34, 0.34)),
    ("lower_right", (0.66, 0.66)),
)
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
    def _tooltip_probe_points(icon_box: tuple[int, int, int, int]) -> tuple[tuple[int, int], ...]:
        return tuple(spec.point for spec in ProductionOverviewReader._tooltip_probe_point_specs(icon_box))

    @staticmethod
    def _tooltip_probe_point_specs(icon_box: tuple[int, int, int, int]) -> tuple[_TooltipProbePoint, ...]:
        x1, y1, x2, y2 = icon_box
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        return tuple(
            _TooltipProbePoint(
                point_id=point_id,
                point=(
                    int(x1 + round(width * x_fraction)),
                    int(y1 + round(height * y_fraction)),
                ),
            )
            for point_id, (x_fraction, y_fraction) in _TOOLTIP_PROBE_POINTS
        )

    @staticmethod
    def _tooltip_crop_box(icon_box: tuple[int, int, int, int], *, card_size: tuple[int, int]) -> tuple[int, int, int, int]:
        card_width, card_height = card_size
        x1, y1, x2, _y2 = icon_box
        crop = (
            max(0, x1 - 78),
            max(0, y1 - 36),
            min(card_width, x2 + 16),
            min(card_height, y1 + 14),
        )
        return crop

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
    def _match_tooltip_label(cls, text: str | None) -> str | None:
        normalized = cls._normalize_tooltip_text(text)
        if not normalized:
            return None
        lookup = cls._known_tooltip_labels()
        return lookup.get(normalized) or lookup.get(normalized.replace(" ", ""))

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
        for region in self._tooltip_icon_regions(tab=tab, card_size=(card_w, card_h)):
            crop_box = self._tooltip_crop_box(region.box, card_size=(card_w, card_h))
            for probe in self._tooltip_probe_point_specs(region.box):
                point = probe.point
                global_point = (int(card_x + point[0]), int(card_y + point[1]))
                if not self.actions.click_client_point(global_point, delay=self._scroll_delay_seconds()):
                    raise ValueError("tooltip_probe_click_failed")
                frame = self._capture_screen()
                if frame is None:
                    raise ValueError("tooltip_probe_capture_unavailable")
                tooltip = frame.crop(
                    (
                        int(card_x + crop_box[0]),
                        int(card_y + crop_box[1]),
                        int(card_x + crop_box[2]),
                        int(card_y + crop_box[3]),
                    )
                )
                result = self.perception.read_text(tooltip, prompt=_TOOLTIP_PROMPT, mode="generic")
                matched_label = self._match_tooltip_label(getattr(result, "value", ""))
                if matched_label is None:
                    continue
                if tab != "smelt":
                    return matched_label, f"tooltip_probe_{region.kind}_{getattr(result, 'backend', '') or 'generic'}"
                output_name = self._resolve_smelt_tooltip_output(tooltip_label=matched_label, templates=templates)
                if output_name is not None:
                    return output_name, f"tooltip_probe_{region.kind}_{getattr(result, 'backend', '') or 'generic'}"
        raise ValueError("tooltip_probe_no_valid_label")

    @staticmethod
    def _tooltip_probe_artifact_label(*, tab: str, rect_key: str, icon_kind: str, point_id: str) -> str:
        return f"{tab}_{rect_key}_{icon_kind}_{point_id}"

    @staticmethod
    def _draw_crosshair(draw: ImageDraw.ImageDraw, point: tuple[int, int], *, color: str) -> None:
        x, y = point
        draw.line((x - 8, y, x + 8, y), fill=color, width=2)
        draw.line((x, y - 8, x, y + 8), fill=color, width=2)

    @classmethod
    def _render_tooltip_probe_audit_overlay(
        cls,
        *,
        frame: Image.Image,
        card_rect: tuple[int, int, int, int],
        icon_box: tuple[int, int, int, int],
        probe_point: tuple[int, int],
        tooltip_crop_box: tuple[int, int, int, int],
        label: str,
    ) -> Image.Image:
        annotated = frame.convert("RGB").copy()
        draw = ImageDraw.Draw(annotated)
        card_x, card_y, card_w, card_h = card_rect
        card_box = (card_x, card_y, card_x + card_w, card_y + card_h)
        icon_rect = (
            card_x + icon_box[0],
            card_y + icon_box[1],
            card_x + icon_box[2],
            card_y + icon_box[3],
        )
        crop_rect = (
            card_x + tooltip_crop_box[0],
            card_y + tooltip_crop_box[1],
            card_x + tooltip_crop_box[2],
            card_y + tooltip_crop_box[3],
        )
        draw.rectangle(card_box, outline="#00e5ff", width=3)
        draw.rectangle(icon_rect, outline="#ffd400", width=3)
        draw.rectangle(crop_rect, outline="#00ff66", width=2)
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
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        attempts: list[dict[str, str | tuple[int, int] | tuple[int, int, int, int] | bool | None]] = []
        for region in self._tooltip_icon_regions(tab=tab, card_size=(card_w, card_h)):
            crop_box = self._tooltip_crop_box(region.box, card_size=(card_w, card_h))
            for probe in self._tooltip_probe_point_specs(region.box):
                global_point = (int(card_x + probe.point[0]), int(card_y + probe.point[1]))
                click_ok = bool(self.actions.click_client_point(global_point, delay=self._scroll_delay_seconds()))
                frame = self._capture_screen()
                if frame is None:
                    raise ValueError("tooltip_probe_capture_unavailable")
                label = self._tooltip_probe_artifact_label(
                    tab=tab,
                    rect_key=rect_key,
                    icon_kind=region.kind,
                    point_id=probe.point_id,
                )
                overlay = self._render_tooltip_probe_audit_overlay(
                    frame=frame,
                    card_rect=rect,
                    icon_box=region.box,
                    probe_point=probe.point,
                    tooltip_crop_box=crop_box,
                    label=label,
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
                        "icon_kind": region.kind,
                        "point_id": probe.point_id,
                        "probe_point": probe.point,
                        "global_point": global_point,
                        "icon_box": region.box,
                        "tooltip_crop_box": crop_box,
                        "click_ok": click_ok,
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
        visual_backend = self._visual_active_backend(
            fill_fraction=fill_fraction,
            cancel_signal=cancel_signal,
            timer_signal=timer_signal,
        )
        if visual_backend is not None:
            if normalized and _TIMER_TEXT_RE.search(normalized):
                return True, timer_text, f"{visual_backend}+{timer_backend or 'timer_text'}"
            return True, None, visual_backend
        if normalized and _TIMER_TEXT_RE.search(normalized) and (cancel_signal or timer_signal or fill_fraction >= _ACTIVE_FILL_HINT_MIN):
            return True, timer_text, timer_backend or "timer_text"
        if normalized == "OFF":
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
        for _ in range(_PRODUCTION_TOP_SCROLL_MAX_ATTEMPTS):
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
        for _ in range(_PRODUCTION_TOP_SCROLL_MAX_ATTEMPTS):
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
            self._scroll_back_to_top(top_anchor=top_anchor)
        return cards
