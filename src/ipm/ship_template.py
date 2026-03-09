from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


_DEFAULT_TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "assets" / "ship_template.png"


@dataclass(slots=True, frozen=True)
class ShipTemplateMatch:
    center_x: int
    center_y: int
    bbox: tuple[int, int, int, int]
    width: int
    height: int
    area: int
    score: float
    scale: float


@dataclass(slots=True, frozen=True)
class ShipTemplateDetection:
    status: str
    match: ShipTemplateMatch | None = None
    raw_match: ShipTemplateMatch | None = None
    best_score: float | None = None
    best_scale: float | None = None
    reject_reason: str | None = None


@dataclass(slots=True, frozen=True)
class _PreparedTemplate:
    gray: np.ndarray
    edge: np.ndarray
    mask: np.ndarray
    area: int


def _as_gray(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    rgb = arr[..., :3]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _as_edge(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.Canny(blurred, 40, 120)


def _alpha_mask(image: Image.Image) -> np.ndarray:
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    alpha = rgba[..., 3]
    return (alpha > 0).astype(np.uint8) * 255


def _prepare_template(image: Image.Image) -> _PreparedTemplate:
    gray = _as_gray(image)
    edge = _as_edge(gray)
    mask = _alpha_mask(image)
    area = int(np.count_nonzero(mask))
    return _PreparedTemplate(gray=gray, edge=edge, mask=mask, area=area)


@lru_cache(maxsize=8)
def _load_template_from_path(path: str) -> _PreparedTemplate | None:
    template_path = Path(path)
    if not template_path.exists():
        return None
    try:
        image = Image.open(template_path)
    except OSError:
        return None
    return _prepare_template(image)


def default_ship_template_path() -> str:
    return str(_DEFAULT_TEMPLATE_PATH)


def _normalize_search_region(
    search_region: tuple[int, int, int, int] | None,
    *,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    if search_region is None:
        return None
    width, height = image_size
    left, top, right, bottom = (int(value) for value in search_region)
    left = max(0, min(left, width - 1))
    top = max(0, min(top, height - 1))
    right = max(left + 1, min(right, width))
    bottom = max(top + 1, min(bottom, height))
    if right - left < 2 or bottom - top < 2:
        return None
    return (left, top, right, bottom)


def detect_ship_template(
    image: Image.Image,
    *,
    template_path: str | None = None,
    template_image: Image.Image | None = None,
    search_region: tuple[int, int, int, int] | None = None,
    scales: tuple[float, ...] = (0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25),
    threshold: float = 0.55,
    use_edges: bool = True,
    min_scale: float = 0.0,
    min_width: int = 0,
    min_height: int = 0,
    min_area: int = 0,
) -> ShipTemplateDetection:
    prepared = (
        _prepare_template(template_image)
        if template_image is not None
        else _load_template_from_path(template_path or default_ship_template_path())
    )
    if prepared is None:
        return ShipTemplateDetection(status="template_missing")
    normalized_search_region = _normalize_search_region(search_region, image_size=image.size)
    if search_region is not None and normalized_search_region is None:
        return ShipTemplateDetection(status="search_region_invalid")
    if normalized_search_region is not None:
        offset_left, offset_top, offset_right, offset_bottom = normalized_search_region
        working_image = image.crop((offset_left, offset_top, offset_right, offset_bottom))
    else:
        offset_left = 0
        offset_top = 0
        working_image = image
    scene_gray = _as_gray(working_image)
    scene_edge = _as_edge(scene_gray)
    best_score: float | None = None
    best_scale: float | None = None
    best_match: ShipTemplateMatch | None = None
    positive_scales = tuple(scale for scale in scales if float(scale) > 0.0)
    if not positive_scales:
        positive_scales = (1.0,)
    scene_work_shape = scene_gray.shape
    for scale in positive_scales:
        tpl_w = max(1, int(round(prepared.gray.shape[1] * float(scale))))
        tpl_h = max(1, int(round(prepared.gray.shape[0] * float(scale))))
        if tpl_w > scene_work_shape[1] or tpl_h > scene_work_shape[0]:
            continue
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        gray = cv2.resize(prepared.gray, (tpl_w, tpl_h), interpolation=interpolation)
        edge = cv2.resize(prepared.edge, (tpl_w, tpl_h), interpolation=interpolation)
        mask = cv2.resize(prepared.mask, (tpl_w, tpl_h), interpolation=cv2.INTER_NEAREST)
        if int(np.count_nonzero(mask)) < 24:
            continue
        template_work = edge if use_edges else gray
        scene_work = scene_edge if use_edges else scene_gray
        result = cv2.matchTemplate(scene_work, template_work, cv2.TM_SQDIFF_NORMED, mask=mask)
        min_val, _max_val, min_loc, _max_loc = cv2.minMaxLoc(result)
        score = 1.0 - float(min_val) if np.isfinite(min_val) else float("nan")
        if (not np.isfinite(min_val)) or (not np.isfinite(score)) or score < 0.0 or score > 1.0:
            result = cv2.matchTemplate(scene_gray, gray, cv2.TM_SQDIFF_NORMED, mask=mask)
            min_val, _max_val, min_loc, _max_loc = cv2.minMaxLoc(result)
            score = 1.0 - float(min_val) if np.isfinite(min_val) else float("nan")
        if (not np.isfinite(min_val)) or (not np.isfinite(score)) or score < 0.0 or score > 1.0:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_scale = float(scale)
            left, top = int(min_loc[0]), int(min_loc[1])
            bbox = (
                left + offset_left,
                top + offset_top,
                left + tpl_w + offset_left,
                top + tpl_h + offset_top,
            )
            best_match = ShipTemplateMatch(
                center_x=left + int(round(tpl_w / 2.0)) + offset_left,
                center_y=top + int(round(tpl_h / 2.0)) + offset_top,
                bbox=bbox,
                width=tpl_w,
                height=tpl_h,
                area=max(1, int(round(prepared.area * (float(scale) ** 2)))),
                score=score,
                scale=float(scale),
            )
    if best_score is None or best_scale is None or best_match is None:
        return ShipTemplateDetection(status="not_found")
    if best_score < float(threshold):
        return ShipTemplateDetection(
            status="below_threshold",
            raw_match=best_match,
            best_score=best_score,
            best_scale=best_scale,
        )
    if float(best_match.scale) < float(min_scale):
        return ShipTemplateDetection(
            status="rejected",
            raw_match=best_match,
            best_score=best_score,
            best_scale=best_scale,
            reject_reason="min_scale",
        )
    if int(best_match.width) < int(min_width):
        return ShipTemplateDetection(
            status="rejected",
            raw_match=best_match,
            best_score=best_score,
            best_scale=best_scale,
            reject_reason="min_width",
        )
    if int(best_match.height) < int(min_height):
        return ShipTemplateDetection(
            status="rejected",
            raw_match=best_match,
            best_score=best_score,
            best_scale=best_scale,
            reject_reason="min_height",
        )
    if int(best_match.area) < int(min_area):
        return ShipTemplateDetection(
            status="rejected",
            raw_match=best_match,
            best_score=best_score,
            best_scale=best_scale,
            reject_reason="min_area",
        )
    return ShipTemplateDetection(
        status="match",
        match=best_match,
        raw_match=best_match,
        best_score=best_score,
        best_scale=best_scale,
    )
