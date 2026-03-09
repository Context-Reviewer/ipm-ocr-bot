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
    best_score: float | None = None
    best_scale: float | None = None


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


def detect_ship_template(
    image: Image.Image,
    *,
    template_path: str | None = None,
    template_image: Image.Image | None = None,
    scales: tuple[float, ...] = (1.0, 0.75, 0.5, 0.35, 0.25, 0.18, 0.12, 0.08),
    threshold: float = 0.55,
    use_edges: bool = True,
) -> ShipTemplateDetection:
    prepared = (
        _prepare_template(template_image)
        if template_image is not None
        else _load_template_from_path(template_path or default_ship_template_path())
    )
    if prepared is None:
        return ShipTemplateDetection(status="template_missing")
    scene_gray = _as_gray(image)
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
        if not np.isfinite(min_val):
            result = cv2.matchTemplate(scene_gray, gray, cv2.TM_SQDIFF_NORMED, mask=mask)
            min_val, _max_val, min_loc, _max_loc = cv2.minMaxLoc(result)
        if not np.isfinite(min_val):
            continue
        score = 1.0 - float(min_val)
        if best_score is None or score > best_score:
            best_score = score
            best_scale = float(scale)
            left, top = int(min_loc[0]), int(min_loc[1])
            bbox = (left, top, left + tpl_w, top + tpl_h)
            best_match = ShipTemplateMatch(
                center_x=left + int(round(tpl_w / 2.0)),
                center_y=top + int(round(tpl_h / 2.0)),
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
        return ShipTemplateDetection(status="below_threshold", best_score=best_score, best_scale=best_scale)
    return ShipTemplateDetection(status="match", match=best_match, best_score=best_score, best_scale=best_scale)
