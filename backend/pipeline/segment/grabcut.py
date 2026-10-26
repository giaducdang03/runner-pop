"""Engine GrabCut -> trimap -> matting (closed-form / KNN)."""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from ..masks import fill_small_holes, keep_components_overlapping
from ..matting import finish_cutout, make_trimap, matte
from ..types import Cutout, Person, Scene, SegParams
from .base import ENGINES, person_init_mask, to_float


def refine_mask_grabcut(rgb: np.ndarray, init_mask: np.ndarray,
                        box: Tuple[int, int, int, int],
                        iters: int = 4, rescue_core: bool = True) -> np.ndarray:
    """
    Dùng mask YOLO làm gợi ý: lõi = chắc chắn FG, vành quanh = có thể,
    ngoài vành = chắc chắn BG. GrabCut chạy trong ROI quanh box để nhanh.
    """
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    pad = int(0.15 * max(bw, bh)) + 8
    rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
    rx2, ry2 = min(w, x2 + pad), min(h, y2 + pad)

    roi = np.ascontiguousarray(rgb[ry1:ry2, rx1:rx2])
    m0 = init_mask[ry1:ry2, rx1:rx2] > 127

    k = max(3, int(0.03 * max(bw, bh)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    sure_fg = cv2.erode(m0.astype(np.uint8), kernel, iterations=2).astype(bool)
    maybe = cv2.dilate(m0.astype(np.uint8), kernel, iterations=2).astype(bool)

    gc = np.full(m0.shape, cv2.GC_BGD, np.uint8)
    gc[maybe] = cv2.GC_PR_BGD
    gc[m0] = cv2.GC_PR_FGD
    gc[sure_fg] = cv2.GC_FGD

    if sure_fg.sum() < 50 or (~maybe).sum() < 50:     # không đủ mẫu cho GMM
        return init_mask

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(cv2.cvtColor(roi, cv2.COLOR_RGB2BGR), gc, None, bgd, fgd,
                    iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return init_mask

    out = np.zeros((h, w), np.uint8)
    fg = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8) * 255
    out[ry1:ry2, rx1:rx2] = fg

    # GrabCut hay cắt nhầm tay/chân có màu da gần màu nền (hoặc người phía sau):
    # gộp lại lõi chắc chắn của mask YOLO để không mất chi.
    if rescue_core:
        out[(cv2.erode((init_mask > 127).astype(np.uint8),
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0)] = 255

    # giữ các thành phần liên thông chồng lên mask YOLO (không chỉ thành phần lớn nhất), lấp lỗ nhỏ
    out = keep_components_overlapping(out, init_mask)
    return fill_small_holes(out, max_hole_frac=0.002)


def cutout_from_mask(rgb: np.ndarray, mask: np.ndarray, band_frac: float, method: str) -> Cutout:
    """mask -> trimap -> matting -> khử nhiễm màu. Dùng chung cho các engine dựa trên mask."""
    tri = make_trimap(mask, band_frac=band_frac)
    rgb_f = to_float(rgb)
    alpha = matte(rgb_f, tri, method=method)
    return finish_cutout(rgb_f, alpha, tri)


def _make_engine(method: str):
    def run(scene: Scene, person: Person, params: SegParams) -> Cutout:
        init = person_init_mask(scene, person)
        mask = refine_mask_grabcut(scene.rgb, init, person.box)
        return cutout_from_mask(scene.rgb, mask, params.band_frac, method)
    return run


ENGINES.register("cf", label="GrabCut + Closed-form")(_make_engine("cf"))
ENGINES.register("knn", label="GrabCut + KNN (chậm hơn)")(_make_engine("knn"))
