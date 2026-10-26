"""Phép toán trên mask nhị phân (uint8 0/255) dùng chung giữa các engine."""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def box_to_mask(box: Tuple[int, int, int, int], shape) -> np.ndarray:
    """Fallback khi không có mask từ YOLO: elip trong box."""
    h, w = shape[:2]
    m = np.zeros((h, w), np.uint8)
    x1, y1, x2, y2 = box
    c = ((x1 + x2) // 2, (y1 + y2) // 2)
    axes = (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2))
    cv2.ellipse(m, c, axes, 0, 0, 360, 255, -1)
    return m


def keep_components_overlapping(mask: np.ndarray, ref: np.ndarray,
                                min_overlap: float = 0.5) -> np.ndarray:
    """Giữ thành phần lớn nhất + mọi thành phần có >= min_overlap diện tích nằm trong `ref`."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 127).astype(np.uint8), 8)
    if n <= 1:
        return mask
    ref_b = ref > 127
    keep = np.zeros(mask.shape, np.uint8)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    for k in range(1, n):
        comp = labels == k
        if k == largest or ref_b[comp].mean() >= min_overlap:
            keep[comp] = 255
    return keep


def fill_small_holes(mask: np.ndarray, max_hole_frac: float = 0.002) -> np.ndarray:
    inv = (mask <= 127).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, 4)
    limit = max_hole_frac * mask.size
    out = mask.copy()
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        touches_border = x == 0 or y == 0 or x + w >= mask.shape[1] or y + h >= mask.shape[0]
        if not touches_border and area < limit:
            out[labels == k] = 255
    return out


def mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask > 127)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
