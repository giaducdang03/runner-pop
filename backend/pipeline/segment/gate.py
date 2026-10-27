"""Khối "gate" dùng chung: giới hạn alpha của model toàn ảnh trong người được chọn."""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from ..masks import keep_components_overlapping


def gate_to_person(alpha: np.ndarray, init: np.ndarray,
                   box: Tuple[int, int, int, int],
                   exclude: Optional[np.ndarray] = None) -> np.ndarray:
    """
    α (float32 HxW 0..1) -> α chỉ còn trong mask người: mask YOLO `init` nở ra
    `max(5, 4% cạnh dài box)` px (để không cắt tay/chân mảnh) rồi làm mềm, trừ vùng
    người khác `exclude` (bool), và bỏ các mảnh rời không chạm mask YOLO.
    Mọi đối số cùng độ phân giải làm việc.
    """
    x1, y1, x2, y2 = box
    r = max(5, int(0.04 * max(x2 - x1, y2 - y1)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    gate = cv2.dilate((init > 127).astype(np.uint8), k).astype(np.float32)
    if exclude is not None:
        gate[exclude] = 0
    gate = cv2.GaussianBlur(gate, (0, 0), r / 3.0)
    if exclude is not None:
        gate[exclude] = 0
    alpha = alpha * gate

    keep = keep_components_overlapping(((alpha > 0.5).astype(np.uint8)) * 255, init)
    keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))  # giữ viền mềm
    return (alpha * (keep > 127)).astype(np.float32)
