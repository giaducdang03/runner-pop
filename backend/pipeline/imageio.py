"""Đọc ảnh / đổi kích thước ảnh làm việc."""
from __future__ import annotations

import io

import cv2
import numpy as np

from .config import MAX_SIDE


def decode_image(data: bytes) -> np.ndarray:
    """bytes -> RGB uint8. Tôn trọng EXIF orientation."""
    from PIL import Image, ImageOps

    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img).convert("RGB")
    return np.asarray(img)


def resize_to_max_side(rgb: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    h, w = rgb.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return rgb
    scale = max_side / s
    return cv2.resize(rgb, (int(round(w * scale)), int(round(h * scale))),
                      interpolation=cv2.INTER_AREA)
