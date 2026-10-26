"""Ghép ảnh cuối - phần nhẹ, slider gọi liên tục."""
from __future__ import annotations

import numpy as np

from .schema import Cutout


def to_gray3(rgb_f: np.ndarray) -> np.ndarray:
    # Rec.709 luma, làm việc trong float
    g = rgb_f @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    return np.repeat(g[..., None], 3, axis=2)


def compose(c: Cutout, intensity: float = 1.0) -> np.ndarray:
    """
    intensity ∈ [0,1]: 0 = gần như ảnh gốc (chỉ viền được khử nhiễm), 1 = nền đen trắng hoàn toàn.
    Trả về uint8 RGB.
    """
    s = float(np.clip(intensity, 0.0, 1.0))
    gray = to_gray3(c.background)
    bg = (1.0 - s) * c.background + s * gray
    a = c.alpha[..., None]
    out = a * c.foreground + (1.0 - a) * bg
    return (np.clip(out, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def alpha_to_png(alpha: np.ndarray) -> np.ndarray:
    return (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
