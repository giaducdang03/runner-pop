"""Ghép ảnh cuối - phần nhẹ, slider gọi liên tục."""
from __future__ import annotations

import cv2
import numpy as np

from .schema import Cutout


def to_gray3(rgb_f: np.ndarray) -> np.ndarray:
    # Rec.709 luma, làm việc trong float
    g = rgb_f @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    return np.repeat(g[..., None], 3, axis=2)


def compose(c: Cutout, intensity: float = 1.0, bg_filter: str = "bw", color: str = "#808080") -> np.ndarray:
    """
    out = α·F + (1−α)·((1−s)·B + s·Bf), với Bf là nền đã lọc bằng `bg_filter` (xem bgfilter.py).
    intensity ∈ [0,1]: 0 = nền gốc (chỉ viền được khử nhiễm), 1 = nền lọc hoàn toàn.
    Mặc định "bw" cho kết quả giống hệt bản đen trắng trước đây. Trả về uint8 RGB.
    """
    from .bgfilter import apply_bg_filter      # import muộn: bgfilter dùng to_gray3 của module này
    s = float(np.clip(intensity, 0.0, 1.0))
    bf = apply_bg_filter(c, bg_filter, color)
    bg = (1.0 - s) * c.background + s * bf
    a = c.alpha[..., None]
    out = a * c.foreground + (1.0 - a) * bg
    return (np.clip(out, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def thumbnail_cutout(c: Cutout, max_side: int = 240) -> Cutout:
    """Bản thu nhỏ của cutout (α, F, B, ảnh) để dựng ảnh xem trước nhanh; cache bộ lọc riêng."""
    h, w = c.alpha.shape
    k = min(1.0, max_side / max(h, w))
    size = (max(1, round(w * k)), max(1, round(h * k)))
    rs = lambda a: cv2.resize(a, size, interpolation=cv2.INTER_AREA)
    return Cutout(image=rs(c.image), alpha=rs(c.alpha), foreground=rs(c.foreground), background=rs(c.background),
                  trimap=cv2.resize(c.trimap, size, interpolation=cv2.INTER_NEAREST))


def alpha_to_png(alpha: np.ndarray) -> np.ndarray:
    return (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
