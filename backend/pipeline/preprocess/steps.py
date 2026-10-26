"""
Các bước tiền xử lý cho bước PHÁT HIỆN. Không bao giờ dùng cho ảnh đầu ra
-> không làm đổi màu người chạy.

Thêm bước mới:

    @STEPS.register("ten_buoc", label="Mô tả ngắn")
    def ten_buoc(rgb: np.ndarray) -> np.ndarray:   # uint8 RGB vào, uint8 RGB ra
        ...

rồi (tuỳ chọn) thêm quy tắc chọn tự động trong quality.plan_steps().
"""
from __future__ import annotations

import cv2
import numpy as np

from ..registry import Registry

STEPS = Registry("preprocess")


def _lab_apply_l(rgb: np.ndarray, fn) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    return cv2.cvtColor(cv2.merge([fn(l), a, b]), cv2.COLOR_LAB2RGB)


@STEPS.register("levels", label="Giãn histogram (ảnh thiếu tương phản)")
def levels(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lo, hi = np.percentile(gray, [1, 99])
    if hi - lo < 8:
        return rgb
    out = (rgb.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8)


@STEPS.register("gamma", label="Làm sáng ảnh tối (gamma)")
def gamma(rgb: np.ndarray) -> np.ndarray:
    def apply(l):
        mean_l = float(l.mean())
        if mean_l >= 90:
            return l
        g = np.clip(mean_l / 110.0, 0.45, 1.0)           # <1 => sáng lên
        lut = (np.linspace(0, 1, 256) ** g * 255).astype(np.uint8)
        return cv2.LUT(l, lut)
    return _lab_apply_l(rgb, apply)


@STEPS.register("clahe", label="Tăng tương phản cục bộ (CLAHE)")
def clahe(rgb: np.ndarray) -> np.ndarray:
    c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return _lab_apply_l(rgb, c.apply)


@STEPS.register("denoise", label="Giảm nhiễu hạt")
def denoise(rgb: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(rgb, None, 5, 5, 5, 15)


@STEPS.register("sharpen", label="Làm nét nhẹ (unsharp mask)")
def sharpen(rgb: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(rgb, (0, 0), 1.5)
    return cv2.addWeighted(rgb, 1.6, blur, -0.6, 0)
