"""Đánh giá chất lượng ảnh -> quyết định chuỗi tiền xử lý cho bước phát hiện."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List

import cv2
import numpy as np

from .. import config as C


@dataclass
class Quality:
    brightness: float    # trung bình độ sáng 0..255
    contrast: float      # p95 - p5 của độ sáng
    sharpness: float     # phương sai Laplacian (chuẩn hoá về ảnh 640 px)
    noise: float         # sigma nhiễu ước lượng (Immerkaer), thang 0..255

    @property
    def is_dark(self) -> bool:
        return self.brightness < C.Q_DARK_BRIGHTNESS

    @property
    def is_low_contrast(self) -> bool:
        return self.contrast < C.Q_LOW_CONTRAST

    @property
    def is_noisy(self) -> bool:
        return self.noise > C.Q_NOISY

    @property
    def is_blurry(self) -> bool:
        return self.sharpness < C.Q_BLURRY

    def to_dict(self) -> dict:
        d = {k: round(float(v), 2) for k, v in asdict(self).items()}
        d["flags"] = [n for n in ("dark", "low_contrast", "noisy", "blurry")
                      if getattr(self, f"is_{n}")]
        return d


def assess(rgb: np.ndarray) -> Quality:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    p5, p95 = np.percentile(gray, [5, 95])

    s = 640.0 / max(h, w)
    small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    sharp = float(cv2.Laplacian(small, cv2.CV_64F).var())

    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    resp = cv2.filter2D(gray.astype(np.float32), -1, k)[1:-1, 1:-1]
    noise = float(np.sqrt(np.pi / 2) * np.abs(resp).sum() / (6.0 * (w - 2) * (h - 2)))

    return Quality(brightness=float(gray.mean()), contrast=float(p95 - p5),
                   sharpness=sharp, noise=noise)


def plan_steps(q: Quality) -> List[str]:
    """
    Chọn các bước tiền xử lý theo chất lượng ảnh, đúng thứ tự áp dụng.
    Thêm quy tắc mới ở đây khi đăng ký bước mới trong steps.py.
    """
    plan: List[str] = []
    if q.is_low_contrast:
        plan.append("levels")
    if q.is_dark:
        plan.append("gamma")
    if q.is_noisy:
        plan.append("denoise")
    plan.append("clahe")              # luôn dùng (hành vi cũ), tăng tương phản cục bộ
    if q.is_blurry:
        plan.append("sharpen")
    return plan
