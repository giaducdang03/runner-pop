"""Kiểu dữ liệu dùng chung giữa các bước của pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .trace import Trace


@dataclass
class Person:
    idx: int
    box: Tuple[int, int, int, int]       # x1, y1, x2, y2 (toạ độ ảnh làm việc)
    conf: float
    mask: Optional[np.ndarray] = field(default=None, repr=False)  # uint8 0/255

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.box
        return max(0, x2 - x1) * max(0, y2 - y1)


@dataclass
class Scene:
    """Kết quả của giai đoạn phát hiện cho một ảnh - đầu vào của mọi engine tách nền."""
    rgb: np.ndarray                                     # uint8 HxWx3 - ảnh làm việc (không bị chỉnh màu)
    persons: List[Person] = field(default_factory=list)
    det_rgb: Optional[np.ndarray] = field(default=None, repr=False)  # ảnh đã tiền xử lý, chỉ để detect
    quality: Dict[str, Any] = field(default_factory=dict)
    preprocess: List[str] = field(default_factory=list)  # các bước tiền xử lý đã áp dụng
    attempt: int = 0                                     # lần thử detect thành công (0 = lần đầu)
    suggested: int = 0                                   # người được tự chọn (chế độ auto)
    trace: Trace = field(default_factory=Trace, repr=False)  # các bước đã chạy, để demo

    def others(self, person: Person) -> List[Person]:
        return [p for p in self.persons if p is not person]


@dataclass
class SegParams:
    """Tham số người dùng chỉnh được cho giai đoạn tách nền."""
    band_frac: float = 0.02      # bề rộng vùng unknown / kích thước người
    focus_frac: float = 0.3      # ngưỡng độ nét (engine "focus")
    trace: Trace = field(default_factory=Trace, repr=False)   # engine ghi các bước vào đây


@dataclass
class Cutout:
    """Kết quả nặng - tính 1 lần cho mỗi (ảnh, người được chọn)."""
    image: np.ndarray        # float32 HxWx3, RGB, 0..1  (ảnh gốc đã resize)
    alpha: np.ndarray        # float32 HxW, 0..1
    foreground: np.ndarray   # float32 HxWx3, RGB, 0..1  (đã khử nhiễm màu viền)
    background: np.ndarray   # float32 HxWx3, RGB, 0..1  (nền thật ước lượng, không lẫn màu người)
    trimap: np.ndarray       # uint8 HxW (0 / 128 / 255) - để debug
    trace: Optional[Trace] = field(default=None, repr=False)
