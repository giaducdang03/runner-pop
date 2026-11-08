"""Kiểu dữ liệu dùng chung giữa các bước của pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .cache import StageCache
from .trace import Trace

if TYPE_CHECKING:
    from .poles import Pole


@dataclass
class Person:
    idx: int
    box: Tuple[int, int, int, int]       # x1, y1, x2, y2 (toạ độ ảnh làm việc)
    conf: float
    mask: Optional[np.ndarray] = field(default=None, repr=False)  # uint8 0/255
    body_box: Optional[Tuple[int, int, int, int]] = None   # box thân người trước khi gộp vật cầm tay; None = trùng box
    hands: List[Tuple[float, float]] = field(default_factory=list)   # tâm nắm tay (toạ độ ảnh làm việc)
    held: List[str] = field(default_factory=list)          # nhãn vật đã gộp: "pole" hoặc tên lớp COCO
    poles: List["Pole"] = field(default_factory=list)      # gậy đã gộp vào mask

    @property
    def foot_y(self) -> int:
        """y của chân thân người (không tính vật cầm tay thò xuống thấp hơn)."""
        return (self.body_box or self.box)[3]

    @property
    def body_h(self) -> int:
        """Chiều cao thân người."""
        b = self.body_box or self.box
        return b[3] - b[1]

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
    device: Dict[str, Any] = field(default_factory=dict)  # thiết bị chạy YOLO (Device.to_dict())
    pole_candidates: Dict[int, List["Pole"]] = field(default_factory=dict)  # idx người -> gậy ứng viên (chưa gộp)

    def others(self, person: Person) -> List[Person]:
        return [p for p in self.persons if p is not person]


@dataclass
class SegParams:
    """Tham số người dùng chỉnh được cho giai đoạn tách nền."""
    band_frac: float = 0.02      # bề rộng vùng unknown / kích thước người
    focus_frac: float = 0.3      # ngưỡng độ nét (engine "focus")
    trace: Trace = field(default_factory=Trace, repr=False)   # engine ghi các bước vào đây
    cache: Optional[StageCache] = field(default=None, repr=False)  # cache trung gian của session
    device_mode: str = "auto"    # auto | cpu | gpu - chọn theo từng request

    def cached(self, key: tuple, build: Callable[[], Any]) -> Any:
        """Dùng cache nếu có, không thì chỉ gọi build()."""
        return self.cache.get_or(key, build) if self.cache is not None else build()


@dataclass
class Cutout:
    """Kết quả nặng - tính 1 lần cho mỗi (ảnh, người được chọn)."""
    image: np.ndarray        # float32 HxWx3, RGB, 0..1  (ảnh gốc đã resize)
    alpha: np.ndarray        # float32 HxW, 0..1
    foreground: np.ndarray   # float32 HxWx3, RGB, 0..1  (đã khử nhiễm màu viền)
    background: np.ndarray   # float32 HxWx3, RGB, 0..1  (nền thật ước lượng, không lẫn màu người)
    trimap: np.ndarray       # uint8 HxW (0 / 128 / 255) - để debug
    trace: Optional[Trace] = field(default=None, repr=False)
    filtered: Dict[tuple, Any] = field(default_factory=dict, repr=False)  # cache nền đã lọc theo (bộ lọc, màu)
