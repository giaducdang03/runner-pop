"""ROI quanh người được chọn: engine chỉ xử lý vùng này rồi dán lại ảnh đầy đủ (FR-90)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from . import config as C
from .schema import Person, Scene


@dataclass(frozen=True)
class Roi:
    """Hình chữ nhật [x1,x2) x [y1,y2) trong ảnh làm việc đầy đủ + hệ số thu nhỏ khi xử lý."""
    x1: int
    y1: int
    x2: int
    y2: int
    scale: float
    full_shape: Tuple[int, int]          # (H, W) của ảnh làm việc đầy đủ

    @property
    def full_size(self) -> Tuple[int, int]:
        """(w, h) của ROI ở độ phân giải gốc."""
        return (self.x2 - self.x1, self.y2 - self.y1)

    @property
    def work_size(self) -> Tuple[int, int]:
        """(w, h) của ROI sau khi nhân `scale` (kích thước engine thực sự chạy)."""
        w, h = self.full_size
        return (max(1, round(w * self.scale)), max(1, round(h * self.scale)))

    @property
    def is_full(self) -> bool:
        h, w = self.full_shape
        return (self.x1, self.y1, self.x2, self.y2) == (0, 0, w, h)


def _overlaps(a, b) -> bool:
    """Hai box giao nhau với diện tích > 0."""
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def person_roi(scene: Scene, person: Person, margin: float = C.ROI_MARGIN,
               min_px: int = C.ROI_MIN_PX, max_side: int = C.WORK_MAX_SIDE) -> Roi:
    """ROI = box người nở `max(min_px, margin * cạnh dài)` mỗi phía, gộp box những người
    khác có box chồng lên box người này, rồi cắt vào ảnh. Chỉ cần box (không cần mask)."""
    h, w = scene.rgb.shape[:2]
    bx1, by1, bx2, by2 = person.box
    pad = max(float(min_px), margin * max(bx2 - bx1, by2 - by1))
    x1, y1, x2, y2 = bx1 - pad, by1 - pad, bx2 + pad, by2 + pad
    for o in scene.others(person):
        if _overlaps(person.box, o.box):
            x1, y1 = min(x1, o.box[0]), min(y1, o.box[1])
            x2, y2 = max(x2, o.box[2]), max(y2, o.box[3])
    x1, y1 = max(0, math.floor(x1)), max(0, math.floor(y1))
    x2, y2 = min(w, math.ceil(x2)), min(h, math.ceil(y2))
    # scale được tính ở Task A2; ở đây giữ 1.0
    return Roi(x1, y1, x2, y2, 1.0, (h, w))


def full_roi(shape, max_side: Optional[int] = None) -> Roi:
    """ROI phủ cả ảnh. `max_side=None` -> scale 1.0."""
    h, w = int(shape[0]), int(shape[1])
    scale = 1.0
    if max_side is not None and max(h, w) > max_side:
        scale = max_side / max(h, w)
    return Roi(0, 0, w, h, scale, (h, w))


def active_roi(scene: Scene, person: Person) -> Roi:
    """ROI mọi engine dùng: toàn ảnh khi FULLRES bật, ngược lại ROI quanh người."""
    if C.FULLRES:
        return full_roi(scene.rgb.shape)
    return person_roi(scene, person)
