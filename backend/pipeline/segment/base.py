"""
Engine tách nền = hàm  (scene, person, params) -> Cutout.

Thêm engine mới: tạo file trong segment/, đăng ký bằng

    @ENGINES.register("ten", label="Mô tả hiện trên giao diện")
    def run(scene: Scene, person: Person, params: SegParams) -> Cutout: ...

rồi import file đó trong segment/__init__.py. Có thể ghép các khối dùng chung:
masks.py (xử lý mask), matting.py (trimap/matting/khử nhiễm màu, finish_cutout).
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from ..masks import box_to_mask
from ..registry import Registry
from ..roi import Roi
from ..schema import Person, Scene

ENGINES = Registry("engine")


def person_init_mask(scene: Scene, person: Person) -> np.ndarray:
    """Mask khởi tạo của người: mask YOLO, hoặc elip trong box nếu nhập box thủ công."""
    if person.mask is not None:
        return person.mask
    return box_to_mask(person.box, scene.rgb.shape)


def to_float(rgb: np.ndarray) -> np.ndarray:
    return rgb.astype(np.float32) / 255.0


def box_to_work(box: Tuple[int, int, int, int], roi: Roi) -> Tuple[int, int, int, int]:
    """Đổi box từ toạ độ ảnh đầy đủ sang hệ ROI ở độ phân giải làm việc (cắt vào ROI)."""
    ww, wh = roi.work_size
    x1, y1, x2, y2 = box

    def conv(v, o, hi):
        return int(min(hi, max(0, round((v - o) * roi.scale))))

    return (conv(x1, roi.x1, ww), conv(y1, roi.y1, wh), conv(x2, roi.x1, ww), conv(y2, roi.y1, wh))


def work_scene(scene: Scene, person: Person, roi: Roi) -> Tuple[Scene, Person]:
    """Scene thu về ROI làm việc: ảnh INTER_AREA, mask mọi người INTER_NEAREST, box đổi toạ độ.
    Dùng để tính mask khởi tạo và vùng loại trừ người khác (FR-37) ngay ở độ phân giải làm việc."""
    def conv(p: Person) -> Person:
        m = None if p.mask is None else roi.to_work(p.mask, cv2.INTER_NEAREST)
        return Person(idx=p.idx, box=box_to_work(p.box, roi), conf=p.conf, mask=m)

    person_w = conv(person)
    persons_w = [person_w if p is person else conv(p) for p in scene.persons]
    if not any(p is person_w for p in persons_w):   # box thủ công không nằm trong scene.persons
        persons_w.append(person_w)
    return Scene(rgb=roi.to_work(scene.rgb), persons=persons_w), person_w
