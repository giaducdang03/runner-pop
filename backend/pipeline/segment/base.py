"""
Engine tách nền = hàm  (scene, person, params) -> Cutout.

Thêm engine mới: tạo file trong segment/, đăng ký bằng

    @ENGINES.register("ten", label="Mô tả hiện trên giao diện")
    def run(scene: Scene, person: Person, params: SegParams) -> Cutout: ...

rồi import file đó trong segment/__init__.py. Có thể ghép các khối dùng chung:
masks.py (xử lý mask), matting.py (trimap/matting/khử nhiễm màu, finish_cutout).
"""
from __future__ import annotations

import numpy as np

from ..masks import box_to_mask
from ..registry import Registry
from ..schema import Person, Scene

ENGINES = Registry("engine")


def person_init_mask(scene: Scene, person: Person) -> np.ndarray:
    """Mask khởi tạo của người: mask YOLO, hoặc elip trong box nếu nhập box thủ công."""
    if person.mask is not None:
        return person.mask
    return box_to_mask(person.box, scene.rgb.shape)


def to_float(rgb: np.ndarray) -> np.ndarray:
    return rgb.astype(np.float32) / 255.0
