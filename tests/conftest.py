"""Fixture dùng chung cho test: dựng Scene tổng hợp, không cần YOLO."""
from __future__ import annotations

import numpy as np
import pytest

from backend.pipeline.schema import Person, Scene


@pytest.fixture
def make_scene():
    """make_scene((H, W), [(box, mask_or_None), ...]) -> Scene"""
    def _make(shape, people):
        h, w = shape
        rgb = np.zeros((h, w, 3), np.uint8)
        persons = [Person(idx=i, box=tuple(box), conf=0.9, mask=mask)
                   for i, (box, mask) in enumerate(people)]
        return Scene(rgb=rgb, persons=persons)
    return _make


@pytest.fixture
def make_cutout():
    """make_cutout(h, w) -> Cutout tổng hợp: nền xanh lá, người là khối đỏ ở giữa (alpha = 1).
    background giữ màu người trong thân (giống ảnh thật: B chỉ được khử nhiễm ở vành)."""
    from backend.pipeline.schema import Cutout

    def _make(h=120, w=160):
        img = np.zeros((h, w, 3), np.float32)
        img[..., 1] = 1.0
        alpha = np.zeros((h, w), np.float32)
        y1, y2, x1, x2 = h // 4, 3 * h // 4, w // 3, 2 * w // 3
        img[y1:y2, x1:x2] = (1.0, 0.0, 0.0)
        alpha[y1:y2, x1:x2] = 1.0
        return Cutout(image=img.copy(), alpha=alpha, foreground=img.copy(), background=img.copy(),
                      trimap=np.where(alpha > 0, 255, 0).astype(np.uint8))
    return _make
