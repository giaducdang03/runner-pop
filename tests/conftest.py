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
