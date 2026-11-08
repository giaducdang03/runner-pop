"""Test dữ liệu người mở rộng (vật cầm tay), truyền qua ROI và occlusion theo chân thân người."""
from __future__ import annotations

import numpy as np

from backend.pipeline.occlusion import exclusion_mask
from backend.pipeline.roi import Roi
from backend.pipeline.schema import Person, Scene
from backend.pipeline.segment.base import box_to_work, work_scene


def test_foot_y_prefers_body_box():
    p = Person(idx=0, box=(0, 0, 10, 50), conf=1, body_box=(0, 0, 10, 40))
    assert p.foot_y == 40 and p.body_h == 40
    q = Person(idx=0, box=(0, 0, 10, 50), conf=1, body_box=None)
    assert q.foot_y == 50 and q.body_h == 50


def test_work_scene_keeps_body_box_hands_held():
    h, w = 200, 200
    roi = Roi(x1=20, y1=10, x2=180, y2=190, scale=0.5, full_shape=(h, w))
    body = (50, 40, 120, 160)
    hx, hy = 100.0, 90.0
    p = Person(idx=0, box=(40, 40, 130, 170), conf=1, body_box=body,
               hands=[(hx, hy)], held=["pole"])
    scene = Scene(rgb=np.zeros((h, w, 3), np.uint8), persons=[p])
    _, pw = work_scene(scene, p, roi)
    assert pw.body_box == box_to_work(body, roi)
    assert pw.hands[0] == ((hx - roi.x1) * 0.5, (hy - roi.y1) * 0.5)
    assert pw.held == ["pole"] and pw.poles == []


def test_occlusion_uses_body_foot_not_held_object():
    h, w = 220, 200
    ma = np.zeros((h, w), np.uint8)
    ma[20:190, 20:100] = 255
    mb = np.zeros((h, w), np.uint8)
    mb[30:152, 60:140] = 255
    a = Person(idx=0, box=(20, 20, 100, 190), conf=1, mask=ma, body_box=(20, 20, 100, 150))
    b = Person(idx=1, box=(60, 30, 140, 152), conf=1, mask=mb)
    scene = Scene(rgb=np.zeros((h, w, 3), np.uint8), persons=[a, b])
    excl = exclusion_mask(scene, a, a.mask)
    overlap = (ma > 127) & (mb > 127)
    assert overlap.any() and not (excl & overlap).any()
