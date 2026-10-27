"""Test khối gate dùng chung + engine BiRefNet chạy trên ROI (FR-31 sửa đổi, FR-90)."""
from __future__ import annotations

import numpy as np

from backend.pipeline.cache import StageCache
from backend.pipeline.roi import person_roi
from backend.pipeline.schema import SegParams
from backend.pipeline.segment import build_cutout
from backend.pipeline.segment import birefnet as bf_mod
from backend.pipeline.segment.gate import gate_to_person


def _case():
    h, w = 200, 200
    init = np.zeros((h, w), np.uint8)
    init[50:150, 80:120] = 255                       # thân
    box = (80, 50, 120, 150)
    alpha = np.ones((h, w), np.float32)               # BiRefNet "thấy" cả ảnh
    return alpha, init, box


def test_gate_zeroes_alpha_outside_dilated_person_mask():
    alpha, init, box = _case()
    gated = gate_to_person(alpha, init, box)
    assert gated[:, :30].max() == 0.0 and gated[:, 170:].max() == 0.0
    assert gated[100, 100] > 0.9


def test_gate_drops_detached_blob():
    alpha, init, box = _case()
    alpha[:] = 0
    alpha[50:150, 80:120] = 1.0
    alpha[160:190, 150:190] = 1.0                    # mảnh rời xa người, ngoài gate -> loại
    gated = gate_to_person(alpha, init, box)
    assert gated[160:190, 150:190].max() == 0.0
    # mảnh rời nằm trong vùng nở của gate nhưng không dính mask YOLO cũng bị loại
    alpha2 = np.zeros_like(alpha)
    alpha2[50:150, 80:120] = 1.0
    alpha2[60:70, 128:134] = 1.0                     # cách thân ~8px, trong gate (r=5.. nở)
    gated2 = gate_to_person(alpha2, init, box)
    assert gated2[100, 100] > 0.9


def test_gate_zeroes_excluded_region():
    alpha, init, box = _case()
    exclude = np.zeros(alpha.shape, bool)
    exclude[50:150, 100:120] = True
    gated = gate_to_person(alpha, init, box, exclude)
    assert gated[exclude].max() == 0.0
    assert gated[100, 85] > 0.5


def test_gate_keeps_thin_limb_inside_mask():
    alpha, init, box = _case()
    init[100:104, 40:80] = 255                       # tay mảnh 4px nằm trong mask YOLO
    box = (40, 50, 120, 150)
    alpha[:] = 0
    alpha[50:150, 80:120] = 1.0
    alpha[100:104, 40:80] = 1.0
    gated = gate_to_person(alpha, init, box)
    limb = gated[101:103, 45:75]
    assert limb.min() > 0.5


def _scene(make_scene):
    h, w = 240, 320
    box = (120, 60, 190, 200)
    m = np.zeros((h, w), np.uint8)
    m[60:200, 120:190] = 255
    scene = make_scene((h, w), [(box, m)])
    rgb = np.full((h, w, 3), 100, np.uint8)
    rgb[60:200, 120:190] = (220, 30, 30)
    scene.rgb = rgb
    return scene, box


def test_birefnet_engine_runs_on_roi_and_caches_alpha(monkeypatch, make_scene):
    scene, box = _scene(make_scene)
    person = scene.persons[0]
    roi = person_roi(scene, person)
    assert not roi.is_full
    calls = []

    def fake(rgb):
        calls.append(rgb.shape)
        a = np.zeros(rgb.shape[:2], np.float32)
        a[:] = 1.0                                   # "thấy" cả ROI; gate phải cắt lại
        return a

    monkeypatch.setattr(bf_mod, "birefnet_alpha", fake)
    cache = StageCache()
    c1 = build_cutout(scene, person, "birefnet", SegParams(band_frac=0.02, cache=cache))
    c2 = build_cutout(scene, person, "birefnet", SegParams(band_frac=0.08, cache=cache))
    assert len(calls) == 1
    assert calls[0][:2] == roi.work_size[::-1]       # chạy trên ảnh ROI, không phải toàn ảnh
    H, W = scene.rgb.shape[:2]
    for c in (c1, c2):
        assert c.alpha.shape == (H, W) and c.trimap.shape == (H, W)
        assert c.foreground.shape == scene.rgb.shape and c.background.shape == scene.rgb.shape
        outside = np.ones((H, W), bool)
        outside[roi.y1:roi.y2, roi.x1:roi.x2] = False
        assert (c.alpha[outside] == 0).all()
        assert c.alpha[130, 155] > 0.9
