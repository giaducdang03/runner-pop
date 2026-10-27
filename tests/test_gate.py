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
    # box cao 400 px -> gate nở r = 16 px; blob nằm TRONG vùng nở nhưng cách thân ~7 px (không chạm mask YOLO)
    h, w = 500, 500
    init = np.zeros((h, w), np.uint8)
    init[50:450, 200:240] = 255
    box = (200, 50, 240, 450)
    alpha = np.zeros((h, w), np.float32)
    alpha[50:450, 200:240] = 1.0
    alpha[200:220, 247:252] = 1.0
    gated = gate_to_person(alpha, init, box)
    assert gated[100, 220] > 0.9                     # thân vẫn còn
    assert gated[200:220, 247:252].max() == 0.0      # mảnh rời bị loại bởi bộ lọc thành phần


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


def test_birefnet_engine_pastes_alpha_at_exact_position(monkeypatch, make_scene):
    scene, box = _scene(make_scene)
    person = scene.persons[0]
    roi = person_roi(scene, person)
    assert roi.scale == 1.0

    def fake(rgb):                                   # alpha đúng hình người (đỏ) trong ROI
        return (rgb[..., 0] > 200).astype(np.float32)

    monkeypatch.setattr(bf_mod, "birefnet_alpha", fake)
    c = build_cutout(scene, person, "birefnet", SegParams(cache=StageCache()))
    x1, y1, x2, y2 = box
    assert c.alpha[y1 - 1, 155] < 0.1 and c.alpha[y1 + 1, 155] > 0.9
    assert c.alpha[y2 - 2, 155] > 0.9 and c.alpha[y2 + 1, 155] < 0.1
    assert c.alpha[130, x1 - 1] < 0.1 and c.alpha[130, x1 + 1] > 0.9
    assert c.alpha[130, x2 - 2] > 0.9 and c.alpha[130, x2 + 1] < 0.1


def test_birefnet_engine_gate_cuts_alpha_inside_roi(monkeypatch, make_scene):
    scene, box = _scene(make_scene)
    person = scene.persons[0]
    roi = person_roi(scene, person)
    monkeypatch.setattr(bf_mod, "birefnet_alpha", lambda rgb: np.ones(rgb.shape[:2], np.float32))
    c = build_cutout(scene, person, "birefnet", SegParams(cache=StageCache()))
    # góc ROI (xa người hơn vùng nở của gate) phải về 0 dù model trả toàn 1
    assert c.alpha[roi.y1, roi.x1] == 0.0 and c.alpha[roi.y2 - 1, roi.x2 - 1] == 0.0
    assert c.alpha[130, 155] > 0.9
