"""Test engine cf/knn chạy trên ROI ở độ phân giải làm việc + chế độ FULLRES (FR-90, FR-91, FR-95)."""
from __future__ import annotations

import numpy as np
import pytest

from backend.pipeline import config as C
from backend.pipeline.cache import StageCache
from backend.pipeline.roi import full_roi, person_roi
from backend.pipeline.schema import SegParams
from backend.pipeline.segment import build_cutout
from backend.pipeline.segment import grabcut as gc_mod


def _synthetic(make_scene, shape, box, others=()):
    """Nền gradient + hình chữ nhật màu đỏ tại `box` (mask YOLO = đúng hình chữ nhật)."""
    h, w = shape

    def rect_mask(b):
        m = np.zeros((h, w), np.uint8)
        m[b[1]:b[3], b[0]:b[2]] = 255
        return m

    people = [(box, rect_mask(box))] + [(b, rect_mask(b)) for b in others]
    scene = make_scene(shape, people)
    gx = np.linspace(40, 200, w, dtype=np.float32)[None, :]
    gy = np.linspace(30, 220, h, dtype=np.float32)[:, None]
    rgb = np.stack([np.broadcast_to(gx, (h, w)), np.broadcast_to(gy, (h, w)),
                    np.full((h, w), 120, np.float32)], axis=-1)
    for i, (b, _) in enumerate(people):
        rgb[b[1]:b[3], b[0]:b[2]] = (220, 30, 30) if i == 0 else (30, 200, 40)
    scene.rgb = rgb.astype(np.uint8)
    return scene


@pytest.fixture
def small_case(make_scene):
    scene = _synthetic(make_scene, (240, 320), (120, 60, 190, 200))
    return scene, build_cutout(scene, scene.persons[0], "cf")


def test_cutout_has_full_image_shapes(small_case):
    scene, c = small_case
    for arr in (c.alpha, c.trimap):
        assert arr.shape == scene.rgb.shape[:2]
    for arr in (c.foreground, c.background):
        assert arr.shape == scene.rgb.shape
    assert c.alpha.dtype == np.float32 and c.trimap.dtype == np.uint8


def test_alpha_is_zero_outside_roi(small_case):
    scene, c = small_case
    roi = person_roi(scene, scene.persons[0])
    assert not roi.is_full
    assert c.alpha[0, 0] == 0.0 and c.alpha[-1, -1] == 0.0
    outside = np.ones(c.alpha.shape, bool)
    outside[roi.y1:roi.y2, roi.x1:roi.x2] = False
    assert (c.alpha[outside] == 0).all() and (c.trimap[outside] == 0).all()
    # ngoài ROI nền = ảnh gốc (FR-36)
    assert np.allclose(c.background[outside], c.image[outside])
    # người được giữ
    assert c.alpha[130, 155] > 0.9


def test_foreground_background_in_unit_range(small_case):
    _, c = small_case
    for arr in (c.foreground, c.background):
        assert arr.min() >= 0.0 and arr.max() <= 1.0


def test_trimap_unknown_band_not_empty_for_tiny_person(make_scene):   # Review Focus #3
    scene = _synthetic(make_scene, (240, 320), (150, 100, 165, 130))     # người ~30 px
    roi = person_roi(scene, scene.persons[0])
    assert roi.scale == 1.0 and roi.work_size == roi.full_size        # không phóng to
    c = build_cutout(scene, scene.persons[0], "knn")
    assert (c.trimap == 128).sum() > 0
    assert c.alpha.shape == scene.rgb.shape[:2]
    assert c.alpha[115, 157] > 0.5


def test_large_person_runs_downscaled_and_pastes_back(make_scene):
    shape, box = (900, 1200), (250, 60, 950, 860)
    scene = _synthetic(make_scene, shape, box)
    params = SegParams(cache=StageCache())
    c = build_cutout(scene, scene.persons[0], "knn", params)
    roi = params.cache.get_or(("roi", 0, box), lambda: None)
    assert roi is not None and roi.scale < 1.0 and max(roi.work_size) == C.WORK_MAX_SIDE
    gcmask = params.cache.get_or(("gcmask", 0, box), lambda: None)
    assert gcmask.shape == roi.work_size[::-1]
    assert c.alpha.shape == shape and c.trimap.shape == shape
    assert c.alpha[460, 600] > 0.9 and c.alpha[460, roi.x1 + 2] < 0.1


def test_fullres_flag_processes_whole_image(monkeypatch, make_scene):
    monkeypatch.setattr(C, "FULLRES", True)
    scene = _synthetic(make_scene, (240, 320), (120, 60, 190, 200))
    person = scene.persons[0]
    params = SegParams(cache=StageCache())
    c = build_cutout(scene, person, "cf", params)
    roi = params.cache.get_or(("roi", person.idx, person.box), lambda: None)
    H, W = scene.rgb.shape[:2]
    assert roi == full_roi(scene.rgb.shape)
    assert roi.is_full and roi.work_size == (W, H)
    assert c.alpha.shape == (H, W)


def test_grabcut_mask_reused_when_only_band_changes(monkeypatch, make_scene):
    scene = _synthetic(make_scene, (240, 320), (120, 60, 190, 200))
    calls = []
    real = gc_mod.refine_mask_grabcut

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(gc_mod, "refine_mask_grabcut", counting)
    cache = StageCache()
    c1 = build_cutout(scene, scene.persons[0], "cf", SegParams(band_frac=0.02, cache=cache))
    c2 = build_cutout(scene, scene.persons[0], "cf", SegParams(band_frac=0.08, cache=cache))
    assert len(calls) == 1
    assert (c2.trimap == 128).sum() > (c1.trimap == 128).sum()


def test_manual_box_cache_keys_do_not_collide(make_scene):
    # hai box thủ công (idx = -1) khác nhau không được dùng chung ROI / mask GrabCut
    from backend.pipeline.schema import Person
    scene = _synthetic(make_scene, (240, 320), (120, 60, 190, 200))
    cache = StageCache()
    a = Person(idx=-1, box=(120, 60, 190, 200), conf=1.0)
    b = Person(idx=-1, box=(20, 20, 70, 120), conf=1.0)
    scene.persons = [a]
    ca = build_cutout(scene, a, "cf", SegParams(cache=cache))
    scene.persons = [b]
    cb = build_cutout(scene, b, "cf", SegParams(cache=cache))
    assert ca.alpha[130, 155] > 0.5 and cb.alpha[130, 155] == 0.0


def test_other_nearer_person_is_excluded(make_scene):
    # người khác đứng gần hơn (chân thấp hơn), box chồng lên -> phần của họ không vào alpha
    scene = _synthetic(make_scene, (300, 400), (120, 40, 200, 220), others=[(180, 60, 260, 280)])
    c = build_cutout(scene, scene.persons[0], "cf")
    assert c.alpha[150, 230] < 0.1          # trong người khác, ngoài người được chọn
    assert c.alpha[130, 150] > 0.9


def test_cutout_from_mask_signature(make_scene):
    from backend.pipeline.segment.base import to_float
    scene = _synthetic(make_scene, (240, 320), (120, 60, 190, 200))
    roi = person_roi(scene, scene.persons[0])
    mask_work = roi.to_work(scene.persons[0].mask)
    c = gc_mod.cutout_from_mask(to_float(scene.rgb), mask_work, roi, SegParams(), "cf")
    assert c.alpha.shape == scene.rgb.shape[:2]
