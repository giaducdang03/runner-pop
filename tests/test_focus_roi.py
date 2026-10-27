"""Engine focus trên ROI + stage `roi` của trace ở cả ba họ engine + dòng "Vùng xử lý" của PDF (FR-90, FR-96)."""
from __future__ import annotations

import numpy as np
import pytest

from backend.pipeline.cache import StageCache
from backend.pipeline.roi import full_roi, person_roi, roi_stage
from backend.pipeline.schema import Person, SegParams
from backend.pipeline.segment import build_cutout
from backend.pipeline.segment import birefnet as bf_mod
from backend.pipeline.segment import focus as focus_mod
from backend.pipeline.trace import Trace
from backend.report import roi_note


def _scene(make_scene, shape=(240, 320), box=(120, 60, 190, 200)):
    """Nền mượt (bokeh) + người có hoạ tiết nhiễu nét; mask YOLO = hình chữ nhật."""
    h, w = shape
    m = np.zeros((h, w), np.uint8)
    m[box[1]:box[3], box[0]:box[2]] = 255
    scene = make_scene(shape, [(box, m)])
    gx = np.linspace(60, 160, w, dtype=np.float32)[None, :]
    rgb = np.stack([np.broadcast_to(gx, (h, w))] * 3, axis=-1).copy()
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (box[3] - box[1], box[2] - box[0], 3))
    rgb[box[1]:box[3], box[0]:box[2]] = noise
    scene.rgb = rgb.astype(np.uint8)
    return scene


def _roi_stage(c):
    return next(s for s in c.trace.stages if s.key == "roi")


def test_focus_engine_returns_full_shape_cutout(make_scene):
    scene = _scene(make_scene)
    c = build_cutout(scene, scene.persons[0], "focus")
    assert c.alpha.shape == scene.rgb.shape[:2]
    assert c.foreground.shape == scene.rgb.shape and c.trimap.shape == scene.rgb.shape[:2]
    roi = person_roi(scene, scene.persons[0])
    assert not roi.is_full
    assert c.alpha[0, 0] == 0.0 and c.alpha[-1, -1] == 0.0


def test_focus_trace_has_roi_stage_with_size_note(make_scene):
    scene = _scene(make_scene)
    c = build_cutout(scene, scene.persons[0], "focus")
    st = _roi_stage(c)
    assert "làm việc" in st.note and st.image is not None
    keys = [s.key for s in c.trace.stages]
    assert keys.index("roi") == keys.index("init_mask") + 1


def test_focus_mask_reused_when_only_band_changes(monkeypatch, make_scene):
    scene = _scene(make_scene)
    calls = []
    real = focus_mod.focus_mask
    monkeypatch.setattr(focus_mod, "focus_mask", lambda *a, **k: calls.append(a[0].shape) or real(*a, **k))
    cache = StageCache()
    p = scene.persons[0]
    c1 = build_cutout(scene, p, "focus", SegParams(band_frac=0.02, cache=cache))
    c2 = build_cutout(scene, p, "focus", SegParams(band_frac=0.08, cache=cache))
    assert len(calls) == 1
    roi = person_roi(scene, p)
    assert calls[0][:2] == roi.work_size[::-1]       # chạy trên ảnh ROI
    # cache hit vẫn ghi stage roi và đủ bước
    assert _roi_stage(c2).note == _roi_stage(c1).note
    assert "focus_mask" in [s.key for s in c2.trace.stages]
    # đổi ngưỡng focus -> tính lại
    build_cutout(scene, p, "focus", SegParams(focus_frac=0.5, cache=cache))
    assert len(calls) == 2


def test_focus_cache_key_includes_person_box(monkeypatch, make_scene):
    scene = _scene(make_scene)
    cache = StageCache()
    calls = []
    real = focus_mod.focus_mask
    monkeypatch.setattr(focus_mod, "focus_mask", lambda *a, **k: calls.append(1) or real(*a, **k))
    b1, b2 = (120, 60, 190, 200), (118, 58, 192, 202)
    build_cutout(scene, Person(idx=-1, box=b1, conf=1.0, mask=None), "focus", SegParams(cache=cache))
    build_cutout(scene, Person(idx=-1, box=b2, conf=1.0, mask=None), "focus", SegParams(cache=cache))
    assert len(calls) == 2


def test_focus_mask_handles_empty_outside_region(make_scene):
    """ROI ôm sát người -> vùng 'outside' rỗng: không crash, sàn nhiễu = 0."""
    scene = _scene(make_scene, shape=(100, 80), box=(0, 0, 80, 100))
    p = scene.persons[0]
    fm = focus_mod.focus_mask(scene.rgb, p, p.mask, 0.3, [])
    assert fm.shape == (100, 80) and fm.dtype == np.uint8
    assert (fm[p.mask > 127] == 255).all()


def test_focus_mask_small_guard_scales_with_scale(make_scene):
    """Lõi nhỏ ở scale thấp không bị coi là 'quá ít điểm' khi so với 50 px ở scale 1."""
    scene = _scene(make_scene, shape=(60, 60), box=(10, 10, 22, 22))
    p = scene.persons[0]
    fm = focus_mod.focus_mask(scene.rgb, p, p.mask, 0.3, [], scale=0.5)
    assert fm.shape == (60, 60)


# ---- stage `roi` dùng chung cho cả ba họ engine (FR-96) ----

def test_roi_stage_helper_draws_box_and_notes():
    rgb = np.full((200, 400, 3), 100, np.uint8)
    roi = full_roi(rgb.shape)
    t = Trace()
    roi_stage(t, rgb, roi)
    st = t.stages[0]
    assert st.key == "roi" and st.label == "Vùng xử lý (ROI) và độ phân giải làm việc"
    assert st.note == "ROI 0,0-400,200 (100% ảnh) · làm việc 400×200px"
    assert st.image


def test_cf_trace_has_roi_stage_even_on_cache_hit(make_scene):
    scene = _scene(make_scene)
    p = scene.persons[0]
    cache = StageCache()
    roi = person_roi(scene, p)
    w, h = roi.work_size
    for band in (0.02, 0.08):                        # lần 2 = cache hit của gcmask
        c = build_cutout(scene, p, "cf", SegParams(band_frac=band, cache=cache))
        st = _roi_stage(c)
        assert f"{roi.x1},{roi.y1}-{roi.x2},{roi.y2}" in st.note
        assert f"làm việc {w}×{h}px" in st.note and "% ảnh" in st.note
        keys = [s.key for s in c.trace.stages]
        if "init_mask" in keys:
            assert keys.index("roi") == keys.index("init_mask") + 1
        assert keys.index("roi") < keys.index("trimap")


def test_birefnet_trace_has_roi_stage(monkeypatch, make_scene):
    scene = _scene(make_scene)
    monkeypatch.setattr(bf_mod, "birefnet_alpha", lambda rgb: np.ones(rgb.shape[:2], np.float32))
    c = build_cutout(scene, scene.persons[0], "birefnet", SegParams(cache=StageCache()))
    st = _roi_stage(c)
    assert "làm việc" in st.note and st.image is not None
    keys = [s.key for s in c.trace.stages]
    assert keys.index("roi") == keys.index("init_mask") + 1


# ---- report ----

def test_roi_note_reads_roi_stage_or_dash():
    t = Trace()
    assert roi_note(t.stages) == "-"
    roi_stage(t, np.zeros((50, 60, 3), np.uint8), full_roi((50, 60)))
    assert roi_note(t.stages) == t.stages[0].note
    assert roi_note(None) == "-"
