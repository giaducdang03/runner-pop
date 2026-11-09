"""Gậy chỉ gộp cho người được xử lý (held.apply_poles trong build_cutout), không gộp lúc detect."""
import numpy as np

import backend.pipeline.segment as seg
from backend.pipeline import config as C, held
from backend.pipeline.poles import Pole
from backend.pipeline.schema import Cutout, Person, Scene
from backend.pipeline.segment import base, build_cutout
from tests.test_detect_device import _Res, _T

SHAPE = (200, 300, 3)


def _m(y1, y2, x1, x2):
    m = np.zeros(SHAPE[:2], np.uint8)
    m[y1:y2, x1:x2] = 255
    return m


def _scene():
    a = Person(idx=0, box=(20, 20, 60, 100), conf=0.9, mask=_m(20, 100, 20, 60))
    b = Person(idx=1, box=(150, 20, 190, 100), conf=0.9, mask=_m(20, 100, 150, 190))
    pa = [Pole(top=(60, 60), tip=(90, 180), half_w=2.0, score=0.9, owner=0),
          Pole(top=(20, 60), tip=(5, 170), half_w=2.0, score=0.9, owner=0)]
    pb = [Pole(top=(190, 60), tip=(230, 190), half_w=2.0, score=0.9, owner=1)]
    sc = Scene(rgb=np.zeros(SHAPE, np.uint8), persons=[a, b], pole_candidates={0: pa, 1: pb})
    return sc, a, b


class _FakeEngine:
    def __init__(self):
        self.boxes = []

    def __call__(self, scene, person, params):
        self.boxes.append(person.box)
        img = np.zeros(SHAPE, np.float32)
        return Cutout(image=img, alpha=np.zeros(SHAPE[:2], np.float32), foreground=img,
                      background=img, trimap=np.zeros(SHAPE[:2], np.uint8))


def _patch_engine(monkeypatch):
    eng = _FakeEngine()
    monkeypatch.setattr(base.ENGINES, "get", lambda m: eng)
    monkeypatch.setattr(seg, "overlay_poles", lambda c, s, p, t, d="auto": c)
    return eng


def test_apply_poles_only_for_given_person_and_idempotent():
    sc, a, b = _scene()
    bm, bbox = b.mask.copy(), b.box
    assert held.apply_poles(sc, a, SHAPE) == 2
    box_a = a.box
    assert a.held.count("pole") == 2 and len(a.poles) == 2 and box_a != (20, 20, 60, 100)
    assert held.apply_poles(sc, a, SHAPE) == 0
    assert a.box == box_a and a.held.count("pole") == 2
    assert b.box == bbox and np.array_equal(b.mask, bm) and b.poles == []


def test_apply_poles_noop_for_manual_box_person():
    sc, _, _ = _scene()
    manual = Person(idx=-1, box=(20, 20, 60, 100), conf=1.0)
    assert held.apply_poles(sc, manual, SHAPE) == 0
    clone = Person(idx=0, box=(20, 20, 60, 100), conf=1.0)    # cùng idx nhưng không phải đối tượng trong scene
    assert held.apply_poles(sc, clone, SHAPE) == 0 and clone.poles == []


def test_build_cutout_merges_only_processed_person_once(monkeypatch):
    sc, a, b = _scene()
    eng = _patch_engine(monkeypatch)
    bm, bbox = b.mask.copy(), b.box
    c1 = build_cutout(sc, a, "x")
    box_a = a.box
    c2 = build_cutout(sc, a, "x")
    assert eng.boxes == [box_a, box_a]            # engine thấy box đã gộp ngay lần đầu
    assert a.box == box_a and a.held.count("pole") == len(sc.pole_candidates[0]) == 2
    assert b.box == bbox and np.array_equal(b.mask, bm) and b.held == []
    assert [s.key for s in c1.trace.stages].count("held") == 1
    assert [s.key for s in c2.trace.stages].count("held") == 0     # lần sau không còn gì để gộp
    sc2, _, b2 = _scene()
    sc2.pole_candidates = {}
    assert all(s.key != "held" for s in build_cutout(sc2, b2, "x").trace.stages)


def test_build_cutout_manual_box_person_works(monkeypatch):
    sc, _, _ = _scene()
    _patch_engine(monkeypatch)
    manual = Person(idx=-1, box=(100, 30, 140, 120), conf=1.0)
    build_cutout(sc, manual, "x")
    assert manual.box == (100, 30, 140, 120) and manual.poles == []


def test_build_cutout_apply_failure_continues_without_poles(monkeypatch):
    sc, a, _ = _scene()
    _patch_engine(monkeypatch)
    monkeypatch.setattr(seg, "apply_poles", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    c = build_cutout(sc, a, "x")
    note = next(s for s in c.trace.stages if s.key == "held").note
    assert "boom" in note and a.poles == []


def _two_person_model():
    class M:
        def predict(self, img, **kw):
            r = _Res(2)
            r.boxes.xyxy = _T([[5, 5, 30, 35], [35, 5, 58, 35]])
            return [r]
    return M()


def _detect(monkeypatch, enabled):
    from backend.pipeline import detect, pose, poles
    from backend.pipeline.trace import Trace
    monkeypatch.setattr(C, "HELD_ENABLE", True)
    monkeypatch.setattr(C, "POLE_ENABLE", enabled)
    monkeypatch.setattr(pose, "detect_hands", lambda img, persons, dev: None)
    monkeypatch.setattr(detect, "get_yolo", lambda w=None, device=None: _two_person_model())
    calls = []

    def fake_find(rgb, persons):
        calls.append(1)
        return [Pole(top=(30, 20), tip=(50, 38), half_w=1.0, score=0.9, owner=0)]
    monkeypatch.setattr(poles, "find_poles", fake_find)
    return detect.detect_scene(np.zeros((40, 60, 3), np.uint8), preprocess="none", trace=Trace()), calls


def test_detect_scene_stores_candidates_without_merging(monkeypatch):
    sc, calls = _detect(monkeypatch, True)
    assert calls == [1] and len(sc.persons) == 2
    assert sc.pole_candidates == {0: [Pole(top=(30, 20), tip=(50, 38), half_w=1.0, score=0.9, owner=0)]}
    assert all(p.poles == [] and p.held == [] and p.body_box is None for p in sc.persons)
    assert sc.persons[0].box == (5, 5, 30, 35)
    assert "1 gậy ứng viên" in next(s for s in sc.trace.stages if s.key == "held").note


def test_detect_scene_pole_disabled_never_calls_find_poles(monkeypatch):
    sc, calls = _detect(monkeypatch, False)
    assert calls == [] and sc.pole_candidates == {}
