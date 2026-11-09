"""Dò gậy trễ theo người được chọn (POLE_SCOPE=selected): chỉ quét tay người đó (+ người có tay sát gậy để phân xử)."""
import numpy as np
import pytest

from backend.pipeline import config as C
from backend.pipeline import held, poles
from backend.pipeline.poles import Pole
from backend.pipeline.schema import Person, Scene

pytestmark = pytest.mark.pole_scope

SHAPE = (100, 100)


def _person(idx, hands, box=(0, 0, 40, 100)):
    p = Person(idx=idx, box=box, conf=0.9, mask=None)
    p.hands = list(hands)
    return p


def _pole(owner, top=(20.0, 30.0), tip=(24.0, 90.0)):
    return Pole(top=top, tip=tip, half_w=1.0, score=0.9, owner=owner)


def test_detect_scene_defers_pole_scan(monkeypatch):
    from backend.pipeline import detect, pose
    from backend.pipeline.trace import Trace
    from tests.test_detect_device import FakeModel
    monkeypatch.setattr(C, "HELD_ENABLE", True)
    monkeypatch.setattr(C, "POLE_ENABLE", True)
    monkeypatch.setattr(pose, "detect_hands", lambda img, persons, dev: persons[0].hands.append((28.0, 20.0)))
    monkeypatch.setattr(detect, "get_yolo", lambda w=None, device=None: FakeModel("m"))
    monkeypatch.setattr(poles, "find_poles", lambda *a, **k: pytest.fail("không được dò lúc phát hiện"))
    sc = detect.detect_scene(np.zeros((40, 60, 3), np.uint8), preprocess="none", trace=Trace(), poles=True)
    assert sc.poles_lazy is True and sc.pole_candidates == {}
    assert "khi xử lý người được chọn" in next(s for s in sc.trace.stages if s.key == "held").note


def test_poles_off_is_never_lazy(monkeypatch):
    from backend.pipeline import detect, pose
    from backend.pipeline.trace import Trace
    from tests.test_detect_device import FakeModel
    monkeypatch.setattr(C, "HELD_ENABLE", True)
    monkeypatch.setattr(pose, "detect_hands", lambda img, persons, dev: None)
    monkeypatch.setattr(detect, "get_yolo", lambda w=None, device=None: FakeModel("m"))
    sc = detect.detect_scene(np.zeros((40, 60, 3), np.uint8), preprocess="none", trace=Trace(), poles=False)
    assert sc.poles_lazy is False


def test_apply_poles_scans_only_chosen_person_once(monkeypatch):
    a, b = _person(0, [(20, 30)]), _person(1, [(70, 30)], box=(50, 0, 90, 100))
    sc = Scene(rgb=np.zeros(SHAPE + (3,), np.uint8), persons=[a, b], poles_lazy=True)
    calls = []

    def fake(rgb, persons, person):
        calls.append(person.idx)
        return [_pole(person.idx)]
    monkeypatch.setattr(held, "find_poles_for", fake)
    assert held.apply_poles(sc, b, SHAPE) == 1
    assert calls == [1] and 0 not in sc.pole_candidates          # người kia không bị dò
    assert held.apply_poles(sc, b, SHAPE) == 0 and calls == [1]  # đã gộp: không dò / gộp lại
    assert held.apply_poles(sc, a, SHAPE) == 1 and calls == [1, 0]


def test_empty_scan_is_remembered(monkeypatch):
    a = _person(0, [(20, 30)])
    sc = Scene(rgb=np.zeros(SHAPE + (3,), np.uint8), persons=[a], poles_lazy=True)
    calls = []
    monkeypatch.setattr(held, "find_poles_for", lambda rgb, persons, person: calls.append(1) or [])
    assert held.apply_poles(sc, a, SHAPE) == 0 and sc.pole_candidates == {0: []}
    assert held.apply_poles(sc, a, SHAPE) == 0 and calls == [1]  # kết quả rỗng được nhớ, không dò lại


def test_non_lazy_scene_does_not_scan(monkeypatch):
    a = _person(0, [(20, 30)])
    sc = Scene(rgb=np.zeros(SHAPE + (3,), np.uint8), persons=[a])
    monkeypatch.setattr(held, "find_poles_for", lambda *x: pytest.fail("scene không lazy"))
    assert held.apply_poles(sc, a, SHAPE) == 0


def test_find_poles_for_single_scan_when_no_rival(monkeypatch):
    a, b = _person(0, [(20, 30)]), _person(1, [(80, 30)], box=(60, 0, 100, 100))
    seen = []

    def fake(rgb, persons):
        seen.append([p.idx for p in persons])
        return [_pole(0)]
    monkeypatch.setattr(poles, "find_poles", fake)
    got = poles.find_poles_for(np.zeros(SHAPE + (3,), np.uint8), [a, b], a)
    assert seen == [[0]] and [g.owner for g in got] == [0]


def test_find_poles_for_adds_rival_whose_hand_is_on_the_pole(monkeypatch):
    a = _person(0, [(20, 30)])
    b = _person(1, [(22, 60)], box=(10, 0, 50, 100))               # tay của b nằm ngay trên trục gậy dò được cho a
    seen = []

    def fake(rgb, persons):
        seen.append([p.idx for p in persons])
        if len(persons) == 1:
            return [_pole(0)]
        return [_pole(1)]                                          # phân xử: gậy thuộc người b
    monkeypatch.setattr(poles, "find_poles", fake)
    got = poles.find_poles_for(np.zeros(SHAPE + (3,), np.uint8), [a, b], a)
    assert seen == [[0], [0, 1]] and got == []                     # gậy bị người kia lấy -> a không có gậy


def test_find_poles_for_no_poles_skips_rival_scan(monkeypatch):
    a, b = _person(0, [(20, 30)]), _person(1, [(22, 60)], box=(10, 0, 50, 100))
    seen = []
    monkeypatch.setattr(poles, "find_poles", lambda rgb, persons: seen.append(len(persons)) or [])
    assert poles.find_poles_for(np.zeros(SHAPE + (3,), np.uint8), [a, b], a) == [] and seen == [1]


def test_find_poles_for_never_raises(monkeypatch):
    a = _person(0, [(20, 30)])

    def boom(rgb, persons):
        raise RuntimeError("x")
    monkeypatch.setattr(poles, "find_poles", boom)
    assert poles.find_poles_for(np.zeros(SHAPE + (3,), np.uint8), [a], a) == []
