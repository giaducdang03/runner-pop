"""Test gộp gậy vào mask/box người và stage trace "held"."""
import numpy as np

from backend.pipeline import config as C, held
from backend.pipeline.poles import Pole
from backend.pipeline.schema import Person
from tests.test_pose_hands import _scene_with_fakes


def test_merge_pole_extends_mask_and_box():
    mask = np.zeros((200, 200), np.uint8)
    mask[20:100, 40:80] = 255
    p = Person(idx=0, box=(40, 20, 80, 100), conf=0.9, mask=mask)
    pole = Pole(top=(100, 60), tip=(100, 180), half_w=3.0, score=0.9, owner=0)
    held.merge_pole(p, pole, (200, 200, 3))
    assert p.mask[120, 100] == 255
    assert p.box[0] <= 40 and p.box[3] >= 180 and p.box[2] >= 103
    assert p.body_box == (40, 20, 80, 100)
    assert p.held == ["pole"] and p.poles == [pole]


def test_detect_scene_without_hands_unchanged(monkeypatch):
    monkeypatch.setattr(C, "HELD_ENABLE", False)
    base = _scene_with_fakes(monkeypatch).persons[0]
    monkeypatch.setattr(C, "HELD_ENABLE", True)
    scene = _scene_with_fakes(monkeypatch)
    p = scene.persons[0]
    assert p.box == base.box
    assert (p.mask is None) == (base.mask is None)
    if p.mask is not None:
        assert np.array_equal(p.mask, base.mask)
    assert "pose lỗi" in next(s for s in scene.trace.stages if s.key == "held").note


# --- vật COCO gần nắm tay -------------------------------------------------------
def _person_with_hand(hand=(60, 100)):
    mask = np.zeros((300, 300), np.uint8)
    mask[20:220, 40:80] = 255
    p = Person(idx=0, box=(40, 20, 80, 220), conf=0.9, mask=mask)      # body_h = 200
    if hand is not None:
        p.hands = [hand]
    return p


def _obj(box, conf=0.8, cls=39, name="bottle", shape=(300, 300)):
    m = np.zeros(shape, np.uint8)
    x1, y1, x2, y2 = box
    m[y1:y2, x1:x2] = 255
    return held.HeldObject(cls=cls, name=name, box=box, conf=conf, mask=m)


def test_object_near_hand_attached():
    p = _person_with_hand((60, 100))
    o = _obj((66, 95, 76, 115))            # cách tay 6 px = 0.03h
    assert held.attach_objects([p], [o]) == 1
    assert p.held == ["bottle"]
    assert p.mask[100, 70] == 255 and p.body_box == (40, 20, 80, 220)


def test_object_touching_shoulder_but_far_from_hands_not_attached():
    p = _person_with_hand((60, 200))
    o = _obj((80, 30, 95, 50))             # chạm vai, cách tay ~150 px = 0.75h
    assert held.attach_objects([p], [o]) == 0
    assert p.held == []


def test_object_touching_without_hands_attached():
    p = _person_with_hand(None)
    o = _obj((78, 100, 95, 120))           # chạm mép mask
    assert held.attach_objects([p], [o]) == 1


def test_low_conf_object_not_attached():
    p = _person_with_hand((60, 100))
    assert held.attach_objects([p], [_obj((66, 95, 76, 115), conf=0.12)]) == 0
    assert p.held == []


def test_object_goes_to_nearest_hand_person():
    a = _person_with_hand((60, 100))
    b = _person_with_hand((90, 100))
    b.idx = 1
    o = _obj((80, 95, 85, 105))
    assert held.attach_objects([a, b], [o]) == 1
    assert (a.held, b.held) == ([], ["bottle"])


def test_result_without_cls_treated_as_person():
    from backend.pipeline import detect
    from tests.test_detect_device import _Res
    res = _Res()
    assert not hasattr(res.boxes, "cls")
    assert len(detect._persons_from_result(res, np.zeros((40, 60, 3), np.uint8))) == 1
    assert detect._objects_from_result(res, np.zeros((40, 60, 3), np.uint8)) == []


def test_held_disabled(monkeypatch):
    from backend.pipeline import pose
    calls = []
    monkeypatch.setattr(C, "HELD_ENABLE", False)
    monkeypatch.setattr(pose, "detect_hands", lambda *a, **k: calls.append(1))
    from backend.pipeline import detect
    from tests.test_detect_device import FakeModel
    models = []
    monkeypatch.setattr(detect, "get_yolo", lambda w=None, device=None: FakeModel("m", calls=models))
    from backend.pipeline.trace import Trace
    detect.detect_scene(np.zeros((40, 60, 3), np.uint8), preprocess="none", trace=Trace())
    assert models[0][1]["classes"] == [C.PERSON_CLASS] and calls == []


def test_detect_scene_merges_object_near_hand(monkeypatch):
    from backend.pipeline import detect, pose
    from backend.pipeline.trace import Trace
    from tests.test_detect_device import FakeModel
    monkeypatch.setattr(C, "HELD_ENABLE", True)
    monkeypatch.setattr(C, "POLE_ENABLE", False)
    monkeypatch.setattr(pose, "detect_hands",
                        lambda img, persons, dev: persons[0].hands.append((28.0, 20.0)))
    monkeypatch.setattr(detect, "get_yolo", lambda w=None, device=None: FakeModel("m"))
    fake_obj = held.HeldObject(39, "bottle", (30, 15, 34, 25), 0.9, np.zeros((40, 60), np.uint8))
    monkeypatch.setattr(detect, "_objects_from_result", lambda res, img: [fake_obj])
    scene = detect.detect_scene(np.zeros((40, 60, 3), np.uint8), preprocess="none", trace=Trace())
    assert scene.persons[0].held == ["bottle"]
    assert next(s for s in scene.trace.stages if s.key == "held").note.startswith("0 gậy, 1 vật")


def _fake_touching_object(monkeypatch):
    from backend.pipeline import detect
    obj = held.HeldObject(39, "bottle", (28, 10, 40, 20), 0.9, None)   # chạm mép phải mask người
    monkeypatch.setattr(detect, "_objects_from_result", lambda res, img: [obj])


def test_pose_failure_does_not_merge_touching_object(monkeypatch):
    # pose lỗi (get_yolo của pose ném) -> chạy như cũ: không gộp dù vật chạm mask người
    monkeypatch.setattr(C, "HELD_ENABLE", False)
    _fake_touching_object(monkeypatch)
    base = _scene_with_fakes(monkeypatch).persons[0]
    monkeypatch.setattr(C, "HELD_ENABLE", True)
    scene = _scene_with_fakes(monkeypatch)
    p = scene.persons[0]
    assert p.box == base.box and p.held == [] and p.body_box is None
    assert np.array_equal(p.mask, base.mask)
    note = next(s for s in scene.trace.stages if s.key == "held").note
    assert "pose lỗi" in note


def test_pole_disabled_skips_find_poles_but_merges_objects(monkeypatch):
    from backend.pipeline import detect, pose, poles
    from backend.pipeline.trace import Trace
    from tests.test_detect_device import FakeModel
    monkeypatch.setattr(C, "HELD_ENABLE", True)
    monkeypatch.setattr(C, "POLE_ENABLE", False)
    monkeypatch.setattr(pose, "detect_hands",
                        lambda img, persons, dev: persons[0].hands.append((28.0, 20.0)))
    calls = []
    monkeypatch.setattr(poles, "find_poles", lambda *a, **k: calls.append(1) or [])
    monkeypatch.setattr(detect, "get_yolo", lambda w=None, device=None: FakeModel("m"))
    fake_obj = held.HeldObject(39, "bottle", (30, 15, 34, 25), 0.9, np.zeros((40, 60), np.uint8))
    monkeypatch.setattr(detect, "_objects_from_result", lambda res, img: [fake_obj])
    scene = detect.detect_scene(np.zeros((40, 60, 3), np.uint8), preprocess="none", trace=Trace())
    p = scene.persons[0]
    assert calls == [] and p.poles == [] and p.held == ["bottle"] and scene.pole_candidates == {}
    assert "tắt dò gậy" in next(s for s in scene.trace.stages if s.key == "held").note
