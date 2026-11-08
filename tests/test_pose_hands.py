"""Test YOLO-pose -> tâm nắm tay: không cần weights, model giả."""
import numpy as np

from backend.pipeline import config as C, pose
from backend.pipeline.schema import Person


def _kp(elbow=(100, 100), wrist=(100, 200), wconf=0.9):
    kp = np.zeros((17, 3), np.float32)
    kp[7] = (*elbow, 0.9)
    kp[9] = (*wrist, wconf)
    return kp


def test_hand_point_extrapolates_from_elbow():
    pts = pose.hand_points(_kp())
    assert len(pts) == 1
    assert np.allclose(pts[0], (100, 235))


def test_no_pole_search_when_wrist_conf_low():
    assert pose.hand_points(_kp(wconf=0.1)) == []


def test_attach_hands_matches_by_iou():
    a = Person(idx=0, box=(0, 0, 100, 300), conf=0.9)
    b = Person(idx=1, box=(400, 0, 500, 300), conf=0.9)
    boxes = np.array([[400, 0, 500, 300], [0, 0, 100, 300]], np.float32)
    kps = np.stack([_kp((450, 100), (450, 200)), _kp((50, 100), (50, 200))])
    pose.attach_hands([a, b], boxes, kps)
    assert np.allclose(a.hands, [(50, 235)])
    assert np.allclose(b.hands, [(450, 235)])


def test_detect_hands_failure_keeps_pipeline(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no pose weights")
    monkeypatch.setattr(pose, "get_yolo", boom)
    p = Person(idx=0, box=(0, 0, 10, 10), conf=0.9)
    pose.detect_hands(np.zeros((20, 20, 3), np.uint8), [p], None)
    assert p.hands == []


def _scene_with_fakes(monkeypatch, pose_fails=True):
    from backend.pipeline import detect
    from backend.pipeline.trace import Trace
    from tests.test_detect_device import FakeModel
    monkeypatch.setattr(detect, "get_yolo", lambda w=None, device=None: FakeModel("m"))

    def boom(*a, **k):
        raise RuntimeError("no pose weights")
    monkeypatch.setattr(pose, "get_yolo", boom)
    return detect.detect_scene(np.zeros((40, 60, 3), np.uint8), preprocess="none", trace=Trace())


def test_pose_failure_is_noted_in_trace(monkeypatch):
    monkeypatch.setattr(C, "HELD_ENABLE", True)
    scene = _scene_with_fakes(monkeypatch)
    note = next(s for s in scene.trace.stages if s.key == "pose").note
    assert "lỗi" in note and "no pose weights" in note and "0 tay" not in note
    assert len(scene.persons) == 1 and scene.persons[0].hands == []


def test_held_disabled_has_no_pose_stage(monkeypatch):
    monkeypatch.setattr(C, "HELD_ENABLE", False)
    scene = _scene_with_fakes(monkeypatch)
    assert all(s.key != "pose" for s in scene.trace.stages)
