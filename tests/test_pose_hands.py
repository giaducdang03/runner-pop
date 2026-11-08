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
