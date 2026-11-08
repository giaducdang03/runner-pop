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
    assert next(s for s in scene.trace.stages if s.key == "held").note == "0 gậy"
