from __future__ import annotations

import numpy as np

from backend.pipeline import config as C
from backend.pipeline.roi import Roi, active_roi, full_roi, person_roi
from backend.pipeline.schema import Person


def _xy(r):
    return (r.x1, r.y1, r.x2, r.y2)


def test_roi_margin_12_percent(make_scene):
    scene = make_scene((1000, 1000), [((100, 100, 300, 500), None)])
    roi = person_roi(scene, scene.persons[0])
    assert _xy(roi) == (52, 52, 348, 548)


def test_roi_margin_floor_16px(make_scene):
    scene = make_scene((1000, 1000), [((400, 400, 430, 430), None)])
    roi = person_roi(scene, scene.persons[0])
    assert roi.x1 == 400 - 16 and roi.y2 == 430 + 16


def test_roi_clamped_to_image(make_scene):
    H, W = 300, 400
    for box in [(0, 0, 100, 100), (300, 200, 400, 300), (0, 100, 400, 300)]:
        scene = make_scene((H, W), [(box, None)])
        roi = person_roi(scene, scene.persons[0])
        assert roi.x1 >= 0 and roi.y1 >= 0 and roi.x2 <= W and roi.y2 <= H
        assert roi.full_shape == (H, W)


def test_roi_unions_overlapping_person_box(make_scene):
    other = (280, 200, 700, 600)
    scene = make_scene((1000, 1000), [((100, 100, 300, 500), None), (other, None)])
    roi = person_roi(scene, scene.persons[0])
    assert roi.x2 >= other[2] and roi.y2 >= other[3]


def test_roi_ignores_non_overlapping_person(make_scene):
    alone = make_scene((1000, 1000), [((100, 100, 300, 500), None)])
    roi_alone = _xy(person_roi(alone, alone.persons[0]))
    scene = make_scene((1000, 1000), [((100, 100, 300, 500), None),
                                      ((600, 600, 800, 900), None)])
    assert _xy(person_roi(scene, scene.persons[0])) == roi_alone


def test_roi_from_manual_box_without_mask(make_scene):
    scene = make_scene((1000, 1000), [])
    p = Person(idx=-1, box=(100, 100, 300, 500), conf=1.0, mask=None)
    roi = person_roi(scene, p)
    assert roi.full_size == (296, 496)
    assert roi.work_size == roi.full_size and not roi.is_full


def test_full_roi_covers_image_and_is_full():
    r = full_roi((240, 320, 3))
    assert r.is_full and r.scale == 1.0
    assert _xy(r) == (0, 0, 320, 240) and r.full_shape == (240, 320)


def test_active_roi_switches_on_fullres(make_scene, monkeypatch):
    scene = make_scene((1000, 1000), [((100, 100, 300, 500), None)])
    monkeypatch.setattr(C, "FULLRES", False)
    assert not active_roi(scene, scene.persons[0]).is_full
    monkeypatch.setattr(C, "FULLRES", True)
    assert active_roi(scene, scene.persons[0]).is_full


def test_roi_scope_all_includes_far_person(make_scene):
    """R16: scope="all" gộp cả người xa không chồng box; scope="person" thì không."""
    scene = make_scene((240, 480), [((100, 60, 170, 200), None), ((380, 60, 450, 200), None)])
    p = scene.persons[0]
    near = person_roi(scene, p)
    far = person_roi(scene, p, scope="all")
    assert near.x2 < 380 and far.x2 >= 450 and far.x1 == near.x1
    manual = Person(idx=-1, box=(100, 60, 170, 200), conf=1.0, mask=None)
    assert person_roi(scene, manual, scope="all").x2 >= 450
    assert active_roi(scene, p, scope="all") == far
