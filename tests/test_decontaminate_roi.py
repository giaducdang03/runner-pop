"""Test khử nhiễm màu chạy ở độ phân giải làm việc, F/B nâng bằng upsample_rgb (FR-93)."""
from __future__ import annotations

import numpy as np
import pytest

from backend.pipeline import matting
from backend.pipeline.roi import person_roi
from backend.pipeline.segment import build_cutout
from tests.test_engine_roi import _synthetic


@pytest.fixture
def case(make_scene):
    scene = _synthetic(make_scene, (240, 320), (120, 60, 190, 200))
    return scene, build_cutout(scene, scene.persons[0], "cf")


def test_foreground_equals_original_where_alpha_is_one(case):
    _, c = case
    m = c.alpha >= 1.0 - 1e-6
    assert m.any()
    assert np.allclose(c.foreground[m], c.image[m], atol=1e-6)


def test_background_equals_original_outside_roi(case):
    _, c = case
    assert np.allclose(c.background[0, 0], c.image[0, 0], atol=1e-6)
    assert np.allclose(c.background[c.alpha <= 0], c.image[c.alpha <= 0], atol=1e-6)


def test_decontaminate_runs_at_work_resolution(make_scene, monkeypatch):
    scene = _synthetic(make_scene, (240, 320), (120, 60, 190, 200))
    seen = []
    real = matting.decontaminate

    def spy(rgb, alpha):
        seen.append(rgb.shape)
        return real(rgb, alpha)

    monkeypatch.setattr(matting, "decontaminate", spy)
    build_cutout(scene, scene.persons[0], "cf")
    roi = person_roi(scene, scene.persons[0])
    assert seen and seen[0][:2] == (roi.work_size[1], roi.work_size[0])
    assert seen[0][:2] != scene.rgb.shape[:2]


def test_fb_full_shape_and_range(case):
    scene, c = case
    assert c.foreground.shape == scene.rgb.shape == c.background.shape
    for a in (c.foreground, c.background):
        assert a.min() >= 0.0 and a.max() <= 1.0
