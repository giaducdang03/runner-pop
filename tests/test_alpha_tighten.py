"""Thu hẹp đuôi alpha để vành viền mềm không loá/nhoè (giữ màu gốc sát viền)."""
from __future__ import annotations

import numpy as np

from backend.pipeline import config as C
from backend.pipeline import matting
from backend.pipeline.segment import build_cutout
from tests.test_engine_roi import _synthetic


def test_tighten_alpha_endpoints_and_tails():
    a = np.array([0.0, 0.5 * C.ALPHA_LO, C.ALPHA_LO, 0.5, C.ALPHA_HI, 0.5 * (1 + C.ALPHA_HI), 1.0], np.float32)
    t = matting.tighten_alpha(a)
    assert t.dtype == np.float32
    assert t[0] == 0.0 and t[1] == 0.0 and t[2] == 0.0       # đuôi thấp -> 0
    assert t[4] == 1.0 and t[5] == 1.0 and t[6] == 1.0       # đuôi cao -> 1
    assert 0.0 < t[3] < 1.0                                  # giữa vẫn mềm
    assert np.all(np.diff(t) >= 0)                            # đơn điệu


def test_tighten_alpha_is_identity_for_binary_alpha():
    a = np.array([[0.0, 1.0], [1.0, 0.0]], np.float32)
    assert np.array_equal(matting.tighten_alpha(a), a)


def test_cutout_alpha_has_no_faint_tail(make_scene):
    scene = _synthetic(make_scene, (240, 320), (120, 60, 190, 200))
    c = build_cutout(scene, scene.persons[0], "cf")
    a = c.alpha
    assert not ((a > 0) & (a < 0.5 * C.ALPHA_LO)).any()
    assert not ((a > 1 - 0.5 * (1 - C.ALPHA_HI)) & (a < 1)).any()
