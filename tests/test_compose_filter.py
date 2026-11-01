import numpy as np
import pytest

from backend.pipeline.bgfilter import BG_FILTERS
from backend.pipeline.compose import compose, to_gray3


def _legacy(c, s):
    gray = to_gray3(c.background)
    bg = (1.0 - s) * c.background + s * gray
    a = c.alpha[..., None]
    return (np.clip(a * c.foreground + (1.0 - a) * bg, 0, 1) * 255.0 + 0.5).astype(np.uint8)


@pytest.mark.parametrize("s", [0.0, 0.3, 1.0])
def test_default_matches_legacy_formula(make_cutout, s):
    c = make_cutout()
    assert np.array_equal(compose(c, s), _legacy(c, s))


def test_intensity_zero_is_original_background_for_every_filter(make_cutout):
    c = make_cutout()
    ref = compose(c, 0.0, "bw")
    for name in BG_FILTERS.names():
        assert np.array_equal(compose(c, 0.0, name), ref), name


def test_solid_filter_changes_background_only(make_cutout):
    c = make_cutout()
    out = compose(c, 1.0, "solid", "#0000ff")
    assert tuple(out[2, 2]) == (0, 0, 255)                     # nền ngoài người
    h, w = c.alpha.shape
    assert tuple(out[h // 2, w // 2]) == (255, 0, 0)           # người giữ màu


def test_cache_is_per_cutout(make_cutout):
    a, b = make_cutout(), make_cutout()
    compose(a, 1.0, "solid", "#ff0000")
    assert b.filtered == {}
