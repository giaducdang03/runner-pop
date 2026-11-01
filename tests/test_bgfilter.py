import numpy as np
import pytest

from backend.pipeline.bgfilter import BG_FILTERS, apply_bg_filter, parse_color
from backend.pipeline.compose import to_gray3


def test_bw_matches_gray3(make_cutout):
    c = make_cutout()
    assert np.allclose(apply_bg_filter(c, "bw"), to_gray3(c.background))


def test_none_returns_background(make_cutout):
    c = make_cutout()
    assert np.array_equal(apply_bg_filter(c, "none"), c.background)


@pytest.mark.parametrize("name", ["blur", "bokeh", "mosaic"])
def test_blur_does_not_leak_person_color(make_cutout, name):
    c = make_cutout(200, 300)
    out = apply_bg_filter(c, name)
    y, x2 = 100, 2 * 300 // 3                       # hàng giữa, sát mép phải khối đỏ
    px = out[y, x2 + 3]
    assert px[0] < 0.02 and px[1] > 0.9


def test_all_filters_valid_output(make_cutout):
    c = make_cutout(101, 77)
    for name in BG_FILTERS.names():
        out = apply_bg_filter(c, name)
        assert out.dtype == np.float32 and out.shape == (101, 77, 3), name
        assert np.isfinite(out).all() and out.min() >= 0.0 and out.max() <= 1.0, name


def test_person_fills_frame_no_nan(make_cutout):
    c = make_cutout()
    c.alpha[:] = 1.0
    c.alpha[0, 0] = c.alpha[-1, -1] = 0.0
    for name in BG_FILTERS.names():
        assert np.isfinite(apply_bg_filter(c, name)).all(), name


def test_solid_uses_color(make_cutout):
    out = apply_bg_filter(make_cutout(), "solid", "#ff8000")
    assert np.allclose(out, (1.0, 128 / 255, 0.0), atol=1e-3)


@pytest.mark.parametrize("bad", ["red", "#12", "#GG0000", ""])
def test_parse_color_rejects_bad(bad):
    with pytest.raises(ValueError):
        parse_color(bad)


def test_apply_is_cached(make_cutout):
    c = make_cutout()
    assert apply_bg_filter(c, "blur") is apply_bg_filter(c, "blur")
    assert apply_bg_filter(c, "solid", "#ff0000") is not apply_bg_filter(c, "solid", "#00ff00")
