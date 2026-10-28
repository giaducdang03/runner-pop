"""Chọn phương pháp đo nét laplacian | freq | hybrid (FR-121, FR-122)."""
from __future__ import annotations

import numpy as np

from backend.pipeline import config as C
from backend.pipeline import focusmap as fm
from backend.pipeline.focusmap import sharpness_map


def _rgb():
    rng = np.random.default_rng(0)
    rgb = np.full((120, 200, 3), 128, np.uint8)
    rgb[:, :100] = rng.integers(0, 255, (120, 100, 3))
    return rgb


def test_default_method_is_laplacian():
    assert C.FOCUS_METHOD == "laplacian"
    m, name = sharpness_map(_rgb())
    assert name == "laplacian" and m.shape == (120, 200) and m.dtype == np.float32


def test_freq_method_selected_by_config(monkeypatch):
    monkeypatch.setattr(C, "FOCUS_METHOD", "freq")
    assert sharpness_map(_rgb())[1] == "freq"
    assert sharpness_map(_rgb(), "hybrid")[1] == "hybrid"       # tham số ghi đè cấu hình


def test_unknown_method_falls_back_to_laplacian():
    assert sharpness_map(_rgb(), "bogus")[1] == "laplacian"


def test_hybrid_is_weighted_mean_of_both():
    rgb = _rgb()
    h, _ = sharpness_map(rgb, "hybrid")
    f, _ = sharpness_map(rgb, "freq")
    l, _ = sharpness_map(rgb, "laplacian")
    w = C.FOCUS_HYBRID_W
    assert np.allclose(h, w * f + (1 - w) * l, atol=1e-5)


def test_noisy_quality_enables_noise_suppression(monkeypatch):
    seen = {}
    real = fm.freq_map
    monkeypatch.setattr(fm, "freq_map", lambda rgb, **kw: seen.update(kw) or real(rgb, **kw))
    sharpness_map(_rgb(), "freq", {"flags": ["noisy"], "noise": 7.5})
    assert seen["noise_sigma"] == 7.5 and seen["drop_high"] is True
    seen.clear()
    sharpness_map(_rgb(), "freq", {"flags": [], "noise": 7.5})
    assert not seen


def test_focus_engine_still_imports_focus_energy():
    from backend.pipeline.segment.focus import focus_energy
    assert focus_energy is fm.focus_energy
