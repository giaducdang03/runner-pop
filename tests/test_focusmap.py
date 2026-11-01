"""Test bản đồ nét theo miền tần số DCT (FR-120, FR-122, FR-124)."""
from __future__ import annotations

import time

import cv2
import numpy as np

from backend.pipeline.focusmap import dct_basis, dct_sharpness, freq_map

H, W = 160, 320


def _noise(seed=0, shape=(H, W)):
    return np.random.default_rng(seed).uniform(0, 255, shape).astype(np.float32)


def test_dct_basis_matches_cv2_dct():
    blk = np.random.default_rng(1).uniform(0, 255, (16, 16)).astype(np.float32)
    D = dct_basis(16)
    assert np.allclose(D @ blk @ D.T, cv2.dct(blk), atol=1e-3)
    assert np.allclose(D @ D.T, np.eye(16), atol=1e-5)


def test_sharp_region_scores_higher_than_blurred_region():
    g = _noise()
    g[:, W // 2:] = cv2.GaussianBlur(g[:, W // 2:], (0, 0), 3.0)
    m = dct_sharpness(g)
    gx = m.shape[1]
    assert m[:, : gx // 2 - 2].mean() > 2 * m[:, gx // 2 + 2:].mean()


def test_flat_image_is_finite_and_zero():
    m = dct_sharpness(np.full((H, W), 117.0, np.float32))
    assert np.isfinite(m).all() and m.max() == 0.0


def test_output_shape_equals_input():
    rgb = np.random.default_rng(2).integers(0, 255, (123, 211, 3), dtype=np.uint8)
    m = freq_map(rgb)
    assert m.shape == rgb.shape[:2] and m.dtype == np.float32
    assert 0.0 <= m.min() and m.max() <= 1.0


def test_tiny_image_does_not_crash():
    assert freq_map(np.zeros((5, 7, 3), np.uint8)).shape == (5, 7)


def test_noise_sigma_suppresses_noise_only_region():
    rng = np.random.default_rng(3)
    base = np.full((H, W), 128.0, np.float32)
    base[:, : W // 2] = cv2.GaussianBlur(rng.uniform(0, 255, (H, W // 2)).astype(np.float32), (0, 0), 4.0)
    noisy = np.clip(base + rng.normal(0, 8.0, base.shape), 0, 255).astype(np.uint8)
    rgb = np.repeat(noisy[..., None], 3, axis=2)
    region = (slice(None), slice(W // 2 + 20, None))      # nửa phẳng, chỉ có nhiễu
    assert freq_map(rgb, noise_sigma=8.0)[region].mean() < freq_map(rgb)[region].mean()


def test_runs_under_150ms_at_640px():
    rgb = np.random.default_rng(4).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    freq_map(rgb)                                          # khởi động
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        freq_map(rgb)
        ts.append(time.perf_counter() - t0)
    assert sorted(ts)[2] < 0.150
