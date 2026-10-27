"""Test hàm thuần của công cụ đo (không cần model, không cần ảnh thật)."""
import numpy as np
import pytest

from tools.bench import summarize
from tools.eval_quality import iou, sad


def test_iou_and_sad_identical_maps():
    a = np.zeros((20, 20), np.float32)
    a[5:15, 5:15] = 255
    assert iou(a, a) == 1.0 and sad(a, a) == 0.0


def test_iou_partial_overlap_and_empty():
    a = np.zeros((10, 10), np.float32); a[:, :5] = 255
    b = np.zeros((10, 10), np.float32); b[:, :10] = 255
    assert iou(a, b) == pytest.approx(0.5)
    z = np.zeros((4, 4), np.float32)
    assert iou(z, z) == 1.0


def test_sad_matting_convention():
    a = np.zeros((4, 4), np.float32)
    b = np.full((4, 4), 255, np.float32)
    assert sad(a, b) == pytest.approx(16.0)   # 16 px lệch 255 -> 16


def test_median_p95_drops_warmup_runs():
    r = summarize([100, 90, 10, 11, 12, 13], drop=2)
    assert r["median"] == 11.5
    assert r["n"] == 4
    assert r["p95"] == pytest.approx(np.percentile([10, 11, 12, 13], 95))


def test_summarize_keeps_data_if_drop_too_large():
    assert summarize([5.0], drop=2)["median"] == 5.0
