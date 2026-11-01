"""Bộ lọc áp lên nền (sau khi tách người). Mỗi bộ lọc: (bg, valid, *, color) -> ảnh float32 cùng kích thước.

`valid` = pixel nền tin cậy (α < 1). Phần thân người trong `bg` vẫn mang màu người nên các bộ lọc
lấy mẫu lân cận chỉ tính từ pixel hợp lệ (trung bình chuẩn hoá), tránh lem màu người ra nền.
"""
from __future__ import annotations

import re
from typing import Callable, Tuple

import cv2
import numpy as np

from .compose import to_gray3
from .registry import Registry
from .schema import Cutout

BG_FILTERS = Registry("bgfilter")

_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
_EPS = 1e-3


def parse_color(text: str) -> Tuple[float, float, float]:
    """'#RRGGBB' -> (r, g, b) trong 0..1; sai định dạng -> ValueError."""
    if not isinstance(text, str) or not _COLOR_RE.match(text):
        raise ValueError(f"Màu phải có dạng #RRGGBB, nhận được '{text}'.")
    return tuple(int(text[i:i + 2], 16) / 255.0 for i in (1, 3, 5))


def _masked(bg: np.ndarray, valid: np.ndarray, op: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Trung bình chuẩn hoá: op(bg·w) / op(w); nơi không có pixel hợp lệ lân cận giữ nguyên bg."""
    w = valid.astype(np.float32)
    num = op(np.ascontiguousarray(bg * w[..., None]))
    den = op(w)[..., None]
    return np.where(den > _EPS, num / np.maximum(den, _EPS), bg).astype(np.float32)


def _side(bg: np.ndarray) -> int:
    return max(bg.shape[:2])


@BG_FILTERS.register("none", "Ảnh gốc")
def none(bg, valid, **_):
    return bg


@BG_FILTERS.register("bw", "Nền đen trắng")
def bw(bg, valid, **_):
    return to_gray3(bg).astype(np.float32)


@BG_FILTERS.register("blur", "Xóa phông")
def blur(bg, valid, **_):
    sigma = max(2.0, 0.012 * _side(bg))
    return _masked(bg, valid, lambda a: cv2.GaussianBlur(a, (0, 0), sigma))


@BG_FILTERS.register("bokeh", "Bokeh")
def bokeh(bg, valid, **_):
    r = max(3, int(round(0.012 * _side(bg))))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    k = ((yy ** 2 + xx ** 2) <= r * r).astype(np.float32)
    k /= k.sum()
    return _masked(bg, valid, lambda a: cv2.filter2D(a, -1, k, borderType=cv2.BORDER_REFLECT))


@BG_FILTERS.register("mosaic", "Mosaic")
def mosaic(bg, valid, **_):
    h, w = bg.shape[:2]
    b = max(8, int(round(0.02 * _side(bg))))
    size = (max(1, -(-w // b)), max(1, -(-h // b)))                  # (cols, rows), làm tròn lên
    wt = valid.astype(np.float32)
    num = cv2.resize(bg * wt[..., None], size, interpolation=cv2.INTER_AREA)
    den = cv2.resize(wt, size, interpolation=cv2.INTER_AREA)[..., None]
    small = cv2.resize(bg, size, interpolation=cv2.INTER_AREA)
    small = np.where(den > _EPS, num / np.maximum(den, _EPS), small).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


@BG_FILTERS.register("sepia", "Sepia")
def sepia(bg, valid, **_):
    m = np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]], np.float32)
    return np.clip(bg @ m.T, 0.0, 1.0).astype(np.float32)


@BG_FILTERS.register("warm", "Tông ấm")
def warm(bg, valid, **_):
    return np.clip(bg * np.array([1.10, 1.00, 0.88], np.float32), 0.0, 1.0)


@BG_FILTERS.register("cool", "Tông lạnh")
def cool(bg, valid, **_):
    return np.clip(bg * np.array([0.88, 1.00, 1.12], np.float32), 0.0, 1.0)


@BG_FILTERS.register("sketch", "Phác họa")
def sketch(bg, valid, **_):
    gray = to_gray3(bg).astype(np.float32)
    sigma = max(2.0, 0.006 * _side(bg))
    inv = _masked(1.0 - gray, valid, lambda a: cv2.GaussianBlur(a, (0, 0), sigma))
    return np.clip(gray / np.maximum(1.0 - inv, _EPS), 0.0, 1.0).astype(np.float32)


@BG_FILTERS.register("vignette", "Tối góc")
def vignette(bg, valid, **_):
    h, w = bg.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = (((xx - (w - 1) / 2) / max(w / 2, 1)) ** 2 + ((yy - (h - 1) / 2) / max(h / 2, 1)) ** 2) / 2.0
    return np.clip(bg * (1.0 - 0.6 * np.clip(r2, 0.0, 1.0))[..., None], 0.0, 1.0).astype(np.float32)


@BG_FILTERS.register("solid", "Màu trơn")
def solid(bg, valid, *, color=(0.5, 0.5, 0.5), **_):
    return np.broadcast_to(np.asarray(color, np.float32), bg.shape).copy()


def apply_bg_filter(c: Cutout, name: str, color: str = "#808080") -> np.ndarray:
    """Nền đã lọc ở cường độ đầy đủ; cache trong cutout theo (name, color). Tên lạ -> KeyError, màu sai -> ValueError."""
    key = (name, color)
    if key not in c.filtered:
        fn = BG_FILTERS.get(name)
        c.filtered[key] = np.clip(fn(c.background, c.alpha < 0.999, color=parse_color(color)), 0.0, 1.0).astype(np.float32)
    return c.filtered[key]
