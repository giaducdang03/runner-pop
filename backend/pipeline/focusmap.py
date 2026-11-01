"""
Bản đồ độ nét dùng chung cho engine `focus` và chọn người tự động.

    laplacian_map   năng lượng Laplacian mịn (cách cũ, mặc định)
    freq_map        tỉ lệ năng lượng dải cao / tổng AC của DCT 16x16 trên khối chồng 50%
    sharpness_map   chọn phương pháp theo C.FOCUS_METHOD (laplacian | freq | hybrid)

Mọi bản đồ là float32 HxW trong 0..1 (đã chuẩn hoá bằng p99 của chính nó).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple

import cv2
import numpy as np

from . import config as C


# --- năng lượng Laplacian (chuyển từ segment/focus.py) -----------------------
def focus_energy(rgb: np.ndarray):
    """
    Trả về (E, R):
      E = năng lượng chi tiết mịn cục bộ (cao = nét, thấp = mờ/bokeh)
      R = tỉ lệ mịn/thô: cạnh nét có R cao bất kể độ tương phản,
          cạnh bị bokeh làm nhoè (dù tương phản mạnh) có R thấp.
    """
    g0 = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sigma = max(3.0, 0.006 * max(rgb.shape[:2]))
    g1 = cv2.GaussianBlur(g0, (0, 0), 1.0)               # giảm nhiễu hạt
    fine = cv2.GaussianBlur(np.abs(cv2.Laplacian(g1, cv2.CV_32F, ksize=3)), (0, 0), sigma)
    g3 = cv2.GaussianBlur(g0, (0, 0), 3.0)
    coarse = cv2.GaussianBlur(np.abs(cv2.Laplacian(g3, cv2.CV_32F, ksize=3)), (0, 0), sigma)
    return fine, fine / (coarse + 1e-3)


def _norm_p99(m: np.ndarray) -> np.ndarray:
    """Chuẩn hoá về 0..1 bằng p99 của chính bản đồ; bản đồ phẳng (p99 ~ 0) -> toàn 0."""
    m = np.asarray(m, np.float32)
    p = float(np.percentile(m, 99))
    if not np.isfinite(p) or p <= 1e-6:
        return np.zeros_like(m)
    return np.clip(m / p, 0.0, 1.0).astype(np.float32)


def laplacian_map(rgb: np.ndarray) -> np.ndarray:
    return _norm_p99(focus_energy(rgb)[0])


# --- DCT theo khối ------------------------------------------------------------
@lru_cache(maxsize=8)
def dct_basis(n: int) -> np.ndarray:
    """Ma trận DCT-II trực chuẩn n×n: coef = D @ X @ D.T (khớp cv2.dct)."""
    k = np.arange(n, dtype=np.float64)[:, None]
    i = np.arange(n, dtype=np.float64)[None, :]
    D = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2.0 / n)
    D[0, :] = np.sqrt(1.0 / n)
    return D.astype(np.float32)


@lru_cache(maxsize=8)
def _band_masks(n: int, drop_high: bool):
    """Vector 0/1 (n*n) cho: AC, dải cao (u+v >= n); drop_high bỏ u+v >= 2n-4. Kèm số hệ số."""
    u, v = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    s = (u + v).reshape(-1)
    ac = np.ones(n * n, bool)
    ac[0] = False
    high = s >= n
    if drop_high:
        keep = s < 2 * n - 4
        ac &= keep
        high &= keep
    ac_f, high_f = ac.astype(np.float32), (high & ac).astype(np.float32)
    return ac_f, high_f, float(ac_f.sum()), float(high_f.sum())


def dct_sharpness(gray_f32: np.ndarray, block: int = C.FOCUS_BLOCK, stride: int = C.FOCUS_STRIDE,
                  drop_high: bool = False, noise_sigma: float = 0.0) -> np.ndarray:
    """Độ nét trên LƯỚI KHỐI (gy×gx, float32, 0..1). `gray_f32` thang 0..255.

    Độ nét khối = log1p(E_cao / E_AC) sau khi trừ nền nhiễu `noise_sigma² × số hệ số` của từng
    dải (kẹp ≥ 0); khối không có năng lượng AC -> 0. Ảnh nhỏ hơn khối được đệm biên."""
    g = np.asarray(gray_f32, np.float32)
    h, w = g.shape
    ph, pw = max(0, block - h), max(0, block - w)
    if ph or pw:
        g = np.pad(g, ((0, ph), (0, pw)), mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(g, (block, block))[::stride, ::stride]
    gy, gx = win.shape[:2]
    D = dct_basis(block)
    coef = np.matmul(np.matmul(D, win), D.T)                 # (gy, gx, block, block)
    e = (coef * coef).reshape(gy, gx, block * block)
    ac_m, high_m, n_ac, n_high = _band_masks(block, bool(drop_high))
    e_ac, e_high = e @ ac_m, e @ high_m
    if noise_sigma > 0:
        n2 = float(noise_sigma) ** 2
        e_ac = np.maximum(e_ac - n2 * n_ac, 0.0)
        e_high = np.maximum(e_high - n2 * n_high, 0.0)
    ok = e_ac > 1e-2                                          # dưới ngưỡng = phẳng (sai số float)
    score = np.where(ok, np.log1p(e_high / (e_ac + 1e-6)), 0.0).astype(np.float32)
    return _norm_p99(score)


def freq_map(rgb: np.ndarray, drop_high: bool = False, noise_sigma: float = 0.0) -> np.ndarray:
    """Bản đồ nét tần số cho cả ảnh: xám -> thu về cạnh dài FOCUS_MAX_SIDE -> DCT theo khối ->
    làm mịn Gaussian -> resize về đúng kích thước ảnh. float32 HxW, 0..1."""
    h, w = rgb.shape[:2]
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    s = min(1.0, C.FOCUS_MAX_SIDE / max(h, w))
    if s < 1.0:
        g = cv2.resize(g, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    gh, gw = g.shape
    b, st = C.FOCUS_BLOCK, C.FOCUS_STRIDE
    # đệm để ô lưới i nằm quanh điểm ảnh i*stride (khối căn giữa)
    ny, nx = -(-gh // st), -(-gw // st)
    half = b // 2
    g = np.pad(g, ((half, max(0, (ny - 1) * st + b - gh - half)),
                   (half, max(0, (nx - 1) * st + b - gw - half))), mode="edge")
    grid = dct_sharpness(g, b, st, drop_high=drop_high, noise_sigma=noise_sigma)[:ny, :nx]
    grid = cv2.GaussianBlur(grid, (0, 0), 1.0)
    m = cv2.resize(grid, (nx * st, ny * st), interpolation=cv2.INTER_LINEAR)[:gh, :gw]
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(m, 0.0, 1.0).astype(np.float32)


# --- chọn phương pháp ---------------------------------------------------------
def sharpness_map(rgb: np.ndarray, method: Optional[str] = None,
                  quality: Optional[dict] = None) -> Tuple[np.ndarray, str]:
    """Bản đồ nét theo `method` (None -> C.FOCUS_METHOD); trả (bản đồ, tên phương pháp đã dùng).
    Ảnh nhiều nhiễu / tối (cờ trong quality["flags"]) -> phép đo tần số bỏ dải quá cao và trừ nền nhiễu."""
    m = (method or C.FOCUS_METHOD or "laplacian").strip().lower()
    if m not in ("laplacian", "freq", "hybrid"):
        m = "laplacian"
    if m == "laplacian":
        return laplacian_map(rgb), m
    q = quality or {}
    noisy = bool({"noisy", "dark"} & set(q.get("flags", [])))
    kw = {"drop_high": True, "noise_sigma": float(q.get("noise", 0.0))} if noisy else {}
    f = freq_map(rgb, **kw)
    if m == "freq":
        return f, m
    w = C.FOCUS_HYBRID_W
    return (w * f + (1.0 - w) * laplacian_map(rgb)).astype(np.float32), m


# --- độ tin cậy ------------------------------------------------------------------
def focus_confidence(smap: np.ndarray, person_mask: np.ndarray) -> float:
    """median(nét người) / median(nét nền); nền = ngoài mask người nở 5% cạnh dài.
    < C.FOCUS_CONF_MIN: ảnh không có phông mờ rõ. Ảnh phẳng / không có nền -> 1.0."""
    pm = np.asarray(person_mask)
    pm = (pm > 127) if pm.dtype == np.uint8 else pm.astype(bool)
    if not pm.any():
        return 1.0
    k = max(3, int(0.05 * max(pm.shape)) | 1)
    grown = cv2.dilate(pm.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    bg = ~grown
    if not bg.any():
        return 1.0
    p, b = float(np.median(smap[pm])), float(np.median(smap[bg]))
    if p <= 0.0 and b <= 0.0:
        return 1.0
    return float(min(p / max(b, 1e-3 * max(p, 1e-6)), 99.0))
