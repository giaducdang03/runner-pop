"""Trimap -> alpha matting -> khử nhiễm màu viền (pymatting, CPU)."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .roi import Roi, upsample_alpha, upsample_rgb
from .schema import Cutout
from .trace import Trace


def make_trimap(mask: np.ndarray, band_frac: float = 0.02) -> np.ndarray:
    """
    band_frac: bề rộng vùng 'chưa biết' so với cạnh dài của box người
    (≈2% => ~25 px với ảnh 1280 px).
    """
    ys, xs = np.nonzero(mask > 127)
    if len(xs) == 0:
        return np.zeros_like(mask)
    size = max(xs.max() - xs.min(), ys.max() - ys.min())
    r = max(3, int(round(band_frac * size)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    m = (mask > 127).astype(np.uint8)
    fg = cv2.erode(m, k)
    bg = 1 - cv2.dilate(m, k)
    tri = np.full(mask.shape, 128, np.uint8)
    tri[fg == 1] = 255
    tri[bg == 1] = 0
    return tri


def matte(rgb_f: np.ndarray, trimap: np.ndarray, method: str = "cf") -> np.ndarray:
    """
    Alpha matting trên CPU bằng pymatting.
    method: 'cf' (closed-form, chất lượng cao, chậm hơn) | 'knn' (nhanh hơn)
    Chỉ giải trong ROI quanh vùng unknown để tiết kiệm RAM/thời gian.
    """
    from pymatting import estimate_alpha_cf, estimate_alpha_knn

    h, w = trimap.shape
    ys, xs = np.nonzero(trimap > 0)            # FG hoặc unknown
    if len(xs) == 0:
        return np.zeros((h, w), np.float32)
    pad = 16
    x1, x2 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    y1, y2 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)

    img_roi = np.ascontiguousarray(rgb_f[y1:y2, x1:x2], dtype=np.float64)
    tri_roi = trimap[y1:y2, x1:x2].astype(np.float64) / 255.0
    tri_roi[(tri_roi > 0.4) & (tri_roi < 0.6)] = 0.5

    solver = estimate_alpha_knn if method == "knn" else estimate_alpha_cf
    try:
        a_roi = solver(img_roi, tri_roi)
    except Exception:
        # ichol của closed-form đôi khi không hội tụ -> thử KNN, cuối cùng là mask làm mềm
        try:
            a_roi = estimate_alpha_knn(img_roi, tri_roi)
        except Exception:
            a_roi = cv2.GaussianBlur((tri_roi >= 0.99).astype(np.float64), (0, 0), 2.0)

    alpha = np.zeros((h, w), np.float32)
    alpha[y1:y2, x1:x2] = np.clip(a_roi, 0, 1).astype(np.float32)
    alpha[trimap == 255] = 1.0
    alpha[trimap == 0] = 0.0
    return alpha


def decontaminate(rgb_f: np.ndarray, alpha: np.ndarray):
    """
    Ước lượng đồng thời màu foreground thật F và màu nền thật B ở pixel viền,
    sao cho  image ≈ alpha*F + (1-alpha)*B.
      - F sạch  -> không còn viền màu / quầng sáng của nền cũ
      - B sạch  -> phần nền sau viền mềm không lẫn độ sáng của người
    Trả về (F, B), cả hai float32 HxWx3.
    """
    from pymatting import estimate_foreground_ml
    h, w = alpha.shape
    ys, xs = np.nonzero(alpha > 0.001)
    if len(xs) == 0:
        return rgb_f.copy(), rgb_f.copy()
    pad = 8
    x1, x2 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    y1, y2 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)
    img_roi = np.ascontiguousarray(rgb_f[y1:y2, x1:x2], dtype=np.float64)
    a_roi = np.ascontiguousarray(alpha[y1:y2, x1:x2], dtype=np.float64)
    fg_roi, bg_roi = estimate_foreground_ml(img_roi, a_roi, return_background=True)
    fg, bg = rgb_f.copy(), rgb_f.copy()
    fg[y1:y2, x1:x2] = np.clip(fg_roi, 0, 1).astype(np.float32)
    bg[y1:y2, x1:x2] = np.clip(bg_roi, 0, 1).astype(np.float32)
    # ngoài vùng người alpha=0 -> nền = ảnh gốc (tránh sai số ước lượng)
    bg[alpha <= 0.0] = rgb_f[alpha <= 0.0]
    return fg, bg


def finish_cutout(rgb_f: np.ndarray, alpha_work: np.ndarray, trimap_work: np.ndarray,
                  roi: Roi, trace: Optional[Trace] = None) -> Cutout:
    """Bước cuối chung của mọi engine: nâng α/trimap từ độ phân giải làm việc của `roi`
    về ảnh đầy đủ, khử nhiễm màu + đóng gói.
    `rgb_f`: ảnh ĐẦY ĐỦ float32 0..1; `alpha_work`, `trimap_work`: kích thước `roi.work_size`.
    Ngoài ROI: α = 0, trimap = 0, B = ảnh gốc (FR-90, FR-36)."""
    alpha = np.clip(upsample_alpha(alpha_work, roi, rgb_f, unknown_work=(trimap_work == 128)),
                    0.0, 1.0)
    tri = trimap_work
    if tri.shape[::-1] != roi.full_size:
        tri = cv2.resize(tri, roi.full_size, interpolation=cv2.INTER_NEAREST)
    trimap = roi.paste(tri.astype(np.uint8), fill=0)
    with (trace or Trace()).stage("decontaminate", "Khử nhiễm màu viền (ước lượng F / B)",
                                  note="ảnh ≈ α·F + (1−α)·B, tính ở độ phân giải làm việc") as s:
        # khử nhiễm ở độ phân giải làm việc (FR-93), rồi nâng F/B bằng guided filter
        rgb_work = roi.to_work(rgb_f)
        fg_work, bg_work = decontaminate(rgb_work, alpha_work.astype(np.float32))
        fg = np.clip(upsample_rgb(fg_work, roi, rgb_f, base=rgb_f), 0.0, 1.0).astype(np.float32)
        bg = np.clip(upsample_rgb(bg_work, roi, rgb_f, base=rgb_f), 0.0, 1.0).astype(np.float32)
        solid = alpha >= 1.0 - 1e-6
        fg[solid] = rgb_f[solid]                 # α = 1 -> F chính là ảnh gốc
        clear = alpha <= 0.0
        bg[clear] = rgb_f[clear]                 # α = 0 (kể cả ngoài ROI) -> B là ảnh gốc
        a = alpha[..., None]
        s.show(a * fg + (1 - a) * 0.5)          # người tách khỏi nền, đặt trên nền xám
    return Cutout(image=rgb_f, alpha=alpha, foreground=fg, background=bg, trimap=trimap)
