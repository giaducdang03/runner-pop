"""Kiểm tra gậy bằng đoạn thẳng LSD: gậy thật có hai mép thẳng mà OpenCV LSD bắt được, còn đường
dò giả (chạy qua cỏ, hàng rào, mép chân) gần như không có đoạn LSD song song dọc trục.

Độ phủ ("clahe_cov") = tỉ lệ mẫu trên trục top -> tip (mỗi POLE_LSD_STEP px) nằm trong một đoạn LSD
song song với trục (lệch <= POLE_LSD_PAR_DEG) và cách trục <= half_w + POLE_LSD_MARGIN px. LSD chạy
trên ảnh xám đã cân bằng CLAHE.

Đo ở độ phân giải chuẩn: ảnh lớn hơn POLE_LSD_MAX_SIDE được thu nhỏ (INTER_AREA) về cạnh dài đó trước
khi chạy LSD, toạ độ / bề rộng gậy đổi theo, các tham số px (MARGIN, STEP) tính ở độ phân giải này.
Ngưỡng được chọn trên ảnh 1600 px; LSD không bất biến theo tỉ lệ (ảnh 1600 phóng lên 2560 làm gậy thật
của người áo đỏ trong photo_5394198 tụt từ ~0.6 xuống 0.14), nên giữ đúng miền đã đo, lại nhanh hơn.

LSD chạy trên TOÀN ảnh (một lần, lười - chỉ khi có ứng viên cần kiểm tra), ~120-180 ms mỗi ảnh 1600 px:
chạy trên vùng cắt quanh gậy nhanh hơn (~34 ms) nhưng ngưỡng NFA của LSD phụ thuộc kích thước ảnh nên
tập đoạn khác đi (độ phủ lệch tới 0.27, 17/106 gậy lệch > 0.05) - không dùng.
Không bao giờ raise: lỗi thì log và trả None (cổng coi None là "đạt")."""
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

from . import config as C

if TYPE_CHECKING:
    from .poles import Pole

log = logging.getLogger(__name__)


class LineIndex:
    """Ảnh xám CLAHE và tập đoạn LSD (N x 4: x1, y1, x2, y2, toạ độ ảnh đã thu về độ phân giải chuẩn,
    = toạ độ ảnh làm việc * scale) tính lười một lần."""

    def __init__(self, rgb: np.ndarray):
        self._rgb = rgb
        self.scale = min(1.0, C.POLE_LSD_MAX_SIDE / max(rgb.shape[:2]))
        self._segs: Optional[np.ndarray] = None
        self._failed = False

    def segments(self) -> Optional[np.ndarray]:
        """Đoạn LSD của cả ảnh; None nếu tính lỗi (chỉ thử một lần)."""
        if self._segs is None and not self._failed:
            try:
                img = self._rgb
                if self.scale < 1.0:
                    h, w = img.shape[:2]
                    img = cv2.resize(img, (round(w * self.scale), round(h * self.scale)),
                                     interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                clahe = cv2.createCLAHE(C.POLE_LSD_CLAHE_CLIP, C.POLE_LSD_CLAHE_TILE).apply(gray)
                s = cv2.createLineSegmentDetector().detect(clahe)[0]
                self._segs = np.zeros((0, 4), np.float64) if s is None else s.reshape(-1, 4).astype(np.float64)
            except Exception as e:  # noqa: BLE001
                log.warning("LSD thất bại, bỏ qua kiểm tra gậy: %s", e)
                self._failed = True
        return self._segs

    def covered(self, pole: "Pole") -> Optional[np.ndarray]:
        """Mẫu dọc trục top -> tip (mỗi POLE_LSD_STEP px ở độ phân giải chuẩn) có đoạn LSD song song; None nếu lỗi."""
        try:
            segs = self.segments()
            if segs is None:
                return None
            s = self.scale
            return _covered(segs, np.asarray(pole.top, np.float64) * s, np.asarray(pole.tip, np.float64) * s,
                            float(pole.half_w) * s + C.POLE_LSD_MARGIN)
        except Exception as e:  # noqa: BLE001
            log.warning("đo đoạn LSD dọc gậy thất bại: %s", e)
            return None

    def coverage(self, pole: "Pole") -> Optional[float]:
        """Độ phủ LSD (0..1) của trục gậy; None nếu lỗi."""
        try:
            segs = self.segments()
            if segs is None:
                return None
            s = self.scale
            return _coverage(segs, np.asarray(pole.top, np.float64) * s, np.asarray(pole.tip, np.float64) * s,
                             float(pole.half_w) * s + C.POLE_LSD_MARGIN)
        except Exception as e:  # noqa: BLE001
            log.warning("kiểm tra gậy bằng LSD thất bại, coi như đạt: %s", e)
            return None


def _coverage(segs: np.ndarray, top: np.ndarray, tip: np.ndarray, tol: float) -> float:
    """Tỉ lệ mẫu u = 0, STEP, ... <= L trên trục top + u*d có đoạn song song (bên nào cũng được)
    đi qua trong khoảng |v| <= tol (v nội suy tuyến tính dọc đoạn)."""
    cov = _covered(segs, top, tip, tol)
    return float(cov.mean()) if len(cov) else 0.0


def _covered(segs: np.ndarray, top: np.ndarray, tip: np.ndarray, tol: float) -> np.ndarray:
    """Mảng bool theo mẫu u = 0, STEP, ... <= L trên trục: mẫu có đoạn LSD song song đi qua (xem _coverage)."""
    L = float(np.linalg.norm(tip - top))
    if L <= 0:
        return np.zeros(0, bool)
    d = (tip - top) / L
    n = np.array([-d[1], d[0]])
    us = np.arange(0, L + 1e-6, C.POLE_LSD_STEP)
    miss = np.zeros(len(us), bool)
    if len(segs) == 0:
        return miss
    p1, p2 = segs[:, :2] - top, segs[:, 2:] - top
    sd = p2 - p1
    sl = np.hypot(sd[:, 0], sd[:, 1]) + 1e-9
    keep = np.abs(sd @ d) / sl >= math.cos(math.radians(C.POLE_LSD_PAR_DEG))
    if not keep.any():
        return miss
    u1, v1 = p1[keep] @ d, p1[keep] @ n
    u2, v2 = p2[keep] @ d, p2[keep] @ n
    sw = u2 < u1
    u1, u2 = np.where(sw, u2, u1), np.where(sw, u1, u2)
    v1, v2 = np.where(sw, v2, v1), np.where(sw, v1, v2)
    near = (u2 >= -tol) & (u1 <= L + tol) & (np.minimum(v1, v2) <= tol) & (np.maximum(v1, v2) >= -tol)
    if not near.any():
        return miss
    u1, v1, u2, v2 = u1[near], v1[near], u2[near], v2[near]
    du = np.maximum(u2 - u1, 1e-6)
    inr = (us[None] >= u1[:, None]) & (us[None] <= u2[:, None])                   # K x M
    vi = v1[:, None] + (v2 - v1)[:, None] * (us[None] - u1[:, None]) / du[:, None]
    return (inr & (np.abs(vi) <= tol)).any(axis=0)
