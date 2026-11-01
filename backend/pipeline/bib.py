"""Xác định số BIB của người được tách: OCR trên vùng thân (ngực/bụng) trong box người."""
from __future__ import annotations

import re
import threading
from typing import List, Optional, Sequence, Tuple

import numpy as np
# Import ở mức module (main thread lúc khởi động) để tránh deadlock khi import lười trong nhiều thread.
from rapidocr_onnxruntime import RapidOCR

MIN_CONF = 0.5
MAX_DIGITS = 6
# tiền tố chữ (tối đa 2) dính liền số; không nhận khi số dính vào chuỗi chữ dài hoặc quá 8 chữ số
_BIB_RE = re.compile(rf"(?<![A-Za-z0-9])[A-Za-z]{{0,2}}\d{{1,{MAX_DIGITS}}}(?!\d)")
TORSO = (0.15, 0.85)          # đoạn chiều cao box người có khả năng mang BIB (ngực -> thắt lưng)

_OCR: Optional[RapidOCR] = None
_OCR_LOCK = threading.Lock()


def _engine() -> RapidOCR:
    global _OCR
    with _OCR_LOCK:
        if _OCR is None:
            _OCR = RapidOCR()
        return _OCR


def pick_bib(items: Sequence[Tuple[Sequence[Sequence[float]], str, float]]) -> str:
    """items: (4 đỉnh, chuỗi, độ tin cậy). Lấy BIB dạng [1-2 chữ cái]+1..6 chữ số (vd M5520), tin cậy đủ cao;
    nhiều ứng viên -> chữ cao nhất (BIB thường là chữ to nhất trên áo). Không có -> ''."""
    best, best_h = "", -1.0
    for pts, text, conf in items:
        m = _BIB_RE.search(re.sub(r"\s+", "", text or ""))
        if not m or float(conf) < MIN_CONF:
            continue
        digits = m.group(0).upper()
        ys = [p[1] for p in pts]
        h = max(ys) - min(ys)
        if h > best_h:
            best, best_h = digits, h
    return best


def read_bib(rgb: np.ndarray, box: Tuple[int, int, int, int], mask: Optional[np.ndarray] = None) -> str:
    """Đọc số BIB của người trong `box` (ảnh RGB uint8). Trả '' nếu không đọc được."""
    x1, y1, x2, y2 = box
    h = y2 - y1
    ty1, ty2 = y1 + int(TORSO[0] * h), y1 + int(TORSO[1] * h)
    crop = rgb[ty1:ty2, x1:x2]
    if crop.size == 0:
        return ""
    if mask is not None:                       # ngoài người -> xám trung tính, tránh đọc số của người bên cạnh
        m = mask[ty1:ty2, x1:x2] > 127
        crop = np.where(m[..., None], crop, np.uint8(127))
    # OCR rất nhạy với chữ nhỏ -> phóng to vùng thân nếu hẹp
    scale = max(1.0, 640.0 / max(crop.shape[:2]))
    if scale > 1.0:
        import cv2
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    result, _ = _engine()(np.ascontiguousarray(crop[..., ::-1]))      # RapidOCR dùng BGR
    return pick_bib(result or [])
