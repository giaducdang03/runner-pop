"""ROI quanh người được chọn: engine chỉ xử lý vùng này rồi dán lại ảnh đầy đủ (FR-90)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from . import config as C
from .schema import Person, Scene


@dataclass(frozen=True)
class Roi:
    """Hình chữ nhật [x1,x2) x [y1,y2) trong ảnh làm việc đầy đủ + hệ số thu nhỏ khi xử lý."""
    x1: int
    y1: int
    x2: int
    y2: int
    scale: float
    full_shape: Tuple[int, int]          # (H, W) của ảnh làm việc đầy đủ

    @property
    def full_size(self) -> Tuple[int, int]:
        """(w, h) của ROI ở độ phân giải gốc."""
        return (self.x2 - self.x1, self.y2 - self.y1)

    @property
    def work_size(self) -> Tuple[int, int]:
        """(w, h) của ROI sau khi nhân `scale` (kích thước engine thực sự chạy)."""
        w, h = self.full_size
        return (max(1, round(w * self.scale)), max(1, round(h * self.scale)))

    @property
    def is_full(self) -> bool:
        h, w = self.full_shape
        return (self.x1, self.y1, self.x2, self.y2) == (0, 0, w, h)

    def crop(self, img: np.ndarray) -> np.ndarray:
        """Cắt ROI từ ảnh `full_shape`, giữ độ phân giải gốc."""
        return img[self.y1:self.y2, self.x1:self.x2]

    def to_work(self, img: np.ndarray, interp: int = cv2.INTER_AREA) -> np.ndarray:
        """Cắt ROI rồi thu nhỏ về `work_size` (không đổi nếu scale = 1)."""
        patch = self.crop(img)
        if self.work_size == self.full_size:
            return patch
        return cv2.resize(patch, self.work_size, interpolation=interp)

    def paste(self, patch: np.ndarray, fill=0) -> np.ndarray:
        """Dán patch kích thước `full_size` vào ảnh `full_shape`; ngoài ROI = `fill`.
        Nhận patch 2D hoặc 3D."""
        h, w = self.full_shape
        out = np.full((h, w) + patch.shape[2:], fill, dtype=patch.dtype)
        out[self.y1:self.y2, self.x1:self.x2] = patch
        return out


def _overlaps(a, b) -> bool:
    """Hai box giao nhau với diện tích > 0."""
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def person_roi(scene: Scene, person: Person, margin: float = C.ROI_MARGIN,
               min_px: int = C.ROI_MIN_PX, max_side: int = C.WORK_MAX_SIDE) -> Roi:
    """ROI = box người nở `max(min_px, margin * cạnh dài)` mỗi phía, gộp box những người
    khác có box chồng lên box người này, rồi cắt vào ảnh. Chỉ cần box (không cần mask)."""
    h, w = scene.rgb.shape[:2]
    bx1, by1, bx2, by2 = person.box
    pad = max(float(min_px), margin * max(bx2 - bx1, by2 - by1))
    x1, y1, x2, y2 = bx1 - pad, by1 - pad, bx2 + pad, by2 + pad
    for o in scene.others(person):
        if _overlaps(person.box, o.box):
            x1, y1 = min(x1, o.box[0]), min(y1, o.box[1])
            x2, y2 = max(x2, o.box[2]), max(y2, o.box[3])
    x1, y1 = max(0, math.floor(x1)), max(0, math.floor(y1))
    x2, y2 = min(w, math.ceil(x2)), min(h, math.ceil(y2))
    # chỉ thu nhỏ, không bao giờ phóng to ROI nhỏ
    scale = min(1.0, max_side / max(x2 - x1, y2 - y1, 1))
    return Roi(x1, y1, x2, y2, scale, (h, w))


def full_roi(shape, max_side: Optional[int] = None) -> Roi:
    """ROI phủ cả ảnh. `max_side=None` -> scale 1.0."""
    h, w = int(shape[0]), int(shape[1])
    scale = 1.0
    if max_side is not None and max(h, w) > max_side:
        scale = max_side / max(h, w)
    return Roi(0, 0, w, h, scale, (h, w))


def active_roi(scene: Scene, person: Person) -> Roi:
    """ROI mọi engine dùng: toàn ảnh khi FULLRES bật, ngược lại ROI quanh người."""
    if C.FULLRES:
        return full_roi(scene.rgb.shape)
    return person_roi(scene, person)


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int = C.GUIDED_RADIUS,
                  eps: float = C.GUIDED_EPS) -> np.ndarray:
    """Guided filter O(1) (He et al.) chỉ dùng boxFilter. `guide` xám float32 0..1,
    `src` float32 cùng HxW (2D hoặc 3D; 3D thì lọc từng kênh)."""
    if src.ndim == 3:
        return np.stack([guided_filter(guide, src[..., c], radius, eps)
                         for c in range(src.shape[2])], axis=-1)
    k = (2 * radius + 1, 2 * radius + 1)
    box = lambda x: cv2.boxFilter(x, -1, k)
    I = guide.astype(np.float32)
    p = src.astype(np.float32)
    mean_i, mean_p = box(I), box(p)
    cov_ip = box(I * p) - mean_i * mean_p
    var_i = box(I * I) - mean_i * mean_i
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return (box(a) * I + box(b)).astype(np.float32)


def _guide_gray(roi: Roi, guide_rgb_f: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(roi.crop(guide_rgb_f), dtype=np.float32),
                        cv2.COLOR_RGB2GRAY)


def upsample_alpha(alpha_work: np.ndarray, roi: Roi, guide_rgb_f: np.ndarray,
                   unknown_work: Optional[np.ndarray] = None) -> np.ndarray:
    """Nâng α từ độ phân giải làm việc về `full_shape`: resize tuyến tính, guided filter
    theo ảnh gốc, chỉ nhận giá trị lọc trong dải chưa biết (ngoài dải giữ 0/1). Ngoài ROI = 0."""
    size = roi.full_size
    a_up = alpha_work.astype(np.float32)
    if a_up.shape[::-1] != size:
        a_up = cv2.resize(a_up, size, interpolation=cv2.INTER_LINEAR)
    if unknown_work is not None:
        u = unknown_work.astype(np.float32)
        if u.shape[::-1] != size:
            u = cv2.resize(u, size, interpolation=cv2.INTER_LINEAR)
        band = u > 0.5
    else:
        band = ((a_up > 0.02) & (a_up < 0.98)).astype(np.uint8)
        band = cv2.dilate(band, np.ones((5, 5), np.uint8)) > 0   # nở 2 px
    filtered = np.clip(guided_filter(_guide_gray(roi, guide_rgb_f), a_up), 0.0, 1.0)
    out = np.where(band, filtered, a_up).astype(np.float32)
    return roi.paste(out, fill=0.0)


def upsample_rgb(rgb_work: np.ndarray, roi: Roi, guide_rgb_f: np.ndarray,
                 base: np.ndarray) -> np.ndarray:
    """Nâng ảnh RGB float (F hoặc B) về `full_shape` bằng resize + guided filter.
    Ngoài ROI lấy `base`."""
    size = roi.full_size
    up = rgb_work.astype(np.float32)
    if up.shape[1::-1] != size:
        up = cv2.resize(up, size, interpolation=cv2.INTER_LINEAR)
    up = guided_filter(_guide_gray(roi, guide_rgb_f), up)
    out = base.astype(np.float32).copy()
    out[roi.y1:roi.y2, roi.x1:roi.x2] = up
    return out
