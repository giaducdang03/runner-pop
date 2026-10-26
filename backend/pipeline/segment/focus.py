"""
Engine "focus": giữ màu người được chọn và những người khác nằm cùng mặt phẳng nét
(ảnh xoá phông). Đường, logo, vật thể không phải người bị bỏ dù nét.
"""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from ..masks import fill_small_holes, mask_bbox
from ..schema import Cutout, Person, Scene, SegParams
from .base import ENGINES, person_init_mask
from .grabcut import cutout_from_mask, refine_mask_grabcut


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


def focus_mask(rgb: np.ndarray, person: Person, init: np.ndarray,
               frac: float = 0.3, others: Optional[List[Person]] = None) -> np.ndarray:
    """
    Mặt phẳng nét: mọi vùng có độ nét >= frac * độ nét của người được chọn.
    Vùng phẳng (áo trơn) không có chi tiết nên được lấp bằng closing + fill hole
    và luôn gộp với mask người.
    """
    h, w = rgb.shape[:2]
    E, R = focus_energy(rgb)
    core = cv2.erode((init > 127).astype(np.uint8),
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
    if core.sum() < 50:
        core = init > 127
    ref = float(np.percentile(E[core], 75))
    x1, y1, x2, y2 = person.box
    outside = np.ones((h, w), bool)
    outside[max(0, y1 - 20):y2 + 20, max(0, x1 - 20):x2 + 20] = False
    floor = 1.5 * float(np.median(E[outside])) if outside.any() else 0.0
    thr = max(frac * ref, floor)

    r_ref = float(np.percentile(R[core], 25))
    m = ((E > thr) & (R > 0.6 * r_ref)).astype(np.uint8) * 255
    size = max(h, w)
    k = max(5, int(0.015 * size) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, k // 3) | 1,) * 2))
    m = cv2.bitwise_or(m, init)
    m = fill_small_holes(m, max_hole_frac=0.01)

    # Chỉ áp dụng cho người: giữ người được chọn + những người khác cũng nằm trên mặt phẳng nét.
    allowed = (init > 127)
    for o in others or []:
        if o.mask is None:
            continue
        om = o.mask > 127
        if om.sum() < 50:
            continue
        if np.median(E[om]) >= 0.6 * ref and np.median(R[om]) >= 0.9 * r_ref:
            allowed |= om
    pad = max(5, int(0.01 * size)) | 1
    allowed_d = cv2.dilate(allowed.astype(np.uint8),
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad, pad))) > 0
    out = np.where(allowed_d & (m > 127), 255, 0).astype(np.uint8)
    out[allowed] = 255
    return out


@ENGINES.register("focus", label="Mặt phẳng nét (ảnh xoá phông: giữ màu người nét)")
def run(scene: Scene, person: Person, params: SegParams) -> Cutout:
    t = params.trace
    init = person_init_mask(scene, person)
    with t.stage("init_mask", "Mask ban đầu từ YOLO-seg") as s:
        s.overlay(scene.rgb, init)
    with t.stage("focus_map", "Bản đồ độ nét (đỏ = nét, xanh = mờ/bokeh)",
                 note="Laplacian mịn + tỉ lệ mịn/thô") as s:
        s.show(focus_energy(scene.rgb)[0], heat=True)
    with t.stage("focus_mask", "Mặt phẳng nét: người nét ngang người được chọn",
                 note=f"ngưỡng {params.focus_frac:.0%} độ nét của người được chọn") as s:
        fm = focus_mask(scene.rgb, person, init, params.focus_frac, scene.others(person))
        s.overlay(scene.rgb, fm)
    with t.stage("grabcut", "GrabCut tinh chỉnh biên", note="khởi tạo từ mặt phẳng nét") as s:
        mask = refine_mask_grabcut(scene.rgb, fm, mask_bbox(fm))
        s.overlay(scene.rgb, mask)
    return cutout_from_mask(scene.rgb, mask, params, "cf")
