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
from ..roi import active_roi, roi_stage
from .base import ENGINES, person_init_mask, to_float, work_scene
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
               frac: float = 0.3, others: Optional[List[Person]] = None,
               scale: float = 1.0) -> np.ndarray:
    """
    Mặt phẳng nét: mọi vùng có độ nét >= frac * độ nét của người được chọn.
    Vùng phẳng (áo trơn) không có chi tiết nên được lấp bằng closing + fill hole
    và luôn gộp với mask người.
    `scale` = hệ số thu nhỏ ROI (roi.scale): ngưỡng số điểm ảnh và lề 20 px quanh box nhân theo
    scale (diện tích theo scale²) để cùng ý nghĩa ở độ phân giải làm việc.
    """
    h, w = rgb.shape[:2]
    min_px = max(1, round(50 * scale ** 2))
    margin = max(1, round(20 * scale))
    E, R = focus_energy(rgb)
    core = cv2.erode((init > 127).astype(np.uint8),
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
    if core.sum() < min_px:
        core = init > 127
    ref = float(np.percentile(E[core], 75))
    x1, y1, x2, y2 = person.box
    outside = np.ones((h, w), bool)
    outside[max(0, y1 - margin):y2 + margin, max(0, x1 - margin):x2 + margin] = False
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
        if om.sum() < min_px:
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
    """Toàn bộ focus_energy / focus_mask / GrabCut chạy trên ảnh ROI ở độ phân giải làm việc."""
    t = params.trace
    # ROI của focus gộp mọi người trong ảnh (cùng mặt phẳng nét, FR-37) -> khoá cache riêng (R16)
    roi = params.cached(("roi", person.idx, person.box, "all"),
                        lambda: active_roi(scene, person, scope="all"))
    w, h = roi.work_size
    scene_w, person_w = work_scene(scene, person, roi)
    init = person_init_mask(scene_w, person_w)
    with t.stage("init_mask", "Mask ban đầu từ YOLO-seg",
                 note=f"ROI {w}×{h} (scale {roi.scale:.2f})") as s:
        s.overlay(scene_w.rgb, init)
    roi_stage(t, scene.rgb, roi)

    built = []

    def build_mask() -> np.ndarray:
        built.append(True)
        with t.stage("focus_map", "Bản đồ độ nét (đỏ = nét, xanh = mờ/bokeh)",
                     note="Laplacian mịn + tỉ lệ mịn/thô") as s:
            s.show(focus_energy(scene_w.rgb)[0], heat=True)
        with t.stage("focus_mask", "Mặt phẳng nét: người nét ngang người được chọn",
                     note=f"ngưỡng {params.focus_frac:.0%} độ nét của người được chọn") as s:
            fm = focus_mask(scene_w.rgb, person_w, init, params.focus_frac,
                            scene_w.others(person_w), scale=roi.scale)
            s.overlay(scene_w.rgb, fm)
        with t.stage("grabcut", "GrabCut tinh chỉnh biên", note="khởi tạo từ mặt phẳng nét") as s:
            mask = refine_mask_grabcut(scene_w.rgb, fm, mask_bbox(fm))
            s.overlay(scene_w.rgb, mask)
        return mask

    # mặt phẳng nét không phụ thuộc band_frac -> đổi band chỉ tính lại từ trimap (FR-94)
    mask_work = params.cached(("focusmask", person.idx, person.box, round(params.focus_frac, 3)),
                              build_mask)
    if not built:   # lấy từ cache: vẫn ghi bước để sơ đồ pipeline đủ bước
        with t.stage("focus_mask", "Mặt phẳng nét: người nét ngang người được chọn",
                     note="dùng lại mask đã tính (cache)") as s:
            s.overlay(scene_w.rgb, mask_work)
    return cutout_from_mask(to_float(scene.rgb), mask_work, roi, params, "cf")
