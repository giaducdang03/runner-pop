"""
Engine "focus": giữ màu người được chọn và những người khác nằm cùng mặt phẳng nét
(ảnh xoá phông). Đường, logo, vật thể không phải người bị bỏ dù nét.
"""
from __future__ import annotations

import time
from typing import List, Optional

import cv2
import numpy as np

from .. import accel
from .. import config as C
from ..focusmap import focus_confidence, focus_energy, sharpness_map
from ..masks import fill_small_holes, mask_bbox
from ..schema import Cutout, Person, Scene, SegParams
from ..roi import active_roi, roi_stage
from .base import ENGINES, person_init_mask, to_float, work_scene
from .grabcut import cutout_from_mask, refine_mask_grabcut


def prune_blurry_rim(init: np.ndarray, E: np.ndarray, ref: float, width: int) -> np.ndarray:
    """
    Mask YOLO của người nét đôi khi lẫn cả người phía sau (mờ, không được phát hiện) dính sát.
    Phần lẫn đó nằm ở rìa mask và có viền ngoài MỜ, còn thân/tay/chân người nét có viền nét.
    Từ các điểm viền ngoài mờ, lấn dần vào trong (tối đa C.FOCUS_RIM_FRAC * bề ngang người) qua
    vùng không nét; gặp điểm nét thì dừng. Trả mask uint8 đã bỏ phần đó (lõi người luôn được giữ).
    """
    m = init > 127
    if ref <= 0 or not m.any():
        return init
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sharp = (E > C.FOCUS_RIM_SHARP * ref).astype(np.uint8)
    sharp_d = cv2.dilate(sharp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    ring = m & ~(cv2.erode(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0)
    blur_ring = ring & (cv2.dilate(sharp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) == 0)
    if not blur_ring.any():
        return init
    free = (m & (sharp_d == 0)).astype(np.uint8)          # đường lấn: trong mask, không nét
    grown = (blur_ring & (free > 0)).astype(np.uint8)
    for _ in range(max(4, int(C.FOCUS_RIM_FRAC * max(width, 1)))):
        nxt = cv2.dilate(grown, k3) & free
        if nxt.sum() == grown.sum():
            break
        grown = nxt
    # thân người = bao đóng của điểm nét trong mask (áo phẳng nằm giữa các điểm nét vẫn được giữ)
    kc = max(5, int(0.12 * max(width, 1)) | 1)
    body = cv2.morphologyEx((sharp_d > 0).astype(np.uint8) * 255 * m.astype(np.uint8), cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)))
    body = fill_small_holes(body, max_hole_frac=0.2) > 127
    keep = m & ~((grown > 0) & ~body)
    keep = cv2.morphologyEx(keep.astype(np.uint8) * 255, cv2.MORPH_OPEN, k3)
    return np.where(m & (keep > 127), 255, 0).astype(np.uint8)


def focus_mask(rgb: np.ndarray, person: Person, init: np.ndarray,
               frac: float = 0.3, others: Optional[List[Person]] = None,
               scale: float = 1.0, smap: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Mặt phẳng nét: mọi vùng có độ nét >= frac * độ nét của người được chọn.
    Vùng phẳng (áo trơn) không có chi tiết nên được lấp bằng closing + fill hole
    và luôn gộp với mask người.
    `scale` = hệ số thu nhỏ ROI (roi.scale): ngưỡng số điểm ảnh và lề 20 px quanh box nhân theo
    scale (diện tích theo scale²) để cùng ý nghĩa ở độ phân giải làm việc.
    `smap` = bản đồ nét dùng chung (cùng kích thước `rgb`); None -> `sharpness_map(rgb)[0]`. Ngưỡng là
    phân vị của chính bản đồ nên `frac` giữ nguyên ngữ nghĩa với mọi phương pháp đo.
    """
    h, w = rgb.shape[:2]
    min_px = max(1, round(50 * scale ** 2))
    margin = max(1, round(20 * scale))
    R = focus_energy(rgb)[1]
    E = smap if smap is not None else sharpness_map(rgb)[0]
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
    init = prune_blurry_rim(init, E, ref, x2 - x1)
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
        e_ratio = float(np.median(E[om])) / max(ref, 1e-9)
        r_ratio = float(np.median(R[om])) / max(r_ref, 1e-9)
        # cổng riêng nới lỏng + cổng kết hợp: người chỉ "hơi mờ" (cả hai tỉ lệ sát ngưỡng) bị loại
        if e_ratio >= 0.6 and r_ratio >= 0.9 and e_ratio * r_ratio >= C.FOCUS_OTHER_MIN:
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
    key = (person.idx, person.box)

    def build_mask():
        built.append(True)

        def compute_map():
            t0 = time.perf_counter()
            m, used = sharpness_map(scene_w.rgb, quality=scene.quality)
            return m, used, (time.perf_counter() - t0) * 1000.0

        with t.stage("focus_map", "Bản đồ độ nét (đỏ = nét, xanh = mờ/bokeh)") as s:
            # bản đồ không phụ thuộc focus_frac -> đổi ngưỡng chỉ tính lại mask (FR-94, FR-125)
            smap, method, ms = params.cached(("focusmap", *key, C.FOCUS_METHOD), compute_map)
            s.note = f"phương pháp {method} · {ms:.0f} ms"
            s.show(smap, heat=True)
        with t.stage("focus_mask", "Mặt phẳng nét: người nét ngang người được chọn",
                     note=f"ngưỡng {params.focus_frac:.0%} độ nét của người được chọn") as s:
            fm = focus_mask(scene_w.rgb, person_w, init, params.focus_frac,
                            scene_w.others(person_w), scale=roi.scale, smap=smap)
            conf = focus_confidence(smap, init)
            s.note += f" · độ tin cậy {conf:.2f}"
            s.overlay(scene_w.rgb, fm)
        with t.stage("grabcut", "GrabCut tinh chỉnh biên", note="khởi tạo từ mặt phẳng nét") as s:
            s.set_device(accel.resolve("cpu"))
            mask = refine_mask_grabcut(scene_w.rgb, fm, mask_bbox(fm))
            s.overlay(scene_w.rgb, mask)
        return mask, conf

    # mặt phẳng nét không phụ thuộc band_frac -> đổi band chỉ tính lại từ trimap (FR-94)
    mask_work, conf = params.cached(
        ("focusmask", *key, round(params.focus_frac, 3), C.FOCUS_METHOD), build_mask)
    t.meta["focus_confidence"] = conf
    if not built:   # lấy từ cache: vẫn ghi bước để sơ đồ pipeline đủ bước
        with t.stage("focus_mask", "Mặt phẳng nét: người nét ngang người được chọn",
                     note=f"dùng lại mask đã tính (cache) · độ tin cậy {conf:.2f}") as s:
            s.overlay(scene_w.rgb, mask_work)
    return cutout_from_mask(to_float(scene.rgb), mask_work, roi, params, "cf")
