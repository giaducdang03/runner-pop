"""Engine GrabCut -> trimap -> matting (closed-form / KNN)."""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from ..masks import fill_small_holes, keep_components_overlapping
from ..matting import finish_cutout, make_trimap, matte
from ..occlusion import exclusion_mask
from ..roi import Roi, active_roi
from ..schema import Cutout, Person, Scene, SegParams
from .base import ENGINES, person_init_mask, to_float


def refine_mask_grabcut(rgb: np.ndarray, init_mask: np.ndarray,
                        box: Tuple[int, int, int, int],
                        iters: int = 4, rescue_core: bool = True,
                        exclude: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Dùng mask YOLO làm gợi ý: lõi = chắc chắn FG, vành quanh = có thể,
    ngoài vành = chắc chắn BG. GrabCut chạy trong ROI quanh box để nhanh.
    `exclude` (bool HxW): vùng của người khác -> luôn là nền chắc chắn, không bao giờ vào kết quả.
    """
    if exclude is not None and exclude.any():
        init_mask = np.where(exclude, 0, init_mask).astype(np.uint8)
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    pad = int(0.15 * max(bw, bh)) + 8
    rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
    rx2, ry2 = min(w, x2 + pad), min(h, y2 + pad)

    roi = np.ascontiguousarray(rgb[ry1:ry2, rx1:rx2])
    m0 = init_mask[ry1:ry2, rx1:rx2] > 127

    k = max(3, int(0.03 * max(bw, bh)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    sure_fg = cv2.erode(m0.astype(np.uint8), kernel, iterations=2).astype(bool)
    maybe = cv2.dilate(m0.astype(np.uint8), kernel, iterations=2).astype(bool)

    gc = np.full(m0.shape, cv2.GC_BGD, np.uint8)
    gc[maybe] = cv2.GC_PR_BGD
    gc[m0] = cv2.GC_PR_FGD
    gc[sure_fg] = cv2.GC_FGD
    if exclude is not None:
        gc[exclude[ry1:ry2, rx1:rx2]] = cv2.GC_BGD

    if sure_fg.sum() < 50 or (~maybe).sum() < 50:     # không đủ mẫu cho GMM
        return init_mask

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(cv2.cvtColor(roi, cv2.COLOR_RGB2BGR), gc, None, bgd, fgd,
                    iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return init_mask

    out = np.zeros((h, w), np.uint8)
    fg = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8) * 255
    out[ry1:ry2, rx1:rx2] = fg

    # GrabCut hay cắt nhầm tay/chân có màu da gần màu nền (hoặc người phía sau):
    # gộp lại lõi chắc chắn của mask YOLO để không mất chi.
    if rescue_core:
        out[(cv2.erode((init_mask > 127).astype(np.uint8),
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0)] = 255

    if exclude is not None:
        out[exclude] = 0

    # giữ các thành phần liên thông chồng lên mask YOLO (không chỉ thành phần lớn nhất), lấp lỗ nhỏ
    out = keep_components_overlapping(out, init_mask)
    return fill_small_holes(out, max_hole_frac=0.002)


def cutout_from_mask(rgb_f: np.ndarray, mask_work: np.ndarray, roi: Roi,
                     params: SegParams, method: str) -> Cutout:
    """mask -> trimap -> matting -> khử nhiễm màu. Dùng chung cho các engine dựa trên mask.
    `rgb_f`: ảnh ĐẦY ĐỦ float32 0..1; `mask_work`: mask uint8 ở `roi.work_size`.
    Trimap + matting chạy ở độ phân giải làm việc, `finish_cutout` nâng về ảnh đầy đủ."""
    t = params.trace
    with t.stage("trimap", "Trimap (FG chắc chắn / vùng chưa biết / BG)",
                 note=f"vùng chưa biết ≈ {params.band_frac:.1%} kích thước người") as s:
        tri = make_trimap(mask_work, band_frac=params.band_frac)
        s.show(tri)
    rgb_work = np.ascontiguousarray(roi.to_work(rgb_f), dtype=np.float32)
    w, h = roi.work_size
    with t.stage("matting", "Alpha matting",
                 note=("closed-form (pymatting)" if method == "cf" else "KNN (pymatting)")
                 + f", ROI {w}×{h}") as s:
        alpha = matte(rgb_work, tri, method=method)
        s.show(alpha)
    return finish_cutout(rgb_f, alpha, tri, roi, t)


def _box_to_work(box: Tuple[int, int, int, int], roi: Roi) -> Tuple[int, int, int, int]:
    """Đổi box từ toạ độ ảnh đầy đủ sang hệ ROI ở độ phân giải làm việc (cắt vào ROI)."""
    ww, wh = roi.work_size
    x1, y1, x2, y2 = box

    def conv(v, o, hi):
        return int(min(hi, max(0, round((v - o) * roi.scale))))

    return (conv(x1, roi.x1, ww), conv(y1, roi.y1, wh), conv(x2, roi.x1, ww), conv(y2, roi.y1, wh))


def _work_scene(scene: Scene, person: Person, roi: Roi) -> Tuple[Scene, Person]:
    """Scene thu về ROI làm việc: ảnh INTER_AREA, mask mọi người INTER_NEAREST, box đổi toạ độ.
    Dùng để tính mask khởi tạo và vùng loại trừ người khác (FR-37) ngay ở độ phân giải làm việc."""
    def conv(p: Person) -> Person:
        m = None if p.mask is None else roi.to_work(p.mask, cv2.INTER_NEAREST)
        return Person(idx=p.idx, box=_box_to_work(p.box, roi), conf=p.conf, mask=m)

    person_w = conv(person)
    persons_w = [person_w if p is person else conv(p) for p in scene.persons]
    if not any(p is person_w for p in persons_w):   # box thủ công không nằm trong scene.persons
        persons_w.append(person_w)
    return Scene(rgb=roi.to_work(scene.rgb), persons=persons_w), person_w


def _make_engine(method: str):
    def run(scene: Scene, person: Person, params: SegParams) -> Cutout:
        t = params.trace
        roi = params.cached(("roi", person.idx, person.box), lambda: active_roi(scene, person))
        w, h = roi.work_size

        built = []

        def build_mask() -> np.ndarray:
            built.append(True)
            scene_w, person_w = _work_scene(scene, person, roi)
            init = person_init_mask(scene_w, person_w)
            with t.stage("init_mask", "Mask ban đầu từ YOLO-seg",
                         note=f"ROI {w}×{h} (scale {roi.scale:.2f})") as s:
                s.overlay(scene_w.rgb, init)
            with t.stage("occlusion", "Loại phần của người khác (thứ tự trước/sau theo vị trí chân)",
                         note=f"{len(scene.others(person))} người khác") as s:
                excl = exclusion_mask(scene_w, person_w, init)
                s.overlay(scene_w.rgb, excl.astype(np.uint8) * 255, color=(235, 60, 60))
            with t.stage("grabcut", "GrabCut tinh chỉnh biên", note="khởi tạo từ mask YOLO, 4 vòng") as s:
                mask = refine_mask_grabcut(scene_w.rgb, init, person_w.box, exclude=excl)
                s.overlay(scene_w.rgb, mask)
            return mask

        # mask GrabCut không phụ thuộc band_frac -> đổi band chỉ tính lại từ trimap (FR-94)
        mask_work = params.cached(("gcmask", person.idx, person.box), build_mask)
        if not built:   # lấy từ cache: vẫn ghi một bước để sơ đồ pipeline đủ bước
            with t.stage("grabcut", "GrabCut tinh chỉnh biên", note="dùng lại mask đã tính (cache)") as s:
                s.overlay(roi.to_work(scene.rgb), mask_work)
        return cutout_from_mask(to_float(scene.rgb), mask_work, roi, params, method)
    return run


ENGINES.register("cf", label="GrabCut + Closed-form")(_make_engine("cf"))
ENGINES.register("knn", label="GrabCut + KNN (chậm hơn)")(_make_engine("knn"))
