"""Engine BiRefNet (rembg + onnxruntime CPU), giới hạn trong mask YOLO của người được chọn."""
from __future__ import annotations

import numpy as np

from .. import config as C
from ..matting import finish_cutout
from ..occlusion import exclusion_mask
from ..roi import active_roi, roi_stage
from ..schema import Cutout, Person, Scene, SegParams
from .base import ENGINES, person_init_mask, to_float, work_scene
from .gate import gate_to_person

_session = None


def birefnet_alpha(rgb: np.ndarray) -> np.ndarray:
    """BiRefNet -> alpha float32 HxW 0..1 cho ảnh đưa vào (ảnh ROI ở độ phân giải làm việc)."""
    global _session
    from PIL import Image
    from rembg import new_session, remove

    if _session is None:
        _session = new_session(C.BIREFNET_MODEL, providers=["CPUExecutionProvider"])
    m = remove(Image.fromarray(rgb), session=_session, only_mask=True)
    return np.asarray(m.convert("L"), dtype=np.float32) / 255.0


@ENGINES.register("birefnet", label="BiRefNet (mask chính xác, chậm nhất)")
def run(scene: Scene, person: Person, params: SegParams) -> Cutout:
    """
    BiRefNet chạy trên ROI quanh người (độ phân giải làm việc). Model không phân biệt
    từng người nên α được giới hạn bằng mask YOLO của người được chọn (khối `gate`).
    """
    t = params.trace
    roi = params.cached(("roi", person.idx, person.box), lambda: active_roi(scene, person))
    w, h = roi.work_size
    scene_w, person_w = work_scene(scene, person, roi)
    init = person_init_mask(scene_w, person_w)
    with t.stage("init_mask", "Mask ban đầu từ YOLO-seg",
                 note=f"ROI {w}×{h} (scale {roi.scale:.2f})") as s:
        s.overlay(scene_w.rgb, init)
    roi_stage(t, scene.rgb, roi)
    with t.stage("birefnet", "BiRefNet: alpha trên ROI",
                 note=f"{C.BIREFNET_MODEL}, ROI {w}×{h} (scale {roi.scale:.2f})") as s:
        alpha_raw = params.cached(("birefnet", person.idx, person.box),
                                  lambda: birefnet_alpha(scene_w.rgb))
        s.show(alpha_raw)
    with t.stage("occlusion", "Loại phần của người khác (thứ tự trước/sau theo vị trí chân)",
                 note=f"{len(scene.others(person))} người khác") as s:
        excl = exclusion_mask(scene_w, person_w, init)
        s.overlay(scene_w.rgb, excl.astype(np.uint8) * 255, color=(235, 60, 60))
    with t.stage("gate", "Giới hạn trong người được chọn",
                 note="mask YOLO nở ra − người khác, bỏ mảnh rời") as s:
        alpha = gate_to_person(alpha_raw, init, person_w.box, excl)
        s.show(alpha)

    tri = np.where(alpha > 0.99, 255, np.where(alpha < 0.01, 0, 128)).astype(np.uint8)
    return finish_cutout(to_float(scene.rgb), alpha, tri, roi, t)
