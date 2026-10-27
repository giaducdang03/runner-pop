"""Engine BiRefNet (rembg + onnxruntime CPU), giới hạn trong mask YOLO của người được chọn."""
from __future__ import annotations

import cv2
import numpy as np

from .. import config as C
from ..masks import keep_components_overlapping
from ..matting import finish_cutout
from ..occlusion import exclusion_mask
from ..roi import full_roi
from ..schema import Cutout, Person, Scene, SegParams
from .base import ENGINES, person_init_mask, to_float

_session = None


def birefnet_alpha(rgb: np.ndarray) -> np.ndarray:
    """BiRefNet -> alpha float32 HxW 0..1 cho toàn ảnh."""
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
    BiRefNet cho biên rất chính xác nhưng không phân biệt từng người, nên giới hạn
    bằng mask YOLO của người được chọn (nở ra vài px để không cắt tay/chân mảnh).
    """
    t = params.trace
    init = person_init_mask(scene, person)
    rgb_f = to_float(scene.rgb)
    with t.stage("init_mask", "Mask ban đầu từ YOLO-seg") as s:
        s.overlay(scene.rgb, init)
    with t.stage("birefnet", "BiRefNet: alpha toàn ảnh", note=C.BIREFNET_MODEL) as s:
        alpha = birefnet_alpha(scene.rgb)
        s.show(alpha)

    x1, y1, x2, y2 = person.box
    r = max(5, int(0.04 * max(x2 - x1, y2 - y1)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    with t.stage("occlusion", "Loại phần của người khác (thứ tự trước/sau theo vị trí chân)",
                 note=f"{len(scene.others(person))} người khác") as s:
        excl = exclusion_mask(scene, person, init)
        s.overlay(scene.rgb, excl.astype(np.uint8) * 255, color=(235, 60, 60))
    with t.stage("gate", "Giới hạn trong người được chọn",
                 note="mask YOLO nở ra − người khác, bỏ mảnh rời") as s:
        gate = cv2.dilate((init > 127).astype(np.uint8), k).astype(np.float32)
        gate[excl] = 0
        gate = cv2.GaussianBlur(gate, (0, 0), r / 3.0)
        gate[excl] = 0
        alpha = alpha * gate

        # bỏ các mảnh rời không thuộc người được chọn
        keep = keep_components_overlapping(((alpha > 0.5).astype(np.uint8)) * 255, init)
        keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))  # giữ viền mềm
        alpha = alpha * (keep > 127)
        s.show(alpha)

    tri = np.where(alpha > 0.99, 255, np.where(alpha < 0.01, 0, 128)).astype(np.uint8)
    # Task A6 sẽ chuyển sang ROI thật; tạm thời xử lý toàn ảnh ở độ phân giải gốc
    return finish_cutout(rgb_f, alpha, tri, full_roi(scene.rgb.shape), t)
