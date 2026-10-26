"""Engine BiRefNet (rembg + onnxruntime CPU), giới hạn trong mask YOLO của người được chọn."""
from __future__ import annotations

import cv2
import numpy as np

from .. import config as C
from ..masks import keep_components_overlapping
from ..matting import finish_cutout
from ..types import Cutout, Person, Scene, SegParams
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
    init = person_init_mask(scene, person)
    rgb_f = to_float(scene.rgb)
    alpha = birefnet_alpha(scene.rgb)

    x1, y1, x2, y2 = person.box
    r = max(5, int(0.04 * max(x2 - x1, y2 - y1)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    gate = cv2.dilate((init > 127).astype(np.uint8), k).astype(np.float32)
    gate = cv2.GaussianBlur(gate, (0, 0), r / 3.0)
    alpha = alpha * gate

    # bỏ các mảnh rời không thuộc người được chọn
    keep = keep_components_overlapping(((alpha > 0.5).astype(np.uint8)) * 255, init)
    keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))  # giữ viền mềm
    alpha = alpha * (keep > 127)

    tri = np.where(alpha > 0.99, 255, np.where(alpha < 0.01, 0, 128)).astype(np.uint8)
    return finish_cutout(rgb_f, alpha, tri)
