"""Alpha của gậy mảnh ở độ phân giải gốc (ảnh đầy đủ): engine chạy ở độ phân giải làm việc nên
gậy vài px bị mất khi thu nhỏ. Mỗi gậy được matting lại trên dải hẹp quanh trục ở ảnh đầy đủ,
rồi phủ lên alpha của bất kỳ engine nào (max), trừ vùng của người khác."""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Optional, Tuple

import cv2
import numpy as np

from . import config as C
from .matting import matte
from .occlusion import exclusion_mask
from .poles import Pole
from .schema import Cutout, Person, Scene
from .trace import Trace

log = logging.getLogger(__name__)

_PAD = 4          # px, dải crop rộng hơn vùng unknown thêm đoạn này mỗi bên


def _poly(pole: Pole, extra: float) -> np.ndarray:
    """Polygon của gậy nở (extra > 0) hoặc co (extra < 0) `extra` px mỗi bên; không co quá 0.5 px."""
    return replace(pole, half_w=max(pole.half_w + extra, -0.5)).polygon()


def _crop_box(pole: Pole, shape) -> Optional[Tuple[int, int, int, int]]:
    """bbox (x1, y1, x2, y2) của dải chữ nhật xoay rộng 2 * (half_w + THIN_BAND_PX + 4), cắt vào ảnh."""
    poly = replace(pole, half_w=pole.half_w + C.THIN_BAND_PX + _PAD - 1.0).polygon()
    h, w = shape[:2]
    x1, y1 = np.floor(poly.min(axis=0)).astype(int)
    x2, y2 = np.ceil(poly.max(axis=0)).astype(int) + 1
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _trimap(pole: Pole, box) -> np.ndarray:
    x1, y1, x2, y2 = box
    shape = (y2 - y1, x2 - x1)
    off = np.array([x1, y1], np.float32)

    def fill(extra: float) -> np.ndarray:
        m = np.zeros(shape, np.uint8)
        cv2.fillConvexPoly(m, np.round(_poly(pole, extra) - off).astype(np.int32), 1)
        return m > 0

    fg = fill(-C.THIN_BAND_PX)
    unk = fill(C.THIN_BAND_PX) & ~fg
    tri = np.zeros(shape, np.uint8)
    tri[unk] = 128
    tri[fg] = 255
    return tri


def overlay_poles(cutout: Cutout, scene: Scene, person: Person,
                  trace: Optional[Trace] = None) -> Cutout:
    """Phủ alpha gậy (matting ở ảnh đầy đủ) lên cutout; person không có gậy thì trả nguyên cutout.
    Pixel alpha tăng: foreground = ảnh gốc, background giữ nguyên. Sửa cutout tại chỗ."""
    if not person.poles:
        return cutout
    trace = trace if trace is not None else Trace()
    with trace.stage("thin", "Alpha gậy ở độ phân giải gốc",
                     note=f"{len(person.poles)} gậy") as s:
        try:
            # mask người không chứa gậy (held.merge_pole không vẽ gậy vào mask, tránh engine tràn);
            # khi tính vùng của người khác, dải gậy vẫn là "của mình" để gậy đè lên người đứng xa
            # hơn không bị xoá
            own = (person.mask.copy() if person.mask is not None
                   else np.zeros(cutout.alpha.shape, np.uint8))
            for pole in person.poles:
                cv2.fillConvexPoly(own, np.round(pole.polygon()).astype(np.int32), 255)
            excl = exclusion_mask(scene, person, own)
            alpha = cutout.alpha.copy()
            for pole in person.poles:
                box = _crop_box(pole, alpha.shape)
                if box is None:
                    continue
                x1, y1, x2, y2 = box
                a_pole = matte(cutout.image[y1:y2, x1:x2], _trimap(pole, box), "cf")
                a_pole[excl[y1:y2, x1:x2]] = 0.0
                alpha[y1:y2, x1:x2] = np.maximum(alpha[y1:y2, x1:x2], a_pole)
            up = alpha > cutout.alpha
            cutout.foreground[up] = cutout.image[up]
            cutout.alpha = alpha
            s.show(alpha)
        except Exception as e:  # noqa: BLE001  (không bao giờ làm hỏng request)
            log.warning("alpha gậy thất bại, giữ cutout của engine: %s", e)
            s.note = f"{len(person.poles)} gậy · lỗi, bỏ qua: {e}"
    return cutout
