"""Gộp vật cầm tay (gậy, ...) vào mask và box của người cầm."""
from __future__ import annotations

import cv2
import numpy as np

from .masks import box_to_mask
from .poles import Pole
from .schema import Person


def merge_pole(person: Person, pole: Pole, shape) -> None:
    """Vẽ gậy vào mask người, nới box theo gậy (cắt vào ảnh), ghi nhãn "pole".
    Lần gộp đầu tiên lưu box thân người vào `body_box`."""
    h, w = shape[:2]
    if person.body_box is None:
        person.body_box = tuple(person.box)
    if person.mask is None:
        person.mask = box_to_mask(person.box, shape)
    poly = pole.polygon()
    cv2.fillConvexPoly(person.mask, np.round(poly).astype(np.int32), 255)
    x1, y1, x2, y2 = person.box
    px1 = int(np.floor(poly[:, 0].min()))
    py1 = int(np.floor(poly[:, 1].min()))
    px2 = int(np.ceil(poly[:, 0].max()))
    py2 = int(np.ceil(poly[:, 1].max()))
    person.box = (max(0, min(x1, px1)), max(0, min(y1, py1)),
                  min(w, max(x2, px2)), min(h, max(y2, py2)))
    person.held.append("pole")
    person.poles.append(pole)
