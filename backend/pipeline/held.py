"""Gộp vật cầm tay (gậy, chai, điện thoại, ...) vào mask và box của người cầm."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from . import config as C
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


def apply_poles(scene, person: Person, shape) -> int:
    """Gộp gậy ứng viên của đúng `person` (nếu thuộc scene.persons); idempotent: đã gộp (person.poles)
    thì không gộp lại. Trả về số gậy vừa gộp."""
    if not any(p is person for p in scene.persons) or person.poles:
        return 0
    cands = scene.pole_candidates.get(person.idx)
    if not cands:
        return 0
    for pole in cands:
        merge_pole(person, pole, shape)
    return len(cands)


@dataclass
class HeldObject:
    """Vật COCO do YOLO-seg phát hiện (ứng viên cầm tay)."""
    cls: int
    name: str
    box: tuple
    conf: float
    mask: Optional[np.ndarray] = None       # uint8 0/255, cỡ ảnh làm việc


def _obj_mask(o: HeldObject, shape) -> np.ndarray:
    return o.mask if o.mask is not None else box_to_mask(o.box, shape)


def _hand_dist(box, hands) -> float:
    """Khoảng cách nhỏ nhất từ box đến một tay (0 nếu tay nằm trong box)."""
    x1, y1, x2, y2 = box
    return min(float(np.hypot(max(x1 - hx, 0, hx - x2), max(y1 - hy, 0, hy - y2))) for hx, hy in hands)


def _owner_score(p: Person, o: HeldObject, omask: np.ndarray):
    """Điểm gần của vật với người p (nhỏ hơn = gần hơn), None nếu không đủ điều kiện.
    Người có tay xếp trước (so theo khoảng cách tay); người không có tay so theo số điểm chạm."""
    if p.hands:
        d = _hand_dist(o.box, p.hands)
        return (0, d) if d <= C.HELD_HAND_DIST * p.body_h else None
    x1, y1, x2, y2 = o.box
    k = max(1, int(round(C.HELD_TOUCH_FRAC * max(x2 - x1, y2 - y1))))
    near = cv2.dilate(p.mask, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
    touch = int(np.count_nonzero((near > 0) & (omask > 0)))
    return (1, -touch) if touch else None


def attach_objects(persons: List[Person], objects: List[HeldObject]) -> int:
    """Gộp vật gần nắm tay (hoặc chạm mask nếu người không có `hands`) vào mask/box người gần nhất.
    Trả về số vật đã gộp."""
    n = 0
    for o in objects:
        if o.conf < C.HELD_MIN_CONF:
            continue
        best, best_s = None, None
        for p in persons:
            if p.mask is None:
                continue
            om = _obj_mask(o, p.mask.shape)
            if np.count_nonzero(om) > C.HELD_MAX_AREA * max(1, np.count_nonzero(p.mask)):
                continue
            s = _owner_score(p, o, om)
            if s is not None and (best_s is None or s < best_s):
                best, best_s = p, s
        if best is None:
            continue
        h, w = best.mask.shape[:2]
        if best.body_box is None:
            best.body_box = tuple(best.box)
        best.mask = np.maximum(best.mask, _obj_mask(o, best.mask.shape))
        x1, y1, x2, y2 = best.box
        ox1, oy1, ox2, oy2 = o.box
        best.box = (max(0, min(x1, ox1)), max(0, min(y1, oy1)),
                    min(w, max(x2, ox2)), min(h, max(y2, oy2)))
        best.held.append(o.name)
        n += 1
    return n
