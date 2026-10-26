"""
Xử lý người chồng nhau: pixel nào KHÔNG được thuộc về người được chọn vì đã là của người khác.

Quy tắc "ai ở phía trước": ảnh chụp ngang tầm mắt thì người có chân thấp hơn trong ảnh (y2 lớn hơn)
đứng gần máy hơn. Ở vùng hai mask chồng nhau, pixel thuộc về người gần hơn.

  - Người khác ở GẦN hơn  -> cả mask của họ bị loại khỏi người được chọn (người được chọn đang bị che).
  - Người khác ở XA hơn / ngang hàng -> chỉ loại phần mask nằm ngoài mask của người được chọn
    (phần chồng nhau giữ lại cho người được chọn).
"""
from __future__ import annotations

import cv2
import numpy as np

from .schema import Person, Scene

NEAR_TOL = 0.03      # chênh lệch y2 nhỏ hơn 3% chiều cao người được coi là ngang hàng


def exclusion_mask(scene: Scene, person: Person, init: np.ndarray) -> np.ndarray:
    """bool HxW: vùng thuộc người khác, không được giao cho `person`."""
    own = init > 127
    own_d = cv2.dilate(own.astype(np.uint8),
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    ref_h = max(1, person.box[3] - person.box[1])
    excl = np.zeros(init.shape, bool)
    for o in scene.others(person):
        if o.mask is None:
            continue
        om = o.mask > 127
        if om.sum() < 50:
            continue
        other_nearer = (o.box[3] - person.box[3]) > NEAR_TOL * ref_h
        excl |= om if other_nearer else (om & ~own_d)
    return excl
