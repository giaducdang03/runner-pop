"""Tự chọn người chạy chính trong ảnh (dùng cho chế độ auto)."""
from __future__ import annotations

import numpy as np

from typing import Optional

from .focusmap import sharpness_map
from .schema import Person, Scene


def _sharpness(smap: np.ndarray, p: Person) -> float:
    if p.mask is not None and (p.mask > 127).sum() > 50:
        return float(smap[p.mask > 127].mean())
    x1, y1, x2, y2 = p.box
    return float(smap[y1:y2, x1:x2].mean()) if x2 > x1 and y2 > y1 else 0.0


def score_persons(scene: Scene, smap: Optional[np.ndarray] = None) -> list:
    """
    Điểm cho từng người: to + gần tâm ảnh + nét + độ tin cậy cao.
    Người nổi bật phía trước nhưng bị mờ (ngoài vùng nét) sẽ bị trừ điểm.
    Thêm tiêu chí mới = nhân thêm một hệ số ở đây.
    `smap` = bản đồ nét dùng chung với engine focus; None -> `sharpness_map(scene.rgb)` (FR-125).
    """
    h, w = scene.rgb.shape[:2]
    if not scene.persons:
        return []
    if smap is None:
        smap = sharpness_map(scene.rgb, quality=scene.quality)[0]
    sharp = [_sharpness(smap, p) for p in scene.persons]
    max_sharp = max(sharp) or 1.0
    diag = float(np.hypot(w, h))

    scores = []
    for p, sh in zip(scene.persons, sharp):
        x1, y1, x2, y2 = p.box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        dist = np.hypot(cx - w / 2, cy - h / 2) / (diag / 2)          # 0 = tâm, ~1 = góc
        area = p.area / float(w * h)
        scores.append(area ** 0.5 * (1.0 - 0.5 * min(1.0, dist)) * (sh / max_sharp) ** 0.7 * p.conf)
    return scores


def suggest_person(scene: Scene, smap: Optional[np.ndarray] = None) -> int:
    """Chỉ số (Person.idx) của người nên được chọn; 0 nếu không có ai."""
    scores = score_persons(scene, smap)
    if not scores:
        return 0
    return scene.persons[int(np.argmax(scores))].idx
