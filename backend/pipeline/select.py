"""Tự chọn người chạy chính trong ảnh (dùng cho chế độ auto)."""
from __future__ import annotations

import cv2
import numpy as np

from .types import Person, Scene


def _sharpness(gray_lap: np.ndarray, p: Person) -> float:
    if p.mask is not None and (p.mask > 127).sum() > 50:
        return float(gray_lap[p.mask > 127].mean())
    x1, y1, x2, y2 = p.box
    return float(gray_lap[y1:y2, x1:x2].mean()) if x2 > x1 and y2 > y1 else 0.0


def score_persons(scene: Scene) -> list:
    """
    Điểm cho từng người: to + gần tâm ảnh + nét + độ tin cậy cao.
    Người nổi bật phía trước nhưng bị mờ (ngoài vùng nét) sẽ bị trừ điểm.
    Thêm tiêu chí mới = nhân thêm một hệ số ở đây.
    """
    h, w = scene.rgb.shape[:2]
    if not scene.persons:
        return []
    gray = cv2.cvtColor(scene.rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap = np.abs(cv2.Laplacian(cv2.GaussianBlur(gray, (0, 0), 1.0), cv2.CV_32F, ksize=3))
    sharp = [_sharpness(lap, p) for p in scene.persons]
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


def suggest_person(scene: Scene) -> int:
    """Chỉ số (Person.idx) của người nên được chọn; 0 nếu không có ai."""
    scores = score_persons(scene)
    if not scores:
        return 0
    return scene.persons[int(np.argmax(scores))].idx
