"""YOLO-pose: tìm tâm nắm tay của từng người (khuỷu -> cổ tay, ngoại suy thêm một đoạn)."""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from . import accel
from . import config as C
from .detect import get_yolo, predict_kwargs
from .schema import Person

log = logging.getLogger(__name__)

_ARMS = ((7, 9), (8, 10))     # (khuỷu, cổ tay) trái/phải theo COCO keypoints
_MATCH_IOU = 0.5


def hand_points(kp: np.ndarray) -> List[Tuple[float, float]]:
    """kp 17x3 (x, y, conf). Mỗi tay đủ tin cậy: cổ tay + HAND_EXTRAP * (cổ tay - khuỷu)."""
    out: List[Tuple[float, float]] = []
    for e, w in _ARMS:
        if kp[e, 2] >= C.HAND_MIN_CONF and kp[w, 2] >= C.HAND_MIN_CONF:
            ex, ey = float(kp[e, 0]), float(kp[e, 1])
            wx, wy = float(kp[w, 0]), float(kp[w, 1])
            out.append((wx + C.HAND_EXTRAP * (wx - ex), wy + C.HAND_EXTRAP * (wy - ey)))
    return out


def _iou(a, b) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def attach_hands(persons: List[Person], pose_boxes: np.ndarray, pose_kps: np.ndarray) -> None:
    """Ghép mỗi pose với người có IoU box lớn nhất (>= 0.5) rồi điền person.hands."""
    for box, kp in zip(pose_boxes, pose_kps):
        best, best_iou = None, _MATCH_IOU
        for p in persons:
            v = _iou(box, p.box)
            if v >= best_iou:
                best, best_iou = p, v
        if best is not None:
            best.hands = hand_points(kp)


def detect_hands(rgb: np.ndarray, persons: List[Person],
                 device: Optional[accel.Device] = None) -> None:
    """Chạy YOLO-pose rồi điền hands cho từng người. Lỗi bất kỳ: log và để hands rỗng."""
    try:
        kw = predict_kwargs(device) if device is not None else {}
        if not kw.get("half"):
            kw.pop("half", None)
        res = get_yolo(C.POSE_WEIGHTS, device).predict(
            rgb, imgsz=C.POSE_IMGSZ, conf=0.25, verbose=False, **kw)[0]
        if res.boxes is None or len(res.boxes) == 0 or res.keypoints is None:
            return
        boxes = res.boxes.xyxy.cpu().numpy()
        kps = res.keypoints.data.cpu().numpy()
        attach_hands(persons, boxes, kps)
    except Exception as e:  # noqa: BLE001
        log.warning("YOLO-pose thất bại, bỏ qua vật cầm tay: %s", e)
        for p in persons:
            p.hands = []
