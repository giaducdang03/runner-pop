"""Phát hiện người (YOLO-seg) + thang thử lại khi ảnh kém chất lượng."""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from . import config as C
from . import preprocess as PP
from .types import Person, Scene

_yolo_model = None


def get_yolo(weights: str = C.YOLO_WEIGHTS):
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO(weights)
    return _yolo_model


def detect_persons(det_rgb: np.ndarray, weights: str = C.YOLO_WEIGHTS,
                   conf: float = 0.25, imgsz: int = C.YOLO_IMGSZ) -> List[Person]:
    """Trả về danh sách người, sắp xếp theo diện tích box giảm dần."""
    model = get_yolo(weights)
    res = model.predict(det_rgb, imgsz=imgsz, conf=conf, classes=[C.PERSON_CLASS],
                        device="cpu", verbose=False,
                        retina_masks=True)[0]   # mask độ phân giải gốc -> không mất tay/chân mảnh
    h, w = det_rgb.shape[:2]
    persons: List[Person] = []
    if res.boxes is None or len(res.boxes) == 0:
        return persons

    boxes = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    masks = res.masks.data.cpu().numpy() if res.masks is not None else None   # N x mh x mw (0/1)

    for i, (bx, cf) in enumerate(zip(boxes, confs)):
        x1, y1, x2, y2 = [int(round(v)) for v in bx]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        m = None
        if masks is not None:
            m = cv2.resize(masks[i], (w, h), interpolation=cv2.INTER_LINEAR)
            m = (m > 0.5).astype(np.uint8) * 255
        persons.append(Person(idx=i, box=(x1, y1, x2, y2), conf=float(cf), mask=m))

    persons.sort(key=lambda p: p.area, reverse=True)
    for new_idx, p in enumerate(persons):
        p.idx = new_idx
    return persons


def detect_scene(rgb: np.ndarray, preprocess: str = "auto",
                 weights: str = C.YOLO_WEIGHTS) -> Scene:
    """
    Tiền xử lý (theo `preprocess`) -> YOLO. Nếu không thấy ai: thử lại với conf thấp hơn,
    imgsz lớn hơn và (khi preprocess="auto") chuỗi tiền xử lý mạnh hơn.
    """
    names, det_img, quality = PP.prepare(rgb, preprocess)
    auto = (preprocess or "auto").strip().lower() == "auto"

    persons: List[Person] = []
    attempt = 0
    for attempt, (conf, imgsz) in enumerate(C.DETECT_ATTEMPTS):
        if attempt > 0 and auto:
            names = [n for n in C.FALLBACK_PREPROCESS]
            det_img = PP.run_chain(rgb, names)
        persons = detect_persons(det_img, weights, conf=conf, imgsz=imgsz)
        if persons:
            break

    return Scene(rgb=rgb, persons=persons, det_rgb=det_img, quality=quality.to_dict(),
                 preprocess=names, attempt=attempt)
