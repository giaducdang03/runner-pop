"""Phát hiện người (YOLO-seg) + thang thử lại khi ảnh kém chất lượng."""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from . import config as C
from . import preprocess as PP
from .masks import box_to_mask
from .schema import Person, Scene
from .select import suggest_person
from .trace import Trace

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


def _draw_boxes(rgb: np.ndarray, persons: List[Person]) -> np.ndarray:
    vis = rgb.copy()
    th = max(2, vis.shape[1] // 400)
    for p in persons:
        x1, y1, x2, y2 = p.box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), th)
        cv2.putText(vis, f"#{p.idx}", (x1 + 4, max(18, y1 + 22)), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.6, vis.shape[1] / 1600), (0, 255, 0), th, cv2.LINE_AA)
    return vis


def _run_chain_traced(trace: Trace, rgb: np.ndarray, names: List[str], prefix: str = "") -> np.ndarray:
    out = rgb
    for n in names:
        with trace.stage(f"pre_{n}", f"{prefix}{PP.STEPS.label(n)}", note=n) as s:
            out = PP.STEPS.get(n)(out)
            s.show(out)
    return out


def detect_scene(rgb: np.ndarray, preprocess: str = "auto",
                 weights: str = C.YOLO_WEIGHTS, trace: Optional[Trace] = None) -> Scene:
    """
    Tiền xử lý (theo `preprocess`) -> YOLO. Nếu không thấy ai: thử lại với conf thấp hơn,
    imgsz lớn hơn và (khi preprocess="auto") chuỗi tiền xử lý mạnh hơn.
    Mọi bước được ghi vào `trace` (truyền vào để theo dõi realtime) và scene.trace.
    """
    trace = trace if trace is not None else Trace()
    h, w = rgb.shape[:2]
    with trace.stage("load", "Ảnh đầu vào (đã resize)", note=f"{w}×{h}px") as s:
        s.show(rgb)

    with trace.stage("quality", "Đánh giá chất lượng ảnh") as s:
        quality = PP.assess(rgb)
        qd = quality.to_dict()
        s.note = ("vấn đề: " + ", ".join(qd["flags"])) if qd["flags"] else "ảnh tốt"
        s.note += (f" · sáng {qd['brightness']:.0f}, tương phản {qd['contrast']:.0f}, "
                   f"nét {qd['sharpness']:.0f}, nhiễu {qd['noise']:.1f}")
    names = PP.parse_spec(preprocess, quality)
    auto = (preprocess or "auto").strip().lower() == "auto"
    det_img = _run_chain_traced(trace, rgb, names)
    if not names:
        with trace.stage("pre_none", "Không tiền xử lý", note="ảnh đưa thẳng vào YOLO"):
            pass

    persons: List[Person] = []
    attempt = 0
    for attempt, (conf, imgsz) in enumerate(C.DETECT_ATTEMPTS):
        if attempt > 0 and auto:
            names = list(C.FALLBACK_PREPROCESS)
            det_img = _run_chain_traced(trace, rgb, names, prefix=f"Thử lại {attempt}: ")
        label = "YOLO-seg phát hiện người" + (f" (thử lại {attempt})" if attempt else "")
        with trace.stage("yolo", label, note=f"conf ≥ {conf}, imgsz {imgsz}") as s:
            persons = detect_persons(det_img, weights, conf=conf, imgsz=imgsz)
            s.note += f" → {len(persons)} người"
            s.show(_draw_boxes(det_img, persons))
        if persons:
            break

    suggested = suggest_person(Scene(rgb=rgb, persons=persons)) if persons else 0
    if persons:
        with trace.stage("select", "Tự chọn người chạy chính",
                         note=f"#{suggested}: to + gần tâm + nét + conf cao") as s:
            s.overlay(rgb, persons[suggested].mask if persons[suggested].mask is not None
                      else box_to_mask(persons[suggested].box, rgb.shape))

    return Scene(rgb=rgb, persons=persons, det_rgb=det_img, quality=quality.to_dict(),
                 preprocess=names, attempt=attempt, suggested=suggested, trace=trace)
