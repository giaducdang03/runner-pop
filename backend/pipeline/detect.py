"""Phát hiện người (YOLO-seg) + thang thử lại khi ảnh kém chất lượng."""
from __future__ import annotations

import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import accel
from . import config as C
from . import preprocess as PP
from .masks import box_to_mask
from .schema import Person, Scene
from .select import suggest_person
from .trace import Trace

_MODELS: Dict[tuple, object] = {}      # (weights, backend, index) -> model
_MODELS_LOCK = threading.Lock()         # detect có thể được gọi từ nhiều luồng job
log = logging.getLogger(__name__)


def _load_yolo(path, **kw):
    """Nạp YOLO (import lười; test thay hàm này bằng bản giả)."""
    from ultralytics import YOLO
    return YOLO(str(path), **kw)


def predict_kwargs(device: accel.Device) -> dict:
    """Tham số thiết bị cho model.predict; MPS không bao giờ half."""
    return {"device": device.torch_device,
            "half": device.backend == "cuda" and device.dtype == "fp16"}


def export_openvino(weights: str = C.YOLO_WEIGHTS) -> Optional[Path]:
    """Xuất YOLO sang IR OpenVINO FP16 vào thư mục cache (chỉ dùng cho backend openvino).
    Đã có thư mục thì dùng lại. Lỗi bất kỳ -> None (FR-102)."""
    ir_dir = accel.openvino_ir_dir(weights)
    try:
        if ir_dir.is_dir():
            return ir_dir
        out = _load_yolo(weights).export(format="openvino", half=True, dynamic=False)
        ir_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(out), str(ir_dir))
        return ir_dir
    except Exception as e:  # noqa: BLE001
        log.warning("Xuất OpenVINO IR thất bại, dùng .pt: %s", e)
        return None


_EXPORT_LOCK = threading.Lock()         # xuất IR tuần tự; không giữ cùng lúc với _MODELS_LOCK
_IR: Dict[str, Optional[Path]] = {}     # weights -> thư mục IR đã xuất thành công
_EXPORT_FAILED: Dict[str, float] = {}   # weights -> thời điểm (monotonic) xuất lỗi gần nhất
_clock = time.monotonic                 # test thay bằng đồng hồ giả


def _ir_for(weights: str) -> Optional[Path]:
    """IR OpenVINO của `weights`, xuất một lần. Lỗi được nhớ trong ACCEL_EXPORT_RETRY_S giây
    (không xuất lại mỗi request), sau đó thử lại; kết quả thành công được cache."""
    with _EXPORT_LOCK:
        ir = _IR.get(weights)
        if ir is not None:
            return ir
        failed_at = _EXPORT_FAILED.get(weights)
        if failed_at is not None and _clock() - failed_at < C.ACCEL_EXPORT_RETRY_S:
            return None
        ir = export_openvino(weights)
        if ir is None:
            _EXPORT_FAILED[weights] = _clock()
            return None
        _EXPORT_FAILED.pop(weights, None)
        _IR[weights] = ir
        return ir


def _build_model(weights: str, device: accel.Device):
    if device.backend == "openvino":
        ir = _ir_for(weights)
        if ir is None:
            # Không nạp .pt dưới khoá openvino: model torch không chạy được trên "intel:gpu".
            raise RuntimeError("OpenVINO IR không dùng được")
        return _load_yolo(ir, task="segment")
    return _load_yolo(weights)


def get_yolo(weights: str = C.YOLO_WEIGHTS, device: Optional[accel.Device] = None):
    """Model YOLO theo backend, cache theo (weights, backend, index). CPU/cuda/mps nạp .pt;
    openvino nạp IR (thất bại thì .pt)."""
    device = device if device is not None else accel.resolve()
    key = (weights, device.backend, device.index)
    with _MODELS_LOCK:
        m = _MODELS.get(key)
    if m is not None:
        return m
    m = _build_model(weights, device)      # chậm: không giữ khoá cache
    with _MODELS_LOCK:
        return _MODELS.setdefault(key, m)


def _detect_with_device(det_rgb: np.ndarray, weights: str, conf: float, imgsz: int,
                        device: accel.Device, stage=None) -> Tuple[List[Person], accel.Device]:
    def run(dev: accel.Device) -> List[Person]:
        model = get_yolo(weights, dev)
        kw = predict_kwargs(dev)
        if not kw["half"]:
            kw.pop("half")      # ultralytics cảnh báo 'half' deprecated khi truyền half=False
        res = model.predict(det_rgb, imgsz=imgsz, conf=conf, classes=[C.PERSON_CLASS],
                            verbose=False, retina_masks=True,   # mask độ phân giải gốc -> không mất tay/chân mảnh
                            **kw)[0]
        return _persons_from_result(res, det_rgb)

    cpu = accel.resolve("cpu")
    if device.kind == "gpu" and device.backend == "openvino" and _ir_for(weights) is None:
        # IR không dùng được: chạy thẳng .pt trên CPU, không thử GPU rồi mới rơi về.
        if stage is not None:
            stage.note = (getattr(stage, "note", "") or "") + " · OpenVINO IR không dùng được, dùng CPU"
        return run(cpu), cpu
    return accel.with_fallback(device, lambda: run(device), lambda: run(cpu), stage=stage)


def detect_persons(det_rgb: np.ndarray, weights: str = C.YOLO_WEIGHTS,
                   conf: float = 0.25, imgsz: int = C.YOLO_IMGSZ,
                   device: Optional[accel.Device] = None) -> List[Person]:
    """Trả về danh sách người, sắp xếp theo diện tích box giảm dần."""
    device = device if device is not None else accel.resolve()
    return _detect_with_device(det_rgb, weights, conf, imgsz, device)[0]


def _persons_from_result(res, det_rgb: np.ndarray) -> List[Person]:
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
                 weights: str = C.YOLO_WEIGHTS, trace: Optional[Trace] = None,
                 device_mode: str = "auto") -> Scene:
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
    device = accel.resolve(device_mode)
    attempt = 0
    for attempt, (conf, imgsz) in enumerate(C.DETECT_ATTEMPTS):
        if attempt > 0 and auto:
            names = list(C.FALLBACK_PREPROCESS)
            det_img = _run_chain_traced(trace, rgb, names, prefix=f"Thử lại {attempt}: ")
        label = "YOLO-seg phát hiện người" + (f" (thử lại {attempt})" if attempt else "")
        with trace.stage("yolo", label, note=f"conf ≥ {conf}, imgsz {imgsz}") as s:
            persons, device = _detect_with_device(det_img, weights, conf, imgsz, device, stage=s)
            s.set_device(device)
            s.note += f" → {len(persons)} người · " + (
                "CPU" if device.kind == "cpu" else f"{device.backend}: {device.name}")
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
                 preprocess=names, attempt=attempt, suggested=suggested, trace=trace,
                 device=device.to_dict())
