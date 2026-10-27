"""
FastAPI backend - tách người chạy, nền đen trắng.

Chạy:  uvicorn backend.main:app --host 127.0.0.1 --port 8000
(từ thư mục gốc của project)
"""
from __future__ import annotations

import io
import os
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from . import pipeline as P
from .pipeline import accel
from .report import build_report
from .pipeline import config as C
from .pipeline.cache import StageCache

MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", "8"))
MAX_UPLOAD_MB = 25

# --------------------------------------------------------------------------
# Warm-up (FR-106): nạp sẵn model ở thread nền, /health chỉ đọc trạng thái
# --------------------------------------------------------------------------
_WARMUP_INITIAL: Dict[str, str] = {"state": "idle", "detail": "not started"}   # idle | running | done | error
_WARMUP: Dict[str, str] = dict(_WARMUP_INITIAL)


def _set_warmup(state: str, detail: str = "") -> None:
    _WARMUP["state"], _WARMUP["detail"] = state, detail


def default_engine(names: List[str]) -> str:
    """Engine nhẹ chọn sẵn: rvm > modnet > cf, không có thì engine đầu tiên."""
    for n in ("rvm", "modnet", "cf"):
        if n in names:
            return n
    return names[0] if names else ""


def _do_warmup() -> str:
    """Làm nóng YOLO + engine cf (JIT numba) + engine mặc định trên ảnh giả. Có thể raise."""
    h = w = 320
    yy, xx = np.mgrid[0:h, 0:w]
    rgb = np.stack([xx * 255 // w, yy * 255 // h, (xx + yy) * 255 // (h + w)], -1).astype(np.uint8)
    rgb[40:300, 100:220] = (200, 60, 40)                     # "người" giả để matting có việc làm
    scene = P.detect_scene(rgb, "none", C.YOLO_WEIGHTS)       # nạp YOLO (ảnh giả không có người thật)
    person = P.Person(idx=-1, box=(100, 40, 220, 300), conf=1.0, mask=None)   # box thủ công
    names = P.engine_names()
    engine = default_engine(names)
    todo = ["cf"] if "cf" in names else []
    if engine and engine not in todo:
        todo.append(engine)
    for m in todo:
        P.build_cutout(scene, person, m, P.SegParams(device_mode="auto"))
    return "engines: " + ", ".join(todo)


def warmup() -> None:
    """Chạy trong thread; không bao giờ raise - lỗi chỉ làm state = error (app vẫn dùng được, nhưng nguội)."""
    if not C.WARMUP_ON_START:
        _set_warmup("done", "disabled")
        return
    try:
        _set_warmup("done", _do_warmup())
    except Exception as e:  # noqa: BLE001
        _set_warmup("error", f"{type(e).__name__}: {e}"[:200])


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if C.WARMUP_ON_START:
        _set_warmup("running", "")          # chỉ lifespan mới đặt "running", ngay trước khi chạy thread
    threading.Thread(target=warmup, daemon=True, name="warmup").start()
    yield


app = FastAPI(title="Runner B&W Background", version="1.0", lifespan=lifespan)


# --------------------------------------------------------------------------
# Session store (in-memory, LRU)
# --------------------------------------------------------------------------
@dataclass
class Job:
    """Một lượt chạy nền (detect / segment). Trace được ghi dần nên giao diện đọc được realtime."""
    trace: P.Trace
    state: str = "running"            # running | done | error
    error: Optional[str] = None
    status: int = 500                 # mã HTTP tương ứng khi lỗi
    result: Optional[dict] = None


@dataclass
class Session:
    rgb: np.ndarray                                  # uint8 ảnh làm việc
    scene: Optional[P.Scene] = None                  # có sau khi detect xong
    cutout: Optional[P.Cutout] = None
    jobs: Dict[str, Job] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)         # cài đặt lần tách nền gần nhất (cho báo cáo)
    cache: StageCache = field(default_factory=StageCache)  # kết quả trung gian theo khoá (FR-94)

    @property
    def persons(self) -> List[P.Person]:
        return self.scene.persons if self.scene else []


_sessions: "OrderedDict[str, Session]" = OrderedDict()
_lock = threading.Lock()


def _put(sid: str, s: Session) -> None:
    with _lock:
        _sessions[sid] = s
        _sessions.move_to_end(sid)
        while len(_sessions) > MAX_SESSIONS:
            _sessions.popitem(last=False)


def _get(sid: str) -> Session:
    with _lock:
        s = _sessions.get(sid)
        if s is None:
            raise HTTPException(404, "Session không tồn tại hoặc đã hết hạn. Hãy tải ảnh lại.")
        _sessions.move_to_end(sid)
        return s


def _encode(rgb_u8: np.ndarray, fmt: str, quality: int = 95) -> Tuple[bytes, str]:
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR) if rgb_u8.ndim == 3 else rgb_u8
    if fmt == "jpeg":
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        media = "image/jpeg"
    else:
        ok, buf = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        media = "image/png"
    if not ok:
        raise HTTPException(500, "Không encode được ảnh.")
    return buf.tobytes(), media


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class PersonOut(BaseModel):
    idx: int
    box: List[int]
    conf: float


class DetectOut(BaseModel):
    session_id: str
    width: int
    height: int
    persons: List[PersonOut]
    quality: dict = {}              # chỉ số chất lượng ảnh + cờ (dark, blurry, ...)
    preprocess: List[str] = []      # các bước tiền xử lý đã áp dụng cho detect
    attempt: int = 0                # 0 = phát hiện ngay lần đầu; >0 = phải thử lại
    suggested: int = 0              # người nên chọn khi chạy chế độ auto


class ProcessOut(BaseModel):
    session_id: str
    person: int
    method: str
    elapsed_s: float
    unknown_px: int


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.get("/health")
def health():
    try:
        import ultralytics  # noqa: F401
        yolo = True
    except Exception:
        yolo = False
    dev = accel.resolve()
    return {"ok": True, "yolo_available": yolo, "weights": C.YOLO_WEIGHTS,
            "device": dev.to_dict(), "backend": dev.backend,
            "available_backends": accel.available_backends(),
            "warmup": _WARMUP["state"], "warmup_detail": _WARMUP["detail"]}


@app.get("/options")
def options():
    """Danh sách bước có thể chọn - frontend dựng giao diện từ đây nên thêm bước mới không cần sửa UI."""
    return {"engines": P.available_engines(), "preprocess": P.available_preprocess(),
            "device": accel.resolve().to_dict(), "devices": ["auto", "cpu", "gpu"],
            "default_engine": default_engine(P.engine_names())}


def _read_upload(file: UploadFile) -> np.ndarray:
    data = file.file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Ảnh vượt quá {MAX_UPLOAD_MB} MB.")
    try:
        return P.resize_to_max_side(P.decode_image(data))
    except Exception:
        raise HTTPException(400, "Không đọc được file ảnh.")


def _detect_out(sid: str, scene: P.Scene) -> DetectOut:
    h, w = scene.rgb.shape[:2]
    return DetectOut(
        session_id=sid, width=w, height=h,
        persons=[PersonOut(idx=p.idx, box=list(p.box), conf=round(p.conf, 3)) for p in scene.persons],
        quality=scene.quality, preprocess=scene.preprocess, attempt=scene.attempt,
        suggested=scene.suggested,
    )


def _run_job(job: Job, fn) -> None:
    """Chạy fn() -> dict, gói kết quả / lỗi vào job."""
    try:
        job.result = fn()
        job.state = "done"
    except ImportError as e:
        job.error, job.status, job.state = f"Thiếu thư viện: {e}", 503, "error"
    except HTTPException as e:
        job.error, job.status, job.state = str(e.detail), e.status_code, "error"
    except KeyError as e:
        job.error, job.status, job.state = str(e.args[0]), 400, "error"
    except Exception as e:  # noqa: BLE001 - báo lỗi về giao diện thay vì chết thread im lặng
        job.error, job.status, job.state = f"{type(e).__name__}: {e}", 500, "error"


@app.post("/detect", response_model=None)
def detect(file: UploadFile = File(...),
           preprocess: str = Query("auto", description="auto | none | danh sách bước, vd: gamma,clahe"),
           device: str = Query("auto", pattern="^(auto|cpu|gpu)$", description="auto | cpu | gpu"),
           background: bool = Query(False, description="true = trả về ngay, theo dõi tại /session/{id}/job/detect")):
    rgb = _read_upload(file)
    sid = uuid.uuid4().hex[:12]
    s = Session(rgb=rgb)
    job = s.jobs["detect"] = Job(trace=P.Trace())
    _put(sid, s)

    def work() -> dict:
        s.cache.clear()
        s.scene = P.detect_scene(rgb, preprocess, C.YOLO_WEIGHTS, trace=job.trace,
                                 device_mode=device)
        return _detect_out(sid, s.scene).model_dump()

    if background:
        threading.Thread(target=_run_job, args=(job, work), daemon=True).start()
        return {"session_id": sid, "state": "running"}
    _run_job(job, work)
    if job.state == "error":
        raise HTTPException(job.status, job.error)
    return job.result


@app.get("/session/{sid}/job/{phase}")
def job_status(sid: str, phase: Literal["detect", "segment"], since: int = Query(0, ge=0)):
    """Tiến độ realtime: các bước mới (từ vị trí `since`), bước đang chạy, và kết quả khi xong."""
    s = _get(sid)
    job = s.jobs.get(phase)
    if job is None:
        raise HTTPException(404, "Chưa có tác vụ này.")
    state = job.state                       # đọc state TRƯỚC stages: 'done' thì stages đã đủ
    return {
        "state": state, "error": job.error, "status": job.status,
        "running": job.trace.current if state == "running" else None,
        "stages": job.trace.to_list(since),
        "result": job.result if state == "done" else None,
    }


@app.get("/session/{sid}/trace")
def trace(sid: str):
    """Các bước pipeline đã chạy (kèm ảnh minh hoạ + thời gian) để vẽ sơ đồ khi demo."""
    s = _get(sid)
    return {
        "detect": s.scene.trace.to_list() if s.scene else [],
        "segment": s.cutout.trace.to_list() if s.cutout and s.cutout.trace else [],
    }


@app.get("/session/{sid}/preview.jpg")
def preview(sid: str):
    """Ảnh làm việc + box đánh số của các người phát hiện được."""
    s = _get(sid)
    vis = cv2.cvtColor(s.rgb, cv2.COLOR_RGB2BGR).copy()
    th = max(2, vis.shape[1] // 400)
    for p in s.persons:
        x1, y1, x2, y2 = p.box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), th)
        label = f"#{p.idx} {p.conf:.2f}"
        fs = max(0.6, vis.shape[1] / 1600)
        cv2.putText(vis, label, (x1 + 4, max(18, y1 + 22)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 2, cv2.LINE_AA)
        cv2.putText(vis, label, (x1 + 4, max(18, y1 + 22)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 0), th, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return Response(buf.tobytes(), media_type="image/jpeg")


@app.post("/process/{sid}", response_model=None)
def process(
    sid: str,
    person: int = Query(0, ge=0, description="Chỉ số người trong /detect"),
    method: str = Query("cf", description="Tên engine, xem GET /options"),
    band: float = Query(0.02, ge=0.005, le=0.08, description="Bề rộng vùng unknown / kích thước người"),
    box: Optional[str] = Query(None, description="x1,y1,x2,y2 - dùng khi YOLO không phát hiện được"),
    focus: float = Query(0.3, ge=0.1, le=1.0, description="Ngưỡng độ nét so với người được chọn (method=focus)"),
    device: str = Query("auto", pattern="^(auto|cpu|gpu)$", description="auto | cpu | gpu"),
    background: bool = Query(False, description="true = trả về ngay, theo dõi tại /session/{id}/job/segment"),
):
    s = _get(sid)
    if s.scene is None:
        raise HTTPException(409, "Phát hiện người chưa xong.")
    if method not in P.engine_names():
        raise HTTPException(400, f"Engine '{method}' không tồn tại. Hiện có: {', '.join(P.engine_names())}")

    if box:
        try:
            x1, y1, x2, y2 = [int(float(v)) for v in box.split(",")]
        except Exception:
            raise HTTPException(400, "box phải có dạng x1,y1,x2,y2")
        h, w = s.rgb.shape[:2]
        x1, x2 = sorted((max(0, x1), min(w, x2)))
        y1, y2 = sorted((max(0, y1), min(h, y2)))
        if x2 - x1 < 20 or y2 - y1 < 20:
            raise HTTPException(400, "box quá nhỏ.")
        chosen = P.Person(idx=-1, box=(x1, y1, x2, y2), conf=1.0, mask=None)
    else:
        if not s.persons:
            raise HTTPException(422, "YOLO không phát hiện người nào. Hãy nhập box thủ công.")
        if person >= len(s.persons):
            raise HTTPException(400, "Chỉ số người không hợp lệ.")
        chosen = s.persons[person]

    job = s.jobs["segment"] = Job(trace=P.Trace())

    def work() -> dict:
        t0 = time.time()
        s.cutout = P.build_cutout(s.scene, chosen, method,
                                  P.SegParams(band_frac=band, focus_frac=focus,
                                              cache=s.cache, device_mode=device), trace=job.trace)
        s.meta = {"method": method, "band": band, "focus": focus, "person": chosen.idx}
        return ProcessOut(
            session_id=sid, person=chosen.idx, method=method,
            elapsed_s=round(time.time() - t0, 2),
            unknown_px=int((s.cutout.trimap == 128).sum()),
        ).model_dump()

    if background:
        threading.Thread(target=_run_job, args=(job, work), daemon=True).start()
        return {"session_id": sid, "state": "running"}
    _run_job(job, work)
    if job.state == "error":
        raise HTTPException(job.status, job.error)
    return job.result


def _cut(sid: str) -> P.Cutout:
    s = _get(sid)
    if s.cutout is None:
        raise HTTPException(409, "Chưa xử lý. Gọi /process trước.")
    return s.cutout


@app.get("/session/{sid}/original.jpg")
def original(sid: str):
    s = _get(sid)
    data, media = _encode(s.rgb, "jpeg", 95)
    return Response(data, media_type=media)


@app.get("/session/{sid}/foreground.png")
def foreground(sid: str):
    c = _cut(sid)
    data, media = _encode((np.clip(c.foreground, 0, 1) * 255 + 0.5).astype(np.uint8), "png")
    return Response(data, media_type=media)


@app.get("/session/{sid}/background.jpg")
def background(sid: str):
    """Nền đã ước lượng (sạch màu người) - frontend dùng để ghép realtime."""
    c = _cut(sid)
    data, media = _encode((np.clip(c.background, 0, 1) * 255 + 0.5).astype(np.uint8), "jpeg", 97)
    return Response(data, media_type=media)


@app.get("/session/{sid}/alpha.png")
def alpha(sid: str):
    c = _cut(sid)
    data, media = _encode(P.alpha_to_png(c.alpha), "png")
    return Response(data, media_type=media)


@app.get("/session/{sid}/trimap.png")
def trimap(sid: str):
    c = _cut(sid)
    data, media = _encode(c.trimap, "png")
    return Response(data, media_type=media)


@app.get("/session/{sid}/compose")
def compose(
    sid: str,
    intensity: float = Query(1.0, ge=0.0, le=1.0),
    fmt: str = Query("png", pattern="^(png|jpeg)$"),
):
    """Ảnh kết quả cuối (dùng để tải về). Slider realtime ở frontend dùng cùng công thức."""
    c = _cut(sid)
    out = P.compose(c, intensity)
    data, media = _encode(out, fmt, 95)
    return Response(data, media_type=media,
                    headers={"Content-Disposition": f'attachment; filename="result.{ "jpg" if fmt=="jpeg" else "png"}"'})


@app.get("/session/{sid}/report.pdf")
def report(sid: str, intensity: float = Query(1.0, ge=0.0, le=1.0)):
    """Báo cáo PDF: thông tin ảnh, cài đặt, hai pipeline với ảnh từng bước, kết quả trước / sau."""
    s = _get(sid)
    if s.scene is None:
        raise HTTPException(409, "Phát hiện người chưa xong.")
    labels = {e["name"]: e["label"] for e in P.available_engines()}
    try:
        data = build_report(s.rgb, s.scene, s.cutout, s.meta, intensity, labels)
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    return Response(data, media_type="application/pdf",
                    headers={"Content-Disposition": 'attachment; filename="report.pdf"'})
