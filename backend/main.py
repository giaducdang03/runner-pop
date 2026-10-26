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
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from . import pipeline as P
from .pipeline import config as C

MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", "8"))
MAX_UPLOAD_MB = 25

app = FastAPI(title="Runner B&W Background", version="1.0")


# --------------------------------------------------------------------------
# Session store (in-memory, LRU)
# --------------------------------------------------------------------------
@dataclass
class Session:
    scene: P.Scene                                   # ảnh làm việc + người phát hiện được
    cutout: Optional[P.Cutout] = None

    @property
    def rgb(self) -> np.ndarray:
        return self.scene.rgb

    @property
    def persons(self) -> List[P.Person]:
        return self.scene.persons


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
    return {"ok": True, "yolo_available": yolo, "weights": C.YOLO_WEIGHTS}


@app.get("/options")
def options():
    """Danh sách bước có thể chọn - frontend dựng giao diện từ đây nên thêm bước mới không cần sửa UI."""
    return {"engines": P.available_engines(), "preprocess": P.available_preprocess()}


@app.post("/detect", response_model=DetectOut)
def detect(file: UploadFile = File(...),
           preprocess: str = Query("auto", description="auto | none | danh sách bước, vd: gamma,clahe")):
    data = file.file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Ảnh vượt quá {MAX_UPLOAD_MB} MB.")
    try:
        rgb = P.resize_to_max_side(P.decode_image(data))
    except Exception:
        raise HTTPException(400, "Không đọc được file ảnh.")

    try:
        scene = P.detect_scene(rgb, preprocess, C.YOLO_WEIGHTS)
    except ImportError:
        raise HTTPException(503, "Chưa cài ultralytics (pip install ultralytics).")
    except KeyError as e:
        raise HTTPException(400, str(e.args[0]))

    sid = uuid.uuid4().hex[:12]
    _put(sid, Session(scene=scene))
    h, w = rgb.shape[:2]
    return DetectOut(
        session_id=sid, width=w, height=h,
        persons=[PersonOut(idx=p.idx, box=list(p.box), conf=round(p.conf, 3)) for p in scene.persons],
        quality=scene.quality, preprocess=scene.preprocess, attempt=scene.attempt,
        suggested=P.suggest_person(scene),
    )


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


@app.post("/process/{sid}", response_model=ProcessOut)
def process(
    sid: str,
    person: int = Query(0, ge=0, description="Chỉ số người trong /detect"),
    method: str = Query("cf", description="Tên engine, xem GET /options"),
    band: float = Query(0.02, ge=0.005, le=0.08, description="Bề rộng vùng unknown / kích thước người"),
    box: Optional[str] = Query(None, description="x1,y1,x2,y2 - dùng khi YOLO không phát hiện được"),
    focus: float = Query(0.3, ge=0.1, le=1.0, description="Ngưỡng độ nét so với người được chọn (method=focus)"),
):
    s = _get(sid)
    t0 = time.time()
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

    s.cutout = P.build_cutout(s.scene, chosen, method,
                              P.SegParams(band_frac=band, focus_frac=focus))
    return ProcessOut(
        session_id=sid, person=chosen.idx, method=method,
        elapsed_s=round(time.time() - t0, 2),
        unknown_px=int((s.cutout.trimap == 128).sum()),
    )


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
