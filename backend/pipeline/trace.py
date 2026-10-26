"""
Ghi lại từng bước của pipeline (tên, thời gian, ảnh trung gian) để giao diện vẽ lại
dưới dạng sơ đồ pipeline khi demo.

    trace = Trace()
    with trace.stage("grabcut", "GrabCut tinh chỉnh biên", note="4 vòng") as s:
        mask = refine(...)
        s.overlay(rgb, mask)          # hoặc s.show(img) / s.show(float_map, heat=True)
    trace.to_list()                   # [{key, label, seconds, note, image: "data:image/jpeg;base64,..."}]
"""
from __future__ import annotations

import base64
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

THUMB = 720      # cạnh dài tối đa của ảnh minh hoạ (đủ rõ khi phóng to)


@dataclass
class Stage:
    key: str
    label: str
    seconds: float
    note: str = ""
    image: Optional[bytes] = None      # JPEG


def to_vis(img: np.ndarray, heat: bool = False) -> np.ndarray:
    """Mọi kiểu ảnh/mask/bản đồ -> uint8 HxWx3 RGB."""
    a = np.asarray(img)
    if a.dtype == bool:
        a = a.astype(np.uint8) * 255
    elif a.dtype != np.uint8:
        a = a.astype(np.float32)
        if heat:
            a = a / (np.percentile(a, 99) + 1e-6)
        a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
        if heat and a.ndim == 2:
            return cv2.cvtColor(cv2.applyColorMap(a, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    return a


def overlay(rgb: np.ndarray, mask: np.ndarray, color=(40, 220, 90)) -> np.ndarray:
    """Tô màu vùng mask lên ảnh + vẽ đường viền."""
    m = mask > 127
    out = rgb.astype(np.float32).copy()
    out[m] = out[m] * 0.55 + np.array(color, np.float32) * 0.45
    out = out.astype(np.uint8)
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, max(1, rgb.shape[1] // 500))
    return out


def _encode(rgb: np.ndarray) -> bytes:
    h, w = rgb.shape[:2]
    s = THUMB / max(h, w)
    if s < 1:
        rgb = cv2.resize(rgb, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else b""


class _Handle:
    def __init__(self, note: str):
        self.note = note
        self._image: Optional[bytes] = None

    def show(self, img: np.ndarray, heat: bool = False) -> None:
        self._image = _encode(to_vis(img, heat))

    def overlay(self, rgb: np.ndarray, mask: np.ndarray, color=(40, 220, 90)) -> None:
        self._image = _encode(overlay(rgb, mask, color))


class Trace:
    def __init__(self):
        self.stages: List[Stage] = []
        self.current: Optional[str] = None      # nhãn bước đang chạy (để giao diện hiện realtime)

    @contextmanager
    def stage(self, key: str, label: str, note: str = ""):
        h = _Handle(note)
        self.current = label
        t0 = time.perf_counter()
        try:
            yield h
        finally:
            self.stages.append(Stage(key, label, time.perf_counter() - t0, h.note, h._image))
            self.current = None

    def to_list(self, start: int = 0) -> List[dict]:
        out = []
        for s in list(self.stages)[start:]:
            d = {"key": s.key, "label": s.label, "seconds": round(s.seconds, 3), "note": s.note, "image": None}
            if s.image:
                d["image"] = "data:image/jpeg;base64," + base64.b64encode(s.image).decode()
            out.append(d)
        return out
