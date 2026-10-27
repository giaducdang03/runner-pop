"""Phần dùng chung của công cụ đo: nạp ảnh, dựng Scene, chạy engine, bật/tắt FULLRES."""
from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: str, limit: int) -> List[Path]:
    """`limit` ảnh đầu theo thứ tự tên đã sắp xếp (deterministic)."""
    files = sorted((p for p in Path(folder).iterdir() if p.suffix.lower() in EXTS), key=lambda p: p.name)
    return files[:limit] if limit > 0 else files


@contextmanager
def fullres(on: bool) -> Iterator[None]:
    """Bật/tắt FULLRES ngay trong tiến trình: `C.FULLRES` được đọc từ env MỘT lần lúc import,
    còn `active_roi` đọc lại `C.FULLRES` mỗi lần gọi nên đảo cờ lúc chạy là đủ (không cần subprocess)."""
    from backend.pipeline import config as C
    old = C.FULLRES
    C.FULLRES = on
    try:
        yield
    finally:
        C.FULLRES = old


def load_scene(path: Path):
    """Đọc ảnh -> Scene (YOLO, chạy 1 lần/ảnh) và người chính. Người = None nếu không thấy ai."""
    from backend import pipeline as P
    from backend.pipeline import config as C
    rgb = P.resize_to_max_side(P.decode_image(path.read_bytes()))
    scene = P.detect_scene(rgb, "auto", C.YOLO_WEIGHTS)
    if not scene.persons:
        return scene, None
    return scene, scene.persons[min(scene.suggested, len(scene.persons) - 1)]


def run_engine(scene, person, engine: str, device: str = "auto", full: bool = False):
    """Chạy một engine với cache mới (không dùng lại kết quả cũ). Trả (cutout, giây, trace)."""
    from backend import pipeline as P
    from backend.pipeline.cache import StageCache
    trace = P.Trace()
    params = P.SegParams(cache=StageCache(), device_mode=device)
    with fullres(full):
        t0 = time.perf_counter()
        cutout = P.build_cutout(scene, person, engine, params, trace=trace)
        dt = time.perf_counter() - t0
    return cutout, dt, trace
