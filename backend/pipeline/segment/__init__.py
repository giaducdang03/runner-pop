"""
Giai đoạn tách nền. Import các engine để chúng tự đăng ký vào ENGINES.
Thứ tự import = thứ tự hiện trên giao diện.
"""
from __future__ import annotations

import logging
from dataclasses import replace

from ..compose import compose
from ..held import apply_poles
from ..thin import overlay_poles
from ..schema import Cutout, Person, Scene, SegParams
from ..trace import Trace
from .base import ENGINES
from . import birefnet, focus, grabcut  # noqa: F401  (đăng ký engine)

log = logging.getLogger(__name__)


def _merge_poles(scene: Scene, person: Person, trace: Trace) -> None:
    """Gộp gậy ứng viên của người được xử lý (dò trước nếu scene dò trễ); lỗi chỉ ghi log, request vẫn chạy không gậy."""
    if not scene.pole_candidates and not scene.poles_lazy:
        return
    scanning = (scene.poles_lazy and person.idx not in scene.pole_candidates
                and any(p is person for p in scene.persons))
    try:
        if scanning:
            with trace.stage("held", "Dò gậy của người được chọn") as s:
                n = apply_poles(scene, person, scene.rgb.shape)
                s.note = f"{n} gậy"
            return
        n = apply_poles(scene, person, scene.rgb.shape)
    except Exception as e:      # không để lỗi gộp gậy làm hỏng request
        log.warning("apply_poles: %s", e)
        with trace.stage("held", "Gộp gậy của người được chọn", note=f"lỗi, bỏ qua gậy: {e}"):
            pass
        return
    if n:
        with trace.stage("held", "Gộp gậy của người được chọn", note=f"{n} gậy"):
            pass


def build_cutout(scene: Scene, person: Person, method: str = "cf",
                 params: SegParams | None = None, trace: Trace | None = None) -> Cutout:
    """Chạy engine `method`; mọi bước được ghi vào cutout.trace để giao diện vẽ pipeline."""
    trace = trace if trace is not None else Trace()
    params = replace(params or SegParams(), trace=trace)
    _merge_poles(scene, person, trace)   # trước engine/cache: cache key dùng person.box
    cutout = ENGINES.get(method)(scene, person, params)
    cutout = overlay_poles(cutout, scene, person, trace, params.device_mode)   # gậy mảnh ở độ phân giải gốc, mọi engine
    with trace.stage("compose", "Ghép: người giữ màu + nền (xem trước, đen trắng)",
                     note="out = α·F + (1−α)·gray(B)") as s:
        s.show(compose(cutout, 1.0))
    cutout.trace = trace
    return cutout


def available() -> list:
    return ENGINES.describe()


__all__ = ["ENGINES", "build_cutout", "available"]
