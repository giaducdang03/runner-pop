"""
Giai đoạn tách nền. Import các engine để chúng tự đăng ký vào ENGINES.
Thứ tự import = thứ tự hiện trên giao diện.
"""
from __future__ import annotations

from dataclasses import replace

from ..compose import compose
from ..schema import Cutout, Person, Scene, SegParams
from ..trace import Trace
from .base import ENGINES
from . import birefnet, focus, grabcut  # noqa: F401  (đăng ký engine)


def build_cutout(scene: Scene, person: Person, method: str = "cf",
                 params: SegParams | None = None, trace: Trace | None = None) -> Cutout:
    """Chạy engine `method`; mọi bước được ghi vào cutout.trace để giao diện vẽ pipeline."""
    trace = trace if trace is not None else Trace()
    params = replace(params or SegParams(), trace=trace)
    cutout = ENGINES.get(method)(scene, person, params)
    with trace.stage("compose", "Ghép: người giữ màu, nền đen trắng",
                     note="out = α·F + (1−α)·gray(B)") as s:
        s.show(compose(cutout, 1.0))
    cutout.trace = trace
    return cutout


def available() -> list:
    return ENGINES.describe()


__all__ = ["ENGINES", "build_cutout", "available"]
