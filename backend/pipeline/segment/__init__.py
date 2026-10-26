"""
Giai đoạn tách nền. Import các engine để chúng tự đăng ký vào ENGINES.
Thứ tự import = thứ tự hiện trên giao diện.
"""
from __future__ import annotations

from ..types import Cutout, Person, Scene, SegParams
from .base import ENGINES
from . import birefnet, focus, grabcut  # noqa: F401  (đăng ký engine)


def build_cutout(scene: Scene, person: Person, method: str = "cf",
                 params: SegParams | None = None) -> Cutout:
    return ENGINES.get(method)(scene, person, params or SegParams())


def available() -> list:
    return ENGINES.describe()


__all__ = ["ENGINES", "build_cutout", "available"]
