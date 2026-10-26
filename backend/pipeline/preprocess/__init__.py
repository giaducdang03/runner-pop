"""
Tiền xử lý cho bước phát hiện.

    names, det_img, quality = prepare(rgb, "auto")        # chọn theo chất lượng ảnh
    names, det_img, quality = prepare(rgb, "gamma,clahe") # chỉ định tay
    names, det_img, quality = prepare(rgb, "none")        # không xử lý
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .quality import Quality, assess, plan_steps
from .steps import STEPS


def parse_spec(spec: str, q: Quality) -> List[str]:
    spec = (spec or "auto").strip().lower()
    if spec == "auto":
        return plan_steps(q)
    if spec in ("", "none"):
        return []
    names = [s.strip() for s in spec.split(",") if s.strip()]
    for n in names:
        STEPS.get(n)                      # báo lỗi sớm nếu tên sai
    return names


def run_chain(rgb: np.ndarray, names: List[str]) -> np.ndarray:
    out = rgb
    for n in names:
        out = STEPS.get(n)(out)
    return out


def prepare(rgb: np.ndarray, spec: str = "auto") -> Tuple[List[str], np.ndarray, Quality]:
    q = assess(rgb)
    names = parse_spec(spec, q)
    return names, run_chain(rgb, names), q


def available() -> List[dict]:
    return STEPS.describe()


__all__ = ["prepare", "run_chain", "parse_spec", "assess", "plan_steps", "available", "Quality", "STEPS"]
