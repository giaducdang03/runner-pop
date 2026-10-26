"""Registry nhỏ dùng chung cho các bước có thể cắm thêm (tiền xử lý, engine tách nền)."""
from __future__ import annotations

from typing import Callable, Dict, List


class Registry:
    """
        steps = Registry("preprocess")

        @steps.register("gamma", label="Làm sáng ảnh tối")
        def gamma(rgb): ...

        steps.get("gamma")(rgb)
    """

    def __init__(self, kind: str):
        self.kind = kind
        self._fns: Dict[str, Callable] = {}
        self._labels: Dict[str, str] = {}

    def register(self, name: str, label: str = ""):
        def deco(fn: Callable) -> Callable:
            if name in self._fns:
                raise ValueError(f"{self.kind} '{name}' đã được đăng ký")
            self._fns[name] = fn
            self._labels[name] = label or name
            return fn
        return deco

    def get(self, name: str) -> Callable:
        try:
            return self._fns[name]
        except KeyError:
            raise KeyError(f"Không có {self.kind} '{name}'. Hiện có: {', '.join(self._fns)}") from None

    def __contains__(self, name: str) -> bool:
        return name in self._fns

    def names(self) -> List[str]:
        return list(self._fns)

    def describe(self) -> List[dict]:
        return [{"name": n, "label": self._labels[n]} for n in self._fns]
