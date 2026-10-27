"""Cache kết quả trung gian theo khoá (FR-94): phần nặng chỉ chạy một lần cho mỗi khoá."""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Callable


class StageCache:
    """LRU nhỏ, khoá là tuple hashable bất kỳ.

    Khoá dựng từ một người phải gồm cả `person.box` (box thủ công đều có idx = -1).
    """

    def __init__(self, maxsize: int = 32) -> None:
        self.maxsize = maxsize
        self._data: "OrderedDict[tuple, Any]" = OrderedDict()
        self._lock = threading.RLock()

    def get_or(self, key: tuple, build: Callable[[], Any]) -> Any:
        """Trả bản đã lưu nếu khoá trùng, ngược lại gọi `build()` và lưu lại."""
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                return self._data[key]
            value = build()
            self._data[key] = value
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)
            return value

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        return len(self._data)
