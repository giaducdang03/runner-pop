"""Hàm thuần cho giao diện (không gọi streamlit) để test được."""
from __future__ import annotations

from typing import List, Optional


def device_line(dev: dict) -> str:
    """Dòng hiển thị thiết bị đang dùng ở sidebar."""
    return (f"Thiết bị: {dev.get('name', '?')} · {dev.get('backend', '?')} "
            f"({dev.get('runtime', '?')}/{dev.get('dtype', '?')})")


def pick_default_engine(opts: dict, names: List[str]) -> str:
    """Engine chọn sẵn: `default_engine` từ backend nếu có trong danh sách, không thì engine đầu tiên."""
    d = opts.get("default_engine")
    return d if d in names else names[0]


def should_auto_run(auto: bool, has_target: bool, sig, last_sig: Optional[object], warming: bool) -> bool:
    """Chạy tự động khi: bật auto, có người/box, tổ hợp tham số mới, và backend không đang warm-up."""
    return bool(auto and has_target and not warming and last_sig != sig)
