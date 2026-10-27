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


def is_warming(warmup_state: Optional[str]) -> bool:
    """Chỉ "running" mới chặn giao diện; idle / done / error / thiếu khoá đều coi là không còn warm-up."""
    return warmup_state == "running"


def should_poll(warmup_state: Optional[str]) -> bool:
    """Có cần hỏi lại /health định kỳ không (cho tới khi warm-up xong)."""
    return is_warming(warmup_state)


def should_auto_run(auto: bool, has_target: bool, sig, last_sig: Optional[object], warming: bool,
                    engine_ready: bool = True) -> bool:
    """Chạy tự động khi: bật auto, có người/box, tổ hợp tham số mới, không warm-up và model đã nạp xong."""
    return bool(auto and has_target and not warming and engine_ready and last_sig != sig)


def engine_ready(state: Optional[str]) -> bool:
    """Model của engine đã dùng được chưa. Chỉ idle / running chặn giao diện (model đang nạp, GPU_LOCK bị giữ).
    done -> sẵn sàng; error -> không chặn (người dùng vẫn được thử); None / lạ -> backend không có /prepare
    hoặc không kết nối được, coi như không chặn để app vẫn dùng được."""
    return state not in ("idle", "running")


def should_poll_prepare(state: Optional[str]) -> bool:
    """Hỏi lại GET /prepare định kỳ khi đang nạp (idle = vừa POST xong, chưa kịp sang running)."""
    return state in ("idle", "running")


def prepare_message(engine_label: str, device_mode: str) -> str:
    return (f"Đang nạp model cho '{engine_label}' trên {device_mode}… "
            "(lần đầu có thể mất vài phút do biên dịch)")


def prepare_note(device_mode: str, used_device: str) -> str:
    """Ghi chú khi yêu cầu GPU nhưng thực tế phải dùng CPU (fallback); rỗng nếu không cần."""
    if device_mode == "gpu" and used_device.strip().upper().startswith("CPU"):
        return f"Đã dùng CPU thay vì GPU ({used_device})."
    return ""
