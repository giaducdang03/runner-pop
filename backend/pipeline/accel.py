"""Lớp thiết bị: registry backend (cpu/cuda/mps/openvino), chọn thiết bị theo cấu hình,
khoá GPU, cache model có ngân sách bộ nhớ và rơi về CPU khi GPU lỗi.

Thêm backend mới = thêm một mục vào `BACKENDS`. Các gói tuỳ chọn (torch, openvino,
onnxruntime) chỉ được import trong hàm; thiếu gói hay probe lỗi => backend "không có".

Thứ tự khoá (tránh deadlock): LUÔN lấy GPU_LOCK trước, rồi mới tới khoá cache.
`compiled()` tuân thủ thứ tự này; không bao giờ gọi build() khi chỉ giữ khoá cache.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from . import config as C


@dataclass(frozen=True)
class Device:
    kind: str          # "cpu" | "gpu"
    backend: str
    name: str
    runtime: str       # "torch" | "openvino" | "onnxruntime"
    dtype: str         # "fp32" | "fp16"
    index: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def torch_device(self) -> str:
        if self.kind == "cpu":
            return "cpu"
        if self.backend == "cuda":
            return f"cuda:{self.index}"
        if self.backend == "mps":
            return "mps"
        if self.backend == "openvino":
            # Dạng có chỉ số "intel:gpu.N" chưa kiểm chứng trên phần cứng thật.
            return "intel:gpu" if self.index == 0 else f"intel:gpu.{self.index}"
        return "cpu"

    @property
    def ov_device_name(self) -> str:
        """Tên thiết bị để OpenVINO compile: "GPU" (chỉ số 0) hoặc "GPU.<index>"."""
        return "GPU" if self.index == 0 else f"GPU.{self.index}"

    @property
    def ort_providers(self) -> list:
        if self.kind == "gpu" and self.backend == "cuda":
            return [("CUDAExecutionProvider", {"device_id": self.index}), "CPUExecutionProvider"]
        if self.kind == "gpu" and self.backend == "mps":
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]


def device_label(device: "Device") -> str:
    """Nhãn ngắn cho trace/PDF: "CPU" hoặc "<backend>: <tên> (<dtype>)"."""
    if device.kind == "cpu":
        return "CPU"
    return f"{device.backend}: {device.name} ({device.dtype})"


@dataclass(frozen=True)
class Backend:
    name: str
    # Trả tên thiết bị nếu bộ tăng tốc chính dùng được, ngược lại None. Exception = None.
    probe: Callable[[], Optional[str]]


# --- probe (import trong hàm; mọi lỗi được resolve/available_backends nuốt) ---
def _probe_cuda() -> Optional[str]:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return str(torch.cuda.get_device_name(C.ACCEL_DEVICE_INDEX))


def _probe_mps() -> Optional[str]:
    if sys.platform != "darwin":
        return None
    try:
        import torch
    except ImportError:
        return None
    return "Apple GPU (MPS)" if torch.backends.mps.is_available() else None


def _probe_openvino() -> Optional[str]:
    try:
        import openvino as ov
    except ImportError:
        return None
    core = ov.Core()
    names = [n for n in core.available_devices if n == "GPU" or n.startswith("GPU.")]
    idx = int(C.ACCEL_DEVICE_INDEX)
    # Máy có nhiều GPU chỉ liệt kê "GPU.0", "GPU.1"...; máy một GPU có thể chỉ có "GPU".
    candidates = ["GPU", "GPU.0"] if idx == 0 else [f"GPU.{idx}"]
    for cand in candidates:
        if cand in names:
            return str(core.get_property(cand, "FULL_DEVICE_NAME"))
    return None


# Thứ tự khai báo = thứ tự ưu tiên của "auto".
BACKENDS: Dict[str, Backend] = {
    "cuda": Backend("cuda", _probe_cuda),
    "mps": Backend("mps", _probe_mps),
    "openvino": Backend("openvino", _probe_openvino),
}

_RUNTIME = {"cuda": "torch", "mps": "torch", "openvino": "openvino"}
_FP16_DEFAULT = {"cuda", "openvino"}


def _safe_probe(backend: Backend) -> Optional[str]:
    try:
        return backend.probe() or None
    except Exception:
        return None


def available_backends() -> Dict[str, str]:
    """{tên backend: tên thiết bị} của các backend có probe thành công."""
    out: Dict[str, str] = {}
    for name, be in list(BACKENDS.items()):
        dev_name = _safe_probe(be)
        if dev_name:
            out[name] = dev_name
    return out


def _cpu_device() -> Device:
    # CPU luôn fp32: ACCEL_DTYPE chỉ áp cho backend GPU.
    return Device("cpu", "cpu", "CPU", "torch", "fp32")


def _dtype_for(backend: str) -> str:
    if C.ACCEL_DTYPE in ("fp16", "fp32"):
        return C.ACCEL_DTYPE
    return "fp16" if backend in _FP16_DEFAULT else "fp32"


def resolve(mode: Optional[str] = None, backend: Optional[str] = None) -> Device:
    """Chọn thiết bị. Không bao giờ raise; mọi trường hợp không dùng được -> CPU."""
    try:
        mode = (mode or C.ACCEL_DEVICE or "auto").lower()
        backend = (backend or C.ACCEL_BACKEND or "auto").lower()
        if mode == "cpu" or backend == "cpu":
            return _cpu_device()
        if backend == "auto":
            for name, be in list(BACKENDS.items()):
                dev_name = _safe_probe(be)
                if dev_name:
                    return _make_gpu(name, dev_name)
            return _cpu_device()
        be = BACKENDS.get(backend)
        if be is None:
            return _cpu_device()
        dev_name = _safe_probe(be)
        return _make_gpu(backend, dev_name) if dev_name else _cpu_device()
    except Exception:
        return _cpu_device()


def _make_gpu(name: str, dev_name: str) -> Device:
    return Device("gpu", name, dev_name, _RUNTIME.get(name, "torch"),
                  _dtype_for(name), int(C.ACCEL_DEVICE_INDEX))


def openvino_ir_dir(weights: str) -> Path:
    """Thư mục IR OpenVINO của `weights`: <ACCEL_CACHE_DIR>/<tên file không đuôi>_openvino_model/."""
    return Path(C.ACCEL_CACHE_DIR) / f"{Path(weights).stem}_openvino_model"


# --- khoá GPU (FR-108): mọi việc GPU chạy tuần tự, áp cho mọi backend ---------
GPU_LOCK = threading.RLock()


# --- cache model đã nạp, LRU, có ngân sách bộ nhớ -----------------------------
_CACHE_LOCK = threading.RLock()
_MODELS: "OrderedDict[str, list]" = OrderedDict()   # key -> [obj, nbytes, last_used]


def loaded_bytes() -> int:
    with _CACHE_LOCK:
        return sum(e[1] for e in _MODELS.values())


def compiled(key: str, build: Callable[[], Any], nbytes: int) -> Any:
    """Trả model đã nạp theo `key`; chưa có thì `build()`. Vượt ngân sách thì giải phóng
    LRU trước khi nạp; một model không vừa dù đã giải phóng hết -> MemoryError."""
    with _CACHE_LOCK:
        entry = _MODELS.get(key)
        if entry is not None:
            entry[2] = time.monotonic()
            _MODELS.move_to_end(key)
            return entry[0]
    # Miss: thả khoá cache, lấy GPU_LOCK trước rồi mới tới khoá cache (xem docstring module).
    with GPU_LOCK, _CACHE_LOCK:
        entry = _MODELS.get(key)          # thread khác có thể đã nạp xong
        if entry is not None:
            entry[2] = time.monotonic()
            _MODELS.move_to_end(key)
            return entry[0]
        budget = int(C.ACCEL_MEM_BUDGET_MB) * 1024 * 1024
        if nbytes > budget:
            raise MemoryError(
                f"model '{key}' cần {nbytes} byte, vượt ngân sách {budget} byte")
        while _MODELS and loaded_bytes() + nbytes > budget:
            _MODELS.popitem(last=False)
        obj = build()
        _MODELS[key] = [obj, int(nbytes), time.monotonic()]
        return obj


def release(key: Optional[str] = None) -> None:
    """Giải phóng một model (hoặc tất cả khi key=None)."""
    with _CACHE_LOCK:
        if key is None:
            _MODELS.clear()
        else:
            _MODELS.pop(key, None)


def unload_idle(idle_s: Optional[float] = None) -> int:
    """Giải phóng model không dùng quá `idle_s` giây (mặc định C.ACCEL_UNLOAD_IDLE;
    <= 0 = không giải phóng). Trả số model đã giải phóng."""
    if idle_s is None:
        idle_s = C.ACCEL_UNLOAD_IDLE
    if idle_s <= 0:
        return 0
    now = time.monotonic()
    with _CACHE_LOCK:
        stale = [k for k, e in _MODELS.items() if now - e[2] >= idle_s]
        for k in stale:
            del _MODELS[k]
        return len(stale)


def with_fallback(device: Device, gpu_fn: Callable[[], Any], cpu_fn: Callable[[], Any],
                  stage=None) -> Tuple[Any, Device]:
    """Chạy gpu_fn dưới GPU_LOCK khi device là GPU; mọi Exception (kể cả MemoryError)
    -> chạy cpu_fn, trả Device CPU và ghi chú vào stage.note (FR-104)."""
    if device.kind != "gpu":
        return cpu_fn(), device if device.kind == "cpu" else _cpu_device()
    try:
        with GPU_LOCK:
            return gpu_fn(), device
    except Exception as e:  # noqa: BLE001 - mọi lỗi GPU đều phải rơi về CPU
        if stage is not None:
            reason = str(e) or type(e).__name__
            stage.note = (getattr(stage, "note", "") or "") + f" · GPU lỗi, đã dùng CPU: {reason}"
        return cpu_fn(), _cpu_device()


# --- nhớ lỗi build (R22): lỗi bền không bị thử lại ở mỗi request ----------------
_FAIL_LOCK = threading.Lock()
_BUILD_FAILED: Dict[str, float] = {}    # khoá -> thời điểm (monotonic) build lỗi gần nhất
_clock = time.monotonic                 # test thay bằng đồng hồ giả


def remember_failure(key: str) -> None:
    with _FAIL_LOCK:
        _BUILD_FAILED[key] = _clock()


def failed_recently(key: str) -> bool:
    """True nếu `key` build lỗi chưa quá C.ACCEL_EXPORT_RETRY_S giây."""
    with _FAIL_LOCK:
        t = _BUILD_FAILED.get(key)
        if t is None:
            return False
        if _clock() - t < C.ACCEL_EXPORT_RETRY_S:
            return True
        del _BUILD_FAILED[key]
        return False


def forget_failures(key: Optional[str] = None) -> None:
    """Xoá ghi nhớ lỗi của `key` (hoặc tất cả khi key=None)."""
    with _FAIL_LOCK:
        if key is None:
            _BUILD_FAILED.clear()
        else:
            _BUILD_FAILED.pop(key, None)


# --- runner ONNX đa backend (BiRefNet, RVM, MODNet) ---------------------------
Runner = Callable[[Dict[str, np.ndarray]], List[np.ndarray]]

# EP bắt buộc phải đứng đầu session khi chạy GPU qua onnxruntime (R19).
_ORT_ACCEL_EP = {"cuda": "CUDAExecutionProvider", "mps": "CoreMLExecutionProvider"}


def _runner_key(key: str, device: Device) -> str:
    return f"{key}:{device.backend}:{device.index}"


def onnx_runner_failed(key: str, device: Device) -> bool:
    """True nếu runner `key` trên `device` vừa build lỗi (còn trong thời gian nhớ lỗi):
    nơi gọi nên chạy thẳng đường CPU thay vì thử GPU rồi mới rơi về."""
    return failed_recently(_runner_key(key, device))


def _ov_convert(ov, onnx_path: str, static_input):
    """Tách riêng để test bắt được đối số; `input` = {tên input: shape} cố định."""
    if static_input:
        return ov.convert_model(onnx_path, input=static_input)
    return ov.convert_model(onnx_path)


def _build_openvino(onnx_path: str, device: Device, static_input) -> Runner:
    import openvino as ov

    model = _ov_convert(ov, onnx_path, static_input)
    cfg = {"CACHE_DIR": C.ACCEL_CACHE_DIR,
           "INFERENCE_PRECISION_HINT": "f16" if device.dtype == "fp16" else "f32"}
    cm = ov.Core().compile_model(model, device.ov_device_name, cfg)
    outputs = list(cm.outputs)

    def run(feeds: Dict[str, np.ndarray]) -> List[np.ndarray]:
        res = cm(feeds)
        return [np.asarray(res[o]) for o in outputs]
    return run


def _build_ort(onnx_path: str, device: Device) -> Runner:
    import onnxruntime as ort

    want = _ORT_ACCEL_EP.get(device.backend) if device.kind == "gpu" else None
    if want is not None and want not in ort.get_available_providers():
        raise RuntimeError(f"onnxruntime thiếu {want} (cần onnxruntime-gpu/CoreML)")
    sess = ort.InferenceSession(onnx_path, providers=device.ort_providers)
    if want is not None:
        active = sess.get_providers()
        if not active or active[0] != want:
            # R19: không để model lặng lẽ chạy CPU trong khi Device báo GPU.
            raise RuntimeError(f"session onnxruntime không chạy {want} (đang: {active})")

    def run(feeds: Dict[str, np.ndarray]) -> List[np.ndarray]:
        return list(sess.run(None, feeds))
    return run


def onnx_runner(key: str, onnx_path: str, device: Device,
                static_input: Optional[Dict[str, Tuple[int, ...]]] = None) -> Runner:
    """Runner cho model ONNX trên `device`: nhận {tên input: mảng}, trả list output theo
    thứ tự của model. openvino: convert (shape tĩnh `static_input`) + compile trên GPU;
    cuda/mps/cpu: onnxruntime với `device.ort_providers`. Cache qua `compiled()` theo
    (key, backend, index). Build lỗi -> RuntimeError (để `with_fallback` về CPU) và lỗi
    được nhớ C.ACCEL_EXPORT_RETRY_S giây (R22)."""
    ck = _runner_key(key, device)
    if failed_recently(ck):
        raise RuntimeError(f"runner '{ck}' vừa build lỗi, chưa thử lại")

    def build() -> Runner:
        try:
            if device.kind == "gpu" and device.backend == "openvino":
                runner = _build_openvino(onnx_path, device, static_input)
            else:
                runner = _build_ort(onnx_path, device)
        except Exception:
            remember_failure(ck)
            raise
        forget_failures(ck)
        return runner

    return compiled(ck, build, nbytes=os.path.getsize(onnx_path))
