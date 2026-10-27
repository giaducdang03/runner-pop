"""Test lớp thiết bị accel.py: chỉ chạy CPU, mọi probe/backend đều được giả lập."""
import threading
import time

import pytest

from backend.pipeline import accel, config as C
from backend.pipeline.accel import (
    Backend, Device, available_backends, compiled, loaded_bytes, release,
    resolve, unload_idle, with_fallback,
)


def _set_backends(monkeypatch, **probes):
    """Thay BACKENDS bằng probe giả, giữ thứ tự khai báo."""
    monkeypatch.setattr(accel, "BACKENDS", {n: Backend(n, p) for n, p in probes.items()})


def _none():
    return None


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(C, "ACCEL_DTYPE", "auto")
    monkeypatch.setattr(C, "ACCEL_DEVICE_INDEX", 0)
    release()
    yield
    release()


def test_resolve_cpu_when_no_backend_available(monkeypatch):
    _set_backends(monkeypatch, cuda=_none, mps=_none, openvino=_none)
    assert resolve("auto", "auto").kind == "cpu" and resolve("gpu", "auto").kind == "cpu"


def test_auto_picks_first_available_in_priority_order(monkeypatch):
    _set_backends(monkeypatch, cuda=lambda: "RTX", mps=_none, openvino=lambda: "Arc")
    assert resolve("auto", "auto").backend == "cuda"
    assert available_backends() == {"cuda": "RTX", "openvino": "Arc"}


def test_explicit_backend_not_available_falls_to_cpu(monkeypatch):
    _set_backends(monkeypatch, cuda=_none, mps=_none, openvino=lambda: "Arc")
    assert resolve("gpu", "cuda").kind == "cpu"
    assert resolve("gpu", "openvino").backend == "openvino"


def test_unknown_backend_name_falls_to_cpu():
    assert resolve("gpu", "tensorrt").kind == "cpu"


def test_probe_exception_is_treated_as_unavailable(monkeypatch):
    def boom():
        raise RuntimeError("driver")
    _set_backends(monkeypatch, cuda=boom, mps=_none, openvino=lambda: "Arc")
    assert resolve("auto", "auto").backend == "openvino"
    assert resolve("gpu", "cuda").kind == "cpu"
    assert "cuda" not in available_backends()


def test_mode_cpu_ignores_available_gpu(monkeypatch):
    _set_backends(monkeypatch, cuda=lambda: "RTX")
    assert resolve("cpu").kind == "cpu" and resolve("gpu", "cpu").kind == "cpu"


def test_defaults_come_from_config(monkeypatch):
    _set_backends(monkeypatch, cuda=lambda: "RTX")
    monkeypatch.setattr(C, "ACCEL_DEVICE", "auto")
    monkeypatch.setattr(C, "ACCEL_BACKEND", "auto")
    monkeypatch.setattr(C, "ACCEL_DEVICE_INDEX", 2)
    d = resolve()
    assert d.backend == "cuda" and d.index == 2 and d.torch_device == "cuda:2"


def test_dtype_defaults_per_backend_and_override(monkeypatch):
    _set_backends(monkeypatch, cuda=lambda: "RTX", mps=lambda: "M1", openvino=lambda: "Arc")
    assert resolve("gpu", "cuda").dtype == "fp16" and resolve("gpu", "mps").dtype == "fp32"
    assert resolve("gpu", "openvino").dtype == "fp16" and resolve("cpu").dtype == "fp32"
    monkeypatch.setattr(C, "ACCEL_DTYPE", "fp32")
    assert resolve("gpu", "cuda").dtype == "fp32"


CPU = Device("cpu", "cpu", "CPU", "torch", "fp32")
CUDA0 = Device("gpu", "cuda", "RTX", "torch", "fp16", 0)
MPS = Device("gpu", "mps", "M1", "torch", "fp32")
OV = Device("gpu", "openvino", "Arc", "openvino", "fp16")


def test_torch_device_strings():
    assert [d.torch_device for d in (CPU, CUDA0, MPS, OV)] == ["cpu", "cuda:0", "mps", "intel:gpu"]


def test_ort_providers_per_backend():
    assert CUDA0.ort_providers == [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    assert MPS.ort_providers == ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    assert CPU.ort_providers == ["CPUExecutionProvider"]


def test_device_to_dict():
    assert CUDA0.to_dict()["backend"] == "cuda" and CUDA0.to_dict()["index"] == 0


class _Stage:
    note = ""


def test_with_fallback_returns_cpu_result_on_gpu_error():
    h = _Stage()
    out, dev = with_fallback(CUDA0, lambda: 1 / 0, lambda: "cpu-ok", stage=h)
    assert out == "cpu-ok" and dev.kind == "cpu" and "CPU" in h.note


def test_with_fallback_memory_error_and_gpu_success():
    def oom():
        raise MemoryError("vram")
    out, dev = with_fallback(CUDA0, oom, lambda: "cpu")
    assert out == "cpu" and dev.kind == "cpu"
    out, dev = with_fallback(CUDA0, lambda: "gpu", lambda: "cpu")
    assert out == "gpu" and dev is CUDA0
    out, dev = with_fallback(CPU, lambda: "gpu", lambda: "cpu")
    assert out == "cpu" and dev.kind == "cpu"


def test_compiled_builds_once_and_counts_bytes():
    builds = []

    def build():
        builds.append(1)
        return object()
    a = compiled("m", build, 1024)
    b = compiled("m", build, 1024)
    assert a is b and len(builds) == 1 and loaded_bytes() == 1024
    release("m")
    assert loaded_bytes() == 0


def test_compiled_evicts_lru_over_budget(monkeypatch):
    monkeypatch.setattr(C, "ACCEL_MEM_BUDGET_MB", 1)
    kb = 512 * 1024
    compiled("a", lambda: "A", kb)
    compiled("b", lambda: "B", kb)
    compiled("a", lambda: "A2", kb)          # chạm a -> b là LRU
    compiled("c", lambda: "C", kb)
    assert loaded_bytes() <= 1024 * 1024
    assert compiled("a", lambda: "A3", kb) == "A"   # a còn
    assert compiled("b", lambda: "B2", kb) == "B2"  # b đã bị giải phóng


def test_compiled_raises_memory_error_when_single_model_too_big(monkeypatch):
    monkeypatch.setattr(C, "ACCEL_MEM_BUDGET_MB", 1)
    compiled("a", lambda: "A", 1024)
    with pytest.raises(MemoryError):
        compiled("big", lambda: "X", 2 * 1024 * 1024)
    assert loaded_bytes() <= 1024 * 1024


def test_unload_idle_uses_monotonic_clock(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(accel.time, "monotonic", lambda: now[0])
    compiled("a", lambda: "A", 10)
    now[0] = 105.0
    compiled("b", lambda: "B", 10)
    now[0] = 110.0
    assert unload_idle(7) == 1           # a idle 10s, b idle 5s
    assert loaded_bytes() == 10
    assert unload_idle(0) == 0           # 0 = không giải phóng


def test_gpu_work_is_serialized():
    state = {"cur": 0, "max": 0}
    guard = threading.Lock()
    first_in = threading.Event()
    second_started = threading.Event()

    def gpu_fn_factory(is_first):
        def gpu_fn():
            with guard:
                state["cur"] += 1
                state["max"] = max(state["max"], state["cur"])
            if is_first:
                first_in.set()
                # giữ khoá cho tới khi thread 2 đã sắp gọi with_fallback (tối đa 2s)
                second_started.wait(2.0)
                time.sleep(0.05)
            with guard:
                state["cur"] -= 1
            return "ok"
        return gpu_fn

    def run1():
        with_fallback(CUDA0, gpu_fn_factory(True), lambda: "cpu")

    def run2():
        first_in.wait(2.0)
        second_started.set()
        with_fallback(CUDA0, gpu_fn_factory(False), lambda: "cpu")

    ts = [threading.Thread(target=run1), threading.Thread(target=run2)]
    [t.start() for t in ts]
    [t.join(5) for t in ts]
    assert state["max"] == 1 and state["cur"] == 0


# --- vòng sửa 1 ---------------------------------------------------------------
def test_lock_order_gpu_lock_before_cache_lock_no_deadlock():
    """A giữ GPU_LOCK rồi gọi compiled(); B gọi compiled() với build cần GPU_LOCK.
    Thứ tự cũ (khoá cache -> GPU_LOCK) làm hai thread treo nhau."""
    b_in_build = threading.Event()
    done = []

    def build_b():
        b_in_build.set()
        with accel.GPU_LOCK:
            return "B"

    def run_a():
        def gpu_fn():
            b_in_build.wait(0.5)             # đủ để B (thứ tự cũ) giữ khoá cache
            return compiled("lock-a", lambda: "A", 10)
        with_fallback(CUDA0, gpu_fn, lambda: "cpu")
        done.append("a")

    def run_b():
        compiled("lock-b", build_b, 10)
        with_fallback(CUDA0, lambda: "g", lambda: "cpu")
        done.append("b")

    ts = [threading.Thread(target=run_a, daemon=True), threading.Thread(target=run_b, daemon=True)]
    ts[0].start()
    time.sleep(0)                             # nhường A khởi động trước
    ts[1].start()
    [t.join(3) for t in ts]
    assert not any(t.is_alive() for t in ts), "deadlock giữa GPU_LOCK và khoá cache"
    assert sorted(done) == ["a", "b"]


def test_compiled_rechecks_key_after_taking_locks():
    builds = []
    barrier = threading.Barrier(2)

    def build():
        builds.append(1)
        return object()

    res = []

    def run():
        barrier.wait(2)
        res.append(compiled("same", build, 10))

    ts = [threading.Thread(target=run) for _ in range(2)]
    [t.start() for t in ts]
    [t.join(3) for t in ts]
    assert len(builds) == 1 and res[0] is res[1]


class _FakeCore:
    names = ["CPU", "GPU"]

    @property
    def available_devices(self):
        return list(self.names)

    def get_property(self, dev, prop):
        assert prop == "FULL_DEVICE_NAME"
        return f"name-of-{dev}"


def _fake_ov(monkeypatch, names):
    import sys
    import types
    core = type("Core", (_FakeCore,), {"names": names})
    monkeypatch.setitem(sys.modules, "openvino", types.SimpleNamespace(Core=core))


def test_openvino_probe_single_gpu(monkeypatch):
    _fake_ov(monkeypatch, ["CPU", "GPU"])
    assert accel._probe_openvino() == "name-of-GPU"
    monkeypatch.setattr(C, "ACCEL_DEVICE_INDEX", 1)
    assert accel._probe_openvino() is None


def test_openvino_probe_numbered_gpus_uses_index(monkeypatch):
    _fake_ov(monkeypatch, ["CPU", "GPU.0", "GPU.1"])
    assert accel._probe_openvino() == "name-of-GPU.0" or accel._probe_openvino() == "name-of-GPU"
    monkeypatch.setattr(C, "ACCEL_DEVICE_INDEX", 1)
    assert accel._probe_openvino() == "name-of-GPU.1"
    monkeypatch.setattr(C, "ACCEL_DEVICE_INDEX", 2)
    assert accel._probe_openvino() is None


def test_openvino_device_names():
    assert OV.ov_device_name == "GPU" and OV.torch_device == "intel:gpu"
    ov1 = Device("gpu", "openvino", "Arc", "openvino", "fp16", 1)
    assert ov1.ov_device_name == "GPU.1" and ov1.torch_device == "intel:gpu.1"


def test_cpu_device_is_always_fp32(monkeypatch):
    monkeypatch.setattr(C, "ACCEL_DTYPE", "fp16")
    assert resolve("cpu").dtype == "fp32"
    assert resolve("gpu", "tensorrt").dtype == "fp32"
    _, dev = with_fallback(CUDA0, lambda: 1 / 0, lambda: "x")
    assert dev.kind == "cpu" and dev.dtype == "fp32"
