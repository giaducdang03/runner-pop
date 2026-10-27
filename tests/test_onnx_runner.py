"""Test accel.onnx_runner: chỉ chạy CPU, onnxruntime/openvino đều là module giả."""
import sys
import types

import numpy as np
import pytest

from backend.pipeline import accel, config as C
from backend.pipeline.accel import Device, onnx_runner

CPU = Device("cpu", "cpu", "CPU", "torch", "fp32")


def gpu(backend, index=0, dtype=None):
    rt = "openvino" if backend == "openvino" else "onnxruntime"
    dt = dtype or ("fp16" if backend == "openvino" else "fp32")
    return Device("gpu", backend, f"fake-{backend}", rt, dt, index)


class FakeORT(types.ModuleType):
    """onnxruntime giả: ghi lại providers yêu cầu, session báo EP đang chạy."""

    def __init__(self, available, active_first=None):
        super().__init__("onnxruntime")
        self.available = list(available)
        self.active_first = active_first        # None = EP đầu tiên được yêu cầu
        self.created = []                        # providers của từng session đã tạo
        ort = self

        class InferenceSession:
            def __init__(self, path, providers=None, **kw):
                ort.created.append(list(providers))
                names = [p[0] if isinstance(p, tuple) else p for p in providers]
                if ort.active_first is not None:
                    names = [ort.active_first] + [n for n in names if n != ort.active_first]
                self._active = names

            def get_providers(self):
                return list(self._active)

            def get_inputs(self):
                return [types.SimpleNamespace(name="input")]

            def run(self, outs, feeds):
                return [feeds["input"] * 2, feeds["input"] + 1]

        self.InferenceSession = InferenceSession

    def get_available_providers(self):
        return list(self.available)


class FakeOV(types.ModuleType):
    """openvino giả: ghi lại đối số convert_model / compile_model."""

    def __init__(self):
        super().__init__("openvino")
        self.convert_calls = []
        self.compile_calls = []
        ov = self

        class _Compiled:
            outputs = ["out0", "out1"]

            def __call__(self, feeds):
                x = feeds["input"]
                return {"out0": x * 2, "out1": x + 1}

        class Core:
            def compile_model(self, model, device_name, config=None):
                ov.compile_calls.append((model, device_name, dict(config or {})))
                return _Compiled()

        self.Core = Core

    def convert_model(self, path, input=None):
        self.convert_calls.append((path, input))
        return f"model:{path}"


@pytest.fixture
def onnx_file(tmp_path):
    p = tmp_path / "m.onnx"
    p.write_bytes(b"x" * 128)
    return str(p)


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "ACCEL_CACHE_DIR", str(tmp_path / "cache"))
    accel.release()
    accel.forget_failures()
    yield
    accel.release()
    accel.forget_failures()


def _install(monkeypatch, name, mod):
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


def test_cuda_runner_requests_cuda_provider_first(monkeypatch, onnx_file):
    ort = _install(monkeypatch, "onnxruntime",
                   FakeORT(["CUDAExecutionProvider", "CPUExecutionProvider"]))
    run = onnx_runner("k", onnx_file, gpu("cuda"))
    captured_providers = ort.created[0]
    assert captured_providers[0] == ("CUDAExecutionProvider", {"device_id": 0})
    out = run({"input": np.ones((1, 2), np.float32)})
    assert [o.tolist() for o in out] == [[[2.0, 2.0]], [[2.0, 2.0]]]


def test_mps_runner_requests_coreml_provider(monkeypatch, onnx_file):
    ort = _install(monkeypatch, "onnxruntime",
                   FakeORT(["CoreMLExecutionProvider", "CPUExecutionProvider"]))
    onnx_runner("k", onnx_file, gpu("mps"))
    captured_providers = ort.created[0]
    assert captured_providers[0] == "CoreMLExecutionProvider"


def test_cpu_runner_uses_only_cpu_provider(monkeypatch, onnx_file):
    ort = _install(monkeypatch, "onnxruntime",
                   FakeORT(["CUDAExecutionProvider", "CPUExecutionProvider"]))
    onnx_runner("k", onnx_file, CPU)
    assert ort.created[0] == ["CPUExecutionProvider"]


def test_runner_raises_when_accelerator_ep_missing(monkeypatch, onnx_file):
    ort = _install(monkeypatch, "onnxruntime", FakeORT(["CPUExecutionProvider"]))
    with pytest.raises(RuntimeError):
        onnx_runner("k", onnx_file, gpu("cuda"))
    assert ort.created == []                      # không tạo session nào


def test_runner_raises_when_session_silently_runs_on_cpu(monkeypatch, onnx_file):
    # R19: EP có trong danh sách nhưng session thực tế chạy CPU (onnxruntime thường đè -gpu)
    _install(monkeypatch, "onnxruntime",
             FakeORT(["CUDAExecutionProvider", "CPUExecutionProvider"],
                     active_first="CPUExecutionProvider"))
    with pytest.raises(RuntimeError):
        onnx_runner("k", onnx_file, gpu("cuda"))
    assert accel.loaded_bytes() == 0              # lỗi không được lưu vào cache


def test_openvino_runner_sets_cache_dir_and_f16_hint(monkeypatch, onnx_file):
    ov = _install(monkeypatch, "openvino", FakeOV())
    run = onnx_runner("k", onnx_file, gpu("openvino", dtype="fp16"))
    _, dev_name, props = ov.compile_calls[0]
    assert dev_name == "GPU"
    assert props["CACHE_DIR"] == C.ACCEL_CACHE_DIR and props["INFERENCE_PRECISION_HINT"] == "f16"
    out = run({"input": np.ones((1, 2), np.float32)})
    assert [o.tolist() for o in out] == [[[2.0, 2.0]], [[2.0, 2.0]]]


def test_openvino_runner_fp32_hint_and_device_index(monkeypatch, onnx_file):
    ov = _install(monkeypatch, "openvino", FakeOV())
    onnx_runner("k", onnx_file, gpu("openvino", index=1, dtype="fp32"))
    _, dev_name, props = ov.compile_calls[0]
    assert dev_name == "GPU.1" and props["INFERENCE_PRECISION_HINT"] == "f32"


def test_openvino_runner_converts_with_static_shape(monkeypatch, onnx_file):
    ov = _install(monkeypatch, "openvino", FakeOV())
    onnx_runner("k", onnx_file, gpu("openvino"), static_input={"input": (1, 3, 1024, 1024)})
    path, convert_input = ov.convert_calls[0]
    assert path == onnx_file
    assert convert_input == {"input": (1, 3, 1024, 1024)}


def test_runner_is_cached_per_backend_and_index(monkeypatch, onnx_file):
    ort = _install(monkeypatch, "onnxruntime",
                   FakeORT(["CUDAExecutionProvider", "CPUExecutionProvider"]))
    onnx_runner("k", onnx_file, gpu("cuda", 0))
    onnx_runner("k", onnx_file, gpu("cuda", 0))
    assert len(ort.created) == 1
    onnx_runner("k", onnx_file, gpu("cuda", 1))
    assert len(ort.created) == 2
    assert accel.loaded_bytes() == 2 * 128        # nbytes = kích thước file ONNX


def test_failed_build_is_remembered_then_retried(monkeypatch, onnx_file):
    # R22: build lỗi được nhớ ACCEL_EXPORT_RETRY_S giây, không tạo lại session mỗi request
    now = [1000.0]
    monkeypatch.setattr(accel, "_clock", lambda: now[0])
    monkeypatch.setattr(C, "ACCEL_EXPORT_RETRY_S", 300.0)
    ort = _install(monkeypatch, "onnxruntime",
                   FakeORT(["CUDAExecutionProvider", "CPUExecutionProvider"],
                           active_first="CPUExecutionProvider"))
    dev = gpu("cuda")
    with pytest.raises(RuntimeError):
        onnx_runner("k", onnx_file, dev)
    assert accel.onnx_runner_failed("k", dev)
    with pytest.raises(RuntimeError):
        onnx_runner("k", onnx_file, dev)
    assert len(ort.created) == 1                  # lần 2 không thử tạo session
    now[0] += 301
    assert not accel.onnx_runner_failed("k", dev)
    ort.active_first = None                       # driver đã sửa
    onnx_runner("k", onnx_file, dev)
    assert len(ort.created) == 2 and not accel.onnx_runner_failed("k", dev)


def test_failure_memo_is_per_backend_and_index(monkeypatch, onnx_file):
    _install(monkeypatch, "onnxruntime", FakeORT(["CPUExecutionProvider"]))
    with pytest.raises(RuntimeError):
        onnx_runner("k", onnx_file, gpu("cuda", 0))
    assert accel.onnx_runner_failed("k", gpu("cuda", 0))
    assert not accel.onnx_runner_failed("k", gpu("cuda", 1))
    assert not accel.onnx_runner_failed("k", CPU)
