"""GPU chết giữa chừng: lần gọi thứ nhất chạy GPU bình thường, lần thứ hai ném lỗi.
Pipeline vẫn phải trả kết quả đầy đủ, rơi về CPU, có ghi chú trong trace, không raise.
Toàn bộ giả lập (không GPU thật, không tải trọng số)."""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from backend.pipeline import accel, config as C, detect
from backend.pipeline.accel import Device
from backend.pipeline.cache import StageCache
from backend.pipeline.schema import Person, Scene, SegParams
from backend.pipeline.segment import build_cutout
from backend.pipeline.segment import birefnet as bf
from backend.pipeline.trace import Trace

CUDA = Device("gpu", "cuda", "fake-cuda", "onnxruntime", "fp32", 0)
S = C.BIREFNET_SIZE


@pytest.fixture
def dying_gpu(monkeypatch):
    """rembg giả (CPU) + onnx_runner giả: gọi lần 1 ổn, lần 2 ném 'device lost'."""
    from PIL import Image

    state = {"gpu_calls": 0, "cpu_calls": 0}
    fake_rembg = types.ModuleType("rembg")
    fake_rembg.new_session = lambda name, providers=None: "sess"

    def remove(img, session=None, only_mask=False):
        state["cpu_calls"] += 1
        return Image.new("L", img.size, 255)
    fake_rembg.remove = remove
    monkeypatch.setitem(sys.modules, "rembg", fake_rembg)
    monkeypatch.setattr(bf, "_session", None)
    monkeypatch.setattr(bf, "onnx_path", lambda: "fake/birefnet.onnx")

    def fake_runner(key, path, device, static_input=None):
        def run(feeds):
            state["gpu_calls"] += 1
            if state["gpu_calls"] >= 2:
                raise RuntimeError("device lost")
            rng = np.random.default_rng(1)
            return [(rng.standard_normal((1, 1, S, S)) * 4).astype(np.float32)]
        return run
    monkeypatch.setattr(accel, "onnx_runner", fake_runner)
    monkeypatch.setattr(accel, "onnx_runner_failed", lambda key, device: False)
    monkeypatch.setattr(accel, "resolve", lambda *a, **k: CUDA)
    return state


def _two_person_scene(make_scene):
    h, w = 240, 320
    boxes = [(40, 60, 110, 200), (180, 60, 250, 200)]
    people = []
    rgb = np.full((h, w, 3), 100, np.uint8)
    for b in boxes:
        m = np.zeros((h, w), np.uint8)
        m[b[1]:b[3], b[0]:b[2]] = 255
        rgb[b[1]:b[3], b[0]:b[2]] = (220, 30, 30)
        people.append((b, m))
    scene = make_scene((h, w), people)
    scene.rgb = rgb
    return scene


def test_pipeline_survives_gpu_dying_midway(dying_gpu, make_scene):
    scene = _two_person_scene(make_scene)
    cache = StageCache()
    c1 = build_cutout(scene, scene.persons[0], "birefnet", SegParams(cache=cache))
    assert dying_gpu["gpu_calls"] == 1 and dying_gpu["cpu_calls"] == 0
    assert any(s.key == "birefnet" and s.device.startswith("cuda") for s in c1.trace.stages)

    # người khác -> khoá cache BiRefNet khác -> bắt buộc gọi GPU lần hai (đang chết)
    c2 = build_cutout(scene, scene.persons[1], "birefnet", SegParams(cache=cache))
    assert dying_gpu["gpu_calls"] == 2 and dying_gpu["cpu_calls"] == 1
    assert c2.alpha.shape == scene.rgb.shape[:2]
    assert c2.foreground.shape == scene.rgb.shape and c2.background.shape == scene.rgb.shape
    stage = next(s for s in c2.trace.stages if s.key == "birefnet")
    assert "CPU" in stage.note and "GPU lỗi" in stage.note and "device lost" in stage.note
    assert stage.device == "CPU"
    assert any("CPU" in s.note for s in c2.trace.stages)


def test_detect_scene_survives_gpu_dying_midway(monkeypatch):
    calls = {"gpu": 0, "cpu": 0}

    class _T:
        def __init__(self, a):
            self.a = np.asarray(a)

        def cpu(self):
            return self

        def numpy(self):
            return self.a

    class _Res:
        def __init__(self, h=40, w=60):
            self.boxes = type("B", (), {"xyxy": _T([[5, 5, 30, 35]]), "conf": _T([0.9]),
                                        "__len__": lambda self: 1})()
            m = np.zeros((1, h, w), np.float32)
            m[:, 5:35, 5:30] = 1
            self.masks = type("M", (), {"data": _T(m)})()

    class Model:
        def __init__(self, gpu):
            self.gpu = gpu

        def predict(self, img, **kw):
            if self.gpu:
                calls["gpu"] += 1
                if calls["gpu"] >= 2:
                    raise RuntimeError("device lost")
            else:
                calls["cpu"] += 1
            return [_Res()]

    monkeypatch.setattr(detect, "get_yolo",
                        lambda w=None, device=None: Model(device.kind != "cpu"))
    monkeypatch.setattr(accel, "resolve",
                        lambda mode=None, backend=None: accel._cpu_device() if mode == "cpu" else CUDA)
    img = np.zeros((40, 60, 3), np.uint8)
    s1 = detect.detect_scene(img, preprocess="none", trace=Trace())
    assert calls == {"gpu": 1, "cpu": 0} and s1.device["backend"] == "cuda"

    s2 = detect.detect_scene(img, preprocess="none", trace=Trace())     # GPU chết ở lần hai
    assert calls == {"gpu": 2, "cpu": 1}
    assert len(s2.persons) == 1
    note = next(s for s in s2.trace.stages if s.key == "yolo").note
    assert "CPU" in note and "GPU lỗi" in note
    assert s2.device["kind"] == "cpu"
