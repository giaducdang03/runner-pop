"""Test BiRefNet theo thiết bị: CPU qua rembg, GPU qua accel.onnx_runner (đều giả lập,
không tải trọng số, không tạo session thật)."""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from backend.pipeline import accel, config as C
from backend.pipeline.accel import Device
from backend.pipeline.cache import StageCache
from backend.pipeline.schema import SegParams
from backend.pipeline.segment import build_cutout
from backend.pipeline.segment import birefnet as bf

CPU = Device("cpu", "cpu", "CPU", "torch", "fp32")
CUDA = Device("gpu", "cuda", "fake-cuda", "onnxruntime", "fp32", 0)
S = C.BIREFNET_SIZE


def _rgb(h=50, w=70):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _logits(seed=1):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((1, 1, S, S)) * 4).astype(np.float32)


@pytest.fixture
def called(monkeypatch):
    """rembg giả (CPU) + onnx_runner giả (GPU); ghi lại đường nào đã chạy."""
    called = {"rembg": 0, "runner": 0, "feeds": [], "static_input": None}
    from PIL import Image

    fake_rembg = types.ModuleType("rembg")

    def new_session(name, providers=None):
        called["providers"] = providers
        return "sess"

    def remove(img, session=None, only_mask=False):
        called["rembg"] += 1
        return Image.new("L", img.size, 255)

    fake_rembg.new_session, fake_rembg.remove = new_session, remove
    monkeypatch.setitem(sys.modules, "rembg", fake_rembg)
    monkeypatch.setattr(bf, "_session", None)
    monkeypatch.setattr(bf, "onnx_path", lambda: "fake/birefnet.onnx")

    def fake_runner(key, path, device, static_input=None):
        called["runner"] += 1
        called["static_input"] = static_input

        def run(feeds):
            called["feeds"].append({k: v.shape for k, v in feeds.items()})
            return [_logits()]
        return run

    monkeypatch.setattr(accel, "onnx_runner", fake_runner)
    monkeypatch.setattr(accel, "onnx_runner_failed", lambda key, device: False)
    return called


def test_birefnet_alpha_uses_rembg_on_cpu(called):
    a = bf.birefnet_alpha(_rgb(), CPU)
    assert called["rembg"] and not called["runner"]
    assert called["providers"] == ["CPUExecutionProvider"]
    assert a.shape == (50, 70) and a.dtype == np.float32 and a.max() == 1.0


def test_birefnet_alpha_default_device_keeps_old_signature(called, monkeypatch):
    monkeypatch.setattr(accel, "resolve", lambda *a, **k: CPU)
    a = bf.birefnet_alpha(_rgb())
    assert called["rembg"] == 1 and a.shape == (50, 70)


def test_birefnet_alpha_falls_back_to_rembg_when_runner_raises(called, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("CUDAExecutionProvider không có")
    monkeypatch.setattr(accel, "onnx_runner", boom)
    stage = types.SimpleNamespace(note="")
    rgb = _rgb()
    a = bf.birefnet_alpha(rgb, CUDA, stage=stage)
    assert a.shape == rgb.shape[:2] and "CPU" in stage.note
    assert called["rembg"] == 1


def test_birefnet_goes_straight_to_cpu_while_failure_remembered(called, monkeypatch):
    # R22: lỗi build còn được nhớ -> chạy CPU luôn, không thử GPU rồi mới rơi về
    monkeypatch.setattr(accel, "onnx_runner_failed", lambda key, device: True)
    tried = []
    monkeypatch.setattr(accel, "with_fallback", lambda *a, **k: tried.append(1))
    stage = types.SimpleNamespace(note="")
    a = bf.birefnet_alpha(_rgb(), CUDA, stage=stage)
    assert a.shape == (50, 70)
    assert called["rembg"] == 1 and called["runner"] == 0 and not tried
    assert "CPU" in stage.note


def test_birefnet_input_is_fixed_size(called):
    stage = types.SimpleNamespace(note="")
    a = bf.birefnet_alpha(_rgb(37, 91), CUDA, stage=stage)
    (seen_shape,) = called["feeds"][0].values()
    assert seen_shape == (1, 3, C.BIREFNET_SIZE, C.BIREFNET_SIZE)
    assert called["static_input"] == {C.BIREFNET_INPUT_NAME: (1, 3, S, S)}
    assert a.shape == (37, 91) and a.dtype == np.float32
    assert 0.0 <= a.min() and a.max() <= 1.0
    assert not called["rembg"]
    assert "cuda" in stage.note


def test_gpu_path_matches_rembg_pre_and_post_processing(called, monkeypatch):
    """Đối chứng với lớp session BiRefNet của rembg đã cài: cùng tensor vào, cùng alpha ra."""
    monkeypatch.delitem(sys.modules, "rembg")      # dùng rembg thật (chỉ lớp, không session)
    from PIL import Image
    from rembg.sessions.birefnet_general_lite import BiRefNetSessionGeneralLite

    logits = _logits(7)
    seen = {}

    class FakeInner:
        def get_inputs(self):
            return [types.SimpleNamespace(name=C.BIREFNET_INPUT_NAME)]

        def run(self, outs, feeds):
            seen["rembg"] = feeds[C.BIREFNET_INPUT_NAME]
            return [logits]

    sess = object.__new__(BiRefNetSessionGeneralLite)
    sess.inner_session = FakeInner()
    rgb = _rgb(45, 63)
    ref = np.asarray(sess.predict(Image.fromarray(rgb))[0], np.float32) / 255.0

    def runner(key, path, device, static_input=None):
        def run(feeds):
            seen["ours"] = feeds[C.BIREFNET_INPUT_NAME]
            return [logits]
        return run
    monkeypatch.setattr(accel, "onnx_runner", runner)
    ours = bf.birefnet_alpha(rgb, CUDA)
    np.testing.assert_array_equal(seen["ours"], seen["rembg"])
    np.testing.assert_array_equal(ours, ref)


def test_onnx_path_uses_rembg_session_class_without_download(monkeypatch, tmp_path):
    target = tmp_path / f"{C.BIREFNET_MODEL}.onnx"

    class FakeSession:
        @classmethod
        def name(cls, *a, **k):
            return C.BIREFNET_MODEL

        @classmethod
        def download_models(cls, *a, **k):
            return str(target)

    class Other(FakeSession):
        @classmethod
        def name(cls, *a, **k):
            return "u2net"

        @classmethod
        def download_models(cls, *a, **k):
            raise AssertionError("không được tải model khác")

    pkg = types.ModuleType("rembg")
    sessions = types.ModuleType("rembg.sessions")
    sessions.sessions_class = [Other, FakeSession]
    pkg.sessions = sessions
    monkeypatch.setitem(sys.modules, "rembg", pkg)
    monkeypatch.setitem(sys.modules, "rembg.sessions", sessions)
    assert bf.onnx_path() == target


def _scene(make_scene):
    h, w = 240, 320
    box = (120, 60, 190, 200)
    m = np.zeros((h, w), np.uint8)
    m[60:200, 120:190] = 255
    scene = make_scene((h, w), [(box, m)])
    rgb = np.full((h, w, 3), 100, np.uint8)
    rgb[60:200, 120:190] = (220, 30, 30)
    scene.rgb = rgb
    return scene


def test_engine_cache_key_includes_device(monkeypatch, make_scene):
    # R6: khoá cache alpha gồm backend, index, dtype của thiết bị
    scene = _scene(make_scene)
    person = scene.persons[0]
    seen = []

    def fake(rgb, device=None, stage=None):
        seen.append(device)
        return np.ones(rgb.shape[:2], np.float32)

    monkeypatch.setattr(bf, "birefnet_alpha", fake)
    monkeypatch.setattr(accel, "resolve", lambda *a, **k: CUDA)
    cache = StageCache()
    build_cutout(scene, person, "birefnet", SegParams(cache=cache))
    assert seen == [CUDA]
    keys = [k for k in cache._data if k and k[0] == "birefnet"]
    assert keys == [("birefnet", person.idx, person.box, "cuda", 0, "fp32")]
    assert ("roi", person.idx, person.box) in cache._data
    monkeypatch.setattr(accel, "resolve", lambda *a, **k: CPU)
    build_cutout(scene, person, "birefnet", SegParams(cache=cache))
    assert seen == [CUDA, CPU]                     # thiết bị khác -> chạy lại, không dùng nhầm cache
