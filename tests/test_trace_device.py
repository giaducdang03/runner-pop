"""Test trace/PDF ghi thiết bị cho từng bước nặng (FR-107)."""
from __future__ import annotations

import numpy as np
import pytest

from backend import report
from backend.pipeline import accel, config as C, detect
from backend.pipeline.accel import Device, device_label
from backend.pipeline.schema import SegParams
from backend.pipeline.segment import build_cutout
from backend.pipeline.segment import birefnet as bf
from backend.pipeline.trace import Stage, Trace

import test_detect_device as td
from test_birefnet_device import called  # noqa: F401  (fixture rembg/runner giả)
from backend.pipeline.cache import StageCache

CPU = Device("cpu", "cpu", "CPU", "torch", "fp32")
CUDA = Device("gpu", "cuda", "NVIDIA GeForce RTX", "torch", "fp16", 0)


def test_device_label_cpu_and_gpu():
    assert device_label(CPU) == "CPU"
    assert device_label(CUDA) == "cuda: NVIDIA GeForce RTX (fp16)"


def test_stage_records_device_via_handle():
    t = Trace()
    with t.stage("a", "A") as s:
        s.set_device(CUDA)
    with t.stage("b", "B"):
        pass
    assert t.stages[0].device == "cuda: NVIDIA GeForce RTX (fp16)"
    assert t.stages[1].device == ""


def test_to_list_has_device_key_for_every_stage():
    t = Trace()
    with t.stage("a", "A") as s:
        s.set_device(CPU)
    with t.stage("b", "B"):
        pass
    lst = t.to_list()
    assert all("device" in d for d in lst)
    assert [d["device"] for d in lst] == ["CPU", ""]


@pytest.fixture
def yolo_env(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "ACCEL_CACHE_DIR", str(tmp_path / "cache"))
    detect._MODELS.clear(); detect._IR.clear(); detect._EXPORT_FAILED.clear()
    yield
    detect._MODELS.clear(); detect._IR.clear(); detect._EXPORT_FAILED.clear()


def _yolo_stage(scene):
    return next(s for s in scene.trace.stages if s.key == "yolo")


def test_yolo_stage_records_device(monkeypatch, yolo_env):
    monkeypatch.setattr(detect, "get_yolo", lambda w=None, device=None: td.FakeModel("m"))
    monkeypatch.setattr(accel, "resolve", lambda mode=None, backend=None: td.device_for("cuda"))
    scene = detect.detect_scene(np.zeros((40, 60, 3), np.uint8), preprocess="none", trace=Trace())
    assert _yolo_stage(scene).device == "cuda: fake-cuda (fp16)"


def test_yolo_stage_records_cpu_after_gpu_failure(monkeypatch, yolo_env):
    def fake_get(w=None, device=None):
        return td.FakeModel("m", fail=device.kind != "cpu")
    monkeypatch.setattr(detect, "get_yolo", fake_get)
    real = accel.resolve
    monkeypatch.setattr(accel, "resolve",
                        lambda mode=None, backend=None: real("cpu") if mode == "cpu" else td.device_for("cuda"))
    scene = detect.detect_scene(np.zeros((40, 60, 3), np.uint8), preprocess="none", trace=Trace())
    assert _yolo_stage(scene).device == "CPU"


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


def test_cf_matting_and_decontaminate_record_cpu(make_scene):
    scene = _scene(make_scene)
    c = build_cutout(scene, scene.persons[0], "cf")
    by_key = {s.key: s for s in c.trace.stages}
    assert by_key["matting"].device == "CPU"
    assert by_key["decontaminate"].device == "CPU"


def test_grabcut_stage_records_cpu(make_scene):
    scene = _scene(make_scene)
    c = build_cutout(scene, scene.persons[0], "cf")
    assert next(s for s in c.trace.stages if s.key == "grabcut").device == "CPU"


def test_birefnet_stage_records_requested_device(monkeypatch, make_scene):
    scene = _scene(make_scene)
    monkeypatch.setattr(bf, "birefnet_alpha",
                        lambda rgb, device=None, stage=None: np.ones(rgb.shape[:2], np.float32))
    monkeypatch.setattr(accel, "resolve", lambda *a, **k: CUDA)
    c = build_cutout(scene, scene.persons[0], "birefnet", SegParams())
    assert next(s for s in c.trace.stages if s.key == "birefnet").device.startswith("cuda:")


def test_birefnet_alpha_sets_device_on_real_paths(monkeypatch):
    monkeypatch.setattr(bf, "_cpu_alpha", lambda rgb: np.ones(rgb.shape[:2], np.float32))
    t = Trace()
    with t.stage("birefnet", "b") as s:
        bf.birefnet_alpha(np.zeros((8, 8, 3), np.uint8), CPU, stage=s)
    assert t.stages[0].device == "CPU"


def test_device_summary():
    st = [Stage("yolo", "y", 0.1, device="CPU"),
          Stage("birefnet", "b", 0.1, device="cuda: X (fp16)"),
          Stage("matting", "m", 0.1, device="CPU"),
          Stage("roi", "r", 0.1)]
    assert report.device_summary(st) == "YOLO: CPU · BiRefNet: cuda: X (fp16) · Matting: CPU"
    assert report.device_summary([Stage("roi", "r", 0.1)]) == "-"
    assert report.device_summary(None) == "-"
    assert report.device_summary([Stage("yolo", "y", 0.1, device="CPU")],
                                 [Stage("matting", "m", 0.1, device="CPU")]) == "YOLO: CPU · Matting: CPU"


def test_stage_records_device_even_if_body_raises():
    t = Trace()
    with pytest.raises(RuntimeError):
        with t.stage("a", "A") as s:
            s.set_device(CUDA)
            raise RuntimeError("boom")
    assert t.stages[0].device == "cuda: NVIDIA GeForce RTX (fp16)"


def _bf_stage(c):
    return next(s for s in c.trace.stages if s.key == "birefnet")


def _boom(*a, **k):
    raise RuntimeError("gpu boom")


def _resolve(mode=None, backend=None):
    return CPU if mode == "cpu" else CUDA


def test_birefnet_real_fallback_records_cpu_then_cache_hit_keeps_cpu(called, monkeypatch, make_scene):
    scene = _scene(make_scene)
    person = scene.persons[0]
    monkeypatch.setattr(accel, "resolve", _resolve)
    monkeypatch.setattr(accel, "onnx_runner", _boom)
    cache = StageCache()
    c1 = build_cutout(scene, person, "birefnet", SegParams(cache=cache))
    assert _bf_stage(c1).device == "CPU" and called["rembg"] == 1
    c2 = build_cutout(scene, person, "birefnet", SegParams(cache=cache, band_frac=0.05))
    assert called["rembg"] == 1                      # cache hit: không chạy lại
    assert _bf_stage(c2).device == "CPU"             # thiết bị đã tính ra alpha, không phải thiết bị được yêu cầu


def test_birefnet_gpu_success_then_cache_hit_records_gpu(called, monkeypatch, make_scene):
    scene = _scene(make_scene)
    person = scene.persons[0]
    monkeypatch.setattr(accel, "resolve", _resolve)
    cache = StageCache()
    c1 = build_cutout(scene, person, "birefnet", SegParams(cache=cache))
    n = called["runner"]
    c2 = build_cutout(scene, person, "birefnet", SegParams(cache=cache, band_frac=0.05))
    assert called["runner"] == n and called["rembg"] == 0
    assert _bf_stage(c1).device == _bf_stage(c2).device == "cuda: NVIDIA GeForce RTX (fp16)"


def test_birefnet_remembered_failure_records_cpu(called, monkeypatch, make_scene):
    scene = _scene(make_scene)
    monkeypatch.setattr(accel, "resolve", _resolve)
    monkeypatch.setattr(accel, "onnx_runner_failed", lambda key, device: True)
    c = build_cutout(scene, scene.persons[0], "birefnet", SegParams())
    assert called["runner"] == 0 and _bf_stage(c).device == "CPU"


def test_fit_text_short_unchanged_long_truncated_empty():
    measure = lambda t: float(len(t))
    assert report.fit_text("abc", 10, measure) == "abc"
    out = report.fit_text("x" * 50, 10, measure)
    assert out.endswith("…") and measure(out) <= 10 and len(out) > 1
    assert report.fit_text("", 10, measure) == ""
