"""Test YOLO-seg theo backend: chỉ chạy CPU, model/YOLO/export đều giả lập."""
import numpy as np
import pytest

from backend.pipeline import accel, config as C, detect
from backend.pipeline.accel import Device
from backend.pipeline.detect import predict_kwargs
from backend.pipeline.trace import Trace


def device_for(backend):
    if backend == "cpu":
        return Device("cpu", "cpu", "CPU", "torch", "fp32")
    rt = "openvino" if backend == "openvino" else "torch"
    dt = "fp32" if backend == "mps" else "fp16"
    return Device("gpu", backend, f"fake-{backend}", rt, dt, 0)


class _T:
    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _Res:
    def __init__(self, n=1, h=40, w=60):
        if n:
            self.boxes = type("B", (), {"xyxy": _T([[5, 5, 30, 35]] * n),
                                        "conf": _T([0.9] * n),
                                        "__len__": lambda self: n})()
            m = np.zeros((n, h, w), np.float32)
            m[:, 5:35, 5:30] = 1
            self.masks = type("M", (), {"data": _T(m)})()
        else:
            self.boxes = None
            self.masks = None


class FakeModel:
    def __init__(self, tag, fail=False, calls=None):
        self.tag, self.fail, self.calls = tag, fail, calls if calls is not None else []

    def predict(self, img, **kw):
        self.calls.append((self.tag, kw))
        if self.fail:
            raise RuntimeError("gpu boom")
        return [_Res()]


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "ACCEL_CACHE_DIR", str(tmp_path / "cache"))
    detect._MODELS.clear(); detect._IR.clear(); detect._EXPORT_FAILED.clear()
    accel.release()
    yield
    detect._MODELS.clear(); detect._IR.clear(); detect._EXPORT_FAILED.clear()
    accel.release()


IMG = np.zeros((40, 60, 3), np.uint8)


@pytest.mark.parametrize("backend,expected", [
    ("cpu", {"device": "cpu", "half": False}),
    ("cuda", {"device": "cuda:0", "half": True}),
    ("mps", {"device": "mps", "half": False}),
    ("openvino", {"device": "intel:gpu", "half": False}),
])
def test_predict_kwargs_per_backend(backend, expected):
    assert predict_kwargs(device_for(backend)) == expected


def test_detect_persons_passes_device_kwargs_to_model(monkeypatch):
    calls = []
    monkeypatch.setattr(detect, "get_yolo", lambda w=None, device=None: FakeModel("g", calls=calls))
    persons = detect.detect_persons(IMG, "x.pt", device=device_for("cuda"))
    assert len(persons) == 1
    kw = calls[0][1]
    assert kw["device"] == "cuda:0" and kw["half"] is True
    assert kw["verbose"] is False and kw["retina_masks"] is True
    assert kw["classes"] == [C.PERSON_CLASS]


def test_detect_falls_back_to_cpu_when_gpu_predict_raises(monkeypatch):
    calls = []

    def fake_get(w=None, device=None):
        return FakeModel("cpu" if device.kind == "cpu" else "gpu",
                         fail=device.kind != "cpu", calls=calls)
    monkeypatch.setattr(detect, "get_yolo", fake_get)
    persons = detect.detect_persons(IMG, "x.pt", device=device_for("cuda"))
    assert len(persons) == 1
    used_device = calls[-1][1]["device"]
    assert used_device == "cpu" and "half" not in calls[-1][1]


def test_cpu_predict_has_no_half_key(monkeypatch):
    calls = []
    monkeypatch.setattr(detect, "get_yolo", lambda w=None, device=None: FakeModel("c", calls=calls))
    detect.detect_persons(IMG, "x.pt", device=device_for("cpu"))
    assert "half" not in calls[0][1] and calls[0][1]["device"] == "cpu"


def test_openvino_falls_back_to_pt_when_export_fails(monkeypatch):
    calls, loaded = [], []
    monkeypatch.setattr(detect, "_load_yolo",
                        lambda path, **kw: (loaded.append(str(path)), FakeModel("m", calls=calls))[1])
    monkeypatch.setattr(detect, "export_openvino", lambda weights=C.YOLO_WEIGHTS: None)
    t = Trace()
    with t.stage("yolo", "y") as s:
        persons, dev = detect._detect_with_device(IMG, "yolo_test.pt", 0.25, 640,
                                                  device_for("openvino"), stage=s)
    assert len(persons) == 1 and loaded == ["yolo_test.pt"]
    assert len(calls) == 1 and calls[0][1]["device"] == "cpu" and "half" not in calls[0][1]
    assert dev.kind == "cpu" and "OpenVINO IR không dùng được" in s.note and "GPU lỗi" not in s.note


def test_export_failure_not_retried_within_window_then_retried(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(detect, "_clock", lambda: now[0])
    monkeypatch.setattr(C, "ACCEL_EXPORT_RETRY_S", 300.0)
    n = []
    monkeypatch.setattr(detect, "export_openvino", lambda weights=None: (n.append(1), None)[1])
    assert detect._ir_for("w.pt") is None and detect._ir_for("w.pt") is None
    assert len(n) == 1
    now[0] += 301
    assert detect._ir_for("w.pt") is None
    assert len(n) == 2


def test_export_success_after_retry_is_cached(monkeypatch, tmp_path):
    now = [0.0]
    monkeypatch.setattr(detect, "_clock", lambda: now[0])
    res = [None, tmp_path]
    n = []
    monkeypatch.setattr(detect, "export_openvino", lambda weights=None: (n.append(1), res[len(n) - 1])[1])
    assert detect._ir_for("w.pt") is None
    now[0] += 1000
    assert detect._ir_for("w.pt") == tmp_path
    assert detect._ir_for("w.pt") == tmp_path and len(n) == 2


def test_openvino_uses_ir_dir_when_present(monkeypatch, tmp_path):
    ir = accel.openvino_ir_dir("some/dir/yolo_test.pt")
    assert ir.name == "yolo_test_openvino_model" and ir.parent == tmp_path / "cache"
    ir.mkdir(parents=True)
    loaded = []
    monkeypatch.setattr(detect, "_load_yolo", lambda path, **kw: (loaded.append(str(path)), FakeModel("m"))[1])
    detect.get_yolo("some/dir/yolo_test.pt", device_for("openvino"))
    assert loaded == [str(ir)]


def test_export_openvino_reuses_existing_dir(monkeypatch):
    ir = accel.openvino_ir_dir("yolo_test.pt")
    ir.mkdir(parents=True)
    monkeypatch.setattr(detect, "_load_yolo", lambda *a, **k: pytest.fail("không được xuất lại"))
    assert detect.export_openvino("yolo_test.pt") == ir


def test_export_openvino_failure_returns_none(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no openvino")
    monkeypatch.setattr(detect, "_load_yolo", boom)
    assert detect.export_openvino("yolo_test.pt") is None


def test_export_openvino_moves_into_cache(monkeypatch, tmp_path):
    src = tmp_path / "w" / "yolo_test_openvino_model"
    src.mkdir(parents=True)
    (src / "m.xml").write_text("x")

    class M:
        def export(self, **kw):
            assert kw == {"format": "openvino", "half": True, "dynamic": False}
            return str(src)
    monkeypatch.setattr(detect, "_load_yolo", lambda *a, **k: M())
    out = detect.export_openvino("yolo_test.pt")
    assert out == accel.openvino_ir_dir("yolo_test.pt") and (out / "m.xml").exists()


def test_get_yolo_caches_per_backend(monkeypatch):
    n = []
    monkeypatch.setattr(detect, "_load_yolo", lambda path, **kw: (n.append(1), FakeModel("m"))[1])
    a = detect.get_yolo("w.pt", device_for("cuda"))
    b = detect.get_yolo("w.pt", device_for("cuda"))
    c = detect.get_yolo("w.pt", device_for("cpu"))
    assert a is b and a is not c and len(n) == 2


def test_yolo_stage_note_contains_backend_and_device(monkeypatch):
    monkeypatch.setattr(detect, "get_yolo", lambda w=None, device=None: FakeModel("m"))
    monkeypatch.setattr(accel, "resolve", lambda mode=None, backend=None: device_for("cuda"))
    scene = detect.detect_scene(np.zeros((40, 60, 3), np.uint8), preprocess="none", trace=Trace())
    note = next(s for s in scene.trace.stages if s.key == "yolo").note
    assert "cuda" in note
    assert scene.device["backend"] == "cuda"


def test_yolo_stage_note_cpu_and_fallback(monkeypatch):
    def fake_get(w=None, device=None):
        return FakeModel("m", fail=device.kind != "cpu")
    monkeypatch.setattr(detect, "get_yolo", fake_get)
    monkeypatch.setattr(accel, "resolve", lambda mode=None, backend=None: device_for("cpu" if mode == "cpu" else "cuda"))
    scene = detect.detect_scene(np.zeros((40, 60, 3), np.uint8), preprocess="none", trace=Trace())
    note = next(s for s in scene.trace.stages if s.key == "yolo").note
    assert "CPU" in note and "GPU lỗi" in note
    assert scene.device["kind"] == "cpu"
