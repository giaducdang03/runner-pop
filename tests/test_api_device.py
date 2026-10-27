"""API thiết bị / warm-up (B4). Không nạp model thật: mọi thứ nặng được monkeypatch."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import main
from backend import pipeline as P
from backend.pipeline.schema import Person, Scene

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontend"))
import ui_helpers  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_warmup():
    """Trả _WARMUP về trạng thái ban đầu để các test không rò rỉ trạng thái cho nhau."""
    main._set_warmup("idle", "not started")
    yield
    main._set_warmup("idle", "not started")


@pytest.fixture
def client():
    return TestClient(main.app)      # không vào lifespan -> không chạy warm-up


def _jpeg() -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((64, 64, 3), 120, np.uint8))
    return buf.tobytes()


def _session_with_scene() -> str:
    rgb = np.zeros((100, 100, 3), np.uint8)
    scene = Scene(rgb=rgb, persons=[Person(idx=0, box=(10, 10, 80, 90), conf=0.9, mask=None)])
    main._put("sid-dev", main.Session(rgb=rgb, scene=scene))
    return "sid-dev"


def test_health_reports_device_and_warmup(client):
    j = client.get("/health").json()
    assert {"device", "backend", "available_backends", "warmup", "warmup_detail"} <= set(j)
    assert j["ok"] is True and "yolo_available" in j and "weights" in j
    assert j["backend"] == j["device"]["backend"]
    assert j["warmup"] == "idle"          # chưa vào lifespan -> không bao giờ kẹt "running"


def test_options_reports_devices_and_default_engine(client):
    j = client.get("/options").json()
    assert j["devices"] == ["auto", "cpu", "gpu"]
    assert j["default_engine"] in [e["name"] for e in j["engines"]]
    assert "backend" in j["device"] and "preprocess" in j


def test_process_rejects_unknown_device(client):
    sid = _session_with_scene()
    assert client.post(f"/process/{sid}?device=cuda").status_code == 422


def test_detect_rejects_unknown_device(client):
    r = client.post("/detect?device=tpu", files={"file": ("a.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 422


def test_default_engine_priority_and_fallback():
    assert main.default_engine(["cf", "birefnet", "modnet", "rvm"]) == "rvm"
    assert main.default_engine(["cf", "modnet", "focus"]) == "modnet"
    assert main.default_engine(["focus", "cf"]) == "cf"
    assert main.default_engine(["birefnet", "focus"]) == "birefnet"
    assert main.default_engine([]) == ""


def test_detect_forwards_device_mode(client, monkeypatch):
    seen = {}

    def fake_detect(rgb, preprocess, weights, trace=None, device_mode="auto"):
        seen["mode"] = device_mode
        return Scene(rgb=rgb, persons=[])

    monkeypatch.setattr(P, "detect_scene", fake_detect)
    r = client.post("/detect?device=cpu", files={"file": ("a.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 200 and seen["mode"] == "cpu"
    client.post("/detect", files={"file": ("a.jpg", _jpeg(), "image/jpeg")})
    assert seen["mode"] == "auto"


def test_process_forwards_device_mode(client, monkeypatch):
    seen = {}

    def fake_build(scene, person, method, params, trace=None):
        seen["mode"] = params.device_mode
        h, w = scene.rgb.shape[:2]
        z = np.zeros((h, w), np.float32)
        img = np.zeros((h, w, 3), np.float32)
        return P.Cutout(image=img, alpha=z, foreground=img, background=img,
                        trimap=np.zeros((h, w), np.uint8))

    monkeypatch.setattr(P, "build_cutout", fake_build)
    sid = _session_with_scene()
    r = client.post(f"/process/{sid}?device=gpu")
    assert r.status_code == 200 and seen["mode"] == "gpu"


def test_warmup_done_and_error(monkeypatch):
    states = []

    def ok():
        states.append(main._WARMUP["state"])
        return "ok-detail"

    monkeypatch.setattr(main, "_do_warmup", ok)
    main._set_warmup("running", "")
    main.warmup()
    assert states == ["running"] and main._WARMUP["state"] == "done"

    def boom():
        raise RuntimeError("hỏng model")

    monkeypatch.setattr(main, "_do_warmup", boom)
    main._set_warmup("running", "")
    main.warmup()                                   # không được raise
    assert main._WARMUP["state"] == "error" and "hỏng model" in main._WARMUP["detail"]


def test_warmup_disabled(monkeypatch):
    called = []
    monkeypatch.setattr(main, "_do_warmup", lambda: called.append(1))
    monkeypatch.setattr(main.C, "WARMUP_ON_START", False)
    main._set_warmup("running", "")
    main.warmup()
    assert not called and main._WARMUP == {"state": "done", "detail": "disabled"}


def test_lifespan_starts_warmup_thread(monkeypatch):
    import threading
    ev = threading.Event()
    monkeypatch.setattr(main, "warmup", ev.set)
    with TestClient(main.app):
        assert ev.wait(2)


def test_lifespan_goes_running_then_done(monkeypatch):
    import threading
    seen, gate, fin = {}, threading.Event(), threading.Event()

    def fake_warmup():
        seen["during"] = main._WARMUP["state"]
        gate.wait(2)
        main._set_warmup("done", "fake")
        fin.set()

    monkeypatch.setattr(main, "warmup", fake_warmup)
    monkeypatch.setattr(main.C, "WARMUP_ON_START", True)
    with TestClient(main.app) as c:
        assert c.get("/health").json()["warmup"] == "running"
        gate.set()
        assert fin.wait(2)
        assert c.get("/health").json()["warmup"] == "done"
    assert seen["during"] == "running"


def test_initial_warmup_state_is_idle():
    assert main._WARMUP["state"] == "idle"


def test_ui_helpers_warming_states():
    for s in ("idle", "done", "error", None):
        assert not ui_helpers.is_warming(s) and not ui_helpers.should_poll(s)
    assert ui_helpers.is_warming("running") and ui_helpers.should_poll("running")


def test_health_never_blocks_on_warmup(client, monkeypatch):
    main._set_warmup("running", "")
    assert client.get("/health").json()["warmup"] == "running"


def test_ui_helpers_should_auto_run():
    assert ui_helpers.should_auto_run(True, True, "a", None, warming=False) is True
    assert ui_helpers.should_auto_run(True, True, "a", "a", warming=False) is False   # sig không đổi
    assert ui_helpers.should_auto_run(True, True, "a", None, warming=True) is False   # đang warm-up
    assert ui_helpers.should_auto_run(False, True, "a", None, warming=False) is False
    assert ui_helpers.should_auto_run(True, False, "a", None, warming=False) is False


def test_ui_helpers_device_line_and_default_engine():
    dev = {"name": "Intel Arc", "backend": "openvino", "runtime": "openvino", "dtype": "fp16"}
    assert ui_helpers.device_line(dev) == "Thiết bị: Intel Arc · openvino (openvino/fp16)"
    assert ui_helpers.pick_default_engine({"default_engine": "rvm"}, ["cf", "rvm"]) == "rvm"
    assert ui_helpers.pick_default_engine({}, ["cf", "rvm"]) == "cf"
    assert ui_helpers.pick_default_engine({"default_engine": "zzz"}, ["cf"]) == "cf"
