"""POST/GET /prepare (E1). Không nạp model thật: dựng cảnh giả và build_cutout giả."""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import main
from backend import pipeline as P
from backend.pipeline.schema import Person, Scene

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontend"))
import ui_helpers  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_prepare(monkeypatch):
    with main._PREPARE_LOCK:
        main._PREPARE.clear()
    monkeypatch.setattr(main, "_dummy_scene_person",
                        lambda: (Scene(rgb=np.zeros((8, 8, 3), np.uint8), persons=[]),
                                 Person(idx=-1, box=(1, 1, 5, 5), conf=1.0, mask=None)))
    yield
    with main._PREPARE_LOCK:
        main._PREPARE.clear()


@pytest.fixture
def client():
    return TestClient(main.app)


def _cutout(device="GPU (fake)"):
    z = np.zeros((8, 8), np.float32)
    img = np.zeros((8, 8, 3), np.float32)
    tr = P.Trace()
    with tr.stage("cf", "cf"):
        pass
    with tr.stage("birefnet", "BiRefNet") as h:
        h.device = device
    return P.Cutout(image=img, alpha=z, foreground=img, background=img,
                    trimap=np.zeros((8, 8), np.uint8), trace=tr)


def _wait(client, engine, device="auto", want=("done", "error"), t=3.0):
    end = time.time() + t
    while time.time() < end:
        j = client.get(f"/prepare?engine={engine}&device={device}").json()
        if j["state"] in want:
            return j
        time.sleep(0.01)
    raise AssertionError(j)


def test_unknown_engine_400_bad_device_422(client):
    assert client.post("/prepare?engine=zzz").status_code == 400
    assert client.get("/prepare?engine=zzz").status_code == 400
    assert client.post("/prepare?engine=cf&device=tpu").status_code == 422
    assert client.get("/prepare?engine=cf&device=tpu").status_code == 422


def test_get_before_post_is_idle(client):
    j = client.get("/prepare?engine=cf").json()
    assert j == {"engine": "cf", "device_mode": "auto", "state": "idle", "detail": "", "used_device": ""}


def test_single_flight_and_running_to_done(client, monkeypatch):
    gate, calls = threading.Event(), []

    def fake_build(scene, person, method, params, trace=None):
        calls.append((method, params.device_mode, params.cache))
        gate.wait(3)
        return _cutout("Intel Arc (fp16)")

    monkeypatch.setattr(P, "build_cutout", fake_build)
    for _ in range(3):
        assert client.post("/prepare?engine=cf&device=gpu").json()["state"] == "running"
    assert client.get("/prepare?engine=cf&device=gpu").json()["state"] == "running"
    gate.set()
    j = _wait(client, "cf", "gpu")
    assert j["state"] == "done" and j["used_device"] == "Intel Arc (fp16)"
    client.post("/prepare?engine=cf&device=gpu")          # đã done: không chạy lại
    time.sleep(0.05)
    assert calls == [("cf", "gpu", None)]                 # đúng 1 lần, cache=None


def test_error_state_and_retry(client, monkeypatch):
    n = {"i": 0}

    def fake_build(scene, person, method, params, trace=None):
        n["i"] += 1
        if n["i"] == 1:
            raise RuntimeError("x" * 500)
        return _cutout()

    monkeypatch.setattr(P, "build_cutout", fake_build)
    client.post("/prepare?engine=cf")
    j = _wait(client, "cf")
    assert j["state"] == "error" and j["detail"].startswith("RuntimeError: x") and len(j["detail"]) <= 200
    client.post("/prepare?engine=cf")                     # thử lại
    assert _wait(client, "cf")["state"] == "done" and n["i"] == 2


def test_used_device_empty_when_no_device_stage(client, monkeypatch):
    def fake_build(scene, person, method, params, trace=None):
        c = _cutout()
        c.trace = P.Trace()
        return c

    monkeypatch.setattr(P, "build_cutout", fake_build)
    client.post("/prepare?engine=cf")
    assert _wait(client, "cf")["used_device"] == ""


def test_pairs_independent(client, monkeypatch):
    seen = []

    def fake_build(scene, person, method, params, trace=None):
        seen.append((method, params.device_mode))
        return _cutout()

    monkeypatch.setattr(P, "build_cutout", fake_build)
    client.post("/prepare?engine=cf&device=cpu")
    _wait(client, "cf", "cpu")
    assert client.get("/prepare?engine=cf&device=gpu").json()["state"] == "idle"
    other = [n for n in P.engine_names() if n != "cf"][0]
    client.post(f"/prepare?engine={other}&device=cpu")
    _wait(client, other, "cpu")
    assert sorted(seen) == sorted([("cf", "cpu"), (other, "cpu")])


def test_prepare_engine_never_raises(monkeypatch):
    monkeypatch.setattr(P, "build_cutout", lambda *a, **k: (_ for _ in ()).throw(ValueError("boom")))
    main._prepare_run("cf", "auto")
    assert main._PREPARE[("cf", "auto")]["state"] == "error"


# ------------------------------------------------------------------ ui_helpers
def test_engine_ready():
    for s in ("done", "error", None, "", "weird"):
        assert ui_helpers.engine_ready(s) is True
    for s in ("idle", "running"):
        assert ui_helpers.engine_ready(s) is False


def test_should_poll_prepare():
    assert ui_helpers.should_poll_prepare("running") and ui_helpers.should_poll_prepare("idle")
    for s in ("done", "error", None, "weird"):
        assert not ui_helpers.should_poll_prepare(s)


def test_prepare_message():
    m = ui_helpers.prepare_message("BiRefNet", "gpu")
    assert "BiRefNet" in m and "gpu" in m and m.startswith("Đang nạp model cho 'BiRefNet' trên gpu")


def test_should_auto_run_engine_ready():
    assert ui_helpers.should_auto_run(True, True, "a", None, warming=False, engine_ready=True) is True
    assert ui_helpers.should_auto_run(True, True, "a", None, warming=False, engine_ready=False) is False
    assert ui_helpers.should_auto_run(True, True, "a", None, warming=False) is True      # mặc định ready


def test_prepare_note():
    assert ui_helpers.prepare_note("gpu", "CPU (x)") != ""
    assert ui_helpers.prepare_note("gpu", "Intel GPU") == ""
    assert ui_helpers.prepare_note("auto", "CPU") == ""
    assert ui_helpers.prepare_note("gpu", "") == ""


# ------------------------------------------------------------------ fix round 1 (R31)
def test_should_post_prepare():
    assert ui_helpers.should_post_prepare("idle") is True
    for s in ("running", "done", "error", None, "", "weird"):
        assert ui_helpers.should_post_prepare(s) is False


def test_fragment_action():
    assert ui_helpers.fragment_action("running") == "poll"
    assert ui_helpers.fragment_action("idle") == "post"
    for s in ("done", "error", None):
        assert ui_helpers.fragment_action(s) == "rerun"


def test_sidebar_prepare_plan():
    assert ui_helpers.sidebar_prepare_plan("idle") == {"post": True, "ready": False, "show_error": False, "polling": True}
    assert ui_helpers.sidebar_prepare_plan("running") == {"post": False, "ready": False, "show_error": False, "polling": True}
    assert ui_helpers.sidebar_prepare_plan("done") == {"post": False, "ready": True, "show_error": False, "polling": False}
    assert ui_helpers.sidebar_prepare_plan("error") == {"post": False, "ready": True, "show_error": True, "polling": False}
    assert ui_helpers.sidebar_prepare_plan(None)["post"] is False and ui_helpers.sidebar_prepare_plan(None)["ready"]


class _FakeBackend:
    """Backend giả: POST trên idle / error bắt đầu một lần chạy; lần chạy luôn thất bại."""
    def __init__(self):
        self.state, self.runs = "idle", 0

    def get(self):
        return self.state

    def post(self):
        if self.state in ("idle", "error"):
            self.runs += 1
            self.state = "error"        # lỗi dai dẳng: chạy xong là error
        return self.state


def test_persistent_error_does_not_loop():
    be = _FakeBackend()
    for _ in range(6):                  # 6 lần chạy lại toàn bộ script
        state = be.get()
        plan = ui_helpers.sidebar_prepare_plan(state)
        if plan["post"]:
            state = be.post()
        if plan["polling"]:             # fragment: vòng poll tới khi rời running
            act = ui_helpers.fragment_action(be.get())
            if act == "post":
                be.post()
    assert be.runs == 1
    plan = ui_helpers.sidebar_prepare_plan(be.get())
    assert plan["ready"] and plan["show_error"] and not plan["post"]
