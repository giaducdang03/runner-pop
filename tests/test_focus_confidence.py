"""Cảnh báo mức tin cậy của engine focus (FR-126)."""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import main
from backend.pipeline import config as C
from backend.pipeline.focusmap import focus_confidence
from backend.pipeline.segment import build_cutout
from backend.pipeline.trace import Trace
from tests.test_focus_shared_map import _scene


def _mask():
    m = np.zeros((100, 100), np.uint8)
    m[30:70, 30:70] = 255
    return m


def test_confidence_high_when_background_is_blurred():
    smap = np.full((100, 100), 0.05, np.float32)
    smap[30:70, 30:70] = 0.8
    assert focus_confidence(smap, _mask()) > 2.0


def test_confidence_low_when_background_is_sharp():
    smap = np.full((100, 100), 0.7, np.float32)
    smap[30:70, 30:70] = 0.8
    assert focus_confidence(smap, _mask()) < C.FOCUS_CONF_MIN


def test_confidence_on_flat_image_is_one():
    assert focus_confidence(np.zeros((100, 100), np.float32), _mask()) == 1.0


def test_confidence_without_background_or_person_is_one():
    full = np.full((100, 100), 255, np.uint8)
    assert focus_confidence(np.ones((100, 100), np.float32), full) == 1.0
    assert focus_confidence(np.ones((100, 100), np.float32), np.zeros((100, 100), np.uint8)) == 1.0


def test_trace_has_meta_dict():
    t = Trace()
    t.meta["focus_confidence"] = 2.0
    assert t.meta == {"focus_confidence": 2.0}


def test_engine_records_confidence_in_trace_and_note(make_scene):
    scene = _scene(make_scene)
    c = build_cutout(scene, scene.persons[0], "focus")
    assert c.trace.meta["focus_confidence"] > C.FOCUS_CONF_MIN
    st = next(s for s in c.trace.stages if s.key == "focus_mask")
    assert "độ tin cậy" in st.note


@pytest.fixture
def client():
    return TestClient(main.app)


def test_process_response_includes_focus_confidence(client, make_scene):
    scene = _scene(make_scene)
    main._put("sid-conf", main.Session(rgb=scene.rgb, scene=scene))
    j = client.post("/process/sid-conf?method=focus").json()
    assert j["focus_confidence"] is not None
    r = client.post("/process/sid-conf?method=cf").json()
    assert r["focus_confidence"] is None


def test_options_exposes_threshold(client):
    assert client.get("/options").json()["focus_conf_min"] == C.FOCUS_CONF_MIN
