"""API bộ lọc nền: /options, /bgfilter.jpg, /bgfilter_previews, tham số filter cho /compose và /report.pdf."""
from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import main
from backend import pipeline as P
from backend.pipeline.schema import Person, Scene


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture
def sid(make_cutout):
    c = make_cutout(120, 160)
    rgb = (c.image * 255).astype(np.uint8)
    scene = Scene(rgb=rgb, persons=[Person(idx=0, box=(40, 30, 110, 90), conf=0.9, mask=None)])
    main._put("sid-bgf", main.Session(rgb=rgb, scene=scene, cutout=c))
    return "sid-bgf"


def _decode(data: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def test_options_lists_bg_filters(client):
    j = client.get("/options").json()
    names = [f["name"] for f in j["bg_filters"]]
    assert {"none", "bw", "solid"} <= set(names)
    assert all({"name", "label"} <= set(f) for f in j["bg_filters"])


def test_bgfilter_jpg_ok(client, sid):
    r = client.get(f"/session/{sid}/bgfilter.jpg", params={"filter": "blur"})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert _decode(r.content).shape[:2] == (120, 160)


def test_bgfilter_errors(client, sid):
    assert client.get(f"/session/{sid}/bgfilter.jpg", params={"filter": "nope"}).status_code == 400
    assert client.get(f"/session/{sid}/bgfilter.jpg", params={"filter": "solid", "color": "red"}).status_code == 400
    main._put("sid-nocut", main.Session(rgb=np.zeros((10, 10, 3), np.uint8)))
    assert client.get("/session/sid-nocut/bgfilter.jpg").status_code == 409


def test_previews_list(client, sid, monkeypatch):
    import backend.pipeline.bgfilter as bgf
    calls = []
    real = bgf.apply_bg_filter
    monkeypatch.setattr(bgf, "apply_bg_filter", lambda c, n, col="#808080": (calls.append(n), real(c, n, col))[1])
    r = client.get(f"/session/{sid}/bgfilter_previews")
    assert r.status_code == 200
    items = r.json()
    assert [i["name"] for i in items] == P.BG_FILTERS.names()
    for i in items:
        assert i["image"].startswith("data:image/jpeg;base64,")
        img = _decode(base64.b64decode(i["image"].split(",", 1)[1]))
        assert max(img.shape[:2]) <= 240
    n = len(calls)
    client.get(f"/session/{sid}/bgfilter_previews")
    assert len(calls) == n                                    # lần hai lấy từ cache
    assert client.get(f"/session/{sid}/bgfilter_previews", params={"color": "red"}).status_code == 400
    assert client.get("/session/sid-nocut/bgfilter_previews").status_code == 409


def test_compose_default_unchanged(client, sid):
    r = client.get(f"/session/{sid}/compose", params={"intensity": 1.0, "fmt": "png"})
    ref = P.compose(main._get(sid).cutout, 1.0)
    assert np.array_equal(_decode(r.content), cv2.cvtColor(ref, cv2.COLOR_RGB2BGR))


def test_compose_with_filter(client, sid):
    base = client.get(f"/session/{sid}/compose", params={"fmt": "png"}).content
    r = client.get(f"/session/{sid}/compose", params={"fmt": "png", "filter": "solid", "color": "#ff0000"})
    assert r.status_code == 200 and r.content != base
    assert client.get(f"/session/{sid}/compose", params={"filter": "nope"}).status_code == 400


def test_report_passes_filter(client, sid, monkeypatch):
    seen = {}
    monkeypatch.setattr(main, "build_report", lambda *a, **k: seen.update(k) or b"%PDF")
    assert client.get(f"/session/{sid}/report.pdf", params={"filter": "blur"}).status_code == 200
    assert seen["bg_filter"] == "blur"
    assert client.get(f"/session/{sid}/report.pdf", params={"filter": "nope"}).status_code == 400


def test_previews_names_subset(client, sid):
    r = client.get(f"/session/{sid}/bgfilter_previews", params={"names": "blur,sepia"})
    assert r.status_code == 200 and [i["name"] for i in r.json()] == ["blur", "sepia"]
    assert client.get(f"/session/{sid}/bgfilter_previews", params={"names": "nope"}).status_code == 400
