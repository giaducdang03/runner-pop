"""Alpha gậy mảnh ở độ phân giải gốc: phủ lên cutout của mọi engine (không cần weights)."""
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.pipeline.poles import Pole
from backend.pipeline.schema import Person, Scene
from backend.pipeline.segment import birefnet as bf
from backend.pipeline.segment import build_cutout

H, W = 1600, 2400
TOP, TIP = (1200.0, 700.0), (1290.0, 1500.0)
HALF_W = 4.0


def _mask(y1, y2, x1, x2):
    m = np.zeros((H, W), np.uint8)
    m[y1:y2, x1:x2] = 255
    return m


def _scene(with_pole=True, other=False):
    rgb = np.full((H, W, 3), 170, np.uint8)
    rgb[300:1300, 1000:1150] = (200, 60, 60)
    p0, p1 = np.array(TOP), np.array(TIP)
    cv2.line(rgb, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)),
             (20, 20, 20), int(2 * HALF_W))
    people = [Person(idx=0, box=(1000, 300, 1150, 1300), conf=0.9, mask=_mask(300, 1300, 1000, 1150))]
    if with_pole:
        people[0].poles = [Pole(top=TOP, tip=TIP, half_w=HALF_W, score=0.9, owner=0)]
    if other:
        rgb[1000:1500, 1230:1330] = (40, 200, 40)
        people.append(Person(idx=1, box=(1230, 1000, 1330, 1500), conf=0.9,
                             mask=_mask(1000, 1500, 1230, 1330)))
    return Scene(rgb=rgb, persons=people)


@pytest.fixture
def body_only_birefnet(monkeypatch):
    """birefnet_alpha giả: chỉ trả thân người (không thấy gậy)."""
    def fake(rgb, device=None, stage=None):
        a = np.zeros(rgb.shape[:2], np.float32)
        ys, xs = np.nonzero((np.abs(rgb.astype(int) - (200, 60, 60)).sum(-1) < 30))
        a[ys, xs] = 1.0
        return a
    monkeypatch.setattr(bf, "birefnet_alpha", fake)


def _axis_pts(n=20, lo=0.2, hi=0.95):
    return [(TOP[0] + t * (TIP[0] - TOP[0]), TOP[1] + t * (TIP[1] - TOP[1]))
            for t in np.linspace(lo, hi, n)]


def test_thin_pole_alpha_survives_downscale(body_only_birefnet):
    scene = _scene()
    c = build_cutout(scene, scene.persons[0], "birefnet")
    for x, y in _axis_pts():
        assert c.alpha[int(round(y)), int(round(x))] >= 0.8, (x, y)
        assert (c.foreground[int(round(y)), int(round(x))] < 0.2).all()
        assert c.alpha[int(round(y)), int(round(x)) - 20] <= 0.1
    assert any(s["key"] == "thin" for s in c.trace.to_list())


def test_overlay_respects_other_person_exclusion(body_only_birefnet):
    scene = _scene(other=True)
    c = build_cutout(scene, scene.persons[0], "birefnet")
    om = scene.persons[1].mask > 127
    # phần giao của gậy với người đứng gần hơn (foot_y lớn hơn) không được tăng alpha
    line = np.zeros((H, W), np.uint8)
    cv2.line(line, tuple(int(v) for v in TOP), tuple(int(v) for v in TIP), 255, 6)
    inter = om & (line > 0)
    assert inter.sum() > 100
    assert c.alpha[inter].max() <= 0.01


def test_no_poles_cutout_unchanged(body_only_birefnet):
    scene = _scene(with_pole=False)
    before = build_cutout(scene, scene.persons[0], "birefnet")
    snap_alpha, snap_fg = before.alpha.copy(), before.foreground.copy()
    from backend.pipeline import thin
    after = thin.overlay_poles(before, scene, scene.persons[0], None)
    assert np.array_equal(after.alpha, snap_alpha)
    assert np.array_equal(after.foreground, snap_fg)
    assert not any(s["key"] == "thin" for s in before.trace.to_list())


def test_overlay_failure_leaves_cutout_unchanged(body_only_birefnet, monkeypatch):
    from backend.pipeline import thin
    monkeypatch.setattr(thin, "matte", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    scene = _scene()
    c = build_cutout(scene, scene.persons[0], "birefnet")
    assert c.alpha[1000, 1230] < 0.1
    assert any(s["key"] == "thin" for s in c.trace.to_list())


# --- ảnh thật -----------------------------------------------------------------

DATA = Path(__file__).parent / "data" / "photo_5394198.jpg"


@pytest.mark.real
def test_real_image_main_runner_poles_keep_alpha():
    from backend.pipeline import config as C
    from backend.pipeline.detect import detect_scene, get_yolo
    try:
        get_yolo(C.POSE_WEIGHTS)
        get_yolo(C.YOLO_WEIGHTS)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"không nạp được weights YOLO: {e}")
    bgr = cv2.imread(str(DATA))
    h, w = bgr.shape[:2]
    s = 2560 / max(h, w)
    bgr = cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    scene = detect_scene(np.ascontiguousarray(bgr[..., ::-1]))

    def iou(a, b):
        iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
        inter = max(iw, 0) * max(ih, 0)
        return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)

    main = max(scene.persons, key=lambda p: iou(p.box, (1225, 5, 2345, 1703)))
    assert main.poles, "người chạy chính phải có gậy"
    c = build_cutout(scene, main, "cf")
    # mask người đã gộp gậy: bỏ dải gậy (nở 6 px) để còn lại thân người
    body = (main.mask > 127).astype(np.uint8)
    for pl in main.poles:
        cv2.fillConvexPoly(body, np.round(replace(pl, half_w=pl.half_w + 6).polygon()).astype(np.int32), 0)
    body_d = cv2.dilate(body, np.ones((5, 5), np.uint8)) > 0
    checked = 0
    for pl in main.poles:
        top, tip = np.array(pl.top), np.array(pl.tip)
        d = (tip - top) / np.linalg.norm(tip - top)
        n = np.array([-d[1], d[0]])
        for t in np.linspace(0.3, 0.95, 20):
            x, y = (top + t * (tip - top)).round().astype(int)
            if 0 <= x < c.alpha.shape[1] and 0 <= y < c.alpha.shape[0]:
                side = np.round(np.array([x, y]) + 20 * n).astype(int)
                if body_d[y, x]:
                    continue
                checked += 1
                assert c.alpha[y, x] >= 0.8, (pl, t, c.alpha[y, x])
                sx, sy = side
                if 0 <= sx < c.alpha.shape[1] and 0 <= sy < c.alpha.shape[0] and not body_d[sy, sx]:
                    assert c.alpha[sy, sx] <= 0.1, (pl, t, c.alpha[sy, sx])
    assert checked >= 20, checked
