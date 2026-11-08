"""Mở rộng mask gậy (polemask.py): gậy thứ hai cầm song song, tay cầm ở đầu gậy; không tràn ra nền
có cấu trúc (không cần weights, trừ ca ảnh thật)."""
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.pipeline import config as C
from backend.pipeline import polemask, thin
from backend.pipeline.matting import matte
from backend.pipeline.occlusion import exclusion_mask
from backend.pipeline.poles import Pole
from backend.pipeline.schema import Cutout, Person, Scene

H, W = 900, 1200
BODY = (520, 150, 640, 850)                  # x1, y1, x2, y2 -> body_h = 700
TOP, TIP = np.array([500.0, 420.0]), np.array([200.0, 700.0])
HALF_W = 2.0
D = (TIP - TOP) / np.linalg.norm(TIP - TOP)
N = np.array([-D[1], D[0]])
OFF0, OFF = 2.0, 14.0                        # gậy thứ hai: lệch 2 px ở chỗ nắm (TOP), 14 px ở đầu dưới
HANDLE_HW, HANDLE_LEN = 6, 26                # tay cầm ở đầu dưới gậy chính, thò ra ngoài tip


def _pt(t, s=0.0):
    p = TOP + t * (TIP - TOP) + s * N
    return int(round(p[0])), int(round(p[1]))


def _off(t):
    """Lệch ngang của trục gậy thứ hai tại t."""
    return OFF0 + t * (OFF - OFF0)


def _sky(rng):
    y = np.linspace(0, 1, H)[:, None, None]
    img = (1 - y) * np.array([150, 175, 210]) + y * np.array([195, 205, 225])
    img = np.broadcast_to(img, (H, W, 3)).copy()
    return img + rng.normal(0, 2.0, img.shape)


def _draw_poles(img, second=True, handle=True, other=False):
    def line(a, b, col, hw):
        cv2.line(img, tuple(int(round(v)) for v in a), tuple(int(round(v)) for v in b), col,
                 int(2 * hw), cv2.LINE_AA)
    line(TOP, TIP, (25, 25, 30), HALF_W)
    if second:
        line(TOP + OFF0 * N, TIP + OFF * N, (30, 35, 70), HALF_W)
    if other:                      # gậy song song của người khác: lệch đều 24 px, không qua tay người cầm
        line(TOP + 24 * N, TIP + 24 * N, (30, 35, 70), HALF_W)
    if handle:
        line(TIP - 6 * D, TIP + HANDLE_LEN * D, (220, 185, 40), HANDLE_HW)


def _scene(bg="sky", second=True, handle=True, blob=False, other=False):
    rng = np.random.default_rng(0)
    img = _sky(rng)
    if bg == "horizon":            # chân trời: nền dưới y = 560 tối, cắt ngang vùng tìm quanh gậy
        img[560:] = np.array([60, 95, 55]) + rng.normal(0, 2.0, (H - 560, W, 3))
    if bg == "texture":            # mảng kết cấu (ô bàn cờ 6 px) sát dải gậy
        yy, xx = np.mgrid[0:H, 0:W]
        chk = ((yy // 6 + xx // 6) % 2).astype(bool)
        x0, y0 = _pt(0.5)
        patch = (np.abs(xx - x0) < 60) & (np.abs(yy - y0) < 60) & chk
        img[patch] = (90, 90, 90)
    x1, y1, x2, y2 = BODY
    img[y1:y2, x1:x2] = (200, 60, 60)
    _draw_poles(img, second, handle, other)
    if blob:                       # khối rộng tối dính vào dải gậy (bóng, ba lô...)
        cv2.circle(img, _pt(0.5, -18.0), 16, (40, 40, 40), -1)
    rgb = np.clip(img, 0, 255).astype(np.uint8)
    mask = np.zeros((H, W), np.uint8)
    mask[y1:y2, x1:x2] = 255
    p = Person(idx=0, box=BODY, conf=0.9, mask=mask)
    p.body_box = BODY
    p.hands = [tuple(TOP - 4 * D)]                 # nắm tay ở đầu trên gậy
    p.poles = [Pole(top=tuple(TOP), tip=tuple(TIP), half_w=HALF_W, score=0.9, owner=0)]
    return Scene(rgb=rgb, persons=[p])


def _cutout(scene):
    f = scene.rgb.astype(np.float32) / 255.0
    a = (scene.persons[0].mask > 127).astype(np.float32)
    return Cutout(image=f, alpha=a, foreground=f.copy(), background=f.copy(),
                  trimap=(a * 255).astype(np.uint8))


def _run(scene, refine=True, monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setattr(C, "POLE_REFINE", refine)
    return thin.overlay_poles(_cutout(scene), scene, scene.persons[0], None).alpha


def _old_overlay(scene):
    """Bản overlay_poles trước khi có POLE_REFINE (chép nguyên vòng lặp cũ) làm mốc so sánh."""
    c = _cutout(scene)
    person = scene.persons[0]
    own = person.mask.copy()
    for pole in person.poles:
        cv2.fillConvexPoly(own, np.round(pole.polygon()).astype(np.int32), 255)
    excl = exclusion_mask(scene, person, own)
    alpha = c.alpha.copy()
    for pole in person.poles:
        box = thin._crop_box(pole, alpha.shape)
        x1, y1, x2, y2 = box
        a_pole = matte(c.image[y1:y2, x1:x2], thin._trimap(pole, box), "cf")
        a_pole[excl[y1:y2, x1:x2]] = 0.0
        alpha[y1:y2, x1:x2] = np.maximum(alpha[y1:y2, x1:x2], a_pole)
    return alpha


def _at(a, p):
    return float(a[p[1], p[0]])


def test_second_pole_and_handle_recovered(monkeypatch):
    sc = _scene()
    a = _run(sc, True, monkeypatch)
    for t in np.linspace(0.3, 0.95, 14):
        assert _at(a, _pt(t, _off(t))) >= 0.8, (t, _at(a, _pt(t, _off(t))))
    hp = TIP + (HANDLE_LEN - 8) * D                            # giữa phần tay cầm thò ra ngoài tip
    for s in (-3.0, 0.0, 3.0):
        q = hp + s * N
        assert _at(a, (int(round(q[0])), int(round(q[1])))) >= 0.8, s
    for t in np.linspace(0.1, 0.9, 9):                         # xa gậy: nền giữ nguyên
        assert _at(a, _pt(t, -25.0)) <= 0.1
        assert _at(a, _pt(t, OFF + 25.0)) <= 0.1
    # tắt mở rộng: gậy thứ hai không được tô
    b = _run(sc, False, monkeypatch)
    assert np.mean([_at(b, _pt(t, _off(t))) for t in np.linspace(0.3, 0.95, 14)]) < 0.5


def test_parallel_pole_of_someone_else_not_added(monkeypatch):
    sc = _scene(second=False, other=True)
    a = _run(sc, True, monkeypatch)
    for t in np.linspace(0.2, 0.9, 15):
        assert _at(a, _pt(t, 24.0)) <= 0.1, (t, _at(a, _pt(t, 24.0)))
    hp = TIP + (HANDLE_LEN - 8) * D                            # tay cầm vẫn được thêm
    assert _at(a, (int(round(hp[0])), int(round(hp[1])))) >= 0.8
    sc.persons[0].hands = []                                   # không có tay: gậy thứ hai không được thêm
    sc2 = _scene()
    sc2.persons[0].hands = []
    b = _run(sc2, True, monkeypatch)
    assert np.mean([_at(b, _pt(t, _off(t))) for t in np.linspace(0.3, 0.95, 14)]) < 0.5


def test_refine_off_matches_old_behaviour(monkeypatch):
    sc = _scene()
    assert np.array_equal(_run(sc, False, monkeypatch), _old_overlay(sc))


def test_refine_exception_falls_back(monkeypatch):
    sc = _scene()
    monkeypatch.setattr(polemask, "_refine", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    assert np.array_equal(_run(sc, True, monkeypatch), _old_overlay(sc))


def test_horizon_does_not_leak(monkeypatch):
    sc = _scene("horizon", second=False, handle=False)
    a = _run(sc, True, monkeypatch)
    # gậy cắt chân trời y = 560 tại t ~ 0.5: các điểm hai bên gậy, sát chân trời ở cả hai phía
    for t in np.linspace(0.3, 0.7, 9):
        for s in (-22.0, -12.0, 12.0, 22.0):
            assert _at(a, _pt(t, s)) <= 0.1, (t, s, _at(a, _pt(t, s)))
    for x in range(250, 520, 10):                              # ngay trên / dưới chân trời
        for y in (552, 568):
            if abs((np.array([x, y]) - TOP) @ N) > 10:
                assert a[y, x] <= 0.1, (x, y, a[y, x])


def test_texture_does_not_leak(monkeypatch):
    sc = _scene("texture", second=False, handle=False)
    a = _run(sc, True, monkeypatch)
    for t in np.linspace(0.4, 0.6, 9):
        for s in (-30.0, -15.0, -9.0, 9.0, 15.0, 30.0):
            assert _at(a, _pt(t, s)) <= 0.1, (t, s, _at(a, _pt(t, s)))


def test_wide_blob_rejected(monkeypatch):
    sc = _scene(second=False, handle=False, blob=True)
    a = _run(sc, True, monkeypatch)
    for s in (-12.0, -18.0, -24.0, -30.0):
        assert _at(a, _pt(0.5, s)) <= 0.1, (s, _at(a, _pt(0.5, s)))


def test_refine_returns_none_on_bad_input():
    sc = _scene()
    p = sc.persons[0]
    assert polemask.refine_pole_mask(sc.rgb, replace(p.poles[0], tip=p.poles[0].top), 700) is None
    assert polemask.refine_pole_mask(None, p.poles[0], 700) is None
    pm = polemask.refine_pole_mask(sc.rgb, p.poles[0], 700, p.mask > 127, p.hands)
    assert pm is not None and pm.ratio > 1.5


# --- ảnh thật -----------------------------------------------------------------

DATA = Path(__file__).parent / "data" / "photo_5377501.jpg"


@pytest.mark.real
def test_real_two_poles_and_handle(monkeypatch):
    import copy
    from backend.pipeline import segment
    from backend.pipeline.detect import detect_scene, get_yolo
    monkeypatch.setattr(C, "POLE_ENABLE", True)
    monkeypatch.setattr(C, "POLE_REFINE", True)
    try:
        get_yolo(C.POSE_WEIGHTS)
        get_yolo(C.YOLO_WEIGHTS)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"không nạp được weights YOLO: {e}")
    bgr = cv2.imread(str(DATA))
    assert bgr.shape[:2] == (1064, 1600)
    scene = detect_scene(np.ascontiguousarray(bgr[..., ::-1]))
    main = max(scene.persons, key=lambda p: p.area)
    saved = {}

    def spy(cutout, *args, **kw):            # giữ cutout của engine để so với khi tắt mở rộng
        saved["c"] = copy.deepcopy(cutout)
        return thin.overlay_poles(cutout, *args, **kw)
    monkeypatch.setattr(segment, "overlay_poles", spy)
    c = segment.build_cutout(scene, main, "cf")
    assert main.poles, "người chạy phải có gậy"
    a = c.alpha
    # gậy thứ hai (xám, phía trên gậy dò được; đo trên ảnh: tâm tối nhất theo cột x, cách trục gậy
    # dò được 7-8 px theo chiều dọc, ngoài vùng unknown của dải cũ)
    for x, y in ((620, 540), (635, 548), (650, 555), (665, 562), (680, 569)):
        assert a[y, x] >= 0.8, ("gậy thứ hai", x, y, a[y, x])
    # tay cầm vàng của gậy thứ hai và tay cầm tối của gậy dò được (đầu trên trái, ngoài dải gậy)
    for x, y in ((596, 527), (603, 529), (585, 537)):
        assert a[y, x] >= 0.8, ("tay cầm", x, y, a[y, x])
    # mây / chân trời / núi cách gậy >= 40 px
    for x, y in ((600, 600), (640, 610), (700, 625), (560, 590), (660, 520), (820, 690)):
        assert a[y, x] <= 0.1, ("nền", x, y, a[y, x])
    # không tràn: phần alpha tăng thêm do mở rộng nằm hết trong 40 px quanh dải gậy
    near = np.zeros(a.shape, np.uint8)
    for pl in main.poles:
        cv2.fillConvexPoly(near, np.round(replace(pl, half_w=pl.half_w + 40).polygon()).astype(np.int32), 1)
        for e in (pl.top, pl.tip):
            cv2.circle(near, tuple(int(round(v)) for v in e), 40, 1, -1)
    monkeypatch.setattr(C, "POLE_REFINE", False)
    off = thin.overlay_poles(saved["c"], scene, main, None).alpha
    spill = float(np.clip(a - off, 0, None)[near == 0].sum())
    gain = float(np.clip(a - off, 0, None)[near > 0].sum())
    print(f"spill ngoài 40 px = {spill:.1f}, thêm trong 40 px = {gain:.1f}")
    assert spill < 1.0
    assert gain > 500
