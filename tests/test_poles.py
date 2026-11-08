"""Test dò gậy từ nắm tay: ảnh tổng hợp (không cần weights) + một test ảnh thật (marker real)."""
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.pipeline import poles
from backend.pipeline.poles import Pole
from backend.pipeline.roi import Roi
from backend.pipeline.schema import Person

H, W = 600, 400
HAND = (260, 250)


def _canvas():
    rgb = np.full((H, W, 3), 128, np.uint8)
    rgb[100:500, 150:250] = (200, 60, 60)
    return rgb


def _block_mask(y1, y2, x1, x2):
    m = np.zeros((H, W), np.uint8)
    m[y1:y2, x1:x2] = 255
    return m


def _person(idx=0, hands=(HAND,), mask=None, box=(150, 100, 250, 500)):
    if mask is None:
        mask = _block_mask(box[1], box[3], box[0], box[2])
    return Person(idx=idx, box=box, conf=0.9, mask=mask, hands=list(hands))


def _angle(p0, p1):
    return math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))


def _pole_angle(pl: Pole):
    return _angle(pl.top, pl.tip)


def test_finds_dark_pole_from_hand():
    rgb = _canvas()
    cv2.line(rgb, (262, 240), (300, 560), (20, 20, 20), 6)
    found = poles.find_poles(rgb, [_person()])
    assert len(found) == 1
    pl = found[0]
    assert abs(_pole_angle(pl) - _angle((262, 240), (300, 560))) <= 3
    assert pl.score >= 0.5
    assert pl.tip[1] >= 540
    assert pl.owner == 0


def test_edge_of_large_region_is_not_a_pole():
    rgb = _canvas()
    rgb[:, 260:] = 40
    assert poles.find_poles(rgb, [_person()]) == []


def test_horizontal_bar_rejected_by_angle_prior():
    rgb = _canvas()
    cv2.line(rgb, (262, 252), (395, 252), (20, 20, 20), 6)
    assert poles.find_poles(rgb, [_person()]) == []


def test_pole_owned_by_hand_whose_mask_wraps_grip():
    rgb = _canvas()
    rgb[100:500, 320:400] = (60, 60, 200)
    cv2.line(rgb, (262, 240), (300, 560), (20, 20, 20), 6)
    mask_a = _block_mask(100, 500, 150, 250)
    cv2.circle(mask_a, (263, 250), 20, 255, -1)          # bàn tay A bọc quanh chỗ nắm
    a = _person(idx=0, hands=[HAND], mask=mask_a)
    b = _person(idx=1, hands=[(268, 290)], box=(320, 100, 400, 500))   # tay B cách tay A ~40 px
    found = poles.find_poles(rgb, [a, b])
    assert len(found) == 1
    assert found[0].owner == a.idx
    assert not any(pl.owner == b.idx for pl in found)


def test_textured_background_is_not_a_pole():
    """Nền nhiều vân (kiểu lá cây mờ) không được sinh gậy."""
    rng = np.random.default_rng(0)
    tex = rng.integers(40, 216, (H // 8, W // 8, 3), dtype=np.uint8)
    rgb = cv2.resize(tex, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int16)
    rgb = np.clip(rgb + rng.normal(0, 10, rgb.shape), 0, 255).astype(np.uint8)
    rgb[100:500, 150:250] = (200, 60, 60)
    assert poles.find_poles(rgb, [_person()]) == []


def test_tip_stops_where_pole_ends_not_in_sparse_texture():
    """Gậy đặc từ tay xuống y=420, sau đó trên cùng đường thẳng chỉ còn vệt lổn nhổn (như cỏ):
    hụt nhỏ hơn POLE_GAP * h nên đường dò vẫn đi tiếp được, nhưng đầu gậy phải dừng ở cuối đoạn đặc."""
    rgb = _canvas()

    def x_at(y):
        return 262 + (y - 240) * 38 // 320

    cv2.line(rgb, (x_at(240), 240), (x_at(420), 420), (20, 20, 20), 6)
    for y in range(432, 590, 19):                       # chấm ~7 px mỗi 19 px: mật độ ~0.35 như đuôi
        cv2.line(rgb, (x_at(y), y), (x_at(y + 1), y + 1), (20, 20, 20), 6)   # qua cỏ thật, hụt < 0.05h
    found = poles.find_poles(rgb, [_person()])
    assert len(found) == 1
    assert 400 <= found[0].tip[1] <= 440


def test_dashed_bar_is_not_a_continuous_pole():
    """Thanh đứt quãng (khoảng hụt > POLE_GAP * h) không được tính là một gậy liền."""
    rgb = _canvas()
    for y in range(240, 600, 66):
        cv2.line(rgb, (262 + (y - 240) * 38 // 320, y), (262 + (y + 36 - 240) * 38 // 320, y + 36),
                 (20, 20, 20), 6)
    assert poles.find_poles(rgb, [_person()]) == []


def test_half_width_prefers_narrow_strip():
    rgb = _canvas()
    cv2.line(rgb, (262, 240), (300, 560), (20, 20, 20), 3)
    found = poles.find_poles(rgb, [_person()])
    assert len(found) == 1
    assert found[0].half_w <= 0.005 * 400


def test_people_without_hands_get_no_poles():
    rgb = _canvas()
    cv2.line(rgb, (262, 240), (300, 560), (20, 20, 20), 6)
    assert poles.find_poles(rgb, [_person(hands=())]) == []


def test_pole_to_work_and_polygon():
    pl = Pole(top=(110.0, 220.0), tip=(130.0, 420.0), half_w=4.0, score=0.8, owner=2)
    roi = Roi(x1=100, y1=200, x2=300, y2=600, scale=0.5, full_shape=(800, 600))
    w = pl.to_work(roi)
    assert np.allclose(w.top, (5, 10)) and np.allclose(w.tip, (15, 110))
    assert w.half_w == pytest.approx(2.0)
    assert w.score == pl.score and w.owner == pl.owner
    poly = pl.polygon()
    assert poly.shape == (4, 2)
    # bề rộng dải = 2 * half_w + 2
    assert np.linalg.norm(poly[0] - poly[1]) == pytest.approx(2 * 4.0 + 2)


# --- ảnh thật -----------------------------------------------------------------

DATA = Path(__file__).parent / "data" / "photo_5394198.jpg"


def _iou(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def _seg_dist(p, a, b):
    p, a, b = (np.asarray(v, float) for v in (p, a, b))
    t = np.clip(np.dot(p - a, b - a) / max(np.dot(b - a, b - a), 1e-9), 0, 1)
    return float(np.linalg.norm(p - (a + t * (b - a))))


@pytest.mark.real
def test_real_image_poles(monkeypatch):
    from backend.pipeline import config as C
    from backend.pipeline.detect import detect_scene, get_yolo
    monkeypatch.setattr(C, "POLE_ENABLE", True)
    try:
        get_yolo(C.POSE_WEIGHTS)
        get_yolo(C.YOLO_WEIGHTS)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"không nạp được weights YOLO: {e}")
    bgr = cv2.imread(str(DATA))
    h, w = bgr.shape[:2]
    s = 2560 / max(h, w)
    bgr = cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    rgb = np.ascontiguousarray(bgr[..., ::-1])
    scene = detect_scene(rgb)
    # detect_scene chỉ lưu gậy ứng viên (chưa gộp) vào scene.pole_candidates
    found = [pl for pls in scene.pole_candidates.values() for pl in pls]

    main = max(scene.persons, key=lambda p: _iou(p.box, (1225, 5, 2345, 1703)))
    assert _iou(main.box, (1225, 5, 2345, 1703)) >= 0.7
    mine = [pl for pl in found if pl.owner == main.idx]
    angles = sorted(_pole_angle(pl) for pl in mine)
    assert len(mine) == 2, angles
    assert abs(angles[0] - 79) <= 4 and abs(angles[1] - 87.5) <= 4, angles
    for pl in mine:                       # dải gậy không được rộng quá
        assert pl.half_w <= 0.010 * main.body_h, pl   # đo được 0.008h; chừa biên

    p79 = next(pl for pl in mine if abs(_pole_angle(pl) - 79) <= 4)
    for pl in found:
        if pl.owner == main.idx:
            continue
        near = _seg_dist(pl.top, p79.top, p79.tip) < 50
        assert not (abs(_pole_angle(pl) - 79) <= 4 and near), (pl, p79)

    red = max(scene.persons, key=lambda p: _iou(p.box, (315, 187, 1071, 1526)))
    assert _iou(red.box, (315, 187, 1071, 1526)) >= 0.6
    red_angles = [_pole_angle(pl) for pl in found if pl.owner == red.idx]
    assert any(abs(a - 75) <= 6 for a in red_angles), red_angles

    # người áo trắng ở mép phải cầm gậy tre chĩa xuống - trái
    white = max(scene.persons, key=lambda p: _iou(p.box, (2232, 28, 2560, 1661)))
    assert _iou(white.box, (2232, 28, 2560, 1661)) >= 0.6
    white_angles = [_pole_angle(pl) for pl in found if pl.owner == white.idx]
    assert any(abs(a - 111) <= 6 for a in white_angles), white_angles

    # số gậy có giới hạn; người nhỏ ở hậu cảnh (thấp hơn nửa người chạy chính) không có gậy
    assert len(found) <= 6, found
    small = {p.idx for p in scene.persons if p.body_h < 0.5 * main.body_h}
    assert small and not any(pl.owner in small for pl in found), found
