"""Tách gậy bằng MobileSAM (polesam.py) với model giả: không tải trọng số."""
import cv2
import numpy as np
import pytest

from backend.pipeline import config as C
from backend.pipeline import polesam, thin
from backend.pipeline.poles import Pole
from backend.pipeline.schema import Cutout, Person, Scene

pytestmark = pytest.mark.pole_sam

H, W = 900, 900
BODY_H = 700.0
TOP, END = np.array([500.0, 250.0]), np.array([350.0, 650.0])    # gậy thật
OVER = END + 0.3 * (END - TOP)                                     # trục dò kéo quá đầu gậy thật
HAND = (TOP + 0.05 * (END - TOP))


def _image():
    img = np.full((H, W, 3), (200, 210, 225), np.uint8)
    cv2.line(img, tuple(TOP.astype(int)), tuple(END.astype(int)), (20, 20, 25), 5, cv2.LINE_AA)
    return img


class _Masks:
    def __init__(self, m):
        self.data = self

        self._m = m

    def __len__(self):
        return 1

    def __getitem__(self, i):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._m.astype(np.float32)


class _Res:
    def __init__(self, m):
        self.masks = _Masks(m)


class _FakeSam:
    """"SAM" giả: pixel tối của crop (gậy tối trên trời sáng) hoặc cả crop (blob=True)."""
    def __init__(self, blob=False):
        self.blob, self.calls = blob, []

    def __call__(self, crop, points, labels, verbose=False):
        self.calls.append((np.asarray(points[0]), np.asarray(labels[0])))
        m = np.ones(crop.shape[:2], bool) if self.blob else crop.mean(axis=2) < 120
        return [_Res(m)]


def _pole(tip=OVER, top=TOP, shift=0.0):
    d = (tip - top) / np.linalg.norm(tip - top)
    n = np.array([-d[1], d[0]])
    a, b = top + shift * n, tip + shift * n
    return Pole(top=tuple(a), tip=tuple(b), half_w=2.5, score=1.0, owner=0)


@pytest.fixture
def fake(monkeypatch):
    def use(model):
        monkeypatch.setattr(polesam, "_get_model", lambda: model)
        return model
    return use


def test_mask_stops_at_real_tip(fake):
    fake(_FakeSam())
    pm = polesam.sam_pole_mask(_image(), _pole(), BODY_H, [tuple(HAND)])
    assert pm is not None
    x1, y1, x2, y2 = pm.box
    ys, xs = np.nonzero(pm.mask)
    # không có pixel nào quá đầu gậy thật (trục dò kéo quá 0.3 chiều dài)
    assert (ys + y1).max() <= END[1] + 4
    assert (ys + y1).min() <= TOP[1] + 30


def test_blob_rejected(fake):
    fake(_FakeSam(blob=True))
    assert polesam.sam_pole_mask(_image(), _pole(), BODY_H, [tuple(HAND)]) is None


def test_points_snapped_onto_offset_pole(fake):
    sam = fake(_FakeSam())
    pole = _pole(shift=8.0)                       # trục dò lệch 8 px khỏi gậy
    pm = polesam.sam_pole_mask(_image(), pole, BODY_H, [tuple(HAND)])
    assert pm is not None
    pts, lbl = sam.calls[0]
    img = _image()
    x1, y1 = pm.box[:2]
    for x, y in pts[lbl == 1]:
        assert img[int(round(y + y1)), int(round(x + x1))].mean() < 120   # điểm dương nằm trên gậy


def test_no_model_falls_back(fake):
    fake(None)
    assert polesam.sam_pole_mask(_image(), _pole(), BODY_H, [tuple(HAND)]) is None


def test_model_error_falls_back(fake):
    def boom(*a, **k):
        raise RuntimeError("x")
    fake(boom)
    assert polesam.sam_pole_mask(_image(), _pole(), BODY_H, [tuple(HAND)]) is None


def test_overlay_uses_sam_and_cuts_overshoot(fake, monkeypatch):
    fake(_FakeSam())
    monkeypatch.setattr(C, "POLE_SAM", True)
    img = _image().astype(np.float32) / 255.0
    alpha = np.zeros((H, W), np.float32)
    cut = Cutout(image=img.copy(), alpha=alpha, foreground=img.copy(), background=img.copy(),
                 trimap=np.zeros((H, W), np.uint8))
    p = Person(idx=0, box=(300, 150, 600, 900), conf=0.9, mask=None)
    p.hands = [tuple(HAND)]
    p.poles = [_pole()]
    scene = Scene(rgb=_image(), persons=[p])
    out = thin.overlay_poles(cut, scene, p)
    mid = (TOP + 0.5 * (END - TOP)).astype(int)
    over = (END + 0.6 * (OVER - END)).astype(int)
    assert out.alpha[mid[1] - 3:mid[1] + 4, mid[0] - 3:mid[0] + 4].max() > 0.8
    assert out.alpha[over[1] - 3:over[1] + 4, over[0] - 3:over[0] + 4].max() < 0.1
