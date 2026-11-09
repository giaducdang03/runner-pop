"""Tách gậy bằng MobileSAM (polesam.py) với model giả: không tải trọng số."""
import cv2
import numpy as np
import pytest

from backend.pipeline import accel
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
        import types
        self.blob, self.calls = blob, []
        self.model = types.SimpleNamespace(image_encoder="cpu-encoder")
        self.encoder_seen = []

    def __call__(self, crop, points, labels, verbose=False):
        self.calls.append((np.asarray(points[0]), np.asarray(labels[0])))
        self.encoder_seen.append(self.model.image_encoder)
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


GPU = accel.Device(kind="gpu", backend="openvino", name="fake", runtime="openvino", dtype="fp16")


def test_gpu_swaps_encoder_and_restores(fake, monkeypatch):
    sam = fake(_FakeSam())
    monkeypatch.setattr(polesam, "_ov_encoder", lambda dev: object())
    dbg = {}
    pm = polesam.sam_pole_mask(_image(), _pole(), BODY_H, [tuple(HAND)], dbg, GPU)
    assert pm is not None and dbg["gpu"] is True
    assert sam.encoder_seen[0] != "cpu-encoder"          # trong lúc chạy: encoder GPU
    assert sam.model.image_encoder == "cpu-encoder"      # xong: khôi phục encoder CPU


def test_gpu_failure_falls_back_to_cpu_with_note(fake, monkeypatch):
    sam = fake(_FakeSam())

    def boom(dev):
        raise RuntimeError("compile lỗi")
    monkeypatch.setattr(polesam, "_ov_encoder", boom)

    class Stage:
        note = "1 gậy"
    st, dbg = Stage(), {}
    pm = polesam.sam_pole_mask(_image(), _pole(), BODY_H, [tuple(HAND)], dbg, GPU, st)
    assert pm is not None and dbg["gpu"] is False
    assert sam.encoder_seen and all(e == "cpu-encoder" for e in sam.encoder_seen)   # mọi lượt đều chạy CPU
    assert "GPU lỗi" in st.note


def test_gpu_disabled_by_config(fake, monkeypatch):
    sam = fake(_FakeSam())
    monkeypatch.setattr(C, "POLE_SAM_GPU", False)
    monkeypatch.setattr(polesam, "_ov_encoder", lambda dev: pytest.fail("không được dùng GPU"))
    dbg = {}
    assert polesam.sam_pole_mask(_image(), _pole(), BODY_H, [tuple(HAND)], dbg, GPU) is not None
    assert dbg["gpu"] is False


def test_warmup_runs_model_once_and_reports_gpu(fake, monkeypatch):
    sam = fake(_FakeSam())
    monkeypatch.setattr(polesam, "_ov_encoder", lambda dev: object())
    assert polesam.warmup(GPU) is True
    assert len(sam.calls) == 1
    assert polesam.warmup(None) is False


def test_warmup_raises_when_model_missing(fake):
    fake(None)
    with pytest.raises(RuntimeError):
        polesam.warmup(None)


def _long_image(end):
    img = np.full((H, W, 3), (200, 210, 225), np.uint8)
    cv2.line(img, tuple(TOP.astype(int)), tuple(np.asarray(end).astype(int)), (20, 20, 25), 5, cv2.LINE_AA)
    return img


def test_top_handle_point_added_above_grip():
    pole = _pole(tip=END)
    hand = TOP + 0.4 * (END - TOP)                       # tay nắm giữa gậy, cán thò lên trên
    pos, neg = polesam._prompts(pole, BODY_H, [tuple(hand)], None)
    g = float(np.linalg.norm(hand - TOP))
    d = (END - TOP) / np.linalg.norm(END - TOP)
    t = (pos - TOP) @ d
    assert np.isclose(t.min(), C.POLE_SAM_TOP * g, atol=1.0)       # điểm trên cán, tính từ đầu trên
    assert len(neg) == 2 * len(pos)


def test_tail_extended_when_pole_continues_past_detected_tip(fake):
    sam = fake(_FakeSam())
    L_true = np.linalg.norm(END - TOP)
    d = (END - TOP) / L_true
    det_tip = TOP + 0.7 * (END - TOP)                    # bộ dò dừng sớm: gậy thật còn tiếp tới END
    pole = _pole(tip=det_tip)
    dbg = {}
    pm = polesam.sam_pole_mask(_long_image(END), pole, BODY_H, [tuple(HAND)], dbg)
    assert pm is not None and len(sam.calls) == 2 and dbg["ext"] > 0
    x1, y1, x2, y2 = pm.box
    ys, xs = np.nonzero(pm.mask)
    t = (xs + x1 - TOP[0]) * d[0] + (ys + y1 - TOP[1]) * d[1]
    assert t.max() > 0.7 * L_true + 0.15 * np.linalg.norm(det_tip - TOP) + 10     # quá cả vùng cắt lượt một


def test_no_extension_when_mask_ends_at_detected_tip(fake):
    sam = fake(_FakeSam())
    pole = _pole(tip=END)                                # đầu gậy dò được trùng đầu gậy thật
    pm = polesam.sam_pole_mask(_image(), pole, BODY_H, [tuple(HAND)])
    assert pm is not None and len(sam.calls) == 1        # không tốn lượt SAM thứ hai


def test_extension_disabled_by_config(fake, monkeypatch):
    sam = fake(_FakeSam())
    monkeypatch.setattr(C, "POLE_SAM_EXT_STEP", 0.0)
    det_tip = TOP + 0.7 * (END - TOP)
    polesam.sam_pole_mask(_long_image(END), _pole(tip=det_tip), BODY_H, [tuple(HAND)])
    assert len(sam.calls) == 1
