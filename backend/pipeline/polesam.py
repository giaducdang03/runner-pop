"""Mask gậy bằng MobileSAM: nhắc SAM bằng điểm dọc trục gậy đã dò được (poles.py).

Bộ dò gậy chỉ cho trục thẳng + bề rộng đều; gậy thật có cán, quai, núm, nhiều đoạn màu, gậy thứ hai
cầm song song. SAM tách theo hình dạng thật nên lấy được cán phía trên tay, dừng đúng đầu gậy (dù trục dò
kéo quá vào đá / cỏ) và bắt cả gậy song song.

- Điểm dương chỉ đặt ở đoạn gần chỗ nắm (trục dò được chắc nhất ở đó; đầu xa có thể chạy quá đầu gậy
  thật, điểm rơi vào đá làm SAM tô cả mảng đá). Điểm âm hai bên trục.
- Sau SAM: bỏ pixel cách đường trục > POLE_SAM_BAND * h, giữ thành phần chạm điểm dương; diện tích quá
  lớn (tràn) hoặc quá nhỏ (SAM trượt) thì trả None để dùng cách cũ (polemask / dải gậy).
Không bao giờ ném lỗi; model nạp lười, dùng chung, khoá khi suy luận."""
from __future__ import annotations

import logging
import threading
from typing import Optional, Sequence

import cv2
import numpy as np

from . import config as C
from .polemask import PoleMask
from .poles import Pole

log = logging.getLogger(__name__)

_lock = threading.Lock()
_model = None
_model_err: Optional[str] = None


def _get_model():
    """MobileSAM (ultralytics), nạp một lần; lần đầu tự tải trọng số vào POLE_SAM_WEIGHTS.
    Nạp lỗi thì nhớ lỗi, không thử lại (trả None)."""
    global _model, _model_err
    if _model is None and _model_err is None:
        try:
            from pathlib import Path
            from ultralytics import SAM
            Path(C.POLE_SAM_WEIGHTS).parent.mkdir(parents=True, exist_ok=True)
            _model = SAM(C.POLE_SAM_WEIGHTS)
        except Exception as e:  # noqa: BLE001
            _model_err = str(e)
            log.warning("không nạp được MobileSAM, dùng cách tách gậy cũ: %s", e)
    return _model


def _to_u8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    return np.clip(np.asarray(img, np.float32) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _snap(lab: np.ndarray, pos: np.ndarray, n: np.ndarray, hw: float, h: float) -> np.ndarray:
    """Dời mỗi điểm dương theo pháp tuyến (±max(6, POLE_SAM_SNAP * h) px) về chỗ có mặt cắt dạng thanh rõ
    nhất: điểm = |tâm − TB hai bên| − |trái − phải| (Lab 8 bit, hai bên cách tâm hw + 4 px). Trục dò có thể
    lệch khỏi gậy vài chục px (gậy sát mép ảnh, ảnh nghiêng); điểm dương rơi vào nền làm SAM tô cả mảng nền.
    Điểm không có chỗ nào đạt POLE_SAM_SNAP_MIN thì bỏ."""
    H, W = lab.shape[:2]
    R = max(6.0, C.POLE_SAM_SNAP * h)
    offs = np.arange(-R, R + 0.5, 1.0)
    side = (hw + 4.0) * n

    def at(pts):
        x = np.clip(np.round(pts[..., 0]), 0, W - 1).astype(int)
        y = np.clip(np.round(pts[..., 1]), 0, H - 1).astype(int)
        return lab[y, x]

    grid = pos[:, None] + offs[None, :, None] * n                  # N x O x 2
    c, left, right = at(grid), at(grid - side), at(grid + side)
    score = (np.linalg.norm(c - 0.5 * (left + right), axis=-1)
             - np.linalg.norm(left - right, axis=-1))
    score -= 0.05 * np.abs(offs)[None]                              # hoà thì gần trục dò hơn thắng
    k = np.argmax(score, axis=1)
    ok = score[np.arange(len(pos)), k] >= C.POLE_SAM_SNAP_MIN
    return grid[np.arange(len(pos)), k][ok]


def _prompts(pole: Pole, h: float, hands: Sequence, lab: Optional[np.ndarray] = None):
    """(điểm dương N x 2, điểm âm M x 2) theo toạ độ ảnh. Chỗ nắm = hình chiếu lên trục của tay gần trục
    nhất (cách trục <= 0.05h), không có thì đầu trên. Điểm dương cách chỗ nắm POLE_SAM_POS * h về mỗi phía
    (bỏ nắm tay), không quá nửa đoạn tới đầu gậy phía đó; phía ngắn hơn 0.1h thì bỏ. Có `lab` (ảnh Lab đã
    blur) thì dời điểm dương về tâm gậy (_snap)."""
    a = np.asarray(pole.top, np.float64)
    b = np.asarray(pole.tip, np.float64)
    L = float(np.linalg.norm(b - a))
    d = (b - a) / L
    n = np.array([-d[1], d[0]])
    g, best = 0.0, None
    for hand in hands or ():
        q = np.asarray(hand, np.float64) - a
        t, s = float(q @ d), abs(float(q @ n))
        if s <= 0.05 * h and -0.05 * h <= t <= L + 0.05 * h and (best is None or s < best):
            g, best = float(np.clip(t, 0.0, L)), s
    pos = []
    for sgn, room in ((1.0, L - g), (-1.0, g)):
        if room < 0.1 * h:
            continue
        for r in C.POLE_SAM_POS:
            rr = r * h
            if rr <= 0.5 * room:
                pos.append(a + (g + sgn * rr) * d)
    if not pos:                                      # gậy ngắn: rải đều trên đoạn giữa
        pos = [a + t * L * d for t in (0.3, 0.5, 0.7)]
    pos = np.array(pos)
    if lab is not None:
        pos = _snap(lab, pos, n, float(pole.half_w), h)
    off = (pole.half_w + C.POLE_SAM_NEG * h) * n
    neg = np.concatenate([pos + off, pos - off])
    return pos, neg


def _axis_dist(xs: np.ndarray, ys: np.ndarray, pole: Pole) -> np.ndarray:
    a = np.asarray(pole.top, np.float64)
    d = np.asarray(pole.tip, np.float64) - a
    d /= max(float(np.linalg.norm(d)), 1e-9)
    return np.abs((xs - a[0]) * d[1] - (ys - a[1]) * d[0])


def sam_pole_mask(image: np.ndarray, pole: Pole, body_h: float, hands: Sequence = (),
                  dbg: Optional[dict] = None) -> Optional[PoleMask]:
    """Mask gậy (PoleMask trong crop) từ MobileSAM hoặc None = dùng cách cũ. `image`: ảnh đầy đủ RGB
    (uint8 hoặc float 0..1), `pole`/`hands`/`body_h` cùng hệ toạ độ ảnh đó."""
    try:
        return _sam(image, pole, float(body_h), hands, dbg if dbg is not None else {})
    except Exception as e:  # noqa: BLE001
        log.warning("MobileSAM tách gậy thất bại, dùng cách cũ: %s", e)
        return None


def _sam(image, pole: Pole, h: float, hands, dbg: dict) -> Optional[PoleMask]:
    H, W = image.shape[:2]
    a, b = np.asarray(pole.top), np.asarray(pole.tip)
    L = float(np.linalg.norm(b - a))
    if L < 4 or h <= 0:
        return None
    model = _get_model()
    if model is None:
        return None
    u8 = _to_u8(image)
    lab = cv2.cvtColor(cv2.GaussianBlur(u8, (0, 0), 1.0), cv2.COLOR_RGB2LAB).astype(np.float32)
    pos, neg = _prompts(pole, h, hands, lab)
    dbg.update(pos=pos, neg=neg)
    if len(pos) < 2:
        log.info("MobileSAM gậy: < 2 điểm dương nằm trên gậy")
        return None
    pad = max(C.POLE_SAM_PAD * L, 0.05 * h)
    x1 = int(max(0, np.floor(min(a[0], b[0]) - pad)))
    y1 = int(max(0, np.floor(min(a[1], b[1]) - pad)))
    x2 = int(min(W, np.ceil(max(a[0], b[0]) + pad)))
    y2 = int(min(H, np.ceil(max(a[1], b[1]) + pad)))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    off = np.array([x1, y1], np.float64)
    pts = np.concatenate([pos, neg]) - off
    inside = (pts[:, 0] >= 0) & (pts[:, 0] < x2 - x1) & (pts[:, 1] >= 0) & (pts[:, 1] < y2 - y1)
    lbl = np.r_[np.ones(len(pos), int), np.zeros(len(neg), int)]
    pts, lbl = pts[inside], lbl[inside]
    if not (lbl == 1).any():
        return None
    crop = cv2.cvtColor(np.ascontiguousarray(u8[y1:y2, x1:x2]), cv2.COLOR_RGB2BGR)
    with _lock:
        r = model(crop, points=[pts.tolist()], labels=[lbl.tolist()], verbose=False)
    if not r or r[0].masks is None or len(r[0].masks.data) == 0:
        log.info("MobileSAM gậy: mask rỗng")
        return None
    m = r[0].masks.data[0].cpu().numpy() > 0.5
    if m.shape != (y2 - y1, x2 - x1):
        m = cv2.resize(m.astype(np.uint8), (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST) > 0
    dbg["raw"] = m.copy()

    # bỏ phần xa trục (tràn sang vật khác), giữ thành phần chạm điểm dương
    ys, xs = np.nonzero(m)
    far = _axis_dist(xs + x1, ys + y1, pole) > C.POLE_SAM_BAND * h
    m[ys[far], xs[far]] = False
    k_n, lbl_img = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
    seeds = np.zeros(m.shape, np.uint8)
    r_seed = max(2, int(round(pole.half_w + 2)))
    for px, py in (pos - off).round().astype(int):
        if 0 <= px < m.shape[1] and 0 <= py < m.shape[0]:
            cv2.circle(seeds, (int(px), int(py)), r_seed, 1, -1)
    hit = np.unique(lbl_img[(seeds > 0) & m])
    m = np.isin(lbl_img, hit[hit > 0])
    area = float(m.sum())
    strip_area = max(1.0, 2.0 * (pole.half_w + 1.0) * L)
    dbg.update(mask=m, area=area, strip_area=strip_area)
    log.info("MobileSAM gậy: raw %d px, giữ %d px, dải %d px, giới hạn %d px",
             int(dbg["raw"].sum()), int(area), int(strip_area), int(C.POLE_SAM_AREA_H * h * h))
    if area < C.POLE_SAM_MIN_AREA * strip_area:
        return None                                  # SAM trượt khỏi gậy
    if area > C.POLE_SAM_AREA_H * h * h:
        return None                                  # nghi tràn sang nền / người
    # độ mảnh: bề rộng trung bình = diện tích / chiều dài dọc trục của mask (mảng nền thì rộng)
    ys, xs = np.nonzero(m)
    d = (b - a) / L
    t = (xs + x1 - a[0]) * d[0] + (ys + y1 - a[1]) * d[1]
    width = area / max(1.0, float(t.max() - t.min()))
    dbg["width"] = width
    if width > max(C.POLE_SAM_WIDTH_MUL * (2 * pole.half_w + 2), C.POLE_SAM_WIDTH_H * h):
        log.info("MobileSAM gậy: mask quá rộng (TB %.1f px)", width)
        return None
    return PoleMask(box=(x1, y1, x2, y2), mask=m, ratio=area / strip_area)
