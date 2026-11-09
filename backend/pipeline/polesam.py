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

import contextlib
import logging
import threading
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from . import accel
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


_ENC_NBYTES = 64 * 1024 * 1024        # ước lượng bộ nhớ encoder đã compile, cho ngân sách accel.compiled


def _ov_encoder(device: "accel.Device"):
    """Encoder TinyViT của MobileSAM đã compile trên GPU OpenVINO (suy luận FP32). Lần đầu: convert từ PyTorch rồi lưu
    IR vào <ACCEL_CACHE_DIR>/<tên trọng số>_encoder_openvino/ (đổi trọng số thì IR riêng); các lần sau đọc IR,
    OpenVINO cache blob đã compile nên không biên dịch lại. Build lỗi -> RuntimeError và nhớ lỗi
    ACCEL_EXPORT_RETRY_S giây để không thử lại ở mỗi gậy (nơi gọi rơi về CPU)."""
    key = f"mobilesam_encoder:{device.backend}:{device.index}"
    if accel.failed_recently(key):
        raise RuntimeError("encoder MobileSAM trên GPU vừa build lỗi, chưa thử lại")

    def build():
        try:
            import openvino as ov
            import torch
            core = ov.Core()
            d = Path(C.ACCEL_CACHE_DIR) / f"{Path(C.POLE_SAM_WEIGHTS).stem}_encoder_openvino"
            xml = d / "encoder.xml"
            if xml.exists():
                m = core.read_model(str(xml))
            else:
                net = _get_model().model
                with torch.no_grad():
                    m = ov.convert_model(net.image_encoder, example_input=torch.zeros(1, 3, 1024, 1024))
                d.mkdir(parents=True, exist_ok=True)
                ov.save_model(m, str(xml), compress_to_fp16=True)
            # luôn FP32 khi suy luận (không theo device.dtype): FP16 làm mask gậy lệch (IoU CPU/GPU TB 0.92,
            # có gậy bị từ chối nhầm), FP32 khớp CPU (IoU TB 0.999); trọng số IR lưu FP16 cho gọn, không ảnh hưởng
            cfg = {"CACHE_DIR": C.ACCEL_CACHE_DIR, "INFERENCE_PRECISION_HINT": "f32"}
            cm = core.compile_model(m, device.ov_device_name, cfg)
        except Exception:
            accel.remember_failure(key)
            raise
        accel.forget_failures(key)
        return cm
    return accel.compiled(key, build, _ENC_NBYTES)


@contextlib.contextmanager
def _encoder_on_gpu(model, cm):
    """Tạm thay image_encoder của SAM bằng encoder đã compile (nhận / trả tensor), khôi phục khi xong.
    Gọi dưới _lock nên không luồng nào thấy encoder đang bị thay."""
    import torch

    class _OvEncoder(torch.nn.Module):
        def forward(self, x):
            out = cm(x.detach().cpu().numpy())
            return torch.from_numpy(np.asarray(out[cm.output(0)]))

    net = model.model
    orig = net.image_encoder
    net.image_encoder = _OvEncoder()
    try:
        yield
    finally:
        net.image_encoder = orig


def _infer(model, crop_bgr: np.ndarray, pts, lbl, device=None, stage=None):
    """Chạy SAM trên `crop_bgr` với điểm `pts` / nhãn `lbl` (list). Encoder chạy GPU OpenVINO khi `device` là GPU
    OpenVINO và POLE_SAM_GPU bật (lỗi -> CPU, ghi chú vào `stage`). Trả (kết quả ultralytics, encoder_chạy_GPU)."""
    def run():
        return model(crop_bgr, points=[pts], labels=[lbl], verbose=False)

    def on_cpu():
        with _lock:
            return run()

    def on_gpu():
        cm = _ov_encoder(device)
        with _lock, _encoder_on_gpu(model, cm):
            return run()

    if C.POLE_SAM_GPU and device is not None and device.kind == "gpu" and device.backend == "openvino":
        r, used = accel.with_fallback(device, on_gpu, on_cpu, stage=stage)
        return r, used.kind == "gpu"
    return on_cpu(), False


def warmup(device=None) -> bool:
    """Nạp MobileSAM (+ convert / compile encoder GPU nếu `device` là GPU OpenVINO) và chạy thử một lượt trên ảnh
    giả, để request thật không phải chờ nạp / biên dịch (lần suy luận GPU đầu tiên còn chậm hơn các lần sau).
    Trả True nếu encoder chạy trên GPU. Raise nếu không nạp được model."""
    model = _get_model()
    if model is None:
        raise RuntimeError(_model_err or "không nạp được MobileSAM")
    crop = np.full((256, 256, 3), 200, np.uint8)
    cv2.line(crop, (170, 20), (90, 240), (20, 20, 25), 5, cv2.LINE_AA)
    _, gpu = _infer(model, crop, [[130, 130], [113, 175], [160, 130]], [1, 1, 0], device)
    return gpu


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
    blur) thì dời điểm dương về tâm gậy (_snap). Thêm một điểm trên cán / quai phía trên chỗ nắm
    (POLE_SAM_TOP * khoảng cách từ đầu trên tới chỗ nắm, tính từ đầu trên): cán rộng hơn thân gậy nên
    không dời, và đoạn từ tay lên đầu gậy thường quá ngắn cho khoảng cách POLE_SAM_POS."""
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
    pos = np.array(pos).reshape(-1, 2)
    if lab is not None:
        pos = _snap(lab, pos, n, float(pole.half_w), h)
    if C.POLE_SAM_TOP > 0 and g >= 0.04 * h:
        pos = np.concatenate([pos.reshape(-1, 2), (a + C.POLE_SAM_TOP * g * d)[None]])
    off = (pole.half_w + C.POLE_SAM_NEG * h) * n
    neg = np.concatenate([pos + off, pos - off])
    return pos, neg


def _axis_dist(xs: np.ndarray, ys: np.ndarray, pole: Pole) -> np.ndarray:
    a = np.asarray(pole.top, np.float64)
    d = np.asarray(pole.tip, np.float64) - a
    d /= max(float(np.linalg.norm(d)), 1e-9)
    return np.abs((xs - a[0]) * d[1] - (ys - a[1]) * d[0])


def sam_pole_mask(image: np.ndarray, pole: Pole, body_h: float, hands: Sequence = (),
                  dbg: Optional[dict] = None, device: Optional["accel.Device"] = None,
                  stage=None) -> Optional[PoleMask]:
    """Mask gậy (PoleMask trong crop) từ MobileSAM hoặc None = dùng cách cũ. `image`: ảnh đầy đủ RGB
    (uint8 hoặc float 0..1), `pole`/`hands`/`body_h` cùng hệ toạ độ ảnh đó. `device`: GPU OpenVINO thì encoder
    chạy trên GPU (lỗi -> CPU, ghi chú vào `stage`); `dbg["gpu"]` = True nếu encoder lần nào chạy trên GPU."""
    try:
        return _sam(image, pole, float(body_h), hands, dbg if dbg is not None else {}, device, stage)
    except Exception as e:  # noqa: BLE001
        log.warning("MobileSAM tách gậy thất bại, dùng cách cũ: %s", e)
        return None


def _segment(u8, lab, pole: Pole, h: float, hands, extra, device, stage, dbg: dict):
    """Một lượt SAM trên vùng cắt quanh `pole` (cộng điểm dương `extra` nếu có, kèm điểm âm hai bên chúng).
    Trả (mask bool trong crop, (x1, y1, x2, y2), điểm dương) đã bỏ phần xa trục và chỉ giữ thành phần chạm
    điểm dương; None nếu không chạy được / mask rỗng. Chưa kiểm tra diện tích / độ rộng (xem _check)."""
    H, W = u8.shape[:2]
    a, b = np.asarray(pole.top), np.asarray(pole.tip)
    L = float(np.linalg.norm(b - a))
    model = _get_model()
    if model is None:
        return None
    pos, neg = _prompts(pole, h, hands, lab)
    if extra is not None and len(extra):
        d = (b - a) / L
        off_n = (pole.half_w + C.POLE_SAM_NEG * h) * np.array([-d[1], d[0]])
        pos = np.concatenate([pos, extra])
        neg = np.concatenate([neg, extra + off_n, extra - off_n])
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
    r, gpu = _infer(model, crop, pts.tolist(), lbl.tolist(), device, stage)
    dbg["gpu"] = bool(dbg.get("gpu")) or gpu
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
    _, lbl_img = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
    seeds = np.zeros(m.shape, np.uint8)
    r_seed = max(2, int(round(pole.half_w + 2)))
    for px, py in (pos - off).round().astype(int):
        if 0 <= px < m.shape[1] and 0 <= py < m.shape[0]:
            cv2.circle(seeds, (int(px), int(py)), r_seed, 1, -1)
    hit = np.unique(lbl_img[(seeds > 0) & m])
    m = np.isin(lbl_img, hit[hit > 0])
    return m, (x1, y1, x2, y2), pos


def _check(m: np.ndarray, box, pole: Pole, h: float, dbg: dict) -> Optional[PoleMask]:
    """Mask đạt khi đủ lớn so với dải gậy, không tràn (diện tích) và đủ mảnh (bề rộng TB); không thì None."""
    x1, y1 = box[0], box[1]
    a, b = np.asarray(pole.top), np.asarray(pole.tip)
    L = float(np.linalg.norm(b - a))
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
    return PoleMask(box=tuple(box), mask=m, ratio=area / strip_area)


def _extend(u8, lab, pole: Pole, h: float, hands, pm: PoleMask, device, stage, dbg: dict) -> PoleMask:
    """Nối / kéo dài gậy dọc trục. Hai ca: (A) mask dừng TRƯỚC đầu gậy dò được (đoạn mờ, gậy xám trên nền xám,
    bị che) trong khi bộ dò đã xác nhận gậy tới đó; (B) mask chạm mép vùng cắt quá đầu gậy (gậy còn tiếp: bộ dò
    dừng sớm vì đoạn dưới bị lá che). Mask dừng tự nhiên trong khoảng [đầu gậy, mép vùng cắt) thì giữ nguyên.
    Đặt điểm dương dọc trục từ chỗ mask dừng tới POLE_SAM_EXT_BEYOND * h quá đầu gậy (cách nhau POLE_SAM_EXT_STEP * h,
    nằm trong ảnh), chạy SAM lần hai trên vùng cắt dài hơn. Chỉ nhận đoạn thêm là đoạn đầu liền mạch và mảnh
    (bề rộng TB mỗi ô <= max(3 * bề rộng dải, POLE_SAM_EXT_WIDTH_H * h)), dài >= POLE_SAM_EXT_MIN * h và phủ
    >= POLE_SAM_EXT_COVER khoảng đã đặt điểm: ca gậy dò quá đầu gậy thật (điểm rơi vào đá / cỏ) không cho
    đường mảnh liền mạch nên bị từ chối. Phần thân giữ nguyên từ lượt một. Không đạt thì trả pm cũ."""
    H, W = u8.shape[:2]
    a, b = np.asarray(pole.top), np.asarray(pole.tip)
    L = float(np.linalg.norm(b - a))
    d = (b - a) / L
    bx1, by1 = pm.box[0], pm.box[1]
    ys, xs = np.nonzero(pm.mask)
    if ys.size == 0:
        return pm
    t_start = float((((xs + bx1 - a[0]) * d[0] + (ys + by1 - a[1]) * d[1])).max())
    pad = max(C.POLE_SAM_PAD * L, 0.05 * h)
    tol = C.POLE_SAM_EXT_TOL * h
    if L - tol <= t_start < L + pad - tol:
        return pm                                    # mask dừng tự nhiên quanh đầu gậy: không có gì để nối / kéo dài
    step = max(2.0, C.POLE_SAM_EXT_STEP * h)
    ts = np.arange(t_start + 0.5 * step, L + C.POLE_SAM_EXT_BEYOND * h + 1e-6, step)
    pts = [a + t * d for t in ts]
    keep_ts = [t for t, p in zip(ts, pts) if 3 <= p[0] < W - 3 and 3 <= p[1] < H - 3]
    pts = [a + t * d for t in keep_ts]
    if not pts:
        return pm
    t_last = float(keep_ts[-1])
    pole2 = replace(pole, tip=(float(pts[-1][0]), float(pts[-1][1])))
    if t_last <= L:                                  # chỉ nối trong đoạn dò được: vùng cắt vẫn phải phủ tới đầu gậy dò được
        pole2 = pole
    ed: dict = {}
    seg = _segment(u8, lab, pole2, h, hands, np.array(pts), device, stage, ed)
    dbg["gpu"] = bool(dbg.get("gpu")) or bool(ed.get("gpu"))
    if seg is None:
        return pm
    m2, box2, _ = seg
    ys2, xs2 = np.nonzero(m2)
    if ys2.size == 0:
        return pm
    t2 = (xs2 + box2[0] - a[0]) * d[0] + (ys2 + box2[1] - a[1]) * d[1]
    t0 = t_start - 0.01 * h
    sel = t2 > t0
    if not sel.any():
        return pm
    binw = max(2.0, 0.003 * h)
    cnt = np.bincount(((t2[sel] - t0) // binw).astype(int))
    wmax = max(C.POLE_SAM_WIDTH_MUL * (2 * pole.half_w + 2), C.POLE_SAM_EXT_WIDTH_H * h)
    good = (cnt > 0) & (cnt / binw <= wmax)
    n_ok = len(good) if good.all() else int(np.argmin(good))   # đoạn đầu liền mạch và mảnh
    t_end = t0 + n_ok * binw
    got, want = t_end - t_start, t_last - t_start
    if got < max(C.POLE_SAM_EXT_MIN * h, C.POLE_SAM_EXT_COVER * want):
        log.info("MobileSAM gậy: không nối / kéo dài được (mảnh liền mạch %.0f / %.0f px)", got, want)
        return pm
    keep = sel & (t2 <= t_end)
    ux1, uy1 = min(pm.box[0], box2[0]), min(pm.box[1], box2[1])
    ux2, uy2 = max(pm.box[2], box2[2]), max(pm.box[3], box2[3])
    u = np.zeros((uy2 - uy1, ux2 - ux1), bool)
    u[pm.box[1] - uy1:pm.box[3] - uy1, pm.box[0] - ux1:pm.box[2] - ux1] = pm.mask
    u[ys2[keep] + box2[1] - uy1, xs2[keep] + box2[0] - ux1] = True
    cd: dict = {"raw": dbg.get("raw", u)}
    out = _check(u, (ux1, uy1, ux2, uy2), pole, h, cd)
    if out is None:
        return pm
    dbg["ext"] = float(got)
    dbg["mask"] = u
    log.info("MobileSAM gậy: nối / kéo dài thêm %.0f px (yêu cầu %.0f px)", got, want)
    return out


def _sam(image, pole: Pole, h: float, hands, dbg: dict, device=None, stage=None) -> Optional[PoleMask]:
    a, b = np.asarray(pole.top), np.asarray(pole.tip)
    L = float(np.linalg.norm(b - a))
    if L < 4 or h <= 0:
        return None
    u8 = _to_u8(image)
    lab = cv2.cvtColor(cv2.GaussianBlur(u8, (0, 0), 1.0), cv2.COLOR_RGB2LAB).astype(np.float32)
    dbg["gpu"] = False
    seg = _segment(u8, lab, pole, h, hands, None, device, stage, dbg)
    if seg is None:
        return None
    m, box, _ = seg
    pm = _check(m, box, pole, h, dbg)
    if pm is None or C.POLE_SAM_EXT_STEP <= 0:
        return pm
    try:
        return _extend(u8, lab, pole, h, hands, pm, device, stage, dbg)
    except Exception as e:  # noqa: BLE001  (lỗi kéo dài thì giữ mask lượt một)
        log.warning("MobileSAM kéo dài đuôi gậy thất bại, giữ mask lượt một: %s", e)
        return pm
