"""Dò gậy (gậy leo núi, gậy tre...) từ nắm tay: quét tia đi xuống từ mỗi tay, kiểm tra mặt cắt
ngang dạng "thanh" (tâm khác hẳn hai bên, hai bên giống nhau) dọc theo tia."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import config as C
from .schema import Person

if TYPE_CHECKING:
    from .roi import Roi

log = logging.getLogger(__name__)

_ANGLE_STEP = 0.5        # độ
_OX_STEP = 6.0           # px, bước dịch tia theo pháp tuyến
_R_STEP = 3.0            # px, bước lấy mẫu dọc tia
_MIN_SAMPLES = 30        # tia có ít mẫu trong ảnh hơn mức này thì bỏ
_SIDE_PAD = 4.0          # px, mẫu hai bên cách mép gậy thêm đoạn này
_EXCLUDE_DEG = 3.0       # người thua bỏ các góc trong ±3° của gậy đã bị lấy
_OWNER_DISK = 0.04       # bán kính đĩa xét mask quanh chỗ nắm (* h)
_SAME_POLE_PAD = 6.0     # hai trục cách nhau < 2*hw + 6 px = cùng một gậy
_BLUR_SIGMA = 1.2
_LAT_MIN = 8.0           # px, khi dò đầu gậy được lệch ngang khỏi tia tối đa max(8, 3*hw)
_LAT_HW = 3.0


@dataclass
class Pole:
    top: Tuple[float, float]     # đầu trên (gần/trên nắm tay), toạ độ ảnh làm việc
    tip: Tuple[float, float]     # đầu gậy chạm đất
    half_w: float                # nửa bề rộng gậy (px)
    score: float                 # tỉ lệ mẫu có mặt cắt dạng thanh trên đoạn tia tìm kiếm
    owner: int                   # idx người cầm

    @property
    def angle(self) -> float:
        """Góc trục gậy (độ), 90 = thẳng xuống."""
        return math.degrees(math.atan2(self.tip[1] - self.top[1], self.tip[0] - self.top[0]))

    def to_work(self, roi: "Roi") -> "Pole":
        """Đổi sang hệ toạ độ ROI ở độ phân giải làm việc."""
        def conv(pt):
            return ((pt[0] - roi.x1) * roi.scale, (pt[1] - roi.y1) * roi.scale)
        return replace(self, top=conv(self.top), tip=conv(self.tip), half_w=self.half_w * roi.scale)

    def polygon(self) -> np.ndarray:
        """4 đỉnh (float32 4x2) của dải quanh trục top -> tip, rộng 2 * half_w + 2."""
        a = np.asarray(self.top, np.float32)
        b = np.asarray(self.tip, np.float32)
        d = b - a
        ln = float(np.linalg.norm(d))
        d = d / ln if ln > 0 else np.array([0.0, 1.0], np.float32)
        n = np.array([-d[1], d[0]], np.float32) * (self.half_w + 1.0)
        return np.stack([a - n, a + n, b + n, b - n]).astype(np.float32)


@dataclass
class _Scan:
    """Kết quả quét một tay theo từng góc."""
    person: Person
    hand: np.ndarray
    h: float
    angles: np.ndarray       # độ
    scores: np.ndarray       # điểm tốt nhất theo góc (-1 = không đủ mẫu)
    oxs: np.ndarray          # độ dịch tia tốt nhất theo góc
    hws: np.ndarray          # nửa bề rộng (px) có nhiều mẫu đạt nhất theo góc
    excluded: List[float]    # góc bị loại (độ), ±_EXCLUDE_DEG quanh mỗi góc
    lost_to: List["Pole"]    # gậy tay này đã thua; ứng viên mới trùng gậy đó cũng bị loại


@dataclass
class _Cand:
    scan: _Scan
    angle: float
    pole: Pole


class _Lab:
    """Ảnh Lab (chia 255) đã blur nhẹ, lấy mẫu theo điểm (cắt vào ảnh)."""

    def __init__(self, rgb: np.ndarray):
        blur = cv2.GaussianBlur(rgb, (0, 0), _BLUR_SIGMA)
        lab = cv2.cvtColor(blur, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
        self.h, self.w = lab.shape[:2]
        self.flat = lab.reshape(-1, 3)

    def inside(self, pts: np.ndarray) -> np.ndarray:
        return (pts[..., 0] >= 0) & (pts[..., 0] < self.w) & (pts[..., 1] >= 0) & (pts[..., 1] < self.h)

    def sample(self, pts: np.ndarray) -> np.ndarray:
        x = np.clip(pts[..., 0], 0, self.w - 1).astype(np.int64)
        y = np.clip(pts[..., 1], 0, self.h - 1).astype(np.int64)
        return self.flat[y * self.w + x]


def _dir(angle: float) -> Tuple[np.ndarray, np.ndarray]:
    t = math.radians(angle)
    d = np.array([math.cos(t), math.sin(t)])
    return d, np.array([-d[1], d[0]])


def _support(lab: _Lab, pts: np.ndarray, n: np.ndarray, hws_px: np.ndarray) -> np.ndarray:
    """sup[hw, ...]: mẫu có mặt cắt dạng thanh với nửa bề rộng hw (tâm khác TB hai bên,
    hai bên giống nhau - loại viền của vùng lớn)."""
    c = lab.sample(pts)
    out = []
    for hw in hws_px:
        off = (hw + _SIDE_PAD) * n
        left = lab.sample(pts - off)
        right = lab.sample(pts + off)
        bar = np.linalg.norm(c - 0.5 * (left + right), axis=-1)
        side = np.linalg.norm(left - right, axis=-1)
        out.append((bar > C.POLE_BAR_CONTRAST) & (side < C.POLE_SIDE_RATIO * bar))
    return np.stack(out)


def _scan_hand(lab: _Lab, person: Person, hand) -> Optional[_Scan]:
    h = float(person.body_h)
    if h <= 0:
        return None
    hand = np.asarray(hand, np.float64)
    hws_px = np.asarray(C.POLE_HALF_W, np.float64) * h
    angles = np.arange(C.POLE_ANGLES[0], C.POLE_ANGLES[1] + _ANGLE_STEP / 2, _ANGLE_STEP)
    tol = C.POLE_HAND_TOL * h
    oxs = np.arange(-tol, tol + 1, _OX_STEP)
    rs = np.arange(C.POLE_SEARCH[0] * h, C.POLE_SEARCH[1] * h, _R_STEP)
    scores = np.full(len(angles), -1.0)
    best_ox = np.zeros(len(angles))
    best_hw = np.full(len(angles), hws_px[0])
    for i, a in enumerate(angles):
        d, n = _dir(a)
        origins = hand[None] + oxs[:, None] * n[None]                 # K x 2
        pts = origins[:, None] + rs[None, :, None] * d[None, None]    # K x R x 2
        ok = lab.inside(pts)
        cnt = ok.sum(axis=1)
        sup = _support(lab, pts, n, hws_px) & ok[None]                # HW x K x R
        any_sup = sup.any(axis=0).sum(axis=1)
        sc = np.where(cnt >= _MIN_SAMPLES, any_sup / np.maximum(cnt, 1), -1.0)
        k = int(np.argmax(sc))
        if sc[k] < 0:
            continue
        scores[i] = sc[k]
        best_ox[i] = oxs[k]
        best_hw[i] = hws_px[int(np.argmax(sup[:, k].sum(axis=1)))]
    return _Scan(person, hand, h, angles, scores, best_ox, best_hw, [], [])


def _trace(lab: _Lab, o: np.ndarray, d: np.ndarray, n: np.ndarray, rs: np.ndarray,
           hws_px: np.ndarray, gap: float, lat_max: float) -> Optional[np.ndarray]:
    """Đi dọc o + r*d theo thứ tự rs, bám theo gậy (lệch ngang tối đa ±1 px mỗi bước, trong
    ±lat_max). Trả điểm đạt cuối cùng trước khoảng hụt > gap hoặc trước mép ảnh (None nếu
    không có mẫu nào đạt)."""
    pts = o[None] + rs[:, None] * d[None]
    ok = lab.inside(pts)
    if not ok.all():
        stop = int(np.argmin(ok))
        rs = rs[:stop]
    if len(rs) == 0:
        return None
    lats = np.arange(-round(lat_max), round(lat_max) + 1, 1.0)
    grid = o[None, None] + rs[:, None, None] * d + lats[None, :, None] * n    # R x L x 2
    sup = _support(lab, grid, n, hws_px).any(axis=0) & lab.inside(grid)     # R x L
    mid = len(lats) // 2
    cur, last, last_r = mid, None, rs[0]
    for i in range(len(rs)):
        lo, hi = max(0, cur - 1), min(len(lats), cur + 2)
        hit = np.flatnonzero(sup[i, lo:hi])
        if len(hit):
            cur = lo + int(hit[np.argmin(np.abs(lo + hit - cur))])
            last, last_r = (i, cur), rs[i]
        elif abs(rs[i] - last_r) > gap:
            break
    if last is None:
        return None
    return grid[last[0], last[1]]


def _candidate(lab: _Lab, scan: _Scan) -> Optional[_Cand]:
    """Góc tốt nhất (bỏ các góc bị loại) của một tay; ứng viên trùng một gậy tay này đã thua
    thì loại thêm ±3° quanh góc đó và tìm tiếp."""
    while True:
        cand = _best(lab, scan)
        if cand is None or not any(_same_pole(cand.pole, q) for q in scan.lost_to):
            return cand
        scan.excluded.append(cand.angle)


def _best(lab: _Lab, scan: _Scan) -> Optional[_Cand]:
    """Góc tốt nhất (bỏ các góc bị loại) của một tay; truy đầu gậy và đầu trên."""
    sc = scan.scores.copy()
    for ex in scan.excluded:
        sc[np.abs(scan.angles - ex) <= _EXCLUDE_DEG] = -1.0
    i = int(np.argmax(sc))
    if sc[i] < C.POLE_MIN_SCORE:
        return None
    a, h = float(scan.angles[i]), scan.h
    d, n = _dir(a)
    o = scan.hand + scan.oxs[i] * n
    hws_px = np.asarray(C.POLE_HALF_W, np.float64) * h
    gap = C.POLE_GAP * h
    diag = math.hypot(lab.w, lab.h)
    lat = max(_LAT_MIN, _LAT_HW * float(scan.hws[i]))
    tip = _trace(lab, o, d, n, np.arange(C.POLE_SEARCH[0] * h, diag, _R_STEP), hws_px, gap, lat)
    top = _trace(lab, o, d, n, -np.arange(0.0, C.POLE_GRIP_UP * h + _R_STEP / 2, _R_STEP),
                 hws_px, gap, lat)
    tip = o + C.POLE_SEARCH[1] * h * d if tip is None else tip
    top = o if top is None else top
    pole = Pole(top=(float(top[0]), float(top[1])), tip=(float(tip[0]), float(tip[1])),
                half_w=float(scan.hws[i]), score=float(sc[i]), owner=scan.person.idx)
    return _Cand(scan, a, pole)


def _line_dist(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = b - a
    ln = float(np.linalg.norm(d))
    if ln == 0:
        return np.linalg.norm(pts - a, axis=1)
    return np.abs((pts[:, 0] - a[0]) * d[1] - (pts[:, 1] - a[1]) * d[0]) / ln


def _same_pole(p: Pole, q: Pole) -> bool:
    """Khoảng cách vuông góc trung bình giữa hai trục < 2*hw + 6 px."""
    a0, a1 = np.asarray(p.top), np.asarray(p.tip)
    b0, b1 = np.asarray(q.top), np.asarray(q.tip)
    t = np.linspace(0, 1, 21)[:, None]
    da = _line_dist(a0 + t * (a1 - a0), b0, b1).mean()
    db = _line_dist(b0 + t * (b1 - b0), a0, a1).mean()
    return 0.5 * (da + db) < 2 * max(p.half_w, q.half_w) + _SAME_POLE_PAD


def _nearest_on_axis(pole: Pole, pt: np.ndarray) -> Tuple[np.ndarray, float]:
    a, b = np.asarray(pole.top), np.asarray(pole.tip)
    d = b - a
    t = float(np.clip(np.dot(pt - a, d) / max(float(np.dot(d, d)), 1e-9), 0, 1))
    q = a + t * d
    return q, float(np.linalg.norm(pt - q))


def _grip_ratio(cand: _Cand, shape) -> Tuple[float, float]:
    """(tỉ lệ pixel mask của người trong đĩa 0.04h quanh điểm trên trục gần nắm tay nhất,
    khoảng cách tay -> trục)."""
    p = cand.scan.person
    q, dist = _nearest_on_axis(cand.pole, cand.scan.hand)
    r = max(1, int(round(_OWNER_DISK * cand.scan.h)))
    H, W = shape
    cx, cy = int(round(q[0])), int(round(q[1]))
    x1, x2, y1, y2 = max(0, cx - r), min(W, cx + r + 1), max(0, cy - r), min(H, cy + r + 1)
    if x1 >= x2 or y1 >= y2:
        return 0.0, dist
    yy, xx = np.mgrid[y1:y2, x1:x2]
    disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    if p.mask is not None:
        m = p.mask[y1:y2, x1:x2] > 0
    else:
        bx1, by1, bx2, by2 = p.box
        m = (xx >= bx1) & (xx < bx2) & (yy >= by1) & (yy < by2)
    n = int(disk.sum())
    return (float((m & disk).sum()) / n if n else 0.0), dist


def _resolve(lab: _Lab, cands: Dict[int, Optional[_Cand]], shape) -> None:
    """Hai ứng viên trùng một gậy: gậy thuộc người có mask bọc chỗ nắm nhiều hơn (hoà thì tay
    gần trục hơn); người thua loại các góc quanh gậy đó rồi lấy ứng viên mới. Lặp đến khi ổn."""
    for _ in range(4 * len(cands) + 4):
        keys = [k for k, c in cands.items() if c is not None]
        clash = next(((i, j) for x, i in enumerate(keys) for j in keys[x + 1:]
                      if _same_pole(cands[i].pole, cands[j].pole)), None)
        if clash is None:
            return
        i, j = clash
        ri, di = _grip_ratio(cands[i], shape)
        rj, dj = _grip_ratio(cands[j], shape)
        if abs(ri - rj) > 1e-6:
            win, lose = (i, j) if ri > rj else (j, i)
        else:
            win, lose = (i, j) if di <= dj else (j, i)
        loser = cands[lose].scan
        loser.excluded.append(cands[win].angle)
        loser.lost_to.append(cands[win].pole)
        cands[lose] = _candidate(lab, loser)
    log.warning("dò gậy: không giải quyết hết tranh chấp sau nhiều vòng")


def find_poles(rgb: np.ndarray, persons: List[Person]) -> List[Pole]:
    """Gậy cầm trên tay của mọi người (toạ độ ảnh làm việc). Người không có tay -> không có gậy.
    Không bao giờ raise: lỗi thì log và trả []."""
    try:
        if not any(p.hands for p in persons):
            return []
        lab = _Lab(rgb)
        cands: Dict[int, Optional[_Cand]] = {}
        k = 0
        for p in persons:
            for hand in p.hands:
                scan = _scan_hand(lab, p, hand)
                cands[k] = None if scan is None else _candidate(lab, scan)
                k += 1
        _resolve(lab, cands, rgb.shape[:2])
        return [c.pole for c in cands.values() if c is not None]
    except Exception as e:  # noqa: BLE001
        log.warning("dò gậy thất bại, bỏ qua: %s", e)
        return []
