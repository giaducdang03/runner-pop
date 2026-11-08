"""Dò gậy (gậy leo núi, gậy tre...) từ nắm tay: quét tia đi xuống từ mỗi tay, kiểm tra mặt cắt
ngang dạng "thanh" (tâm khác hẳn hai bên, hai bên giống nhau) dọc theo tia.

Hai lượt:
1. Quét xuống (nắm tay ở đầu trên gậy, trục trong POLE_ANGLES) -> phân xử chủ gậy -> cổng LSD.
2. Tay chưa có gậy sau lượt 1: quét hai phía (POLE_LINE_ANGLES) - gậy là đường thẳng qua chỗ nắm, kéo dài
   về cả hai phía (gậy cầm giữa thân, gậy gần nằm ngang); điểm = mật độ mẫu đạt trên đoạn đã dò được, xếp
   hạng theo số mẫu đạt -> phân xử -> cổng LSD. Gậy lượt 1 giữ nguyên hình học như trước."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import config as C
from .linecheck import LineIndex
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
_LINE_STEP = 1.0         # độ, bước hướng trục khi quét hai phía (đường dò sau đó bám ngang và khớp lại trục)
_LINE_REF_H = 750.0      # quét hai phía: người cao hơn mức này (px) thì giãn bước r / ox theo h / mức này
_LINE_TRIES = 4          # quét hai phía: số lần dựng ứng viên tối đa mỗi tay (mỗi lần hỏng loại ±3° rồi thử tiếp)
_LINE_CHUNK = 200_000    # số điểm mẫu tối đa mỗi khối khi quét hai phía (giới hạn bộ nhớ)


@dataclass
class Pole:
    top: Tuple[float, float]     # đầu trên (gần/trên nắm tay; gậy hai phía: đầu cao hơn), toạ độ ảnh làm việc
    tip: Tuple[float, float]     # đầu gậy chạm đất (gậy hai phía: đầu còn lại)
    half_w: float                # nửa bề rộng gậy (px)
    score: float                 # quét xuống: tỉ lệ mẫu dạng thanh trên đoạn tia tìm kiếm;
                                 # quét hai phía: mật độ mẫu đạt trên đoạn đã dò được
    owner: int                   # idx người cầm

    @property
    def angle(self) -> float:
        """Góc trục gậy top -> tip (độ), 90 = thẳng xuống, 0 = ngang sang phải."""
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
    spans: np.ndarray        # [r đầu, r cuối] của chuỗi liền được tính điểm, theo góc
    oxs: np.ndarray          # độ dịch tia tốt nhất theo góc
    hws: np.ndarray          # nửa bề rộng (px) có nhiều mẫu đạt nhất theo góc
    excluded: List[float]    # góc bị loại (độ), ±_EXCLUDE_DEG quanh mỗi góc
    lost_to: List["Pole"]    # gậy tay này đã thua; ứng viên mới trùng gậy đó cũng bị loại
    line: bool = False       # quét hai phía: angles là hướng trục, scores = số mẫu đạt (-1 = không đạt)
    dens: Optional[np.ndarray] = None   # quét hai phía: mật độ mẫu đạt trên đoạn đã dò, theo góc
    tries: int = 0           # quét hai phía: số lần đã dựng ứng viên


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
        self.lines = LineIndex(rgb)      # đoạn LSD để kiểm tra gậy, tính lười khi cần

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


def _support_fast(lab: _Lab, pts: np.ndarray, n: np.ndarray, hws_px: np.ndarray) -> np.ndarray:
    """Như _support (cùng quy tắc lấy mẫu / ngưỡng) nhưng float32 và so bình phương, cho lượt quét hai phía."""
    pts = pts.astype(np.float32)
    n = n.astype(np.float32)
    c = lab.sample(pts)
    t2 = np.float32(C.POLE_BAR_CONTRAST ** 2)
    r2 = np.float32(C.POLE_SIDE_RATIO ** 2)
    out = []
    for hw in hws_px:
        off = np.float32(hw + _SIDE_PAD) * n
        left = lab.sample(pts - off)
        right = lab.sample(pts + off)
        bar = c - 0.5 * (left + right)
        side = left - right
        b2 = np.einsum("...i,...i->...", bar, bar)
        s2 = np.einsum("...i,...i->...", side, side)
        out.append((b2 > t2) & (s2 < r2 * b2))
    return np.stack(out)


def _longest_chain(sup: np.ndarray, gap_n: float) -> Optional[Tuple[int, int, int]]:
    """sup 1D (mẫu theo thứ tự dọc tia). Chuỗi có nhiều mẫu đạt nhất, hai mẫu đạt liên tiếp
    cách nhau không quá gap_n bước (cùng quy tắc hụt như khi dò đầu gậy).
    Trả (số mẫu đạt, chỉ số đầu, chỉ số cuối) hoặc None nếu không có mẫu nào."""
    idx = np.flatnonzero(sup)
    if len(idx) == 0:
        return None
    cut = np.flatnonzero(np.diff(idx) > gap_n)
    starts = np.concatenate([[0], cut + 1])
    ends = np.concatenate([cut, [len(idx) - 1]])
    j = int(np.argmax(ends - starts))
    return int(ends[j] - starts[j] + 1), int(idx[starts[j]]), int(idx[ends[j]])


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
    gap_n = C.POLE_GAP * h / _R_STEP
    scores = np.full(len(angles), -1.0)
    spans = np.zeros((len(angles), 2))          # đoạn [r đầu, r cuối] của chuỗi được tính điểm
    best_ox = np.zeros(len(angles))
    best_hw = np.full(len(angles), hws_px[0])
    for i, a in enumerate(angles):
        d, n = _dir(a)
        origins = hand[None] + oxs[:, None] * n[None]                 # K x 2
        pts = origins[:, None] + rs[None, :, None] * d[None, None]    # K x R x 2
        ok = lab.inside(pts)
        cnt = ok.sum(axis=1)
        sup = _support(lab, pts, n, hws_px) & ok[None]                # HW x K x R
        any_sup = sup.any(axis=0)
        # điểm = số mẫu đạt của chuỗi liền dài nhất / số mẫu trong ảnh (mẫu rải rác không tính)
        chains = [_longest_chain(any_sup[k], gap_n) if cnt[k] >= _MIN_SAMPLES else None
                  for k in range(len(oxs))]
        sc = np.array([-1.0 if c is None else c[0] / cnt[k] for k, c in enumerate(chains)])
        k = int(np.argmax(sc))
        if sc[k] < 0:
            continue
        _, lo, hi = chains[k]
        scores[i] = sc[k]
        spans[i] = (rs[lo], rs[hi])
        best_ox[i] = oxs[k]
        best_hw[i] = _narrowest(sup[:, k, lo:hi + 1].sum(axis=1), hws_px)
    return _Scan(person, hand, h, angles, scores, spans, best_ox, best_hw, [], [])


def _trace(lab: _Lab, o: np.ndarray, d: np.ndarray, n: np.ndarray, rs: np.ndarray,
           hws_px: np.ndarray, gap: float, lat_max: float,
           span: Optional[Tuple[float, float]] = None) -> Optional[np.ndarray]:
    """Đi dọc o + r*d theo thứ tự rs (dừng ở mép ảnh), bám theo gậy: lệch ngang tối đa ±1 px
    mỗi bước, trong ±lat_max. Các mẫu đạt được chia thành chuỗi (hụt > gap thì đứt).
    span None: lấy chuỗi bắt đầu ngay từ rs[0] (hụt tính cả từ rs[0]); có span: lấy chuỗi
    nhiều mẫu nhất giao với đoạn span. Trả các điểm đạt (N x 2, theo thứ tự đi) của chuỗi đó,
    điểm cuối là đầu gậy (None nếu không có)."""
    pts = o[None] + rs[:, None] * d[None]
    ok = lab.inside(pts)
    if not ok.all():
        rs = rs[:int(np.argmin(ok))]
    if len(rs) == 0:
        return None
    lats = np.arange(-round(lat_max), round(lat_max) + 1, 1.0)
    grid = o[None, None] + rs[:, None, None] * d + lats[None, :, None] * n    # R x L x 2
    sup = _support(lab, grid, n, hws_px).any(axis=0) & lab.inside(grid)     # R x L
    cur = len(lats) // 2
    chains: List[List[Tuple[int, int]]] = []
    last_r = rs[0]
    for i in range(len(rs)):
        lo, hi = max(0, cur - 1), min(len(lats), cur + 2)
        hit = np.flatnonzero(sup[i, lo:hi])
        if len(hit):
            cur = lo + int(hit[np.argmin(np.abs(lo + hit - cur))])
            if not chains or abs(rs[i] - last_r) > gap:
                chains.append([])
            chains[-1].append((i, cur))
            last_r = rs[i]
        elif span is None and abs(rs[i] - last_r) > gap:
            break
    chains = [c for c in chains if c]
    if span is not None:
        chains = [c for c in chains if rs[c[0][0]] <= span[1] and rs[c[-1][0]] >= span[0]]
    if not chains:
        return None
    best = max(chains, key=len) if span is not None else chains[0]
    return np.array([grid[i, j] for i, j in best])


def _trim_sparse_tail(path: np.ndarray, o: np.ndarray, d: np.ndarray, h: float) -> np.ndarray:
    """Cắt đuôi thưa của đường dò: thân gậy thật cho mẫu đạt gần liên tục (mật độ ~1), còn khi đường
    dò chạy tiếp qua cỏ / nền nhiều vân thì chỉ thỉnh thoảng gặp mẫu đạt (mật độ 0-0.6) nhưng vẫn
    không hụt quá POLE_GAP. Chia đường theo cửa sổ POLE_TAIL_WIN * h; đầu gậy = cuối cặp cửa sổ đặc
    (>= POLE_TAIL_DENSE) cuối cùng, nếu phần sau nó thưa (TB < POLE_TAIL_SPARSE); không thì giữ nguyên.
    `path` theo thứ tự đi (điểm đầu gần nắm tay), dùng được cho cả đường đi xuống và đi lên."""
    t = np.abs((path - o) @ d - float((path[0] - o) @ d))      # quãng đường dọc tia từ điểm đầu
    win = C.POLE_TAIL_WIN * h
    if win <= 0 or t[-1] < 2 * win:
        return path
    idx = (t // win).astype(int)
    nwin = int(idx[-1]) + 1
    counts = np.bincount(idx, minlength=nwin).astype(np.float64)
    lengths = np.full(nwin, win)
    lengths[-1] = t[-1] - (nwin - 1) * win + _R_STEP            # cửa sổ cuối chỉ dài tới đầu gậy
    dens = counts * _R_STEP / np.maximum(lengths, _R_STEP)
    dense = dens >= C.POLE_TAIL_DENSE
    pairs = np.flatnonzero(dense[:-1] & dense[1:])
    if len(pairs) == 0:
        return path
    j = int(pairs[-1]) + 1                                      # cửa sổ đặc cuối cùng
    if j >= nwin - 1 or dens[j + 1:].mean() >= C.POLE_TAIL_SPARSE:
        return path
    return path[idx <= j]


def _narrowest(hits: np.ndarray, hws_px: np.ndarray) -> float:
    """Bề rộng hẹp nhất có số mẫu đạt >= POLE_HW_NARROW x tốt nhất (dải rộng dễ đạt trên nền
    nhiều vân nên không lấy thẳng argmax)."""
    return float(hws_px[int(np.flatnonzero(hits >= C.POLE_HW_NARROW * hits.max())[0])])


def _refine_hw(lab: _Lab, path: np.ndarray, n: np.ndarray, hws_px: np.ndarray) -> float:
    """Ước lại nửa bề rộng dọc đường đã bám theo gậy, cho tâm lệch ngang ±_OX_STEP px (đường
    bám có thể nằm lệch tâm gậy; gậy hai màu có vệt sáng thì dải hẹp chỉ đạt khi đặt đúng tâm)."""
    lats = np.arange(-_OX_STEP, _OX_STEP + 0.5, 1.0)
    grid = path[:, None] + lats[None, :, None] * n                       # N x L x 2
    sup = _support(lab, grid, n, hws_px) & lab.inside(grid)[None]        # HW x N x L
    return _narrowest(sup.any(axis=2).sum(axis=1), hws_px)


def _candidate(lab: _Lab, scan: _Scan) -> Optional[_Cand]:
    """Góc tốt nhất (bỏ các góc bị loại) của một tay; ứng viên trùng một gậy tay này đã thua
    thì loại thêm ±3° quanh góc đó và tìm tiếp."""
    while True:
        sc = scan.scores.copy()
        for ex in scan.excluded:
            sc[np.abs(scan.angles - ex) <= _EXCLUDE_DEG] = -1.0
        i = int(np.argmax(sc))
        if scan.line:
            # điểm quét hai phía là số mẫu đạt, -1 = không đạt mật độ / chiều dài
            if sc[i] <= 0 or scan.tries >= _LINE_TRIES:
                return None
            scan.tries += 1
            cand = _build_line(lab, scan, i)
        else:
            if sc[i] < C.POLE_MIN_SCORE:
                return None
            cand = _build(lab, scan, i, float(sc[i]))
        if cand is not None and not any(_same_pole(cand.pole, q) for q in scan.lost_to):
            return cand
        scan.excluded.append(float(scan.angles[i]))   # gậy quá ngắn / trùng gậy đã thua
        if cand is not None and scan.line:            # đường dò bám sang gậy đã thua: loại cả hướng của nó
            scan.excluded.append(_orient(cand.pole))


def _lsd_ok(lab: _Lab, pole: Pole) -> bool:
    """Gậy có đủ mép thẳng LSD dọc trục (độ phủ >= POLE_LSD_MIN). POLE_LSD_MIN = 0 tắt kiểm tra;
    LSD lỗi (độ phủ None) thì coi như đạt để không làm hỏng request."""
    if C.POLE_LSD_MIN <= 0:
        return True
    cov = lab.lines.coverage(pole)
    return cov is None or cov >= C.POLE_LSD_MIN


def _below_grip(pole: Pole, hand: np.ndarray) -> float:
    """Chiều dài phần gậy phía dưới chỗ nắm: từ hình chiếu của tay lên trục tới đầu thấp hơn (tip), px."""
    a, b = np.asarray(pole.top), np.asarray(pole.tip)
    L = float(np.linalg.norm(b - a))
    if L <= 0:
        return 0.0
    return L - float((np.asarray(hand) - a) @ (b - a)) / L


def _lsd_trim(lab: _Lab, pole: Pole, h: float) -> Optional[Pole]:
    """Gậy quét hai phía: cắt hai đầu về đoạn có mép LSD - đường dò bám có thể chạy quá đầu gậy qua cỏ / lá
    (mặt cắt vẫn lác đác dạng thanh) nhưng ở đó gần như không có đoạn LSD song song. Mỗi đầu lùi vào tới mẫu
    có đoạn LSD đầu tiên mà cửa sổ POLE_TAIL_WIN * h tính từ đó vào trong có độ phủ >= POLE_LINE_TRIM_COV.
    Gậy còn lại ngắn hơn POLE_LINE_MIN_LEN * h thì trả None; LSD lỗi thì giữ nguyên."""
    cov = lab.lines.covered(pole)
    if cov is None or len(cov) < 2:
        return pole
    step = C.POLE_LSD_STEP / lab.lines.scale                    # px ảnh làm việc mỗi mẫu
    win = max(1, int(round(C.POLE_TAIL_WIN * h / step)))
    cs = np.concatenate([[0], np.cumsum(cov)])
    M = len(cov)
    j = np.arange(M)
    fwd = (cs[np.minimum(j + win, M)] - cs[j]) / (np.minimum(j + win, M) - j)    # cửa sổ từ j đi về tip
    bwd = (cs[j + 1] - cs[np.maximum(j + 1 - win, 0)]) / (j + 1 - np.maximum(j + 1 - win, 0))
    ok_a = np.flatnonzero(cov & (fwd >= C.POLE_LINE_TRIM_COV))
    ok_b = np.flatnonzero(cov & (bwd >= C.POLE_LINE_TRIM_COV))
    if len(ok_a) == 0 or len(ok_b) == 0 or ok_b[-1] <= ok_a[0]:
        return None
    a, b = np.asarray(pole.top), np.asarray(pole.tip)
    L = float(np.linalg.norm(b - a))
    d = (b - a) / L
    ua, ub = ok_a[0] * step, min(L, ok_b[-1] * step)
    if ub - ua < C.POLE_LINE_MIN_LEN * h:
        return None
    top, tip = a + ua * d, a + ub * d
    return replace(pole, top=(float(top[0]), float(top[1])), tip=(float(tip[0]), float(tip[1])))


def _build(lab: _Lab, scan: _Scan, i: int, score: float) -> Optional[_Cand]:
    """Truy đầu gậy và đầu trên cho góc thứ i; None nếu không dò được đầu gậy hoặc gậy
    ngắn hơn POLE_MIN_LEN * h."""
    a, h = float(scan.angles[i]), scan.h
    d, n = _dir(a)
    o = scan.hand + scan.oxs[i] * n
    hws_px = np.asarray(C.POLE_HALF_W, np.float64) * h
    gap = C.POLE_GAP * h
    diag = math.hypot(lab.w, lab.h)
    lat = max(C.POLE_LAT_MIN_PX, C.POLE_LAT_HW * float(scan.hws[i]))
    path = _trace(lab, o, d, n, np.arange(C.POLE_SEARCH[0] * h, diag, _R_STEP), hws_px, gap, lat,
                 span=(float(scan.spans[i, 0]), float(scan.spans[i, 1])))
    up = _trace(lab, o, d, n, -np.arange(0.0, C.POLE_GRIP_UP * h + _R_STEP / 2, _R_STEP),
                hws_px, gap, lat)
    if path is None:                      # không dò được đầu gậy = không có bằng chứng
        return None
    path = _trim_sparse_tail(path, o, d, h)       # đuôi thưa qua cỏ / nền không phải gậy
    if up is not None:
        up = _trim_sparse_tail(up, o, d, h)
    tip = path[-1]
    top = o if up is None else up[-1]
    hw = _refine_hw(lab, path, n, hws_px)
    if float(np.linalg.norm(tip - top)) < C.POLE_MIN_LEN * h:
        return None
    pole = Pole(top=(float(top[0]), float(top[1])), tip=(float(tip[0]), float(tip[1])),
                half_w=hw, score=score, owner=scan.person.idx)
    return _Cand(scan, a, pole)


def _orient(pole: Pole) -> float:
    """Hướng trục trong [0, 180) độ (0 = ngang, 90 = dọc)."""
    return float(np.degrees(np.arctan2(pole.tip[1] - pole.top[1], pole.tip[0] - pole.top[0])) % 180.0)


def _scan_line(lab: _Lab, person: Person, hand) -> Optional[_Scan]:
    """Quét hai phía: với mỗi hướng trục a (POLE_LINE_ANGLES) và độ dịch ox, lấy mẫu dọc hai nửa đường
    thẳng qua chỗ nắm, r trong [r0, r1] * h về mỗi phía (bỏ ±r0 quanh tay cho nắm tay / găng). Mỗi phía lấy
    chuỗi bám từ r0 (hụt tính từ r0, đứt khi hụt > POLE_GAP hoặc ra khỏi ảnh) - cùng quy tắc với _trace.
    Mật độ = số mẫu đạt / số mẫu từ r0 tới mẫu đạt cuối (cộng hai phía); chiều dài = tổng r của mẫu đạt cuối
    hai phía. Đạt khi mật độ >= POLE_LINE_MIN_SCORE và chiều dài >= POLE_LINE_MIN_LEN * h; xếp hạng theo số
    mẫu đạt. Vector hoá theo khối hướng."""
    h = float(person.body_h)
    if h <= 0:
        return None
    hand = np.asarray(hand, np.float64)
    hws_px = np.asarray(C.POLE_HALF_W, np.float64) * h
    lo_a, hi_a = C.POLE_LINE_ANGLES
    angles = np.arange(lo_a, hi_a + _LINE_STEP / 2, _LINE_STEP)
    tol = C.POLE_HAND_TOL * h
    g = max(1.0, h / _LINE_REF_H)                       # ảnh lớn: số mẫu không tăng theo h
    oxs = np.arange(-tol, tol + 1, _OX_STEP * g)
    rs = np.arange(C.POLE_SEARCH[0] * h, C.POLE_SEARCH[1] * h, _R_STEP * g)
    R, K = len(rs), len(oxs)
    if R == 0:
        return None
    gap_n = C.POLE_GAP * h / (_R_STEP * g)
    sr = np.concatenate([rs, -rs])                      # 2R: phía +d rồi phía -d
    idx = np.arange(R)
    scores = np.full(len(angles), -1.0)
    dens_out = np.zeros(len(angles))
    best_ox = np.zeros(len(angles))
    best_hw = np.full(len(angles), hws_px[0])
    step = max(1, _LINE_CHUNK // (K * 2 * R))
    for s0 in range(0, len(angles), step):
        t = np.radians(angles[s0:s0 + step])
        d = np.stack([np.cos(t), np.sin(t)], axis=1)                       # A x 2
        n = np.stack([-d[:, 1], d[:, 0]], axis=1)
        A = len(t)
        pts = (hand + oxs[None, :, None, None] * n[:, None, None, :]
               + sr[None, None, :, None] * d[:, None, None, :])             # A x K x 2R x 2
        pts = pts.reshape(A, K, 2, R, 2)
        ok = lab.inside(pts)                                               # A x K x 2 x R
        sup_hw = _support_fast(lab, pts, n[:, None, None, None, :], hws_px) & ok[None]   # HW x A x K x 2 x R
        sup = sup_hw.any(axis=0)
        prev = np.maximum.accumulate(np.where(sup, idx, 0), axis=-1)        # mẫu đạt gần nhất (neo ở r0)
        brk = ((idx - prev) > gap_n) | ~ok
        b = np.where(brk.any(axis=-1), np.argmax(brk, axis=-1), R)        # A x K x 2: chỉ số đứt chuỗi
        chain = idx < b[..., None]
        hit = sup & chain
        nh = hit.sum(axis=-1)                                              # A x K x 2
        last = np.where(nh > 0, R - 1 - np.argmax(hit[..., ::-1], axis=-1), -1)
        hits = nh.sum(axis=-1)                                             # A x K
        ext = (last + 1).sum(axis=-1)
        length = np.where(nh > 0, rs[np.maximum(last, 0)], 0.0).sum(axis=-1)
        dens = hits / np.maximum(ext, 1)
        # số mẫu đạt, hoà thì mật độ cao hơn thắng (dens <= 1 nên không đảo thứ tự số mẫu đạt)
        rank = np.where((dens >= C.POLE_LINE_MIN_SCORE) & (length >= C.POLE_LINE_MIN_LEN * h), hits + 0.5 * dens, -1)
        k = np.argmax(rank, axis=1)                                        # A
        ai = np.arange(A)
        scores[s0:s0 + A] = rank[ai, k]
        dens_out[s0:s0 + A] = dens[ai, k]
        best_ox[s0:s0 + A] = oxs[k]
        hw_hits = (sup_hw & chain[None]).sum(axis=(-1, -2))[:, ai, k]       # HW x A
        for j in np.flatnonzero(rank[ai, k] > 0):
            best_hw[s0 + j] = _narrowest(hw_hits[:, j], hws_px)
    return _Scan(person, hand, h, angles, scores, np.zeros((len(angles), 2)), best_ox, best_hw, [], [],
                 line=True, dens=dens_out)


def _center(lab: _Lab, path: np.ndarray, n: np.ndarray, hw: float) -> np.ndarray:
    """Dời mỗi điểm đường dò theo pháp tuyến (±_OX_STEP px) về giữa gậy: chọn các độ lệch có mặt cắt dạng
    thanh với độ tương phản tâm-hai bên >= 0.9 x lớn nhất của điểm đó, lấy trung bình (đường dò bám theo
    gậy có thể chạy sát mép gậy khi tia quét cắt chéo bề rộng gậy). Điểm không có độ lệch nào đạt giữ nguyên."""
    lats = np.arange(-_OX_STEP, _OX_STEP + 0.5, 1.0)
    grid = path[:, None] + lats[None, :, None] * n                       # N x L x 2
    c = lab.sample(grid)
    off = (hw + _SIDE_PAD) * n
    left, right = lab.sample(grid - off), lab.sample(grid + off)
    bar = np.linalg.norm(c - 0.5 * (left + right), axis=-1)
    side = np.linalg.norm(left - right, axis=-1)
    good = (bar > C.POLE_BAR_CONTRAST) & (side < C.POLE_SIDE_RATIO * bar) & lab.inside(grid)
    bar = np.where(good, bar, 0.0)
    top = bar.max(axis=1, keepdims=True)
    sel = good & (bar >= 0.9 * top)
    cnt = sel.sum(axis=1)
    shift = np.where(cnt > 0, (sel * lats[None]).sum(axis=1) / np.maximum(cnt, 1), 0.0)
    return path + shift[:, None] * n


def _build_line(lab: _Lab, scan: _Scan, i: int) -> Optional[_Cand]:
    """Dò hai phía chỗ nắm theo hướng thứ i (mỗi phía bám từ r0, cắt đuôi thưa); hai đầu = điểm cuối hai
    đường dò chiếu lên trục khớp (phía không dò được thì là chỗ nắm). top = đầu cao hơn trong ảnh. None nếu không phía nào dò
    được hoặc |tip − top| < POLE_LINE_MIN_LEN * h."""
    a, h = float(scan.angles[i]), scan.h
    d, n = _dir(a)
    o = scan.hand + scan.oxs[i] * n
    hws_px = np.asarray(C.POLE_HALF_W, np.float64) * h
    gap = C.POLE_GAP * h
    diag = math.hypot(lab.w, lab.h)
    lat = max(C.POLE_LAT_MIN_PX, C.POLE_LAT_HW * float(scan.hws[i]))
    rs = np.arange(C.POLE_SEARCH[0] * h, diag, _R_STEP)
    paths = []
    for sgn in (1.0, -1.0):
        p = _trace(lab, o, d, n, sgn * rs, hws_px, gap, lat)
        paths.append(None if p is None else _trim_sparse_tail(p, o, d, h))
    if paths[0] is None and paths[1] is None:
        return None
    # trục = đường thẳng khớp (PCA) qua các điểm đã bám (tia quét có thể lệch góc / lệch ngang vài px);
    # hai đầu = chiếu điểm cuối hai đường dò lên trục, phía không dò được thì là hình chiếu của chỗ nắm
    allp = _center(lab, np.concatenate([q for q in paths if q is not None]), n, float(scan.hws[i]))
    c = allp.mean(axis=0)
    u = np.linalg.svd(allp - c, full_matrices=False)[2][0] if len(allp) >= 2 else d
    if float(u @ d) < 0:
        u = -u
    ori = math.degrees(math.atan2(u[1], u[0])) % 180.0
    if not (C.POLE_LINE_ANGLES[0] <= ori <= C.POLE_LINE_ANGLES[1]):
        return None                       # trục khớp ra ngoài prior hướng (vd. bám theo thanh nằm ngang)

    def proj(pt):
        return c + float((pt - c) @ u) * u
    e0 = proj(paths[0][-1] if paths[0] is not None else scan.hand)
    e1 = proj(paths[1][-1] if paths[1] is not None else scan.hand)
    paths = [q for q in paths if q is not None]
    if float(np.linalg.norm(e1 - e0)) < C.POLE_LINE_MIN_LEN * h:
        return None
    top, tip = (e0, e1) if e0[1] <= e1[1] else (e1, e0)
    hw = _refine_hw(lab, allp, np.array([-u[1], u[0]]), hws_px)
    pole = Pole(top=(float(top[0]), float(top[1])), tip=(float(tip[0]), float(tip[1])),
                half_w=hw, score=float(scan.dens[i]), owner=scan.person.idx)
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
        hands = [(p, hand) for p in persons for hand in p.hands]
        cands: Dict[int, Optional[_Cand]] = {}
        for k, (p, hand) in enumerate(hands):
            scan = _scan_hand(lab, p, hand)
            cands[k] = None if scan is None else _candidate(lab, scan)
        _resolve(lab, cands, rgb.shape[:2])
        # Cổng LSD đặt SAU phân xử chủ gậy (đo trên 40 ảnh gán nhãn: đặt trong vòng chọn góc để xếp hạng
        # lại không cứu thêm gậy thật nào, chỉ sinh thêm 4-8 gậy mới chưa kiểm chứng, phần lớn trông giả)
        kept = {k: c for k, c in cands.items() if c is not None and _lsd_ok(lab, c.pole)}
        if C.POLE_LINE_ANGLES is not None:
            # lượt 2: tay chưa có gậy quét hai phía; ứng viên trùng gậy đã giữ bị loại (lost_to)
            held = [c.pole for c in kept.values()]
            line: Dict[int, Optional[_Cand]] = {}
            for k, (p, hand) in enumerate(hands):
                if k in kept:
                    continue
                scan = _scan_line(lab, p, hand)
                if scan is not None:
                    scan.lost_to.extend(held)
                line[k] = None if scan is None else _candidate(lab, scan)
            _resolve(lab, line, rgb.shape[:2])
            for k, c in line.items():
                if c is not None and _lsd_ok(lab, c.pole):
                    pole = _lsd_trim(lab, c.pole, c.scan.h) if C.POLE_LSD_MIN > 0 else c.pole
                    if pole is not None and _below_grip(pole, c.scan.hand) >= C.POLE_LINE_MIN_DOWN * c.scan.h:
                        kept[k] = replace(c, pole=pole)
        return [kept[k].pole for k in sorted(kept)]
    except Exception as e:  # noqa: BLE001
        log.warning("dò gậy thất bại, bỏ qua: %s", e)
        return []
