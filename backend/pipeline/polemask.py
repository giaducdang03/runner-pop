"""Mở rộng mask gậy từ dải dò được: vùng khác nền cục bộ quanh dải (gậy thứ hai cầm song song,
đầu gậy dày: tay cầm, quai, chóp).

Làm trong khung xoay theo trục gậy (cột = dọc trục, hàng = lệch ngang). Vùng tìm = |lệch| <= band_r;
nền cục bộ ước lượng từ vành hai bên (band_r < |lệch| <= band_r + ring_w): trung vị màu mỗi bên theo
cửa sổ dọc trục, cộng inpaint Telea. Pixel là ứng viên khi khác nền rõ rệt: khoảng cách Lab tới ĐOẠN
nối màu hai bên (màu pha giữa hai bên của đường chân trời cắt ngang không tính) và tới nền inpaint đều
vượt ngưỡng; ngưỡng tăng theo độ gồ ghề của nền. Ứng viên còn bị lọc theo độ mảnh (biến đổi khoảng
cách), dính vào dải, hướng song song trục hoặc là khối nhỏ ở đầu gậy, và tổng diện tích. Không chắc
chắn thì trả None (dùng dải gậy như cũ)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from . import config as C
from .poles import Pole

log = logging.getLogger(__name__)


@dataclass
class PoleMask:
    """Mask gậy đã mở rộng trong crop `box` (x1, y1, x2, y2) của ảnh đầy đủ."""
    box: Tuple[int, int, int, int]
    mask: np.ndarray            # bool, cỡ crop
    ratio: float                # diện tích mask / diện tích dải gậy


def _disk(r: int) -> np.ndarray:
    r = max(1, int(r))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _to_u8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    return np.clip(np.asarray(img, np.float32) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def refine_pole_mask(image: np.ndarray, pole: Pole, body_h: float,
                     known: Optional[np.ndarray] = None, hands: Sequence = (),
                     dbg: Optional[dict] = None) -> Optional[PoleMask]:
    """Mask gậy mở rộng quanh `pole` (ảnh đầy đủ RGB, uint8 hoặc float 0..1) hoặc None = dùng dải gậy.
    `known`: bool cỡ ảnh, pixel đã chắc là người / người khác (không tính là gậy, không lấy làm mẫu nền).
    `hands`: tâm nắm tay của người cầm (toạ độ ảnh); gậy thứ hai phải hội tụ về chỗ nắm trên gậy dò được.
    `dbg`: dict nhận các ảnh trung gian (gỡ lỗi). Không bao giờ ném lỗi."""
    try:
        return _refine(image, pole, float(body_h), known, hands, dbg if dbg is not None else {})
    except Exception as e:  # noqa: BLE001  (lỗi thì quay về dải gậy)
        log.warning("mở rộng mask gậy thất bại: %s", e)
        return None


def _side_bg(rows: np.ndarray, ok: np.ndarray, win: int) -> np.ndarray:
    """Trung vị theo cửa sổ dọc trục (±win cột) của các pixel hợp lệ trong `rows` (r x n x k) -> n x k
    (NaN nếu cửa sổ không có pixel hợp lệ)."""
    v = np.where(ok[..., None], rows.astype(np.float32), np.nan)
    v = np.pad(v, ((0, 0), (win, win), (0, 0)), constant_values=np.nan)
    w = np.lib.stride_tricks.sliding_window_view(v, 2 * win + 1, axis=1)   # r x n x k x (2win+1)
    w = np.moveaxis(w, 2, -1).transpose(1, 0, 2, 3)                        # n x r x (2win+1) x k
    w = w.reshape(w.shape[0], -1, w.shape[-1])
    w = np.sort(w, axis=1)                                 # NaN dồn về cuối
    cnt = np.isfinite(w[..., 0]).sum(axis=1)               # số pixel hợp lệ mỗi cột
    lo = np.clip((cnt - 1) // 2, 0, None)[:, None, None]
    hi = np.clip(cnt // 2, 0, w.shape[1] - 1)[:, None, None]
    med = 0.5 * (np.take_along_axis(w, lo, 1) + np.take_along_axis(w, hi, 1))[:, 0]
    return np.where((cnt > 0)[:, None], med, np.nan).astype(np.float32)


def _seg_dist(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Khoảng cách từ màu x (r x n x 3) tới đoạn [a, b] (n x 3, mỗi cột một đoạn)."""
    ab = (b - a)[None]
    t = np.sum((x - a[None]) * ab, -1) / np.maximum(np.sum(ab * ab, -1), 1e-6)
    t = np.clip(t, 0.0, 1.0)[..., None]
    return np.linalg.norm(x - (a[None] + t * ab), axis=-1)


def _bar_stats(labf: np.ndarray, xs: np.ndarray, ys: np.ndarray, R: int) -> Tuple[float, float]:
    """Gậy là một thanh mảnh: (trung vị theo cột của khoảng cách Lab giữa màu thanh và nền ngay sát mép
    ngoài thanh, phía xa trục, cách mép 3 px; trung vị bề dày thanh theo cột). Vệt do ước lượng nền sai
    (vành gặp vật khác) nằm trên nền đồng màu nên tương phản với nền sát bên thấp, và thường dày."""
    nrow = labf.shape[0]
    con, thick = [], []
    for c in np.unique(xs):
        rows = ys[xs == c]
        thick.append(rows.max() - rows.min() + 1)
        far = rows.max() + 3 if np.mean(rows) >= R else rows.min() - 3
        if 0 <= far < nrow:
            con.append(float(np.linalg.norm(labf[rows, c].mean(axis=0) - labf[far, c])))
    return (float(np.median(con)) if con else 0.0), float(np.median(thick))


def _refine(image, pole: Pole, h: float, known, hands, dbg: dict) -> Optional[PoleMask]:
    H, W = image.shape[:2]
    hw = float(pole.half_w)
    a = np.asarray(pole.top, np.float64)
    d = np.asarray(pole.tip, np.float64) - a
    L = float(np.linalg.norm(d))
    if L < 4 or h < C.POLE_REFINE_MIN_H or hw > C.POLE_REFINE_MAX_HW * h:
        return None                                         # gậy quá ngắn / người quá nhỏ / gậy dày (tre)
    d /= L
    n = np.array([-d[1], d[0]])
    band_r = int(round(max(C.POLE_REFINE_BAND * h, hw + 4)))
    ring_w = int(round(max(C.POLE_REFINE_RING * h, 6)))
    R = band_r + ring_w
    ncol = int(np.ceil(L)) + 2 * band_r + 1
    nrow = 2 * R + 1
    # khung xoay: (u, v) -> ảnh: a + (u - band_r) * d + (v - R) * n
    M = np.array([[d[0], n[0], a[0] - band_r * d[0] - R * n[0]],
                  [d[1], n[1], a[1] - band_r * d[1] - R * n[1]]], np.float64)
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    src = image if image.dtype in (np.uint8, np.float32) else np.asarray(image, np.float32)
    rot = _to_u8(cv2.warpAffine(src, M, (ncol, nrow), flags=flags, borderMode=cv2.BORDER_REPLICATE))
    uu, vv = np.meshgrid(np.arange(ncol, dtype=np.float64), np.arange(nrow, dtype=np.float64))
    px = M[0, 0] * uu + M[0, 1] * vv + M[0, 2]
    py = M[1, 0] * uu + M[1, 1] * vv + M[1, 2]
    kn = ~((px >= 0) & (px <= W - 1) & (py >= 0) & (py <= H - 1))   # ngoài ảnh
    if known is not None:
        kn |= cv2.warpAffine(np.asarray(known, np.uint8), M, (ncol, nrow), flags=cv2.INTER_NEAREST
                             | cv2.WARP_INVERSE_MAP, borderValue=0) > 0

    v = np.arange(nrow)[:, None] - R                       # lệch ngang (px) của từng hàng
    u = np.arange(ncol)[None, :] - band_r                  # vị trí dọc trục (0 = top, L = tip)
    band = np.broadcast_to(np.abs(v) <= band_r, (nrow, ncol))
    strip = (np.abs(v) <= hw + 1) & (u >= 0) & (u <= L)
    up_rows, dn_rows = slice(0, ring_w), slice(nrow - ring_w, nrow)
    ring_ok = np.zeros((nrow, ncol), bool)
    ring_ok[up_rows] = ~kn[up_rows]
    ring_ok[dn_rows] = ~kn[dn_rows]
    if ring_ok.sum() < 0.3 * 2 * ring_w * ncol:
        return None                                         # vành gần như toàn người / ngoài ảnh

    lab = cv2.cvtColor(cv2.GaussianBlur(rot, (0, 0), 1.0), cv2.COLOR_RGB2LAB)
    labf = lab.astype(np.float32)

    # độ gồ ghề của nền: gradient Lab cục bộ ở vành
    g = np.zeros((nrow, ncol), np.float32)
    for c in range(3):
        gx = cv2.Sobel(labf[..., c], cv2.CV_32F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(labf[..., c], cv2.CV_32F, 0, 1, ksize=3) / 8.0
        g += gx * gx + gy * gy
    g = cv2.blur(np.sqrt(g), (5, 5))
    rough = float(np.percentile(g[ring_ok], 75))
    dbg.update(rough=rough, rot=rot, band_r=band_r, ring_w=ring_w)
    if rough > C.POLE_REFINE_ROUGH_MAX:
        return None                                         # nền quanh dải không trơn: ước lượng không tin được

    # nền hai bên theo cửa sổ dọc trục; nền inpaint
    win = ring_w
    up = _side_bg(labf[up_rows], ring_ok[up_rows], win)
    dn = _side_bg(labf[dn_rows], ring_ok[dn_rows], win)
    # chỉ tin cột có mẫu nền ở CẢ hai bên (một bên toàn người / ngoài ảnh thì nền bên kia không đại diện)
    valid_col = np.isfinite(up[:, 0]) & np.isfinite(dn[:, 0])
    up, dn = np.nan_to_num(up), np.nan_to_num(dn)
    hole = (band | kn).astype(np.uint8)
    if band_r >= 16:        # vùng tìm rộng: inpaint ở nửa độ phân giải rồi phóng lại (nền trơn, đỡ tốn thời gian)
        small = cv2.inpaint(cv2.resize(lab, ((ncol + 1) // 2, (nrow + 1) // 2), interpolation=cv2.INTER_AREA),
                            cv2.resize(hole, ((ncol + 1) // 2, (nrow + 1) // 2), interpolation=cv2.INTER_NEAREST)
                            | cv2.resize(hole, ((ncol + 1) // 2, (nrow + 1) // 2), interpolation=cv2.INTER_AREA),
                            C.POLE_REFINE_INPAINT_R, cv2.INPAINT_TELEA)
        bg = cv2.resize(small, (ncol, nrow), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    else:
        bg = cv2.inpaint(lab, hole, C.POLE_REFINE_INPAINT_R, cv2.INPAINT_TELEA).astype(np.float32)
    diff = np.minimum(_seg_dist(labf, up, dn), np.linalg.norm(labf - bg, axis=-1))
    tex_up = _side_bg(g[up_rows][..., None], ring_ok[up_rows], win)[:, 0]
    tex_dn = _side_bg(g[dn_rows][..., None], ring_ok[dn_rows], win)[:, 0]
    tex = np.nan_to_num(np.fmax(tex_up, tex_dn))
    thr = C.POLE_REFINE_DIFF + C.POLE_REFINE_TEX_K * tex[None, :]

    fg = (diff > thr) & band & ~kn & valid_col[None, :]
    fg0 = fg
    dbg.update(diff=diff, thr=np.broadcast_to(thr, diff.shape), fg0=fg.copy(), strip=strip)
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # độ rộng cục bộ: bỏ phần đủ rộng để chứa đĩa bán kính w_max (vệt tràn, mảng nền); ở hai đầu gậy
    # (tay cầm, quai, chóp) cho phép rộng hơn: w_end
    w_max = max(C.POLE_REFINE_WMUL * hw, C.POLE_REFINE_WMIN * h, 2.0)
    w_end = max(C.POLE_REFINE_WEND * w_max, w_max)
    end_r = band_r                                          # hai đầu: đoạn dài band_r tính từ mỗi đầu dải
    ends = np.broadcast_to((u <= end_r) | (u >= L - end_r), (nrow, ncol))
    # (bỏ dải gậy dò được khi đo: gậy thứ hai sát gậy chính cộng bề rộng hai gậy không bị coi là rộng)
    fg_ns = fg & ~strip
    wide = np.where(ends, cv2.morphologyEx(fg_ns, cv2.MORPH_OPEN, _disk(int(np.ceil(w_end)) + 1)),
                    cv2.morphologyEx(fg_ns, cv2.MORPH_OPEN, _disk(int(np.ceil(w_max)) + 1))).astype(np.uint8)
    fg &= (cv2.dilate(wide, _disk(int(np.ceil(w_max)) + 1)) == 0)    # cả viền sót lại quanh khối rộng

    # chỉ giữ thành phần dính vào dải gậy (cho phép hở `gap` px)
    gap = max(C.POLE_REFINE_GAP * hw, 2.0)
    seed = (np.abs(v) <= hw + 1 + gap) & (u >= -gap) & (u <= L + gap)
    _, lbl = cv2.connectedComponents(((fg > 0) | seed).astype(np.uint8), connectivity=8)
    hit = np.unique(lbl[seed])
    extra = np.isin(lbl, hit[hit > 0]) & (fg > 0) & ~strip
    dbg["conn"] = extra | strip

    # chỗ nắm: hình chiếu lên trục của nắm tay gần trục nhất (cách trục <= band_r + 0.03h)
    grip = None
    for hx, hy in hands or ():
        q = np.array([hx, hy], np.float64) - a
        th, sh = float(q @ d), float(q @ n)
        if abs(sh) <= band_r + 0.03 * h and -band_r <= th <= L + band_r:
            if grip is None or abs(sh) < grip[1]:
                grip = (th, abs(sh))

    # phần thân (ngoài hai đầu): mỗi thành phần khớp một đường thẳng (Huber); giữ pixel cách đường
    # <= w_max nếu đường gần song song trục, phủ đủ dài và hội tụ về chỗ nắm (gậy thứ hai cầm cùng tay;
    # gậy song song của người khác không qua tay người cầm); phần bám theo (viền tay, nền) bỏ
    keep = np.zeros((nrow, ncol), bool)
    max_slope = np.tan(np.radians(C.POLE_REFINE_ANGLE))
    conv_tol = max(2 * w_max, C.POLE_REFINE_CONVERGE * h)
    k_n, lbl = cv2.connectedComponents((extra & ~ends).astype(np.uint8), connectivity=8)
    for k in range(1, k_n):
        ys, xs = np.nonzero(lbl == k)
        if len(xs) < 4:
            continue
        pts = np.stack([xs, ys], 1).astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        dist = np.abs((pts[:, 0] - x0) * vy - (pts[:, 1] - y0) * vx)
        inl = dist <= w_max
        cover = len(np.unique(xs[inl]))                     # số cột (px dọc trục) có pixel nằm trên đường
        slope = abs(vy) / max(abs(vx), 1e-6)
        ok = slope <= max_slope and cover >= C.POLE_REFINE_MIN_LEN * h
        s_grip = bar = None
        if ok:
            near_axis = abs(float(np.median(ys[inl])) - R) <= hw + 3   # viền / mép gậy dò được
            if not near_axis:
                if grip is None:
                    ok = False
                else:
                    s_grip = float(y0 + (grip[0] + band_r - x0) * vy / vx - R)
                    bar, thick = _bar_stats(labf, xs[inl], ys[inl], R)
                    ok = (abs(s_grip) <= conv_tol and bar >= C.POLE_REFINE_DIFF
                          and thick <= 2 * hw + 2 + C.POLE_REFINE_THICK_PX)
        dbg.setdefault("comps", []).append(dict(n=len(xs), cover=cover, slope=round(float(slope), 3),
                                                t=(int(xs.min()) - band_r, int(xs.max()) - band_r),
                                                s=(int(ys.min()) - R, int(ys.max()) - R),
                                                s_grip=None if s_grip is None else round(s_grip, 1),
                                                bar=None if bar is None else (round(bar, 1), thick),
                                                s_inl=(int(ys[inl].min()) - R, int(ys[inl].max()) - R), ok=bool(ok)))
        if ok:
            keep[ys[inl], xs[inl]] = True

    # hai đầu gậy: khối nhỏ (tay cầm, quai, chóp) phải dính vào phần chắc là gậy: lõi dải gậy khác nền
    # thật (dải dò được thò quá đầu gậy thật không làm mỏ neo) hoặc phần thân đã giữ. Khối cắt về
    # |lệch| <= half_w + 1 + w_end; nền ở đầu đó gồ ghề (đất, cỏ: chóp gậy chạm đất) thì chỉ giữ phần
    # dày thêm sát trục (|lệch| <= half_w + 1 + w_max)
    core_fg = fg0 & strip
    # đoạn liên tục (hở <= 2 px) của các cột có lõi dải khác nền: thân gậy thật
    pres = np.convolve(core_fg.any(axis=0).astype(float), np.ones(3), "same") > 0
    run_id = np.cumsum(np.r_[pres[:1], pres[1:] & ~pres[:-1]]) * pres
    keep_d = cv2.dilate(keep.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    core_d = cv2.dilate(core_fg.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    min_run = C.POLE_REFINE_MIN_LEN * h
    k_n, lbl, stats, _ = cv2.connectedComponentsWithStats((extra & ends).astype(np.uint8), connectivity=8)
    end_cap = C.POLE_REFINE_END_AREA * (2 * w_end) ** 2
    for k in range(1, k_n):
        comp = lbl == k
        area = int(stats[k, cv2.CC_STAT_AREA])
        ys, xs = np.nonzero(comp)
        cols = np.unique(xs)
        tex_end = float(np.median(tex[cols]))
        lat = hw + 1 + (w_end if tex_end <= C.POLE_REFINE_END_TEX else w_max)
        comp &= np.abs(v) <= lat
        # mỏ neo: chạm phần thân đã giữ, hoặc chạm lõi dải thuộc một đoạn thân gậy dài >= min_run tính
        # ngoài các cột của chính khối (khối nằm trên đoạn dải thò vào nền không tự làm mỏ neo được)
        anchored = bool((comp & keep_d).any())
        if not anchored:
            for rid in np.unique(run_id[np.nonzero((comp & core_d).any(axis=0))[0]]):
                if rid > 0 and np.sum((run_id == rid) & ~np.isin(np.arange(ncol), cols)) >= min_run:
                    anchored = True
                    break
        ok = area <= end_cap and anchored
        dbg.setdefault("comps", []).append(dict(end=area, a_rel=round(area / (w_end ** 2), 2),
                                                t=(int(xs.min()) - band_r, int(xs.max()) - band_r),
                                                s=(int(ys.min()) - R, int(ys.max()) - R),
                                                tex=round(tex_end, 2), L=round(L), w_end=round(w_end, 1), ok=ok))
        if ok:
            keep |= comp

    mask_r = keep | strip
    strip_area = max(1.0, float(strip.sum()))
    added = float(keep.sum())
    ratio = (strip_area + added) / strip_area
    dbg.update(mask_r=mask_r, ratio=ratio)
    if ratio > C.POLE_REFINE_AREA or added > C.POLE_REFINE_AREA_H * h * h:
        return None                                         # thêm quá nhiều: nghi tràn
    if added == 0:
        return None                                         # không có gì thêm: dùng dải gậy như cũ

    # đưa mask về ảnh: crop = bbox của vùng tìm
    corners = np.array([[0, R - band_r, 1], [ncol, R - band_r, 1],
                        [0, R + band_r + 1, 1], [ncol, R + band_r + 1, 1]], np.float64) @ M.T
    x1, y1 = np.floor(corners.min(axis=0)).astype(int) - 1
    x2, y2 = np.ceil(corners.max(axis=0)).astype(int) + 2
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    Mc = M.copy()
    Mc[:, 2] -= (x1, y1)
    back = cv2.warpAffine(mask_r.astype(np.uint8) * 255, Mc, (x2 - x1, y2 - y1),
                          flags=cv2.INTER_LINEAR, borderValue=0) >= 128
    return PoleMask(box=(x1, y1, x2, y2), mask=back, ratio=ratio)
