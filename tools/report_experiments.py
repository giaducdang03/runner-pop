"""
Thực nghiệm cho báo cáo cuối kỳ: so sánh engine, ablation các khối xử lý, độ chính xác đọc số BIB.

Tập ảnh: 40 ảnh lấy ngẫu nhiên (hạt giống cố định) từ hai bộ ảnh sự kiện: 20 ảnh Mù Cang Chải trail và
20 ảnh HD Bank marathon; danh sách ghi ra eval_set.txt để tái lập.
Tham chiếu (chuẩn tương đối, KHÔNG phải nhãn tay): alpha của BiRefNet chạy toàn ảnh (FULLRES=1).
Độ đo: IoU (alpha nhị phân hoá tại 0.5), Dice, SAD (quy ước matting, alpha thang 0..255), thời gian.

  python tools/report_experiments.py seg --out ../final_report/results
  python tools/report_experiments.py bib --out ../final_report/results --labels ../final_report/results/bib_labels.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest import mock

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_DIRS = ["../dataset/mu_cang_chai_2026_resized", "../dataset/hd_bank_green_marathon_2026_resized"]
EXTS = {".jpg", ".jpeg", ".png"}


def eval_images(dirs, per_dir: int, seed: int):
    """per_dir ảnh ngẫu nhiên (seed cố định) từ mỗi thư mục, sắp theo (thư mục, tên) -> danh sách tái lập được."""
    import random
    out = []
    for d in dirs:
        files = sorted(p for p in Path(d).iterdir() if p.suffix.lower() in EXTS)
        out += sorted(random.Random(seed).sample(files, min(per_dir, len(files))), key=lambda p: p.name)
    return out


# --------------------------------------------------------------------------- độ đo
def _bin(a: np.ndarray) -> np.ndarray:
    return np.asarray(a) >= 127.5


def iou(a, b) -> float:
    ma, mb = _bin(a), _bin(b)
    u = np.logical_or(ma, mb).sum()
    return 1.0 if u == 0 else float(np.logical_and(ma, mb).sum() / u)


def dice(a, b) -> float:
    ma, mb = _bin(a), _bin(b)
    s = ma.sum() + mb.sum()
    return 1.0 if s == 0 else float(2.0 * np.logical_and(ma, mb).sum() / s)


def sad(a, b) -> float:
    return float(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).sum() / 255.0)


# --------------------------------------------------------------------------- cấu hình ablation
@contextmanager
def patched(*patches):
    """Áp nhiều mock.patch cùng lúc."""
    with ExitStack() as st:
        for p in patches:
            st.enter_context(p)
        yield


def _bilinear_upsample(alpha_work, roi, guide_rgb_f, unknown_work=None):
    """Thay guided-filter upsample bằng resize song tuyến tính rồi dán vào ảnh đầy đủ."""
    import cv2
    w, h = roi.full_size
    up = cv2.resize(alpha_work.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return roi.paste(up.astype(np.float32), fill=0.0)


def build_configs():
    """Danh sách (tên, engine, tham số). Báo cáo chỉ dùng hai engine: focus (mặt phẳng nét) và birefnet."""
    M = "backend.pipeline."
    ident_upsample = lambda: patched(mock.patch(M + "matting.upsample_alpha", _bilinear_upsample))
    cfgs = [
        # --- so sánh engine (mục 4.3) ---
        ("focus_laplacian", "focus", dict()),                     # mặc định của ứng dụng
        ("focus_dct", "focus", dict(patch=lambda: patched(
            mock.patch("backend.pipeline.config.FOCUS_METHOD", "freq")))),
        ("focus_hybrid", "focus", dict(patch=lambda: patched(
            mock.patch("backend.pipeline.config.FOCUS_METHOD", "hybrid")))),
        ("birefnet", "birefnet", dict()),
        # --- ablation trên engine focus (mục 4.4) ---
        ("focus_fullres", "focus", dict(full=True)),
        ("focus_no_grabcut", "focus", dict(patch=lambda: patched(
            mock.patch(M + "segment.focus.refine_mask_grabcut", lambda rgb, init, box, **k: init)))),
        ("focus_no_guided_upsample", "focus", dict(patch=ident_upsample)),
        ("focus_no_tighten", "focus", dict(patch=lambda: patched(
            mock.patch(M + "matting.tighten_alpha", lambda a, lo=None, hi=None: a)))),
        ("focus_band_0.005", "focus", dict(band=0.005)),
        ("focus_band_0.01", "focus", dict(band=0.01)),
        ("focus_band_0.04", "focus", dict(band=0.04)),
        ("focus_band_0.08", "focus", dict(band=0.08)),
        ("focus_frac_0.15", "focus", dict(focus_frac=0.15)),
        ("focus_frac_0.5", "focus", dict(focus_frac=0.5)),
        # --- ablation trên engine birefnet ---
        ("birefnet_no_gate", "birefnet", dict(patch=lambda: patched(
            mock.patch(M + "segment.birefnet.gate_to_person", lambda alpha, init, box, excl=None, **k: alpha)))),
        ("birefnet_no_occlusion", "birefnet", dict(patch=lambda: patched(
            mock.patch(M + "segment.birefnet.exclusion_mask",
                       lambda scene_w, person_w, init: np.zeros(scene_w.rgb.shape[:2], bool))))),
        ("birefnet_no_guided_upsample", "birefnet", dict(patch=ident_upsample)),
    ]
    return cfgs


# --------------------------------------------------------------------------- chạy
def run_one(scene, person, engine, device="auto", full=False, band=0.02, focus_frac=0.3, patch=None):
    from backend import pipeline as P
    from backend.pipeline.cache import StageCache
    from tools.common import fullres
    params = P.SegParams(band_frac=band, focus_frac=focus_frac, cache=StageCache(), device_mode=device)
    ctx = patch() if patch else patched()
    with ctx, fullres(full):
        t0 = time.perf_counter()
        cut = P.build_cutout(scene, person, engine, params, trace=P.Trace())
        dt = time.perf_counter() - t0
    return cut.alpha.astype(np.float32) * 255.0, dt


def cmd_seg(a) -> int:
    from tools.common import load_scene
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfgs = build_configs()
    if a.only:
        keep = set(a.only.split(","))
        cfgs = [c for c in cfgs if c[0] in keep]
    suffix = f"_{a.tag}" if a.tag else ""
    rows, warmed = [], set()
    imgs = eval_images(a.dirs, a.per_dir, a.seed)
    (out / "eval_set.txt").write_text("\n".join(f"{p.parent.name}/{p.name}" for p in imgs), encoding="utf-8")
    for k, p in enumerate(imgs, 1):
        sc, person = load_scene(p)
        n_people = len(sc.persons)
        if person is None:
            print(f"[{k}/{len(imgs)}] {p.name}: không phát hiện người -> bỏ", flush=True)
            rows.append(dict(image=p.name, config="(no_person)", persons=0))
            continue
        if "ref" not in warmed:
            run_one(sc, person, "birefnet", full=True)
            warmed.add("ref")
        ref, t_ref = run_one(sc, person, "birefnet", full=True)
        rows.append(dict(image=p.name, config="reference_birefnet_fullres", persons=n_people, time_s=round(t_ref, 3)))
        for name, engine, kw in cfgs:
            if name not in warmed:
                run_one(sc, person, engine, **kw)                    # khởi động nóng: JIT numba, nạp model
                warmed.add(name)
            alpha, dt = run_one(sc, person, engine, **kw)
            rows.append(dict(image=p.name, config=name, persons=n_people, iou=round(iou(alpha, ref), 5),
                             dice=round(dice(alpha, ref), 5), sad=round(sad(alpha, ref), 1),
                             sad_rel=round(sad(alpha, ref) / max(float(ref.sum() / 255.0), 1e-6), 5),
                             time_s=round(dt, 3)))
        print(f"[{k}/{len(imgs)}] {p.name} xong ({n_people} người)", flush=True)
        _dump(out, rows, suffix)
    _dump(out, rows, suffix)
    return 0


def _dump(out: Path, rows, suffix: str = "") -> None:
    cols = ["image", "config", "persons", "iou", "dice", "sad", "sad_rel", "time_s"]
    with open(out / f"seg_per_image{suffix}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    summ = {}
    for r in rows:
        if r.get("config") in (None, "(no_person)"):
            continue
        summ.setdefault(r["config"], []).append(r)
    agg = {}
    for name, rs in summ.items():
        g = lambda key: np.array([x[key] for x in rs if x.get(key, "") != ""], float)
        d = dict(n=len(rs), time_mean=float(g("time_s").mean()), time_median=float(np.median(g("time_s"))))
        if name != "reference_birefnet_fullres":
            io = g("iou")
            d.update(iou_mean=float(io.mean()), iou_p05=float(np.percentile(io, 5)), iou_min=float(io.min()),
                     dice_mean=float(g("dice").mean()), sad_mean=float(g("sad").mean()),
                     sad_rel_mean=float(g("sad_rel").mean()))
            multi = [x for x in rs if int(x["persons"]) >= 2]
            if multi:
                d.update(n_multi=len(multi), iou_multi=float(np.mean([x["iou"] for x in multi])),
                         sad_multi=float(np.mean([x["sad"] for x in multi])))
        agg[name] = d
    (out / f"seg_summary{suffix}.json").write_text(json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- BIB
def bib_variants():
    """(tên, TORSO, dùng_mask, toàn_ảnh)"""
    return [
        ("full_image", (0.0, 1.0), False, True),
        ("person_box", (0.0, 1.0), False, False),
        ("torso_15_65", (0.15, 0.65), False, False),
        ("torso_15_85", (0.15, 0.85), False, False),
        ("torso_15_85_masked", (0.15, 0.85), True, False),
    ]


def cmd_bib(a) -> int:
    from backend.pipeline import bib as B
    from tools.common import load_scene
    labels = {}
    with open(a.labels, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            labels[r["image"]] = r["bib"].strip().upper()
    out = Path(a.out)
    rows = []
    for p in eval_images(a.dirs, a.per_dir, a.seed):
        if p.name not in labels:
            continue
        sc, person = load_scene(p)
        if person is None:
            continue
        h, w = sc.rgb.shape[:2]
        row = dict(image=p.name, gt=labels[p.name])
        for name, torso, use_mask, whole in bib_variants():
            B.TORSO = torso
            box = (0, 0, w, h) if whole else person.box
            t0 = time.perf_counter()
            pred = B.read_bib(sc.rgb, box, person.mask if use_mask else None)
            row[name] = pred
            row[name + "_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        rows.append(row)
        print(p.name, row, flush=True)
    cols = ["image", "gt"] + [c for n, *_ in bib_variants() for c in (n, n + "_ms")]
    with open(out / "bib_per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    summ = {}
    for name, *_ in bib_variants():
        readable = [r for r in rows if r["gt"] not in ("", "?")]
        absent = [r for r in rows if r["gt"] == ""]
        ok = sum(1 for r in readable if r[name] == r["gt"])
        fp = sum(1 for r in absent if r[name] != "")
        summ[name] = dict(n_readable=len(readable), exact=ok, accuracy=ok / max(len(readable), 1),
                          n_no_bib=len(absent), false_reads=fp,
                          ms_mean=float(np.mean([r[name + "_ms"] for r in rows])))
    (out / "bib_summary.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for n in ("seg", "bib"):
        s = sp.add_parser(n)
        s.add_argument("--dirs", nargs="+", default=DEFAULT_DIRS)
        s.add_argument("--per-dir", type=int, default=20)
        s.add_argument("--seed", type=int, default=2026)
        s.add_argument("--out", required=True)
        if n == "seg":
            s.add_argument("--only", default="", help="chỉ chạy các cấu hình này (cách nhau dấu phẩy)")
            s.add_argument("--tag", default="", help="hậu tố tên file kết quả")
        if n == "bib":
            s.add_argument("--labels", required=True)
    a = ap.parse_args(argv)
    for st in (sys.stdout, sys.stderr):
        if hasattr(st, "reconfigure"):
            st.reconfigure(encoding="utf-8")
    return cmd_seg(a) if a.cmd == "seg" else cmd_bib(a)


if __name__ == "__main__":
    raise SystemExit(main())
