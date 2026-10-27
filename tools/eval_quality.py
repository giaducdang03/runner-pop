"""
So chất lượng alpha của một cấu hình với một tham chiếu: mean IoU, mean SAD, p05 IoU, thời gian.

Tập ảnh nghiệm thu chuẩn: dataset/hd_bank_green_marathon_2026_resized, lấy 30 ảnh đầu theo
thứ tự tên đã sắp xếp (deterministic). Chọn MỘT tham chiếu:
  --ref-fullres        so với cùng engine ở FULLRES=1 (đảo C.FULLRES lúc chạy, cùng tiến trình)
  --ref-engine <tên>   so với engine khác
  --ref-device cpu     cùng engine, thiết bị khác

IoU: α nhị phân hoá tại 0.5. SAD = abs(a - a_ref).sum() / 255 (α thang 0..255, quy ước matting);
báo thêm SAD tương đối = SAD / (tổng α tham chiếu / 255).

  python tools/eval_quality.py --dir ../dataset/hd_bank_green_marathon_2026_resized --limit 30 \
      --engine cf --ref-fullres
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU của hai α (thang 0..255) sau khi nhị phân hoá tại 0.5 (=127.5). Cả hai rỗng -> 1.0."""
    ma, mb = np.asarray(a) >= 127.5, np.asarray(b) >= 127.5
    union = np.logical_or(ma, mb).sum()
    return 1.0 if union == 0 else float(np.logical_and(ma, mb).sum() / union)


def sad(a: np.ndarray, b: np.ndarray) -> float:
    """Sum of absolute differences theo quy ước matting: abs(a - b).sum() / 255 (α thang 0..255)."""
    return float(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).sum() / 255.0)


def _a255(cutout) -> np.ndarray:
    return cutout.alpha.astype(np.float32) * 255.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--limit", type=int, default=30, help="số ảnh đầu (theo tên đã sắp xếp)")
    ap.add_argument("--engine", default="cf")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ref-fullres", action="store_true")
    g.add_argument("--ref-engine")
    g.add_argument("--ref-device", choices=["auto", "cpu", "gpu"])
    a = ap.parse_args(argv)
    for st in (sys.stdout, sys.stderr):              # Windows: stdout chuyển hướng mặc định cp1252
        if hasattr(st, "reconfigure"):
            st.reconfigure(encoding="utf-8")

    from tools.common import list_images, load_scene, run_engine

    def ref_kwargs():
        if a.ref_fullres:
            return dict(engine=a.engine, device=a.device, full=True)
        if a.ref_engine:
            return dict(engine=a.ref_engine, device=a.device, full=False)
        return dict(engine=a.engine, device=a.ref_device, full=False)

    ious, sads, rels, t_test, t_ref = [], [], [], [], []
    first = True
    for p in list_images(a.dir, a.limit):
        sc, person = load_scene(p)
        if person is None:
            print(f"(bỏ {p.name}: không phát hiện người)", file=sys.stderr)
            continue
        if first:                                    # warm-up (JIT numba, nạp model) ngoài vùng đo
            run_engine(sc, person, a.engine, a.device)
            run_engine(sc, person, **ref_kwargs())
            first = False
        c, dt, _ = run_engine(sc, person, a.engine, a.device)
        r, dr, _ = run_engine(sc, person, **ref_kwargs())
        A, R = _a255(c), _a255(r)
        ious.append(iou(A, R))
        s = sad(A, R)
        sads.append(s)
        rels.append(s / max(float(R.sum() / 255.0), 1e-6))
        t_test.append(dt)
        t_ref.append(dr)
        print(f"{p.name}: IoU={ious[-1]:.4f} SAD={s:.1f} t={dt:.2f}s ref_t={dr:.2f}s", file=sys.stderr)

    if not ious:
        print("NOT RUN: không có ảnh nào phát hiện được người")
        return 1
    ref_name = ("FULLRES=1" if a.ref_fullres else f"engine {a.ref_engine}" if a.ref_engine
                else f"device {a.ref_device}")
    print(f"engine={a.engine} device={a.device} ref={ref_name} n={len(ious)}")
    print(f"mean IoU   : {np.mean(ious):.4f}")
    print(f"p05 IoU    : {np.percentile(ious, 5):.4f}")
    print(f"min IoU    : {np.min(ious):.4f}")
    print(f"mean SAD   : {np.mean(sads):.1f}")
    print(f"mean SAD / tổng α tham chiếu : {np.mean(rels) * 100:.2f}%")
    print(f"thời gian trung bình: test {np.mean(t_test):.3f}s, ref {np.mean(t_ref):.3f}s "
          f"-> ref/test = {np.mean(t_ref) / np.mean(t_test):.2f}x "
          f"(median {np.median(t_ref) / np.median(t_test):.2f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
