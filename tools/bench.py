"""
Đo thời gian từng stage của trace (trung vị + p95) theo phương pháp mục 5.7.

Tập ảnh nghiệm thu chuẩn: dataset/hd_bank_green_marathon_2026_resized, lấy 30 ảnh đầu theo
thứ tự tên đã sắp xếp (deterministic). YOLO chạy 1 lần/ảnh ngoài vùng đo; mỗi lần đo chạy engine
với cache mới. Mỗi (ảnh, engine) chạy `--runs` lần, bỏ `--drop` lần đầu (warm-up) của TỪNG ảnh,
rồi gộp mọi mẫu còn lại để tính trung vị và p95.

  python tools/bench.py --dir ../dataset/hd_bank_green_marathon_2026_resized --limit 30 \
      --engine cf,knn,birefnet,focus --runs 10 --drop 2 [--fullres]
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

if __package__ in (None, ""):                       # chạy trực tiếp: python tools/bench.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def summarize(times: Sequence[float], drop: int = 2) -> Dict[str, float]:
    """Bỏ `drop` lần đầu (warm-up) rồi tính trung vị và p95. Nếu bỏ hết thì giữ lại toàn bộ."""
    t = list(times)[drop:] or list(times)
    return {"median": float(np.median(t)), "p95": float(np.percentile(t, 95)), "n": len(t)}


def _table(engine: str, stage_times: Dict[str, List[float]], labels: Dict[str, str],
           totals: List[float]) -> str:
    rows = [f"### engine `{engine}`", "", "| stage | median (s) | p95 (s) | n |", "|---|---:|---:|---:|"]
    for k, v in stage_times.items():
        s = summarize(v, 0)
        rows.append(f"| {labels[k]} (`{k}`) | {s['median']:.3f} | {s['p95']:.3f} | {s['n']} |")
    s = summarize(totals, 0)
    rows.append(f"| **tổng** | **{s['median']:.3f}** | **{s['p95']:.3f}** | {s['n']} |")
    return "\n".join(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--limit", type=int, default=30, help="số ảnh đầu (theo tên đã sắp xếp)")
    ap.add_argument("--engine", default="cf,knn,birefnet,focus")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--drop", type=int, default=2)
    ap.add_argument("--fullres", action="store_true", help="đo với FULLRES=1 (không ROI) để so sánh")
    a = ap.parse_args(argv)
    for st in (sys.stdout, sys.stderr):              # Windows: stdout chuyển hướng mặc định cp1252
        if hasattr(st, "reconfigure"):
            st.reconfigure(encoding="utf-8")

    from tools.common import list_images, load_scene, run_engine
    from backend import pipeline as P

    images = list_images(a.dir, a.limit)
    print(f"# bench: {len(images)} ảnh, runs={a.runs}, drop={a.drop}, device={a.device}, "
          f"fullres={a.fullres}\n")
    scenes = []
    for p in images:
        sc, person = load_scene(p)
        if person is None:
            print(f"(bỏ {p.name}: không phát hiện người)", file=sys.stderr)
            continue
        scenes.append((sc, person))

    for engine in [e.strip() for e in a.engine.split(",") if e.strip()]:
        if engine not in P.engine_names():
            print(f"### engine `{engine}`\n\nNOT RUN: engine không tồn tại\n")
            continue
        stage_times: Dict[str, List[float]] = defaultdict(list)
        labels: Dict[str, str] = {}
        totals: List[float] = []
        try:
            for sc, person in scenes:
                per_img = defaultdict(list)
                tot = []
                for _ in range(a.runs):
                    _, dt, tr = run_engine(sc, person, engine, a.device, a.fullres)
                    tot.append(dt)
                    for st in tr.stages:
                        per_img[st.key].append(st.seconds)
                        labels[st.key] = st.label
                totals += tot[a.drop:] or tot
                for k, v in per_img.items():
                    stage_times[k] += v[a.drop:] or v
        except Exception as e:                       # thiếu trọng số v.v.
            print(f"### engine `{engine}`\n\nNOT RUN: {type(e).__name__}: {e}\n")
            continue
        print(_table(engine, stage_times, labels, totals) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
