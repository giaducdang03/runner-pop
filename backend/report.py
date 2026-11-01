"""
Xuất báo cáo PDF: ảnh đầu vào, chất lượng, cài đặt, pipeline 1 (tiền xử lý + phát hiện),
pipeline 2 (tách nền) với ảnh trung gian từng bước, và kết quả trước / sau.

Cần font TTF có tiếng Việt (tự tìm Arial / Segoe UI / DejaVu; đặt REPORT_FONT=/đường/dẫn/font.ttf để chỉ định).
"""
from __future__ import annotations

import io
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from fpdf import FPDF
from PIL import Image

from . import pipeline as P
from .pipeline.trace import Stage

# Tên ứng dụng hiện trong tiêu đề và chân trang PDF (giữ khớp với tiêu đề giao diện Streamlit)
APP_TITLE = "RunnerPop – Image Preprocessing Optimization for AI-based Runner Segmentation"

_FONT_CANDIDATES: List[Tuple[str, str]] = [
    (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"),
    (r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\segoeuib.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial Bold.ttf"),
]


def _find_fonts() -> Tuple[str, str]:
    env = os.environ.get("REPORT_FONT")
    if env and Path(env).exists():
        return env, env
    for regular, bold in _FONT_CANDIDATES:
        if Path(regular).exists():
            return regular, bold if Path(bold).exists() else regular
    raise RuntimeError("Không tìm thấy font TTF hỗ trợ tiếng Việt. Đặt biến môi trường REPORT_FONT=/đường/dẫn/font.ttf")


def _jpeg(rgb: np.ndarray, max_side: int = 1400, q: int = 90) -> bytes:
    h, w = rgb.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return buf.tobytes()


class _Report(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        regular, bold = _find_fonts()
        self.add_font("vn", "", regular)
        self.add_font("vn", "B", bold)
        self.set_auto_page_break(False)
        self.set_margins(12, 12, 12)
        self.content_w = 210 - 24
        self.bottom = 297 - 14

    def footer(self):
        self.set_y(-10)
        self.set_font("vn", "", 8)
        self.set_text_color(130)
        self.cell(0, 5, f"{APP_TITLE} · trang {self.page_no()}", align="C")
        self.set_text_color(0)

    # ---- khối văn bản
    def h1(self, text: str, subtitle: str = ""):
        # tiêu đề dài -> tự xuống dòng (multi_cell) ở cỡ chữ vừa phải, không tràn khỏi trang
        self.set_font("vn", "B", 15)
        self.multi_cell(0, 7.5, text, new_x="LMARGIN", new_y="NEXT")
        if subtitle:
            self.set_font("vn", "", 12)
            self.cell(0, 7, subtitle, new_x="LMARGIN", new_y="NEXT")

    def h2(self, text: str, need: float = 14):
        self.ensure(need)               # need = chỗ tối thiểu cho tiêu đề + nội dung đầu tiên (tránh tiêu đề mồ côi cuối trang)
        self.ln(3)
        self.set_font("vn", "B", 13)
        self.set_fill_color(235, 238, 245)
        self.cell(0, 8, "  " + text, fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def kv(self, rows: List[Tuple[str, str]]):
        for k, v in rows:
            self.ensure(6)
            self.set_font("vn", "B", 9.5)
            self.cell(48, 5.5, k)
            self.set_font("vn", "", 9.5)
            self.multi_cell(self.content_w - 48, 5.5, v, new_x="LMARGIN", new_y="NEXT")

    def ensure(self, h: float):
        if self.get_y() + h > self.bottom:
            self.add_page()

    # ---- lưới thẻ các bước pipeline
    def stage_grid(self, stages: List[Stage], cols: int = 3):
        gap = 5
        cw = (self.content_w - gap * (cols - 1)) / cols
        for start in range(0, len(stages), cols):
            row = stages[start:start + cols]
            # đo chiều cao từng thẻ để cả hàng cao bằng nhau
            metrics = []
            for s in row:
                self.set_font("vn", "B", 9)
                label_lines = self.multi_cell(cw - 4, 4.4, s.label, dry_run=True, output="LINES")
                self.set_font("vn", "", 8)
                note_lines = self.multi_cell(cw - 4, 3.8, s.note or " ", dry_run=True, output="LINES")
                img_h = 0.0
                if s.image:
                    iw, ih = Image.open(io.BytesIO(s.image)).size
                    img_h = (cw - 4) * ih / iw
                dev_h = 3.6 if s.device else 0.0
                metrics.append((label_lines, note_lines, img_h, dev_h))
            row_h = max(4 + len(m[0]) * 4.4 + m[3] + m[2] + 2 + len(m[1]) * 3.8 + 3 for m in metrics) + 2
            self.ensure(row_h + 2)
            y0 = self.get_y()
            for i, (s, (ll, nl, ih, dh)) in enumerate(zip(row, metrics)):
                x = self.l_margin + i * (cw + gap)
                self.set_draw_color(190)
                self.rect(x, y0, cw, row_h)
                self.set_xy(x + 2, y0 + 1)
                self.set_font("vn", "", 7.5)
                self.set_text_color(120)
                self.cell(cw - 4, 3.5, f"BƯỚC {start + i + 1} · {s.seconds:.2f}s")
                self.set_text_color(0)
                self.set_xy(x + 2, y0 + 4.5)
                self.set_font("vn", "B", 9)
                self.multi_cell(cw - 4, 4.4, s.label, new_x="LEFT", new_y="NEXT")
                y_dev = y0 + 4.5 + len(ll) * 4.4
                if s.device:
                    self.set_xy(x + 2, y_dev)
                    self.set_font("vn", "", 7.5)
                    self.set_text_color(120)
                    self.cell(cw - 4, 3.6, fit_text(f"Thiết bị: {s.device}", cw - 4, self.get_string_width))
                    self.set_text_color(0)
                y_img = y_dev + dh + 1
                if s.image:
                    self.image(io.BytesIO(s.image), x=x + 2, y=y_img, w=cw - 4)
                self.set_xy(x + 2, y_img + ih + 1.5)
                self.set_font("vn", "", 8)
                self.set_text_color(90)
                self.multi_cell(cw - 4, 3.8, s.note or "", new_x="LEFT", new_y="NEXT")
                self.set_text_color(0)
            self.set_xy(self.l_margin, y0 + row_h + 3)


def roi_note(stages) -> str:
    """Ghi chú của stage `roi` trong trace tách nền (ROI, % ảnh, kích thước làm việc); thiếu thì "-"."""
    for st in stages or []:
        if st.key == "roi":
            return st.note or "-"
    return "-"


def fit_text(text: str, max_w: float, measure) -> str:
    """Cắt `text` cho vừa `max_w` (đo bằng `measure(chuỗi) -> độ rộng`), thêm "…" khi bị cắt."""
    if not text or measure(text) <= max_w:
        return text
    lo, hi = 0, len(text)          # tìm số ký tự lớn nhất sao cho text[:n] + "…" vừa
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if measure(text[:mid] + "…") <= max_w:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + "…"


_HEAVY = (("yolo", "YOLO"), ("birefnet", "BiRefNet"), ("matting", "Matting"),
          ("rvm", "RVM"), ("modnet", "MODNet"))


_LIBS = (("numpy", "numpy"), ("opencv", "opencv-python"), ("pymatting", "PyMatting"), ("ultralytics", "ultralytics"),
         ("torch", "torch"), ("onnxruntime", "onnxruntime"), ("onnxruntime-gpu", "onnxruntime-gpu"),
         ("openvino", "openvino"), ("rembg", "rembg"), ("fpdf2", "fpdf2"))


def _lib_versions() -> str:
    """Phiên bản các thư viện chính đang cài (chỉ đọc metadata, không import nặng); thiếu thì bỏ qua."""
    from importlib import metadata
    out = []
    for label, dist in _LIBS:
        try:
            out.append(f"{label} {metadata.version(dist)}")
        except Exception:
            continue
    return " · ".join(out) or "-"


def runtime_config_rows() -> List[Tuple[str, str]]:
    """Cấu hình backend ĐANG chạy (đọc lúc xuất báo cáo): thiết bị/tăng tốc, tham số xử lý, model, máy & thư viện."""
    import platform
    from .pipeline import accel
    from .pipeline import config as C

    dev = accel.resolve()
    backends = accel.available_backends()
    return [
        ("Thiết bị mặc định", accel.device_label(dev)),
        ("Backend GPU khả dụng", ", ".join(f"{k}: {v}" for k, v in backends.items()) or "không có (chạy CPU)"),
        ("ACCEL_BACKEND / DEVICE", f"{C.ACCEL_BACKEND} / {C.ACCEL_DEVICE} (chỉ số GPU {C.ACCEL_DEVICE_INDEX}, kiểu số {C.ACCEL_DTYPE})"),
        ("Bộ nhớ model tối đa", f"{C.ACCEL_MEM_BUDGET_MB} MB"),
        ("Cache model", str(C.ACCEL_CACHE_DIR)),
        ("Chế độ vùng xử lý", "toàn ảnh (FULLRES=1)" if C.FULLRES else "ROI quanh người"),
        ("ROI", f"nở {C.ROI_MARGIN:.0%} (tối thiểu {C.ROI_MIN_PX} px), cạnh dài làm việc ≤ {C.WORK_MAX_SIDE} px"),
        ("Guided filter", f"bán kính {C.GUIDED_RADIUS}, eps {C.GUIDED_EPS:g}"),
        ("Ảnh đầu vào", f"thu nhỏ về cạnh dài ≤ {C.MAX_SIDE} px"),
        ("Engine mặc định", C.DEFAULT_ENGINE),
        ("Model phát hiện", f"{C.YOLO_WEIGHTS} (imgsz {C.YOLO_IMGSZ})"),
        ("Model BiRefNet", f"{C.BIREFNET_MODEL} (đầu vào {C.BIREFNET_SIZE}×{C.BIREFNET_SIZE})"),
        ("Hệ điều hành", platform.platform()),
        ("CPU", platform.processor() or platform.machine() or "-"),
        ("Python", platform.python_version()),
        ("Thư viện", _lib_versions()),
    ]


def device_summary(*stage_lists) -> str:
    """Tóm tắt thiết bị của các bước nặng, vd "YOLO: CPU · BiRefNet: cuda: X (fp16)"; thiếu thì "-".
    Nhận một hay nhiều danh sách stage; bước lặp lại (thử lại YOLO) lấy lần cuối."""
    found = {}
    for stages in stage_lists:
        for st in stages or []:
            if st.device:
                found[st.key] = st.device
    parts = [f"{label}: {found[key]}" for key, label in _HEAVY if key in found]
    return " · ".join(parts) or "-"


def with_filter_compose_stage(stages: List[Stage], cutout: P.Cutout, bg_filter: str, color: str,
                             intensity: float) -> List[Stage]:
    """Bước "Ghép" được ghi lúc tách nền (chưa biết bộ lọc) -> thay nhãn/ảnh bằng bộ lọc đang áp dụng. Không sửa bản gốc."""
    from dataclasses import replace
    from .pipeline.trace import _encode
    name = P.BG_FILTERS.label(bg_filter) + (f" {color}" if bg_filter == "solid" else "")
    out = []
    for s in stages:
        if s.key == "compose":
            s = replace(s, label=f"Ghép: người giữ màu, nền {name}",
                        note=f"out = α·F + (1−α)·((1−s)·B + s·Bf); Bf = {name}, s = {intensity:.0%}",
                        image=_encode(P.compose(cutout, intensity, bg_filter, color)))
        out.append(s)
    return out


def build_report(rgb: np.ndarray, scene: P.Scene, cutout: Optional[P.Cutout], meta: dict,
                 intensity: float = 1.0, engine_labels: Optional[dict] = None,
                 bg_filter: str = "bw", color: str = "#808080", bib: Optional[str] = None) -> bytes:
    """`bib`: số BIB người dùng đã xác nhận/sửa; None -> dùng số đọc được lúc tách nền (meta["bib"])."""
    pdf = _Report()
    pdf.add_page()
    pdf.h1(APP_TITLE, "Báo cáo xử lý")
    pdf.set_font("vn", "", 9)
    pdf.set_text_color(110)
    pdf.cell(0, 5, datetime.now().strftime("Tạo lúc %H:%M %d/%m/%Y"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0)

    h, w = rgb.shape[:2]
    q = scene.quality or {}
    method = meta.get("method")
    labels = engine_labels or {}
    detect_s = sum(s.seconds for s in scene.trace.stages)
    seg_s = sum(s.seconds for s in cutout.trace.stages) if cutout and cutout.trace else 0.0

    pdf.h2("1. Thông tin ảnh & chất lượng")
    rows = [
        ("Kích thước làm việc", f"{w} × {h} px"),
        ("Chất lượng", (f"sáng {q.get('brightness', 0):.0f}/255 · tương phản {q.get('contrast', 0):.0f} · "
                        f"độ nét {q.get('sharpness', 0):.0f} · nhiễu {q.get('noise', 0):.1f}") if q else "-"),
        ("Vấn đề phát hiện", ", ".join(q.get("flags", [])) or "không có"),
        ("Tiền xử lý (cho phát hiện)", " → ".join(scene.preprocess) or "không"),
        ("Số người phát hiện", f"{len(scene.persons)}" + (f" (phải thử lại {scene.attempt} lần)" if scene.attempt else "")),
        ("Người được chọn", f"#{meta['person']}" if meta.get("person") is not None and meta.get("person") >= 0
         else ("box thủ công" if method else "-")),
    ]
    pdf.kv(rows)

    pdf.h2("2. Cài đặt tách nền")
    if method:
        pdf.kv([
            ("Thuật toán", labels.get(method, method)),
            ("Vùng xử lý", roi_note(cutout.trace.stages if cutout and cutout.trace else None)),
            ("Thiết bị", device_summary(scene.trace.stages, cutout.trace.stages if cutout and cutout.trace else None)),
            ("Độ rộng vùng viền", f"{meta.get('band', 0):.1%} kích thước người"),
            *([("Ngưỡng độ nét", f"{meta.get('focus', 0):.0%}")] if method == "focus" else []),
            ("Bộ lọc nền", P.BG_FILTERS.label(bg_filter) + (f" ({color})" if bg_filter == "solid" else "")),
            ("Cường độ hiệu ứng nền", f"{intensity:.0%}"),
            ("Thời gian", f"phát hiện {detect_s:.1f}s · tách nền {seg_s:.1f}s"),
        ])
    else:
        pdf.kv([("Trạng thái", "Chưa tách nền")])

    pdf.h2("3. Cấu hình đang chạy", need=60)
    try:
        pdf.kv(runtime_config_rows())
    except Exception as e:   # không để việc đọc cấu hình làm hỏng cả báo cáo
        pdf.kv([("Cấu hình", f"không đọc được ({type(e).__name__}: {e})")])

    pdf.h2("4. Pipeline 1 - Tiền xử lý & phát hiện người", need=80)
    pdf.stage_grid(scene.trace.stages)

    if cutout and cutout.trace:
        pdf.h2(f"5. Pipeline 2 - Tách nền ({labels.get(method, method)})", need=80)
        pdf.stage_grid(with_filter_compose_stage(cutout.trace.stages, cutout, bg_filter, color, intensity))

        pdf.add_page()
        pdf.h2("6. Kết quả: trước / sau")
        bib_text = (meta.get("bib", "") if bib is None else bib) or ""
        pdf.set_font("vn", "B", 14)
        pdf.cell(0, 8, f"Số BIB: {bib_text}" if bib_text else "Số BIB: không đọc được", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
        after = P.compose(cutout, intensity, bg_filter, color)
        half = (pdf.content_w - 6) / 2
        y0 = pdf.get_y()
        ih = half * h / w
        pdf.image(io.BytesIO(_jpeg(rgb)), x=pdf.l_margin, y=y0, w=half)
        pdf.image(io.BytesIO(_jpeg(after)), x=pdf.l_margin + half + 6, y=y0, w=half)
        pdf.set_xy(pdf.l_margin, y0 + ih + 1)
        pdf.set_font("vn", "", 9)
        pdf.cell(half, 5, "Trước (ảnh gốc)", align="C")
        pdf.set_x(pdf.l_margin + half + 6)
        pdf.cell(half, 5, f"Sau (giữ màu người, nền {P.BG_FILTERS.label(bg_filter)})", align="C")
        pdf.set_y(y0 + ih + 8)

        pdf.ensure(half * h / w + 14)
        pdf.set_font("vn", "B", 10)
        pdf.cell(0, 6, "Alpha matte", new_x="LMARGIN", new_y="NEXT")
        alpha = P.alpha_to_png(cutout.alpha)
        pdf.image(io.BytesIO(_jpeg(np.repeat(alpha[..., None], 3, axis=2))), x=pdf.l_margin, w=half)

    return bytes(pdf.output())
