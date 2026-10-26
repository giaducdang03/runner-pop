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
        self.cell(0, 5, f"Color Pop Runner · trang {self.page_no()}", align="C")
        self.set_text_color(0)

    # ---- khối văn bản
    def h1(self, text: str):
        self.set_font("vn", "B", 20)
        self.cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")

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
                metrics.append((label_lines, note_lines, img_h))
            row_h = max(4 + len(m[0]) * 4.4 + m[2] + 2 + len(m[1]) * 3.8 + 3 for m in metrics) + 2
            self.ensure(row_h + 2)
            y0 = self.get_y()
            for i, (s, (ll, nl, ih)) in enumerate(zip(row, metrics)):
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
                y_img = y0 + 4.5 + len(ll) * 4.4 + 1
                if s.image:
                    self.image(io.BytesIO(s.image), x=x + 2, y=y_img, w=cw - 4)
                self.set_xy(x + 2, y_img + ih + 1.5)
                self.set_font("vn", "", 8)
                self.set_text_color(90)
                self.multi_cell(cw - 4, 3.8, s.note or "", new_x="LEFT", new_y="NEXT")
                self.set_text_color(0)
            self.set_xy(self.l_margin, y0 + row_h + 3)


def build_report(rgb: np.ndarray, scene: P.Scene, cutout: Optional[P.Cutout], meta: dict,
                 intensity: float = 1.0, engine_labels: Optional[dict] = None) -> bytes:
    pdf = _Report()
    pdf.add_page()
    pdf.h1("Color Pop Runner - Báo cáo xử lý")
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
            ("Độ rộng vùng viền", f"{meta.get('band', 0):.1%} kích thước người"),
            *([("Ngưỡng độ nét", f"{meta.get('focus', 0):.0%}")] if method == "focus" else []),
            ("Cường độ nền đen trắng", f"{intensity:.0%}"),
            ("Thời gian", f"phát hiện {detect_s:.1f}s · tách nền {seg_s:.1f}s"),
        ])
    else:
        pdf.kv([("Trạng thái", "Chưa tách nền")])

    pdf.h2("3. Pipeline 1 - Tiền xử lý & phát hiện người", need=80)
    pdf.stage_grid(scene.trace.stages)

    if cutout and cutout.trace:
        pdf.h2(f"4. Pipeline 2 - Tách nền ({labels.get(method, method)})", need=80)
        pdf.stage_grid(cutout.trace.stages)

        pdf.add_page()
        pdf.h2("5. Kết quả: trước / sau")
        after = P.compose(cutout, intensity)
        half = (pdf.content_w - 6) / 2
        y0 = pdf.get_y()
        ih = half * h / w
        pdf.image(io.BytesIO(_jpeg(rgb)), x=pdf.l_margin, y=y0, w=half)
        pdf.image(io.BytesIO(_jpeg(after)), x=pdf.l_margin + half + 6, y=y0, w=half)
        pdf.set_xy(pdf.l_margin, y0 + ih + 1)
        pdf.set_font("vn", "", 9)
        pdf.cell(half, 5, "Trước (ảnh gốc)", align="C")
        pdf.set_x(pdf.l_margin + half + 6)
        pdf.cell(half, 5, "Sau (giữ màu người, nền đen trắng)", align="C")
        pdf.set_y(y0 + ih + 8)

        pdf.ensure(half * h / w + 14)
        pdf.set_font("vn", "B", 10)
        pdf.cell(0, 6, "Alpha matte", new_x="LMARGIN", new_y="NEXT")
        alpha = P.alpha_to_png(cutout.alpha)
        pdf.image(io.BytesIO(_jpeg(np.repeat(alpha[..., None], 3, axis=2))), x=pdf.l_margin, w=half)

    return bytes(pdf.output())
