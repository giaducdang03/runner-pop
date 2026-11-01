from backend import pipeline as P
from backend.pipeline.trace import Stage
from backend.report import with_filter_compose_stage


def test_compose_stage_shows_selected_filter(make_cutout):
    c = make_cutout()
    stages = [Stage("matte", "Alpha", 0.1), Stage("compose", "Ghép: người giữ màu, nền đen trắng", 0.02, "cũ", b"old")]
    out = with_filter_compose_stage(stages, c, "sepia", "#808080", 0.5)
    assert out[0] is stages[0]
    assert "Sepia" in out[1].label and "đen trắng" not in out[1].label
    assert "50%" in out[1].note and out[1].image and out[1].image != b"old"
    assert out[1].seconds == 0.02 and stages[1].label.startswith("Ghép: người giữ màu, nền đen")   # không sửa bản gốc


def test_other_stages_untouched(make_cutout):
    stages = [Stage("matte", "Alpha", 0.1)]
    assert with_filter_compose_stage(stages, make_cutout(), "bw", "#808080", 1.0) == stages


def _report_text(bib, meta_bib=""):
    import numpy as np
    from backend.pipeline.schema import Cutout, Person, Scene
    from backend.pipeline.trace import Trace
    from backend.report import build_report
    fitz = __import__("pytest").importorskip("fitz")
    h, w = 120, 160
    img = np.zeros((h, w, 3), np.float32); img[..., 1] = 1.0
    al = np.zeros((h, w), np.float32); img[30:90, 50:100] = (1, 0, 0); al[30:90, 50:100] = 1
    c = Cutout(image=img, alpha=al, foreground=img, background=img, trimap=(al * 255).astype(np.uint8), trace=Trace())
    rgb = (img * 255).astype(np.uint8)
    scene = Scene(rgb=rgb, persons=[Person(0, (50, 30, 100, 90), 0.9)])
    meta = {"method": "cf", "person": 0, "band": 0.02, "bib": meta_bib}
    data = build_report(rgb, scene, c, meta, 1.0, {"cf": "cf"}, bib=bib)
    return "".join(p.get_text() for p in fitz.open(stream=data, filetype="pdf"))


def test_report_results_show_bib():
    assert "F5215" in _report_text("F5215")


def test_report_bib_falls_back_to_meta_and_handles_missing():
    assert "M5220" in _report_text(None, meta_bib="M5220")
    assert "không đọc được" in _report_text("")
