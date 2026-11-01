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
