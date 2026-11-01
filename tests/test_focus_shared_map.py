"""Engine focus và chọn người dùng chung một bản đồ nét (FR-123, FR-125)."""
from __future__ import annotations

import numpy as np

from backend.pipeline import config as C
from backend.pipeline import focusmap as fm
from backend.pipeline.cache import StageCache
from backend.pipeline.schema import SegParams
from backend.pipeline.segment import build_cutout
from backend.pipeline.segment import focus as focus_mod
from backend.pipeline.select import score_persons, suggest_person


def _scene(make_scene, shape=(240, 320), box=(120, 60, 190, 200)):
    h, w = shape
    m = np.zeros((h, w), np.uint8)
    m[box[1]:box[3], box[0]:box[2]] = 255
    scene = make_scene(shape, [(box, m)])
    gx = np.linspace(60, 160, w, dtype=np.float32)[None, :]
    rgb = np.stack([np.broadcast_to(gx, (h, w))] * 3, axis=-1).copy()
    rng = np.random.default_rng(0)
    rgb[box[1]:box[3], box[0]:box[2]] = rng.integers(0, 255, (box[3] - box[1], box[2] - box[0], 3))
    scene.rgb = rgb.astype(np.uint8)
    return scene


def test_focus_frac_semantics_unchanged_for_laplacian(make_scene):
    scene = _scene(make_scene)
    p = scene.persons[0]
    m03 = focus_mod.focus_mask(scene.rgb, p, p.mask, 0.3, []) > 127
    m08 = focus_mod.focus_mask(scene.rgb, p, p.mask, 0.8, []) > 127
    assert (m08 & ~m03).sum() == 0


def test_focus_mask_accepts_external_map(make_scene):
    scene = _scene(make_scene)
    p = scene.persons[0]
    smap = fm.sharpness_map(scene.rgb)[0]
    a = focus_mod.focus_mask(scene.rgb, p, p.mask, 0.3, [], smap=smap)
    b = focus_mod.focus_mask(scene.rgb, p, p.mask, 0.3, [])
    assert np.array_equal(a, b)


def test_focus_map_stage_notes_method_and_time(make_scene):
    scene = _scene(make_scene)
    c = build_cutout(scene, scene.persons[0], "focus")
    st = next(s for s in c.trace.stages if s.key == "focus_map")
    assert "phương pháp laplacian" in st.note and "ms" in st.note and st.image is not None


def test_select_accepts_shared_map_and_matches_internal(make_scene):
    scene = _scene(make_scene)
    smap = fm.sharpness_map(scene.rgb)[0]
    assert suggest_person(scene, smap) == suggest_person(scene)
    assert score_persons(scene, smap) == score_persons(scene)


def test_select_prefers_sharp_person(make_scene):
    h, w = 200, 400
    ma, mb = np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)
    ma[50:150, 40:140] = 255
    mb[50:150, 260:360] = 255
    scene = make_scene((h, w), [((40, 50, 140, 150), ma), ((260, 50, 360, 150), mb)])
    rng = np.random.default_rng(1)
    rgb = np.full((h, w, 3), 120, np.uint8)
    rgb[50:150, 40:140] = rng.integers(0, 255, (100, 100, 3))          # nét
    scene.rgb = rgb                                                     # người B phẳng/mờ
    assert suggest_person(scene) == 0


def test_focus_map_computed_once_per_segment_run(monkeypatch, make_scene):
    scene = _scene(make_scene)
    calls = []
    real = focus_mod.sharpness_map
    monkeypatch.setattr(focus_mod, "sharpness_map", lambda *a, **k: calls.append(1) or real(*a, **k))
    cache = StageCache()
    p = scene.persons[0]
    build_cutout(scene, p, "focus", SegParams(cache=cache))
    assert len(calls) == 1
    build_cutout(scene, p, "focus", SegParams(focus_frac=0.6, cache=cache))   # đổi ngưỡng: dùng lại bản đồ
    assert len(calls) == 1


def test_focus_map_cache_key_follows_method(monkeypatch, make_scene):
    scene = _scene(make_scene)
    calls = []
    real = focus_mod.sharpness_map
    monkeypatch.setattr(focus_mod, "sharpness_map", lambda *a, **k: calls.append(1) or real(*a, **k))
    cache = StageCache()
    p = scene.persons[0]
    build_cutout(scene, p, "focus", SegParams(cache=cache))
    monkeypatch.setattr(C, "FOCUS_METHOD", "freq")
    build_cutout(scene, p, "focus", SegParams(cache=cache))
    assert len(calls) == 2


def _two_people(make_scene, texture_b: float):
    """A: người nét (nhiễu mạnh); B: cùng nền, độ nét bằng `texture_b` * A (làm mờ phần còn lại)."""
    import cv2
    h, w = 200, 400
    ma, mb = np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)
    ma[50:160, 60:150] = 255
    mb[50:160, 250:340] = 255
    scene = make_scene((h, w), [((60, 50, 150, 160), ma), ((250, 50, 340, 160), mb)])
    rng = np.random.default_rng(2)
    rgb = np.full((h, w, 3), 110, np.float32)
    rgb[50:160, 60:150] = rng.integers(0, 255, (110, 90, 3))
    patch = rng.integers(0, 255, (110, 90, 3)).astype(np.float32)
    rgb[50:160, 250:340] = patch if texture_b >= 1 else cv2.GaussianBlur(patch, (0, 0), 2.0)
    scene.rgb = rgb.astype(np.uint8)
    return scene


def test_equally_sharp_other_person_is_kept(make_scene):
    scene = _two_people(make_scene, 1.0)
    p = scene.persons[0]
    fm_ = focus_mod.focus_mask(scene.rgb, p, p.mask, 0.3, scene.others(p))
    assert (fm_[scene.persons[1].mask > 127] == 255).mean() > 0.9


def test_visibly_blurred_other_person_is_dropped(make_scene):
    scene = _two_people(make_scene, 0.0)
    p = scene.persons[0]
    fm_ = focus_mod.focus_mask(scene.rgb, p, p.mask, 0.3, scene.others(p))
    assert (fm_[scene.persons[1].mask > 127] == 255).mean() < 0.1


def test_other_person_combined_gate_rejects_marginal_sharpness(monkeypatch, make_scene):
    """E/ref và R/r_ref đều vượt cổng riêng (0.76 / 1.05) nhưng tích 0.80 < FOCUS_OTHER_MIN -> bỏ."""
    scene = _two_people(make_scene, 1.0)
    p = scene.persons[0]
    o = scene.persons[1]
    E = np.full(scene.rgb.shape[:2], 0.1, np.float32)
    E[p.mask > 127] = 1.0
    E[o.mask > 127] = 0.76
    R = np.full(scene.rgb.shape[:2], 1.0, np.float32)
    R[o.mask > 127] = 1.05
    monkeypatch.setattr(focus_mod, "focus_energy", lambda rgb: (E, R))
    out = focus_mod.focus_mask(scene.rgb, p, p.mask, 0.3, scene.others(p), smap=E)
    assert (out[o.mask > 127] == 255).mean() < 0.1


# ---- mask YOLO lẫn người mờ không được phát hiện (viền ngoài mờ) ----

def _leaky_scene(make_scene):
    """Người nét (nhiễu) x 60..150; YOLO mask lẫn thêm dải x 150..190 là người mờ phía sau (phẳng, viền mờ)."""
    import cv2
    h, w = 200, 320
    m = np.zeros((h, w), np.uint8)
    m[40:170, 60:190] = 255
    scene = make_scene((h, w), [((60, 40, 190, 170), m)])
    rng = np.random.default_rng(5)
    rgb = np.full((h, w, 3), 110, np.float32)
    rgb[40:170, 150:190] = (200, 60, 60)                       # người mờ: màu phẳng
    rgb = cv2.GaussianBlur(rgb, (0, 0), 4.0)                   # viền mờ
    rgb[40:170, 60:150] = rng.integers(0, 255, (130, 90, 3))   # người nét
    scene.rgb = rgb.astype(np.uint8)
    return scene


def test_blurry_strip_inside_yolo_mask_is_pruned(make_scene):
    scene = _leaky_scene(make_scene)
    p = scene.persons[0]
    out = focus_mod.focus_mask(scene.rgb, p, p.mask, 0.3, []) > 127
    assert out[60:150, 70:140].mean() > 0.95            # thân nét còn nguyên
    assert out[60:150, 165:185].mean() < 0.3            # dải mờ bị loại


def test_sharp_person_mask_not_pruned(make_scene):
    scene = _scene(make_scene)                          # người nét, mask khớp
    p = scene.persons[0]
    out = focus_mod.focus_mask(scene.rgb, p, p.mask, 0.3, []) > 127
    assert out[p.mask > 127].mean() > 0.99


def test_flat_limb_with_sharp_edges_is_kept(make_scene):
    """Cánh tay phẳng hẹp có viền nét hai bên không bị coi là phần lẫn mờ."""
    h, w = 200, 320
    m = np.zeros((h, w), np.uint8)
    m[40:170, 60:150] = 255
    m[80:100, 150:230] = 255                            # tay phẳng 20 px, dài 80 px
    scene = make_scene((h, w), [((60, 40, 230, 170), m)])
    rng = np.random.default_rng(6)
    rgb = np.full((h, w, 3), 110, np.float32)
    rgb[40:170, 60:150] = rng.integers(0, 255, (130, 90, 3))
    rgb[78:102, 150:232] = rng.integers(0, 255, (24, 82, 3))     # viền nét quanh tay
    rgb[80:100, 150:230] = 180                                   # lòng tay phẳng
    scene.rgb = rgb.astype(np.uint8)
    p = scene.persons[0]
    out = focus_mod.focus_mask(scene.rgb, p, p.mask, 0.3, []) > 127
    assert out[82:98, 160:225].mean() > 0.9
