"""Test độ phân giải làm việc, guided filter và nâng α về độ phân giải gốc (FR-91, FR-92)."""
from __future__ import annotations

import cv2
import numpy as np

from backend.pipeline.roi import Roi, guided_filter, person_roi, upsample_alpha, upsample_rgb

H, W = 1000, 1400


def _big_roi(make_scene):
    scene = make_scene((H, W), [((300, 100, 1100, 900), None)])
    return person_roi(scene, scene.persons[0])


def test_work_side_capped_at_640(make_scene):
    roi = _big_roi(make_scene)
    assert max(roi.work_size) == 640
    assert roi.scale < 1.0


def test_small_roi_is_not_upscaled(make_scene):
    # ROI 120x200 -> scale = 1.0, work_size = (120, 200)
    scene = make_scene((400, 400), [((100, 100, 120, 200), None)])
    roi = Roi(100, 100, 220, 300, 1.0, (400, 400))
    assert roi.scale == 1.0 and roi.work_size == (120, 200)
    small = make_scene((400, 400), [((100, 100, 188, 168), None)])
    r = person_roi(small, small.persons[0], margin=0.0, min_px=16)
    assert r.scale == 1.0 and r.work_size == r.full_size


def test_guided_filter_matches_box_filter_on_constant_guide():
    rng = np.random.default_rng(0)
    src = rng.random((60, 70)).astype(np.float32)
    guide = np.full((60, 70), 0.5, np.float32)
    out = guided_filter(guide, src, 4, 1e-2)
    # guide hằng -> a = 0, b = box(p) -> kết quả = box(box(p)) (công thức He et al. lấy trung bình a, b lần nữa)
    ref = cv2.boxFilter(cv2.boxFilter(src, -1, (9, 9)), -1, (9, 9))
    assert np.allclose(out, ref, atol=1e-3)


def test_guided_filter_handles_3d_src():
    src = np.random.default_rng(1).random((30, 40, 3)).astype(np.float32)
    guide = np.random.default_rng(2).random((30, 40)).astype(np.float32)
    out = guided_filter(guide, src, 3, 1e-3)
    assert out.shape == src.shape and out.dtype == np.float32


def _alpha_setup(make_scene, box=(300, 100, 1100, 900), shape=(H, W)):
    scene = make_scene(shape, [(box, None)])
    roi = person_roi(scene, scene.persons[0])
    ww, wh = roi.work_size
    alpha_work = np.zeros((wh, ww), np.float32)
    guide = np.zeros(shape + (3,), np.float32)
    return roi, alpha_work, guide


def test_upsample_alpha_shape_and_zero_outside_roi(make_scene):
    roi, alpha_work, guide = _alpha_setup(make_scene)
    alpha_work[100:400, 100:400] = 1.0
    a = upsample_alpha(alpha_work, roi, guide)
    assert a.shape == (H, W) and a.dtype == np.float32 and a[0, 0] == 0.0


def test_upsample_alpha_keeps_certain_regions_binary(make_scene):
    roi, alpha_work, guide = _alpha_setup(make_scene)
    ww, wh = roi.work_size
    alpha_work[:, :] = 0.0
    alpha_work[wh // 4: 3 * wh // 4, ww // 4: 3 * ww // 4] = 1.0
    rng = np.random.default_rng(3)
    guide = rng.random(guide.shape).astype(np.float32)
    a = upsample_alpha(alpha_work, roi, guide)
    cx, cy = (roi.x1 + roi.x2) // 2, (roi.y1 + roi.y2) // 2
    fg = a[cy - 20: cy + 20, cx - 20: cx + 20]
    bg = a[roi.y1: roi.y1 + 10, roi.x1: roi.x1 + 10]
    assert set(np.unique(fg)) == {1.0} and set(np.unique(bg)) == {0.0}


def test_upsample_alpha_aligned_for_border_roi(make_scene):
    # ROI chạm biên phải/dưới: tâm khối α sáng phải nằm trong ROI, lệch <= 2 px
    box = (W - 500, H - 600, W, H)
    roi, alpha_work, guide = _alpha_setup(make_scene, box=box)
    assert roi.x2 == W and roi.y2 == H
    ww, wh = roi.work_size
    s = ww / roi.full_size[0]
    # khối sáng 100x100 (px gốc) tại (W-300..W-200, H-300..H-200)
    ex, ey = (W - 250), (H - 250)
    x0, x1 = round((W - 300 - roi.x1) * s), round((W - 200 - roi.x1) * s)
    y0, y1 = round((H - 300 - roi.y1) * s), round((H - 200 - roi.y1) * s)
    alpha_work[y0:y1, x0:x1] = 1.0
    a = upsample_alpha(alpha_work, roi, guide)
    ys, xs = np.nonzero(a > 0.5)
    assert abs(xs.mean() - ex) <= 2 and abs(ys.mean() - ey) <= 2
    assert roi.x1 <= xs.min() and xs.max() < roi.x2


def test_upsample_rgb_keeps_base_outside_roi(make_scene):
    roi, _, guide = _alpha_setup(make_scene)
    ww, wh = roi.work_size
    rgb_work = np.full((wh, ww, 3), 0.7, np.float32)
    base = np.random.default_rng(4).random((H, W, 3)).astype(np.float32)
    out = upsample_rgb(rgb_work, roi, guide, base)
    assert out.shape == (H, W, 3) and out.dtype == np.float32
    assert np.array_equal(out[0, 0], base[0, 0])
    assert abs(float(out[(roi.y1 + roi.y2) // 2, (roi.x1 + roi.x2) // 2, 0]) - 0.7) < 0.05


def _soft_alpha(shape, seed=5):
    """α có dải mềm (giá trị thực sự nằm giữa 0 và 1) ở giữa ảnh."""
    h, w = shape
    a = np.zeros((h, w), np.float32)
    a[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = 1.0
    a = cv2.GaussianBlur(a, (0, 0), 6.0)
    return np.clip(a + np.random.default_rng(seed).normal(0, 0.05, a.shape).astype(np.float32), 0, 1)


def test_upsample_alpha_no_resize_only_pastes():   # R14
    roi = Roi(100, 80, 300, 380, 1.0, (500, 600))
    assert roi.work_size == roi.full_size
    ww, wh = roi.work_size
    alpha_work = _soft_alpha((wh, ww))
    assert ((alpha_work > 0.05) & (alpha_work < 0.95)).sum() > 1000
    guide = np.random.default_rng(6).random((500, 600, 3)).astype(np.float32)
    a = upsample_alpha(alpha_work, roi, guide)
    assert a.shape == (500, 600) and a.dtype == np.float32
    assert np.array_equal(roi.crop(a), alpha_work)
    outside = np.ones(a.shape, bool)
    outside[roi.y1:roi.y2, roi.x1:roi.x2] = False
    assert (a[outside] == 0.0).all()
    # truyền dải unknown cũng không lọc
    a2 = upsample_alpha(alpha_work, roi, guide, unknown_work=(alpha_work > 0.05))
    assert np.array_equal(roi.crop(a2), alpha_work)


def test_upsample_alpha_still_filters_when_downscaled(make_scene):   # R14: guard không tắt filter
    roi, _, _ = _alpha_setup(make_scene)
    assert roi.scale < 1.0
    ww, wh = roi.work_size
    alpha_work = _soft_alpha((wh, ww))
    guide = np.random.default_rng(7).random((H, W, 3)).astype(np.float32)
    a = upsample_alpha(alpha_work, roi, guide)
    plain = cv2.resize(alpha_work, roi.full_size, interpolation=cv2.INTER_LINEAR)
    assert not np.allclose(roi.crop(a), plain, atol=1e-3)


def test_upsample_rgb_no_resize_only_pastes():   # R14
    roi = Roi(100, 80, 300, 380, 1.0, (500, 600))
    ww, wh = roi.work_size
    rgb_work = np.random.default_rng(8).random((wh, ww, 3)).astype(np.float32)
    guide = np.random.default_rng(9).random((500, 600, 3)).astype(np.float32)
    base = np.random.default_rng(10).random((500, 600, 3)).astype(np.float32)
    out = upsample_rgb(rgb_work, roi, guide, base)
    assert out.dtype == np.float32 and out.shape == base.shape
    assert np.array_equal(roi.crop(out), rgb_work)
    assert np.array_equal(out[0, 0], base[0, 0])


# ---- guided filter theo ô: chỉ lọc ô chạm vành viền (nhanh hơn, kết quả giống hệt) ----

def _rand_guide_src(h=300, w=420, seed=21):
    rng = np.random.default_rng(seed)
    guide = rng.random((h, w)).astype(np.float32)
    src = rng.random((h, w, 3)).astype(np.float32)
    return guide, src


def test_guided_filter_masked_equals_full_inside_mask():
    from backend.pipeline.roi import guided_filter_masked
    guide, src = _rand_guide_src()
    mask = np.zeros(guide.shape, bool)
    mask[100:110, 30:380] = True                     # dải mỏng cắt qua nhiều ô
    mask[200:260, 250:262] = True
    full = guided_filter(guide, src, 8, 1e-4)
    part = guided_filter_masked(guide, src, mask, 8, 1e-4, tile=64)
    assert part.dtype == np.float32 and part.shape == src.shape
    assert np.allclose(part[mask], full[mask], atol=1e-5)


def test_guided_filter_masked_leaves_untouched_tiles_as_src():
    from backend.pipeline.roi import guided_filter_masked
    guide, src = _rand_guide_src()
    mask = np.zeros(guide.shape, bool)
    mask[10:20, 10:20] = True
    part = guided_filter_masked(guide, src, mask, 8, 1e-4, tile=64)
    assert np.array_equal(part[200:, 200:], src[200:, 200:])


def test_guided_filter_masked_empty_mask_returns_src():
    from backend.pipeline.roi import guided_filter_masked
    guide, src = _rand_guide_src(80, 90)
    out = guided_filter_masked(guide, src, np.zeros(guide.shape, bool), 8, 1e-4)
    assert np.array_equal(out, src)


def test_upsample_rgb_with_band_matches_full_filter_on_band(make_scene):
    roi, _, guide = _alpha_setup(make_scene)
    ww, wh = roi.work_size
    rgb_work = np.random.default_rng(31).random((wh, ww, 3)).astype(np.float32)
    base = np.zeros((H, W, 3), np.float32)
    band = np.zeros((H, W), bool)
    band[roi.y1 + 50:roi.y1 + 60, roi.x1 + 20:roi.x2 - 20] = True
    full = upsample_rgb(rgb_work, roi, guide, base)
    part = upsample_rgb(rgb_work, roi, guide, base, band=band)
    assert np.allclose(part[band], full[band], atol=1e-5)
    assert np.array_equal(part[0, 0], base[0, 0])        # ngoài ROI vẫn là base


def test_finish_cutout_filters_only_band_tiles(monkeypatch, make_scene):
    """F/B không còn lọc cả ROI: tổng điểm ảnh đưa vào guided filter nhỏ hơn nhiều diện tích ROI."""
    from backend.pipeline import roi as roi_mod
    from backend.pipeline.matting import finish_cutout
    roi, _, _ = _alpha_setup(make_scene)
    ww, wh = roi.work_size
    rgb_f = np.random.default_rng(41).random((H, W, 3)).astype(np.float32)
    tri = np.zeros((wh, ww), np.uint8)
    tri[20:120, 20:120] = 128                    # vành chưa biết nhỏ ở một góc ROI
    tri[40:100, 40:100] = 255
    alpha_work = (tri == 255).astype(np.float32) + 0.5 * (tri == 128)
    real, sizes = roi_mod.guided_filter, []
    monkeypatch.setattr(roi_mod, "guided_filter",
                        lambda g, s, *a, **k: (sizes.append(g.size) if s.ndim == 3 else None) or real(g, s, *a, **k))
    finish_cutout(rgb_f, np.clip(alpha_work, 0, 1), tri, roi)
    roi_px = (roi.x2 - roi.x1) * (roi.y2 - roi.y1)
    rgb_calls = [s for s in sizes if s < roi_px]            # lời gọi theo ô, ảnh RGB (F và B)
    assert rgb_calls and sum(rgb_calls) < 0.5 * roi_px        # F+B cộng lại < nửa ROI (trước: 2 lần ROI)
