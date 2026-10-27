# Kế hoạch triển khai: ROI + gia tốc Intel Arc + engine nhẹ + bản đồ nét tần số

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Giảm thời gian tách nền xuống mức dùng được bằng cách chỉ xử lý ROI quanh người ở độ phân giải thấp, thêm lớp thiết bị đa backend (Intel Arc/iGPU qua OpenVINO, NVIDIA qua CUDA, Mac Apple Silicon qua MPS/CoreML; chọn bằng cấu hình khi chạy app, luôn có đường rơi về CPU), thêm engine nhẹ `rvm`/`modnet`, và thay bản đồ độ nét Laplacian bằng bản đồ DCT theo miền tần số.

**Architecture:** Bốn khối độc lập cắm vào pipeline hiện có mà không đổi kiến trúc: `pipeline/roi.py` (cắt ROI + độ phân giải làm việc + guided-filter upsample) thành khối dùng chung mà mọi engine gọi; `pipeline/accel.py` là điểm duy nhất chọn backend/thiết bị và nạp model (mỗi backend là một mục trong bảng `BACKENDS`, thêm backend mới không phải sửa engine); `segment/light.py` là engine mới đăng ký qua `@ENGINES.register`; `pipeline/focusmap.py` tách phần đo nét ra khỏi engine `focus` để `select.py` dùng lại. Mọi thay đổi giữ nguyên hợp đồng `Cutout` (α, F, B, trimap ở **độ phân giải ảnh làm việc đầy đủ**) nên frontend, `compose`, và báo cáo PDF không phải biết ROI tồn tại.

**Tech Stack:** Python 3.13, NumPy 2.5, OpenCV 5.0 (bản `opencv-python`, **không** có `ximgproc` → guided filter tự viết bằng `cv2.boxFilter`), pymatting 1.1, ultralytics 8.4, rembg 2.0 + onnxruntime 1.31, tuỳ backend (đều là gói tuỳ chọn): OpenVINO ≥ 2025.0 (Intel), onnxruntime-gpu + torch CUDA (NVIDIA), onnxruntime có CoreML + torch MPS (Mac), FastAPI, Streamlit, fpdf2, pytest (mới thêm).

**Spec:** [docs/SRS-addendum-v1.1.md](../../SRS-addendum-v1.1.md) (mục 4.10–4.13, 5.7) — đọc cùng [docs/SRS.md](../../SRS.md) bản gốc.

## Global Constraints

- CPU **luôn** chạy được; GPU (Intel/OpenVINO, NVIDIA/CUDA, Apple/MPS) là tuỳ chọn, chọn backend bằng biến môi trường `ACCEL_BACKEND` lúc chạy app (mặc định `auto`: tự dò), tự rơi về CPU (R1 sửa đổi, FR-104). Không backend nào là phụ thuộc bắt buộc: thiếu gói/driver của backend nào thì backend đó bị coi là không có. *(Đổi so với FR-101 gốc, vốn loại CUDA vì chỉ nhắm Intel Arc — xem mục "Điều chỉnh" trong addendum.)*
- Mọi ngưỡng, hằng số, biến môi trường mới đặt trong `backend/pipeline/config.py` (NFR-31). Không rải `os.environ` ra các module khác.
- Engine mới = một file trong `segment/` + một dòng `@ENGINES.register`, chữ ký `(scene, person, params) -> Cutout`; giao diện tự hiện qua `/options` (FR-81, FR-110).
- Bước mới hiện trên sơ đồ pipeline và PDF chỉ bằng `with trace.stage(...)` (FR-82).
- `Cutout.alpha/foreground/background/trimap` luôn có kích thước đúng bằng `scene.rgb` (ảnh làm việc đầy đủ); ngoài ROI thì α = 0, B = ảnh gốc (FR-90, FR-36).
- Ảnh tiền xử lý tuyệt đối không vào ảnh kết quả (R2). Ảnh người dùng chỉ ở RAM, không ghi đĩa (NFR-40) — **chỉ** cache model được ghi đĩa, vào `ACCEL_CACHE_DIR`.
- Ảnh làm việc ≤ 1280 px cạnh dài, upload ≤ 25 MB (R4). Phần nặng chạy một lần cho mỗi (ảnh, người, engine, tham số upstream) (R5, FR-94); slider cường độ không gọi backend (FR-41).
- Dùng FP16 ở những chỗ backend hỗ trợ trực tiếp (OpenVINO `INFERENCE_PRECISION_HINT`, YOLO `half=True` trên CUDA); onnxruntime CUDA/CoreML chạy FP32 vì không có model FP16 sẵn — chấp nhận và ghi trong `Device.dtype` (nới so với "FP16 tối thiểu" của spec; ghi lại trong addendum). **Không** làm INT8. Ngân sách bộ nhớ model nạp cùng lúc mặc định ≤ 3 GB = `ACCEL_MEM_BUDGET_MB = 3072`, áp cho mọi backend (kể cả GPU có VRAM riêng, cho đơn giản; chỉnh được bằng env).
- Backend chọn **một lần lúc khởi động** (env, đọc trong `config.py`); đổi backend cần khởi động lại. Chỉ lựa chọn `auto|cpu|gpu` đổi được theo từng request.
- Mặc định `FOCUS_METHOD = "laplacian"`; chỉ đổi mặc định sau khi Task D5 đo đạt nghiệm thu (FR-121).
- Nhãn hiển thị bằng tiếng Việt, lấy từ backend (NFR-21). Docstring/comment trong code viết tiếng Việt như các module hiện có.
- Test chạy được trên máy **không có GPU và không có trọng số model**: test nào cần model phải `pytest.importorskip` / `skipif` theo biến môi trường, không được fail.
- Các số ở mục 5.7 là **mục tiêu**, không phải số đo. Task nào báo hiệu năng phải báo số đo thật kèm máy đo; không chép số mục tiêu vào tài liệu như thể đã đạt.

## Review Focus

Năm lớp đầu vào mà spec ngụ ý nhưng không nêu, dễ gây lỗi cho người dùng nhất (test đã được gắn vào task sở hữu code):

1. **Người sát biên ảnh** (box chạm biên trên/dưới/trái/phải): ROI bị cắt cụt, alpha/F/B dán lại phải không bị lệch toạ độ và không crash — Task A1 (clamp) + A2 (`paste` đúng vị trí).
2. **Box thủ công** (`person.mask is None`, `idx = -1`): `person_roi` phải dựng ROI từ box, danh sách người khác rỗng, và khoá cache của `idx = -1` không trùng người #0 — Task A1 + A3.
3. **Người rất nhỏ trong ảnh** (box ~30 px): ROI_MARGIN 12% nhỏ hơn 16 px nên phải dùng sàn 16 px; không được phóng to lên 640 px; dải unknown của trimap vẫn phải khác rỗng — Task A1 + A4.
4. **Ảnh phẳng hoàn toàn** (vùng trời/áo trơn, hoặc ảnh một màu): năng lượng AC = 0 → bản đồ nét không được ra NaN/inf, percentile vẫn hợp lệ, và độ tin cậy focus phải báo "không đáng tin" thay vì im lặng — Task D1 + D4.
5. **Backend được chọn mà không dùng được** (đặt `ACCEL_BACKEND=cuda` trên máy không NVIDIA, thiếu gói, tên backend gõ sai, driver reset giữa chừng, op không hỗ trợ): `resolve()` trả CPU và không raise, `with_fallback` trả kết quả CPU, trace có cảnh báo, người dùng không thấy lỗi — Task B1 (+ B2, B3 cho từng đường model).

---

## Cấu trúc file

**Tạo mới**

| File | Trách nhiệm |
|---|---|
| `backend/pipeline/roi.py` | `Roi`, chọn ROI quanh người, đổi về độ phân giải làm việc, guided filter, nâng α/F/B về độ phân giải gốc (FR-90..93, 95) |
| `backend/pipeline/cache.py` | `StageCache` — cache kết quả trung gian theo khoá (FR-94) |
| `backend/pipeline/accel.py` | Lớp thiết bị duy nhất: registry `BACKENDS` (cuda/mps/openvino), chọn thiết bị theo cấu hình, `onnx_runner` đa backend, cache model, khoá GPU, ngân sách bộ nhớ, fallback CPU (FR-100..104, 108) |
| `requirements-intel.txt`, `requirements-cuda.txt`, `requirements-mac.txt` | Gói theo backend, cài thêm khi cần; file lõi không phụ thuộc chúng |
| `backend/pipeline/focusmap.py` | Đo độ nét: `focus_energy` (chuyển từ `segment/focus.py`), bản đồ DCT tần số, chọn phương pháp, độ tin cậy (FR-120..126) |
| `backend/pipeline/segment/gate.py` | `gate_to_person` — giới hạn α trong mask người được chọn, loại mảnh rời (tách từ `birefnet.py` để `light.py` dùng lại) |
| `backend/pipeline/segment/light.py` | Engine `rvm` và `modnet` (FR-110..114) |
| `tools/bench.py` | Đo trung vị/p95 theo phương pháp mục 5.7 |
| `tools/eval_quality.py` | IoU/SAD giữa hai cấu hình trên một tập ảnh (nghiệm thu 4.10, 4.11, 4.12) |
| `tools/compare_weights.py` | So tỉ lệ phát hiện giữa các `YOLO_WEIGHTS` (FR-116) |
| `tests/` | `conftest.py` + test cho từng khối |
| `requirements-dev.txt` | `pytest` |

**Sửa**

| File | Thay đổi |
|---|---|
| `backend/pipeline/config.py` | Toàn bộ hằng số/biến môi trường mới |
| `backend/pipeline/schema.py` | `SegParams`: thêm `cache`, `device_mode`, `cached()` |
| `backend/pipeline/matting.py` | `finish_cutout` nhận `roi`, khử nhiễm màu ở độ phân giải làm việc (FR-93) |
| `backend/pipeline/segment/grabcut.py` | `cutout_from_mask` + engine `cf`/`knn` chạy trên ROI |
| `backend/pipeline/segment/birefnet.py` | Chạy BiRefNet trên ROI, qua lớp thiết bị, dùng `gate.py` |
| `backend/pipeline/segment/focus.py` | Chạy trên ROI, dùng `focusmap.py` |
| `backend/pipeline/detect.py` | YOLO qua lớp thiết bị (CUDA / MPS / OpenVINO IR FP16, fallback CPU) |
| `backend/pipeline/select.py` | Dùng lại bản đồ nét của `focusmap` (FR-125) |
| `backend/pipeline/__init__.py` | Export các API mới |
| `backend/main.py` | Cache theo session, tham số `device`, `/health`+`/options` báo thiết bị, warm-up, trường `focus_confidence` |
| `backend/report.py` | Ghi thiết bị + ROI vào PDF (FR-96, FR-107) |
| `frontend/app.py` | Hiện thiết bị + ép CPU/GPU, trạng thái "đang chuẩn bị", engine mặc định nhẹ + nút "chất lượng cao", cảnh báo độ tin cậy focus |
| `README.md` | Bảng backend đã/chưa kiểm thử kèm phiên bản driver, cách chọn backend lúc chạy app, giấy phép model, bảng so sánh YOLO_WEIGHTS |
| `requirements.txt` | chỉ thêm `onnx` (gói theo backend nằm ở các file `requirements-*.txt`) |
| `docs/SRS.md` | Cập nhật R1, FR-31, NFR-01..03, mục 7.1 sau khi có số đo (Task B6, D5) |

**Ngoài phạm vi kế hoạch này:** phần "*có thể* dùng lại để tính chỉ số độ nét ở FR-10" của FR-125 (spec ghi là tuỳ chọn, và đổi thang `sharpness` sẽ làm lệch ngưỡng `Q_BLURRY` đang dùng); INT8; TensorRT; DirectML (GPU AMD/Intel trên Windows) và ROCm — thêm sau bằng một mục mới trong `BACKENDS` mà không sửa engine.

---

# Phase A — ROI + độ phân giải làm việc (FR-90..96)

Không phụ thuộc phần cứng. Sau Phase A, hệ thống đã nhanh hơn trên chính CPU hiện tại.

### Task A1: `Roi` và chọn ROI quanh người

**Files:**
- Create: `backend/pipeline/roi.py`
- Modify: `backend/pipeline/config.py`
- Create: `tests/conftest.py`, `tests/test_roi.py`, `requirements-dev.txt`, `pytest.ini`

**Interfaces:**
- Consumes: `Scene`, `Person` từ `pipeline/schema.py`; `Person.box = (x1, y1, x2, y2)`, `Person.mask` là `uint8` 0/255 hoặc `None`.
- Produces:
  - `@dataclass(frozen=True) class Roi` với các trường `x1: int, y1: int, x2: int, y2: int, scale: float, full_shape: Tuple[int, int]` (`full_shape` = `(H, W)` của ảnh làm việc đầy đủ) và thuộc tính `full_size -> Tuple[int, int]` (w, h ở độ phân giải gốc), `work_size -> Tuple[int, int]`, `is_full -> bool`.
  - `person_roi(scene: Scene, person: Person, margin: float = C.ROI_MARGIN, min_px: int = C.ROI_MIN_PX, max_side: int = C.WORK_MAX_SIDE) -> Roi`
  - `full_roi(shape, max_side: Optional[int] = None) -> Roi` (scale = 1.0 khi `max_side is None`)
  - `active_roi(scene: Scene, person: Person) -> Roi` — trả `full_roi(scene.rgb.shape)` khi `C.FULLRES` bật, ngược lại `person_roi(...)`. **Mọi engine gọi hàm này**, không gọi `person_roi` trực tiếp.

- [ ] **Step 1: Viết test thất bại**

`tests/conftest.py` cung cấp fixture dựng `Scene` tổng hợp (không cần YOLO):

```python
@pytest.fixture
def make_scene():
    """make_scene((H, W), [(box, mask_or_None), ...]) -> Scene"""
```

`tests/test_roi.py`:

```python
def test_roi_margin_12_percent():
    # ảnh 1000x1000, box 100,100,300,500 (w=200,h=400) -> nở 0.12*400 = 48 px mỗi phía
    assert (roi.x1, roi.y1, roi.x2, roi.y2) == (52, 52, 348, 548)

def test_roi_margin_floor_16px():
    # box 30x30 -> 0.12*30 = 3.6 px < 16 -> nở 16 px
    assert roi.x1 == box_x1 - 16

def test_roi_clamped_to_image():          # Review Focus #1
    # box chạm biên: ROI không âm, không vượt H/W
    assert roi.x1 >= 0 and roi.y1 >= 0 and roi.x2 <= W and roi.y2 <= H

def test_roi_unions_overlapping_person_box():
    # người khác CHỒNG box người được chọn -> ROI gộp cả box của họ
    assert roi.x2 >= other_box[2]

def test_roi_ignores_non_overlapping_person():
    # người khác KHÔNG chồng -> ROI không đổi
    assert (roi.x1, roi.y1, roi.x2, roi.y2) == roi_alone

def test_roi_from_manual_box_without_mask():   # Review Focus #2
    # person.mask is None, idx = -1 -> vẫn dựng được ROI từ box
    assert roi.full_size == (expected_w, expected_h)

def test_full_roi_covers_image_and_is_full():
    assert full_roi((H, W, 3)).is_full and full_roi((H, W, 3)).scale == 1.0
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `python -m pytest tests/test_roi.py -v`
Expected: FAIL — `ModuleNotFoundError: backend.pipeline.roi`

- [ ] **Step 3: Thêm hằng số vào `backend/pipeline/config.py`**

Nhóm mới `# --- ROI / độ phân giải làm việc ---`: `ROI_MARGIN = 0.12`, `ROI_MIN_PX = 16`, `WORK_MAX_SIDE = 640`, `GUIDED_RADIUS = 8`, `GUIDED_EPS = 1e-4`, `FULLRES = os.environ.get("FULLRES", "0") == "1"`.

Hạ tầng test (làm một lần, dùng cho mọi task sau): `requirements-dev.txt` chứa `pytest`; `pytest.ini` đặt `[pytest]` + `pythonpath = .` + `testpaths = tests` để `import backend...` và `import tools...` chạy được khi gọi `python -m pytest` từ thư mục `background_blur/`. Mọi lệnh `python` trong kế hoạch này chạy từ thư mục đó với `.venv` đã kích hoạt.

- [ ] **Step 4: Hiện thực `Roi`, `person_roi`, `full_roi`, `active_roi` trong `backend/pipeline/roi.py`**

Biên ROI: nở `max(min_px, margin * max(bw, bh))` mỗi phía quanh box người được chọn, gộp (union) box của những người khác **có box giao với box người được chọn** (dùng giao diện tích > 0 trên box, không cần mask — rẻ và đủ cho FR-37), rồi clamp vào ảnh. `scale` tính ở Task A2 (bước này đặt `scale = 1.0` và để test A2 đẩy nó đi).

- [ ] **Step 5: Chạy test, xác nhận pass**

Run: `python -m pytest tests/test_roi.py -v`
Expected: PASS (7 test)

- [ ] **Step 6: Commit**

```bash
git add requirements-dev.txt tests/conftest.py tests/test_roi.py backend/pipeline/roi.py backend/pipeline/config.py
git commit -m "feat(roi): chọn ROI quanh người được chọn (FR-90)"
```

---

### Task A2: Độ phân giải làm việc + guided filter + nâng về độ phân giải gốc

**Files:**
- Modify: `backend/pipeline/roi.py`
- Create: `tests/test_upsample.py`

**Interfaces:**
- Consumes: `Roi` (Task A1).
- Produces:
  - `Roi.crop(img) -> np.ndarray` (cắt ROI, giữ độ phân giải gốc)
  - `Roi.to_work(img, interp=cv2.INTER_AREA) -> np.ndarray` (cắt + thu nhỏ về `work_size`)
  - `Roi.paste(patch, fill=0) -> np.ndarray` (patch kích thước `full_size` → ảnh `full_shape`, ngoài ROI = `fill`; nhận patch 2D và 3D)
  - `guided_filter(guide: np.ndarray, src: np.ndarray, radius: int = C.GUIDED_RADIUS, eps: float = C.GUIDED_EPS) -> np.ndarray` — `guide` là ảnh xám float32 0..1, `src` float32 cùng HxW (2D hoặc 3D; 3D thì lọc từng kênh)
  - `upsample_alpha(alpha_work: np.ndarray, roi: Roi, guide_rgb_f: np.ndarray, unknown_work: Optional[np.ndarray] = None) -> np.ndarray` — trả α float32 kích thước `full_shape`, ngoài ROI = 0
  - `upsample_rgb(rgb_work: np.ndarray, roi: Roi, guide_rgb_f: np.ndarray, base: np.ndarray) -> np.ndarray` — trả float32 `full_shape + (3,)`, ngoài ROI = `base`
  - `person_roi` nay đặt `scale = min(1.0, max_side / max(roi_w, roi_h))`

- [ ] **Step 1: Viết test thất bại**

```python
def test_work_side_capped_at_640():
    assert max(roi.work_size) == 640

def test_small_roi_is_not_upscaled():         # Review Focus #3
    # ROI 120x200 -> scale = 1.0, work_size = (120, 200)
    assert roi.scale == 1.0 and roi.work_size == (120, 200)

def test_guided_filter_matches_box_filter_on_constant_guide():
    # guide hằng số -> guided filter suy biến thành box filter của src
    assert np.allclose(guided_filter(const_guide, src, 4, 1e-2), cv2.boxFilter(src, -1, (9, 9)), atol=1e-3)

def test_upsample_alpha_shape_and_zero_outside_roi():
    a = upsample_alpha(alpha_work, roi, guide)
    assert a.shape == (H, W) and a.dtype == np.float32 and a[0, 0] == 0.0

def test_upsample_alpha_keeps_certain_regions_binary():
    # α_work = 0 hoặc 1 ngoài dải unknown -> sau upsample vẫn đúng 0.0 / 1.0
    assert set(np.unique(a[certain_fg])) == {1.0} and set(np.unique(a[certain_bg])) == {0.0}

def test_upsample_alpha_aligned_for_border_roi():   # Review Focus #1
    # ROI chạm biên phải/dưới: tâm khối α sáng phải nằm trong ROI, lệch <= 2 px
    assert abs(centroid_x - expected_x) <= 2

def test_upsample_rgb_keeps_base_outside_roi():
    assert np.array_equal(out[0, 0], base[0, 0])
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `python -m pytest tests/test_upsample.py -v`
Expected: FAIL — `AttributeError: 'Roi' object has no attribute 'to_work'`

- [ ] **Step 3: Hiện thực trong `backend/pipeline/roi.py`**

`guided_filter` theo công thức O(1) của He et al., chỉ dùng `cv2.boxFilter` (OpenCV bản này không có `ximgproc`): với `r = 2*radius+1`,
`mean_I, mean_p, corr_Ip = boxFilter(I), boxFilter(p), boxFilter(I*p)`;
`a = (corr_Ip - mean_I*mean_p) / (boxFilter(I*I) - mean_I² + eps)`; `b = mean_p - a*mean_I`;
kết quả `= boxFilter(a)*I + boxFilter(b)`.

`upsample_alpha`: resize `alpha_work` lên `roi.full_size` bằng `INTER_LINEAR` → guided filter với guide = xám của `roi.crop(guide_rgb_f)` → **chỉ** nhận giá trị lọc trong dải chưa biết, ngoài dải lấy lại 0/1 từ bản resize. Dải chưa biết = `unknown_work` (resize lên, > 0.5) nếu có, ngược lại `(0.02 < a_up < 0.98)` nở 2 px. Cuối cùng `roi.paste(..., fill=0.0)`.

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `python -m pytest tests/ -v`
Expected: PASS (14 test)

- [ ] **Step 5: Commit**

```bash
git add tests/test_upsample.py backend/pipeline/roi.py
git commit -m "feat(roi): độ phân giải làm việc 640px + guided-filter upsample (FR-91, FR-92)"
```

---

### Task A3: Cache trung gian theo khoá

**Files:**
- Create: `backend/pipeline/cache.py`
- Modify: `backend/pipeline/schema.py`, `backend/pipeline/segment/__init__.py`, `backend/main.py`
- Create: `tests/test_cache.py`

**Interfaces:**
- Produces:
  - `class StageCache` với `__init__(self, maxsize: int = 32)`, `get_or(self, key: tuple, build: Callable[[], Any]) -> Any` (LRU, trả bản đã lưu nếu khoá trùng), `clear(self) -> None`, `__len__`
  - `SegParams` thêm trường `cache: Optional[StageCache] = None` và `device_mode: str = "auto"`, cùng method `cached(self, key: tuple, build: Callable[[], Any]) -> Any` (gọi `self.cache.get_or` nếu có cache, ngược lại `build()`)
  - `build_cutout(scene, person, method="cf", params=None, trace=None)` giữ nguyên chữ ký; `Session` trong `main.py` thêm `cache: StageCache`, truyền qua `SegParams`, và `clear()` khi detect lại.
- Consumes: không có.

- [ ] **Step 1: Viết test thất bại**

```python
def test_get_or_builds_once_per_key():
    assert calls == 1 after two get_or with same key

def test_different_keys_build_separately():
    assert calls == 2

def test_manual_box_person_key_differs_from_person_0():   # Review Focus #2
    # khoá ("gcmask", -1) và ("gcmask", 0) là hai khoá khác nhau
    assert cache.get_or(("gcmask", -1), b1) is not cache.get_or(("gcmask", 0), b2)

def test_lru_evicts_oldest_beyond_maxsize():
    assert len(cache) == 2 for maxsize=2 after 3 keys

def test_segparams_cached_works_without_cache():
    assert SegParams().cached(("k",), lambda: 7) == 7
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `python -m pytest tests/test_cache.py -v`
Expected: FAIL — `ModuleNotFoundError: backend.pipeline.cache`

- [ ] **Step 3: Hiện thực `StageCache` (dựa trên `collections.OrderedDict`) và nối vào `SegParams` + `Session`**

Trong `main.py`: `Session.cache: StageCache = field(default_factory=StageCache)`; hàm `work()` của `/process` truyền `P.SegParams(..., cache=s.cache, device_mode=device)`; hàm `work()` của `/detect` gọi `s.cache.clear()` trước khi detect.

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `python -m pytest tests/ -v`
Expected: PASS (19 test)

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline/cache.py backend/pipeline/schema.py backend/pipeline/segment/__init__.py backend/main.py tests/test_cache.py
git commit -m "feat(cache): cache kết quả trung gian theo khoá (FR-94)"
```

---

### Task A4: Engine `cf`/`knn` chạy trên ROI + chế độ `FULLRES`

**Files:**
- Modify: `backend/pipeline/segment/grabcut.py`, `backend/pipeline/matting.py`
- Create: `tests/test_engine_roi.py`

**Interfaces:**
- Consumes: `active_roi`, `Roi.to_work`, `upsample_alpha` (A1, A2); `SegParams.cached` (A3).
- Produces:
  - `cutout_from_mask(rgb_f: np.ndarray, mask_work: np.ndarray, roi: Roi, params: SegParams, method: str) -> Cutout` — `rgb_f` là ảnh **đầy đủ** float32, `mask_work` là mask ở `roi.work_size`
  - `finish_cutout(rgb_f: np.ndarray, alpha_work: np.ndarray, trimap_work: np.ndarray, roi: Roi, trace: Optional[Trace] = None) -> Cutout` (chữ ký thay đổi — mọi engine phải cập nhật; nội dung khử nhiễm màu làm ở Task A5)
  - Khoá cache của engine `cf`/`knn`: `("roi", person.idx)` cho ROI, `("gcmask", person.idx)` cho mask GrabCut (không phụ thuộc `band_frac` nên đổi band chỉ tính lại từ trimap — FR-94)

- [ ] **Step 1: Viết test thất bại**

Dùng ảnh tổng hợp (hình chữ nhật màu trên nền gradient) + `Scene` từ fixture, chạy `build_cutout` thật (không cần YOLO):

```python
def test_cutout_has_full_image_shapes():
    for arr in (c.alpha, c.trimap): assert arr.shape == scene.rgb.shape[:2]
    for arr in (c.foreground, c.background): assert arr.shape == scene.rgb.shape

def test_alpha_is_zero_outside_roi():
    assert c.alpha[0, 0] == 0.0 and c.alpha[-1, -1] == 0.0

def test_trimap_unknown_band_not_empty_for_tiny_person():   # Review Focus #3
    assert (c.trimap == 128).sum() > 0

def test_fullres_flag_processes_whole_image(monkeypatch):
    # monkeypatch C.FULLRES = True -> roi.is_full và work_size == kích thước ảnh
    assert roi.is_full and roi.work_size == (W, H)

def test_grabcut_mask_reused_when_only_band_changes(monkeypatch):
    # đếm số lần refine_mask_grabcut được gọi qua 2 lượt khác band_frac
    assert calls == 1
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `python -m pytest tests/test_engine_roi.py -v`
Expected: FAIL — `TypeError: cutout_from_mask() missing 1 required positional argument: 'roi'`

- [ ] **Step 3: Chuyển engine `cf`/`knn` sang ROI trong `backend/pipeline/segment/grabcut.py`**

Trong `_make_engine(...).run`: lấy `roi = params.cached(("roi", person.idx), lambda: active_roi(scene, person))`; đổi ảnh, mask khởi tạo và `exclude` sang độ phân giải làm việc bằng `roi.to_work` (mask dùng `INTER_NEAREST`); `refine_mask_grabcut` chạy trên ảnh làm việc với box = box người **đã đổi toạ độ về hệ ROI làm việc** (thêm hàm nội bộ `_box_to_work(box, roi)`); kết quả trả qua `cutout_from_mask(to_float(scene.rgb), mask_work, roi, params, method)`.

`finish_cutout` tạm thời: `alpha = upsample_alpha(alpha_work, roi, rgb_f, unknown_work=(trimap_work == 128))`, `trimap = roi.paste(cv2.resize(trimap_work, roi.full_size, INTER_NEAREST), fill=0)`, F/B vẫn tính ở độ phân giải gốc như hiện tại (Task A5 đổi).

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `python -m pytest tests/ -v`
Expected: PASS (24 test)

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline/segment/grabcut.py backend/pipeline/matting.py tests/test_engine_roi.py
git commit -m "feat(segment): cf/knn chạy trên ROI ở độ phân giải làm việc + FULLRES (FR-90, FR-91, FR-95)"
```

---

### Task A5: Khử nhiễm màu ở độ phân giải làm việc

**Files:**
- Modify: `backend/pipeline/matting.py`
- Create: `tests/test_decontaminate_roi.py`

**Interfaces:**
- Consumes: `upsample_rgb`, `Roi.to_work` (A2).
- Produces: `finish_cutout` (chữ ký như A4) nay tính `decontaminate` trên ảnh làm việc rồi nâng F, B bằng `upsample_rgb`; pixel α ≥ 1−1e-6 lấy F trực tiếp từ `rgb_f`; ngoài ROI B = `rgb_f` (FR-93).

- [ ] **Step 1: Viết test thất bại**

```python
def test_foreground_equals_original_where_alpha_is_one():
    assert np.allclose(c.foreground[c.alpha >= 1.0], c.image[c.alpha >= 1.0], atol=1e-6)

def test_background_equals_original_outside_roi():
    assert np.allclose(c.background[0, 0], c.image[0, 0], atol=1e-6)

def test_decontaminate_runs_at_work_resolution(monkeypatch):
    # chặn pymatting.estimate_foreground_ml, kiểm tra shape đầu vào == roi.work_size
    assert seen_shape[:2] == (roi.work_size[1], roi.work_size[0])
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `python -m pytest tests/test_decontaminate_roi.py -v`
Expected: FAIL — shape đầu vào bằng kích thước ảnh đầy đủ, không phải `work_size`

- [ ] **Step 3: Sửa `finish_cutout` trong `backend/pipeline/matting.py`**

Trình tự: `rgb_work = roi.to_work(rgb_f)` → `fg_work, bg_work = decontaminate(rgb_work, alpha_work)` → `alpha = upsample_alpha(...)` → `fg = upsample_rgb(fg_work, roi, rgb_f, base=rgb_f)`, `bg = upsample_rgb(bg_work, roi, rgb_f, base=rgb_f)` → gán `fg[alpha >= 1 - 1e-6] = rgb_f[...]`, `bg[alpha <= 0] = rgb_f[...]`. Ảnh minh hoạ của stage vẫn vẽ ở độ phân giải gốc.

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `python -m pytest tests/ -v`
Expected: PASS (27 test)

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline/matting.py tests/test_decontaminate_roi.py
git commit -m "feat(matting): khử nhiễm màu ở độ phân giải làm việc, F/B nâng bằng guided filter (FR-93)"
```

---

### Task A6: BiRefNet trên ROI + tách khối `gate`

**Files:**
- Create: `backend/pipeline/segment/gate.py`
- Modify: `backend/pipeline/segment/birefnet.py`
- Create: `tests/test_gate.py`

**Interfaces:**
- Consumes: `active_roi`, `upsample_alpha`, `finish_cutout` (A2, A4, A5); `exclusion_mask` (có sẵn).
- Produces: `gate_to_person(alpha: np.ndarray, init: np.ndarray, box: Tuple[int, int, int, int], exclude: Optional[np.ndarray] = None) -> np.ndarray` — α đã giới hạn trong mask người (nở `max(5, 4% cạnh dài box)`, làm mềm), trừ vùng người khác, loại mảnh rời; mọi đối số **cùng độ phân giải làm việc**. `birefnet.birefnet_alpha(rgb)` giữ nguyên chữ ký nhưng nay được gọi với ảnh ROI làm việc.

- [ ] **Step 1: Viết test thất bại**

```python
def test_gate_zeroes_alpha_outside_dilated_person_mask():
    assert gated[far_from_person] == 0.0

def test_gate_drops_detached_blob():
    assert gated[blob_region].max() == 0.0

def test_gate_zeroes_excluded_region():
    assert gated[exclude].max() == 0.0

def test_gate_keeps_thin_limb_inside_mask():
    assert gated[limb].min() > 0.5
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `python -m pytest tests/test_gate.py -v`
Expected: FAIL — `ModuleNotFoundError: backend.pipeline.segment.gate`

- [ ] **Step 3: Chuyển khối "gate" của `birefnet.py` sang `gate.py` và cho BiRefNet chạy trên ROI**

`birefnet.run`: `roi = params.cached(("roi", person.idx), ...)` → `rgb_work = roi.to_work(scene.rgb)` → `alpha_work = params.cached(("birefnet", person.idx), lambda: birefnet_alpha(rgb_work))` → `gate_to_person(...)` với `init`, `box`, `exclude` đã đổi về hệ làm việc → `trimap_work` theo ngưỡng 0.01/0.99 → `finish_cutout(to_float(scene.rgb), alpha_work, trimap_work, roi, t)`. Ghi ROI vào note của stage.

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `python -m pytest tests/ -v`
Expected: PASS (31 test)

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline/segment/gate.py backend/pipeline/segment/birefnet.py tests/test_gate.py
git commit -m "feat(birefnet): chạy trên ROI, tách khối gate dùng chung (FR-31 sửa đổi, FR-90)"
```

---

### Task A7: Engine `focus` trên ROI + trace ROI

**Files:**
- Modify: `backend/pipeline/segment/focus.py`, `backend/pipeline/trace.py`, `backend/report.py`
- Create: `tests/test_focus_roi.py`

**Interfaces:**
- Consumes: `active_roi`, `Roi`, `cutout_from_mask` (A1, A2, A4).
- Produces:
  - `focus_mask(rgb, person, init, frac=0.3, others=None)` giữ chữ ký, nay nhận ảnh/mask ở độ phân giải làm việc.
  - Stage mới `roi` trong trace tách nền: `key="roi"`, `label="Vùng xử lý (ROI) và độ phân giải làm việc"`, `note=f"ROI {x1},{y1}-{x2},{y2} ({pct:.0%} ảnh) · làm việc {ww}×{wh}px"`, ảnh minh hoạ là ảnh gốc có vẽ khung ROI. Engine nào cũng mở stage này ngay sau `init_mask` (FR-96).
  - `report.py`: mục "2. Cài đặt tách nền" thêm dòng `("Vùng xử lý", "<note của stage roi>")`.
- Khoá cache: `("focusmask", person.idx, round(params.focus_frac, 3))`.

- [ ] **Step 1: Viết test thất bại**

```python
def test_focus_engine_returns_full_shape_cutout():
    assert c.alpha.shape == scene.rgb.shape[:2]

def test_focus_trace_has_roi_stage_with_size_note():
    st = next(s for s in c.trace.stages if s.key == "roi")
    assert "làm việc" in st.note and st.image is not None

def test_focus_mask_reused_when_only_band_changes(monkeypatch):
    assert focus_mask_calls == 1
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `python -m pytest tests/test_focus_roi.py -v`
Expected: FAIL — không có stage `roi`

- [ ] **Step 3: Chuyển engine `focus` sang ROI và thêm stage `roi`**

Thêm hàm dùng chung `roi_stage(trace, scene_rgb, roi)` trong `backend/pipeline/roi.py` (vẽ khung + mở stage) rồi gọi từ cả ba engine `grabcut`/`birefnet`/`focus`. Engine `focus` chạy `focus_energy`, `focus_mask`, `refine_mask_grabcut` trên ảnh làm việc; `mask_bbox` cũng tính trên ảnh làm việc.

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `python -m pytest tests/ -v`
Expected: PASS (34 test)

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline/segment/focus.py backend/pipeline/roi.py backend/report.py tests/test_focus_roi.py
git commit -m "feat(focus): chạy trên ROI; trace và PDF ghi ROI + kích thước làm việc (FR-90, FR-96)"
```

---

### Task A8: Công cụ đo và nghiệm thu Phase A

**Files:**
- Create: `tools/bench.py`, `tools/eval_quality.py`
- Modify: `docs/SRS-addendum-v1.1.md` (chỉ điền bảng số đo), `README.md`

**Interfaces:**
- Consumes: pipeline qua `backend.pipeline` (gọi trực tiếp, không qua HTTP, để không tính thời gian mạng).
- Produces:
  - `tools/bench.py --dir <thư mục ảnh> --limit 30 --engine cf,knn,birefnet,focus [--device auto|cpu|gpu] --runs 10 --drop 2` → in bảng markdown trung vị + p95 theo từng **stage** của trace và tổng (phương pháp mục 5.7).
  - `tools/eval_quality.py --dir <thư mục ảnh> --limit 30 --engine cf [--device auto|cpu|gpu]` cùng **một** trong ba tham chiếu: `--ref-fullres` (so với `FULLRES=1`), `--ref-engine <tên>`, hoặc `--ref-device cpu` (cùng engine, thiết bị khác) → in `mean IoU`, `mean SAD`, `p05 IoU`, và thời gian tương đối; IoU tính trên α nhị phân hoá tại 0.5, SAD = `abs(a - a_ref).sum() / 255` theo quy ước matting.
  - Tập ảnh nghiệm thu chuẩn: `dataset/hd_bank_green_marathon_2026_resized`, lấy 30 ảnh đầu theo thứ tự tên sắp xếp (deterministic) — ghi trong `--help` và README.

- [ ] **Step 1: Viết test thất bại**

```python
# tests/test_tools.py
def test_iou_and_sad_identical_maps():
    assert iou(a, a) == 1.0 and sad(a, a) == 0.0

def test_median_p95_drops_warmup_runs():
    # [100, 90, 10, 11, 12, 13] drop=2 -> trung vị tính trên [10..13]
    assert median == 11.5
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `python -m pytest tests/test_tools.py -v`
Expected: FAIL — `ModuleNotFoundError: tools.eval_quality`

- [ ] **Step 3: Hiện thực hai script (hàm thuần `iou`, `sad`, `summarize(times, drop)` để test gọi được; phần CLI dùng `argparse`)**

- [ ] **Step 4: Chạy test + chạy nghiệm thu thật**

Run: `python -m pytest tests/ -v` → PASS
Run: `python tools/eval_quality.py --dir ../dataset/hd_bank_green_marathon_2026_resized --limit 30 --engine cf --ref-fullres`
Expected: `mean IoU ≥ 0.98`, `mean SAD` tăng ≤ 5%, và `python tools/bench.py ... --engine cf` cho thấy thời gian giảm ≥ 2× so với `FULLRES=1`. Nếu **không** đạt: tăng `WORK_MAX_SIDE` lên 800 hoặc `GUIDED_RADIUS`/`GUIDED_EPS`, đo lại, và ghi giá trị cuối vào `config.py`.

- [ ] **Step 5: Ghi số đo thật vào `docs/SRS-addendum-v1.1.md` (cột CPU của 5.7) và README; commit**

```bash
git add tools/bench.py tools/eval_quality.py tests/test_tools.py docs/SRS-addendum-v1.1.md README.md
git commit -m "test(bench): công cụ đo trung vị/p95 + IoU/SAD, nghiệm thu Phase A (5.7)"
```

---

# Phase B — Lớp thiết bị đa backend: Intel/OpenVINO, NVIDIA/CUDA, Mac/MPS (FR-100..108)

### Task B1: `accel.py` — registry backend, chọn thiết bị theo cấu hình, khoá GPU, fallback

Backend là **một mục trong bảng `BACKENDS`**; thêm backend mới (DirectML, ROCm…) sau này không phải sửa engine hay `detect.py`.

| Backend | Phần cứng | YOLO-seg | Model ONNX (BiRefNet, RVM, MODNet) |
|---|---|---|---|
| `cpu` | mọi máy | `.pt`, torch CPU | onnxruntime `CPUExecutionProvider` |
| `cuda` | NVIDIA | `.pt`, torch `cuda:N`, `half=True` khi FP16 | onnxruntime `CUDAExecutionProvider` |
| `mps` | Mac Apple Silicon | `.pt`, torch `mps`, FP32 | onnxruntime `CoreMLExecutionProvider` |
| `openvino` | Intel Arc / iGPU | OpenVINO IR FP16, `intel:gpu` | OpenVINO `GPU`, FP16, static shape |

**Files:**
- Create: `backend/pipeline/accel.py`, `tests/test_accel.py`
- Create: `requirements-intel.txt` (`openvino>=2025.0`), `requirements-cuda.txt` (`onnxruntime-gpu`, ghi chú cài torch bản CUDA), `requirements-mac.txt` (`onnxruntime` có CoreML EP)
- Modify: `backend/pipeline/config.py`, `requirements.txt` (chỉ thêm `onnx`; **không** thêm gói theo backend vào file lõi — thiếu gói thì backend đó bị coi là không có)

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class Device`: `kind: str` (`"cpu"|"gpu"`), `backend: str`, `name: str`, `runtime: str` (`"torch"|"openvino"|"onnxruntime"`), `dtype: str` (`"fp32"|"fp16"`), `index: int = 0`; `to_dict() -> dict`; thuộc tính `torch_device -> str` (`"cpu"`, `"cuda:<index>"`, `"mps"`, `"intel:gpu"`); `ort_providers -> list` (CPU: `["CPUExecutionProvider"]`; CUDA: `[("CUDAExecutionProvider", {"device_id": index}), "CPUExecutionProvider"]`; MPS: `["CoreMLExecutionProvider", "CPUExecutionProvider"]`)
  - `@dataclass(frozen=True) class Backend`: `name: str`, `probe: Callable[[], Optional[str]]` — trả tên thiết bị nếu bộ tăng tốc chính của backend dùng được, ngược lại `None`; **mọi exception trong probe coi như `None`**
  - `BACKENDS: Dict[str, Backend]` — thứ tự khai báo = thứ tự ưu tiên của `auto`: `cuda`, `mps`, `openvino`. Probe: `cuda` = `torch.cuda.is_available()` → `torch.cuda.get_device_name(index)`; `mps` = `sys.platform == "darwin"` và `torch.backends.mps.is_available()`; `openvino` = `openvino.Core().available_devices` chứa `"GPU"` → `FULL_DEVICE_NAME`. Thiếu một EP của onnxruntime **không** làm probe thất bại — từng model sẽ tự rơi về CPU ở `onnx_runner` (Task B3)
  - `available_backends() -> Dict[str, str]` — `{tên backend: tên thiết bị}` của các backend có probe thành công
  - `resolve(mode: Optional[str] = None, backend: Optional[str] = None) -> Device` — `mode` ∈ `auto|cpu|gpu` (`None` → `C.ACCEL_DEVICE`); `backend` ∈ `auto|cpu` hoặc tên trong `BACKENDS` (`None` → `C.ACCEL_BACKEND`). `mode == "cpu"` hoặc `backend == "cpu"` → CPU. Ngược lại: backend chỉ định tường minh thì dùng đúng backend đó nếu probe ổn, không thì CPU; `backend == "auto"` thì lấy backend đầu tiên trong `BACKENDS` có probe ổn, không có thì CPU. **Không bao giờ raise**, kể cả tên backend lạ. `dtype`: `C.ACCEL_DTYPE` nếu khác `auto`, ngược lại `fp16` cho `cuda`/`openvino`, `fp32` cho `mps`/`cpu`
  - `GPU_LOCK: threading.RLock` (FR-108, áp cho mọi backend)
  - `compiled(key: str, build: Callable[[], Any], nbytes: int) -> Any` — cache model đã nạp; vượt `C.ACCEL_MEM_BUDGET_MB` thì giải phóng LRU trước khi nạp, vẫn không đủ thì raise `MemoryError`
  - `release(key: Optional[str] = None) -> None`, `unload_idle(idle_s: float = C.ACCEL_UNLOAD_IDLE) -> int`, `loaded_bytes() -> int`
  - `with_fallback(device: Device, gpu_fn: Callable[[], Any], cpu_fn: Callable[[], Any], stage=None) -> Tuple[Any, Device]` — chạy `gpu_fn` dưới `GPU_LOCK` khi `device.kind == "gpu"`; **bất kỳ** `Exception` (kể cả `MemoryError`) → chạy `cpu_fn`, trả `Device` CPU, nối `" · GPU lỗi, đã dùng CPU: <lý do>"` vào `stage.note` nếu có `stage` (FR-104)

- [ ] **Step 1: Viết test thất bại**

```python
def test_resolve_cpu_when_no_backend_available(monkeypatch):          # Review Focus #5
    # mọi probe trả None
    assert resolve("auto").kind == "cpu" and resolve("gpu").kind == "cpu"

def test_auto_picks_first_available_in_priority_order(monkeypatch):
    # cuda và openvino cùng khả dụng -> cuda thắng
    assert resolve("auto", "auto").backend == "cuda"

def test_explicit_backend_not_available_falls_to_cpu(monkeypatch):    # Review Focus #5
    # chỉ openvino khả dụng, yêu cầu backend="cuda" -> CPU, không raise
    assert resolve("gpu", "cuda").kind == "cpu"

def test_unknown_backend_name_falls_to_cpu():
    assert resolve("gpu", "tensorrt").kind == "cpu"

def test_probe_exception_is_treated_as_unavailable(monkeypatch):
    # probe cuda raise RuntimeError -> backend="auto" bỏ qua nó, không crash
    assert resolve("auto", "auto").backend != "cuda"

def test_mode_cpu_ignores_available_gpu(monkeypatch):
    assert resolve("cpu").kind == "cpu"

def test_dtype_defaults_per_backend_and_override(monkeypatch):
    assert resolve("gpu", "cuda").dtype == "fp16" and resolve("gpu", "mps").dtype == "fp32"
    # ACCEL_DTYPE="fp32" -> cuda cũng fp32

def test_torch_device_strings():
    assert [d.torch_device for d in (cpu, cuda0, mps, ov)] == ["cpu", "cuda:0", "mps", "intel:gpu"]

def test_ort_providers_per_backend():
    assert cuda0.ort_providers[0] == ("CUDAExecutionProvider", {"device_id": 0})
    assert mps.ort_providers[0] == "CoreMLExecutionProvider" and cpu.ort_providers == ["CPUExecutionProvider"]

def test_with_fallback_returns_cpu_result_on_gpu_error():             # Review Focus #5
    out, dev = with_fallback(gpu_device, lambda: 1 / 0, lambda: "cpu-ok", stage=h)
    assert out == "cpu-ok" and dev.kind == "cpu" and "CPU" in h.note

def test_compiled_builds_once_and_counts_bytes():
    assert builds == 1 and loaded_bytes() == 1024

def test_compiled_evicts_lru_over_budget(monkeypatch):
    # budget 1 MB, nạp 3 model 512 KB -> model đầu bị giải phóng
    assert loaded_bytes() <= 1024 * 1024

def test_gpu_work_is_serialized():                                    # FR-108
    # 2 thread gọi with_fallback, gpu_fn ghi nhật ký vào/ra -> không chồng nhau
    assert max_concurrent == 1
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `python -m pytest tests/test_accel.py -v`
Expected: FAIL — `ModuleNotFoundError: backend.pipeline.accel`

- [ ] **Step 3: Thêm hằng số và hiện thực `accel.py`**

`config.py`, nhóm `# --- gia tốc phần cứng (chọn bằng biến môi trường khi chạy app) ---`:
`ACCEL_DEVICE = os.environ.get("ACCEL_DEVICE", "auto")` (`auto|cpu|gpu`),
`ACCEL_BACKEND = os.environ.get("ACCEL_BACKEND", "auto")` (`auto|cpu|cuda|mps|openvino`),
`ACCEL_DEVICE_INDEX = int(os.environ.get("ACCEL_DEVICE_INDEX", "0"))` (chọn GPU khi máy có nhiều card),
`ACCEL_DTYPE = os.environ.get("ACCEL_DTYPE", "auto")` (`auto|fp16|fp32`),
`ACCEL_CACHE_DIR = os.environ.get("ACCEL_CACHE_DIR", str(Path.home() / ".cache" / "color_pop_runner"))`,
`ACCEL_MEM_BUDGET_MB = int(os.environ.get("ACCEL_MEM_BUDGET_MB", "3072"))`,
`ACCEL_UNLOAD_IDLE = float(os.environ.get("ACCEL_UNLOAD_IDLE", "0"))` (0 = không giải phóng),
`BIREFNET_SIZE = int(os.environ.get("BIREFNET_SIZE", "1024"))`.

Mọi `import torch` / `openvino` / `onnxruntime` nằm **trong hàm** và bọc `try/except ImportError` — thiếu gói chỉ làm backend tương ứng "không có" (FR-104). Backend được chọn một lần khi khởi động (đọc env); đổi backend cần khởi động lại, còn đổi `auto|cpu|gpu` thì làm được theo từng request (Task B4).

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `python -m pytest tests/ -v`
Expected: PASS (49 test)

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline/accel.py backend/pipeline/config.py requirements.txt requirements-intel.txt requirements-cuda.txt requirements-mac.txt tests/test_accel.py
git commit -m "feat(accel): registry backend cuda/mps/openvino chọn bằng cấu hình, khoá GPU, fallback CPU (FR-100, FR-104, FR-108)"
```

---

### Task B2: YOLO-seg theo backend (CUDA / MPS / OpenVINO IR FP16)

**Files:**
- Modify: `backend/pipeline/detect.py`, `backend/pipeline/accel.py`, `backend/pipeline/schema.py`
- Create: `tests/test_detect_device.py`

**Interfaces:**
- Consumes: `Device`, `resolve`, `with_fallback`, `compiled`, `C.ACCEL_CACHE_DIR` (B1).
- Produces:
  - `detect.predict_kwargs(device: Device) -> dict` — `{"device": device.torch_device, "half": device.backend == "cuda" and device.dtype == "fp16"}`; MPS không bao giờ `half`
  - `accel.openvino_ir_dir(weights: str) -> Path` — `<ACCEL_CACHE_DIR>/<stem>_openvino_model/`
  - `detect.export_openvino(weights: str = C.YOLO_WEIGHTS) -> Optional[Path]` — chỉ cho backend `openvino`: `YOLO(weights).export(format="openvino", half=True, dynamic=False)`, chuyển vào thư mục cache, trả `None` nếu xuất lỗi (FR-102)
  - `detect.get_yolo(weights: str = C.YOLO_WEIGHTS, device: Optional[Device] = None)` — `openvino` nạp IR cache (xuất một lần nếu chưa có, thất bại thì `.pt`); `cpu`/`cuda`/`mps` nạp `.pt`; cache theo `(weights, device.backend, device.index)`
  - `detect_persons(det_rgb, weights=..., conf=0.25, imgsz=C.YOLO_IMGSZ, device: Optional[Device] = None)` — `model.predict(..., **predict_kwargs(device))`; lỗi trên GPU → `with_fallback` chạy lại trên CPU
  - `detect_scene(rgb, preprocess="auto", weights=..., trace=None, device_mode: str = "auto")` — thêm `device_mode`; `Scene` thêm trường `device: dict = field(default_factory=dict)`

- [ ] **Step 1: Viết test thất bại**

```python
@pytest.mark.parametrize("backend,expected", [
    ("cpu", {"device": "cpu", "half": False}),
    ("cuda", {"device": "cuda:0", "half": True}),
    ("mps", {"device": "mps", "half": False}),
    ("openvino", {"device": "intel:gpu", "half": False}),
])
def test_predict_kwargs_per_backend(backend, expected):
    assert predict_kwargs(device_for(backend)) == expected

def test_detect_persons_passes_device_kwargs_to_model(monkeypatch):
    # model giả ghi lại kwargs của predict, device=cuda -> device="cuda:0"
    assert seen["device"] == "cuda:0"

def test_detect_falls_back_to_cpu_when_gpu_predict_raises(monkeypatch):   # Review Focus #5
    # predict trên GPU raise RuntimeError, trên CPU trả 1 người
    assert len(persons) == 1 and used_device == "cpu"

def test_openvino_falls_back_to_pt_when_export_fails(monkeypatch):
    # export_openvino -> None -> vẫn trả người, nạp .pt
    assert len(persons) == 1 and used_weights.endswith(".pt")

def test_yolo_stage_note_contains_backend_and_device(monkeypatch):
    assert "cuda" in stage.note or "CPU" in stage.note
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `python -m pytest tests/test_detect_device.py -v`
Expected: FAIL — `ImportError: cannot import name 'predict_kwargs'`

- [ ] **Step 3: Hiện thực; export OpenVINO chạy một lần nhờ `compiled`**

- [ ] **Step 4: Chạy test + kiểm thử thật trên backend máy đang có**

Run: `python -m pytest tests/ -v` → PASS
Chạy `detect_scene` trên một ảnh trong `dataset/gold_set` với `ACCEL_BACKEND=<backend của máy này>` và với `ACCEL_DEVICE=cpu`; ghi thời gian YOLO của từng cấu hình (NFR-60). Backend nào **không có phần cứng để chạy thật** thì ghi "chưa kiểm thử trên phần cứng thật" trong README (Task B6) — chỉ có test giả lập. Nếu backend chọn chạy lỗi trên máy thật: ghi lại lỗi, giữ CPU, **dừng và báo lại**.

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline/detect.py backend/pipeline/accel.py backend/pipeline/schema.py tests/test_detect_device.py
git commit -m "feat(detect): YOLO-seg chạy theo backend cuda/mps/openvino, fallback CPU (FR-102)"
```

---

### Task B3: Runner ONNX đa backend + BiRefNet

Một hàm `accel.onnx_runner` phục vụ BiRefNet (task này), RVM và MODNet (Phase C) — các engine không biết backend nào đang chạy.

**Files:**
- Modify: `backend/pipeline/accel.py`, `backend/pipeline/segment/birefnet.py`
- Create: `tests/test_onnx_runner.py`, `tests/test_birefnet_device.py`

**Interfaces:**
- Consumes: `Device`, `compiled`, `with_fallback`, `resolve`, `C.BIREFNET_SIZE`, `C.ACCEL_CACHE_DIR` (B1); `gate_to_person` (A6).
- Produces:
  - `accel.onnx_runner(key: str, onnx_path: str, device: Device, static_input: Optional[Dict[str, Tuple[int, ...]]] = None) -> Callable[[Dict[str, np.ndarray]], List[np.ndarray]]` — runner nhận `{tên input: mảng}` và trả danh sách output theo thứ tự của model. Cache qua `compiled(f"{key}:{device.backend}:{device.index}", build, nbytes=os.path.getsize(onnx_path))`.
    - `openvino`: `openvino.convert_model(onnx_path, input=static_input)` → `compile_model` trên `"GPU"` với `{"CACHE_DIR": C.ACCEL_CACHE_DIR, "INFERENCE_PRECISION_HINT": "f16" | "f32"}` (theo `device.dtype`)
    - `cuda` / `mps` / `cpu`: `onnxruntime.InferenceSession(onnx_path, providers=device.ort_providers)`; nếu EP của backend không có trong `onnxruntime.get_available_providers()` thì **raise `RuntimeError`** để `with_fallback` chuyển sang CPU (không lặng lẽ chạy CPU mà vẫn báo là GPU)
  - `birefnet.onnx_path() -> Path` — file ONNX mà rembg đã tải (`~/.u2net/<BIREFNET_MODEL>.onnx`), tải qua rembg nếu chưa có
  - `birefnet_alpha(rgb: np.ndarray, device: Optional[Device] = None) -> np.ndarray` — giữ chữ ký cũ, thêm `device`; CPU qua rembg như hiện tại, GPU qua `onnx_runner` với `static_input` cố định `(1, 3, BIREFNET_SIZE, BIREFNET_SIZE)`

- [ ] **Step 1: Viết test thất bại**

```python
# tests/test_onnx_runner.py  (monkeypatch onnxruntime / openvino bằng module giả)
def test_cuda_runner_requests_cuda_provider_first(monkeypatch):
    assert captured_providers[0] == ("CUDAExecutionProvider", {"device_id": 0})

def test_mps_runner_requests_coreml_provider(monkeypatch):
    assert captured_providers[0] == "CoreMLExecutionProvider"

def test_runner_raises_when_accelerator_ep_missing(monkeypatch):
    # get_available_providers() chỉ có CPU, device = cuda
    with pytest.raises(RuntimeError): onnx_runner("k", path, cuda0)

def test_openvino_runner_sets_cache_dir_and_f16_hint(monkeypatch):
    assert props["CACHE_DIR"] == C.ACCEL_CACHE_DIR and props["INFERENCE_PRECISION_HINT"] == "f16"

def test_openvino_runner_converts_with_static_shape(monkeypatch):
    assert convert_input == {"input": (1, 3, 1024, 1024)}

def test_runner_is_cached_per_backend_and_index(monkeypatch):
    assert builds == 1 after two calls with same device, 2 with different index

# tests/test_birefnet_device.py
def test_birefnet_alpha_uses_rembg_on_cpu(monkeypatch):
    assert called["rembg"] and not called["runner"]

def test_birefnet_alpha_falls_back_to_rembg_when_runner_raises(monkeypatch):   # Review Focus #5
    assert a.shape == rgb.shape[:2] and "CPU" in stage.note

def test_birefnet_input_is_fixed_size(monkeypatch):
    assert seen_shape == (1, 3, C.BIREFNET_SIZE, C.BIREFNET_SIZE)
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `python -m pytest tests/test_onnx_runner.py tests/test_birefnet_device.py -v`
Expected: FAIL — `ImportError: cannot import name 'onnx_runner'`

- [ ] **Step 3: Hiện thực**

Trước khi viết tiền/hậu xử lý: **đọc lớp session BiRefNet trong package `rembg` đã cài** (`.venv/Lib/site-packages/rembg/sessions/`) và dùng đúng kích thước đầu vào, mean/std, tên input/output và hậu xử lý của nó — không đoán. Đầu ra chuẩn hoá về float32 0..1, resize về kích thước ảnh đưa vào.

- [ ] **Step 4: Chạy test + đối chứng chất lượng trên backend máy đang có**

Run: `python -m pytest tests/ -v` → PASS
Run: `python tools/eval_quality.py --dir ../dataset/gold_set --limit 12 --engine birefnet --ref-device cpu --device gpu`
Expected: `mean IoU ≥ 0.99` (nghiệm thu GPU vs CPU FP32, áp cho **từng backend có phần cứng thật**). Dưới ngưỡng: với `openvino` đặt `ACCEL_DTYPE=fp32`; với `cuda`/`mps` ghi lại IoU thật và nghi ngờ EP (op không hỗ trợ bị đẩy về CPU từng phần) trước khi chấp nhận.

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline/accel.py backend/pipeline/segment/birefnet.py tests/test_onnx_runner.py tests/test_birefnet_device.py
git commit -m "feat(accel): onnx_runner đa backend; BiRefNet static shape trên GPU, fallback rembg CPU (FR-103)"
```

---

### Task B4: Warm-up + báo thiết bị ra API và giao diện

**Files:**
- Modify: `backend/main.py`, `frontend/app.py`
- Create: `tests/test_api_device.py`

**Interfaces:**
- Consumes: `accel.resolve`, `accel.gpu_name` (B1); `detect_scene(device_mode=...)` (B2).
- Produces:
  - `GET /health` → thêm `{"device": {...}, "backend": "<backend đang dùng>", "available_backends": {"cuda": "<tên GPU>", ...}, "warmup": "running|done|error", "warmup_detail": "<chuỗi>"}`
  - `GET /options` → thêm `{"device": {...}, "devices": ["auto", "cpu", "gpu"], "default_engine": "<tên>"}` (`default_engine` = engine nhẹ đầu tiên có mặt trong thứ tự `rvm`, `modnet`, `cf` — Task C3 dựa vào đây)
  - `POST /detect` và `POST /process/{sid}` thêm `device: str = Query("auto", pattern="^(auto|cpu|gpu)$")`; truyền vào `detect_scene(device_mode=...)` / `SegParams(device_mode=...)`
  - `main.warmup()` chạy trong lifespan bằng thread: nạp YOLO + BiRefNet theo thiết bị đã chọn, chạy `detect_scene` + `build_cutout` trên ảnh giả 320×320 (kích hoạt cả JIT numba của pymatting), cập nhật `_WARMUP` (FR-106)
  - Sidebar: hiện `Thiết bị: <name> · <backend> (<runtime>/<dtype>)` và danh sách backend khả dụng; `st.selectbox("Thiết bị", ["auto", "cpu", "gpu"])` gửi kèm mọi request (đổi **backend** cần khởi động lại backend với `ACCEL_BACKEND` khác — sidebar chỉ ghi chú điều này, không đổi được); khi `warmup == "running"` hiện `st.info("Đang chuẩn bị model…")` và không tự chạy chế độ tự động

- [ ] **Step 1: Viết test thất bại**

```python
# dùng fastapi.testclient.TestClient
def test_health_reports_device_and_warmup():
    assert {"device", "backend", "available_backends", "warmup"} <= set(client.get("/health").json())

def test_options_reports_devices_and_default_engine():
    assert j["devices"] == ["auto", "cpu", "gpu"] and j["default_engine"] in j_engine_names

def test_process_rejects_unknown_device():
    assert client.post(f"/process/{sid}?device=cuda").status_code == 422
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `python -m pytest tests/test_api_device.py -v`
Expected: FAIL — `/health` không có khoá `device`

- [ ] **Step 3: Hiện thực trong `backend/main.py` và `frontend/app.py`**

Dùng `lifespan` của FastAPI (không dùng `@app.on_event`, đã deprecated). Warm-up không được làm `/health` bị chặn → chạy thread, `/health` chỉ đọc `_WARMUP`.

- [ ] **Step 4: Chạy test, xác nhận pass + kiểm thử tay**

Run: `python -m pytest tests/ -v` → PASS
Run: `uvicorn backend.main:app --port 8000` rồi `streamlit run frontend/app.py` → sidebar hiện thiết bị, đổi `cpu`/`gpu` chạy lại được, lần khởi động thứ hai nhanh hơn nhờ cache (ghi số đo cho NFR-65).

- [ ] **Step 5: Commit**

```bash
git add backend/main.py frontend/app.py tests/test_api_device.py
git commit -m "feat(api): báo thiết bị + ép CPU/GPU, warm-up khi khởi động (FR-105, FR-106)"
```

---

### Task B5: Trace và PDF ghi thiết bị cho từng bước nặng

**Files:**
- Modify: `backend/pipeline/detect.py`, `backend/pipeline/segment/birefnet.py`, `backend/pipeline/matting.py`, `backend/report.py`
- Create: `tests/test_trace_device.py`

**Interfaces:**
- Consumes: `Device.to_dict()` (B1).
- Produces: `Stage` trong `trace.py` thêm trường `device: str = ""`; `_Handle` thêm method `set_device(self, device: Device) -> None`; `Trace.to_list()` thêm khoá `"device"`; `report.stage_grid` in `device` dưới nhãn bước; mọi stage nặng (`yolo`, `birefnet`, `matting`, `rvm`, `modnet`) gọi `s.set_device(dev)` (FR-107).

- [ ] **Step 1: Viết test thất bại**

```python
def test_stage_records_device():
    assert any(s["device"] for s in trace.to_list() if s["key"] == "yolo")

def test_to_list_has_device_key_for_every_stage():
    assert all("device" in s for s in trace.to_list())
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `python -m pytest tests/test_trace_device.py -v`
Expected: FAIL — `KeyError: 'device'`

- [ ] **Step 3: Hiện thực**

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `python -m pytest tests/ -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline/trace.py backend/pipeline/detect.py backend/pipeline/segment/birefnet.py backend/pipeline/matting.py backend/report.py tests/test_trace_device.py
git commit -m "feat(trace): ghi thiết bị cho từng bước nặng vào trace và PDF (FR-107)"
```

---

### Task B6: Nghiệm thu Phase B + tài liệu môi trường

**Files:**
- Modify: `README.md`, `docs/SRS.md`, `docs/SRS-addendum-v1.1.md`
- Create: `tests/test_gpu_interrupt.py`

**Interfaces:**
- Consumes: `tools/bench.py`, `tools/eval_quality.py` (A8); `with_fallback` (B1).
- Produces: README mục "Gia tốc phần cứng" gồm (a) bảng backend × trạng thái kiểm thử — mỗi backend ghi **"đã kiểm thử"** kèm phiên bản driver/runtime thật (Intel Graphics + OpenVINO; driver NVIDIA + CUDA + onnxruntime-gpu; macOS + torch + onnxruntime) **hoặc "chưa kiểm thử trên phần cứng thật"** (chỉ có test giả lập) — không được ghi "hỗ trợ" cho backend chưa chạy thật; (b) lệnh cài gói (`requirements-intel.txt` / `-cuda.txt` / `-mac.txt`) và lệnh chạy cho Windows/PowerShell (`$env:ACCEL_BACKEND="cuda"; uvicorn ...`) và bash (`ACCEL_BACKEND=mps uvicorn ...`); (c) mô tả `ACCEL_BACKEND`, `ACCEL_DEVICE`, `ACCEL_DEVICE_INDEX`, `ACCEL_DTYPE`, `ACCEL_CACHE_DIR`, `ACCEL_MEM_BUDGET_MB`, `ACCEL_UNLOAD_IDLE`; `docs/SRS.md` cập nhật R1, FR-31, NFR-01..03 và mục 7.1 theo số đo mới.

- [ ] **Step 1: Viết test thất bại (mô phỏng ngắt GPU giữa chừng)**

```python
def test_pipeline_survives_gpu_dying_midway(monkeypatch):
    # gpu_fn raise RuntimeError("device lost") ở lần gọi thứ hai
    # -> build_cutout vẫn trả Cutout đầy đủ, trace có cảnh báo, không raise
    assert c.alpha.shape == scene.rgb.shape[:2] and any("CPU" in s.note for s in c.trace.stages)
```

- [ ] **Step 2: Chạy test, xác nhận thất bại hoặc pass có lý do**

Run: `python -m pytest tests/test_gpu_interrupt.py -v`
Expected: PASS nếu B1–B3 đã đúng; FAIL thì sửa chỗ còn để lỗi GPU thoát ra ngoài.

- [ ] **Step 3: Đo và ghi số thật**

Run: `python tools/bench.py --dir ../dataset/hd_bank_green_marathon_2026_resized --limit 30 --engine cf,birefnet --device cpu` rồi `--device gpu` với `ACCEL_BACKEND=<backend>` cho **từng backend có phần cứng thật trên máy đo** (backend không có phần cứng thì bỏ qua, không đoán số).
Ghi trung vị + p95 từng bước vào bảng 5.7: cột CPU và một cột GPU **cho mỗi backend đã đo**, ghi rõ tên máy/driver/runtime. Mục nào không đạt mục tiêu thì ghi số thật kèm một dòng lý do — **không** sửa bảng mục tiêu để khớp số đo.

- [ ] **Step 4: Cập nhật README + SRS**

- [ ] **Step 5: Commit**

```bash
git add README.md docs/SRS.md docs/SRS-addendum-v1.1.md tests/test_gpu_interrupt.py
git commit -m "docs: số đo từng backend đã chạy thật, bảng trạng thái kiểm thử, cập nhật R1 và 7.1 (4.11)"
```

---

# Phase C — Engine nhẹ (FR-110..117)

### Task C1: Engine `rvm`

**Files:**
- Create: `backend/pipeline/segment/light.py`
- Modify: `backend/pipeline/segment/__init__.py`, `backend/pipeline/config.py`
- Create: `tests/test_light_rvm.py`

**Interfaces:**
- Consumes: `active_roi`, `Roi.to_work`, `upsample_alpha` (A1, A2); `finish_cutout` (A5); `gate_to_person` (A6); `exclusion_mask` (có sẵn); `accel.resolve`, `accel.with_fallback` (B1), `accel.onnx_runner` (B3).
- Produces:
  - `light.onnx_alpha(session_key: str, onnx_path: str, feeds_fn, device) -> np.ndarray` — khối dùng chung: lấy runner từ `accel.onnx_runner` (mọi backend, tự rơi về CPU qua `with_fallback`) rồi chạy, trả α float32 0..1
  - `@ENGINES.register("rvm", label="RVM (nhanh, nhẹ)")` → `run_rvm(scene, person, params) -> Cutout`
  - `light.cutout_from_alpha(scene, person, params, roi, alpha_work) -> Cutout` — dùng chung cho `rvm`/`modnet`: `gate_to_person` → trimap theo ngưỡng (0.01/0.99) chỉ để debug (FR-113) → `finish_cutout`
  - Hằng số mới: `RVM_ONNX = os.environ.get("RVM_ONNX", "")`, `RVM_PASSES = int(os.environ.get("RVM_PASSES", "1"))` (kẹp 1..3), `RVM_DOWNSAMPLE = float(os.environ.get("RVM_DOWNSAMPLE", "0.25"))`, `MODNET_ONNX = os.environ.get("MODNET_ONNX", "")`
  - Thiếu trọng số → raise `RuntimeError` với thông điệp tiếng Việt chỉ rõ biến môi trường và nơi tải (API trả 500 kèm thông điệp này, không làm sập backend — NFR-10)

- [ ] **Step 1: Viết test thất bại** (dùng model ONNX giả: `onnx_alpha` được monkeypatch)

```python
def test_rvm_registered_and_listed():
    assert "rvm" in ENGINES.names() and any(e["name"] == "rvm" for e in available())

def test_rvm_returns_full_cutout():
    # FR-113: cả 4 thành phần đúng shape, trimap có đủ 3 mức khi α có viền mềm
    assert set(np.unique(c.trimap)) <= {0, 128, 255} and (c.trimap == 128).any()

def test_rvm_alpha_limited_to_selected_person():            # FR-111
    assert c.alpha[other_person_region].max() == 0.0

def test_rvm_excludes_nearer_other_person():                # FR-114
    assert c.alpha[overlap_with_nearer_person].max() == 0.0

def test_rvm_runs_on_work_resolution():
    assert seen_input_shape[:2] == (roi.work_size[1], roi.work_size[0])

def test_rvm_passes_feeds_recurrent_state_back(monkeypatch):
    # RVM_PASSES = 3 -> infer gọi 3 lần, lần sau nhận state khác rỗng của lần trước
    assert infer_calls == 3 and not np.allclose(states[1], 0)

def test_rvm_missing_weights_raises_vietnamese_hint(monkeypatch):
    with pytest.raises(RuntimeError, match="RVM_ONNX"): ...
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `python -m pytest tests/test_light_rvm.py -v`
Expected: FAIL — `ModuleNotFoundError: backend.pipeline.segment.light`

- [ ] **Step 3: Hiện thực `light.py` + import trong `segment/__init__.py`**

RVM: đầu vào `src` (1,3,H,W) float32 0..1 + `r1i..r4i` khởi tạo mảng rỗng (shape động theo model, dùng `np.zeros([1,1,1,1], np.float32)` như bản ONNX chính thức) + `downsample_ratio` (1,) = `C.RVM_DOWNSAMPLE`; đầu ra lấy `pha`. Chạy lặp `RVM_PASSES` lần, mỗi lần nối `r1o..r4o` của lần trước vào `r1i..r4i` (FR-112). Khoá cache: `("light", "rvm", person.idx, C.RVM_PASSES, device.kind)`.

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `python -m pytest tests/ -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline/segment/light.py backend/pipeline/segment/__init__.py backend/pipeline/config.py tests/test_light_rvm.py
git commit -m "feat(light): engine rvm trên ROI qua lớp thiết bị (FR-110..114)"
```

---

### Task C2: Engine `modnet`

**Files:**
- Modify: `backend/pipeline/segment/light.py`
- Create: `tests/test_light_modnet.py`

**Interfaces:**
- Consumes: `light.onnx_alpha`, `light.cutout_from_alpha` (C1).
- Produces: `@ENGINES.register("modnet", label="MODNet (nhanh, nhẹ)")` → `run_modnet(scene, person, params) -> Cutout`; đầu vào (1,3,512,512) chuẩn hoá `(x − 0.5) / 0.5`, đầu ra matte 1 kênh; khoá cache `("light", "modnet", person.idx, device.kind)`.

- [ ] **Step 1: Viết test thất bại**

```python
def test_modnet_registered():
    assert "modnet" in ENGINES.names()

def test_modnet_returns_full_cutout():
    assert c.alpha.shape == scene.rgb.shape[:2]

def test_modnet_input_is_512_square():
    assert seen_input_shape == (1, 3, 512, 512)
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `python -m pytest tests/test_light_modnet.py -v`
Expected: FAIL — `KeyError: Không có engine 'modnet'`

- [ ] **Step 3: Hiện thực (tái dùng toàn bộ khối của C1; chỉ khác tiền/hậu xử lý)**

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `python -m pytest tests/ -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline/segment/light.py tests/test_light_modnet.py
git commit -m "feat(light): engine modnet (FR-110)"
```

---

### Task C3: Chế độ tự động mặc định engine nhẹ + nút "chất lượng cao"

**Files:**
- Modify: `frontend/app.py`
- Create: `tests/test_default_engine.py`

**Interfaces:**
- Consumes: `/options` → `default_engine` (B4).
- Produces: `ENGINE_DEFAULT` lấy từ `OPTS["default_engine"]` (fallback: engine đầu tiên trong danh sách) thay cho `"birefnet"` hiện tại; nút `st.button("Chất lượng cao (BiRefNet)")` đặt `st.session_state["hq_once"] = True` → lượt xử lý kế tiếp dùng `method="birefnet"` rồi xoá cờ; `sig` của chế độ tự động gồm cả cờ này để không chạy lặp (FR-115, FR-51).

- [ ] **Step 1: Viết test thất bại**

```python
# test ở mức backend: /options phải đề xuất engine nhẹ khi có
def test_default_engine_prefers_light_engine():
    assert client.get("/options").json()["default_engine"] == "rvm"

def test_default_engine_falls_back_to_cf_when_light_absent(monkeypatch):
    assert default_engine(["cf", "knn", "birefnet"]) == "cf"
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `python -m pytest tests/test_default_engine.py -v`
Expected: FAIL — `default_engine` trả `"birefnet"`

- [ ] **Step 3: Hiện thực `main.default_engine(names: List[str]) -> str` và sửa sidebar**

- [ ] **Step 4: Chạy test + kiểm thử tay**

Run: `python -m pytest tests/ -v` → PASS
Kiểm thử tay: đổi `band` liên tục ở chế độ tự động không còn kéo BiRefNet; bấm "Chất lượng cao" chạy BiRefNet đúng một lần.

- [ ] **Step 5: Commit**

```bash
git add backend/main.py frontend/app.py tests/test_default_engine.py
git commit -m "feat(ui): mặc định engine nhẹ, BiRefNet chỉ khi chọn hoặc bấm chất lượng cao (FR-115)"
```

---

### Task C4: Nghiệm thu engine nhẹ, so sánh YOLO_WEIGHTS, giấy phép

**Files:**
- Create: `tools/compare_weights.py`
- Modify: `README.md`, `docs/SRS-addendum-v1.1.md`
- Create: `tests/test_compare_weights.py`

**Interfaces:**
- Consumes: `detect_scene` (B2); `tools/eval_quality.py`, `tools/bench.py` (A8).
- Produces: `tools/compare_weights.py --dir <thư mục> --limit 30 --weights yolov8s-seg.pt,yolov8n-seg.pt,yolo11n-seg.pt` → bảng markdown: số ảnh có người, tổng số người, IoU mask so với `yolov8s-seg`, thời gian trung vị (FR-116). README mục "Giấy phép trọng số model" ghi giấy phép **đã xác minh trên repo gốc** của RVM và MODNet kèm link (FR-117).

- [ ] **Step 1: Viết test thất bại**

```python
def test_detection_rate_table_rows_match_weights():
    assert [r["weights"] for r in rows] == ["yolov8s-seg.pt", "yolov8n-seg.pt"]
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `python -m pytest tests/test_compare_weights.py -v`
Expected: FAIL — `ModuleNotFoundError: tools.compare_weights`

- [ ] **Step 3: Hiện thực script**

- [ ] **Step 4: Chạy nghiệm thu thật**

Run: `python tools/eval_quality.py --dir ../dataset/hd_bank_green_marathon_2026_resized --limit 30 --engine rvm --ref-engine birefnet`
Expected: `mean IoU ≥ 0.95`; thời gian `rvm` ≤ 1/5 thời gian `birefnet` trên cùng thiết bị (`tools/bench.py`). Ghi riêng các ca tóc / tay chân mảnh / nền phức tạp (tên ảnh + nhận xét) vào README. **Xác minh giấy phép RVM và MODNet trên repo gốc trước khi commit trọng số hoặc link tải.**

- [ ] **Step 5: Commit**

```bash
git add tools/compare_weights.py tests/test_compare_weights.py README.md docs/SRS-addendum-v1.1.md
git commit -m "docs: nghiệm thu engine nhẹ, bảng YOLO_WEIGHTS, giấy phép model (FR-116, FR-117)"
```

---

# Phase D — Bản đồ nét theo miền tần số (FR-120..126)

Độc lập với Phase B và C; có thể làm song song sau Phase A.

### Task D1: Bản đồ nét DCT vector hoá

**Files:**
- Create: `backend/pipeline/focusmap.py`
- Modify: `backend/pipeline/config.py`
- Create: `tests/test_focusmap.py`

**Interfaces:**
- Produces:
  - `dct_basis(n: int) -> np.ndarray` — ma trận DCT-II trực chuẩn `n×n`
  - `dct_sharpness(gray_f32: np.ndarray, block: int = C.FOCUS_BLOCK, stride: int = C.FOCUS_STRIDE, drop_high: bool = False, noise_sigma: float = 0.0) -> np.ndarray` — bản đồ nét trên **lưới khối** (float32, giá trị 0..1)
  - `freq_map(rgb: np.ndarray, drop_high: bool = False, noise_sigma: float = 0.0) -> np.ndarray` — full `HxW` float32 0..1: xám → thu về cạnh dài `C.FOCUS_MAX_SIDE` → `dct_sharpness` → làm mịn Gaussian → `cv2.resize` về kích thước ảnh
  - Hằng số: `FOCUS_METHOD = os.environ.get("FOCUS_METHOD", "laplacian")`, `FOCUS_BLOCK = 16`, `FOCUS_STRIDE = 8`, `FOCUS_MAX_SIDE = 640`, `FOCUS_HYBRID_W = 0.5`, `FOCUS_CONF_MIN = 1.3`

- [ ] **Step 1: Viết test thất bại**

```python
def test_dct_basis_matches_cv2_dct():
    D = dct_basis(16)
    assert np.allclose(D @ blk @ D.T, cv2.dct(blk), atol=1e-4)

def test_sharp_region_scores_higher_than_blurred_region():
    # nửa phải của ảnh nhiễu trắng bị GaussianBlur sigma 3
    assert m[:, :W // 2].mean() > 2 * m[:, W // 2:].mean()

def test_flat_image_is_finite_and_zero():            # Review Focus #4
    assert np.isfinite(m).all() and m.max() == 0.0

def test_output_shape_equals_input():
    assert freq_map(rgb).shape == rgb.shape[:2]

def test_noise_sigma_suppresses_noise_only_region():  # FR-122
    assert freq_map(noisy, noise_sigma=8.0)[noise_region].mean() < freq_map(noisy)[noise_region].mean()

def test_runs_under_150ms_at_640px():                 # FR-124, NFR-64
    assert median_of_5_runs < 0.150
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `python -m pytest tests/test_focusmap.py -v`
Expected: FAIL — `ModuleNotFoundError: backend.pipeline.focusmap`

- [ ] **Step 3: Hiện thực `dct_sharpness` vector hoá (không vòng lặp Python trên từng khối)**

Lấy khối chồng 50% bằng `np.lib.stride_tricks.sliding_window_view(gray, (block, block))[::stride, ::stride]` → `(gy, gx, block, block)`. DCT 2D theo lô: `coef = D @ X @ D.T` bằng `np.matmul` (giữ đúng kết quả `cv2.dct`, test Step 1 chốt điều này). Năng lượng: `E = coef ** 2`; `AC` = tất cả trừ `(0,0)`; dải cao = hệ số có `u + v >= block`; khi `drop_high` thì bỏ `u + v >= 2 * block - 4`. Trừ nền nhiễu: mỗi dải trừ `noise_sigma ** 2 * <số hệ số của dải>`, kẹp ≥ 0. Độ nét khối = `log1p(E_high / (E_ac + 1e-6))`, chuẩn hoá bằng `p99` của chính bản đồ, kẹp 0..1; `E_ac <= 0` → 0.

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `python -m pytest tests/test_focusmap.py -v`
Expected: PASS (6 test)

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline/focusmap.py backend/pipeline/config.py tests/test_focusmap.py
git commit -m "feat(focusmap): bản đồ nét DCT 16x16 vector hoá, chống nhiễu (FR-120, FR-122, FR-124)"
```

---

### Task D2: Chọn phương pháp đo nét (`laplacian | freq | hybrid`)

**Files:**
- Modify: `backend/pipeline/focusmap.py`, `backend/pipeline/segment/focus.py`
- Create: `tests/test_focus_method.py`

**Interfaces:**
- Consumes: `freq_map` (D1); `focus_energy` (chuyển từ `segment/focus.py` sang `focusmap.py` ở bước này — `segment/focus.py` import lại từ `focusmap` để không vỡ engine).
- Produces:
  - `laplacian_map(rgb: np.ndarray) -> np.ndarray` — `focus_energy(rgb)[0]` chuẩn hoá về 0..1 bằng `p99`
  - `sharpness_map(rgb: np.ndarray, method: Optional[str] = None, quality: Optional[dict] = None) -> Tuple[np.ndarray, str]` — `method` `None` → `C.FOCUS_METHOD`; `hybrid` = `FOCUS_HYBRID_W * freq + (1 - FOCUS_HYBRID_W) * laplacian`; cờ `noisy`/`dark` trong `quality["flags"]` → bật `drop_high=True` và `noise_sigma=quality["noise"]` (FR-122); trả thêm tên phương pháp đã dùng
  - `focus_energy(rgb) -> Tuple[np.ndarray, np.ndarray]` nay sống ở `focusmap.py` (chữ ký không đổi)

- [ ] **Step 1: Viết test thất bại**

```python
def test_default_method_is_laplacian():               # FR-121
    assert sharpness_map(rgb)[1] == "laplacian"

def test_freq_method_selected_by_config(monkeypatch):
    assert sharpness_map(rgb, "freq")[1] == "freq"

def test_hybrid_is_weighted_mean_of_both():
    assert np.allclose(h, 0.5 * f + 0.5 * l, atol=1e-5)

def test_noisy_quality_enables_noise_suppression(monkeypatch):
    # quality flags có "noisy" -> freq_map được gọi với noise_sigma > 0 và drop_high=True
    assert seen["noise_sigma"] > 0 and seen["drop_high"] is True

def test_focus_engine_still_imports_focus_energy():
    from backend.pipeline.segment.focus import focus_energy  # không vỡ import cũ
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `python -m pytest tests/test_focus_method.py -v`
Expected: FAIL — `ImportError: cannot import name 'sharpness_map'`

- [ ] **Step 3: Hiện thực**

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `python -m pytest tests/ -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline/focusmap.py backend/pipeline/segment/focus.py tests/test_focus_method.py
git commit -m "feat(focusmap): FOCUS_METHOD laplacian|freq|hybrid (FR-121, FR-122)"
```

---

### Task D3: Engine `focus` và `select` dùng bản đồ nét chung

**Files:**
- Modify: `backend/pipeline/segment/focus.py`, `backend/pipeline/select.py`
- Create: `tests/test_focus_shared_map.py`

**Interfaces:**
- Consumes: `sharpness_map` (D2); `SegParams.cached` (A3).
- Produces:
  - `focus_mask(rgb, person, init, frac=0.3, others=None, smap: Optional[np.ndarray] = None) -> np.ndarray` — dùng `smap` (mặc định `sharpness_map(rgb)[0]`) thay cho `E`; ngưỡng vẫn là **phân vị** của chính bản đồ (`thr = max(frac * np.percentile(smap[core], 75), floor)`) nên ngữ nghĩa `focus_frac` 10–100% không đổi (FR-123)
  - `score_persons(scene: Scene, smap: Optional[np.ndarray] = None) -> list` và `suggest_person(scene: Scene, smap: Optional[np.ndarray] = None) -> int` — nhận bản đồ dùng chung (FR-125)
  - Stage `focus_map` nay ghi `note = f"phương pháp {method} · {ms:.0f} ms"` và vẽ heatmap (FR-125)
  - Khoá cache: `("focusmap", C.FOCUS_METHOD)`

- [ ] **Step 1: Viết test thất bại**

```python
def test_focus_frac_semantics_unchanged_for_laplacian():
    # cùng ảnh, frac 0.3 vs 0.8 -> mask của 0.8 là tập con của 0.3
    assert (m08 & ~m03).sum() == 0

def test_focus_map_stage_notes_method_and_time():
    assert "phương pháp" in stage.note and "ms" in stage.note

def test_select_accepts_shared_map_and_matches_internal(monkeypatch):
    assert suggest_person(scene, smap) == suggest_person(scene)

def test_focus_map_computed_once_per_segment_run(monkeypatch):
    assert sharpness_map_calls == 1
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `python -m pytest tests/test_focus_shared_map.py -v`
Expected: FAIL — `TypeError: focus_mask() got an unexpected keyword argument 'smap'`

- [ ] **Step 3: Hiện thực**

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `python -m pytest tests/ -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline/segment/focus.py backend/pipeline/select.py tests/test_focus_shared_map.py
git commit -m "feat(focus): dùng bản đồ nét chung cho engine focus và chọn người (FR-123, FR-125)"
```

---

### Task D4: Cảnh báo mức tin cậy của engine `focus`

**Files:**
- Modify: `backend/pipeline/focusmap.py`, `backend/pipeline/segment/focus.py`, `backend/main.py`, `frontend/app.py`
- Create: `tests/test_focus_confidence.py`

**Interfaces:**
- Consumes: `sharpness_map` (D2).
- Produces:
  - `focus_confidence(smap: np.ndarray, person_mask: np.ndarray) -> float` — tỉ lệ `median(smap[người]) / median(smap[nền])` (nền = ngoài mask người nở 5%); `< C.FOCUS_CONF_MIN` nghĩa là ảnh không có phông mờ rõ; ảnh phẳng (hai median = 0) trả `1.0`
  - Engine `focus` ghi `note` của stage `focus_mask` thêm `· độ tin cậy x.xx` và đặt `params.trace.meta["focus_confidence"] = value` (thêm `Trace.meta: dict` vào `trace.py`)
  - `ProcessOut` thêm `focus_confidence: Optional[float] = None`; frontend hiện `st.warning("Ảnh này không có phông mờ rõ — kết quả 'mặt phẳng nét' có thể kém tin cậy. Thử engine khác.")` khi giá trị `< FOCUS_CONF_MIN` (FR-126)

- [ ] **Step 1: Viết test thất bại**

```python
def test_confidence_high_when_background_is_blurred():
    assert focus_confidence(smap, person_mask) > 2.0

def test_confidence_low_when_background_is_sharp():
    assert focus_confidence(smap_sharp_bg, person_mask) < C.FOCUS_CONF_MIN

def test_confidence_on_flat_image_is_one():            # Review Focus #4
    assert focus_confidence(np.zeros((100, 100), np.float32), mask) == 1.0

def test_process_response_includes_focus_confidence():
    assert client.post(f"/process/{sid}?method=focus").json()["focus_confidence"] is not None
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `python -m pytest tests/test_focus_confidence.py -v`
Expected: FAIL — `ImportError: cannot import name 'focus_confidence'`

- [ ] **Step 3: Hiện thực**

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `python -m pytest tests/ -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline/focusmap.py backend/pipeline/segment/focus.py backend/main.py frontend/app.py backend/pipeline/trace.py tests/test_focus_confidence.py
git commit -m "feat(focus): báo mức tin cậy mặt phẳng nét ra giao diện (FR-126)"
```

---

### Task D5: Nghiệm thu bản đồ nét và quyết định mặc định

**Files:**
- Modify: `backend/pipeline/config.py`, `README.md`, `docs/SRS-addendum-v1.1.md`, `docs/SRS.md`

**Interfaces:**
- Consumes: `tools/bench.py`, `tools/eval_quality.py` (A8).
- Produces: giá trị `FOCUS_METHOD` mặc định cuối cùng (giữ `laplacian` hoặc đổi sang `freq`/`hybrid`) kèm số đo lý giải; mục 7.1 của SRS cập nhật phần quầng màu quanh chân.

- [ ] **Step 1: Dựng tập đánh giá có gán nhãn tay**

Chọn 20 ảnh xoá phông từ `dataset/gold_set` + `dataset/mu_cang_chai_2026_resized`, ghi `docs/focus_eval.md`: mỗi ảnh một dòng `tên ảnh | số người đúng mặt phẳng nét (gán tay) | ghi chú`.

- [ ] **Step 2: Đo cả ba phương pháp**

Run: `FOCUS_METHOD=laplacian python tools/bench.py --dir ../dataset/gold_set --limit 12 --engine focus` (và `freq`, `hybrid`)
Ghi: thời gian bản đồ nét (phải ≤ 150 ms ở 640 px), tỉ lệ người đúng mặt phẳng nét được giữ, số ảnh còn quầng màu quanh chân.

- [ ] **Step 3: Quyết định mặc định**

Đổi `FOCUS_METHOD` mặc định **chỉ khi** phương pháp mới giữ tỉ lệ người đúng không thấp hơn `laplacian` **và** giảm số ảnh có quầng màu. Nếu không đạt: giữ `laplacian` và ghi rõ lý do trong README.

- [ ] **Step 4: Chạy lại toàn bộ test**

Run: `python -m pytest tests/ -v`
Expected: PASS (toàn bộ)

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline/config.py README.md docs/focus_eval.md docs/SRS-addendum-v1.1.md docs/SRS.md
git commit -m "docs: nghiệm thu bản đồ nét tần số, chốt FOCUS_METHOD mặc định (4.13)"
```

---

## Bảng đối chiếu yêu cầu → task

| Yêu cầu | Task |
|---|---|
| FR-90 | A1, A4, A6, A7 |
| FR-91 | A2, A4 |
| FR-92 | A2, A4 |
| FR-93 | A5 |
| FR-94 | A3 (hạ tầng), A4/A6/A7 (khoá cache từng engine) |
| FR-95 | A1 (`active_roi`), A4 (test) |
| FR-96 | A7 (stage `roi` + PDF) |
| FR-100 | B1 |
| FR-101 | B1, B2, B3 |
| FR-102 | B2 |
| FR-103 | B3 |
| FR-104 | B1, B6 |
| FR-105 | B4 |
| FR-106 | B4 |
| FR-107 | B5 |
| FR-108 | B1 |
| Ràng buộc ≤ 3 GB / `ACCEL_UNLOAD_IDLE` / FP16 | B1 |
| README bảng backend + driver đã kiểm thử, cách chọn backend lúc chạy | B6 |
| FR-110 | C1, C2 |
| FR-111 | C1 |
| FR-112 | C1 |
| FR-113 | C1 |
| FR-114 | C1 |
| FR-115 | C3 |
| FR-116 | C4 |
| FR-117 | C4 |
| FR-120 | D1 |
| FR-121 | D2, D5 |
| FR-122 | D1, D2 |
| FR-123 | D3 |
| FR-124 | D1 |
| FR-125 | D3 |
| FR-126 | D4 |
| NFR-60..66 (5.7) | A8 (công cụ + cột CPU), B6 (cột GPU), D5 (NFR-64) |
| Nghiệm thu 4.10 | A8 |
| Nghiệm thu 4.11 | B3 (IoU FP16), B6 (ngắt GPU) |
| Nghiệm thu 4.12 | C4 |
| Nghiệm thu 4.13 | D1 (thời gian), D5 (chất lượng) |
