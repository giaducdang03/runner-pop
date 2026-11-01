# Kế hoạch triển khai: Bộ lọc nền sau khi tách người

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Go**Goal:** Cho phép chọn bộ lọc áp lên nền bên dưới người đã tách (đen trắng, mờ, bokeh, sepia, ấm, lạnh, mosaic, phác họa, tối góc, màu trơn) thay cho hiệu ứng đen trắng duy nhất hiện nay, chọn bằng lưới ô xem trước từ chính ảnh của người dùng và một thanh cường độ chung.

**Architecture:** Thêm `pipeline/bgfilter.py` (registry `BG_FILTERS`, cùng kiểu `ENGINES`/`STEPS`). Backend dựng nền đã lọc `Bf` ở cường độ đầy đủ, cache trong `Cutout`; công thức ghép tổng quát hoá thành `out = α·F + (1−α)·((1−s)·B + s·Bf)`. Viewer tải `Bf` như một ảnh thành phần và chỉ trộn trong JS, nên JS không phải cài lại từng bộ lọc. Giao diện chọn là lưới ô (ảnh xem trước do backend dựng trên bản thu nhỏ của chính cutout) và một thanh cường độ Streamlit duy nhất dùng cho viewer, ảnh tải về và PDF. Các bộ lọc lấy mẫu lân cận dùng trung bình chuẩn hoá theo mặt nạ nền hợp lệ để không lem màu người ra nền.

**Tech Stack:** Python 3.13, NumPy, OpenCV (`GaussianBlur`, `filter2D`, `resize`), FastAPI, Streamlit, pytest. Không thêm thư viện mới.

**Spec:** [docs/SRS.md](../../SRS.md) mục 4.14 (FR-130 … FR-136), cùng 3.2, 4.5 (FR-40), 6.1, 6.3.

## Global Constraints

- Bộ lọc `bw` + cường độ `s` phải cho kết quả **giống hệt** công thức cũ `(1−s)·B + s·gray(B)` (FR-131, FR-135); không truyền `filter` thì mọi endpoint hành xử như trước.
- `Bf` luôn float32, shape `(H, W, 3)`, giá trị 0..1, không NaN/inf (FR-132).
- Đổi bộ lọc không chạy lại pipeline tách nền và không thêm bước vào `Trace` (FR-136).
- Tên bộ lọc lạ hoặc `color` không phải `#RRGGBB` → HTTP `400` (FR-133).
- Mặc định: `filter="bw"`, `color="#808080"` (FR-135). Bộ lọc `none` ("Ảnh gốc") trả nền giữ nguyên, chỉ dùng trên lưới xem trước và để quay về ảnh gốc.
- Chú thích/docstring trong code bằng tiếng Việt; commit message bằng tiếng Anh.
- Không viết benchmark hay đo hiệu năng; chỉ cần chạy được và test pass.

## Review Focus

- Mờ/bokeh/mosaic khi người chiếm gần hết khung (hầu như không có pixel nền hợp lệ): không NaN, không ra ảnh đen → Task 1.
- Ảnh có cạnh lẻ (vd 101×77): mosaic và bokeh không lỗi shape → Task 1.
- Quầng sáng quanh người khi làm mờ (màu áo lem sang nền) → Task 1.
- `color` sai định dạng hoặc bộ lọc không tồn tại → `400`, không `500` → Task 3.
- Cường độ 0 trả đúng nền gốc với mọi bộ lọc; đổi bộ lọc xong tách lại người thì không dùng cache cũ → Task 2.

---

## File Structure

| File | Việc |
|---|---|
| `backend/pipeline/bgfilter.py` (mới) | `BG_FILTERS`, các bộ lọc, `parse_color`, `apply_bg_filter` |
| `backend/pipeline/schema.py` | `Cutout.filtered` cache |
| `backend/pipeline/compose.py` | `compose(..., bg_filter, color)`, `thumbnail_cutout` |
| `backend/pipeline/__init__.py` | xuất `BG_FILTERS`, `apply_bg_filter` |
| `backend/main.py` | `/options`, `/bgfilter.jpg`, `/bgfilter_previews`, tham số `filter`/`color` cho `/compose`, `/report.pdf` |
| `backend/report.py` | ghi bộ lọc đã dùng, ghép đúng bộ lọc |
| `frontend/app.py` | ô chọn bộ lọc, chọn màu, viewer dùng `Bf` |
| `tests/test_bgfilter.py`, `tests/test_compose_filter.py`, `tests/test_api_bgfilter.py` (mới) | test |

---

### Task 1: Bộ lọc nền (lõi, chưa nối vào API)

**Files:**
- Create: `backend/pipeline/bgfilter.py`
- Modify: `backend/pipeline/schema.py` (thêm trường vào `Cutout`), `backend/pipeline/__init__.py`
- Test: `tests/test_bgfilter.py`

**Interfaces:**
- Produces:
  - `BG_FILTERS = Registry("bgfilter")`; mỗi bộ lọc: `fn(bg: np.ndarray, valid: np.ndarray, *, color: Tuple[float, float, float] = (0.5, 0.5, 0.5), **_) -> np.ndarray`. `bg` float32 HxWx3 0..1, `valid` bool HxW (True = pixel nền tin cậy), trả float32 HxWx3 0..1.
  - `parse_color(text: str) -> Tuple[float, float, float]` — nhận `#RRGGBB`, ngược lại `raise ValueError`.
  - `apply_bg_filter(c: Cutout, name: str, color: str = "#808080") -> np.ndarray` — `valid = c.alpha < 0.999`; cache trong `c.filtered[(name, color)]`; tên lạ → `KeyError` (từ `Registry.get`); trả đúng một đối tượng cho cùng khoá.
  - `Cutout.filtered: Dict[tuple, np.ndarray] = field(default_factory=dict, repr=False)`.
- Registry tên và nhãn (thứ tự hiện trên UI): `none` "Ảnh gốc", `bw` "Đen trắng", `blur` "Làm mờ", `bokeh` "Bokeh", `sepia` "Sepia", `warm` "Tông ấm", `cool` "Tông lạnh", `mosaic` "Mosaic", `sketch` "Phác họa", `vignette` "Tối góc", `solid` "Màu trơn"; thêm `none` "Ảnh gốc" đứng đầu danh sách (`fn` trả `bg` nguyên vẹn).

- [ ] **Step 1: Viết test (file `tests/test_bgfilter.py`)**. Dùng helper `_cutout(h=120, w=160)` dựng `Cutout` tổng hợp: nền xanh lá `(0,1,0)`, người là khối chữ nhật đỏ `(1,0,0)` ở giữa với `alpha=1` trong khối và `0` ngoài; `foreground = image = background` có khối đỏ (giống ảnh thật: `B` trong thân người vẫn là màu người). Các test:
  - `test_bw_matches_gray3`: `apply_bg_filter(c, "bw")` bằng `to_gray3(c.background)` (hàm trong `compose.py`).
  - `test_blur_does_not_leak_person_color`: với `blur`, `bokeh`, `mosaic`, pixel nền sát khối đỏ (cách 3 px) có kênh đỏ `< 0.02` và kênh xanh `> 0.9`.
  - `test_none_returns_background`: `apply_bg_filter(c, "none")` bằng `c.background`.
  - `test_all_filters_valid_output`: mọi tên trong `BG_FILTERS.names()` trên ảnh lẻ 101×77: dtype float32, shape đúng, hữu hạn, trong [0,1].
  - `test_person_fills_frame_no_nan`: `alpha = 1` ở mọi nơi trừ 2 pixel → mọi bộ lọc hữu hạn.
  - `test_solid_uses_color`: `apply_bg_filter(c, "solid", "#ff8000")` toàn ảnh ≈ `(1, 0.502, 0)`.
  - `test_parse_color_rejects_bad`: `"red"`, `"#12"`, `"#GG0000"`, `""` → `ValueError`.
  - `test_apply_is_cached`: gọi hai lần cùng khoá trả cùng đối tượng (`is`); khác `color` cho `solid` thì khác.

- [ ] **Step 2: Chạy `pytest tests/test_bgfilter.py -q`**, kỳ vọng FAIL (`ModuleNotFoundError: backend.pipeline.bgfilter`).

- [ ] **Step 3: Cài `bgfilter.py` và trường `Cutout.filtered`.**
  - Hàm dùng chung `_masked(bg, valid, op)`: `w = valid.astype(float32)`; `num = op(bg * w[...,None])`, `den = op(w)`; kết quả `num / den` khi `den > 1e-3`, ngược lại giữ `bg` (không chia cho 0). `op` là hàm lọc trên mảng HxWxC hoặc HxW.
  - `blur`: `op = GaussianBlur(sigma = 0.012·max(H,W))`. `bokeh`: `op = filter2D` với nhân đĩa bán kính `max(3, round(0.012·max(H,W)))` chuẩn hoá tổng 1.
  - `mosaic`: khối `max(8, round(0.02·max(H,W)))`; thu nhỏ `INTER_AREA` cả `bg·w` lẫn `w` rồi chia như trên, phóng lại `INTER_NEAREST` về đúng `(H, W)` (cạnh lẻ không được lệch shape).
  - `sketch`: xám của `bg` → nghịch đảo → làm mờ có mặt nạ → color-dodge `gray / (1 − blurred_inv)` (kẹp 0..1) → lặp 3 kênh.
  - `sepia`: nhân ma trận sepia chuẩn rồi kẹp. `warm`: nhân kênh `(1.10, 1.00, 0.88)`. `cool`: `(0.88, 1.00, 1.12)`. `vignette`: nhân `1 − 0.6·r²` với `r` chuẩn hoá khoảng cách tới tâm ảnh (0 ở tâm, 1 ở góc). `solid`: ảnh đồng màu `color`. `bw`: dùng `to_gray3` từ `compose.py` (import trong `bgfilter.py`; `compose.py` không import ngược lại ở mức module — xem Task 2).
  - `__init__.py` xuất `BG_FILTERS`, `apply_bg_filter`.

- [ ] **Step 4: Chạy `pytest tests/test_bgfilter.py -q`**, kỳ vọng PASS.

- [ ] **Step 5: Commit** — `git commit -m "Add background filter registry and filters"` (kèm dòng Co-Authored-By theo cấu hình).

---

### Task 2: Ghép với bộ lọc (`compose`)

**Files:**
- Modify: `backend/pipeline/compose.py`
- Test: `tests/test_compose_filter.py`

**Interfaces:**
- Consumes: `apply_bg_filter(c, name, color) -> ndarray` (Task 1).
- Produces: `compose(c: Cutout, intensity: float = 1.0, bg_filter: str = "bw", color: str = "#808080") -> np.ndarray` (uint8 HxWx3). Công thức `a·F + (1−a)·((1−s)·B + s·Bf)`.
- Import: `compose.py` import `apply_bg_filter` **bên trong hàm** (tránh vòng, vì `bgfilter.py` import `to_gray3` từ `compose.py`).

- [ ] **Step 1: Viết test** (dùng lại helper dựng `Cutout` từ Task 1, đặt trong `tests/conftest.py` thành fixture `make_cutout`):
  - `test_default_matches_legacy_formula`: `compose(c, s)` bằng công thức cũ `a·F + (1−a)·((1−s)·B + s·gray(B))` (tính trong test) với `s ∈ {0, 0.3, 1}`.
  - `test_intensity_zero_is_original_background_for_every_filter`: với mọi bộ lọc, `compose(c, 0.0, name)` bằng `compose(c, 0.0, "bw")`.
  - `test_filter_changes_output`: `compose(c, 1.0, "solid", "#ff0000")` có pixel nền ngoài người bằng đỏ, pixel trong người giữ màu người.
  - `test_cache_cleared_with_new_cutout`: hai `Cutout` khác nhau, cùng `solid` khác màu nền gốc → kết quả không dùng chung cache (cache nằm trong `Cutout`).

- [ ] **Step 2: Chạy `pytest tests/test_compose_filter.py -q`**, kỳ vọng FAIL (chữ ký `compose` chưa nhận `bg_filter`).

- [ ] **Step 3: Sửa `compose`** theo chữ ký trên; `bw` mặc định cho kết quả bit-giống cũ. Cập nhật docstring (mô tả `intensity` là mức trộn giữa nền gốc và nền đã lọc).

- [ ] **Step 4: Chạy `pytest tests/test_compose_filter.py tests/test_bgfilter.py -q`**, kỳ vọng PASS.

- [ ] **Step 5: Commit** — `"Make compose accept a background filter"`.

---

### Task 3: API, báo cáo PDF

**Files:**
- Modify: `backend/main.py` (`options`, thêm `bgfilter`, sửa `compose` và `report`), `backend/report.py` (`build_report`)
- Test: `tests/test_api_bgfilter.py` (cùng kiểu fixture `client` như `tests/test_api_device.py`; dựng `Session` có `cutout` tổng hợp bằng `main._put`)

**Interfaces:**
- Consumes: `P.compose(c, intensity, bg_filter, color)`, `P.apply_bg_filter`, `P.BG_FILTERS` (Task 1–2).
- Produces:
  - `GET /options` thêm khoá `"bg_filters": P.BG_FILTERS.describe()`.
  - `GET /session/{sid}/bgfilter_previews?color=%23808080` → JSON `[{"name": str, "label": str, "image": "data:image/jpeg;base64,..."}]`, theo thứ tự `BG_FILTERS`; mỗi ảnh là kết quả ghép đầy đủ (`compose` với cường độ 1) trên bản thu nhỏ của cutout (cạnh dài 240 px). Hàm hỗ trợ `thumbnail_cutout(c: Cutout, max_side: int = 240) -> Cutout` trong `compose.py` (resize α, F, B, image bằng `INTER_AREA`). Kết quả cache trong `Cutout.filtered` theo `("preview", name, color)`; `400` nếu `color` sai, `409` nếu chưa có cutout.
  - `GET /session/{sid}/bgfilter.jpg?filter=bw&color=%23808080` → `image/jpeg` (chất lượng 97); `409` nếu chưa có cutout, `400` nếu `filter` lạ hoặc `color` sai.
  - `GET /session/{sid}/compose?...&filter=&color=` và `GET /session/{sid}/report.pdf?...&filter=&color=`: cùng quy tắc lỗi.
  - `build_report(rgb, scene, cutout, meta, intensity=1.0, engine_labels=None, bg_filter: str = "bw", color: str = "#808080") -> bytes`; trong bảng cài đặt thay dòng "Cường độ nền đen trắng" bằng hai dòng "Bộ lọc nền" (nhãn từ `BG_FILTERS.label`) và "Cường độ hiệu ứng nền".

- [ ] **Step 1: Viết test**:
  - `test_options_lists_bg_filters`: `bg_filters` chứa `bw` và `solid`, mỗi mục có `name`, `label`.
  - `test_bgfilter_jpg_ok`: 200, `content-type == image/jpeg`, giải mã ra đúng `(H, W)` của cutout.
  - `test_previews_list`: 200, đủ tên theo `BG_FILTERS.names()` (kể cả `none`), mỗi `image` giải mã được và rộng ≤ 240 px; `color=red` → 400; chưa tách → 409; gọi hai lần không dựng lại (monkeypatch đếm lời gọi `apply_bg_filter`).
  - `test_bgfilter_errors`: `filter=nope` → 400; `color=red` → 400; session chưa tách → 409.
  - `test_compose_default_unchanged`: `/compose` không tham số trả byte-giống `P.compose(c, 1.0)` mã hoá PNG.
  - `test_compose_with_filter`: `filter=solid&color=%23ff0000` → 200, ảnh khác bản mặc định.
  - `test_report_passes_filter`: monkeypatch `main.build_report` ghi lại đối số; gọi `/report.pdf?filter=blur` → nhận `bg_filter="blur"`; `filter=nope` → 400.

- [ ] **Step 2: Chạy `pytest tests/test_api_bgfilter.py -q`**, kỳ vọng FAIL (404 / thiếu khoá).

- [ ] **Step 3: Cài đặt.** Hàm kiểm tra dùng chung `_bg_args(filter: str, color: str) -> Tuple[str, str]` trong `main.py`: lỗi `KeyError` hoặc `ValueError` → `HTTPException(400, ...)` (tin nhắn tiếng Việt nêu danh sách bộ lọc hiện có / định dạng `#RRGGBB`). Ba endpoint gọi hàm này trước khi tính. `Content-Disposition` của `/compose` và `/report.pdf` giữ nguyên.

- [ ] **Step 4: Chạy `pytest tests -q`**, kỳ vọng PASS (toàn bộ suite, để chắc `bw` mặc định không phá test cũ).

- [ ] **Step 5: Commit** — `"Add background filter API and report support"`.

---

### Task 4: Giao diện (lưới ô bộ lọc + thanh cường độ)

**Files:**
- Modify: `frontend/app.py` (phần viewer và tải về, quanh dòng 360–500)

**Interfaces:**
- Consumes: `GET /session/{sid}/bgfilter_previews`, `GET /session/{sid}/bgfilter.jpg`, `compose`/`report.pdf` có `filter`, `color`.
- Produces: không (lá của chuỗi).

- [ ] **Step 1: Lưới ô "Chỉnh màu nền".** Tiêu đề + chú thích "Chạm để xem trước hiệu ứng trên ảnh của bạn." Lưới 3 cột bằng `st.columns(3)`; mỗi ô là `st.container(border=True)` chứa `st.image(preview, use_container_width=True)` và một `st.button(label, key=f"bgf_{name}", type="primary" nếu đang chọn, ngược lại "secondary", use_container_width=True)`. Bấm → `st.session_state["bg_filter"] = name` rồi rerun. Mặc định `bw`. Rơi về chỉ một ô `bw` (không ảnh xem trước) nếu backend cũ không có `bg_filters`. Khi chọn `solid`, hiện thêm `st.color_picker("Màu nền", "#808080", key="bg_color")` bên dưới lưới.

- [ ] **Step 2: Thanh cường độ chung.** Dưới lưới: `st.slider("Cường độ hiệu ứng", 0, 100, 100, 5, format="%d%%", key="bg_intensity")`, ẩn khi chọn `none`. Đây là thanh duy nhất: bỏ thanh trong viewer ("Cường độ nền đen trắng") và thanh "Cường độ khi tải về". Viewer, ảnh tải về và PDF cùng dùng giá trị này. Thanh cập nhật khi thả (Streamlit dựng lại viewer, không gọi lại pipeline).

- [ ] **Step 3: Lấy dữ liệu.** `previews` và `bf` (base64 của `/bgfilter.jpg?filter=&color=`) chỉ tải lại khi `(sid, filter, color, seg_ver)` đổi, `seg_ver` là bộ đếm tăng mỗi khi có kết quả tách nền mới (đặt cùng chỗ gán `st.session_state["done"]`). Ảnh xem trước chỉ tải lại khi `(sid, color, seg_ver)` đổi.

- [ ] **Step 4: Viewer JS.** Thêm `__BF__` và `__S__` (cường độ ban đầu 0..1) vào `VIEWER`; tải ảnh thứ năm `Bf`; bỏ mảng `gray` và thanh `#s`; công thức `D[k+c] = a*F + (1-a)*((1-s)*B + s*Bf)` cho ba kênh với `s` cố định từ `__S__`. Giữ ô chọn chế độ xem (so sánh/cạnh nhau/…).

- [ ] **Step 5: Tải về.** `/compose` và `/report.pdf` truyền `filter`, `color`, `intensity` đang chọn; tên file không đổi (theo BIB).

- [ ] **Step 6: Chạy thử tay** (không benchmark): khởi động lại `uvicorn` và `streamlit`, tách một ảnh mẫu trong `samples/`; lưới hiện đủ ô với ảnh xem trước đúng hiệu ứng, ô đang chọn nổi bật; bấm từng ô thì viewer đổi; cường độ 0 ra nền gốc; `solid` đổi theo ô chọn màu; ảnh tải về và PDF khớp viewer. Báo kết quả từng bộ lọc.

- [ ] **Step 7: Commit** — `"Add background filter tile grid and intensity slider"`.

---

## Tự kiểm tra kế hoạch với SRS 4.14

- FR-130/131 → Task 1; FR-132 → Task 1 (test rò màu, người chiếm khung, ảnh lẻ); FR-133 → Task 1 (cache) + Task 3 (400); FR-134, FR-137 → Task 3–4; FR-135 → Task 2–3; FR-136 → Task 3 (không đụng `build_cutout`/`Trace`).
 vùng.
