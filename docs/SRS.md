# Đặc tả yêu cầu phần mềm (SRS) – Color Pop Runner

| | |
|---|---|
| **Sản phẩm** | Color Pop Runner – giữ màu người chạy, đưa nền về đen trắng |
| **Phiên bản tài liệu** | 1.0 |
| **Ngày** | 26/10/2026 |
| **Dự án** | MSA – IPV Final Project |
| **Trạng thái** | Mô tả hệ thống đã hiện thực (mục 4 ghi rõ trạng thái từng yêu cầu) |

---

## 1. Giới thiệu

### 1.1 Mục đích
Tài liệu mô tả yêu cầu của Color Pop Runner: ứng dụng tách người chạy khỏi nền trong ảnh thể thao, giữ nguyên màu người chạy và chuyển phần nền sang đen trắng ("color pop"). Tài liệu dùng cho nhóm phát triển, người đánh giá đồ án và người sẽ mở rộng hệ thống sau này.

### 1.2 Phạm vi
Hệ thống nhận **một ảnh tĩnh** (ảnh chụp người chạy tại giải chạy), tự phát hiện người, tách người ra khỏi nền bằng một trong nhiều thuật toán, rồi ghép lại với nền đen trắng có cường độ chỉnh được. Hệ thống chạy **hoàn toàn trên CPU**, một người dùng một lúc (demo / công cụ cá nhân).

**Ngoài phạm vi:** xử lý video, xử lý hàng loạt nhiều ảnh trong một lượt, tài khoản người dùng, lưu trữ lâu dài, huấn luyện model, chạy trên GPU.

### 1.3 Định nghĩa và viết tắt

| Thuật ngữ | Ý nghĩa |
|---|---|
| **YOLO-seg** | Mô hình phát hiện đối tượng kèm mask phân đoạn (ultralytics `yolov8*-seg`) |
| **GrabCut** | Thuật toán tách vùng tiền cảnh/hậu cảnh khởi tạo từ mask (OpenCV) |
| **Trimap** | Ảnh 3 vùng: tiền cảnh chắc chắn (255), chưa biết (128), hậu cảnh chắc chắn (0) |
| **Alpha matting** | Ước lượng độ trong suốt α ∈ [0,1] từng pixel, giữ được tóc, viền mềm (pymatting) |
| **Closed-form (cf) / KNN** | Hai bộ giải matting của pymatting |
| **BiRefNet** | Mô hình phân đoạn đối tượng nổi bật độ chính xác biên cao (chạy qua rembg + onnxruntime) |
| **Khử nhiễm màu (decontamination)** | Ước lượng màu tiền cảnh F và nền B thật ở viền để không lem màu nền cũ |
| **Mặt phẳng nét (focus plane)** | Vùng nằm trong khoảng lấy nét của ống kính, phần còn lại bị bokeh |
| **Scene / Cutout / Trace** | Kết quả phát hiện; kết quả tách nền; nhật ký các bước pipeline kèm ảnh trung gian |
| **Engine** | Một cách tách nền cắm được vào hệ thống (`cf`, `knn`, `focus`, `birefnet`) |

### 1.4 Tài liệu tham chiếu
- [README.md](../README.md) – cài đặt, chạy, cấu trúc code.
- Ultralytics YOLOv8-seg; OpenCV `grabCut`; pymatting (closed-form, KNN, `estimate_foreground_ml`); rembg (BiRefNet); FastAPI; Streamlit; fpdf2.

### 1.5 Bố cục tài liệu
Mục 2 mô tả tổng quan; mục 3 các giao diện ngoài; mục 4 yêu cầu chức năng; mục 5 yêu cầu phi chức năng; mục 6 kiến trúc và luồng dữ liệu; mục 7 giới hạn đã biết và hướng mở rộng.

---

## 2. Mô tả tổng quan

### 2.1 Bối cảnh sản phẩm
Ảnh chụp tại giải chạy thường có nhiều người, nền phức tạp và chất lượng không đồng đều (tối, nhiễu, hơi mờ). Hiệu ứng "color pop" giúp làm nổi bật vận động viên. Hệ thống gồm hai tiến trình độc lập:

```
Trình duyệt ──▶ Streamlit frontend (cổng 8501) ──HTTP──▶ FastAPI backend (cổng 8000) ──▶ pipeline xử lý ảnh (CPU)
```

### 2.2 Chức năng chính
1. Tải ảnh, đánh giá chất lượng và tiền xử lý **chỉ cho bước phát hiện**.
2. Phát hiện người, tự chọn người chạy chính (chế độ tự động) hoặc để người dùng chọn.
3. Tách nền bằng engine được chọn, khử nhiễm màu viền.
4. Ghép kết quả đen trắng với cường độ chỉnh realtime, so sánh trước/sau.
5. Hiển thị **pipeline từng bước theo thời gian thực** (ảnh trung gian, thời gian, ghi chú), phóng to từng ảnh.
6. Tải ảnh kết quả (PNG/JPEG) và **xuất báo cáo PDF** toàn bộ thông tin.

### 2.3 Người dùng

| Người dùng | Mô tả | Nhu cầu |
|---|---|---|
| Người dùng cuối | Nhiếp ảnh gia / vận động viên | Tạo ảnh color pop nhanh, ít thao tác |
| Người demo / chấm đồ án | Giảng viên, hội đồng | Thấy rõ từng bước xử lý, xuất báo cáo |
| Lập trình viên | Người mở rộng hệ thống | Thêm bước tiền xử lý hoặc engine mới mà không sửa nhiều chỗ |

### 2.4 Môi trường vận hành
- Windows / Linux / macOS, Python ≥ 3.10 (đã chạy thử trên Python 3.13, Windows 11).
- CPU thông thường, RAM khuyến nghị ≥ 8 GB, không cần GPU.
- Trình duyệt hiện đại (Chrome, Edge, Firefox) với JavaScript bật.
- Cần Internet ở lần chạy đầu để tải trọng số model (`yolov8s-seg.pt` ≈ 24 MB, BiRefNet-lite ≈ 224 MB).

### 2.5 Ràng buộc thiết kế
- **R1.** Chỉ dùng CPU.
- **R2.** Ảnh kết quả không bao giờ dùng ảnh đã tiền xử lý; tiền xử lý chỉ phục vụ phát hiện để không đổi màu người chạy.
- **R3.** Session lưu trong RAM, tối đa 8 session (LRU), không lưu đĩa.
- **R4.** Ảnh được thu nhỏ về cạnh dài tối đa 1280 px; file tải lên tối đa 25 MB.
- **R5.** Phần nặng chạy một lần cho mỗi (ảnh, người, engine); thao tác chỉnh cường độ phải nhẹ và không gọi lại backend.

### 2.6 Giả định và phụ thuộc
- Ảnh có ít nhất một người đủ rõ để YOLO phát hiện; nếu không, người dùng nhập box thủ công.
- Hệ thống có quyền tải model lần đầu. Thư viện: `torch`, `ultralytics`, `opencv-python`, `pymatting`, `rembg[cpu]`, `fastapi`, `streamlit`, `fpdf2`.
- Máy có font TTF hỗ trợ tiếng Việt (Arial, Segoe UI, DejaVu) để xuất PDF, hoặc đặt `REPORT_FONT`.

---

## 3. Giao diện bên ngoài

### 3.1 Giao diện người dùng (Streamlit)
- **Sidebar:** công tắc Chế độ tự động; chọn tiền xử lý (Tự động / Không / Tự chọn các bước); chọn thuật toán tách nền; thanh trượt độ rộng vùng viền; thanh trượt ngưỡng độ nét (chỉ bật với engine `focus`); trạng thái backend.
- **Khu vực chính:** tải ảnh → dòng chất lượng ảnh → Pipeline 1 → ảnh các box đã phát hiện + chọn người + box thủ công + nút Xử lý → Pipeline 2 → khung xem kết quả → tải ảnh / báo cáo PDF / debug alpha & trimap.
- **Khung xem kết quả:** các chế độ *So sánh (kéo thanh)*, *Cạnh nhau (trái trước, phải sau)*, *Chỉ kết quả*, *Chỉ ảnh gốc*; thanh cường độ nền đen trắng 0–100%; hỗ trợ ảnh ngang và ảnh dọc.
- **Sơ đồ pipeline:** thẻ nằm ngang nối bằng mũi tên; bấm ảnh để mở modal phóng to, phím ← → chuyển bước, Esc đóng.

### 3.2 Giao diện lập trình (REST API – FastAPI)

| Phương thức & đường dẫn | Mô tả |
|---|---|
| `GET /health` | Tình trạng backend, có ultralytics không |
| `GET /options` | Danh sách engine và bước tiền xử lý hiện có |
| `POST /detect?preprocess=auto\|none\|a,b&background=bool` | Tải ảnh, tiền xử lý, phát hiện người, tạo session |
| `POST /process/{id}?person=&method=&band=&focus=&box=&background=bool` | Tách nền cho người chọn |
| `GET /session/{id}/job/{detect\|segment}?since=N` | Tiến độ realtime: bước mới, bước đang chạy, kết quả / lỗi |
| `GET /session/{id}/trace` | Toàn bộ bước pipeline kèm ảnh minh hoạ |
| `GET /session/{id}/preview.jpg` | Ảnh làm việc kèm box đánh số |
| `GET /session/{id}/{original.jpg\|foreground.png\|background.jpg\|alpha.png\|trimap.png}` | Ảnh thành phần |
| `GET /session/{id}/compose?intensity=&fmt=png\|jpeg` | Ảnh kết quả cuối (tải về) |
| `GET /session/{id}/report.pdf?intensity=` | Báo cáo PDF |

Mã lỗi chính: `400` tham số sai / ảnh không đọc được; `404` session không tồn tại; `409` chưa phát hiện / chưa xử lý; `413` ảnh quá lớn; `422` không phát hiện người; `503` thiếu thư viện.

### 3.3 Giao diện phần cứng / phần mềm
Không có phần cứng đặc thù. Mô hình tải từ Internet (Ultralytics, rembg) và lưu cache cục bộ.

---

## 4. Yêu cầu chức năng

Trạng thái: ✔ = đã triển khai.

### 4.1 Nhập ảnh

| ID | Yêu cầu | TT |
|---|---|---|
| FR-01 | Chấp nhận ảnh `jpg`, `jpeg`, `png`, `webp`; từ chối file lớn hơn 25 MB (HTTP 413) và file không đọc được (HTTP 400). | ✔ |
| FR-02 | Tôn trọng hướng EXIF; thu nhỏ ảnh về cạnh dài ≤ 1280 px (giữ tỉ lệ, không phóng to). | ✔ |

### 4.2 Đánh giá chất lượng & tiền xử lý (cho phát hiện)

| ID | Yêu cầu | TT |
|---|---|---|
| FR-10 | Tính các chỉ số: độ sáng, tương phản (p95−p5), độ nét (phương sai Laplacian chuẩn hoá ảnh 640 px), nhiễu (Immerkaer); gắn cờ `dark`, `low_contrast`, `blurry`, `noisy` theo ngưỡng cấu hình. | ✔ |
| FR-11 | Có các bước tiền xử lý đăng ký được: `levels`, `gamma`, `clahe`, `denoise`, `sharpen`. | ✔ |
| FR-12 | Chế độ **auto** tự chọn chuỗi bước theo cờ chất lượng; chế độ **none** bỏ qua; chế độ **tuỳ chọn** nhận danh sách bước theo thứ tự (tên sai → HTTP 400). | ✔ |
| FR-13 | Ảnh tiền xử lý chỉ dùng để phát hiện, tuyệt đối không dùng làm đầu ra. | ✔ |
| FR-14 | Khi `auto` và không phát hiện người, thử lại theo thang (conf thấp hơn, imgsz lớn hơn, chuỗi tiền xử lý mạnh hơn) cho tới khi có kết quả hoặc hết thang. | ✔ |

### 4.3 Phát hiện & chọn người

| ID | Yêu cầu | TT |
|---|---|---|
| FR-20 | Phát hiện người bằng YOLO-seg, trả về box, độ tin cậy, mask độ phân giải gốc; đánh số theo diện tích box giảm dần. | ✔ |
| FR-21 | Hiển thị ảnh kèm box đánh số để người dùng chọn. | ✔ |
| FR-22 | **Tự chọn người chính** theo điểm: kích thước, độ gần tâm ảnh, độ nét và độ tin cậy; người to nhưng mờ (ngoài vùng nét) bị trừ điểm. | ✔ |
| FR-23 | Cho phép nhập **box thủ công** khi YOLO không phát hiện được hoặc chọn sai. | ✔ |

### 4.4 Tách nền (engine)

| ID | Yêu cầu | TT |
|---|---|---|
| FR-30 | Engine `cf` và `knn`: mask YOLO → GrabCut → trimap → alpha matting (closed-form / KNN) → khử nhiễm màu viền. | ✔ |
| FR-31 | Engine `birefnet`: BiRefNet cho alpha toàn ảnh, giới hạn trong mask YOLO của người được chọn (nở ra vài px), loại mảnh rời, khử nhiễm màu. | ✔ |
| FR-32 | Engine `focus` (ảnh xoá phông): tính bản đồ độ nét; giữ màu người được chọn và những người khác **nằm cùng mặt phẳng nét**; chỉ áp dụng cho người (đường, logo, vật thể không phải người bị bỏ). Ngưỡng độ nét chỉnh được (10–100%). | ✔ |
| FR-33 | Độ rộng vùng chưa biết của trimap chỉnh được (0.5%–8% kích thước người). | ✔ |
| FR-34 | Bước matting closed-form phải có phương án dự phòng (KNN, rồi mask làm mềm) khi bộ giải không hội tụ, để không báo lỗi người dùng. | ✔ |
| FR-35 | Giữ lại tay/chân mảnh mà GrabCut dễ cắt nhầm (gộp lõi chắc chắn của mask YOLO, giữ mọi thành phần liên thông chồng lên mask YOLO). | ✔ |
| FR-37 | Với người chồng nhau đã được YOLO tách thành mask riêng: loại phần của người khác khỏi kết quả của người được chọn; thứ tự trước/sau suy ra từ vị trí chân (người có chân thấp hơn trong ảnh ở gần hơn, giữ pixel chồng nhau). Áp dụng cho engine `cf`, `knn`, `birefnet`; không áp dụng cho `focus` (giữ người cùng mặt phẳng nét theo thiết kế). | ✔ |
| FR-36 | Kết quả mỗi lần tách gồm: α, F (đã khử nhiễm màu), B (nền sạch màu người), trimap. | ✔ |

### 4.5 Ghép kết quả & xem

| ID | Yêu cầu | TT |
|---|---|---|
| FR-40 | Công thức ghép: `out = α·F + (1−α)·((1−s)·B + s·gray(B))`, s ∈ [0,1] là cường độ; frontend và backend dùng cùng công thức. | ✔ |
| FR-41 | Thanh cường độ đổi kết quả tức thì trong trình duyệt, không gọi lại backend. | ✔ |
| FR-42 | Chế độ xem: so sánh kéo thanh, cạnh nhau, chỉ kết quả, chỉ ảnh gốc; ảnh dọc không bị cắt khỏi khung. | ✔ |
| FR-43 | Tải ảnh kết quả PNG hoặc JPEG ở cường độ chọn trước. | ✔ |
| FR-44 | Xem ảnh debug: alpha matte và trimap. | ✔ |

### 4.6 Chế độ tự động

| ID | Yêu cầu | TT |
|---|---|---|
| FR-50 | Khi bật, sau khi tải ảnh hệ thống tự phát hiện, tự chọn người và tách nền ngay, không cần bấm nút. | ✔ |
| FR-51 | Tự xử lý lại khi đổi người, engine hoặc tham số; mỗi tổ hợp chỉ chạy một lần. | ✔ |
| FR-52 | Nếu không phát hiện người và chưa có box thủ công, không tự chạy; thông báo và chờ người dùng nhập box. | ✔ |

### 4.7 Pipeline trực quan realtime

| ID | Yêu cầu | TT |
|---|---|---|
| FR-60 | Ghi lại mọi bước (tên, thời gian, ghi chú, ảnh trung gian) của hai giai đoạn: tiền xử lý & phát hiện; tách nền. | ✔ |
| FR-61 | Backend chạy việc nặng ở nền; frontend hỏi tiến độ định kỳ và hiện từng bước ngay khi hoàn thành, kèm thẻ "đang chạy" cho bước hiện tại. | ✔ |
| FR-62 | Sơ đồ hiển thị dạng thẻ nối mũi tên; bấm vào ảnh mở modal phóng to, chuyển bước bằng ← →, đóng bằng Esc hoặc bấm ra ngoài. | ✔ |
| FR-63 | Lỗi trong tác vụ nền được báo về giao diện với thông điệp và mã trạng thái, không làm treo giao diện. | ✔ |

### 4.8 Báo cáo PDF

| ID | Yêu cầu | TT |
|---|---|---|
| FR-70 | Xuất PDF gồm: thông tin ảnh & chất lượng, cài đặt tách nền, thời gian, Pipeline 1, Pipeline 2 (ảnh từng bước), kết quả trước/sau và alpha. | ✔ |
| FR-71 | Xuất được cả khi chưa tách nền (chỉ gồm phần thông tin và Pipeline 1). | ✔ |
| FR-72 | Hỗ trợ tiếng Việt có dấu; thiếu font thì báo lỗi rõ ràng và cho phép chỉ định bằng `REPORT_FONT`. | ✔ |

### 4.9 Khả năng mở rộng (yêu cầu cấu trúc)

| ID | Yêu cầu | TT |
|---|---|---|
| FR-80 | Thêm bước tiền xử lý mới chỉ bằng một hàm `rgb → rgb` đăng ký với `@STEPS.register`; giao diện tự hiện qua `/options`. | ✔ |
| FR-81 | Thêm engine tách nền mới bằng một file đăng ký `@ENGINES.register` với chữ ký `(scene, person, params) → Cutout`. | ✔ |
| FR-82 | Bước mới hiện trên sơ đồ pipeline và báo cáo PDF chỉ bằng cách bọc đoạn code trong `with trace.stage(...)`. | ✔ |

---

## 5. Yêu cầu phi chức năng

### 5.1 Hiệu năng
Số đo thực tế trên máy phát triển (CPU, ảnh 1280×854, một người chính) – dùng làm mức tham chiếu:

| ID | Yêu cầu | Tham chiếu |
|---|---|---|
| NFR-01 | Phát hiện người (YOLOv8s-seg) | ≈ 3 s |
| NFR-02 | Tách nền `cf` / `focus` | ≈ 1.5–3 s sau khi đã khởi động |
| NFR-03 | Tách nền `birefnet` (lite) | ≈ 25 s |
| NFR-04 | Lần chạy đầu chậm hơn do tải model và biên dịch JIT (numba) – chấp nhận được, có thông báo trên giao diện. | – |
| NFR-05 | Kéo thanh cường độ cập nhật trong < 100 ms cảm nhận được (xử lý tại trình duyệt). | – |
| NFR-06 | Giao diện cập nhật tiến độ pipeline với độ trễ ≲ 0.5 s so với lúc bước hoàn thành. | – |

### 5.2 Độ tin cậy và chịu lỗi
- NFR-10: Lỗi trong tác vụ nền không làm sập backend; trả trạng thái `error` kèm thông điệp.
- NFR-11: Matting có phương án dự phòng khi bộ giải số không hội tụ (FR-34).
- NFR-12: Session hết hạn (bị LRU loại) trả HTTP 404 với hướng dẫn tải ảnh lại.

### 5.3 Khả năng sử dụng
- NFR-20: Người dùng mới chỉ cần tải ảnh là có kết quả (chế độ tự động bật mặc định).
- NFR-21: Giao diện bằng tiếng Việt; nhãn bước và engine lấy từ backend nên nhất quán.
- NFR-22: Mọi tham số có chú thích trợ giúp (tooltip).

### 5.4 Khả năng bảo trì & mở rộng
- NFR-30: Pipeline chia thành các module độc lập (`preprocess`, `detect`, `segment`, `matting`, `masks`, `compose`, `trace`, `select`, `report`); mỗi module một trách nhiệm.
- NFR-31: Ngưỡng và hằng số tập trung ở `pipeline/config.py`.
- NFR-32: Có registry chung cho bước cắm thêm (FR-80, FR-81).

### 5.5 Bảo mật và riêng tư
- NFR-40: Ảnh chỉ nằm trong RAM của tiến trình backend, không ghi đĩa, không gửi ra ngoài (trừ việc tải trọng số model).
- NFR-41: Backend mặc định chỉ lắng nghe `127.0.0.1`; không có xác thực – **không** đặt lên mạng công cộng nếu không bổ sung bảo vệ.
- NFR-42: Giới hạn kích thước upload 25 MB; `YOLO_AUTOINSTALL=False` để ultralytics không tự cài gói.

### 5.6 Tương thích và khả chuyển
- NFR-50: Chạy trên Windows, Linux, macOS (font PDF cần thiết lập thêm nếu không có Arial/DejaVu).
- NFR-51: Biến môi trường cấu hình: `YOLO_WEIGHTS` (mặc định `yolov8s-seg.pt`), `BIREFNET_MODEL` (mặc định `birefnet-general-lite`), `API_URL`, `MAX_SESSIONS`, `REPORT_FONT`.

---

## 6. Kiến trúc và luồng dữ liệu

### 6.1 Cấu trúc module

```
backend/
  main.py                  API, session LRU, tác vụ nền (Job)
  report.py                dựng báo cáo PDF
  pipeline/
    config.py              ngưỡng / hằng số
    schema.py              Person, Scene, SegParams, Cutout
    registry.py            Registry cho bước cắm thêm
    imageio.py             đọc ảnh, resize
    preprocess/            quality.py (đánh giá + kế hoạch), steps.py (STEPS)
    detect.py              YOLO + thang thử lại -> Scene
    select.py              tự chọn người chính
    occlusion.py           loại phần của người khác, thứ tự trước/sau theo vị trí chân
    segment/               ENGINES: grabcut.py (cf, knn), focus.py, birefnet.py
    matting.py, masks.py   khối dùng chung
    compose.py             ghép đen trắng
    trace.py               nhật ký bước + ảnh trung gian
frontend/app.py            giao diện Streamlit
```

### 6.2 Luồng xử lý

```
Tải ảnh ─▶ resize + EXIF ─▶ đánh giá chất lượng ─▶ chuỗi tiền xử lý ─▶ YOLO-seg ──(không thấy người)──▶ thử lại
                                                                         │
                                                                  Scene (người, mask, box)
                                                                         │
                                                         tự chọn người / người dùng chọn / box thủ công
                                                                         │
                          ┌──────────────── engine ────────────────┐
                          │ cf/knn : GrabCut → trimap → matting     │
                          │ focus  : bản đồ độ nét → mask nét → GrabCut → trimap → matting
                          │ birefnet: alpha toàn ảnh → giới hạn theo mask YOLO
                          └─────────────────────────────────────────┘
                                                                         │
                                          khử nhiễm màu viền  →  Cutout (α, F, B, trimap)
                                                                         │
                          trình duyệt: ghép realtime theo cường độ  /  backend: compose, PDF
```

### 6.3 Mô hình dữ liệu chính

| Đối tượng | Nội dung |
|---|---|
| `Person` | idx, box, conf, mask (uint8 0/255) |
| `Scene` | ảnh làm việc, danh sách `Person`, ảnh đã tiền xử lý, chất lượng, các bước tiền xử lý đã dùng, số lần thử lại, người được gợi ý, `Trace` |
| `SegParams` | band_frac, focus_frac, trace |
| `Cutout` | image, alpha, foreground, background, trimap, trace |
| `Session` | ảnh, `Scene`, `Cutout`, các `Job` (detect / segment), cài đặt lần tách gần nhất |
| `Job` | trace đang ghi dần, state (`running` / `done` / `error`), lỗi, kết quả |

---

## 7. Giới hạn đã biết và hướng mở rộng

### 7.1 Giới hạn đã biết
- Hai người chạy chồng nhau mà YOLO gộp thành **một** mask (không có mask riêng của người kia) thì người phía sau vẫn có thể lọt một phần vào kết quả; `occlusion` chỉ xử lý được khi người kia có mask riêng. BiRefNet không tự phân biệt từng người.
- Engine `focus` giả định ảnh có phông mờ rõ; ảnh nền cũng sắc nét sẽ cho kết quả kém tin cậy. Quầng màu nhạt quanh chân có thể còn do GrabCut.
- Chất lượng viền phụ thuộc mask GrabCut: matting chỉ sửa được lỗi nằm trong dải chưa biết.
- BiRefNet-lite chậm (~25 s/ảnh trên CPU); chạy nhiều lần liên tục trong chế độ tự động có thể gây chờ lâu.
- Kết quả mới được kiểm tra trên số ít ảnh; chưa có bộ kiểm thử tự động và chưa đánh giá định lượng (IoU, SAD…).
- Bản xem pipeline "đang chạy" là HTML tĩnh; chỉ bản hoàn chỉnh mới phóng to được.

### 7.2 Hướng mở rộng
- Tách từng người khi hai người chồng nhau (SAM2 với điểm nhấp chuột, hoặc cho người dùng vẽ bớt vùng thừa).
- Thêm bước tiền xử lý mới (dehaze, khử rung nhẹ, tăng độ phân giải cho người nhỏ) và quy tắc chọn tự động tương ứng.
- Xử lý hàng loạt thư mục ảnh; hỗ trợ video.
- Bộ kiểm thử tự động và đánh giá định lượng; đóng gói Docker.
- Xác thực và hàng đợi tác vụ nếu cần phục vụ nhiều người dùng.
