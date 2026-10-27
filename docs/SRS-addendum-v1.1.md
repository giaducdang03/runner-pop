# SRS Addendum v1.1 – Tối ưu tốc độ, gia tốc GPU tích hợp, engine nhẹ, bản đồ nét tần số

Phần bổ sung cho [SRS.md](SRS.md). Các mục dưới đây là mục **4.10–4.13** và **5.7** của SRS.
Trạng thái mọi yêu cầu: **Đề xuất** (chưa triển khai).

> Các con số "mục tiêu" là ngưỡng nghiệm thu đề xuất, **chưa phải số đo thật** trên Intel Arc.
> Phải benchmark để xác nhận. CUDA/TensorRT không dùng được trên Intel Arc nên runtime chính là
> **OpenVINO**. GPU tích hợp dùng chung RAM hệ thống (không có VRAM riêng) nên có thêm ràng buộc bộ nhớ.

## Điều chỉnh sau khi duyệt (27/10/2026): GPU đa nhà sản xuất

Mục 4.11 ban đầu nhắm riêng Intel Arc và loại CUDA. Theo yêu cầu mới, lớp thiết bị hỗ trợ **nhiều backend**,
chọn bằng cấu hình lúc chạy app (biến môi trường `ACCEL_BACKEND`, mặc định `auto` = tự dò theo thứ tự
`cuda` → `mps` → `openvino` → `cpu`). Các điểm thay đổi so với FR-100..108 bên dưới:

| Backend | Phần cứng | YOLO-seg | Model ONNX |
|---|---|---|---|
| `cpu` | mọi máy | `.pt` (torch CPU) | onnxruntime CPU |
| `cuda` | NVIDIA | `.pt`, `cuda:N`, FP16 | onnxruntime CUDA EP |
| `mps` | Mac Apple Silicon | `.pt`, `mps`, FP32 | onnxruntime CoreML EP |
| `openvino` | Intel Arc / iGPU | OpenVINO IR FP16 | OpenVINO GPU, FP16, static shape |

- **FR-100**: biến cấu hình thêm `ACCEL_BACKEND` (`auto|cpu|cuda|mps|openvino`), `ACCEL_DEVICE_INDEX`, `ACCEL_DTYPE`; `ACCEL_DEVICE` (`auto|cpu|gpu`) giữ nguyên và vẫn đổi được theo từng request.
- **FR-101**: OpenVINO không còn là runtime duy nhất; CUDA và MPS/CoreML được phép. TensorRT, DirectML, ROCm nằm ngoài phạm vi (thêm sau như một backend mới).
- **FR-102/103**: nhánh "xuất OpenVINO IR / static shape / `CACHE_DIR`" chỉ áp cho backend `openvino`; backend khác dùng `.pt` và onnxruntime.
- **FP16 "tối thiểu"** nới thành: FP16 khi backend hỗ trợ trực tiếp (OpenVINO, YOLO trên CUDA); onnxruntime CUDA/CoreML chạy FP32. Nghiệm thu IoU ≥ 0.99 áp cho **từng backend có phần cứng thật**.
- **Ràng buộc bộ nhớ ≤ 3 GB** áp cho mọi backend (cấu hình được bằng `ACCEL_MEM_BUDGET_MB`); với GPU rời có VRAM riêng có thể nâng.
- **FR-105**: `/health` thêm `backend` và `available_backends`. Đổi backend cần khởi động lại; sidebar chỉ ép `auto/cpu/gpu`.
- **Trạng thái kiểm thử**: backend nào chưa chạy thật trên phần cứng phải ghi "chưa kiểm thử" trong README, không ghi "hỗ trợ".

## Thay đổi cho yêu cầu hiện có

| Mục | Hiện tại | Sửa thành |
|---|---|---|
| R1 | Chỉ dùng CPU | CPU luôn chạy được; GPU (Intel/OpenVINO, NVIDIA/CUDA, Mac/MPS) là tuỳ chọn, chọn bằng `ACCEL_BACKEND`, tự rơi về CPU |
| FR-31 | BiRefNet chạy toàn ảnh rồi giới hạn theo mask | BiRefNet chạy trên vùng ROI quanh người (FR-90), rồi giới hạn theo mask YOLO |
| NFR-01..03 | Chỉ có mốc CPU | Thêm cột mốc cho GPU tích hợp (mục 5.7) |
| 7.1 | Giới hạn "BiRefNet chậm" | Cập nhật sau khi có benchmark mới |

## 4.10 Tối ưu vùng xử lý và độ phân giải làm việc

| ID | Yêu cầu | TT |
|---|---|---|
| FR-90 | **Cắt ROI.** Với mọi engine, vùng xử lý nặng (GrabCut, trimap, matting, BiRefNet) chỉ chạy trên ROI = box người được chọn nở thêm `ROI_MARGIN` (mặc định 12%, tối thiểu 16 px). ROI phải gộp thêm box của những người **chồng lên** người được chọn để FR-37 vẫn hoạt động. Ngoài ROI, α = 0. | Đề xuất |
| FR-91 | **Độ phân giải làm việc.** Cạnh dài của ROI được thu nhỏ về tối đa `WORK_MAX_SIDE` (mặc định 640 px) trước khi matting; **không phóng to** khi ROI nhỏ hơn. | Đề xuất |
| FR-92 | Nâng α về độ phân giải gốc bằng **guided filter** (ảnh gốc ROI làm guide; bán kính, eps đặt trong `config.py`). Chỉ chỉnh α trong dải chưa biết; vùng chắc chắn giữ nguyên 0/1. | Đề xuất |
| FR-93 | Khử nhiễm màu tính ở độ phân giải làm việc, rồi nâng F và B về độ phân giải gốc bằng guided filter. Pixel có α = 1 lấy **trực tiếp từ ảnh gốc**, không qua F ước lượng. | Đề xuất |
| FR-94 | **Cache trung gian** theo khoá (session, người, engine, tham số upstream): bản đồ nét, mask GrabCut, ROI. Đổi band chỉ tính lại từ trimap trở đi; đổi cường độ không gọi backend (giữ FR-41). | Đề xuất |
| FR-95 | **Chế độ đối chứng.** Biến `FULLRES=1` buộc chạy toàn ảnh, toàn độ phân giải, để so sánh chất lượng và làm chuẩn benchmark. | Đề xuất |
| FR-96 | Trace ghi rõ ROI (toạ độ, tỉ lệ so với ảnh), kích thước làm việc và thời gian từng bước; hiện trong sơ đồ pipeline và PDF (FR-60, FR-70). | Đề xuất |

**Tiêu chí nghiệm thu:** trên tập ≥ 30 ảnh, so với `FULLRES=1`, mean IoU của α nhị phân hoá ≥ 0.98 và
SAD trung bình tăng không quá 5%; thời gian `cf`/`knn` giảm ≥ 2× trên cùng máy.

## 4.11 Gia tốc phần cứng (đa backend: Intel Arc, NVIDIA, Apple — xem mục "Điều chỉnh" ở trên)

| ID | Yêu cầu | TT |
|---|---|---|
| FR-100 | Lớp thiết bị `pipeline/accel.py` là **điểm duy nhất** chọn thiết bị cho mọi model. Biến `ACCEL_DEVICE` ∈ `auto \| cpu \| gpu` (mặc định `auto`: dùng GPU Intel nếu OpenVINO phát hiện được, nếu không dùng CPU). | Đề xuất |
| FR-101 | Runtime chính: **OpenVINO** (GPU plugin cho Intel Arc). Phương án thay thế trên Windows: ONNX Runtime với OpenVINO Execution Provider hoặc DirectML. Không phụ thuộc CUDA/TensorRT. | Đề xuất |
| FR-102 | **YOLO-seg:** xuất sang OpenVINO IR (FP16) ở lần chạy đầu và cache lại; tải lại không cần xuất lại. Nếu xuất lỗi, dùng `.pt` trên CPU. | Đề xuất |
| FR-103 | **BiRefNet:** dùng ONNX → OpenVINO, FP16, kích thước đầu vào cố định (static shape) để tránh biên dịch lại theo từng ảnh. Lưu cache model đã biên dịch (`ACCEL_CACHE_DIR`) để rút ngắn lần khởi động sau. | Đề xuất |
| FR-104 | **Tự rơi về CPU:** nếu GPU không có, thiếu bộ nhớ, op không được hỗ trợ hoặc kết quả lỗi, hệ thống chuyển sang CPU, ghi cảnh báo vào trace và **không báo lỗi** cho người dùng. | Đề xuất |
| FR-105 | `GET /health` và `GET /options` trả về thiết bị đang dùng (tên GPU, dtype, runtime); sidebar hiển thị và cho phép **ép CPU/GPU**. | Đề xuất |
| FR-106 | **Warm-up khi khởi động:** nạp và biên dịch các model, chạy một ảnh giả (gồm cả JIT numba); frontend hiện trạng thái "đang chuẩn bị" cho tới khi xong (cải thiện NFR-04). | Đề xuất |
| FR-107 | Trace và báo cáo PDF ghi thiết bị cho **từng bước nặng**. | Đề xuất |
| FR-108 | **Tuần tự hoá model nặng:** tại một thời điểm chỉ một model chạy trên GPU (khoá), vì GPU tích hợp dùng chung RAM. Các bước CPU (GrabCut, matting, bản đồ nét) vẫn có thể chạy song song với bước GPU. | Đề xuất |

**Ràng buộc riêng cho GPU tích hợp:**
- Tổng bộ nhớ model nạp cùng lúc **≤ 3 GB**; có tuỳ chọn giải phóng model không dùng (`ACCEL_UNLOAD_IDLE`).
- Không bắt buộc INT8; **FP16 là mức tối thiểu**. INT8 chỉ thêm sau khi đo thấy mất chất lượng chấp nhận được.
- README ghi rõ phiên bản driver Intel Graphics và OpenVINO đã kiểm thử.

**Tiêu chí nghiệm thu:** mask FP16 GPU so với FP32 CPU có IoU ≥ 0.99 trên tập kiểm thử; ngắt GPU
giữa chừng (mô phỏng lỗi) vẫn trả kết quả qua CPU.

## 4.12 Engine tách nền nhẹ

| ID | Yêu cầu | TT |
|---|---|---|
| FR-110 | Thêm engine `rvm` (Robust Video Matting, MobileNetV3) và/hoặc `modnet`, đăng ký bằng `@ENGINES.register`, chữ ký `(scene, person, params) → Cutout` (FR-81). Chỉ cần thêm **một file**, giao diện tự hiện qua `/options`. | Đề xuất |
| FR-111 | Engine chạy trên ROI (FR-90) ở độ phân giải làm việc (FR-91), α được giới hạn trong mask YOLO của người chọn nở vài px, loại mảnh rời (như FR-31). | Đề xuất |
| FR-112 | Engine chạy qua lớp thiết bị (FR-100): xuất ONNX → OpenVINO FP16, có fallback CPU. Với RVM, trạng thái hồi quy khởi tạo rỗng cho ảnh tĩnh; cho phép chạy lặp 1–3 lần trên cùng ảnh (`RVM_PASSES`) và tham số `downsample_ratio` cấu hình được. | Đề xuất |
| FR-113 | Kết quả đầy đủ `Cutout`: α, F (khử nhiễm màu), B, trimap. Với engine không dùng trimap, trimap sinh từ α theo ngưỡng chỉ để hiển thị debug (FR-44). | Đề xuất |
| FR-114 | FR-37 (loại phần của người khác theo thứ tự trước/sau) áp dụng cho engine mới. | Đề xuất |
| FR-115 | Chế độ tự động có thể **mặc định dùng engine nhẹ**; `birefnet` chỉ chạy khi người dùng chọn hoặc bấm "chất lượng cao" (tránh chờ lâu khi đổi tham số liên tục). | Đề xuất |
| FR-116 | `YOLO_WEIGHTS` hỗ trợ `yolov8n-seg` / `yolo11n-seg`; tài liệu ghi lại tỉ lệ phát hiện so với `yolov8s-seg` để người dùng cân nhắc. | Đề xuất |
| FR-117 | **Giấy phép:** trước khi đưa trọng số vào, kiểm tra giấy phép của từng model (RVM được cho là GPL-3.0, MODNet là Apache-2.0 – **phải xác minh lại trên repo gốc**) và ghi vào README, phòng trường hợp phát hành lại. | Đề xuất |

**Tiêu chí nghiệm thu:** trên cùng tập ảnh, so với `birefnet` làm tham chiếu: mean IoU ≥ 0.95, và ghi
nhận riêng các ca tóc, tay chân mảnh, nền phức tạp. Thời gian tách nền (không tính warm-up) thấp hơn
`birefnet` ít nhất 5× trên cùng thiết bị.

## 4.13 Bản đồ độ nét theo miền tần số (engine focus)

| ID | Yêu cầu | TT |
|---|---|---|
| FR-120 | Thêm phương pháp đo nét `freq` cho FR-32: ảnh xám thu về cạnh dài 640 px, chia khối 16×16 bước 8 (chồng 50%), DCT 2D từng khối (`cv2.dct`), độ nét = tỉ lệ năng lượng dải cao / tổng năng lượng AC (chuẩn hoá log), làm mịn rồi nâng về kích thước ảnh. | Đề xuất |
| FR-121 | `FOCUS_METHOD` ∈ `laplacian \| freq \| hybrid` (trung bình có trọng số hai bản đồ). Mặc định giữ `laplacian` cho tới khi đạt nghiệm thu, sau đó đổi sang `freq` hoặc `hybrid`. | Đề xuất |
| FR-122 | **Chống nhiễu:** khi cờ `noisy` hoặc `dark` bật, loại dải tần cao nhất khỏi phép tính và trừ nền nhiễu ước lượng (Immerkaer) để vùng nhiễu không bị coi là nét. | Đề xuất |
| FR-123 | Ngưỡng `focus_frac` (10–100%) giữ nguyên ngữ nghĩa, áp theo **phân vị** của bản đồ mới. | Đề xuất |
| FR-124 | Chạy hoàn toàn trên CPU, vector hoá; không phụ thuộc GPU. | Đề xuất |
| FR-125 | Trace hiển thị bản đồ nét dạng heatmap và ghi phương pháp, thời gian. Bản đồ này **dùng lại** cho FR-22 (chấm điểm chọn người chính) và *có thể* dùng lại để tính chỉ số độ nét ở FR-10. | Đề xuất |
| FR-126 | **Báo cáo mức tin cậy:** nếu độ chênh nét giữa vùng người và nền thấp (ảnh không có phông mờ rõ), giao diện cảnh báo thay vì im lặng cho kết quả kém (khớp 7.1). | Đề xuất |

**Tiêu chí nghiệm thu:** thời gian tính bản đồ ≤ 150 ms ở 640 px trên CPU; trên tập ảnh xoá phông có
gán nhãn tay, tỉ lệ người đúng mặt phẳng nét được giữ không thấp hơn `laplacian`, và số ảnh còn quầng
màu quanh chân giảm.

## 5.7 Hiệu năng mục tiêu theo thiết bị (bổ sung cho 5.1)

Đo trên ảnh 1280×854, một người chính, không tính warm-up. Các cột GPU là **mục tiêu cần xác nhận**.

| ID | Bước | CPU (hiện tại) | GPU tích hợp Intel Arc (mục tiêu) |
|---|---|---|---|
| NFR-60 | Phát hiện YOLOv8s-seg | ≈ 3 s | ≤ 0.6 s |
| NFR-61 | Tách nền `cf`/`knn` (có ROI + upsample) | ≈ 1.5–3 s → ≤ 1 s. **Đo thật (CPU, bản rút gọn):** `cf` 0.70 s, `knn` 1.01 s (ROI); `FULLRES=1`: `cf` 0.64 s, `knn` 1.17 s. Trung vị, 5 ảnh đầu × 4 lần bỏ 1, Intel64 Family 6 Model 170 (22 luồng). Xem ghi chú dưới bảng | ≤ 1 s (matting vẫn trên CPU) |
| NFR-62 | Tách nền `birefnet` (ROI) | ≈ 25 s → ≤ 10 s. **Đo thật (CPU, bản rút gọn, ROI):** bước `birefnet` trung vị 23.298 s (p95 25.356 s), tổng build_cutout trung vị 23.595 s (p95 25.770 s). 3 ảnh đầu × 3 lần bỏ 1 (n = 6 mẫu), máy và lệnh ở khối "Số đo Phase B" dưới bảng. **Chưa đạt** mục tiêu ≤ 10 s: BiRefNet-lite chạy ở đầu vào cố định 1024×1024 nên ROI không làm nó nhanh hơn | ≤ 5 s |
| NFR-63 | Tách nền `rvm`/`modnet` | ≤ 2 s | ≤ 0.8 s |
| NFR-64 | Bản đồ nét `freq` | ≤ 150 ms | ≤ 150 ms (CPU) |
| NFR-65 | Khởi động + warm-up | ≤ 60 s lần đầu (kể cả xuất/biên dịch model), ≤ 20 s các lần sau nhờ cache | – |
| NFR-66 | Bộ nhớ model nạp cùng lúc | ≤ 3 GB | ≤ 3 GB (dùng chung RAM hệ thống) |

**Số đo Phase A (CPU, thật):** máy Intel64 Family 6 Model 170, 22 luồng; ảnh `dataset/hd_bank_green_marathon_2026_resized`
(1600×1067, resize xuống 1280 px), engine `cf`, so `FULLRES=1`.
- Chất lượng (`tools/eval_quality.py --limit 30 --engine cf --ref-fullres`, **30 ảnh**): mean IoU 0.9858, p05 IoU 0.9372,
  min IoU 0.9044, mean SAD 914.3 (= 2.02% tổng α tham chiếu). Đạt ngưỡng IoU ≥ 0.98.
- Thời gian cùng lần chạy: ROI 0.775 s so với FULLRES 1.025 s (trung bình) → nhanh hơn 1.32× (trung vị 1.38×). **Chưa đạt** mục tiêu ≥ 2×.
  Nguyên nhân chưa được xác lập. Giả thuyết cũ "ảnh nghiệm thu chụp người chiếm phần lớn khung hình nên ROI gần bằng cả ảnh" **đã đo và bị bác bỏ**:
  diện tích ROI / diện tích ảnh trên 30 ảnh đầu (người chính do YOLO gợi ý, `person_roi`): trung vị 2.8%, trung bình 8.0%, nhỏ nhất 1.7%, lớn nhất 34.6%
  (xem "Số đo Phase B"). Còn lại là giả thuyết, **giả thuyết, chưa đo**: các stage chạy trên toàn ảnh làm việc (`decontaminate` ≈ 0.19–0.20 s, `compose` ≈ 0.10 s
  trong lần đo ROI ở Phase B) không giảm theo ROI nên chi phí cố định này chặn mức tăng tốc; chưa đo `FULLRES=1` theo từng stage để xác nhận.
  Quy mô tập ảnh (30 ảnh) nhỏ nên con số 1.32× chỉ mang tính chỉ báo, **giả thuyết, chưa đo** là nó đại diện cho ảnh ngoài tập này.
- `tools/bench.py` bản **rút gọn** (`--limit 5 --runs 4 --drop 1`, không phải 10 lần bỏ 2): tổng trung vị `cf` 0.699 s (p95 0.823) ROI / 0.644 s (p95 0.670) FULLRES;
  `knn` 1.008 s (p95 1.463) / 1.165 s (p95 1.333); `focus` 1.448 s (p95 1.735) / 6.266 s (p95 10.913).
- YOLO (NFR-60), `rvm`/`modnet`, `freq`: NOT RUN (chưa có engine / không nằm trong bench engine). `birefnet`: xem "Số đo Phase B" (khối Phase A ghi nhầm là chưa có trọng số; trọng số BiRefNet-lite có sẵn ở `~/.rembg/models/birefnet-general-lite/`).

**Số đo Phase B (CPU, thật, bản rút gọn):** máy Intel Core Ultra 7 155H (22 luồng logic), RAM 31.6 GB, Windows 11 (10.0.26200);
Python 3.13.14, torch 2.14.1+cpu, onnxruntime 1.31.0, ultralytics 8.4.189, rembg 2.0.86. Không có backend GPU nào trên máy này
(`accel.available_backends()` = `{}`), nên **các cột GPU không có số đo: NOT RUN: máy đo không có GPU cuda/mps/openvino**.
Bản rút gọn, không phải 30 ảnh × 10 lần bỏ 2 theo phương pháp đầy đủ; n = 6 mẫu mỗi stage nên p95 chỉ mang tính chỉ báo.
- Lệnh: `python tools/bench.py --dir ../dataset/hd_bank_green_marathon_2026_resized --limit 3 --engine birefnet --device cpu --runs 3 --drop 1`
  → tổng trung vị 23.595 s (p95 25.770 s); `birefnet` 23.298 s (p95 25.356 s), `decontaminate` 0.190 s, `compose` 0.104 s, các stage còn lại ≤ 0.005 s.
- Lệnh: `python tools/bench.py --dir ../dataset/hd_bank_green_marathon_2026_resized --limit 3 --engine cf --device cpu --runs 3 --drop 1`
  → tổng trung vị 0.626 s (p95 0.694 s); `grabcut` 0.198 s, `matting` 0.091 s, `decontaminate` 0.201 s, `compose` 0.105 s.
- Diện tích ROI / ảnh (30 ảnh đầu, `person_roi` cho người chính do YOLO gợi ý): trung vị 2.8%, trung bình 8.0%, min 1.7%, max 34.6%.

**Phương pháp đo:** dùng trace hiện có, mỗi cấu hình chạy 10 lần bỏ 2 lần đầu, báo cáo **trung vị và
p95**; bảng kết quả đưa vào báo cáo đồ án và lấp phần "chưa có đánh giá định lượng" ở mục 7.1 (thêm
IoU/SAD so với `FULLRES=1` và `birefnet`).

## Thứ tự triển khai gợi ý

1. FR-90..96 (ROI + độ phân giải thấp): không phụ thuộc phần cứng, nên làm trước và đo ngay.
2. FR-100..108 (lớp thiết bị, OpenVINO): làm sớm vì engine mới ở bước 3 cũng dựa vào nó.
3. FR-110..117 (engine `rvm`/`modnet`).
4. FR-120..126 (bản đồ nét tần số): độc lập, làm song song bất kỳ lúc nào.
