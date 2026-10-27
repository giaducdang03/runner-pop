# Runner – giữ màu người chạy, nền đen trắng

YOLO-seg → GrabCut → trimap → alpha matting (pymatting) → khử nhiễm màu viền → ghép.
Backend FastAPI, giao diện Streamlit. Luôn chạy được trên CPU; GPU (Intel/NVIDIA/Mac) là tuỳ chọn, xem mục *Gia tốc phần cứng*.

## Cài đặt (Windows PowerShell, trong venv)

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

> Nếu `opencv-python` xung đột với bản `opencv-contrib`/`headless` đã có, hãy gỡ bản cũ trước.
> Lần chạy đầu ultralytics tự tải trọng số `yolov8s-seg.pt` (~24 MB).
> Lần xử lý đầu tiên chậm hơn vì pymatting biên dịch JIT (numba).

## Chạy (2 terminal, đứng ở thư mục gốc project)

```powershell
# terminal 1 – backend
uvicorn backend.main:app --host 127.0.0.1 --port 8000

# terminal 2 – giao diện
streamlit run frontend/app.py
```

Biến môi trường tuỳ chọn: `YOLO_WEIGHTS` (mặc định `yolov8s-seg.pt`, có thể dùng `yolov8n-seg.pt` cho nhanh hơn),
`API_URL` (mặc định `http://127.0.0.1:8000`).

## Gia tốc phần cứng

CPU **luôn** chạy được. GPU là tuỳ chọn: thiếu gói hay driver của backend nào thì backend đó bị coi là không có và app tự chạy bằng CPU.

### Backend và trạng thái kiểm thử

| Backend | Phần cứng | Đường YOLO | Đường ONNX (BiRefNet) | File cài đặt | Trạng thái kiểm thử |
|---|---|---|---|---|---|
| `cpu` | Mọi máy | PyTorch CPU | rembg + onnxruntime CPU | `requirements.txt` | **Đã kiểm thử trên phần cứng thật** (Intel Core Ultra 7 155H, Windows 11; phiên bản ở bảng dưới) |
| `cuda` | GPU NVIDIA | PyTorch CUDA (`half=True`) | onnxruntime-gpu (`CUDAExecutionProvider`, FP32) | `requirements-cuda.txt` | **chưa kiểm thử trên phần cứng thật — chỉ có test giả lập** |
| `mps` | Mac Apple Silicon | PyTorch MPS (FP32) | onnxruntime (`CoreMLExecutionProvider`, FP32) | `requirements-mac.txt` | **chưa kiểm thử trên phần cứng thật — chỉ có test giả lập** |
| `openvino` | GPU Intel (Arc / iGPU) | OpenVINO IR (FP16, xuất một lần vào `ACCEL_CACHE_DIR`) | OpenVINO compile ONNX (FP16) | `requirements-intel.txt` | **chưa kiểm thử trên phần cứng thật — chỉ có test giả lập** |

Máy phát triển không có GPU backend nào (`accel.available_backends()` trả `{}`), nên chưa có số đo hay kết quả chạy thật cho `cuda`, `mps`, `openvino`.

#### Phiên bản driver / runtime

| Backend | Driver / hệ điều hành | Runtime | Ghi chú |
|---|---|---|---|
| `cpu` | Windows 11 (10.0.26200), Python 3.13.14 | torch 2.14.1+cpu, onnxruntime 1.31.0, ultralytics 8.4.189, rembg 2.0.86 | đã đo thật |
| `cuda` | chưa kiểm thử | chưa kiểm thử | chưa kiểm thử |
| `mps` | chưa kiểm thử | chưa kiểm thử | chưa kiểm thử |
| `openvino` | chưa kiểm thử | chưa kiểm thử (máy đo chưa cài gói `openvino`) | chưa kiểm thử |

### Cài đặt theo backend

Cài phần chung trước (mục *Cài đặt* ở trên), rồi thêm gói của backend muốn dùng:

```powershell
# Intel Arc / iGPU (OpenVINO)
pip install -r requirements-intel.txt

# NVIDIA (CUDA) — torch bản CUDA phải lấy từ PyTorch index, không phải PyPI
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip uninstall -y onnxruntime          # BẮT BUỘC, chạy TRƯỚC khi cài onnxruntime-gpu
pip install -r requirements-cuda.txt  # onnxruntime-gpu
```

> **Cảnh báo (R19):** `rembg[cpu]` trong `requirements.txt` kéo theo gói `onnxruntime` bản CPU. Hai gói cùng cung cấp module
> `onnxruntime` và bản CPU che bản GPU, nên `CUDAExecutionProvider` không xuất hiện. Luôn `pip uninstall -y onnxruntime`
> **trước** `pip install onnxruntime-gpu`. Nếu quên, app không lỗi nhưng chạy CPU và trace ghi đã rơi về CPU.

```bash
# Mac Apple Silicon (MPS + CoreML)
pip install -r requirements-mac.txt
```

### Chọn backend lúc khởi động

Backend được chọn **một lần lúc khởi động** bằng biến môi trường; đổi backend cần khởi động lại backend FastAPI.

```powershell
# Windows PowerShell
$env:ACCEL_BACKEND="cuda"; uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

```bash
# bash / zsh (macOS, Linux, Git Bash)
ACCEL_BACKEND=mps uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `ACCEL_BACKEND` | `auto` | `auto` (tự dò theo thứ tự cuda, mps, openvino) \| `cpu` \| `cuda` \| `mps` \| `openvino` |
| `ACCEL_DEVICE` | `auto` | Chế độ thiết bị mặc định khi request không nêu: `auto` \| `cpu` \| `gpu` |
| `ACCEL_DEVICE_INDEX` | `0` | Chọn GPU khi máy có nhiều card |
| `ACCEL_DTYPE` | `auto` | `auto` \| `fp16` \| `fp32` (chỉ áp cho backend GPU; CPU luôn fp32) |
| `ACCEL_CACHE_DIR` | `~/.cache/color_pop_runner` | Thư mục cache model đã xuất (vd. OpenVINO IR). Chỉ cache model được ghi đĩa; ảnh người dùng chỉ ở RAM |
| `ACCEL_MEM_BUDGET_MB` | `3072` | Ngân sách bộ nhớ model nạp cùng lúc (MB), áp cho mọi backend |
| `ACCEL_UNLOAD_IDLE` | `0` | Giây rảnh trước khi giải phóng model; `0` = không giải phóng |
| `ACCEL_EXPORT_RETRY_S` | `300` | Xuất/build model lỗi thì không thử lại trong khoảng này (giây) |
| `BIREFNET_SIZE` | `1024` | Cạnh vuông cố định của đầu vào BiRefNet trên GPU |
| `WARMUP` | `1` | `0` để tắt nạp sẵn model khi khởi động backend |
| `DEFAULT_ENGINE` | `birefnet` | Engine chọn sẵn trên giao diện và được nạp sẵn lúc khởi động (`cf`, `knn`, `focus`, `birefnet`). Không có trong danh sách thì dùng `cf` |
| `FULLRES` | `0` | `1` = bỏ ROI, xử lý toàn ảnh (để so sánh) |

Tên backend gõ sai hoặc backend không dùng được thì app dùng CPU, không báo lỗi lúc khởi động.

### Backend cố định, thiết bị đổi theo request

Backend (cuda/mps/openvino) cố định từ lúc khởi động. Trên thanh bên chỉ lựa chọn `auto | cpu | gpu` đổi được theo từng request:
`cpu` ép CPU; `auto` và `gpu` dùng GPU của backend đã chọn nếu có, không có thì CPU.

### Khi GPU lỗi: rơi về CPU

Mọi lỗi GPU (thiếu gói, driver reset giữa chừng, hết bộ nhớ, op không hỗ trợ) đều không làm request thất bại. Bước đó chạy lại bằng
**CPU một cách tường minh**: thiết bị của bước hiện là `CPU` trên sơ đồ pipeline/PDF và trace ghi chú `GPU lỗi, đã dùng CPU: <lý do>`.
Lỗi bền (build model hỏng) được nhớ trong `ACCEL_EXPORT_RETRY_S` giây để không thử lại ở mỗi request. Hành vi này được kiểm bằng
`tests/test_gpu_interrupt.py` (GPU giả lập chết ở lần gọi thứ hai).

## Cách dùng

1. Tải ảnh → YOLO đánh số các người phát hiện được (mặc định #0 = box lớn nhất).
2. Chọn người chạy → **Xử lý**.
3. Kéo **cường độ nền đen trắng** (realtime, không gọi lại backend) và dùng chế độ **So sánh** để kéo thanh trước/sau.
4. Tải ảnh kết quả (PNG/JPEG) ở cường độ mong muốn. Mục *Debug* cho xem alpha và trimap.

## Kiến trúc – phần nào nặng, phần nào nhẹ

| Bước | Nơi chạy | Ghi chú |
|---|---|---|
| Detect (YOLO, có CLAHE/gamma cho ảnh tối – chỉ dùng cho detection) | backend, 1 lần | ~2 s |
| GrabCut → trimap → matting → khử nhiễm màu | backend, 1 lần / người | ~3–5 s (closed-form), ảnh 1280 px |
| Slider cường độ + so sánh trước/sau | trình duyệt (canvas JS) | tức thì |
| Ảnh tải về | backend `/compose` | cùng công thức với JS |

Công thức ghép: `out = α·F + (1−α)·((1−s)·B + s·gray(B))`, với F/B là màu foreground/background đã được ước lượng
sao cho `ảnh ≈ α·F + (1−α)·B` → viền không lẫn màu nền cũ, nền sau viền không lẫn độ sáng của người.

## API

`GET /session/{id}/trace` (các bước pipeline + ảnh minh hoạ) · `GET /options` (engine + bước tiền xử lý hiện có) · `POST /detect?preprocess=auto|none|gamma,clahe` · `GET /session/{id}/preview.jpg` · `POST /process/{id}?person=&method=cf|knn&band=&box=x1,y1,x2,y2`
· `GET /session/{id}/{original.jpg|foreground.png|background.jpg|alpha.png|trimap.png}` · `GET /session/{id}/compose?intensity=&fmt=`
· `GET /health`. Session lưu trong RAM (tối đa 8), phù hợp 1 người dùng/demo.

## Cấu trúc code & cách mở rộng

```
backend/
  main.py                  API + session
  pipeline/
    config.py              ngưỡng / hằng số (chất lượng ảnh, thang thử lại detect, model)
    schema.py              Person, Scene, SegParams, Cutout
    trace.py               ghi từng bước + ảnh trung gian -> giao diện vẽ sơ đồ pipeline khi demo
    registry.py            Registry dùng chung cho các bước cắm thêm
    imageio.py             đọc ảnh, resize
    preprocess/            tiền xử lý CHỈ cho phát hiện
      quality.py           assess() chỉ số chất lượng + plan_steps() quy tắc chọn tự động
      steps.py             các bước đã đăng ký (levels, gamma, clahe, denoise, sharpen)
    detect.py              YOLO + thang thử lại -> Scene
    segment/               engine tách nền -> Cutout
      base.py              ENGINES registry + helper
      grabcut.py / focus.py / birefnet.py
    occlusion.py           loại phần của người khác (thứ tự trước/sau theo vị trí chân)
    matting.py, masks.py   khối dùng chung (trimap, matting, khử nhiễm màu, phép toán mask)
    compose.py             ghép đen trắng
```

- **Thêm bước tiền xử lý**: viết hàm `rgb -> rgb` với `@STEPS.register("ten", label=...)` trong `preprocess/steps.py`;
  muốn tự động dùng thì thêm quy tắc vào `plan_steps()`. Giao diện tự hiện bước mới (lấy từ `/options`).
- **Hiện bước trên sơ đồ demo**: bọc đoạn code bằng `with trace.stage("key", "Nhãn") as s:` rồi `s.show(img)` / `s.overlay(rgb, mask)`;
  giao diện tự vẽ thành thẻ trong sơ đồ.
- **Thêm engine tách nền**: file mới trong `segment/` với `@ENGINES.register("ten", label=...)`,
  chữ ký `(scene, person, params) -> Cutout`, rồi import nó trong `segment/__init__.py`.
- **Khi không phát hiện được người** (`preprocess=auto`): `detect.py` thử lại theo `DETECT_ATTEMPTS`
  (conf thấp hơn, imgsz lớn hơn, chuỗi `FALLBACK_PREPROCESS` mạnh hơn).

## Công cụ đo (tools/)

Tập ảnh nghiệm thu chuẩn: `../dataset/hd_bank_green_marathon_2026_resized`, lấy 30 ảnh đầu theo thứ tự tên đã sắp xếp. Chạy từ thư mục này:

```powershell
# thời gian từng stage (trung vị + p95), 10 lần bỏ 2 lần đầu — phương pháp mục 5.7; thêm --fullres để so với không ROI
.venv\Scripts\python.exe tools/bench.py --dir ../dataset/hd_bank_green_marathon_2026_resized --limit 30 --engine cf,knn,birefnet,focus --runs 10 --drop 2
# chất lượng so với FULLRES=1 (cũng có --ref-engine <tên> hoặc --ref-device cpu): mean IoU, mean SAD, p05 IoU, thời gian tương đối
.venv\Scripts\python.exe tools/eval_quality.py --dir ../dataset/hd_bank_green_marathon_2026_resized --limit 30 --engine cf --ref-fullres
```

`FULLRES` được đọc từ môi trường một lần lúc import, nên công cụ đảo `config.FULLRES` ngay trong tiến trình (`active_roi` đọc lại mỗi lần gọi).
Engine thiếu trọng số (vd. `birefnet`) in `NOT RUN: ...`. Số đo thật nằm ở mục 5.7 của `docs/SRS-addendum-v1.1.md`.

## Giới hạn đã biết

- Chất lượng viền phụ thuộc mask GrabCut: matting chỉ sửa được lỗi nằm **trong dải unknown** (mặc định 2% kích thước người).
  Nếu mask sai lớn (quần áo tối trên nền tối ban đêm, người đứng sát khán giả) hãy tăng *Độ rộng vùng viền* hoặc nhập box thủ công.
- Kết quả mới được kiểm tra trên 1 ảnh mẫu (người trên nền khán đài mờ), chưa có ảnh ban đêm/cầu/đường phố thật – nên thử và
  chỉnh `band_frac`, `YOLO_CONF`, số vòng GrabCut trong `backend/pipeline.py`.
- Nếu cần mask tốt hơn nữa, có thể thay bước GrabCut bằng BiRefNet/SAM2 (chạy CPU sẽ chậm hơn đáng kể).
