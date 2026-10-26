# Runner – giữ màu người chạy, nền đen trắng (CPU)

YOLO-seg → GrabCut → trimap → alpha matting (pymatting) → khử nhiễm màu viền → ghép.
Backend FastAPI, giao diện Streamlit. Chạy hoàn toàn trên CPU.

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

## Giới hạn đã biết

- Chất lượng viền phụ thuộc mask GrabCut: matting chỉ sửa được lỗi nằm **trong dải unknown** (mặc định 2% kích thước người).
  Nếu mask sai lớn (quần áo tối trên nền tối ban đêm, người đứng sát khán giả) hãy tăng *Độ rộng vùng viền* hoặc nhập box thủ công.
- Kết quả mới được kiểm tra trên 1 ảnh mẫu (người trên nền khán đài mờ), chưa có ảnh ban đêm/cầu/đường phố thật – nên thử và
  chỉnh `band_frac`, `YOLO_CONF`, số vòng GrabCut trong `backend/pipeline.py`.
- Nếu cần mask tốt hơn nữa, có thể thay bước GrabCut bằng BiRefNet/SAM2 (chạy CPU sẽ chậm hơn đáng kể).
