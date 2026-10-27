"""Hằng số cấu hình dùng chung. Mỗi nhóm bước có mục riêng để dễ chỉnh."""
import os
from pathlib import Path

# Không cho ultralytics tự pip-install gói khi chạy (tránh thay đổi môi trường ngoài ý muốn)
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

# --- ảnh làm việc ---------------------------------------------------------
MAX_SIDE = 1280            # ảnh được thu nhỏ về cạnh dài này để chạy (CPU)

# --- phát hiện người ------------------------------------------------------
YOLO_IMGSZ = 640
PERSON_CLASS = 0
# Thang thử lại khi không phát hiện được: (conf, imgsz). Thử lần lượt, dừng ở lần có kết quả.
DETECT_ATTEMPTS = [(0.25, 640), (0.15, 960), (0.10, 1280)]
# Chuỗi tiền xử lý mạnh dùng cho các lần thử lại (chỉ khi preprocess="auto")
FALLBACK_PREPROCESS = ["levels", "gamma", "denoise", "clahe"]

# --- đánh giá chất lượng ảnh (ngưỡng cho kế hoạch "auto") -----------------
Q_DARK_BRIGHTNESS = 90         # độ sáng trung bình (0..255) dưới mức này = ảnh tối
Q_LOW_CONTRAST = 110           # p95 - p5 của độ sáng dưới mức này = thiếu tương phản
Q_NOISY = 6.0                  # sigma nhiễu ước lượng (0..255) trên mức này = nhiều nhiễu
Q_BLURRY = 60.0                # phương sai Laplacian (ảnh 640px) dưới mức này = mờ

# --- backend model --------------------------------------------------------
YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "yolov8s-seg.pt")
BIREFNET_MODEL = os.environ.get("BIREFNET_MODEL", "birefnet-general-lite")

# --- ROI / độ phân giải làm việc -----------------------------------------
ROI_MARGIN = 0.12          # ROI nở thêm tỉ lệ này * cạnh dài của box người mỗi phía
ROI_MIN_PX = 16            # sàn nở tối thiểu (px) mỗi phía, cho người rất nhỏ
WORK_MAX_SIDE = 640        # cạnh dài tối đa của ROI khi chạy engine (scale < 1 nếu lớn hơn)
GUIDED_RADIUS = 8          # bán kính guided filter (px, ở độ phân giải làm việc)
GUIDED_EPS = 1e-4          # eps của guided filter
FULLRES = os.environ.get("FULLRES", "0") == "1"   # bật = bỏ ROI, chạy toàn ảnh gốc

# --- gia tốc phần cứng (chọn bằng biến môi trường khi chạy app) ---------------
ACCEL_DEVICE = os.environ.get("ACCEL_DEVICE", "auto")        # auto|cpu|gpu
ACCEL_BACKEND = os.environ.get("ACCEL_BACKEND", "auto")      # auto|cpu|cuda|mps|openvino
ACCEL_DEVICE_INDEX = int(os.environ.get("ACCEL_DEVICE_INDEX", "0"))   # chọn GPU khi có nhiều card
ACCEL_DTYPE = os.environ.get("ACCEL_DTYPE", "auto")          # auto|fp16|fp32
ACCEL_CACHE_DIR = os.environ.get(
    "ACCEL_CACHE_DIR", str(Path.home() / ".cache" / "color_pop_runner"))
ACCEL_MEM_BUDGET_MB = int(os.environ.get("ACCEL_MEM_BUDGET_MB", "3072"))   # ngân sách model nạp cùng lúc
ACCEL_UNLOAD_IDLE = float(os.environ.get("ACCEL_UNLOAD_IDLE", "0"))        # giây; 0 = không giải phóng
ACCEL_EXPORT_RETRY_S = float(os.environ.get("ACCEL_EXPORT_RETRY_S", "300"))   # giây; xuất IR lỗi thì thử lại sau
BIREFNET_SIZE = int(os.environ.get("BIREFNET_SIZE", "1024"))
# Tên input của ONNX BiRefNet do rembg phát hành (đọc từ birefnet-general-lite.onnx, shape tĩnh
# 1x3x1024x1024); rembg lấy get_inputs()[0].name, ở đây cố định để convert OpenVINO shape tĩnh.
BIREFNET_INPUT_NAME = os.environ.get("BIREFNET_INPUT_NAME", "input_image")
