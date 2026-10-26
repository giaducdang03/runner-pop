"""Hằng số cấu hình dùng chung. Mỗi nhóm bước có mục riêng để dễ chỉnh."""
import os

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
