"""Hằng số cấu hình dùng chung. Mỗi nhóm bước có mục riêng để dễ chỉnh."""
import os
from pathlib import Path

# Không cho ultralytics tự pip-install gói khi chạy (tránh thay đổi môi trường ngoài ý muốn)
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

# --- ảnh làm việc ---------------------------------------------------------
MAX_SIDE = int(os.environ.get("MAX_SIDE", "2560"))   # ảnh được thu nhỏ về cạnh dài này để chạy; ROI vẫn xử lý ≤ WORK_MAX_SIDE

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
ALPHA_LO = 0.10          # alpha dưới mức này -> 0 (cắt đuôi loá quanh người)
ALPHA_HI = 0.90          # alpha trên mức này -> 1 (dùng màu gốc sát viền hơn)

# --- bản đồ nét (engine focus + chọn người) ---------------------------------
FOCUS_METHOD = os.environ.get("FOCUS_METHOD", "laplacian")   # laplacian | freq | hybrid
FOCUS_BLOCK = 16           # cạnh khối DCT (px, ở độ phân giải đo nét)
FOCUS_STRIDE = 8           # bước trượt khối (khối chồng 50%)
FOCUS_MAX_SIDE = 640       # cạnh dài tối đa khi đo nét theo tần số
FOCUS_HYBRID_W = 0.5       # trọng số của bản đồ tần số trong chế độ hybrid
FOCUS_CONF_MIN = 1.3       # tỉ lệ nét người/nền dưới mức này = ảnh không có phông mờ rõ
FOCUS_OTHER_MIN = 0.85     # người khác chỉ được giữ màu khi (nét/nét người chọn) × (tỉ lệ mịn/thô) ≥ mức này
FOCUS_RIM_SHARP = 0.35      # điểm có độ nét ≥ mức này × độ nét người chọn được coi là "nét" khi cắt phần lẫn mờ
FOCUS_RIM_FRAC = 0.30       # phần lẫn mờ được cắt sâu tối đa tỉ lệ này × bề ngang người từ rìa mask

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
WARMUP_ON_START = os.environ.get("WARMUP", "1") != "0"   # nạp sẵn model khi khởi động backend
# Engine chọn sẵn trên giao diện và được nạp sẵn lúc khởi động; không có trong danh sách engine thì dùng thứ tự ưu tiên
DEFAULT_ENGINE = os.environ.get("DEFAULT_ENGINE", "birefnet").strip().lower()
BIREFNET_SIZE = int(os.environ.get("BIREFNET_SIZE", "1024"))
# Tên input của ONNX BiRefNet do rembg phát hành (đọc từ birefnet-general-lite.onnx, shape tĩnh
# 1x3x1024x1024); rembg lấy get_inputs()[0].name, ở đây cố định để convert OpenVINO shape tĩnh.
BIREFNET_INPUT_NAME = os.environ.get("BIREFNET_INPUT_NAME", "input_image")
