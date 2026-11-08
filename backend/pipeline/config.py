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

# --- đồ vật cầm trên tay ------------------------------------------------------
HELD_ENABLE = os.environ.get("HELD_ENABLE", "1") != "0"
# Dò gậy mặc định BẬT: bộ xác nhận LSD (POLE_LSD_MIN) loại gậy giả, trên mẫu gán nhãn giữ 41/44 gậy
# thật và loại 44/49 gậy giả (trước khi lọc ~46% gậy dò được là giả). POLE_ENABLE=0 để tắt.
# Gậy ứng viên chỉ được gộp cho người được xử lý (build_cutout); vật COCO không bị ảnh hưởng.
POLE_ENABLE = os.environ.get("POLE_ENABLE", "1") != "0"
POSE_WEIGHTS = os.environ.get("POSE_WEIGHTS", "yolov8s-pose.pt")
POSE_IMGSZ = 960
HAND_MIN_CONF = 0.3
HAND_EXTRAP = 0.35          # nắm tay = cổ tay + 0.35 * (cổ tay − khuỷu)
POLE_MIN_SCORE = 0.5        # tỉ lệ mẫu trên tia có mặt cắt dạng thanh tối thiểu để nhận là gậy
POLE_ANGLES = (45.0, 135.0)          # độ, 90 = thẳng xuống; 0.5° mỗi bước
POLE_SEARCH = (0.06, 0.55)           # đoạn tia [r0, r1] * h tính từ nắm tay
POLE_HAND_TOL = 0.03                 # dịch tia ±0.03h theo pháp tuyến, bước 6 px
POLE_HALF_W = (0.002, 0.003, 0.005, 0.008, 0.012)   # nửa bề rộng gậy * h (0.012 để bắt gậy tre)
POLE_BAR_CONTRAST = 0.05             # |Lab(tâm) − TB(Lab hai bên)| (Lab chia 255, ảnh blur σ=1.2)
POLE_SIDE_RATIO = 0.7                # |Lab(trái) − Lab(phải)| < 0.7 * contrast (loại viền)
POLE_GAP = 0.05                      # khoảng hụt tối đa (* h) khi dò đầu gậy
POLE_GRIP_UP = 0.12                  # dò ngược lên trên nắm tay tối đa 0.12h (tay cầm)
POLE_MIN_LEN = 0.2                   # gậy dò được (|tip − top|) ngắn hơn 0.2h thì bỏ
POLE_HW_NARROW = 0.9                 # chọn bề rộng hẹp nhất có số mẫu đạt >= 0.9 × tốt nhất
POLE_TAIL_WIN = 0.03                 # cắt đuôi thưa: cửa sổ đo mật độ mẫu đạt dọc gậy (* h)
POLE_TAIL_DENSE = 0.8                # cửa sổ có mật độ >= mức này là thân gậy (thân gậy thật ~1.0)
POLE_TAIL_SPARSE = 0.6               # phần sau cặp cửa sổ đặc cuối có mật độ TB < mức này = đuôi qua cỏ, bỏ
POLE_LAT_MIN_PX = 8.0                # khi dò đầu gậy được lệch ngang khỏi tia tối đa
POLE_LAT_HW = 3.0                    # max(POLE_LAT_MIN_PX, POLE_LAT_HW * hw) px, ±1 px mỗi bước
# Quét hai phía nắm tay (gậy cầm giữa thân, gậy gần nằm ngang): chạy cho tay chưa có gậy sau lượt quét xuống
# + cổng LSD. Gậy = đường thẳng qua chỗ nắm, hai phía (mỗi phía có thể ngắn / ra khỏi ảnh), bỏ qua ±r0 quanh tay.
POLE_LINE_ANGLES = (15.0, 165.0)     # hướng trục (độ, 0 = ngang, 90 = dọc); None = tắt lượt quét hai phía
POLE_LINE_MIN_SCORE = 0.6            # mật độ mẫu đạt trên đoạn đã dò được (không chia cho cả đoạn tìm kiếm)
POLE_LINE_MIN_LEN = 0.15             # tổng chiều dài hai phía (|tip − top|) tối thiểu * h
POLE_LINE_MIN_DOWN = 0.12            # phần gậy dưới chỗ nắm (tới đầu thấp hơn) tối thiểu * h: đường chỉ đi lên trên
                                     # tay / quanh tay là cánh tay, đùi, cột đèn (trên mẫu: giả <= 0.08h, thật >= 0.17h)
POLE_LINE_TRIM_COV = 0.5             # cắt hai đầu gậy hai phía về đoạn có độ phủ LSD (cửa sổ POLE_TAIL_WIN) >= mức này
# Kiểm tra gậy bằng đoạn thẳng LSD (linecheck.py): độ phủ trục gậy bởi đoạn LSD song song, gần trục
POLE_LSD_MIN = 0.25                  # độ phủ tối thiểu để giữ gậy (gậy thật ~0.6, giả ~0.02); 0 = tắt kiểm tra (như trước)
POLE_LSD_PAR_DEG = 4.0               # đoạn LSD lệch góc so với trục <= 4° mới tính là song song
POLE_LSD_MARGIN = 3.0                # đoạn cách trục <= half_w + 3 px
POLE_LSD_STEP = 3.0                  # px, bước lấy mẫu dọc trục khi đo độ phủ
POLE_LSD_CLAHE_CLIP = 2.0            # CLAHE trên ảnh xám trước khi chạy LSD
POLE_LSD_CLAHE_TILE = (8, 8)
POLE_LSD_MAX_SIDE = 1600             # ảnh lớn hơn được thu nhỏ về cạnh dài này trước khi chạy LSD (miền đã chọn ngưỡng)

THIN_BAND_PX = 2            # px, vùng unknown quanh mép gậy khi matting alpha gậy ở ảnh đầy đủ
# Mở rộng mask gậy (polemask.py): vùng khác nền cục bộ quanh dải gậy (gậy thứ hai song song, tay cầm, chóp)
POLE_REFINE = os.environ.get("POLE_REFINE", "1") != "0"   # 0 = chỉ matting dải gậy dò được (như trước)
POLE_REFINE_MAX_HW = 0.005           # dải gậy nửa bề rộng > 0.005 * h (gậy tre, gậy dày đã phủ đủ) thì không mở rộng
POLE_REFINE_BAND = 0.04              # vùng tìm: cách trục <= 0.04 * h
POLE_REFINE_RING = 0.015             # bề rộng vành lấy mẫu nền ngoài vùng tìm (* h, tối thiểu 6 px)
POLE_REFINE_INPAINT_R = 5            # bán kính inpaint (Telea) khi ước lượng nền
POLE_REFINE_DIFF = 22.0              # khoảng cách Lab (thang 8 bit của OpenCV) tối thiểu tới nền ước lượng
POLE_REFINE_TEX_K = 1.0              # ngưỡng += K * gradient Lab cục bộ của nền (bờ, kết cấu nâng ngưỡng)
# Phân vị 75 của gradient Lab ở vành lấy mẫu nền vượt mức này thì không mở rộng (nền gồ ghề: lá, cỏ gần,
# đám đông). Trên 48 gậy thật/giả (19 ảnh): trời/tường/đường ~1-5, ruộng/cỏ xa ~8-11 (vẫn cải thiện),
# bụi lá gần 14.8 làm tràn đốm lá; 12 giữ các ca cải thiện, bỏ ca tràn.
POLE_REFINE_ROUGH_MAX = 12.0
POLE_REFINE_MIN_H = 450              # người thấp hơn 450 px (ảnh làm việc): gậy ~1 px, vùng tìm toàn nền lộn xộn,
                                     # không mở rộng (trên mẫu: chỉ thêm vài chục px, có ca nở theo gậy giả là lá cỏ)
POLE_REFINE_WMUL = 3.0               # nửa bề rộng cục bộ tối đa = max(3 * half_w, 0.006 * h)
POLE_REFINE_WMIN = 0.006
POLE_REFINE_WEND = 2.0               # ở hai đầu gậy (tay cầm, quai, chóp): nửa bề rộng tối đa x 2.0
POLE_REFINE_GAP = 3.0                # phần mới cách dải gậy tối đa max(3 * half_w, 2) px
POLE_REFINE_MIN_LEN = 0.05           # phần thân mới (khớp đường thẳng, giữ pixel cách đường <= nửa bề rộng tối đa):
                                     # phủ >= 0.05 * h dọc trục
POLE_REFINE_ANGLE = 15.0             # và đường lệch hướng trục <= 15°
POLE_REFINE_CONVERGE = 0.015         # và (nếu không phải mép gậy dò được) đường cắt trục gậy cách chỗ nắm
                                     # <= max(2 * nửa bề rộng tối đa, 0.015 * h); không có tay gần trục thì bỏ
                                     # và là một thanh: tương phản Lab với nền sát mép ngoài >= POLE_REFINE_DIFF,
POLE_REFINE_THICK_PX = 3.0           # bề dày (trung vị theo cột) <= bề rộng dải gậy + 3 px
POLE_REFINE_END_AREA = 3.0           # khối ở đầu gậy: diện tích <= 3 * (2 * nửa bề rộng tối đa ở đầu gậy)^2
POLE_REFINE_END_TEX = 4.0            # gradient Lab nền ở đầu gậy > mức này (đất, cỏ; trời/tường ~1-2, núi xa ~5):
                                     # khối ở đầu chỉ giữ phần sát trục (nửa bề rộng tối đa thường)
POLE_REFINE_AREA = 4.0               # mask / dải gậy vượt mức này thì không mở rộng (nghi tràn)
POLE_REFINE_AREA_H = 0.01            # phần thêm vượt 0.01 * h^2 px thì không mở rộng

# Lớp COCO coi là vật cầm tay (YOLO-seg phát hiện cùng lúc với người); mặc định bỏ phương tiện, động vật, đồ nội thất lớn
_HELD_DEFAULT = ("24,25,26,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,"
                 "50,51,52,53,54,55,63,64,65,66,67,73,74,75,76,77,78,79")
HELD_CLASSES = [int(c) for c in os.environ.get("HELD_CLASSES", _HELD_DEFAULT).replace(" ", "").split(",") if c]
HELD_MIN_CONF = 0.25
HELD_HAND_DIST = 0.08       # box vật cách nắm tay <= 0.08 * chiều cao thân người
HELD_TOUCH_FRAC = 0.03      # không có tay: mask vật chạm mask người nới 0.03 * cạnh dài box vật
HELD_MAX_AREA = 1.5         # diện tích vật <= 1.5 * diện tích mask người
