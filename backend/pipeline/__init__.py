"""
Pipeline tách người chạy khỏi nền (CPU-only), chia thành các giai đoạn độc lập:

    imageio      đọc ảnh, resize về ảnh làm việc
    preprocess/  đánh giá chất lượng + chuỗi bước tiền xử lý CHỈ cho phát hiện   (registry STEPS)
    detect       YOLO-seg + thang thử lại khi không phát hiện được      -> Scene
    segment/     engine tách nền (GrabCut+matting, focus, BiRefNet, ...) -> Cutout (registry ENGINES)
    matting      trimap, alpha matting, khử nhiễm màu (khối dùng chung)
    masks        phép toán mask (khối dùng chung)
    select       tự chọn người chạy chính (chế độ auto)
    compose      ghép người giữ màu + nền đen trắng theo cường độ

Phần nặng chạy MỘT lần, kết quả đóng gói trong `Cutout`. Slider cường độ chỉ gọi `compose()`.
"""
from .bgfilter import BG_FILTERS, apply_bg_filter
from .bib import read_bib
from .compose import alpha_to_png, compose
from .detect import detect_scene
from .imageio import decode_image, resize_to_max_side
from .preprocess import available as available_preprocess
from .segment import available as available_engines
from .segment import ENGINES, build_cutout
from .select import suggest_person
from .schema import Cutout, Person, Scene, SegParams
from .trace import Trace

__all__ = [
    "Cutout", "Person", "Scene", "SegParams", "Trace",
    "decode_image", "resize_to_max_side", "detect_scene", "build_cutout",
    "compose", "alpha_to_png", "read_bib", "BG_FILTERS", "apply_bg_filter", "available_engines", "available_preprocess", "engine_names", "suggest_person",
]


def engine_names() -> list:
    return ENGINES.names()
