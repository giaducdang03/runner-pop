#!/bin/sh
# Cài phần chung của image backend (dùng cho mọi target: cpu / intel / cuda).
# Đối số 1: PyTorch index URL (cpu hoặc cuXXX). Chạy trong WORKDIR /app, với quyền root.
set -eu
TORCH_INDEX="$1"

# Thư viện hệ thống: libGL + glib cho opencv-python (ultralytics), DejaVu cho PDF tiếng Việt, curl cho healthcheck.
# Ubuntu 24.04 đổi tên glib thành libglib2.0-0t64, Debian vẫn là libglib2.0-0.
apt-get update
GLIB=libglib2.0-0
if apt-cache show libglib2.0-0t64 >/dev/null 2>&1; then GLIB=libglib2.0-0t64; fi
apt-get install -y --no-install-recommends libgl1 "$GLIB" fonts-dejavu-core curl ca-certificates
rm -rf /var/lib/apt/lists/*

# torch phải cài TRƯỚC requirements.txt, nếu không pip kéo bản torch mặc định từ PyPI.
pip install --no-cache-dir torch torchvision --index-url "$TORCH_INDEX"
pip install --no-cache-dir -r requirements.txt

python -c "import cv2, torch, onnxruntime, pymatting, rembg, rapidocr_onnxruntime, ultralytics, fpdf; print('torch', torch.__version__)"

# Tải sẵn trọng số YOLO vào /app để container không cần mạng cho bước detect.
python -c "from ultralytics import YOLO; YOLO('yolov8s-seg.pt')"
python -c "from ultralytics import YOLO; YOLO('yolov8s-pose.pt')"
