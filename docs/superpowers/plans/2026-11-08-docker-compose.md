# Chạy app bằng Dockerfile + docker compose (cpu / intel / cuda) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `docker compose up --build` chạy backend FastAPI + giao diện Streamlit; thêm file override để chạy backend trên GPU Intel (OpenVINO) hoặc NVIDIA (CUDA).

**Architecture:** Một `Dockerfile` nhiều target: `frontend` (chỉ streamlit + requests), `cpu`, `intel`, `cuda`. Phần cài chung của backend nằm trong `docker/install-backend.sh` (apt deps, torch theo index, requirements, tải sẵn YOLO). `docker-compose.yml` chạy CPU; `docker-compose.intel.yml` / `docker-compose.cuda.yml` override target, `ACCEL_BACKEND` và cấp GPU. Cache model bind mount ra host (`./docker-data/cache` → `/cache`). Frontend lấy mọi ảnh phía server nên trình duyệt không cần gọi `http://backend:8000`.

**Tech Stack:** Docker Desktop (WSL2), compose v2, `python:3.13-slim`, `openvino/ubuntu24_runtime:2026.4.1` (Python 3.12), torch 2.14.1 (index `cpu` / `cu126`).

**Spec:** yêu cầu người dùng 2026-11-08: Dockerfile + compose cho nhiều môi trường, volume mount ra host. Đã xác minh trên máy dev (Core Ultra 7 155H, Windows 11): container `openvino/ubuntu24_runtime` với `--device /dev/dxg -v /usr/lib/wsl:/usr/lib/wsl` thấy `['CPU', 'GPU']`, GPU = `Intel(R) Graphics [0x7d55] (iGPU)`.

## Global Constraints

- Port backend `8000`, frontend `8501` (đổi được bằng `BACKEND_PORT`, `FRONTEND_PORT`).
- Cache trên host: `${RUNNERPOP_CACHE:-./docker-data/cache}` mount vào `/cache`; `ACCEL_CACHE_DIR=/cache/accel`, `U2NET_HOME=REMBG_HOME=/cache/rembg`, `NUMBA_CACHE_DIR=/cache/numba`. `docker-data/` nằm trong `.gitignore`.
- torch luôn cài từ PyTorch index **trước** `requirements.txt`. CUDA: gỡ `onnxruntime` trước khi cài `onnxruntime-gpu` (R19).
- Intel trên Windows/WSL2: `devices: /dev/dxg` + mount `/usr/lib/wsl`; host Linux dùng `/dev/dri`.
- `cuda` chỉ build, chưa chạy được trên phần cứng thật (máy dev không có NVIDIA).
- Comment tiếng Việt, commit tiếng Anh. Không benchmark.

## Review Focus

1. Ảnh vỡ trong trình duyệt khi `API_URL=http://backend:8000` → Task 1 (test `tests/test_frontend_no_browser_urls.py`).
2. OpenCV thiếu `libGL`/glib trên image slim/Ubuntu 24 (`libglib2.0-0t64`) → smoke import trong `install-backend.sh`.
3. PDF báo cáo cần font DejaVu → `fonts-dejavu-core`; Task 4 tải PDF.
4. Container `intel` âm thầm chạy CPU → Task 4 kiểm tra `/health` / trace báo thiết bị GPU OpenVINO.
5. Code chạy trên Python 3.12 (image intel) dù dev trên 3.13 → chạy `pytest` trong container intel.

---

### Task 1: Frontend không để trình duyệt gọi thẳng backend ✅

`frontend/app.py` dòng 322, 528, 529: `st.image(api_get(...).content, ...)`. Test `tests/test_frontend_no_browser_urls.py`. Commit `22319a8`.

### Task 2: Dockerfile nhiều target + install script

**Files:** `Dockerfile`, `docker/install-backend.sh`, `.dockerignore` (allowlist: `backend/ frontend/ docker/ requirements.txt`), `.gitattributes` (`*.sh`, `Dockerfile` → LF).

- [x] Build `cpu` + `frontend`: `docker compose build` → smoke import in ra `torch 2.14.1+cpu`.
- [x] Build `intel`: `docker compose -f docker-compose.yml -f docker-compose.intel.yml build backend`.
- [ ] Build `cuda`: `docker compose -f docker-compose.yml -f docker-compose.cuda.yml build backend` — bỏ qua trên máy dev (không có NVIDIA, image >10 GB); chỉ `config --quiet`.
- [x] Commit.

### Task 3: Compose files

**Files:** `docker-compose.yml`, `docker-compose.intel.yml`, `docker-compose.cuda.yml`, `.gitignore` (+ `docker-data/`).

- [x] `docker compose config --quiet` cho cả 3 tổ hợp.
- [x] `docker compose up -d` → `backend` healthy, `http://localhost:8501` mở được; sau warm-up có `docker-data/cache/rembg/**/*.onnx` trên host.
- [x] Commit.

### Task 4: Kiểm tra chạy thật + README

- [x] CPU: tải ảnh → xử lý → viewer → tải PNG + PDF.
- [x] Intel: `up` với override → `/health` báo backend openvino/GPU; xử lý 1 ảnh, trace các bước YOLO/BiRefNet ở `GPU` không có ghi chú "đã dùng CPU"; `pytest` trong container intel pass (mount `tests/`, `pytest.ini`).
- [x] README: mục *Chạy bằng Docker* (3 lệnh, cache trên host, biến tuỳ chỉnh, ghi chú WSL2 vs Linux, `MSYS_NO_PATHCONV=1` khi dùng `docker run` trong Git Bash). Cập nhật bảng trạng thái kiểm thử `openvino` nếu bước trên đạt.
- [x] Commit.

## Ngoài phạm vi

- Chạy thật target `cuda`; MPS (Docker trên Mac không truy cập được Metal).
- Đẩy image lên registry, CI.
