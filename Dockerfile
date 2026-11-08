# syntax=docker/dockerfile:1
# Các target:
#   frontend  giao diện Streamlit (nhẹ, không có torch)
#   cpu       backend chạy CPU, mọi máy
#   intel     backend OpenVINO trên GPU Intel (Arc / iGPU); base có sẵn Intel compute runtime
#   cuda      backend GPU NVIDIA (torch CUDA + onnxruntime-gpu); cần NVIDIA Container Toolkit trên host
# Dùng qua docker compose (docker-compose.yml + docker-compose.intel.yml / docker-compose.cuda.yml).

ARG PY_IMAGE=python:3.13-slim
ARG OPENVINO_IMAGE=openvino/ubuntu24_runtime:2026.4.1

# ------------------------------------------------------------------ frontend
FROM ${PY_IMAGE} AS frontend
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN pip install --no-cache-dir streamlit requests
COPY frontend/ frontend/
EXPOSE 8501
CMD ["streamlit", "run", "frontend/app.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--server.headless", "true", "--browser.gatherUsageStats", "false"]

# ------------------------------------------------------------------ cpu
FROM ${PY_IMAGE} AS cpu
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 ACCEL_BACKEND=cpu
WORKDIR /app
COPY requirements.txt docker/install-backend.sh ./
RUN sh install-backend.sh https://download.pytorch.org/whl/cpu
COPY backend/ backend/
EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ------------------------------------------------------------------ intel (OpenVINO)
# Base Ubuntu 24.04, Python 3.12 trong /opt/venv, openvino đã có trên PYTHONPATH -> không cài requirements-intel.txt.
FROM ${OPENVINO_IMAGE} AS intel
USER root
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 ACCEL_BACKEND=openvino
WORKDIR /app
COPY requirements.txt docker/install-backend.sh ./
RUN sh install-backend.sh https://download.pytorch.org/whl/cpu
COPY backend/ backend/
EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ------------------------------------------------------------------ cuda (NVIDIA)
# Chưa kiểm thử trên phần cứng thật. CUDA/cuDNN lấy từ các gói nvidia-* mà wheel torch kéo theo.
FROM ${PY_IMAGE} AS cuda
ARG CUDA_TORCH_INDEX=https://download.pytorch.org/whl/cu126
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 ACCEL_BACKEND=cuda
WORKDIR /app
COPY requirements.txt docker/install-backend.sh ./
# R19: rembg[cpu] kéo onnxruntime bản CPU, che bản GPU -> gỡ trước khi cài onnxruntime-gpu
RUN sh install-backend.sh "$CUDA_TORCH_INDEX" \
 && pip uninstall -y onnxruntime \
 && pip install --no-cache-dir onnxruntime-gpu
# để onnxruntime-gpu tìm được libcudnn/libcublas... mà wheel torch cài vào site-packages/nvidia/*/lib
RUN python -c "import glob,os,site; open('/etc/ld.so.conf.d/nvidia-pip.conf','w').write('\n'.join(sorted(glob.glob(os.path.join(site.getsitepackages()[0],'nvidia','*','lib')))) + '\n')" \
 && ldconfig
COPY backend/ backend/
EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
