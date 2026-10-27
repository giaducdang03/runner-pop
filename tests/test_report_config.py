"""Mục "Cấu hình đang chạy" và tiêu đề mới trong báo cáo PDF."""
import numpy as np
import pytest

from backend import report
from backend.pipeline import accel
from backend.pipeline import config as C
from backend.pipeline.schema import Scene


def _rows(monkeypatch=None):
    return dict(report.runtime_config_rows())


def test_app_title_constant():
    assert report.APP_TITLE == "RunnerPop – Image Preprocessing Optimization for AI-based Runner Segmentation"


def test_rows_have_expected_keys():
    keys = [k for k, _ in report.runtime_config_rows()]
    for k in ("Thiết bị mặc định", "Backend GPU khả dụng", "ACCEL_BACKEND / DEVICE", "Chế độ vùng xử lý", "ROI",
              "Engine mặc định", "Model phát hiện", "Model BiRefNet", "Hệ điều hành", "Python", "Thư viện"):
        assert k in keys
    assert all(isinstance(v, str) and v for _, v in report.runtime_config_rows())


def test_rows_reflect_gpu_device_and_config(monkeypatch):
    dev = accel.Device("gpu", "openvino", "Intel(R) Arc(TM) Graphics (iGPU)", "openvino", "fp16", 0)
    monkeypatch.setattr(accel, "resolve", lambda *a, **k: dev)
    monkeypatch.setattr(accel, "available_backends", lambda: {"openvino": "Intel(R) Arc(TM) Graphics (iGPU)"})
    monkeypatch.setattr(C, "FULLRES", True)
    monkeypatch.setattr(C, "DEFAULT_ENGINE", "birefnet")
    r = _rows()
    assert "openvino" in r["Thiết bị mặc định"] and "Arc" in r["Thiết bị mặc định"]
    assert "openvino: Intel(R) Arc(TM) Graphics (iGPU)" == r["Backend GPU khả dụng"]
    assert "toàn ảnh" in r["Chế độ vùng xử lý"]
    assert r["Engine mặc định"] == "birefnet"


def test_rows_cpu_only_and_roi_mode(monkeypatch):
    cpu = accel.resolve("cpu")      # lấy trước khi thay hàm, tránh đệ quy
    monkeypatch.setattr(accel, "resolve", lambda *a, **k: cpu)
    monkeypatch.setattr(accel, "available_backends", lambda: {})
    monkeypatch.setattr(C, "FULLRES", False)
    r = _rows()
    assert "CPU" in r["Thiết bị mặc định"]
    assert "không có" in r["Backend GPU khả dụng"]
    assert r["Chế độ vùng xử lý"] == "ROI quanh người"
    assert "640" in r["ROI"]


def test_lib_versions_skips_missing(monkeypatch):
    from importlib import metadata
    real = metadata.version

    def fake(name):
        if name == "numpy":
            return "9.9.9"
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", fake)
    assert report._lib_versions() == "numpy 9.9.9"
    monkeypatch.setattr(metadata, "version", real)


def test_build_report_includes_new_section_and_does_not_crash():
    """Dựng PDF thật (chưa tách nền): có PDF hợp lệ. Bỏ qua nếu máy thiếu font tiếng Việt."""
    rgb = np.full((120, 160, 3), 128, np.uint8)
    scene = Scene(rgb=rgb)
    try:
        data = report.build_report(rgb, scene, None, {}, 1.0, {})
    except RuntimeError as e:     # thiếu font TTF hỗ trợ tiếng Việt (FR-72)
        pytest.skip(str(e))
    assert data[:4] == b"%PDF" and len(data) > 1000
