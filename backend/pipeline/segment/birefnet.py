"""Engine BiRefNet, giới hạn trong mask YOLO của người được chọn.

CPU: rembg + onnxruntime CPU (như cũ). GPU (cuda/mps/openvino): `accel.onnx_runner` với
input shape cố định, tiền/hậu xử lý chép đúng lớp session BiRefNet của rembg
(`rembg/sessions/birefnet_general.py` + `BaseSession.normalize`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .. import accel
from .. import config as C
from ..matting import finish_cutout
from ..occlusion import exclusion_mask
from ..roi import active_roi, roi_stage
from ..schema import Cutout, Person, Scene, SegParams
from .base import ENGINES, person_init_mask, to_float, work_scene
from .gate import gate_to_person

_session = None

# Hằng số tiền xử lý của rembg BiRefNetSessionGeneral.predict (ImageNet mean/std).
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def onnx_path() -> Path:
    """File ONNX BiRefNet mà rembg dùng (`<REMBG_HOME>/models/<tên>/<tên>.onnx`, hoặc
    `~/.u2net/<tên>.onnx` kiểu cũ); chưa có thì rembg tải về."""
    from rembg.sessions import sessions_class

    for sc in sessions_class:
        if sc.name() == C.BIREFNET_MODEL:
            return Path(sc.download_models())
    raise ValueError(f"rembg không có model '{C.BIREFNET_MODEL}'")


def _cpu_alpha(rgb: np.ndarray) -> np.ndarray:
    """Đường CPU: rembg + onnxruntime CPU, giữ nguyên như trước."""
    global _session
    from PIL import Image
    from rembg import new_session, remove

    if _session is None:
        _session = new_session(C.BIREFNET_MODEL, providers=["CPUExecutionProvider"])
    m = remove(Image.fromarray(rgb), session=_session, only_mask=True)
    return np.asarray(m.convert("L"), dtype=np.float32) / 255.0


def _preprocess(rgb: np.ndarray) -> np.ndarray:
    """Như `BaseSession.normalize`: resize LANCZOS về SxS, chia max, (x-mean)/std, NCHW fp32."""
    from PIL import Image

    size = (C.BIREFNET_SIZE, C.BIREFNET_SIZE)
    im = np.array(Image.fromarray(rgb).convert("RGB").resize(size, Image.Resampling.LANCZOS))
    im = im / max(np.max(im), 1e-6)
    x = np.zeros((im.shape[0], im.shape[1], 3))
    for c in range(3):
        x[:, :, c] = (im[:, :, c] - _MEAN[c]) / _STD[c]
    return np.expand_dims(x.transpose((2, 0, 1)), 0).astype(np.float32)


def _postprocess(out: np.ndarray, hw) -> np.ndarray:
    """Như `BiRefNetSessionGeneral.predict`: sigmoid kênh 0, chuẩn hoá min-max, lượng tử
    uint8, resize LANCZOS về kích thước ảnh; trả float32 0..1."""
    from PIL import Image

    pred = 1 / (1 + np.exp(-out[:, 0, :, :]))
    ma, mi = np.max(pred), np.min(pred)
    pred = (pred - mi) / (ma - mi) if ma > mi else np.zeros_like(pred)   # rembg: chia 0 -> NaN
    pred = np.squeeze(pred)
    mask = Image.fromarray((pred * 255).astype("uint8"), mode="L")
    mask = mask.resize((hw[1], hw[0]), Image.Resampling.LANCZOS)
    return np.asarray(mask, dtype=np.float32) / 255.0


def _runner_key() -> str:
    return f"birefnet:{C.BIREFNET_MODEL}"


def _gpu_alpha(rgb: np.ndarray, device: accel.Device) -> np.ndarray:
    s = C.BIREFNET_SIZE
    run = accel.onnx_runner(_runner_key(), str(onnx_path()), device,
                            static_input={C.BIREFNET_INPUT_NAME: (1, 3, s, s)})
    out = run({C.BIREFNET_INPUT_NAME: _preprocess(rgb)})
    return _postprocess(np.asarray(out[0]), rgb.shape[:2])


def _note(stage, text: str) -> None:
    if stage is not None:
        stage.note = (getattr(stage, "note", "") or "") + text


def birefnet_alpha(rgb: np.ndarray, device: Optional[accel.Device] = None,
                   stage=None) -> np.ndarray:
    """BiRefNet -> alpha float32 HxW 0..1 cho ảnh đưa vào (ảnh ROI ở độ phân giải làm việc).
    CPU: rembg; GPU: onnx_runner, lỗi thì rơi về rembg CPU và ghi chú vào `stage.note`."""
    device = device if device is not None else accel.resolve()
    if device.kind != "gpu":
        _note(stage, " · CPU")
        return _cpu_alpha(rgb)
    if accel.onnx_runner_failed(_runner_key(), device):
        # R22: lỗi build còn được nhớ -> chạy CPU luôn, không thử GPU mỗi request.
        _note(stage, f" · {device.backend} vừa lỗi, dùng CPU")
        return _cpu_alpha(rgb)
    alpha, used = accel.with_fallback(device, lambda: _gpu_alpha(rgb, device),
                                      lambda: _cpu_alpha(rgb), stage=stage)
    _note(stage, " · CPU" if used.kind == "cpu" else f" · {used.backend}: {used.name}")
    return alpha


@ENGINES.register("birefnet", label="BiRefNet (mask chính xác, chậm nhất)")
def run(scene: Scene, person: Person, params: SegParams) -> Cutout:
    """
    BiRefNet chạy trên ROI quanh người (độ phân giải làm việc). Model không phân biệt
    từng người nên α được giới hạn bằng mask YOLO của người được chọn (khối `gate`).
    """
    t = params.trace
    roi = params.cached(("roi", person.idx, person.box), lambda: active_roi(scene, person))
    w, h = roi.work_size
    scene_w, person_w = work_scene(scene, person, roi)
    init = person_init_mask(scene_w, person_w)
    with t.stage("init_mask", "Mask ban đầu từ YOLO-seg",
                 note=f"ROI {w}×{h} (scale {roi.scale:.2f})") as s:
        s.overlay(scene_w.rgb, init)
    roi_stage(t, scene.rgb, roi)
    with t.stage("birefnet", "BiRefNet: alpha trên ROI",
                 note=f"{C.BIREFNET_MODEL}, ROI {w}×{h} (scale {roi.scale:.2f})") as s:
        device = accel.resolve(params.device_mode)
        alpha_raw = params.cached(
            ("birefnet", person.idx, person.box, device.backend, device.index, device.dtype),
            lambda: birefnet_alpha(scene_w.rgb, device, stage=s))
        s.show(alpha_raw)
    with t.stage("occlusion", "Loại phần của người khác (thứ tự trước/sau theo vị trí chân)",
                 note=f"{len(scene.others(person))} người khác") as s:
        excl = exclusion_mask(scene_w, person_w, init)
        s.overlay(scene_w.rgb, excl.astype(np.uint8) * 255, color=(235, 60, 60))
    with t.stage("gate", "Giới hạn trong người được chọn",
                 note="mask YOLO nở ra − người khác, bỏ mảnh rời") as s:
        alpha = gate_to_person(alpha_raw, init, person_w.box, excl)
        s.show(alpha)

    tri = np.where(alpha > 0.99, 255, np.where(alpha < 0.01, 0, 128)).astype(np.uint8)
    return finish_cutout(to_float(scene.rgb), alpha, tri, roi, t)
