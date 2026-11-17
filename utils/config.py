"""统一配置加载与校验。

根目录 config.yaml 是全项目唯一配置文件：
cameras[]（id/url/task/内联标定）+ models + pole + marker + runtime。
"""
from pathlib import Path

import yaml

from .common import WORKSPACE_ROOT, log

VALID_TASKS = {"pole", "marker"}

DEFAULTS = {
    "runtime": {
        "output_dir": "output",
        "log_level": "INFO",
        "display": False,
        "capture": {"mode": "threads", "reconnect_s": 5, "rtsp_low_latency": True},
        "batch": {
            "pole": {"timeout_ms": 500},
            "marker": {"max_frames": 8, "timeout_ms": 60},
        },
        "queue": {"pole_frame": 2, "marker_frame": 16, "result": 256, "spectrum": 512},
        "init": {
            "persist": True,
            "anchor": {"iou_min": 0.3, "release_after_s": 10.0, "ema_alpha": 0.05},
        },
    },
    "pole": {"sample_fps": 1.0},
    "marker": {
        "fft": {"window_s": 30.0, "snr": 5.0, "min_samples": 32,
                 "min_span_s": 10.0},
        "calibration": {"real_diameter_mm": 100.0, "frames": 8, "min_frames": 1},
    },
}


def _merge_defaults(cfg: dict) -> dict:
    """只补缺省键，不覆盖用户显式配置。"""
    for section, values in DEFAULTS.items():
        node = cfg.setdefault(section, {})
        if isinstance(values, dict):
            for key, val in values.items():
                if isinstance(val, dict):
                    sub = node.setdefault(key, {})
                    if isinstance(sub, dict):
                        for k2, v2 in val.items():
                            sub.setdefault(k2, v2)
                else:
                    node.setdefault(key, val)
    return cfg


def _validate_cameras(cfg: dict) -> None:
    cameras = cfg.get("cameras")
    if not cameras or not isinstance(cameras, list):
        raise ValueError("cameras 必须是非空列表（至少一个摄像头）")
    seen = set()
    for i, cam in enumerate(cameras):
        if not isinstance(cam, dict):
            raise ValueError(f"cameras[{i}] 必须是字典")
        cam_id = cam.get("id")
        if not cam_id:
            raise ValueError(f"cameras[{i}].id 缺失")
        if cam_id in seen:
            raise ValueError(f"摄像头 id 重复: {cam_id}")
        seen.add(cam_id)
        task = cam.get("task")
        if task not in VALID_TASKS:
            raise ValueError(f"cameras[{i}].task 必须是 {VALID_TASKS} 之一，当前: {task}")
        url = cam.get("url")
        # print('=========================',cam)
        if not url:
            raise ValueError(f"cameras[{i}].url 缺失")
        calib = cam.get("calibration")
        if task == "pole" and not calib:
            raise ValueError(
                f"pole 摄像头 {cam_id} 必须提供 calibration（K/dist，外参可选）"
            )
        if calib:
            K = calib.get("K")
            if not K or len(K) != 3 or any(len(r) != 3 for r in K):
                raise ValueError(f"cameras[{i}].calibration.K 必须是 3x3 矩阵")
            dist = calib.get("dist", [])
            if not isinstance(dist, list):
                raise ValueError(f"cameras[{i}].calibration.dist 必须是列表")
            R = calib.get("R")
            if R and (len(R) != 3 or any(len(r) != 3 for r in R)):
                raise ValueError(f"cameras[{i}].calibration.R 必须是 3x3 矩阵")
            t = calib.get("t")
            if t and len(t) != 3:
                raise ValueError(f"cameras[{i}].calibration.t 必须是长度 3 的向量")


def _validate_models(cfg: dict) -> None:
    models = cfg.get("models") or {}
    pole_cams = [c for c in cfg["cameras"] if c["task"] == "pole"]
    marker_cams = [c for c in cfg["cameras"] if c["task"] == "marker"]
    if pole_cams and not models.get("pole_seg"):
        raise ValueError("存在 pole 摄像头但 models.pole_seg 未配置")
    if marker_cams and not models.get("marker_det"):
        raise ValueError("存在 marker 摄像头但 models.marker_det 未配置")


def camera_undistort(cam: dict, pole_mode: str, tag: str = "config") -> bool:
    """返回该相机最终是否去畸变；采集与可视化共用这一处判定。

    cameras[].undistort 缺省时沿用原口径（image_relative 去畸变、world_3d 不去）。
    显式 false 可在 image_relative 下关闭去畸变；显式 true 在 world_3d 下会被
    忽略并告警——world_3d 的 pixel_ray_world 已用 K/dist 逐点反投影，若再对整帧
    cv2.undistort，会把同一畸变扣两次，倾角结果直接错。
    """
    base = pole_mode == "image_relative"
    value = cam.get("undistort")
    if value is None:
        return base
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            value = True
        elif text in {"0", "false", "no", "off"}:
            value = False
        else:
            raise ValueError(
                f"相机 {cam.get('id')} 的 undistort 无法解析: {value!r}"
                "（请用 true/false）")
    value = bool(value)
    if value and not base:
        log(tag, f"相机 {cam.get('id')} undistort=true 与 pole.measurement.mode="
                 f"{pole_mode} 冲突，已忽略：world_3d 在像素点层面已用 K/dist "
                 "反投影，整帧再去畸变会重复")
        return False
    return value


def load_config(path) -> dict:
    """加载并校验统一配置；路径解析为绝对路径。

    cameras[].enabled 相机级开关：显式 enabled: false 的相机在**这里**就被
    剔除（等同未配置），下游（supervisor / capture / inference / dashboard）
    全部自动看不到它——该相机不拉流、不推理、不产出，也不出现在大屏下拉与
    /video 流里；某分支相机全关时 supervisor 不再 spawn 该分支进程。缺省或
    true 均视为启用，旧配置无需改动即可兼容。
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg = _merge_defaults(cfg)
    _validate_cameras(cfg)

    disabled = [c.get("id") for c in cfg.get("cameras", [])
                if c.get("enabled") is False]
    if disabled:
        cfg["cameras"] = [c for c in cfg["cameras"]
                          if c.get("enabled") is not False]
        log("config", f"相机开关关闭，本次运行剔除: {disabled}")
    if not cfg.get("cameras"):
        raise ValueError("所有摄像头都被 enabled:false 关闭，无可运行任务")

    _validate_models(cfg)

    # 统一解析 output_dir / 模型路径为绝对路径
    rt = cfg["runtime"]
    out = Path(rt["output_dir"])
    rt["output_dir"] = str(out if out.is_absolute() else WORKSPACE_ROOT / out)
    cfg["_config_dir"] = str(path.parent)
    cfg["_config_path"] = str(path)
    return cfg


def cameras_by_task(cfg: dict, task: str) -> list:
    return [c for c in cfg["cameras"] if c["task"] == task]
