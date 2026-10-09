"""公共件：日志、相机打开、队列背压、模型路径选择、JSON 序列化。

本模块（以及 capture/supervisor）绝不 import ultralytics/OpenVINO——
模型只在推理子进程内加载（对齐 JRY_BAA_121422 的 spawn 纪律）。
"""
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]

# 队列哨兵：表示"该通道的发送方已结束"。
SENTINEL = None


def log(tag: str, msg) -> None:
    """统一前缀日志（stdout，spawn 子进程各自打印）。"""
    print(f"[{tag}] {msg}", file=sys.stdout, flush=True)


def open_camera(source, low_latency: bool = True):
    """打开摄像头/RTSP/视频文件源，失败返回 None。

    沿用 marker_subpixel_tracker 的低延迟 RTSP 设置（TCP + nobuffer）。
    """
    value = source
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if low_latency and isinstance(value, str) and value.lower().startswith(("rtsp://", "rtsps://")):
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            "rtsp_transport;tcp|fflags;nobuffer|probesize;32|analyzeduration;0",
        )
        cap = cv2.VideoCapture(value, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(value)
    if not cap.isOpened():
        cap.release()
        return None
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


def is_realtime_source(url) -> bool:
    """RTSP/摄像头编号视为实时源（用墙钟时间戳）；文件用帧号换算时间。"""
    if isinstance(url, int):
        return True
    text = str(url).strip().lower()
    return text.startswith(("rtsp://", "rtsps://")) or text.isdigit()


def put_latest(q, item) -> None:
    """帧队列背压：满时丢最旧再放（保新丢旧，各路内部保持有序）。"""
    from queue import Empty, Full

    try:
        q.put_nowait(item)
    except Full:
        try:
            q.get_nowait()
        except Empty:
            pass
        try:
            q.put_nowait(item)
        except Full:
            pass


def put_blocking(q, item, timeout: float = 5.0) -> bool:
    """结果队列：结果不可丢，阻塞写；超时返回 False（仅日志告警）。"""
    try:
        q.put(item, timeout=timeout)
        return True
    except Exception:
        return False


def get_any(queues, timeout: float):
    """轮询多条帧队列，返回 (cam_id, item)；item 可能是 SENTINEL，无则 None。

    queues 为 [(cam_id, Queue), ...]，按队列数均分超时以保持公平。
    """
    if not queues:
        return None
    per = max(0.005, timeout / max(1, len(queues)))
    for cam_id, q in queues:
        try:
            return cam_id, q.get(timeout=per)
        except Exception:
            continue
    return None


def to_float(v):
    """数值清洗：转为 float，非有限值（NaN/Inf）或不可转返回 None。"""
    try:
        v = float(v)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def to_float_list(v):
    """数值列表清洗：逐元素 to_float；不可迭代返回 None。"""
    if v is None:
        return None
    try:
        return [to_float(x) for x in v]
    except TypeError:
        return None


def jsonable(obj):
    """numpy → 原生 Python，供 json 序列化。NaN 保留为 null。"""
    import math

    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return [jsonable(v) for v in obj.tolist()]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return str(obj)


def is_openvino_model(path) -> bool:
    path = Path(path)
    return path.is_dir() and any(path.glob("*.xml"))


def select_model_weights(model_cfg: dict, config_dir: Path) -> Path:
    """纯 pathlib 版模型选择：优先 OpenVINO 目录，回退 .pt 权重。

    逻辑对齐 marker_subpixel_tracker 原 select_model_weights，
    但不 import 任何推理框架（Supervisor 安全）。
    """
    candidates = []
    for key in ("openvino_dir", "openvino_model", "openvino_weights"):
        value = model_cfg.get(key)
        if value:
            p = Path(value)
            candidates.append(p if p.is_absolute() else WORKSPACE_ROOT / p)
    configured = model_cfg.get("weights")
    if configured:
        stem = Path(configured).stem
        for name in (f"{stem}_openvino_model",):
            candidates.append(WORKSPACE_ROOT / name)
    for candidate in candidates:
        if is_openvino_model(candidate):
            return candidate
    if configured:
        p = Path(configured)
        p = p if p.is_absolute() else WORKSPACE_ROOT / p
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"未找到可用模型。OpenVINO 候选: {candidates}; "
        f"weights 回退: {configured}"
    )


def now_ts() -> float:
    return time.time()
