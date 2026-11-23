# -*- coding: utf-8 -*-
"""靶标视频 → 累计位移 → 主频：video_calibration 界面的后端核心。

只复用项目已有算法原子，不重写检测/亚像素/谱算法：

    utils.common.select_model_weights       模型选择（OpenVINO 优先）
    utils.inference.DetInference            批量检测
    utils.inference.make_byte_tracker       多目标 ID 关联
    utils.inference.tracker_update          取带 id 的原始检测框
    marker_subpixel_tracker.pipeline.localize_one          亚像素中心
    marker_subpixel_tracker.features.measure_outer_diameter_px  外圈直径自标定
    marker_subpixel_tracker.spectrum._segment_spectrum     主频（legacy/六门同源）

编排逻辑（mm_per_px 标定 / ref 锚点 / cum 位移）与
marker_subpixel_tracker/worker.py 第 361-393 行同源；区别只是不进队列、
不写主链路产物、不启动 supervisor，供本地界面同步/异步调用。

注意（与项目约定一致）：
- marker 分支固定不去畸变，本模块不做 cv2.undistort。
- 时间轴用 frame_id / src_fps 等间隔生成，保证 FFT 频率轴正确。
- 输入用 cum_* 0 阶累计位移（视觉天然量），不差分到加速度。
"""
from __future__ import annotations

import os
import sys
from typing import Any, Callable

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------- 配置
def load_settings(config_path: str = "") -> dict[str, Any]:
    """直接读 config.yaml 的相关小节，不走 load_config（它会剔除 disabled 相机）。"""
    import yaml

    config_path = config_path or os.path.join(_PROJECT_ROOT, "config.yaml")
    if not os.path.isabs(config_path):
        config_path = os.path.join(_PROJECT_ROOT, config_path)
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    marker = cfg.get("marker") or {}
    det = (cfg.get("models") or {}).get("marker_det") or {}
    return {
        "config_path": config_path,
        "config_dir": os.path.dirname(config_path),
        "det": det,
        "localization": marker.get("localization") or {},
        "calibration": marker.get("calibration") or {},
        "fft": marker.get("fft") or {},
    }


# ---------------------------------------------------------------- 主测量
def measure_video(video_path: str, settings: dict[str, Any] | None = None,
                  *, conf: float | None = None, iou: float | None = None,
                  device: str | None = None, max_frames: int = 0,
                  frame_stride: int = 1,
                  progress: Callable[[str, int, int, str], None] | None = None
                  ) -> dict[str, Any]:
    """逐帧检测 + 亚像素 + 自标定，返回按 target 分组的累计位移样本。

    frame_stride=2 表示每 2 帧取 1 帧；时间戳仍用原 frame_id/src_fps，
    因此 dt 会翻倍、Nyquist 减半（分析频率上限随之降低），不做混叠修正。
    """
    import cv2

    from utils.common import select_model_weights
    from utils.inference import DetInference, make_byte_tracker, tracker_update
    from marker_subpixel_tracker.pipeline import localize_one
    from marker_subpixel_tracker.features import measure_outer_diameter_px

    settings = settings or load_settings()
    det_cfg = settings["det"]
    loc_cfg = settings["localization"]
    cal_cfg = settings["calibration"]
    weights = select_model_weights(det_cfg, settings["config_dir"])
    conf_v = float(conf if conf is not None else det_cfg.get("confidence", 0.25))
    iou_v = float(iou if iou is not None else det_cfg.get("iou", 0.6))
    dev = device or det_cfg.get("device", "cpu")
    cls_id = int(det_cfg.get("class_id", 0))
    tracker_name = det_cfg.get("tracker", "bytetrack.yaml")
    real_mm = float(cal_cfg.get("real_diameter_mm", 100.0))
    calib_frames = int(cal_cfg.get("frames", 8))
    calib_min = int(cal_cfg.get("min_frames", 1))
    if not loc_cfg:
        raise ValueError("config.yaml 缺少 marker.localization 亚像素参数")

    def emit(stage: str, done: int, total: int, msg: str) -> None:
        if progress:
            try:
                progress(stage, int(done), int(total), str(msg))
            except Exception:
                pass

    emit("model", 0, 1, f"加载模型 {os.path.basename(str(weights))}")
    engine = DetInference(weights, conf_v, cls_id, dev, tracker_name,
                          "video_gate", iou=iou_v)
    bt = make_byte_tracker(tracker_name)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if src_fps <= 0 or src_fps != src_fps:
        src_fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    limit = min(total_frames, max_frames) if max_frames > 0 else total_frames
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    emit("decode", 0, limit or 1, f"FPS={src_fps:.3f} 帧数={total_frames} 尺寸={width}x{height}")

    # raw_track_id -> 该目标的标定/状态（与 worker.py session.calib 同构）
    states: dict[int, dict[str, Any]] = {}
    samples: dict[str, list[dict[str, Any]]] = {}
    fid = 0
    processed = 0
    n_det = 0
    n_located = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if limit and fid >= limit:
                break
            cur_fid = fid
            fid += 1
            if frame_stride > 1 and (cur_fid % frame_stride) != 0:
                continue
            processed += 1
            res = engine.predict([frame])[0]
            dets = tracker_update(bt, res, frame)
            n_det += len(dets)
            for det in dets:
                rid = int(det["id"])
                st = states.setdefault(rid, {
                    "tried": 0, "diams": [], "mm_per_px": None,
                    "ready": False, "ref": None,
                    "prev_center": None, "prev_box": None,
                })
                center, _lm, continuous = localize_one(
                    frame, det["box"], loc_cfg,
                    st["prev_center"], st["prev_box"])
                st["prev_box"] = det["box"]
                # 亚像素外圈直径自标定：前 calib_frames 帧尝试，满 min_frames 次即就绪
                if not st["ready"] and st["tried"] < calib_frames:
                    st["tried"] += 1
                    d = measure_outer_diameter_px(frame, det["box"])
                    if d and float(d[0]) > 0:
                        st["diams"].append(float(d[0]))
                        if len(st["diams"]) >= calib_min:
                            st["mm_per_px"] = real_mm / float(np.mean(st["diams"]))
                            st["ready"] = True
                if center is None:
                    st["prev_center"] = None
                    continue
                n_located += 1
                xy = np.asarray(center, dtype=float)
                st["prev_center"] = xy.copy()
                if not st["ready"] or st["mm_per_px"] is None:
                    continue
                if st["ref"] is None:
                    st["ref"] = xy.copy()
                dx_px = float(xy[0] - st["ref"][0])
                dy_px = float(xy[1] - st["ref"][1])
                mmp = float(st["mm_per_px"])
                pid = f"T{rid}"
                samples.setdefault(pid, []).append({
                    "ts": cur_fid / src_fps,
                    "frame_id": cur_fid,
                    "pid": pid,
                    "raw_track_id": rid,
                    "cx_px": float(xy[0]), "cy_px": float(xy[1]),
                    "dx_px": dx_px, "dy_px": dy_px,
                    "cum_dx_mm": dx_px * mmp, "cum_dy_mm": dy_px * mmp,
                    "mm_per_px": mmp,
                    "continuous": bool(continuous),
                })
            if progress and (processed % 10 == 0 or processed == limit):
                emit("decode", processed, limit or 1,
                     f"已处理 {processed}/{limit} 帧，检出 {n_det} 框，定位 {n_located} 次")
    finally:
        cap.release()

    emit("done", processed, limit or 1, "测量完成")
    # 只保留有足够样本的 target
    samples = {pid: v for pid, v in samples.items() if len(v) >= 8}
    return {
        "video": os.path.basename(video_path),
        "video_path": os.path.abspath(video_path),
        "fps": round(src_fps, 5),
        "width": width, "height": height,
        "n_frames": fid, "n_processed": processed,
        "n_detections": n_det, "n_located": n_located,
        "frame_stride": frame_stride,
        "targets": {
            pid: {
                "mm_per_px": round(float(v[-1]["mm_per_px"]), 6),
                "n_valid": len(v),
                "duration_s": round(v[-1]["ts"] - v[0]["ts"], 3),
                "samples": v,
            } for pid, v in sorted(samples.items())
        },
        "settings": {
            "conf": conf_v, "iou": iou_v, "device": dev,
            "real_diameter_mm": real_mm, "weights": str(weights),
        },
    }


# ---------------------------------------------------------------- 频率提取
def _uniform(ts: list[float], ys: list[float], dt: float
             ) -> tuple[np.ndarray, np.ndarray]:
    """线性插值到等间隔栅格（丢帧/跟踪丢失帧补齐，供 FFT 使用）。"""
    t = np.asarray(ts, dtype=float)
    y = np.asarray(ys, dtype=float)
    keep = np.concatenate(([True], np.diff(t) > 1e-9))
    t, y = t[keep], y[keep]
    if len(t) < 2:
        return t, y
    n = max(8, int(round((t[-1] - t[0]) / dt)) + 1)
    tu = t[0] + np.arange(n) * dt
    return tu, np.interp(tu, t, y)


def extract_frequencies(measurement: dict[str, Any], *,
                        axis: str = "dy", window_s: float = 30.0,
                        snr: float = 5.0, min_samples: int = 32,
                        min_span_s: float = 10.0,
                        gate_cfg: dict[str, Any] | None = None
                        ) -> dict[str, Any]:
    """对测量结果的每个 target 按固定不重叠窗提取主频，口径同主链路/看板。

    gate_cfg=None → legacy 单段 FFT + SNR 门；给出标定 JSON 加载结果 →
    启用 G0~G3 + 丢帧率门 + ID 连续性门，判定在 px 域、输出幅值在 mm 域。
    """
    from marker_subpixel_tracker.spectrum import _segment_spectrum
    from marker_subpixel_tracker.gate_core import evaluate_window

    if axis not in ("dx", "dy"):
        raise ValueError("axis 只能是 dx / dy")
    W = float(window_s)
    dt = 1.0 / float(measurement["fps"])
    out: dict[str, Any] = {
        "axis": axis, "window_s": W, "dt": round(dt, 6),
        "nyquist_hz": round(0.5 / dt, 4),
        "mode": "calibrated" if gate_cfg else "legacy",
        "targets": {},
    }
    for pid, info in measurement["targets"].items():
        smp = info["samples"]
        t_all = [s["ts"] for s in smp]
        pk = f"cum_{axis}_mm"
        px_key = f"{axis}_px"
        t_max = t_all[-1]
        windows: list[dict[str, Any]] = []
        seg = 0
        while seg * W <= t_max + 1e-9:
            lo, hi = seg * W, (seg + 1) * W
            win = [s for s in smp if lo <= s["ts"] < hi]
            is_tail = bool(win) and hi > t_max + 1e-9
            if is_tail and smp[-1] not in win:
                win.append(smp[-1])
            if len(win) >= 4:
                tu, y_mm = _uniform([s["ts"] for s in win],
                                    [float(s[pk]) for s in win], dt)
                _, y_px = _uniform([s["ts"] for s in win],
                                   [float(s[px_key]) for s in win], dt)
                need_n = (4 if is_tail else int(min_samples))
                need_span = (0.0 if is_tail else min(float(min_span_s), W))
                if gate_cfg:
                    # 标定模式：判定在 px 域、输出幅值在 mm 域，两序列必须分传
                    r = evaluate_window(tu, y_px, y_mm, float(snr), cfg=gate_cfg,
                                        window_s=W, min_samples=need_n,
                                        min_span_s=need_span)
                    if r is not None and r.get("n", 0) < 4:
                        r = None
                else:
                    r = _segment_spectrum(tu, y_mm, float(snr), window_s=W,
                                          min_samples=need_n, min_span_s=need_span)
                if r is not None:
                    windows.append({
                        "seg": seg, "t0": round(lo, 2), "t1": round(min(hi, t_max), 2),
                        "n": int(r.get("n") or 0),
                        "f1_hz": r.get("f1_hz"), "snr": r.get("snr"),
                        "valid": bool(r.get("valid")),
                        "reason": r.get("reason") or "",
                        "gates": r.get("gates") or {},
                        "freq": r.get("freq") or [], "power": r.get("power") or [],
                    })
            seg += 1
            if seg > 10000:  # 防御：异常长时间轴
                break
        valid = [w["f1_hz"] for w in windows if w["valid"] and w["f1_hz"]]
        out["targets"][pid] = {
            "n_valid_samples": len(smp), "mm_per_px": info["mm_per_px"],
            "windows": windows,
            "n_windows": len(windows), "n_valid_windows": len(valid),
            "f1_list_hz": [round(v, 4) for v in valid],
            "f1_mean_hz": round(float(np.mean(valid)), 4) if valid else None,
            "f1_median_hz": round(float(np.median(valid)), 4) if valid else None,
            "f1_std_hz": round(float(np.std(valid)), 4) if len(valid) > 1 else None,
        }
    return out
