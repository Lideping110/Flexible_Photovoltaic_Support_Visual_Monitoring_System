"""拉流进程：每个分支一个进程，内部每摄像头一个读流线程。

- pole 分支：线程内按 sample_fps 抽帧（grab+retrieve 保时序），
  并用本摄像头 K/dist 去畸变（image_relative 模式）。
- marker 分支：逐帧解码（sample_fps=None）。
- 帧消息 {"cam_id","frame_id","ts","frame"} → 有界帧队列（丢旧保新）；
  状态事件 → 结果队列（阻塞写，不丢）。
- RTSP 断线重连（DEGRADED 告警）；视频文件读到尾发 EOF 事件后退出线程。
"""
import threading
import time

import cv2
import numpy as np

from .common import (SENTINEL, is_realtime_source, log, open_camera,
                     put_blocking, put_latest)


def _undistorter(calibration: dict):
    if not calibration:
        return lambda frame: frame
    K = np.asarray(calibration["K"], dtype=np.float64)
    dist = np.asarray(calibration.get("dist", []), dtype=np.float64).reshape(-1, 1)

    def und(frame):
        try:
            return cv2.undistort(frame, K, dist)
        except cv2.error:
            return frame

    return und


def _camera_loop(cam, out_q, result_q, stop_event, branch, sample_fps,
                 undistort, reconnect_s):
    """单个摄像头的读流线程。out_q=帧队列；result_q=结果队列（事件）。"""
    from .common import now_ts

    cam_id, url = cam["id"], cam["url"]
    realtime = is_realtime_source(url)
    und = _undistorter(cam.get("calibration")) if undistort else (lambda f: f)
    tag = f"capture-{cam_id}"

    def _event(evt, detail=""):
        put_blocking(result_q, {
            "branch": branch, "kind": "event", "cam_id": cam_id,
            "event": evt, "ts": now_ts(), "detail": detail,
        })

    while not stop_event.is_set():
        cap = open_camera(url)
        if cap is None:
            _event("camera_open_failed", str(url))
            if not realtime:
                out_q.put(SENTINEL)  # 视频文件打不开：一次性失败即下线
                return
            stop_event.wait(reconnect_s)
            continue

        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
        _event("camera_connected")

        frame_id = 0
        sample_period = (1.0 / max(1e-6, float(sample_fps))
                         if sample_fps is not None else None)
        next_sample_t = 0.0
        fail_count = 0
        started = time.perf_counter()

        while not stop_event.is_set():
            ok = cap.grab()
            if not ok:
                fail_count += 1
                if realtime and fail_count >= 30:
                    _event("camera_degraded", "连续 30 次取流失败")
                    break
                if not realtime:
                    cap.release()
                    _event("camera_eof", str(url))
                    out_q.put(SENTINEL)
                    log(tag, f"视频结束: {url}")
                    return
                continue

            source_t = frame_id / src_fps if src_fps > 0 else 0.0
            if (sample_fps is not None and source_t + 1e-9 < next_sample_t):
                frame_id += 1
                continue  # 抽帧：只 grab 不 retrieve

            ok, frame = cap.retrieve()
            if not ok or frame is None:
                fail_count += 1
                continue
            fail_count = 0

            ts = (time.perf_counter() - started) if realtime else source_t
            put_latest(out_q, {
                "cam_id": cam_id, "frame_id": frame_id, "ts": ts,
                "frame": und(frame),
            })
            if sample_period is not None:
                next_sample_t += sample_period
            frame_id += 1

        cap.release()
        if realtime and not stop_event.is_set():
            stop_event.wait(reconnect_s)  # 断线重连循环

    out_q.put(SENTINEL)


def run_capture(branch: str, cameras: list, frame_queues: dict, result_q,
                stop_event, runtime_cfg: dict, pole_cfg: dict = None):
    """拉流进程入口（每分支一个）。frame_queues: {cam_id: Queue}。"""
    tag = f"capture-{branch}"
    log(tag, f"启动，摄像头: {[c['id'] for c in cameras]}")

    reconnect_s = float(runtime_cfg.get("reconnect_s", 5.0))
    sample_fps = None
    undistort = False
    if branch == "pole":
        sample_fps = float((pole_cfg or {}).get("sample_fps", 1.0))
        mode = ((pole_cfg or {}).get("measurement") or {}).get("mode", "image_relative")
        undistort = mode == "image_relative"
        log(tag, f"抽帧 {sample_fps} fps, 去畸变={undistort}")

    threads = []
    for cam in cameras:
        q = frame_queues[cam["id"]]
        t = threading.Thread(
            target=_camera_loop,
            args=(cam, q, result_q, stop_event, branch, sample_fps,
                  undistort, reconnect_s),
            daemon=True,
        )
        t.start()
        threads.append(t)
        time.sleep(0.1)  # 错峰打开摄像头

    for t in threads:
        while t.is_alive() and not stop_event.is_set():
            t.join(timeout=0.5)

    # 兜底：确保每条帧队列都收到哨兵（推理进程据此判定该路下线）
    for cam in cameras:
        try:
            frame_queues[cam["id"]].put(SENTINEL, timeout=2.0)
        except Exception:
            pass
    log(tag, "退出")
