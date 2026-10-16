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
from pathlib import Path

import cv2
import numpy as np

from .common import (SENTINEL, is_realtime_source, log, open_camera,
                     put_backpressure, put_blocking, put_latest)


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


class RollingBuffer:
    """RTSP 实时流滚动缓冲：按固定时长分段落盘，供振动报警切窗。

    每个实时相机一个实例；capture 读流线程每帧把 (ts, frame) 推入 push()，由它按
    seg_s 切分写入 output/rtsp_buffer/<cam_id>/seg_XXXXXX.mp4（mp4v，原始帧、不叠加），
    并保留最近 keep_s 秒的段。切窗时 cut() 从覆盖 [t0,t1] 的段里拼出片段，按需转 H.264
    供 web 播放。

    ts 口径：必须与该相机 displacement 时序同源（capture 计算的源相对秒），切窗才能对齐。
    """

    def __init__(self, cam_id, output_dir, seg_s=10.0, keep_s=120.0, fps=25.0):
        self.cam_id = cam_id
        self.seg_s = float(seg_s)
        self.keep_s = float(keep_s)
        self.fps = float(fps)
        self.dir = Path(output_dir) / "rtsp_buffer" / cam_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.segs = []            # [{"idx","ts0","ts1","path","fps"}]
        self._writer = None
        self._cur_idx = None
        self._cur_count = 0
        self._lock = threading.Lock()

    def push(self, ts, frame):
        if frame is None:
            return
        try:
            h, w = frame.shape[:2]
            idx = int(ts // self.seg_s)
            with self._lock:
                if self._writer is None or idx != self._cur_idx:
                    self._close()
                    self._cur_idx = idx
                    ts0 = idx * self.seg_s
                    ts1 = ts0 + self.seg_s
                    path = self.dir / f"seg_{idx:06d}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    vw = cv2.VideoWriter(str(path), fourcc, self.fps, (w, h))
                    if vw.isOpened():
                        self._writer = vw
                        self._cur_count = 0
                        self.segs.append({"idx": idx, "ts0": ts0, "ts1": ts1,
                                          "path": str(path), "fps": self.fps})
                        # 修剪超过 keep_s 的旧段，避免磁盘无限增长
                        cutoff = ts - self.keep_s
                        while self.segs and self.segs[0]["ts1"] < cutoff:
                            old = self.segs.pop(0)
                            try:
                                Path(old["path"]).unlink(missing_ok=True)
                            except Exception:
                                pass
                    else:
                        self._writer = None
                if self._writer is not None:
                    self._writer.write(frame)
                    self._cur_count += 1
        except Exception as exc:
            log("capture", f"滚动缓冲写入失败 {self.cam_id}: {exc}")

    def _close(self):
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception:
                pass
            self._writer = None
        # 收尾时把刚结束段的真实 fps 补登（按实际写入帧数），切窗帧↔时映射更准
        if self.segs and self._cur_count:
            self.segs[-1]["fps"] = max(1.0, self._cur_count / self.seg_s)

    def cut(self, t0, t1, out_path, transcode=True):
        from pathlib import Path as _P
        out_path = _P(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            segs = [s for s in self.segs
                    if not (s["ts1"] <= t0 or s["ts0"] >= t1)]
            segs.sort(key=lambda s: s["ts0"])
        if not segs:
            return {"status": "skip", "reason": "缓冲中无覆盖该窗口的片段"}
        probe = cv2.VideoCapture(str(segs[0]["path"]))
        ok, frame0 = probe.read()
        probe.release()
        if not ok or frame0 is None:
            return {"status": "skip", "reason": "首段无法读取"}
        h, w = frame0.shape[:2]
        vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             self.fps, (w, h))
        written = 0
        for s in segs:
            cap = cv2.VideoCapture(str(s["path"]))
            if not cap.isOpened():
                cap.release()
                continue
            i = 0
            while True:
                ok, fr = cap.read()
                if not ok or fr is None:
                    break
                fts = s["ts0"] + i / s["fps"]
                if t0 - 1e-3 <= fts <= t1 + 1e-3:
                    vw.write(fr)
                    written += 1
                i += 1
            cap.release()
        vw.release()
        if written == 0:
            try:
                out_path.unlink(missing_ok=True)
            except Exception:
                pass
            return {"status": "skip", "reason": "窗口内无帧"}
        if transcode:
            try:
                from viz.export import _to_h264
                _to_h264(out_path)
            except Exception as exc:
                log("capture", f"RTSP 片段 H.264 转码失败（保留 mp4v）: {exc}")
        return {"status": "ok", "out": str(out_path), "frames": written}


def _clip_cut_loop(clip_q, buffers, stop_event):
    """消费 clip_q 切窗请求：从对应相机滚动缓冲拼出片段（capture 进程内线程驱动）。"""
    def _do(req):
        cam = req.get("cam_id")
        buf = buffers.get(cam)
        if buf is None:
            return
        try:
            res = buf.cut(req["t0"], req["t1"], req["out_path"],
                         req.get("transcode", True))
            log("capture", f"RTSP 片段切窗 {cam} {req['t0']:.1f}~{req['t1']:.1f}s "
                           f"→ {res.get('status')} {req['out_path']}")
        except Exception as exc:
            log("capture", f"RTSP 片段切窗失败 {cam}: {exc}")

    while not stop_event.is_set():
        try:
            req = clip_q.get(timeout=0.5)
        except Exception:
            continue
        if req is None:
            break
        _do(req)
    # 停止后排空剩余请求（避免漏切末段报警窗口）
    while True:
        try:
            req = clip_q.get_nowait()
        except Exception:
            break
        if req is None:
            break
        _do(req)


def _camera_loop(cam, out_q, result_q, stop_event, branch, sample_fps,
                 undistort, reconnect_s, block_on_full=False, rolling=None):
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
            log(tag, f"{cam_id} 打开摄像头失败（网络/凭证/通道？），{reconnect_s}s 后重连: {url}")
            if not realtime:
                out_q.put(SENTINEL)  # 视频文件打不开：一次性失败即下线
                return
            stop_event.wait(reconnect_s)
            continue

        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
        _event("camera_connected")
        log(tag, f"{cam_id} 摄像头已连接（realtime={realtime}, 源FPS≈{src_fps:.1f}）: {url}")

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
                    log(tag, f"{cam_id} 连续 30 次取流失败，断开并重连")
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
            msg = {"cam_id": cam_id, "frame_id": frame_id, "ts": ts,
                   "frame": und(frame)}
            # 实时流滚动缓冲：把同源 ts 的帧推入缓冲，供振动报警切窗复用
            if rolling is not None:
                try:
                    rolling.push(ts, frame)
                except Exception:
                    pass
            if block_on_full:
                # 文件源：背压阻塞写，队列满则等待消费方腾位，绝不丢帧，
                # 保证逐帧分析（marker 亚像素跟踪）的连续性。
                put_backpressure(out_q, msg, stop_event)
            else:
                # 实时源：丢旧保新，限制端到端延迟（丢帧可接受）。
                put_latest(out_q, msg)
            if sample_period is not None:
                next_sample_t += sample_period
            frame_id += 1
            if frame_id % 100 == 0:
                log(tag, f"{cam_id} 已取流 {frame_id} 帧（realtime={realtime}）")

        cap.release()
        if realtime and not stop_event.is_set():
            stop_event.wait(reconnect_s)  # 断线重连循环

    out_q.put(SENTINEL)


def run_capture(branch: str, cameras: list, frame_queues: dict, result_q,
                stop_event, runtime_cfg: dict, pole_cfg: dict = None,
                ready_event=None, clip_q=None, output_dir="output",
                rtsp_rolling_s=120.0, rtsp_seg_s=10.0):
    """拉流进程入口（每分支一个）。frame_queues: {cam_id: Queue}。

    ready_event: 推理进程预热完成后置位；capture 在其置位前不开始抽帧，
    避免消费方冷启动期间帧被有界队列（丢旧保新）冲掉。
    """
    tag = f"capture-{branch}"
    log(tag, f"启动，摄像头: {[c['id'] for c in cameras]}")

    if ready_event is not None:
        log(tag, "等待推理进程预热（模型编译）就绪…")
        while not ready_event.is_set() and not stop_event.is_set():
            stop_event.wait(0.5)
        if stop_event.is_set():
            log(tag, "收到停止信号，放弃启动")
            return
        log(tag, "推理就绪，开始抽帧")

    reconnect_s = float(runtime_cfg.get("reconnect_s", 5.0))
    sample_fps = None
    undistort = False
    if branch == "pole":
        sample_fps = float((pole_cfg or {}).get("sample_fps", 1.0))
        mode = ((pole_cfg or {}).get("measurement") or {}).get("mode", "image_relative")
        undistort = mode == "image_relative"
        log(tag, f"抽帧 {sample_fps} fps, 去畸变={undistort}")

    threads = []
    buffers = {}          # cam_id -> RollingBuffer（仅实时 marker 相机）
    for cam in cameras:
        q = frame_queues[cam["id"]]
        # 文件源（非实时）背压不丢帧；实时源丢旧保新以限制延迟
        block_on_full = not is_realtime_source(cam["url"])
        rolling = None
        # 仅 marker 分支的实时相机需要滚动缓冲（振动报警切窗）；pole 不产振动报警，
        # 且 RTSP 实时流的 ts 与 displacement 同源，切窗才能对齐。
        if branch == "marker" and is_realtime_source(cam["url"]):
            rolling = RollingBuffer(cam["id"], output_dir, rtsp_seg_s,
                                    rtsp_rolling_s, 25.0)
            buffers[cam["id"]] = rolling
        t = threading.Thread(
            target=_camera_loop,
            args=(cam, q, result_q, stop_event, branch, sample_fps,
                  undistort, reconnect_s, block_on_full, rolling),
            daemon=True,
        )
        t.start()
        threads.append(t)
        time.sleep(0.1)  # 错峰打开摄像头

    # 实时源振动报警切窗：由本进程内线程消费 clip_q，从滚动缓冲拼出片段
    if buffers and clip_q is not None:
        threading.Thread(
            target=_clip_cut_loop, args=(clip_q, buffers, stop_event),
            daemon=True).start()

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
