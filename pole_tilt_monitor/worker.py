"""立柱监测分支推理进程（单实例，分割模型仅加载一次）。

流程（对齐 docs/architecture/03_branch_pipelines.svg）：
  整秒攒批（等齐所有在线 pole 摄像头或超时）
  → 批量 predict() 前向（模型一份）
  → 按 cam_id 拆分，各自 BYTETracker 关联（跨摄像头混批不串轨）
  → 持久 ID 锚点匹配（遮挡换轨纠正，见 utils/anchor.py）
  → 复用本包 monitor 的倾角测量（RANSAC/质量门控/EMA/基准）
  → 结构化 measurement 记录发往 Recorder

初始基准持久化：首帧保存 init_frame.jpg + init_state.json（原子写）；
重启检测到已保存则直接加载锚点与基准，delta 延续重启前。
"""
import time
from pathlib import Path

import numpy as np

from utils.anchor import AnchorRegistry
from utils.common import (SENTINEL, get_any, is_realtime_source, log, now_ts,
                          put_blocking, select_model_weights, to_float,
                          to_float_list)
from utils.inference import SegInference, make_byte_tracker, tracker_update
from utils.persist import (atomic_write_json, build_init_state, init_paths,
                           load_json, save_anchor_frame, write_calibration_yaml)


def _build_pole_monitor(calib_path, geom, bc, fc, relative_mode):
    """按测量模式构造立柱倾角测量器（复用本包 monitor）。"""
    from .monitor import ImageRelativePoleMonitor, PoleMonitor

    common = dict(
        endpoint_ratio=geom.get("endpoint_ratio", 0.10),
        ransac_residual_px=geom.get("ransac_residual_px", 2.5),
        ransac_iterations=geom.get("ransac_iterations", 300),
        ema_alpha=fc.get("alpha", 0.25),
        quality_threshold=geom.get("quality_threshold", 0.60),
        quality_weights=geom.get("quality_weights"),
        max_centerline_residual_px=geom.get("max_centerline_residual_px", 4.0),
        max_endpoint_spread_px=geom.get("max_endpoint_spread_px", 4.0),
        body_trim_ratio=geom.get("body_trim_ratio", 0.10),
    )
    if relative_mode:
        return ImageRelativePoleMonitor(
            str(calib_path),
            baseline_frames=bc.get("relative_frames", 1),
            **common)
    return PoleMonitor(
        str(calib_path),
        geom.get("pole_length_m", 3.0),
        geom.get("ground_z_m", 0.0),
        baseline_frames=bc.get("frames", 60),
        min_ray_plane_angle_deg=geom.get("min_ray_plane_angle_deg", 5.0),
        max_length_error_ratio=geom.get("max_length_error_ratio", 0.03),
        **common)


class PoleCameraSession:
    """单个 pole 摄像头的推理上下文。

    封装 ByteTrack、锚点匹配、倾角测量与初始基准持久化，替代原先
    run_pole_inference 内 states[dict] + 闭包的组织方式。
    """

    def __init__(self, cam, model_cfg, pole_cfg, anchor_cfg, runtime,
                 refresh_init, result_q, tag):
        geom = pole_cfg.get("measurement", {})
        bc = pole_cfg.get("baseline", {})
        fc = pole_cfg.get("filter", {})
        self.relative_mode = geom.get("mode", "image_relative") == "image_relative"

        self.cam = cam
        self.cam_id = cam["id"]
        self.tag = tag
        self.result_q = result_q
        self.refresh_init = refresh_init
        self.persist_flag = bool(runtime["init"]["persist"])

        init_dir, self.state_path, self.frame_path = init_paths(
            runtime["output_dir"], "pole", self.cam_id)
        calib_path = write_calibration_yaml(
            init_dir.parent / "calibration.yaml", cam.get("calibration", {}))
        self.monitor = _build_pole_monitor(
            calib_path, geom, bc, fc, self.relative_mode)
        self.tracker = make_byte_tracker(
            model_cfg.get("tracker", "bytetrack.yaml"))
        self.anchors = AnchorRegistry(
            self.cam_id, "pole", iou_min=anchor_cfg["iou_min"],
            release_after_s=anchor_cfg["release_after_s"],
            ema_alpha=anchor_cfg["ema_alpha"], prefix="P")
        self.init_done = False
        self.last_frame = None  # 最近一帧，供 finalize_init 重绘首帧图
        self.init_bg_frame = None  # 初始帧背景：首帧采一次，之后只在其上叠加已就绪基线的拟合线
        self._drawn_key = None  # 上次重绘时的"已就绪基线集合"，变化才重绘（幂等防抖）
        self._init_wait_t0 = None  # 首次成功初始化前的等待计时（未建立则不锁死）
        # 立柱报警留存节流：同一 (cam_id, pid) 最短冷却间隔才再写盘，避免每秒一张刷屏
        self._alarm_cooldowns = {}

    def handle_init(self, dets, ts):
        """初始化：加载已保存初始状态，或从带检测的帧建立锚点。

        **只有锚点真正建立（或从盘加载成功）后才置 init_done。** 曾无条件
        init_done=True：首帧恰好没检测到目标（RTSP 刚连上曝光未稳、或镜头里
        压根没有立柱）→ ensure_init 永不重试 → 该路整轮零数据且**静默无报错**
        （init/ 空、无 measurements.jsonl、大屏只显示空表）。现改为未建立就
        持续重试，并每 10s 告警一次，杜绝"无声死亡"。
        """
        if self.init_done:
            return
        saved = load_json(self.state_path) if self.persist_flag else None
        if saved and not self.refresh_init and saved.get("cam_id") == self.cam_id:
            self.anchors.load_anchors(saved.get("targets", []), ts)
            injected = 0
            for t in saved.get("targets", []):
                bl = t.get("baseline")
                if bl and bl.get("values"):
                    self.monitor.baseline[t["pid"]] = np.asarray(
                        bl["values"], dtype=np.float64)
                    injected += 1
            put_blocking(self.result_q, {
                "branch": "pole", "kind": "event", "cam_id": self.cam_id,
                "event": "init_loaded", "ts": now_ts(),
                "detail": f"锚点 {len(saved.get('targets', []))} 个，注入基准 {injected} 条",
            })
            log(self.tag, f"{self.cam_id} 加载持久化初始状态 "
                          f"({len(saved.get('targets', []))} 锚点)")
            self.init_done = True
            return
        if dets:
            self.anchors.ensure_init(dets, ts)
            put_blocking(self.result_q, {
                "branch": "pole", "kind": "event", "cam_id": self.cam_id,
                "event": "init_created", "ts": now_ts(),
                "detail": f"首帧锚点 {len(self.anchors.anchors)} 个",
            })
            log(self.tag, f"{self.cam_id} 首帧建立锚点 {len(self.anchors.anchors)} 个")
            self.init_done = True
            return
        # 本帧无检测：不锁死，下一帧继续尝试；定期告警避免静默
        if self._init_wait_t0 is None:
            self._init_wait_t0 = time.perf_counter()
        waited = time.perf_counter() - self._init_wait_t0
        if waited >= 10.0:
            log(self.tag, f"{self.cam_id} 尚未检测到目标（已等待 {waited:.0f}s），"
                          f"持续重试初始化——请确认该相机画面中确有立柱、"
                          f"且标定(K/dist)属于该相机")
            self._init_wait_t0 = time.perf_counter()

    def persist_init(self, frame):
        """首帧图片 + 初始状态原子落盘（仅首次运行）。

        背景固定为首帧（init_bg_frame，只在第一次拿到帧时采集一次），但**每根
        柱子基线就绪时都重绘 init_frame.jpg**——把当前所有"已就绪基线"的拟合线
        画到同一张首帧背景上。这样初始状态画面随基线逐步补齐而逐步出现拟合线，
        不必等到进程退出 finalize_init 才一次性补全（修复"开始时只有第一根、
        视频跑完后后面才拟合上"的观感，且 RTSP 实时流无需退出即可自愈）。
        """
        if not self.anchors.initialized or self.anchors.loaded_from_disk:
            return
        if frame is not None:
            self.last_frame = frame
            if self.init_bg_frame is None:
                self.init_bg_frame = frame.copy()  # 背景只采一次，保持"首帧"语义
        bg = self.init_bg_frame if self.init_bg_frame is not None else self.last_frame
        # 仅当"已就绪基线集合"变化时才重绘帧（幂等防抖）：即使将来调用方改为
        # 逐帧调用，也不会退化成逐帧 JPEG 编码+写盘；新增基线必触发重绘。
        key = tuple(pid for pid in sorted(self.anchors.anchors)
                    if self.anchors.get_baseline(pid) is not None)
        if bg is not None and key != self._drawn_key:
            # 保留首帧背景，叠加当前所有已就绪拟合线
            save_anchor_frame(bg, self.anchors.anchors, self.frame_path)
            self._drawn_key = key
        atomic_write_json(self.state_path, build_init_state(
            self.cam_id, "pole",
            [{**t, "baseline": self._baseline_payload(t["baseline"])}
             for t in self.anchors.to_targets()]))

    def finalize_init(self):
        """退出时（或运行结束时）用最终锚点集重绘 init_frame.jpg 并重写 state。

        与 persist_init 共用同一首帧背景（init_bg_frame）；若运行期内已随基线
        逐步重绘过，这里只是做最后一次一致性收尾，保证两者（帧与 json）始终
        来自同一次运行、内容一致。
        """
        if not self.anchors.initialized or self.anchors.loaded_from_disk:
            return
        bg = self.init_bg_frame if self.init_bg_frame is not None else self.last_frame
        if bg is not None:
            save_anchor_frame(bg, self.anchors.anchors, self.frame_path)
        atomic_write_json(self.state_path, build_init_state(
            self.cam_id, "pole",
            [{**t, "baseline": self._baseline_payload(t["baseline"])}
             for t in self.anchors.to_targets()]))

    def maybe_persist_baseline(self, pid, quality, frame):
        """首次运行：该 pid 基准就绪且尚未写入锚点时，持久化锚点初始数据。"""
        if (self.anchors.initialized
                and not self.anchors.loaded_from_disk
                and pid in self.monitor.baseline
                and self.anchors.get_baseline(pid) is None):
            self.anchors.set_baseline(
                pid, [float(v) for v in self.monitor.baseline[pid]],
                quality=quality)
            self.persist_init(frame)

    def _baseline_payload(self, baseline):
        if baseline is None:
            return None
        if isinstance(baseline, dict):
            return baseline
        return {"mode": "image_relative" if self.relative_mode else "world_3d",
                "values": [float(v) for v in baseline]}


def run_pole_inference(cfg: dict, cameras: list, frame_queues: dict, result_q,
                       stop_event, refresh_init: bool = False,
                       ready_event=None):
    """立柱分支推理进程入口。"""
    tag = "pole-inf"
    model_cfg = cfg["models"]["pole_seg"]
    weights = select_model_weights(model_cfg, Path(cfg["_config_dir"]))
    log(tag, f"模型: {weights}")

    pole_cfg = cfg.get("pole", {})
    runtime = cfg["runtime"]
    anchor_cfg = runtime["init"]["anchor"]
    alarm_cfg = pole_cfg.get("alarm", {})
    batch_timeout = float(runtime["batch"]["pole"]["timeout_ms"]) / 1000.0

    engine = SegInference(
        weights, model_cfg.get("confidence", 0.25),
        model_cfg.get("class_id", 0), model_cfg.get("device", "cpu"),
        model_cfg.get("tracker", "bytetrack.yaml"), tag,
    )

    # 预热模型：把首帧冷启动（OpenVINO 编译/JIT）移出采集关键路径，
    # 完成后置位就绪事件，capture 才会开始抽帧，避免早期帧被有界队列冲掉。
    engine.warmup()
    if ready_event is not None:
        ready_event.set()
        log(tag, "预热完成，已通知 capture 开始抽帧")

    sessions = {
        cam["id"]: PoleCameraSession(
            cam, model_cfg, pole_cfg, anchor_cfg, runtime,
            refresh_init, result_q, tag)
        for cam in cameras
    }

    online = {c["id"] for c in cameras}
    queues = [(c["id"], frame_queues[c["id"]]) for c in cameras]
    # 已成功出过至少一帧的相机。攒批只等"已建立连接的相机"齐批；
    # 从未连上的相机（如不可达 RTSP）不阻塞其他相机，避免其把整条队列
    # 抽干（pending[cam] 覆盖写）导致前段帧被吞（实测：加不可达的
    # pole_cam_02 后 pole_cam_01 前 ~15s 检测全丢）。相机一旦出帧即入 seen，
    # 后续批次正常参与同步；若中途下线（SENTINEL）则从 seen 移除。
    seen = set()
    log(tag, f"启动，摄像头: {sorted(online)}")

    # 文件源相机处理完成事件（收到该路 SENTINEL 且批次算完才发），供 supervisor
    # 即时导出标注视频；流源永不写。
    # 走 result_q 事件而非推理端直接写文件：标注视频读的是 **Recorder 落盘的
    # jsonl**，"推理算完"≠"Recorder 写完"。camera_done 排在该路所有记录之后
    # （result_q FIFO），由 Recorder 收到时亲自写 .rec_done —— 完成标记由
    # 最后一环发出，才是确定性事件，而非靠"jsonl 不增长"猜（无目标时段不写行会误判）。
    done_pending = set()

    def _flush_done():
        while done_pending:
            cid = done_pending.pop()
            try:
                put_blocking(result_q, {
                    "branch": "pole", "kind": "camera_done",
                    "cam_id": cid, "ts": now_ts()})
                log(tag, f"{cid} 推理完成，已发 camera_done 事件")
            except Exception as exc:
                log(tag, f"{cid} 完成事件发送失败: {exc}")

    try:
        while online and not stop_event.is_set():
            # ---- 攒批：等齐"已出过帧"的相机或超时 ----
            _flush_done()  # 上一轮批次已处理完的相机，发完成事件
            pending = {}
            batch_t0 = None
            while True:
                # 只等已经出过帧的相机齐批；从未连上的相机不阻塞本批。
                # 这样不可达 RTSP（pole_cam_02）不会把 pole_cam_01 的整条队列
                # 抽干、吞掉前段帧。seen 为空时（尚未有任何相机连上）继续等待。
                need = [cid for cid in online if cid in seen]
                if seen and len(pending) >= len(need):
                    break
                timeout = 1.0 if not pending else max(
                    0.01, batch_t0 + batch_timeout - time.perf_counter())
                got = get_any(queues, timeout)
                if got is None:
                    break
                cam_id, item = got
                if item is SENTINEL:
                    online.discard(cam_id)
                    seen.discard(cam_id)
                    queues = [(cid, q) for cid, q in queues if cid != cam_id]
                    log(tag, f"{cam_id} 下线（剩余 {sorted(online)}）")
                    done_pending.add(cam_id)
                    continue
                if cam_id in online:
                    pending[cam_id] = item
                    seen.add(cam_id)
                    if batch_t0 is None:
                        batch_t0 = time.perf_counter()
            if not pending:
                continue

            # ---- 批量前向 → 按 cam_id 拆分 ----
            cam_ids = [c["id"] for c in cameras if c["id"] in pending]
            frames = [pending[cid]["frame"] for cid in cam_ids]
            try:
                results = engine.predict(frames)
            except Exception as exc:  # noqa: BLE001
                log(tag, f"前向失败，丢弃本批: {exc}")
                continue
            if not isinstance(results, list):
                results = [results]

            for cid, frame, result in zip(cam_ids, frames, results):
                session = sessions[cid]
                msg = pending[cid]
                dets = tracker_update(session.tracker, result, frame,
                                      with_mask=True)

                session.handle_init(dets, msg["ts"])
                mapping = session.anchors.assign(dets, msg["ts"])
                frame_pids = []

                for det in dets:
                    pid = mapping.get(det["id"])
                    if pid is None:
                        continue  # 未匹配到锚点的轨迹（新出现目标无空置锚点）
                    m = session.monitor.process(pid, det["mask"], msg["frame_id"])
                    if m is None:
                        continue

                    quality = (float(m.get("quality", float("nan")))
                               if m.get("quality") is not None else None)
                    session.maybe_persist_baseline(pid, quality, frame)

                    delta_lr = to_float(m.get("delta_lr"))
                    delta_fb = to_float(m.get("delta_fb"))
                    delta_total = to_float(m.get("delta_total"))
                    alarm = bool(
                        (delta_lr is not None and abs(delta_lr) >= alarm_cfg.get("left_right_deg", 2.0))
                        or (delta_fb is not None and abs(delta_fb) >= alarm_cfg.get("front_back_deg", 2.0))
                        or (delta_total is not None and abs(delta_total) >= alarm_cfg.get("total_deg", 2.5))
                    )
                    record = {
                        "frame_id": int(msg["frame_id"]), "ts": float(msg["ts"]),
                        "pid": pid, "raw_track_id": int(det["id"]),
                        "status": m.get("status", "INVALID"),
                        "quality": to_float(m.get("quality")),
                        "lr_deg": to_float(m.get("lr")), "fb_deg": to_float(m.get("fb")),
                        "total_deg": to_float(m.get("total")),
                        "delta_lr_deg": delta_lr, "delta_fb_deg": delta_fb,
                        "delta_total_deg": delta_total,
                        "delta_dx_m": to_float(m.get("delta_dx")),
                        "delta_dy_m": to_float(m.get("delta_dy")),
                        "delta_dh_m": to_float(m.get("delta_dh")),
                        "box": [float(v) for v in det["box"]],
                        "top": to_float_list(m.get("top")),
                        "bottom": to_float_list(m.get("bottom")),
                        "alarm": alarm,
                    }
                    put_blocking(result_q, {
                        "branch": "pole", "kind": "measurement",
                        "cam_id": cid, "ts": float(msg["ts"]), "data": record,
                    })
                    # 立柱报警留存：原始帧 + 重新编码帧（画框+拟合直线）成对落盘，
                    # 供大屏报警记录页回溯。受 session 级 cooldown 节流，不会刷屏。
                    if alarm:
                        try:
                            from utils.alarm_store import save_pole_alarm_snapshot
                            # 命名区分实时/文件源：实时→当前时间，文件→frame_id
                            # （is_realtime_source 由相机 url 判定，保证与 capture 口径一致）
                            save_pole_alarm_snapshot(
                                runtime["output_dir"], cid, msg["ts"], frame,
                                record, alarm_cfg, session._alarm_cooldowns,
                                cooldown_s=float(alarm_cfg.get("cooldown_s", 5.0)),
                                is_realtime=is_realtime_source(session.cam["url"]),
                                frame_id=msg["frame_id"])
                        except Exception as exc:
                            log(tag, f"立柱报警留存失败 {cid}/{pid}: {exc}")
                    frame_pids.append(pid)

                # 即使本帧没有任何有效测量，也发布当前帧的目标集合。
                # Recorder 据此删除已经离开画面的旧目标，避免 latest.json 永久
                # 保留最后一次测量。
                put_blocking(result_q, {
                    "branch": "pole", "kind": "frame_snapshot",
                    "cam_id": cid, "ts": float(msg["ts"]),
                    "data": {"frame_id": int(msg["frame_id"]),
                             "pids": sorted(set(frame_pids))},
                })
    finally:
        _flush_done()  # 兜底：发出剩余完成事件
        # 退出前用最终锚点集重绘 init_frame.jpg，确保与 init_state.json 一致
        for session in sessions.values():
            try:
                session.finalize_init()
            except Exception:
                pass
        log(tag, "退出")
