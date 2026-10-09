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
from utils.common import (SENTINEL, get_any, log, now_ts, put_blocking,
                          select_model_weights, to_float, to_float_list)
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
        self.frame_saved = False

    def handle_init(self, dets, ts):
        """首帧初始化：加载已保存初始状态，或从首帧检测建立锚点。"""
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
        elif dets:
            self.anchors.ensure_init(dets, ts)
            put_blocking(self.result_q, {
                "branch": "pole", "kind": "event", "cam_id": self.cam_id,
                "event": "init_created", "ts": now_ts(),
                "detail": f"首帧锚点 {len(self.anchors.anchors)} 个",
            })
            log(self.tag, f"{self.cam_id} 首帧建立锚点 {len(self.anchors.anchors)} 个")
        self.init_done = True

    def persist_init(self, frame):
        """首帧图片 + 初始状态原子落盘（仅首次运行）。"""
        if not self.anchors.initialized or self.anchors.loaded_from_disk:
            return
        if not self.frame_saved and frame is not None:
            save_anchor_frame(frame, self.anchors.anchors, self.frame_path)
            self.frame_saved = True
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
                       stop_event, refresh_init: bool = False):
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

    sessions = {
        cam["id"]: PoleCameraSession(
            cam, model_cfg, pole_cfg, anchor_cfg, runtime,
            refresh_init, result_q, tag)
        for cam in cameras
    }

    online = {c["id"] for c in cameras}
    queues = [(c["id"], frame_queues[c["id"]]) for c in cameras]
    log(tag, f"启动，摄像头: {sorted(online)}")

    try:
        while online and not stop_event.is_set():
            # ---- 攒批：等齐所有在线摄像头或超时 ----
            pending = {}
            batch_t0 = None
            while len(pending) < len(online):
                timeout = 1.0 if not pending else max(
                    0.01, batch_t0 + batch_timeout - time.perf_counter())
                got = get_any(queues, timeout)
                if got is None:
                    break
                cam_id, item = got
                if item is SENTINEL:
                    online.discard(cam_id)
                    queues = [(cid, q) for cid, q in queues if cid != cam_id]
                    log(tag, f"{cam_id} 下线（剩余 {sorted(online)}）")
                    continue
                if cam_id in online:
                    pending[cam_id] = item
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
    finally:
        # 首帧图片兜底保存（首帧无检测时 anchors 未初始化的情况）
        for session in sessions.values():
            try:
                session.persist_init(None)
            except Exception:
                pass
        log(tag, "退出")
