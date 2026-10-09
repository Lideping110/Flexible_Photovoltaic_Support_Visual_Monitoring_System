"""靶标跟踪分支推理进程（单实例，检测模型仅加载一次）。

流程（对齐 docs/architecture/03_branch_pipelines.svg）：
  滑动攒批（满 max_frames 帧或 timeout 超时，跨摄像头混批）
  → 批量 predict() 前向（模型一份）
  → 按 cam_id 拆分，各自 BYTETracker 关联
  → 持久 ID 持续跟踪（ByteTrack raw id 绑定，box 仅存档不匹配）
  → λmin 锚点窗口选峰 + cornerSubPix 亚像素（复用本包 pipeline/features）
  → mm/px 标定（外圈圆直径；可从 init 恢复）→ 累计位移
  → displacement 记录发 Recorder；位移时序样本发 SpectrumAnalyzer

初始基准持久化：首帧保存 init_frame.jpg + init_state.json（靶标初始位置
+ mm/px 标定值）；重启自动加载，mm/px 立即生效、位移相对 init 参考点延续。
"""
import time
from pathlib import Path

import numpy as np

from utils.common import (SENTINEL, get_any, log, now_ts, put_blocking,
                          put_latest, select_model_weights, to_float)
from utils.inference import DetInference, make_byte_tracker, tracker_update
from utils.persist import (atomic_write_json, build_init_state, init_paths,
                           load_json, save_anchor_frame)

from .identity import MarkerTrackRegistry


class MarkerCameraSession:
    """单个 marker 摄像头的推理上下文。

    封装 ByteTrack、持续跟踪身份、多目标跟踪与 mm/px 标定状态，替代原先
    run_marker_inference 内 states[dict] + 闭包的组织方式。
    """

    def __init__(self, cam, model_cfg, anchor_cfg, runtime,
                 refresh_init, result_q, tag):
        from .tracker import MultiTargetTracker

        self.cam = cam
        self.cam_id = cam["id"]
        self.tag = tag
        self.result_q = result_q
        self.refresh_init = refresh_init
        self.persist_flag = bool(runtime["init"]["persist"])

        init_dir, self.state_path, self.frame_path = init_paths(
            runtime["output_dir"], "marker", self.cam_id)
        self.tracker = make_byte_tracker(
            model_cfg.get("tracker", "bytetrack.yaml"))
        # 靶标身份靠 ByteTrack 持续跟踪维持（box 仅存档，不做框匹配）
        self.registry = MarkerTrackRegistry(
            self.cam_id, prefix="M",
            release_after_s=anchor_cfg["release_after_s"])
        self.multi = MultiTargetTracker(max_age=30)
        # pid -> {"tried", "diams", "mm_per_px", "ready", "ref"}
        self.calib = {}
        self.init_done = False
        self.frame_saved = False

    def handle_init(self, dets, ts):
        """首帧初始化：加载已保存初始状态，或从首帧检测建立锚点。"""
        if self.init_done:
            return
        saved = load_json(self.state_path) if self.persist_flag else None
        if saved and not self.refresh_init and saved.get("cam_id") == self.cam_id:
            self.registry.load_targets(saved.get("targets", []), ts)
            for t in saved.get("targets", []):
                bl = t.get("baseline") or {}
                pid = t["pid"]
                self.calib[pid] = {
                    "tried": 0, "diams": [],
                    "mm_per_px": bl.get("mm_per_px"),
                    "ready": bl.get("mm_per_px") is not None,
                    "ref": (np.asarray(bl["center"], dtype=np.float64)
                            if bl.get("center") else None),
                }
            put_blocking(self.result_q, {
                "branch": "marker", "kind": "event", "cam_id": self.cam_id,
                "event": "init_loaded", "ts": now_ts(),
                "detail": f"锚点 {len(saved.get('targets', []))} 个",
            })
            log(self.tag, f"{self.cam_id} 加载持久化初始状态")
        elif dets:
            self.registry.ensure_init(dets, ts)
            put_blocking(self.result_q, {
                "branch": "marker", "kind": "event", "cam_id": self.cam_id,
                "event": "init_created", "ts": now_ts(),
                "detail": f"首帧锁定靶标 {len(self.registry.targets)} 个",
            })
        self.init_done = True

    def persist_init(self, frame):
        """首帧图片 + 初始状态（含 mm/px 标定）原子落盘（仅首次运行）。"""
        if not self.registry.initialized or self.registry.loaded_from_disk:
            return
        if not self.frame_saved and frame is not None:
            save_anchor_frame(frame, self.registry.anchors, self.frame_path)
            self.frame_saved = True
        targets = []
        for t in self.registry.to_targets():
            bl = None
            cal = self.calib.get(t["pid"])
            if cal and cal.get("ready"):
                bl = {"center": ([float(v) for v in cal["ref"]]
                                if cal["ref"] is not None else None),
                      "mm_per_px": cal.get("mm_per_px")}
            targets.append({**t, "baseline": bl})
        atomic_write_json(self.state_path, build_init_state(
            self.cam_id, "marker", targets))


def run_marker_inference(cfg: dict, cameras: list, frame_queues: dict, result_q,
                         spectrum_q, stop_event, refresh_init: bool = False):
    """靶标分支推理进程入口。"""
    from .features import measure_outer_diameter_px
    from .pipeline import localize_one

    tag = "marker-inf"
    model_cfg = cfg["models"]["marker_det"]
    weights = select_model_weights(model_cfg, Path(cfg["_config_dir"]))
    log(tag, f"模型: {weights}")

    marker_cfg = cfg.get("marker", {})
    loc_cfg = marker_cfg.get("localization", {})
    calib_cfg = marker_cfg.get("calibration", {})
    runtime = cfg["runtime"]
    anchor_cfg = runtime["init"]["anchor"]
    max_frames = int(runtime["batch"]["marker"]["max_frames"])
    batch_timeout = float(runtime["batch"]["marker"]["timeout_ms"]) / 1000.0

    real_diameter_mm = float(calib_cfg.get("real_diameter_mm", 100.0))
    calib_frames = int(calib_cfg.get("frames", 8))
    calib_min = int(calib_cfg.get("min_frames", 1))

    engine = DetInference(
        weights, model_cfg.get("confidence", 0.25),
        model_cfg.get("class_id", 0), model_cfg.get("device", "intel:gpu"),
        model_cfg.get("tracker", "bytetrack.yaml"), tag,
    )

    sessions = {
        cam["id"]: MarkerCameraSession(
            cam, model_cfg, anchor_cfg, runtime, refresh_init, result_q, tag)
        for cam in cameras
    }

    online = {c["id"] for c in cameras}
    queues = [(c["id"], frame_queues[c["id"]]) for c in cameras]
    log(tag, f"启动，摄像头: {sorted(online)}")

    try:
        while online and not stop_event.is_set():
            # ---- 滑动攒批：满 N 帧或超时 ----
            batch = {}  # cam_id -> list of frame msgs
            batch_t0 = None
            total = 0
            while total < max_frames:
                timeout = 1.0 if total == 0 else max(
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
                    batch.setdefault(cam_id, []).append(item)
                    total += 1
                    if batch_t0 is None:
                        batch_t0 = time.perf_counter()
            if total == 0:
                continue

            for cam_id, msgs in batch.items():
                session = sessions[cam_id]
                frames = [m["frame"] for m in msgs]
                try:
                    results = engine.predict(frames)
                except Exception as exc:  # noqa: BLE001
                    log(tag, f"{cam_id} 前向失败，丢弃本批: {exc}")
                    continue
                if not isinstance(results, list):
                    results = [results]
                for msg, frame, result in zip(msgs, frames, results):
                    dets = tracker_update(session.tracker, result, frame)
                    session.handle_init(dets, msg["ts"])
                    mapping = session.registry.assign(dets, msg["ts"])
                    session.multi.prune(msg["frame_id"])

                    for det in dets:
                        pid = mapping.get(det["id"])
                        if pid is None:
                            continue
                        prev = session.multi.previous(pid)
                        prev_center = prev[0] if prev else None
                        prev_box = prev[1] if prev else None
                        center, lm, continuous = localize_one(
                            frame, det["box"], loc_cfg, prev_center, prev_box)

                        # ---- mm/px 标定（首帧自标定，或从 init 恢复）----
                        cal = session.calib.setdefault(pid, {
                            "tried": 0, "diams": [], "mm_per_px": None,
                            "ready": False, "ref": None})
                        if (not cal["ready"] and center is not None
                                and cal["tried"] < calib_frames):
                            cal["tried"] += 1
                            res = measure_outer_diameter_px(frame, det["box"])
                            if res is not None:
                                cal["diams"].append(float(res[0]))
                            if (len(cal["diams"]) >= calib_min
                                    or (cal["tried"] >= calib_frames and cal["diams"])):
                                cal["mm_per_px"] = real_diameter_mm / float(
                                    np.mean(cal["diams"]))
                                cal["ready"] = True
                                log(tag, f"{cam_id} {pid} mm/px="
                                         f"{cal['mm_per_px']:.6f}")

                        dx = dy = None
                        if (center is not None
                                and session.multi.delta_ok(pid, continuous, msg["frame_id"])):
                            dx, dy = map(float, center - prev_center)

                        cum_dx = cum_dy = None
                        if center is not None and cal["ready"] and cal["mm_per_px"]:
                            if cal["ref"] is None:
                                cal["ref"] = center.copy()
                            cum_dx = float((center[0] - cal["ref"][0]) * cal["mm_per_px"])
                            cum_dy = float((center[1] - cal["ref"][1]) * cal["mm_per_px"])

                        if center is not None:
                            session.multi.record(pid, center, np.asarray(
                                det["box"], dtype=np.float32), msg["frame_id"])

                        record = {
                            "frame_id": int(msg["frame_id"]),
                            "ts": float(msg["ts"]),
                            "pid": pid, "raw_track_id": int(det["id"]),
                            "conf": float(det["conf"]),
                            "cx": to_float(center[0]) if center is not None else None,
                            "cy": to_float(center[1]) if center is not None else None,
                            "dx_px": to_float(dx), "dy_px": to_float(dy),
                            "continuous": bool(continuous),
                            "lambda_min": to_float(lm),
                            "cum_dx_mm": to_float(cum_dx), "cum_dy_mm": to_float(cum_dy),
                            "mm_per_px": to_float(cal["mm_per_px"]),
                            "box": [float(v) for v in det["box"]],
                        }
                        put_blocking(result_q, {
                            "branch": "marker", "kind": "displacement",
                            "cam_id": cam_id, "ts": float(msg["ts"]), "data": record,
                        })
                        if cum_dx is not None:
                            put_latest(spectrum_q, {
                                "cam_id": cam_id, "pid": pid,
                                "t": float(msg["ts"]),
                                "dx_mm": cum_dx, "dy_mm": cum_dy,
                            })

                    # 首次运行：标定就绪后持久化初始数据
                    if (session.registry.initialized
                            and not session.registry.loaded_from_disk):
                        session.persist_init(frame)
    finally:
        for session in sessions.values():
            try:
                session.persist_init(None)
            except Exception:
                pass
        if spectrum_q is not None:
            try:
                spectrum_q.put(SENTINEL, timeout=2.0)
            except Exception:
                pass
        log(tag, "退出")
