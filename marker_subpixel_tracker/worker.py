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
            release_after_s=anchor_cfg["release_after_s"],
            relock_max_px=anchor_cfg.get("relock_max_px", 200.0))
        self.multi = MultiTargetTracker(max_age=30)
        # pid -> {"tried", "diams", "mm_per_px", "ready", "ref"}
        self.calib = {}
        self.init_done = False
        self.frame_saved = False
        self.last_frame = None  # 最近一帧，供 finalize_init 重绘首帧图
        self._init_wait_t0 = None  # 首次成功初始化前的等待计时（未建立则不锁死）

    def handle_init(self, dets, ts):
        """初始化：加载已保存初始状态，或从带检测的帧建立锚点。

        **只有锚点真正建立（或从盘加载成功）后才置 init_done。** 曾无条件
        init_done=True：首帧恰好没检测到目标（RTSP 刚连上曝光未稳、或镜头里
        压根没有靶标）→ ensure_init 永不重试 → 该路整轮零数据且**静默无报错**。
        现改为未建立就持续重试，并每 10s 告警一次，杜绝"无声死亡"。
        """
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
            self.init_done = True
            return
        if dets:
            self.registry.ensure_init(dets, ts)
            put_blocking(self.result_q, {
                "branch": "marker", "kind": "event", "cam_id": self.cam_id,
                "event": "init_created", "ts": now_ts(),
                "detail": f"首帧锁定靶标 {len(self.registry.targets)} 个",
            })
            self.init_done = True
            return
        # 本帧无检测：不锁死，下一帧继续尝试；定期告警避免静默
        if self._init_wait_t0 is None:
            self._init_wait_t0 = time.perf_counter()
        waited = time.perf_counter() - self._init_wait_t0
        if waited >= 10.0:
            log(self.tag, f"{self.cam_id} 尚未检测到靶标（已等待 {waited:.0f}s），"
                          f"持续重试初始化——请确认该相机画面中确有靶标")
            self._init_wait_t0 = time.perf_counter()

    def _save_anchors(self):
        """构造带亚像素中心点的锚点 dict，供 save_anchor_frame 画红色十字跟踪点。

        中心取自标定参考点 calib[pid]["ref"]（即各靶标初始亚像素跟踪点），
        与 displacement.jsonl 的 cx/cy 同源；标定未就绪时退化为仅画框（无十字）。
        """
        out = {}
        for pid, a in self.registry.anchors.items():
            item = dict(a)  # 原 {"box": ...}
            cal = self.calib.get(pid)
            if cal and cal.get("ready") and cal.get("ref") is not None:
                item["center"] = [float(v) for v in cal["ref"]]
            out[pid] = item
        return out

    def persist_init(self, frame):
        """首帧图片 + 初始状态（含 mm/px 标定）原子落盘（仅首次运行）。
        早期快照：首个标定就绪时先存一版帧；最终一致性由 finalize_init 保证。
        """
        if not self.registry.initialized or self.registry.loaded_from_disk:
            return
        if frame is not None:
            self.last_frame = frame
        if not self.frame_saved and self.last_frame is not None:
            save_anchor_frame(self.last_frame, self._save_anchors(), self.frame_path)
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

    def finalize_init(self):
        """退出时用最终锚点集重绘 init_frame.jpg 并重写 state，确保两者一致。
        修复 finally 中 persist_init(None) 因 frame=None 跳过帧重绘、却仍写 state
        导致的「帧与状态不一致」（如帧停留旧运行的 1 个靶标而 state 已是 2 个）。
        """
        if not self.registry.initialized or self.registry.loaded_from_disk:
            return
        if self.last_frame is not None:
            save_anchor_frame(self.last_frame, self._save_anchors(), self.frame_path)
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
                         spectrum_q, stop_event, refresh_init: bool = False,
                         ready_event=None):
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

    # 预热模型：把首帧冷启动移出采集关键路径，完成后置位就绪事件，
    # 通知 capture 开始抽帧（避免早期帧被有界队列冲掉）。
    engine.warmup()
    if ready_event is not None:
        ready_event.set()
        log(tag, "预热完成，已通知 capture 开始抽帧")

    sessions = {
        cam["id"]: MarkerCameraSession(
            cam, model_cfg, anchor_cfg, runtime, refresh_init, result_q, tag)
        for cam in cameras
    }

    online = {c["id"] for c in cameras}
    queues = [(c["id"], frame_queues[c["id"]]) for c in cameras]
    log(tag, f"启动，摄像头: {sorted(online)}")

    # 文件源相机处理完成事件（收到该路 SENTINEL 且批次算完才发），供 supervisor
    # 即时导出标注视频，无需等 main 整体退出；流源永不收到 SENTINEL，故永不发。
    #
    # 为何走 result_q 事件而非推理端直接写文件：标注视频读的是 **Recorder 落盘的
    # jsonl**，"推理算完"≠"Recorder 写完"。把 camera_done 排在该路所有记录之后
    # （result_q 是 FIFO），由 Recorder 收到时亲自写 .rec_done —— 完成标记由
    # 最后一环发出，才是确定性事件。绝不靠"jsonl 不增长了"去猜：无目标时段不写
    # 行，jsonl 会长时间不增长，那个判据会误判成"已完成"。
    done_pending = set()

    def _flush_done():
        while done_pending:
            cid = done_pending.pop()
            try:
                put_blocking(result_q, {
                    "branch": "marker", "kind": "camera_done",
                    "cam_id": cid, "ts": now_ts()})
                log(tag, f"{cid} 推理完成，已发 camera_done 事件")
            except Exception as exc:
                log(tag, f"{cid} 完成事件发送失败: {exc}")

    # ---- 实时叠加（仅本地 GUI 调试用）----
    # 复用 viz.render._draw_marker，保证与离线标注视频视觉一致（黄框/红十字/
    # M1/M2+raw id+d=()mm+mm/px 标签）。GUI 窗口仅在 runtime.display=true 时启用；
    # 默认 headless（display=false），生产/服务器环境不受影响。无显示器环境用
    # 一次性 probe 检测，失败则优雅降级关闭窗口，绝不崩溃。
    # 注意：main 进程不向磁盘/Web 发布任何实时帧——Web 大屏的实时画面由
    # dashboard 直接拉 RTSP 子码流显示（见 dashboard.py），两进程解耦。
    _disp_cv2 = None
    try:
        from viz.render import _draw_marker
    except Exception:
        draw_marker = None
    else:
        draw_marker = _draw_marker
    if bool(runtime.get("display", False)):
        import cv2 as _disp_cv2
        try:
            _disp_cv2.namedWindow("__probe__", _disp_cv2.WINDOW_AUTOSIZE)
            _disp_cv2.destroyWindow("__probe__")
            log(tag, "实时叠加窗口已启用")
        except Exception as exc:  # noqa: BLE001
            log(tag, f"display=true 但无法创建 GUI 窗口（环境无显示器？），"
                      f"已自动关闭实时显示: {exc}")
            _disp_cv2 = None

    try:
        while online and not stop_event.is_set():
            # ---- 滑动攒批：满 N 帧或超时 ----
            _flush_done()  # 上一轮批次已处理完的相机，发完成事件（略延迟 1 轮，无碍）
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
                    done_pending.add(cam_id)
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
                    # 复制一份帧用于绘制/发布，避免污染首帧存档（persist_init 用原帧）
                    _disp = frame.copy()
                    _dh, _dw = frame.shape[:2]
                    mapping = session.registry.assign(dets, msg["ts"])
                    session.multi.prune(msg["frame_id"])
                    frame_pids = []
                    frame_records = []

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
                        frame_pids.append(pid)
                        frame_records.append(record)
                        if cum_dx is not None:
                            put_latest(spectrum_q, {
                                "cam_id": cam_id, "pid": pid,
                                "t": float(msg["ts"]),
                                "dx_mm": cum_dx, "dy_mm": cum_dy,
                            })

                    # 发布本帧实际输出的目标集合；空集合也有意义，Recorder
                    # 可据此清掉已经离开画面的旧目标。
                    put_blocking(result_q, {
                        "branch": "marker", "kind": "frame_snapshot",
                        "cam_id": cam_id, "ts": float(msg["ts"]),
                        "data": {"frame_id": int(msg["frame_id"]),
                                 "pids": sorted(set(frame_pids))},
                    })
                    # 实时叠加：把本帧所有 det 的标注画到副本帧上。
                    # 没有检测时 frame_records 为空，不能引用未定义的 record。
                    if draw_marker is not None:
                        for rec in frame_records:
                            draw_marker(_disp, rec, _dw, _dh)

                    # 实时叠加窗口：一帧所有 det 画完后弹出（窗口名区分摄像头）；
                    # 无 GUI（display=false）时跳过，实时画面统一在 Web 大屏由
                    # dashboard 直接拉 RTSP 子码流显示，main 不再发布任何图片。
                    if _disp_cv2 is not None:
                        _disp_cv2.imshow(f"marker:{cam_id}", _disp)
                        _disp_cv2.waitKey(1)

                    # 首次运行：标定就绪后持久化初始数据
                    if (session.registry.initialized
                            and not session.registry.loaded_from_disk):
                        session.persist_init(frame)
    finally:
        _flush_done()  # 兜底：循环因某路下线退出前，发出剩余完成事件
        if _disp_cv2 is not None:
            try:
                _disp_cv2.destroyAllWindows()
            except Exception:
                pass
        for session in sessions.values():
            try:
                session.finalize_init()
            except Exception:
                pass
        if spectrum_q is not None:
            try:
                spectrum_q.put(SENTINEL, timeout=2.0)
            except Exception:
                pass
        log(tag, "退出")
