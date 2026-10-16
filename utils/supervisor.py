"""Supervisor 主进程：进程编排 + 健康监控 + 优雅退出。

职责（对齐 JRY_BAA_121422 mainStream）：
- spawn 启动子进程（模型只在推理进程内加载，本进程绝不 import 推理框架）；
- 有界队列连接单向 DAG；
- is_alive() 僵尸检测：推理/分析/记录任一死亡 → 统一 terminate；
- 全部生产者正常结束 → 通知 Recorder 冲刷退出，正常收尾；
- Ctrl+C 优雅退出。

进程拓扑（进程总数恒为 7，与摄像头数无关）：
  1 Supervisor + 1 PoleCapture + 1 PoleInference
  + 1 MarkerCapture + 1 MarkerInference + 1 SpectrumAnalyzer + 1 Recorder
"""
import multiprocessing as mp
import time
import shutil
from pathlib import Path

from .common import SENTINEL, is_realtime_source, log
from .config import cameras_by_task, load_config


def clear_run_outputs(cfg: dict, refresh_init: bool = False) -> None:
    """清理会把上一轮结果误显示到新一轮的运行产物。

    ``latest.json`` 在没有新结果时不会被 Recorder 覆盖，初始化文件也可能在
    新一轮首个有效检测前继续被 Dashboard 读到。因此 refresh-init 必须在
    Dashboard 启动前清掉这些精确文件；JSONL 也一并删除，避免短暂重用旧时间轴。
    """
    if not refresh_init:
        return
    output_dir = Path(cfg["runtime"]["output_dir"])
    vibration_dir = output_dir / "vibration"
    rtsp_buffer_dir = output_dir / "rtsp_buffer"
    if vibration_dir.exists():
        shutil.rmtree(vibration_dir)
    if rtsp_buffer_dir.exists():
        shutil.rmtree(rtsp_buffer_dir)
    # 全局报警事件日志：跨相机累加，若不随 refresh-init 清除，新一轮运行会在
    # 看板"报警记录"页把历史报警一并显示，造成"记录越来越多"的假象。清掉重启干净。
    events_log = output_dir / "events.jsonl"
    try:
        events_log.unlink()
        log("supervisor", f"refresh-init 清除旧结果: {events_log}")
    except FileNotFoundError:
        pass
    except OSError as exc:
        log("supervisor", f"refresh-init 清除失败 {events_log}: {exc}")
    for cam in cfg.get("cameras", []):
        branch = cam.get("task")
        cid = cam.get("id")
        if branch not in ("pole", "marker") or not cid:
            continue
        cam_dir = output_dir / branch / cid
        targets = [
            cam_dir / "latest.json",
            cam_dir / "measurements.jsonl",
            cam_dir / "displacement.jsonl",
            cam_dir / "spectrum.jsonl",
            cam_dir / f"{cid}_annotated.mp4",
            cam_dir / ".rec_done",
            cam_dir / ".export_failed",
            cam_dir / "init" / "init_state.json",
            cam_dir / "init" / "init_frame.jpg",
        ]
        
        for path in targets:
            try:
                path.unlink()
                log("supervisor", f"refresh-init 清除旧结果: {path}")
            except FileNotFoundError:
                pass
            except OSError as exc:
                log("supervisor", f"refresh-init 清除失败 {path}: {exc}")


def _amp_threshold(cfg):
    """振动报警的位移幅值阈值（mm）；<=0 表示关闭振幅报警。"""
    try:
        return float((cfg.get("marker", {}) or {}).get("alarm", {})
                     .get("amplitude_mm", 0.0))
    except Exception:
        return 0.0


def _ph_deflection(cfg):
    try:
        return float((cfg.get("marker", {}) or {}).get("placeholder", {})
                     .get("deflection_mm", 0.0))
    except Exception:
        return 0.0


def _ph_cable_force(cfg):
    try:
        return float((cfg.get("marker", {}) or {}).get("placeholder", {})
                     .get("cable_force_idx", 100.0))
    except Exception:
        return 100.0


def _defl_threshold(cfg):
    """振动报警的挠度阈值（mm）；<=0 表示关闭挠度报警。

    与振幅报警一致沿用 >0 启用约定。当前挠度取自 marker.placeholder（占位值，
    物理模型未定），阈值比较作用于该占位值；真实模型就绪后该占位值被真实量替换，
    报警逻辑无需改动。
    """
    try:
        return float((cfg.get("marker", {}) or {}).get("alarm", {})
                     .get("deflection_mm", 0.0))
    except Exception:
        return 0.0


def _cable_threshold(cfg):
    """振动报警的索力阈值（相对量纲，与占位值 cable_force_idx 同量纲）；<=0 表示关闭。

    索力占位值默认 100（首帧基准），设例如 80 即可在占位值下触发报警；真实模型
    就绪后由真实索力替换占位值，比较逻辑不变。
    """
    try:
        return float((cfg.get("marker", {}) or {}).get("alarm", {})
                     .get("cable_force_idx", 0.0))
    except Exception:
        return 0.0


def _retain_vibration_clips(cfg):
    """收尾：为 amplitude_alarm 窗口抽取原始视频片段 + 振动分析数据。

    详见 viz.clip_retention.retain_vibration_clips。RTSP 实时源无落地文件，跳过片段
    抽取（已知缺口：main 不录制 RTSP）；分析数据仍随 events.jsonl 保留。
    """
    try:
        from viz.clip_retention import retain_vibration_clips as _rvc
    except Exception as exc:
        log("supervisor", f"振动留存模块不可用: {exc}")
        return
    try:
        produced = _rvc(cfg)
        if produced:
            log("supervisor", f"振动留存完成，共 {len(produced)} 个窗口片段/分析")
    except Exception as exc:
        log("supervisor", f"振动留存异常: {exc}")


def _export_annotated_videos(cfg):
    """main 跑完后，为每个 file 源相机生成标注视频到各自相机文件夹下。

    stream 源（rtsp/http…）无本地文件，跳过；viz_overlay 仍可手动重渲染到 output/viz。
    产物：<output>/<branch>/<cam_id>/<cam_id>_annotated.mp4（H.264，检测框+拟合直线已编码）。
    """
    try:
        from viz.export import export_cam_video
    except Exception as exc:
        log("supervisor", f"标注视频导出不可用（缺 cv2/ffmpeg？）: {exc}")
        return
    output_dir = cfg["runtime"]["output_dir"]
    pole_mode = (((cfg.get("pole", {}) or {}).get("measurement", {}) or {})
                 .get("mode", "image_relative"))
    config_dir = cfg.get("_config_dir") or "."
    for cam in cfg.get("cameras", []):
        url = cam.get("url", "") or ""
        if is_realtime_source(url):
            continue  # 实时流无本地文件，无法离线回贴
        src = Path(url)
        src_abs = src if src.is_absolute() else Path(config_dir) / src
        c = {"id": cam["id"], "task": cam["task"], "url": str(src_abs),
             "calibration": cam.get("calibration")}
        out_dir = Path(output_dir) / cam["task"] / cam["id"]
        try:
            st = export_cam_video(c, output_dir, pole_mode, out_dir)
        except Exception as exc:
            log("supervisor", f"[标注视频] {cam['id']} 导出异常: {exc}")
            continue
        if st.get("status") == "ok":
            log("supervisor", f"[标注视频] {cam['id']} → {st.get('out')} "
                              f"(h264={st.get('h264')}, "
                              f"已标注帧={st.get('frames_annotated')})")
            # 本轮已成功，清除可能存在的失败标记
            try:
                (Path(output_dir) / cam["task"] / cam["id"]
                 / ".export_failed").unlink()
            except Exception:
                pass
        else:
            log("supervisor", f"[标注视频] {cam['id']} 跳过: {st.get('reason')}")


def _poll_done_exports(cfg, exported):
    """文件源相机数据全部落盘后即时导出标注视频，无需等 main 整体退出。

    完成判据是 **Recorder 亲手写下的 .rec_done**（确定性事件）：推理端该路跑完时
    发 camera_done 到 result_q，Recorder 收到时该路 jsonl 已 flush 完毕才写标记。
    不用"jsonl 不增长了"这类猜测判据——无目标时段本就不写行，会误判成已完成。
    流源永不产生 camera_done，故无 .rec_done，跳过。
    """
    output_dir = cfg["runtime"]["output_dir"]
    config_dir = cfg.get("_config_dir") or "."
    pole_mode = (((cfg.get("pole", {}) or {}).get("measurement", {}) or {})
                 .get("mode", "image_relative"))
    for cam in cfg.get("cameras", []):
        cid = cam.get("id")
        if not cid or cid in exported:
            continue
        url = cam.get("url", "") or ""
        if is_realtime_source(url):
            continue  # 实时流：无完成概念，标记永不出现
        branch = cam.get("task")
        if branch not in ("pole", "marker"):
            continue
        if not (Path(output_dir) / branch / cid / ".rec_done").is_file():
            continue
        # Recorder 已确认该路结果全部落盘 → 直接导出该路标注视频
        src = Path(url)
        src_abs = src if src.is_absolute() else Path(config_dir) / src
        c = {"id": cid, "task": branch, "url": str(src_abs),
             "calibration": cam.get("calibration")}
        out_dir = Path(output_dir) / branch / cid
        try:
            from viz.export import export_cam_video
        except Exception as exc:
            log("supervisor", f"[即时标注视频] 导出模块不可用: {exc}")
            exported.add(cid)
            continue
        res = None
        for _ in range(3):
            try:
                res = export_cam_video(c, output_dir, pole_mode, out_dir)
                break
            except OSError:
                time.sleep(1.0)  # 偶发文件占用，稍后重试
        if res is None:
            # 导出失败：写失败标记（大屏如实提示），留待 main 退出时 _finish 兜底再导
            try:
                (Path(output_dir) / branch / cid / ".export_failed").write_text(
                    "export failed", encoding="utf-8")
            except Exception:
                pass
            log("supervisor", f"[即时标注视频] {cid} 导出失败，留待退出时再导出")
            exported.add(cid)
            continue
        exported.add(cid)
        if res.get("status") == "ok":
            log("supervisor", f"[即时标注视频] {cid} → {res.get('out')} "
                              f"(h264={res.get('h264')}, 已标注帧={res.get('frames_annotated')})")
            try:
                (Path(output_dir) / branch / cid / ".export_failed").unlink()
            except Exception:
                pass
        else:
            log("supervisor", f"[即时标注视频] {cid} 跳过: {res.get('reason')}")


def _spawn(name, target, args):
    p = mp.Process(target=target, args=args, name=name)
    p.start()
    time.sleep(0.3)  # 错峰启动，错开模型加载
    return p


def run(config_path: str, refresh_init: bool = False) -> int:
    mp.set_start_method("spawn", force=True)

    cfg = load_config(config_path)
    runtime = cfg["runtime"]
    pole_cams = cameras_by_task(cfg, "pole")
    marker_cams = cameras_by_task(cfg, "marker")
    if not pole_cams and not marker_cams:
        log("supervisor", "无摄像头配置，退出")
        return 1

    # 强制重建初始基准时先移除上一轮初始化、快照、分析和视频产物。否则在
    # 新轮次尚未检测到有效目标期间，Dashboard 会继续读到旧结果，把“正在
    # 重标定”误显示成“已完成初始化”。只处理每个已配置相机的精确路径。
    clear_run_outputs(cfg, refresh_init=refresh_init)

    queue_cfg = runtime["queue"]
    result_q = mp.Queue(int(queue_cfg["result"]))
    spectrum_q = mp.Queue(int(queue_cfg["spectrum"])) if marker_cams else None
    # 振动报警切窗请求队列：spectrum 进程写入、capture 进程（仅实时 marker 相机）
    # 消费，从滚动缓冲切出 {当前时间}_original.mp4。无 marker 实时源时为 None。
    clip_q = mp.Queue(512) if marker_cams else None
    stop_event = mp.Event()

    # 每个摄像头一条有界帧队列
    frame_queues = {}
    for cam in cfg["cameras"]:
        size = (int(queue_cfg["pole_frame"]) if cam["task"] == "pole"
                else int(queue_cfg["marker_frame"]))
        frame_queues[cam["id"]] = mp.Queue(size)

    pole_queues = {c["id"]: frame_queues[c["id"]] for c in pole_cams}
    marker_queues = {c["id"]: frame_queues[c["id"]] for c in marker_cams}
    # 实时（RTSP）marker 相机：振动报警时由 capture 滚动缓冲切窗，而非收尾抽源文件
    realtime_marker = {c["id"] for c in marker_cams
                       if is_realtime_source(c.get("url", ""))}
    # 文件源 marker 相机：源文件绝对路径，供 spectrum 报警时即时抽取片段
    # （文件源无滚动缓冲，源文件已在磁盘，可报警瞬间成片，不依赖收尾/周期）。
    _config_dir = cfg.get("_config_dir") or "."
    def _abs_src(url):
        p = Path(url)
        return str(p if p.is_absolute() else Path(_config_dir) / p)
    file_marker_sources = {
        c["id"]: _abs_src(c.get("url", ""))
        for c in marker_cams
        if not is_realtime_source(c.get("url", ""))
    }

    # ---- 启动进程 ----
    from .capture import run_capture
    from .recorder import run_recorder

    procs = {}
    procs["recorder"] = _spawn(
        "recorder", run_recorder,
        (result_q, runtime["output_dir"], stop_event))

    if pole_cams:
        from pole_tilt_monitor.worker import run_pole_inference

        # 推理就绪事件：capture 在消费方预热完成前不投递帧，避免冷启动丢帧
        pole_ready = mp.Event()
        procs["pole_capture"] = _spawn(
            "pole_capture", run_capture,
            ("pole", pole_cams, pole_queues, result_q, stop_event,
             runtime["capture"], cfg.get("pole", {}), pole_ready))
        procs["pole_inference"] = _spawn(
            "pole_inference", run_pole_inference,
            (cfg, pole_cams, pole_queues, result_q, stop_event,
             refresh_init, pole_ready))

    if marker_cams:
        from marker_subpixel_tracker.worker import run_marker_inference
        from marker_subpixel_tracker.spectrum import run_spectrum_analyzer

        marker_ready = mp.Event()
        procs["marker_capture"] = _spawn(
            "marker_capture", run_capture,
            ("marker", marker_cams, marker_queues, result_q, stop_event,
             runtime["capture"], None, marker_ready, clip_q,
             str(runtime["output_dir"]),
             float(runtime["capture"].get("rtsp_rolling_s", 120.0)),
             float(runtime["capture"].get("rtsp_seg_s", 10.0))))
        procs["marker_inference"] = _spawn(
            "marker_inference", run_marker_inference,
            (cfg, marker_cams, marker_queues, result_q, spectrum_q,
             stop_event, refresh_init, marker_ready))
        procs["spectrum"] = _spawn(
            "spectrum", run_spectrum_analyzer,
            (float(cfg["marker"]["fft"]["window_s"]),
             float(cfg["marker"]["fft"]["snr"]),
             int(cfg["marker"]["fft"]["min_samples"]),
             spectrum_q, result_q, stop_event,
             float(cfg["marker"]["fft"].get("min_span_s", 0.0)),
             _amp_threshold(cfg), _ph_deflection(cfg), _ph_cable_force(cfg),
             _defl_threshold(cfg), _cable_threshold(cfg),
             realtime_marker, clip_q, str(runtime["output_dir"]),
             file_marker_sources))

    log("supervisor", f"已启动 {len(procs)} 个进程: "
                      f"{list(procs.keys())}")
    critical = {k: v for k, v in procs.items()
                if k in ("pole_inference", "marker_inference",
                         "spectrum", "recorder")}
    workers = {k: v for k, v in procs.items() if k != "recorder"}

    # 启动即清除文件源上一轮的标注视频、完成标记与失败标记，避免重新跑同一文件时
    # 大屏在"处理中"阶段误显示旧结果、或残留标记让本轮被误判为已就绪；
    # 本轮跑完会由 Recorder 重写 .rec_done、supervisor 重新生成标注视频。
    for cam in cfg.get("cameras", []):
        url = cam.get("url", "") or ""
        if is_realtime_source(url):
            continue
        d = Path(cfg["runtime"]["output_dir"]) / cam["task"] / cam["id"]
        for fn in (f"{cam['id']}_annotated.mp4", ".export_failed", ".rec_done"):
            try:
                d.joinpath(fn).unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass

    exported = set()        # 已即时导出（或已判定失败留待兜底）的相机，避免重复导出

    def _terminate_all(exit_code):
        stop_event.set()
        for name, p in procs.items():
            if p.is_alive():
                p.terminate()
        for p in procs.values():
            try:
                p.join(timeout=5)
            except Exception:
                pass
        log("supervisor", f"全部进程已终止（exit={exit_code}）")
        return exit_code

    def _finish():
        # 生产者全部结束 → 通知 Recorder 冲刷退出
        try:
            result_q.put(SENTINEL, timeout=5)
        except Exception:
            pass
        r = procs["recorder"]
        r.join(timeout=15)
        if r.is_alive():
            r.terminate()
            r.join(timeout=5)
        # 收尾后自动生成每相机标注视频（file 源）到各自相机文件夹下
        _export_annotated_videos(cfg)
        # 振动报警留存：文件源相机抽取原始片段 + 振动分析数据（在标注视频之后，
        # 此时 events.jsonl 已由 Recorder 全部 flush 完毕）
        _retain_vibration_clips(cfg)
        log("supervisor", "正常收尾完成")
        return 0

    # 振动报警视频留存周期触发：文件源报警视频需从源文件抽取，原本只在 _finish()
    # 收尾时抽取；但含 RTSP 长驻相机时 main 永不退出、收尾永不到达，文件源报警视频
    # 永不生成（output/vibration 不建）。改为周期抽取（_retain_vibration_clips 幂等，
    # 已抽取窗口跳过），报警后最多 RETAIN_INTERVAL_S 内落盘，RTSP 长驻不再阻塞。
    retain_interval_s = 15.0
    last_retain = time.monotonic()

    try:
        while True:
            # 僵尸检测：关键进程"异常"退出（exitcode != 0）才整体终止；
            # 正常结束（exitcode == 0，如某分支视频源读尽）不影响其他分支。
            for name, p in critical.items():
                if not p.is_alive() and p.exitcode != 0:
                    log("supervisor",
                        f"关键进程 {name} 异常退出 (exitcode={p.exitcode})")
                    return _terminate_all(1)

            # 文件源相机推理结束 → 即时导出该路标注视频（无需等 main 整体退出）
            _poll_done_exports(cfg, exported)

            # 振动报警视频周期留存（文件源报警经此从源文件抽片段，RTSP 源补全 json
            # 侧车与 _h264 副本；_retain_vibration_clips 幂等，已抽取窗口跳过）。
            # 关键修复：含 RTSP 长驻相机时 main 绝不退出，仅靠收尾抽片段会让文件源
            # 报警视频永不生成、连 output/vibration 目录都不建。
            if marker_cams and (time.monotonic() - last_retain) >= retain_interval_s:
                last_retain = time.monotonic()
                _retain_vibration_clips(cfg)

            # 全部生产者正常结束 → 收尾
            if all(not p.is_alive() for p in workers.values()):
                return _finish()

            time.sleep(0.4)
    except KeyboardInterrupt:
        log("supervisor", "收到 Ctrl+C，优雅退出")
        stop_event.set()
        # 给推理/分析进程发哨兵，令其冲刷退出
        for q in frame_queues.values():
            try:
                q.put(SENTINEL, timeout=1)
            except Exception:
                pass
        if spectrum_q is not None:
            try:
                spectrum_q.put(SENTINEL, timeout=1)
            except Exception:
                pass
        for name, p in workers.items():
            p.join(timeout=15)
            if p.is_alive():
                p.terminate()
        return _finish()
    finally:
        for p in procs.values():
            if p.is_alive():
                p.terminate()
