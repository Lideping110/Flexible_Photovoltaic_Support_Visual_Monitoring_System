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
    stop_event = mp.Event()

    # 每个摄像头一条有界帧队列
    frame_queues = {}
    for cam in cfg["cameras"]:
        size = (int(queue_cfg["pole_frame"]) if cam["task"] == "pole"
                else int(queue_cfg["marker_frame"]))
        frame_queues[cam["id"]] = mp.Queue(size)

    pole_queues = {c["id"]: frame_queues[c["id"]] for c in pole_cams}
    marker_queues = {c["id"]: frame_queues[c["id"]] for c in marker_cams}

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
             runtime["capture"], None, marker_ready))
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
             float(cfg["marker"]["fft"].get("min_span_s", 0.0))))

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
        log("supervisor", "正常收尾完成")
        return 0

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
