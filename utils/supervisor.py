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

from .common import SENTINEL, log
from .config import cameras_by_task, load_config


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

        procs["pole_capture"] = _spawn(
            "pole_capture", run_capture,
            ("pole", pole_cams, pole_queues, result_q, stop_event,
             runtime["capture"], cfg.get("pole", {})))
        procs["pole_inference"] = _spawn(
            "pole_inference", run_pole_inference,
            (cfg, pole_cams, pole_queues, result_q, stop_event, refresh_init))

    if marker_cams:
        from marker_subpixel_tracker.worker import run_marker_inference
        from marker_subpixel_tracker.spectrum import run_spectrum_analyzer

        procs["marker_capture"] = _spawn(
            "marker_capture", run_capture,
            ("marker", marker_cams, marker_queues, result_q, stop_event,
             runtime["capture"], None))
        procs["marker_inference"] = _spawn(
            "marker_inference", run_marker_inference,
            (cfg, marker_cams, marker_queues, result_q, spectrum_q,
             stop_event, refresh_init))
        procs["spectrum"] = _spawn(
            "spectrum", run_spectrum_analyzer,
            (float(cfg["marker"]["fft"]["window_s"]),
             float(cfg["marker"]["fft"]["snr"]),
             int(cfg["marker"]["fft"]["min_samples"]),
             spectrum_q, result_q, stop_event))

    log("supervisor", f"已启动 {len(procs)} 个进程: "
                      f"{list(procs.keys())}")
    critical = {k: v for k, v in procs.items()
                if k in ("pole_inference", "marker_inference",
                         "spectrum", "recorder")}
    workers = {k: v for k, v in procs.items() if k != "recorder"}

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
