"""主频分析进程（SpectrumAnalyzer，全局唯一，属靶标分支）。

消费各 marker 摄像头的累计位移时序样本（cam_id, pid, t, dx_mm, dy_mm），
按固定不重叠 window_s 秒窗口分段，对每个目标的 dx、dy 两个轴分别做 FFT 提取主频，
SNR 门控通过后将两轴结果都产出到 spectrum.jsonl。Dashboard 展示层再固定选择 dy，
用于评估靶标上下振动。
"""
import time
from datetime import datetime

import numpy as np

from utils.common import SENTINEL, log, put_blocking
from .gate_core import check_fingerprint, evaluate_window, load_gate_config


def _clip_stamp():
    """振动报警片段命名：实时流用当前时间，格式 YYYYMMDD_HHMMSS_ffffff。"""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _opt_float(v):
    """可选数值字段：None 保留 None，其余转 float（新消息缺字段时兜底）。"""
    return None if v is None else float(v)


def _segment_spectrum(t, y, snr_min, nb=200, *, cfg=None, meta=None,
                      window_s=None, min_samples=None, min_span_s=None):
    """对一段时序做谱分析，返回完整结果字典或 None（样本不足）。

    兼容旧签名 _segment_spectrum(t, y, snr_min)：cfg=None 时走 legacy 口径
    （单段 FFT + SNR 门），现有看板行为零变化。cfg 为 load_gate_config() 的
    返回值时启用四门+丢帧率+ID 连续性门，判定在 px 域、输出幅值在 mm 域。

    返回字段保持旧结构（freq/power/f1_hz/snr/n/dt），另加：
      max_amp_mm, gates, reason, valid, mode
    f1_hz=None 表示门控未通过（与旧 SNR 未达标语义一致）。
    """
    # legacy 路径：y 既作 mm 也作 px（不影响，legacy 只算 FFT）
    r = evaluate_window(t, y, y, snr_min, cfg=cfg, meta=meta,
                        window_s=window_s, min_samples=min_samples,
                        min_span_s=min_span_s)
    if r is None or r.get("n", 0) < 4 or r.get("reason") in (
            "样本不足", "时间戳异常", "谱长度不足"):
        return None
    return r


def _segment_fft(t, y, snr_min, **kwargs):
    """对一段时序做谱，返回 (f1_hz, snr, n) 或 None（门控未通过）。

    委托 _segment_spectrum，保证「主频 / 功率谱下发前端 / 标定门控」零分叉。
    """
    r = _segment_spectrum(t, y, snr_min, **kwargs)
    if r is None:
        return None
    return r["f1_hz"], r["snr"], r["n"]


def run_spectrum_analyzer(window_s: float, snr_min: float, min_samples: int,
                          in_q, result_q, stop_event, min_span_s=None,
                          amp_threshold=0.0,
                          deflection_threshold=0.0, cable_force_threshold=0.0,
                          placeholder_deflection=0.0,
                          placeholder_cable_force=100.0,
                          realtime_cam_ids=None, clip_q=None,
                          output_dir="output", file_sources=None,
                          gate_config=None):
    """主频分析进程入口。gate_config=load_gate_config() 返回值；None 走 legacy。"""
    tag = "spectrum"
    window = float(window_s)
    span_need = (min(float(min_span_s), window)
                 if min_span_s is not None else 0.0)
    amp_threshold = float(amp_threshold)
    deflection_threshold = float(deflection_threshold)
    cable_force_threshold = float(cable_force_threshold)
    placeholder_deflection = float(placeholder_deflection)
    placeholder_cable_force = float(placeholder_cable_force)
    # (cam_id, pid) -> {"samples": [dict], "last_seg": int}
    # sample dict: t, dx_mm, dy_mm, dx_px, dy_px, frame_id, raw_track_id, continuous
    buffers = {}
    _gate_mode = "calibrated" if gate_config else "legacy"
    log(tag, f"启动，窗口 {window}s, SNR>={snr_min}, 最少 {min_samples} 样本"
             f", 最小跨度 {span_need:g}s, 门控={_gate_mode}"
             f"(标定={gate_config.get('path') if gate_config else '无，走 legacy'})"
             f", 振幅报警阈值={amp_threshold:g}mm")
    if gate_config:
        for w in check_fingerprint(gate_config, 1.0 / max(
                float(gate_config.get("fingerprint", {}).get("rate_hz") or 25.0), 1.0)):
            log(tag, f"[标定告警] {w}")

    def _finalize(key, seg, allow_partial=False):
        buf = buffers.get(key)
        if buf is None:
            return
        t0, t1 = seg * window, (seg + 1) * window
        samples = [s for s in buf["samples"] if t0 <= s["t"] < t1]
        buf["samples"] = [s for s in buf["samples"] if s["t"] >= t1]
        if not samples:
            return
        cam_id, pid = key
        # 真实位移幅值（mm）：本窗内累计位移模长最大值（与看板同源）。
        amp = float(max((s["dx_mm"] ** 2 + s["dy_mm"] ** 2) ** 0.5 for s in samples))
        amp_alarm = amp_threshold > 0 and amp >= amp_threshold
        # 主频取 SNR 更高的轴（legacy）或第一个通过四门的轴（calibrated）。
        best = {"axis": None, "f1": None, "snr": -1.0}
        best_gates = None
        # 原始频谱文件保留两个方向；Dashboard 展示层固定选择 dy（上下振动）。
        for axis, k_mm, k_px in (("dx", "dx_mm", "dx_px"), ("dy", "dy_mm", "dy_px")):
            ts = [s["t"] for s in samples]
            ys_mm = [s[k_mm] for s in samples]
            ys_px = [s.get(k_px) for s in samples]
            if gate_config and (any(v is None for v in ys_px)):
                ys_px = ys_mm  # 旧消息无 px 字段时退化为 mm（标定域失配风险，记 detail）
            elif not ys_px or any(v is None for v in ys_px):
                ys_px = ys_mm
            # EOF 尾段按实际样本计算，至少保留 FFT 所需的 4 个点。
            if len(ts) < (4 if allow_partial else min_samples):
                continue
            if not allow_partial and span_need > 0.0 and (len(ts) < 2 or
                                    (ts[-1] - ts[0]) < span_need - 1e-9):
                continue
            meta = {"frame_id": [s.get("frame_id") for s in samples],
                    "raw_track_id": [s.get("raw_track_id") for s in samples],
                    "continuous": [s.get("continuous") for s in samples]}
            r = _segment_spectrum(ts, ys_px, snr_min, cfg=gate_config, meta=meta,
                                  window_s=window, min_samples=min_samples,
                                  min_span_s=span_need)
            if r is None:
                continue
            f1, snr, n = r["f1_hz"], r["snr"], r["n"]
            result_t1 = samples[-1]["t"] if allow_partial else t1
            data = {"pid": pid, "axis": axis, "t0": t0, "t1": result_t1,
                    "f1_hz": f1, "snr": snr, "n_samples": n}
            if gate_config:
                data["gates"] = r.get("gates") or {}
                data["gate_reason"] = r.get("reason") or ""
                data["mode"] = "calibrated"
                data["max_amp_mm"] = r.get("max_amp_mm")
            put_blocking(result_q, {
                "branch": "marker", "kind": "spectrum", "cam_id": cam_id,
                "ts": result_t1,  # 尾段记录实际终点，而非虚构的整窗终点
                "data": data,
            })
            if f1 is not None and snr == snr and snr < 1e300 and snr > best["snr"]:
                best = {"axis": axis, "f1": f1, "snr": snr}
                best_gates = r.get("gates")
        # ---- 三种振动报警条件（共享同一窗口的 clip 与分析数据）----
        # 振幅为真实量；挠度/索力物理模型未定，保留占位值并显式标注。
        # 三个阈值沿用振幅约定（>0 启用，<=0 关闭）。任一条件触发即在该窗口
        # 留存报警数据（事件 + 振动分析 + 视频片段），片段在窗口内共享。
        defl_alarm = deflection_threshold > 0 and placeholder_deflection >= deflection_threshold
        cable_alarm = cable_force_threshold > 0 and placeholder_cable_force >= cable_force_threshold
        any_alarm = amp_alarm or defl_alarm or cable_alarm
        if any_alarm:
            # 振动报警事件：全局事件日志（events.jsonl）承载"报警"语义；
            # 原始视频片段：
            #  - 实时流(RTSP)：本进程发出切窗请求，由 capture 滚动缓冲切出
            #    {当前时间}_original.mp4（H.264，web 可播），事件内先预留 clip 路径；
            #  - 文件源：收尾阶段 supervisor 从源文件抽取 {t0}_{t1}_original.mp4
            #    （保持原编码，不转 web）。
            # 三个条件任一触发都复用同一窗口片段，切窗请求每个窗口只发一次。
            seg_id = f"{cam_id}_{pid}_w{seg}"
            result_t1 = samples[-1]["t"] if allow_partial else t1
            is_rt = bool(realtime_cam_ids and cam_id in realtime_cam_ids)
            source = "realtime" if is_rt else "file"
            clip_url = None
            if is_rt:
                cname = f"{_clip_stamp()}_original.mp4"
                clip_url = f"/media/output/vibration/{cname}"
            analysis = {
                "window": seg, "t0": t0, "t1": result_t1,
                "frequency_hz": best["f1"],
                "frequency_axis": best["axis"],
                "amplitude_mm": amp,
                "amplitude_threshold_mm": amp_threshold,
                "deflection_mm": placeholder_deflection,
                "deflection_placeholder": True,
                "deflection_threshold_mm": deflection_threshold,
                "cable_force_idx": placeholder_cable_force,
                "cable_force_placeholder": True,
                "cable_force_threshold_idx": cable_force_threshold,
                "source": source,
                "clip": clip_url,
            }
            # 文件源：报警即同步抽取源文件片段（不依赖收尾/周期），与 RTSP 实时切窗对等。
            # 源文件已在磁盘上，无需"录制"，故报警瞬间即可成片；侧车 json 仍由 retain 周期补。
            if not is_rt:
                _src = (file_sources or {}).get(cam_id)
                if _src:
                    try:
                        from pathlib import Path as _P
                        from viz.clip_retention import (
                            extract_clip_native, ensure_web_h264)
                        _cname = f"{t0:.1f}_{result_t1:.1f}_original.mp4"
                        _out = str(_P(output_dir) / "vibration" / _cname)
                        _r1 = extract_clip_native(
                            _src, float(t0), float(result_t1), _out)
                        if _r1.get("status") == "ok":
                            analysis["clip_native"] = (
                                f"/media/output/vibration/{_cname}")
                            _r2 = ensure_web_h264(
                                _out, _P(output_dir) / "vibration")
                            if _r2.get("status") in ("ok", "reuse"):
                                analysis["clip"] = (
                                    f"/media/output/vibration/"
                                    f"{_P(_cname).stem}_h264.mp4")
                            else:
                                analysis["clip"] = analysis["clip_native"]
                            analysis["clip_status"] = (
                                "ok" if analysis["clip"] else "ok_no_web_copy")
                        else:
                            analysis["clip_status"] = "extract_failed"
                    except Exception as exc:
                        log(tag, f"文件源振动片段抽取失败 {cam_id}: {exc}")
                        analysis["clip_status"] = "extract_failed"
                clip_url = analysis["clip"]  # 同步给下方 event 的 clip 字段
            # 实时流：本窗口仅发一次切窗请求（三条件共享片段）；文件源已在上方同步抽取
            if is_rt and clip_q is not None:
                try:
                    from pathlib import Path as _P
                    out_path = str(_P(output_dir) / "vibration" / cname)
                    clip_q.put({
                        "cam_id": cam_id, "t0": float(t0), "t1": float(result_t1),
                        "out_path": out_path, "transcode": True,
                    })
                except Exception as exc:
                    log(tag, f"RTSP 振动片段切窗请求发送失败 {cam_id}: {exc}")
            # 三种振动报警条件共享同一窗口的 analysis 与视频片段：合并为「一条」
            # vibration_alarm 事件（id 不再带 _defl/_cable 后缀），避免同一(相机,
            # 目标,窗)被拆成 3 条记录（振幅/挠度/索力全触发时 6 条→2 条，符合
            # "一次分析→一次上报→每目标最多 1 条"）。triggered 列出实际触发条件，
            # 看板据其渲染，无需再分 3 行。
            if any_alarm:
                triggered = []
                if amp_alarm:
                    triggered.append("amplitude")
                if defl_alarm:
                    triggered.append("deflection")
                if cable_alarm:
                    triggered.append("cable_force")
                parts = []
                if amp_alarm:
                    parts.append(f"振幅 {amp:.2f}mm(阈{amp_threshold:.2f})")
                if defl_alarm:
                    parts.append(f"挠度 {placeholder_deflection:.2f}mm占位"
                                 f"(阈{deflection_threshold:.2f})")
                if cable_alarm:
                    parts.append(f"索力 {placeholder_cable_force:.2f}占位"
                                 f"(阈{cable_force_threshold:.2f})")
                analysis = dict(analysis)
                analysis["triggered"] = triggered
                put_blocking(result_q, {
                    "branch": "marker", "kind": "event", "cam_id": cam_id,
                    "event": "vibration_alarm", "id": seg_id,
                    "ts": result_t1, "pid": pid, "level": 1,
                    "triggered": triggered,
                    "detail": (f"靶标 {pid} 振动报警（" + "；".join(parts) + "）"),
                    "clip": clip_url,  # 实时流预留 /media/output/vibration/<当前时间>_original.mp4
                    "analysis": analysis,
                })

    try:
        while not stop_event.is_set():
            try:
                item = in_q.get(timeout=1.0)
            except Exception:
                continue
            if item is SENTINEL:
                break
            if item.get("kind") == "camera_done":
                # 单路文件 EOF 不必等待其他相机（尤其是永不结束的 RTSP）。
                cid = item["cam_id"]
                for key in [key for key in buffers if key[0] == cid]:
                    _finalize(key, buffers[key]["last_seg"], allow_partial=True)
                    buffers.pop(key, None)
                put_blocking(result_q, {
                    "branch": "marker", "kind": "spectrum_done",
                    "cam_id": cid, "ts": item.get("ts"),
                })
                continue
            key = (item["cam_id"], item["pid"])
            buf = buffers.setdefault(key, {"samples": [], "last_seg": -1})
            seg = int(float(item["t"]) // window)
            # 进入新窗口时，结算上一窗口（可能跨越多个空窗）
            if seg > buf["last_seg"]:
                for k in range(buf["last_seg"], seg):
                    _finalize(key, k)
                buf["last_seg"] = seg
            # 整份透传：新增的 px/frame_id/raw_track_id/continuous 供门控使用，
            # 旧 worker 发来的消息缺字段时用 None 兜底，gate_core 会判 SKIP 或退化。
            buf["samples"].append({
                "t": float(item["t"]),
                "dx_mm": float(item["dx_mm"]), "dy_mm": float(item["dy_mm"]),
                "dx_px": _opt_float(item.get("dx_px")),
                "dy_px": _opt_float(item.get("dy_px")),
                "frame_id": item.get("frame_id"),
                "raw_track_id": item.get("raw_track_id"),
                "continuous": item.get("continuous"),
            })
    finally:
        # 冲刷所有已结束窗口
        for key in list(buffers.keys()):
            buf = buffers[key]
            for k in range(buf["last_seg"],
                           int(buf["samples"][-1]["t"] // window) + 1
                           if buf["samples"] else buf["last_seg"] + 1):
                _finalize(key, k, allow_partial=True)
        log(tag, "退出")
