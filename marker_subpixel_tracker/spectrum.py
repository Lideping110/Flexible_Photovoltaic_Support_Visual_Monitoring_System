"""主频分析进程（SpectrumAnalyzer，全局唯一，属靶标分支）。

消费各 marker 摄像头的累计位移时序样本（cam_id, pid, t, dx_mm, dy_mm），
按固定不重叠 window_s 秒窗口分段，对每个目标的 dx、dy 两个轴分别做 FFT 提取主频，
SNR 门控通过后将两轴结果都产出到 spectrum.jsonl。Dashboard 展示层再固定选择 dy，
用于评估靶标上下振动。
"""
import time

import numpy as np

from utils.common import SENTINEL, log, put_blocking


def _segment_spectrum(t, y, snr_min, nb=200):
    """对一段时序做 FFT，返回完整结果字典或 None（SNR 不达标 / 样本不足）。

    与主链路固定窗分段（SpectrumAnalyzer._finalize）共用同一套功率谱 / SNR 算法，
    仅额外把功率谱降采样到 nb 个点（freq/power）供前端直接画曲线，无需前端再算一次 FFT。

    返回 {"freq":[...], "power":[...], "f1_hz":float|None, "snr":float, "n":int, "dt":float}
    f1_hz 为 None 表示 SNR 未达标（被门控拒绝，与主链路口径一致）。
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(t)
    if n < 4:
        return None
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        return None
    y = y - y.mean()
    spectrum = np.abs(np.fft.rfft(y)) ** 2
    freqs = np.fft.rfftfreq(n, d=dt)
    if len(spectrum) < 3:
        return None
    body = spectrum[1:]  # 排除直流
    if body.size == 0:
        return None
    k = int(np.argmax(body)) + 1
    peak_power = float(spectrum[k])
    noise = float(np.median(body))
    snr = (peak_power / noise) if noise > 0 else float("inf")
    f1 = float(freqs[k]) if not (snr < float(snr_min) or freqs[k] <= 0) else None
    # 降采样到 nb 个点供画曲线（保持 0~Nyquist 的频率覆盖）
    if len(freqs) > nb:
        idx = np.linspace(0, len(freqs) - 1, nb).astype(int)
        f_d = freqs[idx]
        p_d = spectrum[idx]
    else:
        f_d = freqs
        p_d = spectrum
    return {
        "freq": [float(x) for x in f_d],
        "power": [float(x) for x in p_d],
        "f1_hz": f1,
        "snr": snr,
        "n": int(n),
        "dt": dt,
    }


def _segment_fft(t, y, snr_min):
    """对一段时序做 FFT，返回 (f1_hz, snr, n) 或 None（SNR 不达标）。

    现委托 _segment_spectrum 实现，保证与「功率谱下发前端」走同一条算法（口径零分叉）。
    """
    r = _segment_spectrum(t, y, snr_min)
    if r is None:
        return None
    return r["f1_hz"], r["snr"], r["n"]


def run_spectrum_analyzer(window_s: float, snr_min: float, min_samples: int,
                          in_q, result_q, stop_event, min_span_s=None):
    """主频分析进程入口。"""
    tag = "spectrum"
    window = float(window_s)
    span_need = (min(float(min_span_s), window)
                 if min_span_s is not None else 0.0)
    # (cam_id, pid) -> {"samples": [(t, dx, dy)], "last_seg": int}
    buffers = {}
    log(tag, f"启动，窗口 {window}s, SNR>={snr_min}, 最少 {min_samples} 样本"
             f", 最小跨度 {span_need:g}s")

    def _finalize(key, seg, allow_partial=False):
        buf = buffers.get(key)
        if buf is None:
            return
        t0, t1 = seg * window, (seg + 1) * window
        samples = [(t, dx, dy) for t, dx, dy in buf["samples"]
                   if t0 <= t < t1]
        buf["samples"] = [(t, dx, dy) for t, dx, dy in buf["samples"] if t >= t1]
        cam_id, pid = key
        # 原始频谱文件保留两个方向；Dashboard 展示层固定选择 dy（上下振动）。
        for axis, idx in (("dx", 1), ("dy", 2)):
            ts = [s[0] for s in samples]
            ys = [s[idx] for s in samples]
            # EOF 尾段按实际样本计算，至少保留 FFT 所需的 4 个点。
            if len(ts) < (4 if allow_partial else min_samples):
                continue
            # min_samples 只约束数量；高频或突发重采样可能在很短的时间
            # 内凑够数量，仍不能形成可靠的低频谱。Dashboard 使用同一门限。
            if not allow_partial and span_need > 0.0 and (len(ts) < 2 or
                                    (ts[-1] - ts[0]) < span_need - 1e-9):
                continue
            r = _segment_fft(ts, ys, snr_min)
            if r is None:
                continue
            f1, snr, n = r
            result_t1 = samples[-1][0] if allow_partial else t1
            put_blocking(result_q, {
                "branch": "marker", "kind": "spectrum", "cam_id": cam_id,
                "ts": result_t1,  # 尾段记录实际终点，而非虚构的整窗终点
                "data": {"pid": pid, "axis": axis, "t0": t0, "t1": result_t1,
                         "f1_hz": f1, "snr": snr, "n_samples": n},
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
            buf["samples"].append((float(item["t"]),
                                   float(item["dx_mm"]), float(item["dy_mm"])))
    finally:
        # 冲刷所有已结束窗口
        for key in list(buffers.keys()):
            buf = buffers[key]
            for k in range(buf["last_seg"], int(buf["samples"][-1][0] // window) + 1
                           if buf["samples"] else buf["last_seg"] + 1):
                _finalize(key, k, allow_partial=True)
        log(tag, "退出")
