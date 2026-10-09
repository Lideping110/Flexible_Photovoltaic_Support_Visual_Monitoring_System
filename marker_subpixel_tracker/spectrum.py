"""主频分析进程（SpectrumAnalyzer，全局唯一，属靶标分支）。

消费各 marker 摄像头的累计位移时序样本（cam_id, pid, t, dx_mm, dy_mm），
按固定不重叠 window_s 秒窗口分段，对每个 (cam_id, pid, axis) 做 FFT 提取主频，
SNR 门控通过后产出 spectrum 记录发往 Recorder。
"""
import time

import numpy as np

from utils.common import SENTINEL, log, put_blocking


def _segment_fft(t, y, snr_min):
    """对一段时序做 FFT，返回 (f1_hz, snr, n) 或 None（SNR 不达标）。"""
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
    if snr < float(snr_min) or freqs[k] <= 0:
        return None
    return float(freqs[k]), float(snr), int(n)


def run_spectrum_analyzer(window_s: float, snr_min: float, min_samples: int,
                          in_q, result_q, stop_event):
    """主频分析进程入口。"""
    tag = "spectrum"
    window = float(window_s)
    # (cam_id, pid) -> {"samples": [(t, dx, dy)], "last_seg": int}
    buffers = {}
    log(tag, f"启动，窗口 {window}s, SNR>={snr_min}, 最少 {min_samples} 样本")

    def _finalize(key, seg):
        buf = buffers.get(key)
        if buf is None:
            return
        t0, t1 = seg * window, (seg + 1) * window
        samples = [(t, dx, dy) for t, dx, dy in buf["samples"]
                   if t0 <= t < t1]
        buf["samples"] = [(t, dx, dy) for t, dx, dy in buf["samples"] if t >= t1]
        cam_id, pid = key
        for axis, idx in (("dx", 1), ("dy", 2)):
            ts = [s[0] for s in samples]
            ys = [s[idx] for s in samples]
            if len(ts) < min_samples:
                continue
            r = _segment_fft(ts, ys, snr_min)
            if r is None:
                continue
            f1, snr, n = r
            put_blocking(result_q, {
                "branch": "marker", "kind": "spectrum", "cam_id": cam_id,
                "ts": t1,  # 与位移记录同一时间基准（视频相对秒）
                "data": {"pid": pid, "axis": axis, "t0": t0, "t1": t1,
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
                _finalize(key, k)
        log(tag, "退出")
