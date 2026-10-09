# -*- coding: utf-8 -*-
"""实时 RTSP 视频流 → ByteTrack 多目标跟踪 → 亚像素角点定位 → 实时可视化。

独立脚本，复用 marker_subpixel_tracker/src 的真实算法链路（与 main.py 完全一致），
多目标版本：画面里的**每一个** ByteTrack 轨迹都独立定位、独立测位移。

    RTSP 流 → ByteTrack(model.track 一次前向) → 全部轨迹
           → 每条轨迹独立：crop_box(无扩边) → 灰度化
           → structure_tensor λ_min(Shi-Tomasi 角点响应)
           → predict_anchor 框位移预测锚点(快速移动不滞后)
           → detect_marker_center 锚点窗口选峰 + cornerSubPix 亚像素精化
           → MultiTargetTracker 逐目标帧连续性门控 dx/dy
           → 画面实时叠加所有目标的可视化

运行：
    uv run python marker_tracing.py
    uv run python marker_tracing.py --rtsp rtsp://... --device intel:gpu
    uv run python marker_tracing.py --weights marker26_det.pt --device cpu   # 回退纯 CPU

推理加速：若存在 marker26_det_openvino_model/ 目录(OpenVINO IR 导出产物)，默认自动
用它走 Intel GPU(intel:gpu)。Intel iGPU 实测提速约 1.8 倍(34.7ms -> 19.3ms/帧)。

退出：显示窗口按 q。
"""
import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from marker_subpixel_tracker.src.pipeline import localize_one
from marker_subpixel_tracker.src.tracker import ByteTrackTracker, MultiTargetTracker
from marker_subpixel_tracker.src.features import measure_outer_diameter_px

SCRIPT_DIR = Path(__file__).resolve().parent
# DEFAULT_RTSP = "rtsp://admin:hhjt220220@192.168.1.65:554/streaming/Channels/101"
DEFAULT_RTSP = "rtsp://admin:hhjt110110@192.168.1.64:554/streaming/Channels/101"
DEFAULT_CONFIG = SCRIPT_DIR / "marker_subpixel_tracker" / "config" / "config.yaml"
# OpenVINO IR 导出产物目录(marker26_det.pt 转出)，存在则默认走 Intel GPU 加速
OPENVINO_DIR = SCRIPT_DIR / "marker26s_det_openvino_model"

# RTSP 实时性关键：走 TCP 传输(抗花屏) + nobuffer(降延迟) + 小探测(快速起播)。
# 必须在 cv2.VideoCapture 之前设置才生效。
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|fflags;nobuffer|probesize;32|analyzeduration;0",
)


def resolve_weights(cfg_path, weights):
    """按 config 文件的权重相对路径解析到真实文件(复用 main.py 的思路)。"""
    p = Path(weights)
    if p.is_absolute():
        return p
    for base in (SCRIPT_DIR, cfg_path.parent, cfg_path.parent.parent):
        cand = base / p
        if cand.exists():
            return cand
    return SCRIPT_DIR / p


def is_openvino_dir(p):
    """判断路径是否为 OpenVINO 模型目录(含 .xml 拓扑文件)。"""
    return Path(p).is_dir() and any(Path(p).glob("*.xml"))


def open_stream(rtsp):
    """打开 RTSP 流并尽可能压低缓冲延迟。"""
    cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开 RTSP 流: {rtsp}")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 缓冲压到最小，降低实时延迟
    return cap


def select_primary(measured):
    """选主目标：有中心且框面积最大者(静止单靶标时即该靶标)。"""
    cands = [m for m in measured if m["center"] is not None]
    if not cands:
        return None
    return max(cands, key=lambda m: (m["box"][2] - m["box"][0]) *
               (m["box"][3] - m["box"][1]))


# 多目标曲线配色(按 (tid, 分量) 取唯一色)
_PALETTE = [
    (0, 0, 255), (255, 0, 0), (0, 140, 255), (0, 180, 0), (255, 0, 255),
    (255, 128, 0), (128, 0, 255), (0, 200, 200), (255, 255, 0), (60, 60, 255),
]

# 频谱分段配色: 每个 30s 整段(0-30/30-60/60-90...)一种颜色, 画细曲线用
_SEG_COLORS = [
    (0, 0, 255), (255, 0, 0), (0, 150, 0), (255, 0, 255), (0, 140, 255),
    (180, 0, 180), (0, 190, 190), (255, 128, 0), (128, 128, 0), (90, 90, 90),
]


def draw_displacement_curve(per_hist, per_calib, plot_axis="dy",
                            t_lo=None, t_hi=None):
    """逐目标位移时程曲线(mm)。per_hist: tid -> [(t_sec, dx_mm, dy_mm)]。

    横轴=真实时间(秒, 与下方时频谱图共享同一时间轴); 纵轴=位移(mm)。
    每个 track_id 一条曲线, 颜色按 id 区分; 各目标独立 mm/px 标定。
    t_lo/t_hi: 共享时间范围(不传则按数据自适应), 用于上下两图对齐。
    plot_axis: "dy"(默认) / "dx" / "both"。
    """
    W, H = 640, 300
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)
    # 收集要绘制的序列: (tid, 分量名, 数组, 颜色)
    ready = [tid for tid, st in per_calib.items()
             if st.get("ready") and len(per_hist.get(tid, [])) >= 2]
    series = []
    for i, tid in enumerate(sorted(ready)):
        arr = np.array(per_hist[tid], float)
        xs = arr[:, 0]; dxs = arr[:, 1]; dys = arr[:, 2]
        if plot_axis in ("dx", "both"):
            series.append((tid, "dx", xs, dxs, _PALETTE[(2 * i) % len(_PALETTE)]))
        if plot_axis in ("dy", "both"):
            series.append((tid, "dy", xs, dys, _PALETTE[(2 * i + 1) % len(_PALETTE)]))
    if not series:
        cv2.putText(canvas, "采集中... (需先完成各目标标定)", (50, H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1, cv2.LINE_AA)
        cv2.putText(canvas, "每目标独立 mm/px 标定", (50, H // 2 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return canvas

    cv2.rectangle(canvas, (44, 12), (W - 10, H - 30), (210, 210, 210), 1)
    yall = np.concatenate([s[3] for s in series])
    ymin, ymax = float(yall.min()), float(yall.max())
    pad = max(1e-4, (ymax - ymin) * 0.15)
    ymin -= pad; ymax += pad
    # 时间轴(真实秒)跨所有目标共享, 与下方时频谱图对齐
    if t_lo is None or t_hi is None:
        t_lo = float(min(s[2][0] for s in series))
        t_hi = float(max(s[2][-1] for s in series))

    def tx(x):
        return 44 + (W - 54) * (0.0 if t_hi == t_lo else (x - t_lo) / (t_hi - t_lo))

    def ty(y):
        return 12 + (H - 42) * (1.0 - (y - ymin) / (ymax - ymin))

    for tid, comp, xs, arr, col in series:
        pts = np.column_stack([tx(xs), ty(arr)]).astype(int)
        cv2.polylines(canvas, [pts], False, col, 2)
    if ymin < 0 < ymax:
        zy = int(ty(0.0))
        cv2.line(canvas, (44, zy), (W - 10, zy), (0, 0, 0), 1)
    # 纵轴量程刻度(位移 mm)
    cv2.putText(canvas, f"{ymax:.2f}", (6, int(ty(ymax)) + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{ymin:.2f}", (6, int(ty(ymin)) + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    # 横轴时间刻度(秒)
    for tk in np.linspace(t_lo, t_hi, 6):
        xx = int(tx(tk))
        cv2.line(canvas, (xx, H - 30), (xx, H - 26), (160, 160, 160), 1)
        cv2.putText(canvas, f"{tk:.0f}", (xx - 10, H - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1, cv2.LINE_AA)
    # 标题 + 图例(每个序列一行, 色块+id+分量+当前值)
    cv2.putText(canvas, "位移时程 (mm) · 横轴=时间",
                (50, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    ly = H - 14
    for tid, comp, xs, arr, col in reversed(series):
        txt = f"id{tid}·{comp}={arr[-1]:.3f}"
        cv2.rectangle(canvas, (50, ly - 9), (62, ly + 1), col, -1)
        cv2.putText(canvas, txt, (66, ly), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 1, cv2.LINE_AA)
        ly -= 16
        if ly < 40:
            break
    return canvas


def compute_segment_fft(samples, t0, t1, fmin=0.05, fmax_cap=16.0,
                        snr_thresh=5.0):
    """对**非重叠整段** [t0, t1) 的位移样本做 FFT, 返回 (freqs, amp, f1, snr)。

    samples: list of (t_sec, val)。

    与"滑动窗"的关键区别: 段边界固定为整段 —— 0-30s / 30-60s / 60-90s ...,
    各段互不重叠、不出现 1-31 这种跨段窗; 只有某段跑满才输出该段频谱。

    流程: 取段内样本 → 等间隔重采样(网格落在样本实际跨度内, 不外推) →
    去均值(去直流) → Hann 窗 → rfft → **单边幅值谱**(mm, Hann 增益已补偿,
    正弦分量峰值≈其真实幅值)。主频 f1 = [fmin, min(fmax_cap, 奈奎斯特)] 内
    幅值最大处(三点抛物线插值减弱栅栏效应)。频率分辨率 df ≈ 1/(t1-t0)。

    **能量门限(过滤静止段假频)**: 以幅值谱中位数估计噪声地板, 峰值/噪声地板
    即信噪比 snr; 若 snr < snr_thresh(默认5倍噪声地板) 判该段"无有效振动",
    返回 None —— 不出谱/不画/不打印, 直接消除纯噪声 argmax 抓到的假频。
    段内样本实际跨度 < 段长*0.7 或样本 < 16 也返回 None(数据不足)。
    """
    if not samples:
        return None
    ts = np.array([s[0] for s in samples], float)
    ys = np.array([s[1] for s in samples], float)
    sel = (ts >= t0) & (ts < t1)
    ts_w = ts[sel]; ys_w = ys[sel]
    if len(ts_w) < 16:
        return None
    span = float(ts_w[-1] - ts_w[0])
    if span <= 0 or span < (t1 - t0) * 0.7:
        return None                       # 该段样本太少(如标定占用/中途起播)
    fs_est = len(ts_w) / span             # 实际采样率估计
    N = max(16, len(ts_w))                # 网格点数=样本数(不外推)
    tg = np.linspace(ts_w[0], ts_w[-1], N)
    yg = np.interp(tg, ts_w, ys_w)
    yg = yg - np.mean(yg)                 # 去趋势(移除直流, 避免 0Hz 尖峰)
    w = np.hanning(N)
    X = np.abs(np.fft.rfft(yg * w))
    freqs = np.fft.rfftfreq(N, d=1.0 / fs_est)
    amp = 2.0 * X / (float(np.sum(w)) + 1e-12)     # 单边幅值谱(mm)
    fhi = min(fmax_cap, fs_est / 2.0)
    band = (freqs >= fmin) & (freqs <= fhi)
    if not np.any(band):
        return None
    fb = freqs[band]; ab = amp[band]
    peak = float(ab.max())
    noise_floor = float(np.median(ab))      # 噪声地板: 幅值谱中位数(对窄带信号稳健)
    if noise_floor <= 1e-9:
        return None                         # 幅值谱近全零 → 无有效振动
    snr = peak / noise_floor                # 峰值信噪比(相对噪声地板)
    if snr < snr_thresh:
        return None                         # 能量不足, 无有效振动 → 不出谱(过滤假频)
    i = int(np.argmax(ab))
    # 主频抛物线插值(减弱栅栏效应/scalloping): 峰值不在栅格中心时会被摊到
    # 相邻两根谱线, 直接用 argmax 会有最多半个 bin(0.0167Hz@30s)的偏差 ——
    # 对 T=4μL²f₁² 意味着约 1.4% 的索力误差。三点抛物线插值把它降到 ~0.1%。
    f1 = float(fb[i])
    if 0 < i < len(ab) - 1:
        y0v, y1v, y2v = float(ab[i - 1]), float(ab[i]), float(ab[i + 1])
        denom = y0v - 2.0 * y1v + y2v
        if abs(denom) > 1e-12:
            delta = float(np.clip(0.5 * (y0v - y2v) / denom, -0.5, 0.5))
            f1 = float(fb[i] + delta * (fb[i + 1] - fb[i]))
    return freqs, amp, f1, snr


def draw_spectrum_panel(seg_spec, plot_axis="dy", win_sec=30.0,
                        fmin=0.05, fmax=None):
    """底部**频谱图**: 横轴=频率(Hz), 纵轴=幅值(mm) —— 不是 spectrogram。

    分段口径: 每 win_sec 秒**一整段**(0-30s / 30-60s / 60-90s ...)做一次 FFT,
    每段一条**细曲线**(颜色按段区分, 线宽 1), 图例标出"起-止s + 主频 f1"。
    多目标/双分量时按 (tid, comp) 分竖直子带, 各子带独立坐标系(频率轴/幅值轴)。

    seg_spec: dict (tid, comp) -> [ {k, t0, t1, freqs, amp, f1}, ... ]
    fmax: 横轴上限(Hz); None 或 <=0 时按数据自适应(取各段奈奎斯特最小值)。
    """
    W, H = 640, 320
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)
    # 选要画的子带(每 tid+comp 一条), 顺序: 先 dy 后 dx, 各按 tid
    keys = []
    for (tid, comp) in sorted(seg_spec):
        if plot_axis == "dy" and comp != "dy":
            continue
        if plot_axis == "dx" and comp != "dx":
            continue
        if seg_spec[(tid, comp)]:
            keys.append((tid, comp))
    keys.sort(key=lambda k: (k[0], 0 if k[1] == "dy" else 1))
    if not keys:
        cv2.putText(canvas, f"采集中... 满 {win_sec:.0f}s 后输出第 1 段频谱",
                    (40, H // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (80, 80, 80), 1, cv2.LINE_AA)
        cv2.putText(canvas, "分段口径: 0-30s / 30-60s / 60-90s ...(非重叠整段)",
                    (40, H // 2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 1, cv2.LINE_AA)
        return canvas

    # 频率上限: 未指定则自适应(各段奈奎斯特的最小值)
    if fmax is None or fmax <= 0:
        fmax = min(float(e["freqs"][-1]) for k in keys for e in seg_spec[k])
    fmax = float(max(fmax, fmin + 0.1))

    n_b = len(keys)
    top_m, bot_m = 30, 32
    band_h = max(84, (H - top_m - bot_m) // n_b)

    for bi, key in enumerate(keys):
        tid, comp = key
        entries = seg_spec[key]
        y0 = top_m + bi * band_h
        y1 = y0 + band_h - 8
        # 该子带幅值上限(所有段的最大幅值 × 1.15 留白)
        amax = max(float(e["amp"][e["freqs"] <= fmax].max()) for e in entries)
        amax = max(amax * 1.15, 1e-6)

        def fx(f):
            return 44 + (W - 54) * ((f - fmin) / (fmax - fmin))

        def ay(a):
            return y1 - (y1 - y0) * (a / amax)

        cv2.rectangle(canvas, (44, y0), (W - 10, y1), (210, 210, 210), 1)
        # 频率刻度(横轴)
        for fk in np.linspace(fmin, fmax, 6):
            xk = int(fx(fk))
            cv2.line(canvas, (xk, y0), (xk, y1), (236, 236, 236), 1)
            cv2.putText(canvas, f"{fk:.1f}", (xk - 10, y1 + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (90, 90, 90), 1,
                        cv2.LINE_AA)
        # 幅值刻度(纵轴)
        cv2.putText(canvas, f"{amax:.3g}", (4, y0 + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (60, 60, 60), 1, cv2.LINE_AA)
        cv2.putText(canvas, "0", (24, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.36, (60, 60, 60), 1, cv2.LINE_AA)
        # 每段一条细曲线(线宽1, 与上方时移曲线同风格)
        for e in entries:
            col = _SEG_COLORS[e["k"] % len(_SEG_COLORS)]
            band = e["freqs"] <= fmax
            pts = np.column_stack([fx(e["freqs"][band]), ay(e["amp"][band])])
            pts = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], False, col, 1, cv2.LINE_AA)
        # 子带标签(左上) + 图例(右上, 每段一行: 色线 + 段区间 + f1)
        cv2.putText(canvas, f"id{tid}·{comp}", (48, y0 + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        lx = W - 14
        ly = y0 + 13
        for e in reversed(entries):          # 最新段排最上
            col = _SEG_COLORS[e["k"] % len(_SEG_COLORS)]
            txt = f"{e['t0']:.0f}-{e['t1']:.0f}s f1={e['f1']:.3f}Hz"
            (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
            x0t = lx - tw - 20
            cv2.line(canvas, (x0t, ly - 4), (x0t + 14, ly - 4), col, 2)
            cv2.putText(canvas, txt, (x0t + 18, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 1, cv2.LINE_AA)
            ly += 13
            if ly > y1 - 4:
                break
    # 底部轴标签
    cv2.putText(canvas, "频率/Hz", (W // 2 - 20, H - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "幅值/mm", (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"频谱图: 每 {win_sec:.0f}s 一整段(非重叠) 横轴=频率 纵轴=幅值",
                (60, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 0, 0), 1,
                cv2.LINE_AA)
    return canvas


def draw_realtime(frame, measured, fps, per_calib=None):
    """在原帧上叠加**所有**目标的跟踪框、中心点与实时状态面板。
    per_calib: tid -> 标定状态, 用于按各目标独立 mm/px 显示累计位移。"""
    out = frame.copy()
    h, w = out.shape[:2]

    for m in measured:
        # 跟踪框(黄) + id/conf
        x1, y1, x2, y2 = np.round(m["box"]).astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
        id_label = f"id={m['track_id']} conf={m['conf']:.2f}"
        cv2.putText(out, id_label, (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
                    cv2.LINE_AA)
        # 中心点(红十字 + 圆点) + 坐标
        if m["center"] is not None:
            p = tuple(np.round(m["center"]).astype(int))
            cv2.drawMarker(out, p, (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
            cv2.circle(out, p, 6, (0, 0, 255), 2)
            cv2.putText(out, f"({m['center'][0]:.2f}, {m['center'][1]:.2f})",
                        (p[0] + 16, p[1] - 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2, cv2.LINE_AA)

    # 左上状态面板(黑底白字)：目标数 + fps + 每个目标的 center/dx-dy
    panel_lines = [f"targets={len(measured)}   fps={fps:.1f}"]
    for m in measured:
        if m["center"] is None:
            panel_lines.append(f"id={m['track_id']} 丢失")
            continue
        st = per_calib.get(m["track_id"]) if per_calib else None
        if st and st.get("ready"):
            # cum_* = 相对该目标标定参考帧的累计位移(绝对位移, mm) —— 时移曲线用这个
            d = (f"位移 dx={m['cum_dx_mm']:.3f} dy={m['cum_dy_mm']:.3f} mm "
                 f"(scale={st['mm_per_px']:.4f})")
        elif not np.isnan(m["dx"]):
            # dx/dy = 帧间增量(当前帧中心 - 上一帧中心, px), 类速度量, 非绝对位移
            d = f"帧间 dx={m['dx']:.3f} dy={m['dy']:.3f} px (标定中)"
        else:
            d = "dx=nan dy=nan"
        panel_lines.append(
            f"id={m['track_id']} ({m['center'][0]:.1f},{m['center'][1]:.1f}) {d}")
    ph = 22
    panel_h = ph * len(panel_lines) + 10
    cv2.rectangle(out, (8, 8), (min(w - 8, 470), 8 + panel_h), (20, 20, 20), -1)
    for i, line in enumerate(panel_lines):
        cv2.putText(out, line, (16, 28 + i * ph), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def save_step(radius_dir, name, img):
    """保存一步标定中间图到 radius_dir(文件名全英文, 规避 OpenCV 中文 imwrite 坑)。

    与 calibrate_scale.py 的 save() 对齐: 文件名形如 00_original / 02_crop /
    07_circle / 99_final, 方便逐帧走通"原图→检测→裁剪→灰度→模糊→二值→候选→选中圆"。
    这个 helper 让两个脚本的步骤图命名一致、便于交叉核对。
    """
    if img is None:
        return None
    path = Path(radius_dir) / f"{name}.jpg"
    cv2.imwrite(str(path), img)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtsp", default=DEFAULT_RTSP, help="RTSP 流地址")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="marker_subpixel_tracker 的配置文件")
    parser.add_argument("--weights", default=None,
                        help="模型路径(.pt 或 OpenVINO 目录)，默认优先 OpenVINO")
    parser.add_argument("--device", default=None,
                        help="覆盖默认 device，如 cpu / intel:gpu / intel:npu")
    parser.add_argument("--max-side", type=int, default=960,
                        help="显示时画面长边缩放到的像素数(同比例缩小，0=不缩放)")
    parser.add_argument("--calib-frames", type=int, default=8,
                        help="每个新 id 出现后, 尝试提取外圆直径的帧数(局部窗口, "
                             "非全局前N帧)")
    parser.add_argument("--calib-min", type=int, default=1,
                        help="每个新 id 攒够多少个有效外圆直径即定标 mm/px"
                             "(默认3, 不必等满 calib-frames)")
    parser.add_argument("--real-diameter-mm", type=float, default=100.0,
                        help="靶标外圆真实直径(mm)，默认 100=10cm")
    parser.add_argument("--no-calib", action="store_true",
                        help="跳过自动标定，仅像素显示")
    parser.add_argument("--mm-per-px", type=float, default=None,
                        help="手动给定 mm/px，所有目标共用并跳过自动标定")
    parser.add_argument("--csv", default="displacement_mm.csv",
                        help="位移时程 CSV 输出路径")
    parser.add_argument("--radius-dir", default="radius",
                        help="标定阶段外接圆+直径图片保存目录(默认 radius/)")
    parser.add_argument("--plot-axis", default="dy", choices=("dy", "dx", "both"),
                        help="时移曲线绘制哪个分量: dy(默认,竖向/索力相关) / "
                             "dx(横向) / both(同时画 dx+dy)")
    parser.add_argument("--fft-win", type=float, default=30.0,
                        help="FFT 分段长度(秒)，默认30；按**非重叠整段**切分: "
                             "0-30s / 30-60s / 60-90s ... 每段做一次 FFT"
                             "(段未跑满不出谱)，频率分辨率≈1/段长(30s→0.033Hz)")
    parser.add_argument("--seg-max", type=int, default=6,
                        help="频谱图最多保留/绘制最近多少段(默认6)，"
                             "每段一条细曲线，颜色按段区分")
    parser.add_argument("--fmax", type=float, default=0.0,
                        help="频谱图横轴上限(Hz)，默认0=自适应(取奈奎斯特)")
    parser.add_argument("--fft-snr", type=float, default=5.0,
                        help="FFT 能量门限(信噪比阈值, 默认5.0): 幅值谱峰值/噪声地板"
                             "(中位数) < 该值判为无有效振动, 不出谱/不画/不打印,"
                             "过滤静止段假频; 调大更严格, 调小更灵敏")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]
    tracking_cfg = cfg.get("tracking", {})
    loc_cfg = cfg.get("localization", {})

    # 权重选择：--weights 优先；否则存在 OpenVINO 模型则默认用它(Intel 加速)
    if args.weights:
        weights = Path(args.weights).resolve()
    elif OPENVINO_DIR.is_dir():
        weights = OPENVINO_DIR
    else:
        weights = resolve_weights(cfg_path, model_cfg["weights"])
    if not weights.exists():
        raise FileNotFoundError(f"模型不存在: {weights}")

    # 设备选择：--device 优先；OpenVINO 模型默认走 Intel GPU，否则用 config 的 cpu
    if args.device:
        device = args.device
    elif is_openvino_dir(weights):
        device = "intel:gpu"
    else:
        device = model_cfg["device"]

    # 初始化跟踪器(多目标，不锁定单 ID)与逐目标状态管理
    tracker = ByteTrackTracker(
        str(weights), model_cfg["confidence"], model_cfg["class_id"],
        device, tracker=tracking_cfg.get("tracker", "bytetrack.yaml"),
        persist=tracking_cfg.get("persist", True),
    )
    multi = MultiTargetTracker()

    print(f"[marker_tracing] RTSP: {args.rtsp}")
    print(f"[marker_tracing] 权重: {weights}  设备: {device}")
    print("[marker_tracing] 多目标模式：画面内每个 ByteTrack 轨迹独立定位")
    print("[marker_tracing] 显示窗口按 q 退出")

    # ---- 像素→mm 标定 与 位移时程 状态(逐目标独立) ----
    # per_calib: tid -> 标定状态(各目标独立外圆直径→mm/px, 及各自参考中心)
    # per_hist: tid -> [(frame_index, dx_mm, dy_mm)] 各自位移时程
    mm_manual = float(args.mm_per_px) if args.mm_per_px else None
    real_d = args.real_diameter_mm
    calib_frames = args.calib_frames
    calib_min = max(1, args.calib_min)
    per_calib = {}
    per_hist = {}
    # per_ts: tid -> [(t_sec, dx_mm, dy_mm)] 带真实时间戳的位移序列(FFT 用)
    per_ts = {}
    csv_path = Path(args.csv)
    csv_f = open(csv_path, "w", encoding="utf-8", newline="")
    # 标定阶段外接圆+直径图片保存目录
    radius_dir = Path(args.radius_dir)
    radius_dir.mkdir(parents=True, exist_ok=True)
    print(f"[标定] 标定帧外接圆图片将保存到 {radius_dir.resolve()}/")
    # 列说明: cx_px/cy_px=当前帧亚像素中心; dx_px/dy_px=帧间增量(=上一帧中心差);
    # dx_mm/dy_mm=相对该目标标定参考帧的累计位移(mm) —— FFT取频用这两列;
    # t_sec=当前帧真实时间(秒, 自起播), 离线等间隔重采样+FFT 用
    csv_f.write("frame,track_id,cx_px,cy_px,dx_px,dy_px,dx_mm,dy_mm,continuous,t_sec\n")
    print(f"[标定] 外圆真实直径={real_d} mm, 每个新 id 出现即局部标定"
          f"(窗口{calib_frames}帧, 攒够{calib_min}样本即定标); "
          f"--mm-per-px 给定则所有目标共用; CSV→{csv_path}")

    cap = open_stream(args.rtsp)
    frame_index = 0
    consecutive_fail = 0
    fps_ema = 0.0
    last_t = time.perf_counter()
    t_start = last_t                 # 实时起始时刻(用于 t_sec 真实时间轴)
    # 频谱: **非重叠整段**累计。seg_idx=下一个待计算的段号;
    # seg_spec: (tid, comp) -> [ {k, t0, t1, freqs, amp, f1}, ... ] 最近 seg_max 段
    seg_len = float(args.fft_win)
    seg_max = max(1, args.seg_max)
    seg_idx = 0                      # 0-30s 是第 0 段, 30-60s 是第 1 段 ...
    seg_spec = {}
    FMIN, FMAX_CAP = 0.05, 16.0      # 主频搜索下限 / 频率上限上限

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                consecutive_fail += 1
                if consecutive_fail > 30:
                    print("[marker_tracing] 断流超过 30 帧，尝试重连...")
                    cap.release()
                    time.sleep(2)
                    cap = open_stream(args.rtsp)
                    consecutive_fail = 0
                continue
            consecutive_fail = 0

            # ---- 算法链路：ByteTrack → 全部轨迹逐条定位 ----
            tracks = tracker.update(frame)
            multi.prune(frame_index)

            measured = []
            for track in tracks:
                tid = track["id"]
                box = np.asarray(track["box"], dtype=float)
                prev = multi.previous(tid)
                prev_center, prev_box = prev if prev is not None else (None, None)
                center, lambda_min, continuous = localize_one(
                    frame, box, loc_cfg, prev_center, prev_box)
                dx = dy = float("nan")
                if center is not None and multi.delta_ok(tid, continuous,
                                                         frame_index):
                    delta = center - prev_center
                    dx, dy = map(float, delta)
                measured.append({
                    "track_id": tid, "box": box, "center": center,
                    "conf": track["conf"], "lambda_min": lambda_min,
                    "continuous": continuous, "dx": dx, "dy": dy,
                })
                if center is not None:
                    multi.record(tid, center, box, frame_index)

            # ---- 实时 FPS(EMA 平滑) ----
            now = time.perf_counter()
            dt = now - last_t
            last_t = now
            t_sec = now - t_start            # 当前帧真实时间(秒), FFT 时间窗用
            inst = 1.0 / dt if dt > 0 else 0.0
            fps_ema = inst if fps_ema == 0.0 else 0.9 * fps_ema + 0.1 * inst

            # ---- 逐目标: 像素→mm 标定 + 累计位移 + 时程 ----
            for m in measured:
                m["cum_dx_mm"] = m["cum_dy_mm"] = float("nan")
                tid = m["track_id"]
                if m["center"] is None:
                    continue
                st = per_calib.setdefault(
                    tid, {"diams": [], "mm_per_px": None,
                          "ref_cx": None, "ref_cy": None, "ready": False,
                          "calib_tried": 0, "give_up": False})
                if mm_manual is not None:
                    # 手动给定比例: 所有目标共用; 原点=开始画图首帧(下方统一设)
                    st["mm_per_px"] = mm_manual
                    st["ready"] = True
                elif not args.no_calib and not st["ready"] and not st["give_up"]:
                    # 每新 id 出现后, 在其前 calib_frames 帧内尝试提取外圆直径;
                    # 攒够 calib_min 个有效直径即定标(不必等满 calib_frames),
                    # 之后该 id 永久复用此比例。margin 取检测框短边的 1/10。
                    if st["calib_tried"] < calib_frames:
                        bx1, by1, bx2, by2 = np.round(m["box"]).astype(int)
                        sbox = min(bx2 - bx1, by2 - by1)
                        margin_px = max(1, int(sbox / 10))
                        dres = measure_outer_diameter_px(
                            frame, m["box"], margin=margin_px,
                            return_debug=True)
                        cal_idx = st["calib_tried"]   # 0-based 标定帧序号
                        st["calib_tried"] += 1
                        res = dres[0] if isinstance(dres, tuple) else dres
                        dbg = dres[1] if isinstance(dres, tuple) else None
                        # ---- 全套步骤图(与 calibrate_scale.py 对齐) ----
                        # 命名: id{tid}_cal{序号}_00_original / 01_detect /
                        # 02_crop / 03_gray / 04_blur / 05_binary /
                        # 06_candidates / 07_circle / 99_final
                        # 仅在"标定窗口"(每新 id 前 calib_frames 帧)存,
                        # 不存全程, 避免实时跑产生海量图。
                        tag = f"id{tid}_cal{cal_idx:02d}"
                        save_step(radius_dir, f"{tag}_00_original", frame)
                        det_vis = frame.copy()
                        cv2.rectangle(det_vis, (bx1, by1), (bx2, by2),
                                      (0, 255, 255), 2)
                        cv2.putText(det_vis, f"id={tid}",
                                    (bx1, max(18, by1 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 255), 2, cv2.LINE_AA)
                        save_step(radius_dir, f"{tag}_01_detect", det_vis)
                        if isinstance(dbg, dict):
                            save_step(radius_dir, f"{tag}_02_crop",
                                      dbg.get("crop"))
                            save_step(radius_dir, f"{tag}_03_gray",
                                      dbg.get("gray"))
                            save_step(radius_dir, f"{tag}_04_blur",
                                      dbg.get("blurred"))
                            save_step(radius_dir, f"{tag}_05_binary",
                                      dbg.get("binary"))
                            save_step(radius_dir, f"{tag}_06_candidates",
                                      dbg.get("candidates"))
                        if res is not None:
                            d, (ccx, ccy), r = res
                            st["diams"].append(d)
                            # 07_circle: 选中圆叠加(绿圈已在 dbg["chosen"] 内,
                            # 仅补直径文字); 失败退化分支 chosen=None 用 crop 兜底
                            chosen = dbg.get("chosen") if isinstance(dbg, dict) else None
                            crop0 = dbg.get("crop") if isinstance(dbg, dict) else None
                            circle_vis = (chosen.copy() if chosen is not None
                                          else (crop0.copy() if crop0 is not None
                                                else frame.copy()))
                            cv2.putText(circle_vis, f"D={d:.2f}px r={r:.2f}",
                                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        (0, 255, 0), 2, cv2.LINE_AA)
                            save_step(radius_dir, f"{tag}_07_circle", circle_vis)
                            # 标定帧在画面画外圆(绿)+直径标注, 供人工核对
                            cv2.circle(frame, (int(ccx), int(ccy)),
                                       int(r), (0, 255, 0), 2)
                            cv2.putText(frame,
                                        f"D={d:.2f}px r={r:.2f}",
                                        (int(ccx - r), max(18, int(ccy - r) - 8)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                        (0, 255, 0), 2, cv2.LINE_AA)
                            save_step(radius_dir, f"{tag}_99_final", frame)
                            print(f"[标定] id={tid} 帧{frame_index} D={d:.2f}px "
                                  f"-> 全套步骤图 {tag}_00~99 已存")
                        else:
                            # 提取失败帧: 00~06 已存, 仅提示(无需存 crop/FAIL)
                            print(f"[标定] id={tid} 帧{frame_index} 外圆提取失败, "
                                  f"已存 {tag}_00~06 步骤图供核对")
                    # 攒够 calib_min 个有效直径即定标; 或试满窗且有样本也定标
                    if len(st["diams"]) >= calib_min:
                        mean_d = float(np.mean(st["diams"]))
                        st["mm_per_px"] = real_d / mean_d
                        st["ready"] = True
                        print(f"[标定] id={tid} 外圆像素均值直径={mean_d:.2f}px, "
                              f"真实{real_d}mm → {st['mm_per_px']:.5f} mm/px")
                    elif st["calib_tried"] >= calib_frames and st["diams"]:
                        mean_d = float(np.mean(st["diams"]))
                        st["mm_per_px"] = real_d / mean_d
                        st["ready"] = True
                        print(f"[标定] id={tid} 仅{len(st['diams'])}样本定标 "
                              f"D={mean_d:.2f}px → {st['mm_per_px']:.5f} mm/px")
                    elif st["calib_tried"] >= calib_frames:
                        # 试满窗仍 0 样本: 放弃该 id 自动标定(仅像素显示)
                        st["give_up"] = True
                        print(f"[标定] id={tid} 标定窗内未提取到外圆, "
                              f"放弃自动标定(仅像素)")
                # 就绪后: 以"开始画图首帧"为原点, 算相对原点的累计位移(mm)
                # 该帧 dy_mm=dx_mm=0; 之后各帧 = 当前中心 - 原点中心
                if st["ready"] and st["mm_per_px"] is not None:
                    if st["ref_cx"] is None:
                        # 第一帧绘图即原点(位移=0)
                        st["ref_cx"], st["ref_cy"] = map(float, m["center"])
                        dx_mm = dy_mm = 0.0
                    else:
                        dx_mm = (m["center"][0] - st["ref_cx"]) * st["mm_per_px"]
                        dy_mm = (m["center"][1] - st["ref_cy"]) * st["mm_per_px"]
                    m["cum_dx_mm"] = dx_mm
                    m["cum_dy_mm"] = dy_mm
                    per_hist.setdefault(tid, []).append(
                        (t_sec, dx_mm, dy_mm))
                    # FFT 时间序列: 带真实时间戳。整段口径下必须保留完整段,
                    # 故保留最近 (seg_max+1) 段样本, 供各段独立切片做 FFT。
                    per_ts.setdefault(tid, []).append((t_sec, dx_mm, dy_mm))
                    cut = t_sec - (seg_max + 1) * seg_len
                    per_ts[tid] = [(t, dx, dy) for (t, dx, dy) in per_ts[tid]
                                   if t >= cut]
                    cx, cy = m["center"]
                    csv_f.write(f"{frame_index},{tid},{cx:.2f},{cy:.2f},"
                                f"{m['dx']:.4f},{m['dy']:.4f},"
                                f"{dx_mm:.5f},{dy_mm:.5f},{int(m['continuous'])},"
                                f"{t_sec:.3f}\n")
                    csv_f.flush()

            # ---- 可视化(所有目标) ----
            annotated = draw_realtime(frame, measured, fps_ema, per_calib)
            # 画完框/标记点后，同比例缩小到长边 max_side(默认 640)再显示，
            # 用 INTER_AREA 避免缩小产生锯齿。计算仍在原分辨率上进行，不受影响。
            if args.max_side and args.max_side > 0:
                dh, dw = annotated.shape[:2]
                if max(dh, dw) > args.max_side:
                    s = args.max_side / float(max(dh, dw))
                    annotated = cv2.resize(
                        annotated, None, fx=s, fy=s,
                        interpolation=cv2.INTER_AREA)
            cv2.imshow("marker_tracing - RTSP", annotated)
            if any(st.get("ready") for st in per_calib.values()):
                # 上: 位移时程(横轴=时间)
                all_t = [p[0] for tid in per_hist
                         for p in per_hist[tid]] if per_hist else []
                t_lo = float(min(all_t)) if all_t else 0.0
                t_hi = float(max(t_sec, max(all_t))) if all_t else float(t_sec)
                disp = draw_displacement_curve(per_hist, per_calib, args.plot_axis,
                                               t_lo=t_lo, t_hi=t_hi)
                # ---- 非重叠整段 FFT: 段 k = [k*seg_len, (k+1)*seg_len) ----
                # 段跑满才输出该段频谱; 每段一条曲线(0-30s/30-60s/60-90s...),
                # 不是滑动窗(不会出现 1-31、2-32 这种重叠窗)。
                ready_tids = sorted(
                    tid for tid, st in per_calib.items() if st.get("ready"))
                # 断流/卡顿导致段号严重落后时直接跳段(旧段样本已裁剪, 补算无意义)
                lag = int(t_sec / seg_len)
                if lag - 1 > seg_idx:
                    seg_idx = lag - 1
                while t_sec >= (seg_idx + 1) * seg_len:
                    t0 = seg_idx * seg_len
                    t1 = (seg_idx + 1) * seg_len
                    comps = []
                    if args.plot_axis in ("dy", "both"):
                        comps.append(("dy", 1))
                    if args.plot_axis in ("dx", "both"):
                        comps.append(("dx", 0))
                    for tid in ready_tids:
                        buf = per_ts.get(tid)
                        if not buf:
                            continue
                        for comp, idx in comps:
                            ys = [(t, (dx if idx == 0 else dy))
                                  for (t, dx, dy) in buf]
                            r = compute_segment_fft(ys, t0, t1, FMIN, FMAX_CAP,
                                                    snr_thresh=args.fft_snr)
                            if r is None:
                                continue        # 段未满 / 能量不足(无有效振动) → 跳过
                            freqs, amp, f1, snr = r
                            key = (tid, comp)
                            seg_spec.setdefault(key, []).append(
                                {"k": seg_idx, "t0": t0, "t1": t1,
                                 "freqs": freqs, "amp": amp, "f1": f1})
                            if len(seg_spec[key]) > seg_max:
                                seg_spec[key] = seg_spec[key][-seg_max:]
                            if comp == "dy" or per_calib[tid].get("f1") is None:
                                per_calib[tid]["f1"] = f1
                            print(f"[FFT] id={tid} 段 {t0:.0f}-{t1:.0f}s "
                                  f"主频 f1={f1:.3f}Hz "
                                  f"(峰值幅值 {float(np.max(amp)):.4f}mm, "
                                  f"信噪比 {snr:.1f}×噪声地板, "
                                  f"索力 T=4μL²f₁²)")
                    seg_idx += 1
                spec = draw_spectrum_panel(seg_spec, args.plot_axis, seg_len,
                                           fmin=FMIN, fmax=args.fmax)
                # 上: 位移时程(横轴=时间)  下: 频谱图(横轴=频率)  拼成一张图
                combined = np.vstack([disp, spec])
                cv2.imshow("位移时程 + 频谱图", combined)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_index += 1
    finally:
        try:
            csv_f.close()
        except Exception:
            pass
        cap.release()
        cv2.destroyAllWindows()
        print(f"[marker_tracing] 结束，共处理 {frame_index} 帧；位移时程已写 {csv_path}")


if __name__ == "__main__":
    main()
