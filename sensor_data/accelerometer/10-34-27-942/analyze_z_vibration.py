# -*- coding: utf-8 -*-
"""
analyze_z_vibration.py — WT901SDCL-BT50 柔性光伏支架 加速度：预处理 + 频率提取 + 索力门控估算
=============================================================================================
运行方式（在本目录下）：
    python analyze_z_vibration.py                 # 全部数据，默认轴（AXIS 常量）
    python analyze_z_vibration.py 100             # 只分析前 100 s
    python analyze_z_vibration.py 100 Y           # 只分析前 100 s 的 Y 轴
    python analyze_z_vibration.py all Y           # 全部数据的 Y 轴
自动读取同目录 data_0.csv（缺失时回退 data_0.txt）。输出文件名带轴前缀与窗口后缀。

工程门控（防止静止/弱风噪声假峰流入索力结果）：
    G1 显著性   : 谱峰 ≥ 噪声底 × SIGNIF_RATIO，否则判 INVALID（静止数据必然有噪声谱，
                  但没有结构频率；峰值检测永远会返回"最高噪声鼓包"，必须用比值拦截）
    G2 时窗稳定 : 数据分成 4 个独立子窗分别提频，真实模态各窗一致（变异系数 ≤5%），
                  噪声假峰在各窗间随机游走
    G3 多轴同频 : 结构模态是三维运动，X/Y/Z 中至少一个其他轴在 ±15% 内出现同频峰
                  （峰/底 ≥5×）；单轴孤立峰多为该轴噪声
    G4 谐波族   : 索结构模态近似整数倍频 fₙ ≈ n·f₁，2f₁/3f₁ 处有峰（≥5×底）加分
    质量等级    : VALID（G1+G2+G3 全过）/ SUSPECT（仅 G1 过）/ INVALID（G1 不过）
    索力输出    : 仅 VALID 且配置了 CABLE_MASS_PER_LEN / CABLE_SPAN 时输出
                  T = 4·m·L²·f₁²（张紧索，忽略弯曲刚度；ΔT/T = 2·Δf/f）

预处理流水线（详见报告第二节）：均匀重采样 → 去尖峰 → 去直流(重力) → 线性去趋势 →
0.3 Hz 零相位高通 → Hann 窗 Welch PSD → 谱峰抛物线插值 + 过零率 + Hilbert 交叉验证。
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch, find_peaks, hilbert, medfilt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))

# ===================== 可调参数 =====================
AXIS         = "Z"     # 默认分析轴：X / Y / Z（可用命令行第 2 个参数覆盖）
MAX_SECONDS  = None    # 只分析前 N 秒；None = 全部（可用命令行第 1 个参数覆盖）
HP_CUTOFF    = 0.3     # 高通截止频率 (Hz)
HP_ORDER     = 2       # Butterworth 阶数
SPIKE_WIN    = 7       # 去尖峰：滚动中值窗口（点）
SPIKE_NSIG   = 6.0     # 去尖峰：稳健 σ 倍数阈值
NPERSEG      = 512     # Welch 段长（点）
OVERLAP_FRAC = 0.75    # Welch 重叠比例
N_TOP_PEAKS  = 5       # 报告列出的谱峰数
SIGNIF_RATIO = 10.0    # G1：谱峰/噪声底 显著性判据（10 dB）
NYQ_GUARD    = 0.95    # 剔除 f > NYQ_GUARD·(fs/2) 的边界伪影峰
N_WINDOWS    = 4       # G2：稳定性检验的子窗数
STAB_CV      = 0.05    # G2：各子窗频率变异系数阈值（≤5% 视为稳定）
CONFIRM_RATIO = 5.0    # G3/G4：其他轴同频峰、谐波峰的最低 峰/底 比值

# ---- 索力估算参数（保持 None 则只判频率、不输出索力）----
CABLE_MASS_PER_LEN = None   # 索线密度 m (kg/m)，如 φ20 钢绞线约 2.1 kg/m
CABLE_SPAN         = None   # 索的计算长度/跨度 L (m)
# ====================================================


def find_data_file():
    for name in ("data_0.csv", "data_0.txt"):
        p = os.path.join(HERE, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError("未找到 data_0.csv / data_0.txt")


def load_data(path):
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path, encoding="utf-8", skipinitialspace=True, index_col=False)
    else:
        df = pd.read_csv(path, sep="\t", encoding="utf-8", skipinitialspace=True,
                         index_col=False, engine="python", on_bad_lines="skip")
    df.columns = [str(c).strip() for c in df.columns]
    ct = df["片上时间()"].astype(str).str.strip()
    t = pd.to_datetime(ct.str.slice(0, 19) + "." + ct.str.slice(20),
                       format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
    ok = t.notna()
    df = df[ok].reset_index(drop=True)
    t = t[ok].reset_index(drop=True)
    t_sec = (t - t.iloc[0]).dt.total_seconds().to_numpy()
    return df, t, t_sec


def axis_series(df, axis):
    return pd.to_numeric(df[f"加速度{axis}(g)"], errors="coerce").to_numpy(float)


def resample_uniform(t_sec, x):
    dt_med = np.median(np.diff(t_sec))
    fs = 1.0 / dt_med
    gaps = int(np.sum(np.diff(t_sec) > 1.5 * dt_med))
    grid = np.arange(0.0, t_sec[-1] + 1e-9, dt_med)
    return fs, grid, np.interp(grid, t_sec, x), gaps


def despike(x):
    med = medfilt(x, kernel_size=SPIKE_WIN)
    resid = x - med
    sigma = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    mask = np.abs(resid) > SPIKE_NSIG * sigma
    xs = x.copy()
    xs[mask] = med[mask]
    return xs, int(mask.sum()), sigma


def highpass(x, fs):
    b, a = butter(HP_ORDER, HP_CUTOFF / (fs / 2.0), btype="high")
    return filtfilt(b, a, x)


def preprocess_axis(xg, tg, fs):
    """去尖峰 → 去直流 → 线性去趋势 → 零相位高通。返回 (x_pre, 统计dict)。"""
    dc = float(np.mean(xg))
    x1, n_spike, sig_mad = despike(xg)
    slope = float(np.polyfit(tg, x1, 1)[0])
    x2 = x1 - dc
    x3 = x2 - np.polyval(np.polyfit(tg, x2, 1), tg)
    x4 = highpass(x3, fs)
    return x4, dict(dc=dc, n_spike=n_spike, sig_mad=sig_mad, slope=slope)


def one_sided_amp(seg, fs, window):
    n = len(seg)
    w = np.ones(n) if window == "rect" else np.hanning(n)
    X = np.fft.rfft(seg * w)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    return f, np.abs(X) * 2.0 / np.sum(w)


def half_power_damping(f, P, i0):
    """半功率带宽法粗估阻尼比 ζ ≈ (f2-f1)/(2 f0)。低阻尼时受 Δf 分辨率限制，为上限估计。"""
    half = P[i0] / 2.0
    i1 = i0
    while i1 > 0 and P[i1] > half:
        i1 -= 1
    i2 = i0
    while i2 < len(P) - 1 and P[i2] > half:
        i2 += 1
    if i1 == 0 or i2 == len(P) - 1:
        return np.nan, np.nan
    f1 = f[i1] + (half - P[i1]) * (f[i1 + 1] - f[i1]) / (P[i1 + 1] - P[i1])
    f2 = f[i2 - 1] + (P[i2 - 1] - half) * (f[i2] - f[i2 - 1]) / (P[i2 - 1] - P[i2])
    return (f2 - f1) / (2.0 * f[i0]), f2 - f1


def refine_peak_logparabola(f, P, i0):
    df_res = f[1] - f[0]
    if 0 < i0 < len(P) - 1 and P[i0 - 1] > 0 and P[i0 + 1] > 0:
        la, lb, lc = np.log(P[i0 - 1]), np.log(P[i0]), np.log(P[i0 + 1])
        den = la - 2 * lb + lc
        if den != 0:
            return f[i0] + 0.5 * (la - lc) / den * df_res
    return f[i0]


def welch_psd(x, fs, nperseg):
    nperseg = min(nperseg, len(x))
    noverlap = int(nperseg * OVERLAP_FRAC)
    f, P = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap,
                 window="hann", detrend="constant", scaling="density")
    n_seg = max(1, (len(x) - noverlap) // (nperseg - noverlap))
    return f, P, n_seg, nperseg


def noise_floor_of(f, P, fs):
    band = (f >= max(2 * HP_CUTOFF, f[1])) & (f <= NYQ_GUARD * fs / 2)
    return float(np.median(P[band])), float(f[band][0]), float(f[band][-1])


def window_stability(x_pre, fs, f0):
    """G2：分成 N_WINDOWS 个独立子窗分别提频，返回各窗频率列表。"""
    seg = len(x_pre) // N_WINDOWS
    nps = min(256, seg)
    freqs = []
    for k in range(N_WINDOWS):
        w = x_pre[k * seg:(k + 1) * seg]
        if len(w) < max(64, nps):
            continue
        fw, Pw, _, _ = welch_psd(w, fs, nps)
        fl, _, _ = noise_floor_of(fw, Pw, fs)
        band = (fw >= 0.7 * f0) & (fw <= 1.3 * f0)
        if not band.any() or fl <= 0:
            continue
        i = int(np.argmax(np.where(band, Pw, -np.inf)))
        if Pw[i] >= CONFIRM_RATIO * fl:
            freqs.append(refine_peak_logparabola(fw, Pw, i))
    return freqs


def harmonic_check(f, P, floor, f1, fs):
    """G4：检查 2f₁、3f₁ 附近是否有 ≥CONFIRM_RATIO×底 的峰。返回 {n: 峰/底比}。"""
    out = {}
    for n in (2, 3):
        target = n * f1
        if target > NYQ_GUARD * fs / 2:
            break
        band = (f >= 0.93 * target) & (f <= 1.07 * target)
        if band.any():
            out[n] = float(P[band].max() / floor)
    return out


def multi_axis_confirm(df, t_sec_full, tg_full, keep_u, fs, axis, f0, nperseg):
    """G3：其他两轴是否在 ±15% 内出现同频显著峰。返回 {轴: (频率, 峰/底) 或 None}。"""
    res = {}
    for ax2 in ("X", "Y", "Z"):
        if ax2 == axis:
            continue
        x2 = axis_series(df, ax2)
        _, tg2, xg2, _ = resample_uniform(t_sec_full, x2)
        pre2, _ = preprocess_axis(xg2[keep_u], tg2[keep_u], fs)
        f2, P2, _, _ = welch_psd(pre2, fs, nperseg)
        fl2, _, _ = noise_floor_of(f2, P2, fs)
        band = (f2 >= 0.85 * f0) & (f2 <= 1.15 * f0)
        if band.any() and fl2 > 0:
            i2 = int(np.argmax(np.where(band, P2, -np.inf)))
            r2 = float(P2[i2] / fl2)
            res[ax2] = (float(refine_peak_logparabola(f2, P2, i2)), r2) \
                if r2 >= CONFIRM_RATIO else None
        else:
            res[ax2] = None
    return res


def main():
    # 命令行：argv[1] = 秒数或 all/none/full；argv[2] = 轴 X/Y/Z
    max_seconds = MAX_SECONDS
    axis = AXIS.upper()
    if len(sys.argv) > 1:
        arg = sys.argv[1].strip().lower()
        max_seconds = None if arg in ("all", "none", "full") else float(sys.argv[1])
    if len(sys.argv) > 2:
        axis = sys.argv[2].strip().upper()
    if axis not in ("X", "Y", "Z"):
        raise ValueError("轴参数必须是 X / Y / Z")
    prefix = f"{axis.lower()}_analysis_"

    path = find_data_file()
    df, t, t_sec_full = load_data(path)
    x_raw_full = axis_series(df, axis)
    fs, tg_full, xg_full, gaps_full = resample_uniform(t_sec_full, x_raw_full)

    if max_seconds is not None:
        keep_u = tg_full <= max_seconds
        keep_r = t_sec_full <= max_seconds
        suffix = f"_first{int(max_seconds)}s"
    else:
        keep_u = np.ones(len(tg_full), bool)
        keep_r = np.ones(len(t_sec_full), bool)
        suffix = ""
    t_sec, x_raw = t_sec_full[keep_r], x_raw_full[keep_r]
    tg, xg = tg_full[keep_u], xg_full[keep_u]
    gaps = int(np.sum(np.diff(tg) > 1.5 / fs))
    dur = tg[-1] - tg[0]

    print(f"数据文件      : {os.path.basename(path)}   分析轴: {axis}")
    print(f"窗口内样本数  : {len(x_raw)}（重采样后 {len(xg)}）  时长: {dur:.1f} s   "
          f"采样率: {fs:.2f} Hz (奈奎斯特 {fs/2:.1f} Hz)")

    # ---------- 预处理 + Welch ----------
    x_pre, st = preprocess_axis(xg, tg, fs)
    ac_rms_mg = float(np.std(x_pre)) * 1000.0
    f, P, n_seg, nperseg = welch_psd(x_pre, fs, NPERSEG)
    df_res = f[1] - f[0]
    noise_floor, f_lo_edge, f_hi_edge = noise_floor_of(f, P, fs)

    pk, _ = find_peaks(P, prominence=0.05 * P.max())
    pk = pk[f[pk] <= NYQ_GUARD * fs / 2]
    if len(pk) == 0:
        pk = np.array([int(np.argmax(np.where(
            (f >= f_lo_edge) & (f <= f_hi_edge), P, -np.inf)))])
    peaks = pk[np.argsort(P[pk])[::-1][:N_TOP_PEAKS]]

    i0 = peaks[0]
    f0_bin = f[i0]
    f0 = refine_peak_logparabola(f, P, i0)
    ratio0 = float(P[i0] / noise_floor)
    g1 = ratio0 >= SIGNIF_RATIO                       # G1 显著性
    zeta, bw = half_power_damping(f, P, i0)
    band = max(3 * df_res, bw if not np.isnan(bw) else 3 * df_res)
    sel = (f >= f0_bin - band) & (f <= f0_bin + band)
    amp_eq_mg = float(np.sqrt(2.0 * np.trapezoid(P[sel], f[sel])) * 1000.0)

    # ---------- 交叉验证（过零 + Hilbert） ----------
    lo = max(HP_CUTOFF * 1.2, f0 * 0.6)
    hi = min(fs / 2 * 0.92, f0 * 1.4)
    bb, ba = butter(2, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    xb = filtfilt(bb, ba, x_pre)
    zc_idx = np.where(np.diff(np.sign(xb)) != 0)[0]
    f_zc = (len(zc_idx) - 1) / 2.0 / dur if len(zc_idx) > 2 else np.nan
    inst = np.diff(np.unwrap(np.angle(hilbert(xb)))) / (2 * np.pi) * fs
    f_inst_med = float(np.median(inst))
    f_inst_iqr = (float(np.percentile(inst, 5)), float(np.percentile(inst, 95)))
    consistent = (abs(f_zc - f0) <= 0.15 * f0) and (abs(f_inst_med - f0) <= 0.15 * f0)

    # ---------- G2 时窗稳定性 ----------
    win_freqs = window_stability(x_pre, fs, f0) if g1 else []
    stab_cv = (float(np.std(win_freqs) / np.mean(win_freqs))
               if len(win_freqs) >= 2 else np.nan)
    g2 = (len(win_freqs) >= max(2, N_WINDOWS - 1)) and \
         (not np.isnan(stab_cv)) and stab_cv <= STAB_CV

    # ---------- G3 多轴同频 ----------
    maxis = multi_axis_confirm(df, t_sec_full, tg_full, keep_u, fs, axis, f0, nperseg) \
        if g1 else {}
    confirmed_axes = {k: v for k, v in maxis.items() if v is not None}
    g3 = len(confirmed_axes) >= 1

    # ---------- G4 谐波族 ----------
    harm = harmonic_check(f, P, noise_floor, f0, fs) if g1 else {}
    harm_found = {n: r for n, r in harm.items() if r >= CONFIRM_RATIO}

    # ---------- 质量等级 ----------
    if not g1:
        quality = "INVALID"
        verdict = (f"G1 未通过：峰/噪声底 = {ratio0:.1f}×（判据 {SIGNIF_RATIO:.0f}×）。"
                   f"数据接近静止/弱激励，交流残差 {ac_rms_mg:.2f} mg 与传感器自身噪声"
                   f"（规格 0.75~1 mg rms）同量级；{f0:.2f} Hz 只是噪声底中最高的随机"
                   f"起伏，不代表结构频率，禁止用于索力计算。")
    elif not (g2 and g3):
        quality = "SUSPECT"
        miss = []
        if not g2:
            miss.append(f"G2 时窗不稳定（{len(win_freqs)}/{N_WINDOWS} 窗有峰"
                        f"{'' if np.isnan(stab_cv) else f', 变异系数 {stab_cv*100:.1f}%'}）")
        if not g3:
            miss.append("G3 其他轴无同频峰")
        verdict = (f"G1 通过（峰/底 {ratio0:.1f}×）但 {'；'.join(miss)}。"
                   f"可能为局部干扰或单轴噪声，建议延长记录/复核安装后重判。")
    else:
        quality = "VALID"
        verdict = (f"G1~G3 全部通过：峰/底 {ratio0:.1f}×，{len(win_freqs)} 个子窗频率一致"
                   f"（变异系数 {stab_cv*100:.1f}%），其他轴同频确认"
                   f"（{'、'.join(f'{k}轴 {v[0]:.2f}Hz({v[1]:.0f}×)' for k, v in confirmed_axes.items())}）。"
                   f"{f0:.2f} Hz 可认定为结构振动频率，允许进入索力估算。")

    # ---------- 索力估算（仅 VALID 且配置了索参数） ----------
    tension_lines = []
    if CABLE_MASS_PER_LEN and CABLE_SPAN:
        if quality == "VALID":
            m, L = CABLE_MASS_PER_LEN, CABLE_SPAN
            T1 = 4.0 * m * (L * f0) ** 2          # T = 4 m L² f₁²  (N)
            tension_lines.append(f"    基频 f₁ = {f0:.3f} Hz → T = 4·m·L²·f₁² = {T1/1000:.2f} kN")
            Ts = [T1]
            for n, r in harm_found.items():
                fn_band = (f >= 0.93 * n * f0) & (f <= 1.07 * n * f0)
                fn = f[fn_band][int(np.argmax(P[fn_band]))]
                Tn = 4.0 * m * (L * fn / n) ** 2
                Ts.append(Tn)
                tension_lines.append(f"    {n} 阶 {fn:.3f} Hz（峰/底 {r:.1f}×）→ T = {Tn/1000:.2f} kN")
            tension_lines.append(f"    多阶平均 T = {np.mean(Ts)/1000:.2f} kN，"
                                 f"离散 ±{(np.max(Ts)-np.min(Ts))/2/np.mean(Ts)*100:.1f}%")
            tension_lines.append(f"    不确定度传递：ΔT/T = 2·Δf/f（频率 ±{stab_cv*100:.1f}%"
                                 f" → 索力约 ±{2*stab_cv*100:.1f}%，未含 m、L 误差）")
        else:
            tension_lines.append(f"    ★ 质量等级 {quality}，未通过门控 —— 本次不输出索力"
                                 f"（监测系统中应保持上一次 VALID 值并标记数据无效）。")
    else:
        tension_lines.append("    未配置索参数（CABLE_MASS_PER_LEN / CABLE_SPAN 为 None），"
                             "仅输出频率判定。公式：T = 4·m·L²·f₁²（张紧索，忽略弯曲刚度）。")

    # ---------- 绘图 ----------
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=150)
    win_txt = f"前 {max_seconds:.0f} s" if max_seconds is not None else "全长"

    ax = axes[0, 0]
    ax.plot(t_sec, x_raw * 1000, lw=0.6, color="#1f77b4")
    ax.axhline(st["dc"] * 1000, color="r", ls="--", lw=1,
               label=f"直流分量(重力投影) = {st['dc']*1000:.1f} mg")
    ax.set_title(f"① 原始 {axis} 轴加速度（{win_txt}）：振动叠加在巨大的重力直流分量上")
    ax.set_xlabel("时间 (s)"); ax.set_ylabel("加速度 (mg)")
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[0, 1]
    zoom = tg < min(60, dur)
    dc_removed = xg - st["dc"]
    ax.plot(tg[zoom], dc_removed[zoom] * 1000, lw=0.7, color="#999999", label="仅去直流")
    ax.plot(tg[zoom], x_pre[zoom] * 1000, lw=0.9, color="#d62728",
            label="去尖峰+去趋势+0.3Hz高通")
    ax.set_title(f"② 预处理效果（前 {tg[zoom][-1]:.0f} s）：慢漂移/准静态晃动被去除，只留振动")
    ax.set_xlabel("时间 (s)"); ax.set_ylabel("加速度 (mg)")
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    fr, ar = one_sided_amp(xg[:nperseg], fs, "rect")
    fp, ap = one_sided_amp(x_pre[:nperseg], fs, "hann")
    ax.semilogy(fr[1:], np.maximum(ar[1:], 1e-9), lw=1, color="#7f7f7f",
                label="不去直流 + 矩形窗（泄漏淹没全谱）")
    ax.semilogy(fp[1:], np.maximum(ap[1:], 1e-9), lw=1.2, color="#1f77b4",
                label="预处理 + Hann窗（干净谱峰）")
    ax.set_title("③ 为什么必须去直流并加窗：单段 FFT 幅值谱对比")
    ax.set_xlabel("频率 (Hz)"); ax.set_ylabel("幅值 (g)")
    ax.set_xlim(0, fs / 2); ax.legend(fontsize=9)

    ax = axes[1, 1]
    ax.semilogy(f, np.maximum(P, 1e-12), lw=1.2, color="#2ca02c", label="Welch PSD")
    ax.axhline(noise_floor, color="#888888", ls="--", lw=1,
               label=f"噪声底中位数 = {noise_floor:.2e}")
    ax.axhline(SIGNIF_RATIO * noise_floor, color="#ff7f0e", ls=":", lw=1.2,
               label=f"G1 判据 = {SIGNIF_RATIO:.0f}×噪声底")
    for ip in peaks[:3]:
        f_lab = refine_peak_logparabola(f, P, ip) if ip == i0 else f[ip]
        ax.annotate(f"{f_lab:.3f} Hz", (f[ip], P[ip]),
                    textcoords="offset points", xytext=(8, 6),
                    color="k", fontsize=10, arrowprops=dict(arrowstyle="->", lw=0.8))
        ax.plot(f[ip], P[ip], "rv", ms=6)
    ax.set_title(f"④ Welch PSD（{n_seg} 段平均）—— 质量等级 [{quality}]  最高峰 {f0:.3f} Hz"
                 f"（峰/底 {ratio0:.1f}×）")
    ax.set_xlabel("频率 (Hz)"); ax.set_ylabel("PSD (g²/Hz)")
    ax.set_xlim(0, fs / 2)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(f"WT901SDCL-BT50 {axis} 轴 风致振动分析（{win_txt}, fs={fs:.0f} Hz, "
                 f"{dur:.0f} s）—— 判定: {quality}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = os.path.join(HERE, f"{prefix}overview{suffix}.png")
    fig.savefig(fig_path)
    plt.close(fig)

    # ---------- CSV ----------
    pd.DataFrame({"f_Hz": f, "PSD_g2perHz": P}).to_csv(
        os.path.join(HERE, f"{prefix}psd{suffix}.csv"), index=False)
    pd.DataFrame({
        "time_s": tg,
        "datetime": pd.to_datetime(t.iloc[0]) + pd.to_timedelta(tg, unit="s"),
        f"accel_{axis}_raw_g": xg,
        f"accel_{axis}_preprocessed_g": x_pre,
    }).to_csv(os.path.join(HERE, f"{prefix}preprocessed{suffix}.csv"), index=False)

    # ---------- 报告 ----------
    peak_lines = [f"    f = {f[ip]:.3f} Hz   PSD = {P[ip]:.3e} g²/Hz   峰/底 = {P[ip]/noise_floor:.1f}×"
                  for ip in peaks]
    win_line = (f"前 {max_seconds:.0f} s（全记录 {tg_full[-1]:.1f} s 中截取）"
                if max_seconds is not None else "全部数据")
    zeta_txt = f"{zeta:.3f}" if not np.isnan(zeta) else "N/A"
    wf_txt = ("、".join(f"{v:.3f}" for v in win_freqs) if win_freqs else "无")
    ax_txt = ("；".join(f"{k}轴: {v[0]:.3f} Hz（峰/底 {v[1]:.1f}×）" if v else f"{k}轴: 无同频峰"
                        for k, v in maxis.items()) if maxis else "未检验（G1 未过）")
    harm_txt = ("；".join(f"{n}f₁: 峰/底 {r:.1f}×{'√' if r >= CONFIRM_RATIO else '×'}"
                          for n, r in harm.items()) if harm else "未检验（G1 未过）")
    report = f"""WT901SDCL-BT50 {axis} 轴加速度 风致振动分析报告（含索力门控）
数据文件: {os.path.basename(path)}   分析窗口: {win_line}
================================================================
一、数据概况
    窗口内样本数      : {len(x_raw)}（均匀重采样后 {len(xg)}）
    推断采样率        : {fs:.2f} Hz（片上时间间隔 {1000/fs:.0f} ms）
    可用频带          : 0 ~ {fs/2:.1f} Hz（奈奎斯特）
    分析时长          : {dur:.1f} s   丢帧间隙: {gaps} 处（已插值补齐）

二、预处理量化证据
    直流分量(重力投影): {st['dc']*1000:.1f} mg（是交流残差 {ac_rms_mg:.2f} mg 的
                        {abs(st['dc'])/max(np.std(x_pre),1e-12):.0f} 倍，必须先扣除）
    单点尖峰          : 替换 {st['n_spike']} 点（σ_MAD={st['sig_mad']*1000:.2f} mg）
    线性趋势斜率      : {st['slope']*1000*60:.2f} mg/min
    {HP_CUTOFF} Hz 零相位高通 / Hann 窗 / Welch {n_seg} 段平均（Δf={df_res*1000:.1f} mHz）

三、频率提取
    ★ 质量等级        : 【{quality}】
    ★ 自动判定        : {verdict}
    噪声底            : {noise_floor:.2e} g²/Hz（{f_lo_edge:.1f}~{f_hi_edge:.1f} Hz PSD 中位数）
    最高谱峰 f0       : {f0:.3f} Hz（峰值 {P[i0]:.3e}，峰/底 {ratio0:.1f}×）
    等效正弦幅值      : ≈ {amp_eq_mg:.2f} mg    阻尼比 ζ(上限粗估): {zeta_txt}
    过零率 / Hilbert  : {f_zc:.3f} Hz / {f_inst_med:.3f} Hz（一致: {'是' if consistent else '否'}）
    前 {len(peaks)} 个谱峰        :
{chr(10).join(peak_lines)}

四、工程门控明细
    G1 显著性(≥{SIGNIF_RATIO:.0f}×) : {'通过' if g1 else '未通过'}（实测 {ratio0:.1f}×）
    G2 时窗稳定     : {'通过' if g2 else '未通过'}（{len(win_freqs)}/{N_WINDOWS} 子窗有峰: {wf_txt} Hz
                      {'，变异系数 %.1f%%' % (stab_cv*100) if not np.isnan(stab_cv) else ''}）
    G3 多轴同频     : {'通过' if g3 else '未通过'}（{ax_txt}）
    G4 谐波族(加分) : {harm_txt}

五、索力估算（门控输出）
{chr(10).join(tension_lines)}

六、结论与提醒
    1. 只有质量等级 VALID 的频率才允许换算索力；SUSPECT/INVALID 时监测系统应保持
       上一次 VALID 值并标注"数据无效/无激励"，绝不用噪声假峰刷新索力。
    2. 索力对频率误差极其敏感：ΔT/T = 2·Δf/f。若把静止噪声假峰当作模态频率，
       索力误差按平方放大（如频率高估 60% → 索力高估约 1.6² ≈ 2.6 倍）。
    3. 本记录采样率 {fs:.0f} Hz：如需识别 >{fs/2:.0f} Hz 高阶模态或多阶索力反演，请把输出速率调至 200 Hz。
    4. 建议每次部署先录一段静止数据作噪声底基准（本底 {noise_floor:.2e} g²/Hz），
       并定期复测（噪声底随温度/老化缓变）。
"""
    rep_path = os.path.join(HERE, f"{prefix}report{suffix}.txt")
    with open(rep_path, "w", encoding="utf-8") as fh:
        fh.write(report)

    print()
    print(report)
    print(f"输出文件:\n  {fig_path}\n  {rep_path}")


if __name__ == "__main__":
    main()
