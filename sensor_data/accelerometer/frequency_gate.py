# -*- coding: utf-8 -*-
"""四门控频率计算（WT901SDCL-BT50 加速度时程）。

独立脚本，不依赖 scipy / pandas / flask，只用 numpy + 标准库。
被 gate_server.py 以模块方式调用，也可直接命令行运行：

    .venv/Scripts/python.exe frequency_gate.py --file data_0.csv
    .venv/Scripts/python.exe frequency_gate.py --file data_0.txt --axis "加速度Z(g)" --json out.json

四道门（全部计算、逐门展示，不短路）：
  G0 幅度门   : 去趋势后 RMS >= g0_rms_factor × 本底 RMS
  G1 显著性门 : Welch 带内 max(P)/median(P) >= g1_peak_ratio
  G2 稳定性门 : 四个子窗主频 CV <= g2_cv_max
  G3 频率下限 : f1 > g3_factor / 窗实际跨度，且 f1 >= f1_min_hz

只有四门全过（VALID）才刷新索力；INVALID 保持上一次 VALID 并标注时效。
索力是状态量，不是瞬时量。
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import io
import json
import math
import os
import re
import sys
from typing import Any

import numpy as np

# ---------------------------------------------------------------- 默认参数
DEFAULTS: dict[str, Any] = {
    "axis": "",                 # 空 = 自动选含「加速度Z」的列
    "window_s": 30.0,           # 固定不重叠窗长（与主链路 seg=int(t//W) 一致）
    "fs": 0.0,                  # 0 = 从时间戳自动估计
    "unit_scale": 0.0,          # 0 = 自动：加速度 g→mg(×1000)，其余 ×1
    "detrend": True,            # 每窗去均值 + 线性去趋势
    "f_lo": 0.3,                # 分析带下限 Hz（拦温漂慢波 / 准静态抬升）
    "f_hi": 4.7,                # 分析带上限 Hz（10Hz 采样 Nyquist=5）
    "nperseg_s": 8.0,           # Welch 单段长度（秒）
    "n_sub": 4,                 # G2 子窗数
    "g0_rms_factor": 1.2,       # G0 倍数
    "noise_floor_mg": 1.286,    # 本底 RMS；0 = 自动用各窗 RMS 的 10 分位数
    "g1_peak_ratio": 10.0,      # G1 峰/中位数
    "g2_cv_max": 0.08,          # G2 CV 上限（8%）
    "g3_factor": 2.0,           # G3 f1 > g3_factor / T
    "f1_min_hz": 0.0,           # G3 附加绝对下限
    "n_peaks": 6,               # 输出显著峰个数
    "peak_min_ratio": 3.0,      # 各阶峰相对谱中位数的最小倍数
    "min_samples": 32,          # 少于该样本数的窗不参与判定
    "min_span_s": 10.0,         # 窗实际跨度不足则跳过
    "L_m": 33.3,                # 索跨长（绝对索力用，需现场标定）
    "mu_kg_m": 1.0,             # 线密度 kg/m（绝对索力用，需现场标定）
    "f1_ref_hz": 1.979,         # 基准一阶频率（相对索力指数用）
    # ---- AZV 吸收增强（2026-11-13）----
    "despike": True,            # 去尖峰：滚动中值 + MAD 稳健 σ
    "spike_win": 7,             # 去尖峰窗口（点）
    "spike_nsig": 6.0,          # 去尖峰 σ 倍数
    "hp_cutoff": 0.3,           # 零相位高通截止 Hz；0=关闭（Butterworth 2 阶 filtfilt）
    "refine_f1": True,          # f1 用 log 抛物线亚格点插值
    "g2_mode": "band",          # G2 模式：band=带内验峰(AZV) / full=全带argmax(旧)
    "g2_band_frac": 0.3,        # band 模式：子窗搜索带 f0±30%
    "g2_sub_ratio": 5.0,        # band 模式：子窗带内峰 >= 5×该子窗谱中位
    "g2_min_subs": 3,           # band 模式：至少 N 个子窗有有效峰才计算 CV
    "multi_axis": True,         # 多轴同频确认（软确认，不否决 VALID）
    "ma_band_frac": 0.15,       # 多轴搜索带 f0±15%
    "ma_ratio": 5.0,            # 其他轴带内峰 >= 5×该轴谱中位
    "harmonic_check": True,     # 谐波族检查 2f1/3f1（加分证据）
    "harm_ratio": 5.0,          # 谐波峰 >= 5×谱中位
    "xcheck": True,             # 过零率 + Hilbert 瞬时频率交叉验证
}

PARAM_KEYS = tuple(DEFAULTS.keys())


# ---------------------------------------------------------------- 标定参数加载
def load_calibration(src: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """读取 calibrate_accel.py 产出的标定文件 → (生产参数, 元信息)。

    src 可为文件路径或已解析的 dict。只做校验与告警，不因告警拒绝加载
    （现场不能因少填一项就整链停摆），但所有告警会随结果返回给前端。
    """
    if isinstance(src, dict):
        calib, path = src, "<inline>"
    else:
        path = str(src)
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
        with open(path, "r", encoding="utf-8") as fh:
            calib = json.load(fh)

    warnings: list[str] = []
    if int(calib.get("schema_version", 0)) != 1:
        warnings.append(f"标定文件 schema_version={calib.get('schema_version')}，本工具期望 1")
    params = dict(calib.get("params") or {})
    bad = [k for k in params if k not in DEFAULTS]
    if bad:
        warnings.append(f"标定含非生产字段，已忽略：{bad}")
        params = {k: v for k, v in params.items() if k in DEFAULTS}

    v = calib.get("validity") or {}
    until = v.get("valid_until")
    if until:
        try:
            if _dt.date.fromisoformat(str(until)) < _dt.date.today():
                warnings.append(f"标定已于 {until} 过期，必须复标")
        except ValueError:
            warnings.append(f"valid_until 格式非法：{until}")
    pending = [str(it.get("key")) for it in (calib.get("site_required") or [])
               if it.get("value") in (None, "")]
    if pending:
        warnings.append("L3 现场量未标定（" + ", ".join(pending) +
                        "）：当前索力/挠度输出仅为占位值，禁止用于判定")
    failed = list((calib.get("checks") or {}).get("failed") or [])
    if failed:
        warnings.append("标定报告存在 FAIL 项：" + "; ".join(map(str, failed)))

    meta: dict[str, Any] = {
        "path": path,
        "schema_version": calib.get("schema_version"),
        "sensor": calib.get("sensor") or {},
        "noise_floor_mg": params.get("noise_floor_mg"),
        "calibrated_on": v.get("calibrated_on"), "valid_until": until,
        "source": calib.get("source") or [],
        "site_pending": pending,
        "fingerprint": calib.get("fingerprint") or {},
        "confusion_band_hz": ((calib.get("levels") or {}).get("L2_thresholds") or {}
                              ).get("confusion_band_hz"),
        "warnings": warnings, "runtime_warnings": [],
    }
    return params, meta


def check_runtime_calibration(p: dict[str, Any], fs: float, axis: str,
                              unit_scale: float) -> None:
    """运行时指纹比对：换传感器 / 速率 / 量程 / 轴 / 谱参数后必须告警，不静默放行。"""
    meta = p.get("_calib")
    if not isinstance(meta, dict) or meta.get("error"):
        return
    fp = meta.get("fingerprint") or {}
    w: list[str] = []
    rate = float(fp.get("rate_hz") or 0)
    if rate > 0 and abs(fs - rate) / rate > 0.05:
        w.append(f"采样率 {fs:.3f}Hz 与标定时 {rate:.3f}Hz 偏差 >5%，本底与 G1 失效")
    if axis and fp.get("axis_default") and axis != fp["axis_default"]:
        w.append(f"当前轴 {axis} ≠ 标定轴 {fp['axis_default']}：本底/阈值按标定轴测定，换轴需重标")
    cols = list(fp.get("axis_columns") or [])
    if cols and axis not in cols:
        w.append(f"主标定轴 {axis} 不在标定文件的加速度列 {cols} 中，可能换了设备/列定义")
    us = float(fp.get("unit_scale") or 0)
    if us > 0 and abs(float(unit_scale) - us) > 1e-6:
        w.append(f"单位换算 {unit_scale} ≠ 标定时 {us}，量程/单位可能已改")
    for key, why in (("window_s", "G2 子窗数与 G3 下限失效"),
                     ("nperseg_s", "带内格点数变化，G1 阈值必须重标")):
        ref = float(fp.get(key) or p[key])
        if abs(float(p[key]) - ref) > 1e-6:
            w.append(f"{key}={p[key]} ≠ 标定时 {ref}：{why}")
    meta["runtime_warnings"] = w


def merge_params(raw: dict | None) -> dict[str, Any]:
    """前端传来的参数与默认值合并，并做类型/范围防御。

    raw["calib"] = 标定文件路径或标定 dict 时，标定参数作为默认基底；
    raw 中其余键（用户手填）优先级更高，可显式覆盖。
    """
    p = dict(DEFAULTS)
    # 幂等：analyze_file/CLI 会把已合并的参数再传进来一次，
    # 必须透传已解析的标定元信息，否则运行期指纹校验会在第二道合并里丢失。
    if raw and isinstance(raw.get("_calib"), dict):
        p["_calib"] = dict(raw["_calib"])
    if raw and raw.get("calib"):
        try:
            cparams, cmeta = load_calibration(raw["calib"])
            p.update(cparams)
            p["_calib"] = cmeta
        except Exception as exc:
            p["_calib"] = {"error": f"标定加载失败：{exc}",
                           "warnings": [str(exc)], "runtime_warnings": []}
    if not raw:
        return p
    for k, v in raw.items():
        if k in ("calib", "_calib") or k not in DEFAULTS or v is None or v == "":
            continue
        try:
            if isinstance(DEFAULTS[k], bool):
                p[k] = str(v).lower() in ("1", "true", "yes", "on")
            elif isinstance(DEFAULTS[k], float):
                p[k] = float(v)
            elif isinstance(DEFAULTS[k], int):
                p[k] = int(float(v))
            else:
                p[k] = str(v)
        except (TypeError, ValueError):
            continue
    # 硬边界，避免前端乱填导致崩溃
    p["window_s"] = max(5.0, min(600.0, p["window_s"]))
    p["f_lo"] = max(0.0, p["f_lo"])
    p["f_hi"] = max(p["f_lo"] + 0.05, min(100.0, p["f_hi"]))
    p["nperseg_s"] = max(1.0, min(120.0, p["nperseg_s"]))
    p["n_sub"] = max(2, min(12, p["n_sub"]))
    p["n_peaks"] = max(1, min(20, p["n_peaks"]))
    p["g2_cv_max"] = max(0.001, min(5.0, p["g2_cv_max"]))
    p["g1_peak_ratio"] = max(1.0, p["g1_peak_ratio"])
    p["g0_rms_factor"] = max(0.0, p["g0_rms_factor"])
    p["g3_factor"] = max(0.0, p["g3_factor"])
    p["min_samples"] = max(4, p["min_samples"])
    if p.get("g2_mode") not in ("band", "full"):
        p["g2_mode"] = "band"
    p["spike_win"] = max(3, min(31, int(p["spike_win"])))
    p["spike_nsig"] = max(2.0, min(50.0, p["spike_nsig"]))
    p["hp_cutoff"] = max(0.0, p["hp_cutoff"])
    p["g2_band_frac"] = max(0.05, min(1.0, p["g2_band_frac"]))
    p["g2_sub_ratio"] = max(1.0, min(100.0, p["g2_sub_ratio"]))
    p["g2_min_subs"] = max(2, min(12, p["g2_min_subs"]))
    p["ma_band_frac"] = max(0.02, min(0.5, p["ma_band_frac"]))
    p["ma_ratio"] = max(1.0, min(100.0, p["ma_ratio"]))
    p["harm_ratio"] = max(1.0, min(100.0, p["harm_ratio"]))
    return p


# ---------------------------------------------------------------- 表格解析
_CHIP_RE = re.compile(
    r"^\s*\d{4}-\d{1,2}-\d{1,2}\s+(\d{1,2}):(\d{1,2}):(\d{1,2})[:.](\d{1,3})\s*$")
_HOST_RE = re.compile(
    r"^\s*(\d{1,2}):(\d{1,2}):(\d{1,2})[.:](\d{1,3})\s*$")


def parse_table(text: str) -> tuple[list[str], list[list[str]]]:
    """解析 csv / txt（BOM + 逗号或制表符 + 末尾多余空列）。"""
    text = text.lstrip("\ufeff")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError("文件内容不足（至少要有表头和一行数据）")
    delim = "\t" if "\t" in lines[0] else ","
    reader = csv.reader(io.StringIO("\n".join(lines)), delimiter=delim)
    rows = [[c.strip() for c in r] for r in reader if r and any(c.strip() for c in r)]
    header = rows[0]
    while header and not header[-1]:
        header.pop()
    data = [r[:len(header)] + [""] * max(0, len(header) - len(r)) for r in rows[1:]]
    return header, data


def numeric_columns(header: list[str], data: list[list[str]]) -> list[str]:
    """挑出可稳定解析为数值的列（排除时间、设备名等非数值列）。"""
    out: list[str] = []
    probe = data[:min(200, len(data))]
    for i, h in enumerate(header):
        if not h:
            continue
        ok = 0
        for r in probe:
            try:
                float(r[i])
                ok += 1
            except (ValueError, IndexError):
                pass
        if ok >= max(5, int(len(probe) * 0.8)):
            out.append(h)
    return out


def _time_col(header: list[str]) -> tuple[int | None, str]:
    for i, h in enumerate(header):
        if "片上时间" in h:          # 精确 100ms 栅格，优先
            return i, "chip"
    for i, h in enumerate(header):
        if "时间" in h:
            return i, "host"
    return None, "index"


def _to_seconds(s: str, kind: str) -> float | None:
    if kind == "chip":
        m = _CHIP_RE.match(s)
        if m:
            h, mi, sec, ms = m.groups()
            return int(h) * 3600 + int(mi) * 60 + int(sec) + int(ms) / 1000.0
    m = _HOST_RE.match(s)
    if m:
        h, mi, sec, ms = m.groups()
        return int(h) * 3600 + int(mi) * 60 + int(sec) + int(ms) / 1000.0
    return None


def build_series(header: list[str], data: list[list[str]], axis: str,
                 p: dict[str, Any],
                 diag: dict[str, Any] | None = None
                 ) -> tuple[np.ndarray, np.ndarray, float, str]:
    """返回 (均匀时间轴, 数值序列, fs, 单位名)。

    diag 可选：回填 bad_timestamp_rows / clock_wraps，便于前端提示数据质量。
    时间戳解析失败时不再用 sample index 顶替（会触发伪跨天回绕），
    改用相邻有效时间戳按样本序号线性插值。
    """
    if axis not in header:
        raise ValueError(f"找不到列：{axis}")
    ai = header.index(axis)
    ti, kind = _time_col(header)
    scale = p["unit_scale"] if p["unit_scale"] > 0 else (1000.0 if "加速度" in axis else 1.0)
    unit = "mg" if "加速度" in axis and scale == 1000.0 else ("g" if "加速度" in axis else "原单位")

    tv: list[float | None] = []
    yv: list[float] = []
    for k, r in enumerate(data):
        try:
            val = float(r[ai])
        except (ValueError, IndexError):
            continue
        yv.append(val * scale)
        sec = _to_seconds(r[ti], kind) if ti is not None else None
        tv.append(sec)
    if len(yv) < 8:
        raise ValueError("有效数值样本太少（<8），无法分析")

    # 时间戳解析失败 = 整包损坏（实测该行加速度 14.167g，其余行 ~1g）。
    # 直接剔除该行，绝不退回 sample index：sample index ~0..N 与片上时间 ~5e4
    # 相邻会造成 d < -3600 的伪跨天回绕，把 304s 数据拉成 86704s。
    good = [i for i, s in enumerate(tv) if s is not None]
    bad_rows = [i for i, s in enumerate(tv) if s is None]
    yv_arr = np.asarray(yv, dtype=float)
    if len(good) >= 2:
        idx = np.asarray(good, dtype=int)
        t = np.asarray([tv[i] for i in good], dtype=float)
        y = yv_arr[idx]
    else:
        # 整列时间都解析不了：退回样本序号，fs 由参数给定时仍是正确的等间隔栅格
        t = np.arange(len(tv), dtype=float)
        y = yv_arr

    # 跨天回绕修正：只认接近 86400s 的回跳（23:59:59 → 00:00:00）
    # 原来的 d < -3600 过宽，任何坏时间戳造成的负跳变都会被误判成跨天。
    d = np.diff(t)
    wrap = (d < -80000.0) & (d > -90000.0)
    n_wrap = int(wrap.sum())
    if n_wrap:
        t = t + np.concatenate(([0.0], np.cumsum(np.where(wrap, 86400.0, 0.0))))
    if diag is not None:
        diag["bad_timestamp_count"] = len(bad_rows)
        diag["bad_timestamp_rows"] = bad_rows[:50]
        diag["clock_wraps"] = n_wrap
    order = np.argsort(t, kind="stable")
    t, y = t[order], y[order]
    keep = np.concatenate(([True], np.diff(t) > 1e-6))
    t, y = t[keep], y[keep]

    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        raise ValueError("时间戳异常（中位数间隔 <= 0）")
    fs = p["fs"] if p["fs"] > 0 else 1.0 / dt
    # 等间隔重采样：FFT 的前提
    n = max(8, int(round((t[-1] - t[0]) * fs)) + 1)
    tu = t[0] + np.arange(n) / fs
    yu = np.interp(tu, t, y)
    return tu, yu, float(fs), unit


# ---------------------------------------------------------------- 信号与谱
def detrend(y: np.ndarray) -> np.ndarray:
    """去均值 + 去线性趋势（消温漂慢波与准静态抬升的一阶分量）。"""
    x = np.arange(len(y), dtype=float)
    a = np.vstack([x, np.ones_like(x)]).T
    coef = np.linalg.lstsq(a, y, rcond=None)[0]
    return y - a @ coef


# ------------------------------------------------ AZV 吸收：稳健预处理件
def medfilt_clamp(x: np.ndarray, k: int) -> np.ndarray:
    """滑动中值（边界 clamp），等效 scipy.signal.medfilt 的工程近似。"""
    if k <= 1 or len(x) < k:
        return x.copy()
    pad = k // 2
    idx = np.clip(np.arange(len(x))[:, None] + np.arange(-pad, pad + 1)[None, :],
                  0, len(x) - 1)
    return np.median(x[idx], axis=1)


def despike(x: np.ndarray, win: int, nsig: float) -> tuple[np.ndarray, int]:
    """中值残差 + MAD 稳健 σ 剔单点尖峰，超阈点回填中值。"""
    if win <= 2 or len(x) < win + 2:
        return x.copy(), 0
    med = medfilt_clamp(x, win)
    resid = x - med
    mad = float(np.median(np.abs(resid - np.median(resid))))
    sigma = 1.4826 * mad
    if sigma <= 0:
        return x.copy(), 0
    mask = np.abs(resid) > nsig * sigma
    xs = x.copy()
    xs[mask] = med[mask]
    return xs, int(mask.sum())


def _butter_hp2(fc: float, fs: float) -> tuple[tuple[float, float, float],
                                               tuple[float, float, float]]:
    """二阶 Butterworth 高通双二阶节系数（预扭曲双线性变换，Q=1/√2）。"""
    om = math.tan(math.pi * fc / fs)
    d = 1.0 + math.sqrt(2.0) * om + om * om
    b0 = 1.0 / d
    a1 = 2.0 * (om * om - 1.0) / d
    a2 = (1.0 - math.sqrt(2.0) * om + om * om) / d
    return (b0, -2.0 * b0, b0), (1.0, a1, a2)


def _biquad_fwd(x: np.ndarray, b, a) -> np.ndarray:
    y = np.empty_like(x)
    xm1 = xm2 = ym1 = ym2 = 0.0
    # 零初态 + 输入起始值预充，减弱启动瞬态
    xm1 = xm2 = float(x[0])
    for i, xv in enumerate(x):
        yv = b[0] * xv + b[1] * xm1 + b[2] * xm2 - a[1] * ym1 - a[2] * ym2
        xm2, xm1 = xm1, xv
        ym2, ym1 = ym1, yv
        y[i] = yv
    return y


def highpass_filtfilt(x: np.ndarray, fc: float, fs: float) -> np.ndarray:
    """零相位高通：odd-padding + 正反向双二阶（等效 filtfilt）。"""
    if fc <= 0 or fc >= fs / 2 or len(x) < 8:
        return x.copy()
    b, a = _butter_hp2(fc, fs)
    n = len(x)
    npad = int(min(max(8.0, 3.0 * fs / fc), n // 2))
    left = 2.0 * x[0] - x[npad:0:-1]
    right = 2.0 * x[-1] - x[-2:-npad - 2:-1]
    xp = np.concatenate([left, x, right])
    yf = _biquad_fwd(xp, b, a)
    yr = _biquad_fwd(yf[::-1], b, a)
    return yr[::-1][npad:npad + n]


def hilbert(x: np.ndarray) -> np.ndarray:
    """解析信号（FFT 法，等效 scipy.signal.hilbert 返回的复序列）。"""
    n = len(x)
    Xf = np.fft.fft(x)
    H = np.zeros(n)
    if n % 2 == 0:
        H[1:n // 2] = 2.0
        H[n // 2] = 1.0
    else:
        H[1:(n + 1) // 2] = 2.0
    return np.fft.ifft(Xf * H)


def refine_logparabola(f: np.ndarray, P: np.ndarray, i0: int) -> float:
    """log 域三点抛物线插值，亚格点主频（AZV refine_peak_logparabola）。"""
    if 0 < i0 < len(P) - 1 and P[i0 - 1] > 0 and P[i0 + 1] > 0:
        la, lb, lc = math.log(float(P[i0 - 1])), math.log(float(P[i0])), math.log(float(P[i0 + 1]))
        den = la - 2 * lb + lc
        if den != 0:
            df = float(f[1] - f[0])
            return float(f[i0] + 0.5 * (la - lc) / den * df)
    return float(f[i0])


def preprocess_window(y: np.ndarray, fs: float, p: dict[str, Any]) -> tuple[np.ndarray, int]:
    """窗内统一预处理链：去尖峰 → 去趋势 → 零相位高通。返回 (序列, 剔除点数)。"""
    n_spike = 0
    x = y
    if p["despike"]:
        x, n_spike = despike(x, int(p["spike_win"]), float(p["spike_nsig"]))
    if p["detrend"] and len(x) >= 4:
        x = detrend(x)
    elif len(x):
        x = x - x.mean()
    if p["hp_cutoff"] > 0:
        x = highpass_filtfilt(x, float(p["hp_cutoff"]), fs)
    return x, n_spike


def welch_psd(x: np.ndarray, fs: float, nperseg: int) -> tuple[np.ndarray, np.ndarray] | None:
    """手写 Welch（Hanning 窗 + 50% 重叠平均），不依赖 scipy。"""
    n = len(x)
    nperseg = int(min(max(8, nperseg), n))
    if nperseg < 8:
        return None
    step = max(1, nperseg // 2)
    w = np.hanning(nperseg)
    scale = fs * float(np.sum(w * w))
    acc: np.ndarray | None = None
    cnt = 0
    for start in range(0, n - nperseg + 1, step):
        seg = x[start:start + nperseg]
        seg = seg - seg.mean()
        power = (np.abs(np.fft.rfft(seg * w)) ** 2) / scale
        acc = power if acc is None else acc + power
        cnt += 1
    if acc is None or cnt == 0:
        return None
    f = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    return f, acc / cnt


def _downsample(x: list[float], y: list[float], limit: int = 260):
    if len(x) <= limit:
        return x, y
    idx = np.linspace(0, len(x) - 1, limit).astype(int)
    return [x[i] for i in idx], [y[i] for i in idx]


def _band_peak(f: np.ndarray, p: np.ndarray, lo: float, hi: float):
    band = (f >= lo) & (f <= hi)
    if not band.any():
        return None, 0.0, 0.0
    idx = np.where(band)[0]
    pb = p[band]
    k = idx[int(np.argmax(pb))]
    med = float(np.median(pb)) if pb.size else 0.0
    ratio = float(p[k] / med) if med > 0 else float("inf")
    return float(f[k]), ratio, med


def local_peaks(f: np.ndarray, p: np.ndarray, lo: float, hi: float,
                med: float, min_ratio: float, f1: float, n_max: int) -> list[dict]:
    """带内局部极大 → 各阶候选频率。"""
    if med <= 0 or not math.isfinite(med):
        return []
    idx = [i for i in range(1, len(p) - 1)
           if lo <= f[i] <= hi and p[i] > p[i - 1] and p[i] >= p[i + 1]
           and p[i] / med >= min_ratio]
    if f1 and f1 > 0:
        i1 = int(np.argmin(np.abs(f - f1)))
        if i1 not in idx:
            idx.append(i1)
    idx = sorted(idx, key=lambda i: p[i], reverse=True)[:n_max]
    out = []
    for i in sorted(idx, key=lambda i: f[i]):
        order = f[i] / f1 if f1 and f1 > 0 else 0.0
        out.append({"f_hz": round(float(f[i]), 4),
                    "power": float(p[i]),
                    "ratio": round(float(p[i] / med), 2),
                    "order": round(float(order), 3),
                    "n_est": int(round(order)) if order > 0 else None})
    return out


def window_rms(y: np.ndarray, do_detrend: bool) -> float:
    x = detrend(y) if do_detrend and len(y) >= 4 else y - y.mean()
    return float(np.sqrt(np.mean(x * x)))


# ---------------------------------------------------------------- 单窗分析
def analyze_window(seg: int, t: np.ndarray, y: np.ndarray, fs: float,
                   p: dict[str, Any], noise_floor: float,
                   other_axes: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
    span = float(t[-1] - t[0]) + 1.0 / fs
    rec: dict[str, Any] = {
        "idx": int(seg), "t0": round(float(t[0]), 3), "t1": round(float(t[-1]), 3),
        "n": int(len(y)), "span_s": round(span, 3), "skipped": False,
        "g0": False, "g1": False, "g2": False, "g3": False, "valid": False,
        "f1_hz": None, "f1_raw_hz": None, "cv": None, "sub_freqs": [],
        "sub_n_valid": 0, "peaks": [],
        "welch": {"freq": [], "power": []},
        "rms_mg": None, "noise_floor_mg": round(float(noise_floor), 4),
        "g0_threshold": None, "peak_ratio": None, "g3_threshold_hz": None,
        "cable_force_idx": None, "cable_force_n": None,
        "n_spike": 0, "confirmed": False, "confirm": None,
        "reason": "",
    }
    if len(y) < p["min_samples"] or span < p["min_span_s"]:
        rec["skipped"] = True
        rec["reason"] = "样本不足或窗实际跨度不足，跳过"
        return rec

    x, n_spike = preprocess_window(y, fs, p)
    rec["n_spike"] = n_spike
    rms = float(np.sqrt(np.mean(x * x)))
    rec["rms_mg"] = round(rms, 4)
    g0_thr = p["g0_rms_factor"] * noise_floor
    rec["g0_threshold"] = round(float(g0_thr), 4)
    rec["g0"] = bool(rms >= g0_thr)

    lo, hi = float(p["f_lo"]), min(float(p["f_hi"]), fs / 2.0 - 1e-6)
    nperseg = int(p["nperseg_s"] * fs)
    wr = welch_psd(x, fs, nperseg)
    if wr is None:
        rec["reason"] = "Welch 无法计算（窗太短）"
        return rec
    f, power = wr
    f1_raw, ratio, med = _band_peak(f, power, lo, hi)
    i1 = int(np.argmin(np.abs(f - (f1_raw if f1_raw is not None else lo))))
    f1 = refine_logparabola(f, power, i1) if (p["refine_f1"] and f1_raw is not None) else f1_raw
    rec["welch"] = {"freq": [], "power": []}
    fd, pd = _downsample([float(v) for v in f], [float(v) for v in power])
    rec["welch"] = {"freq": [round(v, 4) for v in fd], "power": [round(v, 12) for v in pd]}
    rec["peak_ratio"] = round(float(ratio), 2)
    rec["g1"] = bool(ratio >= p["g1_peak_ratio"])
    rec["f1_raw_hz"] = round(float(f1_raw), 4) if f1_raw is not None else None
    rec["f1_hz"] = round(float(f1), 4) if f1 is not None else None

    # G2：稳定性（band=带内验峰 AZV 模式 / full=旧全带 argmax）
    subs: list[float | None] = []
    n = len(x)
    band_frac, sub_ratio = float(p["g2_band_frac"]), float(p["g2_sub_ratio"])
    for k in range(p["n_sub"]):
        a, b = int(k * n / p["n_sub"]), int((k + 1) * n / p["n_sub"])
        part = x[a:b]
        if len(part) < 8:
            subs.append(None)
            continue
        sr = welch_psd(part, fs, min(nperseg, len(part)))
        if sr is None:
            subs.append(None)
            continue
        sf, sp = sr
        if p["g2_mode"] == "band" and f1_raw is not None:
            # AZV 模式：只在 f0±band_frac 带内找峰，且峰须显著于该子窗谱中位
            slo, shi = f1_raw * (1 - band_frac), f1_raw * (1 + band_frac)
            sband = (sf >= slo) & (sf <= shi)
            if not sband.any():
                subs.append(None)
                continue
            smed = float(np.median(sp[(sf >= lo) & (sf <= hi)]))
            i = int(np.argmax(np.where(sband, sp, -np.inf)))
            if smed > 0 and float(sp[i]) >= sub_ratio * smed:
                subs.append(round(refine_logparabola(sf, sp, i), 4))
            else:
                subs.append(None)
        else:
            sf1, _, _ = _band_peak(sf, sp, lo, hi)
            subs.append(round(sf1, 4) if sf1 is not None else None)
    rec["sub_freqs"] = subs
    good = [v for v in subs if v and v > 0]
    rec["sub_n_valid"] = len(good)
    if p["g2_mode"] == "band":
        need = max(2, min(int(p["g2_min_subs"]), p["n_sub"]))
        if len(good) >= need and float(np.mean(good)) > 0:
            cv = float(np.std(good) / np.mean(good))
            rec["cv"] = round(cv, 4)
            rec["g2"] = bool(cv <= p["g2_cv_max"])
        else:
            rec["cv"] = None
            rec["g2"] = False
    elif len(good) >= 2 and float(np.mean(good)) > 0:
        cv = float(np.std(good) / np.mean(good))
        rec["cv"] = round(cv, 4)
        rec["g2"] = bool(cv <= p["g2_cv_max"])
    else:
        rec["cv"] = None
        rec["g2"] = False

    # G3：f1 > g3_factor / 窗实际跨度，且不低于绝对下限
    g3_thr = max(p["g3_factor"] / span, float(p["f1_min_hz"]))
    rec["g3_threshold_hz"] = round(float(g3_thr), 4)
    rec["g3"] = bool(f1 is not None and f1 > g3_thr)

    rec["peaks"] = local_peaks(f, power, lo, hi, med,
                               float(p["peak_min_ratio"]), f1 or 0.0, p["n_peaks"])
    rec["valid"] = bool(rec["g0"] and rec["g1"] and rec["g2"] and rec["g3"])
    failed = [g for g in ("G0", "G1", "G2", "G3") if not rec[g.lower()]]
    rec["reason"] = "四门全过" if rec["valid"] else ("未通过：" + "/".join(failed))

    # ---- 确认证据（软确认：增强置信展示，不否决 VALID 判定）----
    if rec["valid"] and f1 is not None:
        conf: dict[str, Any] = {"axes": {}, "axes_ok": 0, "harmonics": {},
                                "cross": None, "zeta": None, "amp_eq_mg": None}
        # 等效正弦幅值：√(2·∫P df)，Welch 密度谱
        if med > 0:
            try:
                bw = max(3 * float(f[1] - f[0]), abs(f1 - (f1_raw or f1)) * 4 + 3 * float(f[1] - f[0]))
                sel = (f >= (f1_raw or f1) - bw) & (f <= (f1_raw or f1) + bw)
                conf["amp_eq_mg"] = round(float(np.sqrt(2.0 * np.trapezoid(power[sel], f[sel]))), 4)
            except Exception:
                pass
        # 半功率阻尼 ζ（受 Δf 分辨率限制，为上限估计）
        try:
            half = power[i1] / 2.0
            j1 = i1
            while j1 > 0 and power[j1] > half:
                j1 -= 1
            j2 = i1
            while j2 < len(power) - 1 and power[j2] > half:
                j2 += 1
            if j1 > 0 and j2 < len(power) - 1:
                df_res = float(f[1] - f[0])
                fw1 = f[j1] + (half - power[j1]) * (f[j1 + 1] - f[j1]) / (power[j1 + 1] - power[j1])
                fw2 = f[j2 - 1] + (power[j2 - 1] - half) * (f[j2] - f[j2 - 1]) / (power[j2 - 1] - power[j2])
                conf["zeta"] = round(float((fw2 - fw1) / (2.0 * f[i1])), 4)
        except Exception:
            pass
        # 多轴同频
        if p["multi_axis"] and other_axes:
            mlo, mhi = f1 * (1 - float(p["ma_band_frac"])), f1 * (1 + float(p["ma_band_frac"]))
            for tag, ys2 in other_axes.items():
                x2, _ = preprocess_window(ys2, fs, p)
                w2 = welch_psd(x2, fs, nperseg)
                if w2 is None:
                    conf["axes"][tag] = None
                    continue
                f2, p2 = w2
                b2 = (f2 >= mlo) & (f2 <= mhi)
                if not b2.any():
                    conf["axes"][tag] = None
                    continue
                m2 = float(np.median(p2[(f2 >= lo) & (f2 <= hi)]))
                i2 = int(np.argmax(np.where(b2, p2, -np.inf)))
                r2 = float(p2[i2]) / m2 if m2 > 0 else 0.0
                ok = bool(r2 >= p["ma_ratio"])
                if ok:
                    conf["axes_ok"] += 1
                conf["axes"][tag] = {"f_hz": round(float(refine_logparabola(f2, p2, i2)), 4),
                                     "ratio": round(r2, 2), "ok": ok}
        # 谐波族 2f1 / 3f1
        if p["harmonic_check"] and med > 0:
            for hn in (2, 3):
                target = hn * f1
                if target > fs / 2 * 0.95:
                    break
                hband = (f >= 0.93 * target) & (f <= 1.07 * target)
                if hband.any():
                    hr = float(power[hband].max()) / med
                    if hr >= p["harm_ratio"]:
                        conf["harmonics"][str(hn)] = round(hr, 2)
        # 过零率 + Hilbert 瞬时频率交叉验证
        if p["xcheck"]:
            try:
                zc = int(np.sum(np.diff(np.sign(x)) != 0))
                f_zc = (zc - 1) / 2.0 / span if zc > 2 else None
                inst = np.diff(np.unwrap(np.angle(hilbert(x)))) * fs / (2.0 * np.pi)
                f_hi_ = float(np.median(inst))
                ok = bool(f_zc is not None and abs(f_zc - f1) <= 0.15 * f1
                          and abs(f_hi_ - f1) <= 0.15 * f1)
                conf["cross"] = {"f_zc": round(float(f_zc), 4) if f_zc else None,
                                 "f_hilbert": round(f_hi_, 4), "ok": ok}
            except Exception:
                conf["cross"] = None
        rec["confirm"] = conf
        rec["confirmed"] = bool(conf.get("axes_ok", 0) >= 1)

    if rec["valid"] and f1:
        rec["cable_force_idx"] = round(100.0 * (f1 / p["f1_ref_hz"]) ** 2, 2)
        rec["cable_force_n"] = round(4.0 * p["L_m"] ** 2 * p["mu_kg_m"] * f1 ** 2, 1)
    return rec


# ---------------------------------------------------------------- 总入口
def analyze_text(text: str, filename: str, params: dict | None) -> dict[str, Any]:
    p = merge_params(params)
    header, data = parse_table(text)
    cols = numeric_columns(header, data)
    axis = p["axis"] or next((c for c in cols if "加速度Z" in c), cols[0] if cols else "")
    if axis not in cols:
        raise ValueError(f"列 {axis!r} 不是数值列，可选：{cols}")

    tdiag: dict[str, Any] = {}
    t, y, fs, unit = build_series(header, data, axis, p, tdiag)
    actual_scale = (p["unit_scale"] if p["unit_scale"] > 0
                    else (1000.0 if "加速度" in axis else 1.0))
    check_runtime_calibration(p, fs, axis, actual_scale)
    t0 = float(t[0])
    seg_ids = np.floor((t - t0) / p["window_s"]).astype(int)
    groups: list[tuple[int, np.ndarray]] = []
    for seg in sorted(set(seg_ids.tolist())):
        idx = np.where(seg_ids == seg)[0]
        groups.append((int(seg), idx))

    # 多轴同频：自动找其余加速度轴（列名含「加速度X/Y/Z」），统一到同一时间栅格
    other_series: dict[str, np.ndarray] = {}
    if p["multi_axis"]:
        for tag in ("X", "Y", "Z"):
            col = next((c for c in cols if f"加速度{tag}" in c), None)
            if not col or col == axis:
                continue
            try:
                _, ys2, fs2, _ = build_series(header, data, col, p)
                if len(ys2) == len(y):
                    other_series[tag] = ys2
            except Exception:
                pass

    # 本底：优先用户给定，否则用各窗预处理后 RMS 的 10 分位数（与 G0 同口径）
    rms_all = []
    for _, idx in groups:
        if len(idx) < p["min_samples"]:
            continue
        try:
            xr, _ = preprocess_window(y[idx], fs, p)
            rms_all.append(float(np.sqrt(np.mean(xr * xr))))
        except Exception:
            continue
    nf = (float(p["noise_floor_mg"]) if p["noise_floor_mg"] > 0
          else float(np.percentile(rms_all, 10)) if rms_all else float(np.std(y)))
    nf = max(nf, 1e-9)

    windows = [analyze_window(seg, t[idx] - t0, y[idx], fs, p, nf,
                              {tag: s2[idx] for tag, s2 in other_series.items()}
                              if other_series else None)
               for seg, idx in groups]

    # 索力是状态量：只有 VALID 才刷新，INVALID 保持上次 VALID 并标时效
    last_valid: dict[str, Any] | None = None
    valid_count = 0
    for w in windows:
        if w["skipped"]:
            w["state"] = "SKIPPED"
            # 跳过窗不改变状态量，但仍应携带「当前应保持的索力」
            w["cable_current"] = dict(last_valid) if last_valid else None
            continue
        if w["valid"]:
            valid_count += 1
            last_valid = {"idx": w["idx"], "t1": w["t1"], "f1_hz": w["f1_hz"],
                          "cable_force_idx": w["cable_force_idx"],
                          "cable_force_n": w["cable_force_n"]}
        w["state"] = "VALID" if w["valid"] else "INVALID"
        w["cable_current"] = dict(last_valid) if last_valid else None

    # 「当前」必须是最后一个已评估的窗，不能是被跳过的尾部短窗
    last = next((w for w in reversed(windows) if not w["skipped"]), None)
    age_s = None
    if last_valid and last:
        age_s = round(float(last["t1"] - last_valid["t1"]), 2)
    state = {
        "total_windows": len(windows),
        "evaluated": sum(1 for w in windows if not w["skipped"]),
        "valid_count": valid_count,
        "last_valid": last_valid,
        "current_cable": last["cable_current"] if last else None,
        "age_s": age_s,
        "stale": bool(last_valid and last and not last["valid"]),
        "has_output": bool(last_valid is not None),
        "status": ("VALID" if last and last["valid"] else
                   "STALE" if last_valid else "NO_VALID"),
    }

    ts = [round(float(v - t0), 3) for v in t]
    ys = [round(float(v), 5) for v in y]
    td, yd = _downsample(ts, ys, 2000)
    return {
        "meta": {
            "filename": filename, "axis": axis, "unit": unit, "fs": round(fs, 4),
            "n_samples": int(len(y)), "duration_s": round(float(t[-1] - t0), 3),
            "nyquist_hz": round(fs / 2, 4), "unit_scale": 1000.0 if unit == "mg" else 1.0,
            "params": {k: p[k] for k in PARAM_KEYS},
            "calib": p.get("_calib") or None,
            "time_quality": tdiag,
        },
        "axes": cols,
        "noise_floor": round(nf, 4),
        "series": {"t": td, "y": yd},
        "windows": windows,
        "state": state,
    }


def analyze_file(path: str, params: dict | None = None) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        text = fh.read()
    import os
    return analyze_text(text, os.path.basename(path), params)


# ---------------------------------------------------------------- CLI
def _fmt(v: Any, width: int = 8) -> str:
    if v is None:
        return "-" * width
    if isinstance(v, float):
        return f"{v:>{width}.4f}"
    return f"{str(v):>{width}s}"


def main(argv: list[str] | None = None) -> int:
    import os
    ap = argparse.ArgumentParser(description="四门控频率计算（加速度时程）")
    ap.add_argument("--file", default="data_0.csv", help="csv / txt 路径")
    ap.add_argument("--axis", default="", help="列名，默认自动选加速度Z")
    ap.add_argument("--json", default="", help="结果另存为 JSON")
    ap.add_argument("--list-axes", action="store_true", help="只列出可选数值列")
    ap.add_argument("--calib", default="",
                    help="标定文件 accel_calib.json；指定后其参数作为默认基底")
    for k, v in DEFAULTS.items():
        if k in ("axis",):
            continue
        if isinstance(v, bool):
            ap.add_argument("--" + k.replace("_", "-"), default=None,
                            type=lambda s: str(s).lower() in ("1", "true", "yes", "on"),
                            help=f"默认 {v}")
        else:
            ap.add_argument("--" + k.replace("_", "-"), default=None,
                            type=float if isinstance(v, float) else
                            int if isinstance(v, int) else str,
                            help=f"默认 {v}")
    args = ap.parse_args(argv)
    raw = {k: getattr(args, k) for k in DEFAULTS if k != "axis"}
    raw = {k: v for k, v in raw.items() if v is not None}
    if args.axis:
        raw["axis"] = args.axis
    if args.calib:
        raw["calib"] = args.calib
    params = merge_params(raw)
    path = args.file if os.path.isabs(args.file) else os.path.join(os.path.dirname(__file__), args.file)

    if args.list_axes:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
            h, d = parse_table(fh.read())
        print("可选数值列：", numeric_columns(h, d))
        return 0

    res = analyze_file(path, params)
    m = res["meta"]
    print(f"\n文件 {m['filename']} | 轴 {m['axis']} ({m['unit']}) | fs={m['fs']}Hz "
          f"| {m['n_samples']} 点 / {m['duration_s']}s | Nyquist={m['nyquist_hz']}Hz")
    print(f"本底 RMS = {res['noise_floor']} {m['unit']} | 窗长 {params['window_s']}s "
          f"| 分析带 {params['f_lo']}~{min(params['f_hi'], m['nyquist_hz'])}Hz")
    print(f"门限：G0>={params['g0_rms_factor']}×本底  G1>={params['g1_peak_ratio']} "
          f" G2<={params['g2_cv_max']*100:.1f}%  G3>{params['g3_factor']}/T")
    cm = m.get("calib")
    if cm:
        s = cm.get("sensor") or {}
        print(f"标定批次：{s.get('id')} / {s.get('model')} | "
              f"±{s.get('range_g')}g @ {s.get('rate_hz')}Hz | "
              f"标定于 {cm.get('calibrated_on')} 有效至 {cm.get('valid_until')} | "
              f"本底 {cm.get('noise_floor_mg')}mg")
        for w in list(cm.get("warnings") or []) + list(cm.get("runtime_warnings") or []):
            print(f"  [标定告警] {w}")
    print("\n" + "=" * 118)
    print(f"{'窗':>3s} {'区间s':>14s} {'n':>5s} {'RMS':>8s} {'G0':>5s} "
          f"{'峰/底':>8s} {'G1':>5s} {'CV%':>7s} {'G2':>5s} {'f1':>8s} {'2/T':>7s} {'G3':>5s} "
          f"{'判定':>8s} {'索力指数':>9s}")
    print("-" * 118)
    for w in res["windows"]:
        if w["skipped"]:
            print(f"{w['idx']:>3d} {w['t0']:>6.1f}~{w['t1']:<7.1f} {w['n']:>5d}  {w['reason']}")
            continue
        cv = "-" if w["cv"] is None else f"{w['cv']*100:.1f}"
        print(f"{w['idx']:>3d} {w['t0']:>6.1f}~{w['t1']:<7.1f} {w['n']:>5d} "
              f"{w['rms_mg']:>8.3f} {'PASS' if w['g0'] else 'FAIL':>5s} "
              f"{w['peak_ratio']:>8.2f} {'PASS' if w['g1'] else 'FAIL':>5s} "
              f"{cv:>7s} {'PASS' if w['g2'] else 'FAIL':>5s} "
              f"{_fmt(w['f1_hz']):>8s} {w['g3_threshold_hz']:>7.4f} "
              f"{'PASS' if w['g3'] else 'FAIL':>5s} {w['state']:>8s} "
              f"{_fmt(w['cable_force_idx'], 9):>9s}")
        if w["peaks"]:
            pk = "  ".join(f"{q['f_hz']:.3f}Hz(×{q['ratio']:.1f},n≈{q['n_est']})"
                           for q in w["peaks"][:5])
            print(f"      显著峰: {pk}")
        c = w.get("confirm")
        if c:
            ax_txt = " ".join(f"{k}={'%.3fHz(%.0fx)' % (v['f_hz'], v['ratio']) if v else '—'}{'✓' if v and v['ok'] else '✗'}"
                              for k, v in c["axes"].items()) or "无其他轴"
            hx = c.get("cross") or {}
            print(f"      确认: 多轴[{ax_txt}] 确认数={c['axes_ok']} "
                  f"谐波={c['harmonics'] or '无'} "
                  f"过零/Hilbert={'%.3f/%.3f' % (hx.get('f_zc') or 0, hx.get('f_hilbert') or 0) if hx else '—'}"
                  f"{'✓' if hx.get('ok') else '✗' if hx else ''} "
                  f"ζ={c['zeta']} 等效幅值={c['amp_eq_mg']}mg 去尖峰={w.get('n_spike')}点"
                  + ("  [多轴已确认]" if w.get("confirmed") else ""))
    print("=" * 118)
    s = res["state"]
    print(f"共 {s['total_windows']} 窗（评估 {s['evaluated']}），VALID {s['valid_count']} 窗")
    if s["last_valid"]:
        print(f"索力状态：最后有效窗 w{s['last_valid']['idx']} t={s['last_valid']['t1']}s  "
              f"f1={s['last_valid']['f1_hz']}Hz  索力指数={s['last_valid']['cable_force_idx']}  "
              f"绝对索力={s['last_valid']['cable_force_n']}N")
        print(f"当前显示：{s['status']}  时效 {s['age_s']}s"
              f"{'（沿用上次 VALID）' if s['stale'] else ''}")
    else:
        print("索力状态：NO_VALID —— 没有任何窗口通过四门，**不输出索力**（显示 —）")
    if args.json:
        out = args.json if os.path.isabs(args.json) else os.path.join(os.path.dirname(__file__), args.json)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=2)
        print(f"JSON 已写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
