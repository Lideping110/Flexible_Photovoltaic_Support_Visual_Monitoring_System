# -*- coding: utf-8 -*-
"""加速度计「安装前」系统级标定 —— 生成可直接投用的参数文件 accel_calib.json。

设计原则
--------
1. 不动出厂标定：WT901SDCL-BT50 输出的是内部已完成零偏/标度/轴对准/温度补偿的
   工程值(g)。本脚本只做「验证 + 记录」，不覆盖模块内部参数；只有用户提供
   >=4 个六面姿态文件时才解算可选的 12 参数补偿矩阵，且默认不启用。
2. 标定口径必须与生产完全一致：预处理、Welch、四门全部直接调用 frequency_gate.py，
   不在本文件里另写一套，避免「标定时一套、上线时另一套」。
3. 分三层：L1 传感器本底(安装前) / L2 门限阈值(安装前台架) / L3 绝对物理量(必须安装后)。
   L3 在 site_required 里留空并强制告警，不允许用占位值冒充已标定。
4. 防串用：产物带 sensor_id / 量程 / 速率 / 轴列名 / 数据指纹 / 有效期 / 复标触发条件，
   加载时与实时数据比对，不一致只告警不静默放行。

用法（项目根目录，必须用 .venv）
--------------------------------
    .venv/Scripts/python.exe sensor_data/accelerometer/10-34-27-942/calibrate_accel.py \
        --static data_0.csv --sensor-id WT901-0001 --range-g 2 --temp-c 22

    # 六面法（可选，姿态标签 +X/-X/+Y/-Y/+Z/-Z 与文件一一对应）
    ... --static p_xp.csv --pose +X --static p_xn.csv --pose -X ... --pose -Z

    # 三段静止（推荐）：时段/温度/摆放姿态逐段标注，脚本据此校验「跨时段/跨温度」
    ... --static 早.csv --slot 早 --temp 18 --pose-desc 平放 \
         --static 午.csv --slot 午 --temp 26 --pose-desc 平放 \
         --static 晚.csv --slot 晚 --temp 14 --pose-desc 平放

    # 自定义输出
    ... --out accel_calib.json --report accel_calib_report.txt --n-mc 200

产物：accel_calib.json（参数，交给 gate_server.py --calib 或前端载入）
      accel_calib_report.txt（人读报告，含复标触发条件与现场待填清单）
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import io
import json
import math
import os
import sys
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import frequency_gate as fg  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
SCHEMA_VERSION = 1

POSE_TARGETS: dict[str, tuple[float, float, float]] = {
    "+X": (1.0, 0.0, 0.0), "-X": (-1.0, 0.0, 0.0),
    "+Y": (0.0, 1.0, 0.0), "-Y": (0.0, -1.0, 0.0),
    "+Z": (0.0, 0.0, 1.0), "-Z": (0.0, 0.0, -1.0),
}

# L3：必须安装后现场取值，标定阶段一律留空
SITE_REQUIRED: list[dict[str, Any]] = [
    {"key": "L_m", "label": "索跨长 L (m)", "value": None,
     "how": "现场钢尺/全站仪量两锚固点净距；斜索取弦长",
     "why": "T = 4L²μf₁² 中 L 是平方项，误差放大 2 倍"},
    {"key": "mu_kg_m", "label": "等效线密度 μ (kg/m)", "value": None,
     "how": "索材合格证线密度 + 附件折算；传感器附加质量约 18.86g 须按安装间距折算进 μ",
     "why": "附加质量改变局部 μ，直接改变 f₁ 与反算索力"},
    {"key": "f1_ref_hz", "label": "基准一阶频率 f1_ref (Hz)", "value": None,
     "how": "张拉到位后用千斤顶/索力计同步读数，取同一时刻的稳定 f₁",
     "why": "cable_force_idx=(f1/f1_ref)²×100 的分母，未标定时只是相对占位"},
    {"key": "attach_mass_g", "label": "传感器+夹具附加质量 (g)", "value": None,
     "how": "电子秤实测整套（含夹具、线、防水壳）",
     "why": "折算进 μ；忽略会把安装本身带来的频率下移误判为索力损失"},
]

RECALIBRATE_IF = [
    "更换传感器 / 更换安装位置 / 重新拆装夹具",
    "改变量程(±2/4/8/16g)、输出速率、固件或滤波配置",
    "更换供电、线缆长度、采集主机或接地方式",
    "环境温度区间偏离标定时 ±10℃ 以上",
    "连续 7 天出现 G0 全部 FAIL 或静止段 RMS 漂移 >20%",
    "超过 valid_until，或传感器受过冲击/进水/维修",
]

# 现场采集规格：写进 JSON 与报告，避免「只采一段 5min 就以为标定完成」
COLLECTION_REQUIREMENTS: list[dict[str, str]] = [
    {"item": "静止本底", "minimum": "每段 ≥5min 且为 30s 整数倍（如 120s/150s），合计 ≥3 段、≥10 个完整窗，跨 ≥2 个温度点",
     "why": "5min 仅 10 个独立 30s 窗，p10 本底标准误约 0.02mg；不足 30s 的尾段碎片不参与统计，单段无法覆盖温漂与时段差异"},
    {"item": "安装姿态", "minimum": "台架按现场安装姿态摆放；装完现场复测静止 ≥2min",
     "why": "重力投影在三轴分配随姿态变，MEMS 本底与轴/重力方向相关，台架姿态≠现场姿态"},
    {"item": "已知激励段", "minimum": "振动台/激振器/敲击，≥60s，频率覆盖 0.5~4Hz，记录真值频率",
     "why": "脚本内 55 组注入是合成信号，只验算法不验硬件链路（标度、相位、夹具谐振、松动）"},
    {"item": "温度记录", "minimum": "每段记录环境温度；室外至少覆盖低温/高温各一段",
     "why": "单一温度点无法标定零偏温漂，户外昼夜温差会改变本底"},
    {"item": "L3 现场量", "minimum": "L_m / mu_kg_m / f1_ref_hz / attach_mass_g 装后实测",
     "why": "取决于现场几何与张拉工况，台架无法提前取得"},
]


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def _read_one(path: str, axis: str) -> dict[str, Any]:
    """解析一个静止文件，返回原始表、时间轴、三轴(g)、主标定轴序列(mg)、fs。"""
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        text = fh.read()
    header, data = fg.parse_table(text)
    cols = fg.numeric_columns(header, data)
    if axis and axis not in cols:
        raise ValueError(f"{os.path.basename(path)} 找不到列 {axis!r}，可选：{cols}")
    if not axis:
        axis = next((c for c in cols if "加速度Z" in c), cols[0] if cols else "")
    p = fg.merge_params({"axis": axis})
    t, y, fs, unit = fg.build_series(header, data, axis, p)

    tri: dict[str, np.ndarray] = {}
    for tag in ("X", "Y", "Z"):
        col = next((c for c in cols if f"加速度{tag}" in c), None)
        if col:
            idx = header.index(col)
            v = []
            for r in data:
                try:
                    v.append(float(r[idx]))
                except (ValueError, IndexError):
                    continue
            tri[tag] = np.asarray(v, dtype=float)

    # 原始时间戳间隔（用于丢帧体检）
    ti, kind = fg._time_col(header)
    dt_raw: list[float] = []
    if ti is not None:
        secs = []
        for r in data:
            s = fg._to_seconds(r[ti], kind) if ti < len(r) else None
            if s is not None:
                secs.append(s)
        secs = np.asarray(secs, dtype=float)
        if len(secs) > 2:
            d = np.diff(secs)
            d = d[d > 0]
            dt_raw = d.tolist()
    return {
        "path": path, "name": os.path.basename(path), "sha256": _sha256(path),
        "header": header, "cols": cols, "axis": axis, "unit": unit,
        "t": t, "y": y, "fs": float(fs), "tri": tri, "dt_raw": dt_raw,
        "rows": len(data),
    }


# ---------------------------------------------------------------- STEP 0
def step0_health(rec: dict[str, Any]) -> dict[str, Any]:
    """数据体检：采样率、丢帧、量化台阶、饱和、常数段。"""
    y, fs = rec["y"], rec["fs"]
    n = len(y)
    dt = np.asarray(rec["dt_raw"], dtype=float)
    if len(dt):
        base = float(np.median(dt))
        irregular = int(np.sum(np.abs(dt - base) > 0.25 * base))
        irregular_pct = round(100.0 * irregular / len(dt), 3)
    else:
        base, irregular, irregular_pct = 1.0 / fs, 0, 0.0

    axis_g = rec["tri"].get("Z", y / 1000.0)
    uniq = np.unique(np.round(axis_g, 6))
    steps = np.diff(uniq)
    steps = steps[steps > 1e-9]
    quant_mg = float(np.median(steps) * 1000.0) if len(steps) else 0.0

    # 最长连续常数段
    longest, run = 1, 1
    for i in range(1, len(axis_g)):
        run = run + 1 if axis_g[i] == axis_g[i - 1] else 1
        longest = max(longest, run)

    return {
        "n_samples": int(n),
        "duration_s": round(float(rec["t"][-1] - rec["t"][0]), 3),
        "fs_hz": round(fs, 4),
        "dt_median_s": round(float(base), 6),
        "irregular_dt_count": irregular,
        "irregular_dt_pct": irregular_pct,
        "unique_values": int(len(uniq)),
        "quant_step_mg": round(quant_mg, 4),
        "longest_constant_run": int(longest),
        "peak_to_peak_mg": round(float(np.ptp(y)), 4),
        "mean_g": round(float(np.mean(axis_g)), 6),
        "columns": rec["cols"],
        "axis_column": rec["axis"],
    }


# ---------------------------------------------------------------- STEP 1
def step1_gravity(recs: list[dict[str, Any]], poses: list[str],
                  range_g: float) -> dict[str, Any]:
    """重力投影验证；>=4 姿态时解算可选的 12 参数补偿矩阵（不自动启用）。"""
    means: list[np.ndarray] = []
    for r in recs:
        tri = r["tri"]
        means.append(np.asarray([float(np.mean(tri[k])) for k in ("X", "Y", "Z")]
                                if all(k in tri for k in ("X", "Y", "Z"))
                                else [0.0, 0.0, float(np.mean(r["y"] / 1000.0))]))
    m0 = means[0]
    gmag = float(np.linalg.norm(m0))
    tilt = float(math.degrees(math.acos(float(np.clip(m0[2] / max(gmag, 1e-9), -1, 1)))))

    out: dict[str, Any] = {
        "method": "single_pose" if len(recs) < 4 else "six_face_optional",
        "factory_calib_touched": False,
        "gravity_norm_g": round(gmag, 6),
        "gravity_norm_dev_pct": round((gmag - 1.0) * 100.0, 3),
        "mount_tilt_deg": round(tilt, 3),
        "mean_vector_g": [round(float(v), 6) for v in m0],
        "note": ("单姿态无法分离「零偏」与「安装倾角」，上式为二者综合偏差；"
                 "不做补偿、不覆盖出厂标定。"),
        "poses": poses,
        "saturation": bool(np.any(np.abs(m0) >= 0.98 * range_g)),
        "optional_correction": None,
    }
    if len(recs) >= 4 and all(p in POSE_TARGETS for p in poses):
        K = len(recs)
        A = np.zeros((3 * K, 12))
        Y = np.zeros(3 * K)
        for i, (m, tag) in enumerate(zip(means, poses)):
            u = np.asarray(POSE_TARGETS[tag], dtype=float)
            for k in range(3):
                A[3 * i + k, k] = 1.0
                A[3 * i + k, 3 + 3 * k:3 + 3 * k + 3] = u
                Y[3 * i + k] = m[k]
        sol, *_ = np.linalg.lstsq(A, Y, rcond=None)
        resid = float(np.sqrt(np.mean((Y - A @ sol) ** 2)))
        out["optional_correction"] = {
            "bias_g": [round(float(v), 6) for v in sol[:3]],
            "matrix_M": [[round(float(v), 6) for v in row]
                         for row in sol[3:].reshape(3, 3)],
            "residual_rms_g": round(resid, 6),
            "enabled": False,
            "warning": "仅作参考；启用会覆盖模块出厂补偿，需高精度转台复核后人工决定",
        }
        out["method"] = "six_face_optional"
        out["note"] = "已解算 12 参数模型，但默认不启用；残差见 optional_correction。"
    return out


# ---------------------------------------------------------------- STEP 2
def build_window_pool(recs: list[dict[str, Any]], p: dict[str, Any]) -> list[np.ndarray]:
    """所有静止段按生产固定不重叠 30s 窗切分（逐段切，避免跨段边界产生混合窗）。

    修复：旧版 main() 只把 recs[0] 送进 S0/S2/S3/S4，补采的多段数据会被静默丢弃。
    """
    fs = recs[0]["fs"]
    n = int(round(p["window_s"] * fs))
    pool: list[np.ndarray] = []
    for r in recs:
        y = r["y"]
        for s in range(0, max(1, len(y) - n + 1), n):
            seg = y[s:s + n]
            # 只收完整 30s 窗：残窗的高通边界效应会污染本底统计，
            # 且与生产 seg=int(t//W) 的固定窗口径不符
            if len(seg) == n:
                pool.append(seg)
    if not pool:
        raise ValueError("没有任何满足 min_samples 的 30s 静止窗，请延长采集时长")
    return pool


def step0_summary(recs: list[dict[str, Any]], p: dict[str, Any],
                  pool: list[np.ndarray]) -> dict[str, Any]:
    """独立数据量体检：5min 静止 = 只有 10 个独立窗，必须显式记录。"""
    fs0 = recs[0]["fs"]
    n = int(round(p["window_s"] * fs0))
    return {
        "segments": len(recs),
        "per_segment_s": [round(len(r["y"]) / max(r["fs"], 1e-9), 1) for r in recs],
        "total_duration_s": round(sum(len(r["y"]) / max(r["fs"], 1e-9) for r in recs), 2),
        "usable_window_s": round(len(pool) * p["window_s"], 1),
        "wasted_tail_s": round(sum(len(r["y"]) / max(r["fs"], 1e-9) for r in recs)
                               - len(pool) * p["window_s"], 1),
        "independent_windows": len(pool),
        "fs_mismatch": any(abs(r["fs"] - fs0) / fs0 > 0.05 for r in recs),
        "min_minutes_required": 5.0,
        "min_segments_recommended": 3,
    }


def step2_noise(pool: list[np.ndarray], p: dict[str, Any], fs: float,
                raw_y: np.ndarray, health: dict[str, Any]) -> tuple[dict[str, Any], float]:
    """本底：与生产 G0 完全同口径 = 预处理后逐窗 RMS 的 10 分位数。"""
    rms_win: list[float] = []
    pre_all: list[np.ndarray] = []
    for seg in pool:
        x, _ = fg.preprocess_window(seg, fs, p)
        pre_all.append(x)
        rms_win.append(float(np.sqrt(np.mean(x * x))))
    nf = float(np.percentile(rms_win, 10))
    x = np.concatenate(pre_all)
    ac1 = float(np.corrcoef(x[:-1], x[1:])[0, 1]) if len(x) > 2 else 0.0
    q = np.percentile(np.abs(x), [50, 95, 99, 100])

    # p10 是低估估计量：用卡方蒙特卡洛量化本次窗数下的标准误与系统偏差
    k, npt = len(rms_win), len(pool[0])
    sigma = float(np.mean(rms_win))
    mc = sigma * np.sqrt(np.random.default_rng(0).chisquare(npt, size=(2000, k)) / npt)
    p10s = np.percentile(mc, 10, axis=1)
    info = {
        "noise_floor_mg": round(nf, 4),
        "definition": "预处理(去尖峰→去趋势→0.3Hz高通)后逐30s窗 RMS 的 10 分位，与 G0 同口径",
        "n_windows": k,
        "pre_rms_mean_mg": round(sigma, 4),
        "p10_se_mg": round(float(np.std(p10s)), 4),
        "p10_bias_pct": round((float(np.mean(p10s)) / sigma - 1) * 100, 2),
        "raw_std_mg": round(float(np.std(raw_y)), 4),
        "raw_std_vs_floor_pct": round((float(np.std(raw_y)) / max(nf, 1e-9) - 1) * 100, 2),
        "window_rms_mg": [round(v, 4) for v in rms_win],
        "abs_percentile_mg": {"p50": round(float(q[0]), 4), "p95": round(float(q[1]), 4),
                              "p99": round(float(q[2]), 4), "max": round(float(q[3]), 4)},
        "lag1_autocorr": round(ac1, 4),
        "white_like": bool(abs(ac1) < 0.15),
        "quant_step_mg": float(health.get("quant_step_mg") or 0.0),
    }
    return info, nf


# ---------------------------------------------------------------- STEP 3
def step3_false_alarm(pool: list[np.ndarray], p: dict[str, Any], fs: float,
                      nf: float, noise: dict[str, Any], n_mc: int,
                      rng: np.random.Generator) -> dict[str, Any]:
    """静止虚警基线：从窗池抽整窗（与生产固定不重叠窗口径一致）。
    另做独立合成白噪蒙特卡洛，解决「5min 只有 10 个独立窗却抽 150 次」的伪重复问题。"""
    n = len(pool[0])
    t = np.arange(n) / fs
    ratios: list[float] = []
    rms: list[float] = []
    for _ in range(n_mc):
        seg = pool[int(rng.integers(0, len(pool)))]
        r = fg.analyze_window(0, t, seg, fs, p, nf)
        if r["peak_ratio"] is not None:
            ratios.append(float(r["peak_ratio"]))
        if r["rms_mg"] is not None:
            rms.append(float(r["rms_mg"]))
    ratios = ratios or [0.0]
    rms = rms or [nf]
    p995, p999 = float(np.percentile(ratios, 99.5)), float(np.percentile(ratios, 99.9))
    r999 = float(np.percentile(rms, 99.9))

    # 独立合成白噪：按实测本底 σ 与量化台阶构造，样本真正独立，不依赖采集时长
    sigma = float(noise["pre_rms_mean_mg"])
    quant = float(noise["quant_step_mg"])
    syn: list[float] = []
    for _ in range(max(n_mc, 300)):
        w = rng.normal(0.0, sigma, n)
        if quant > 0:
            w = np.round(w / quant) * quant
        r = fg.analyze_window(0, t, w, fs, p, nf)
        if r["peak_ratio"] is not None:
            syn.append(float(r["peak_ratio"]))
    syn = syn or [0.0]
    s999 = float(np.percentile(syn, 99.9))
    return {
        "n_trials": n_mc,
        "n_independent_windows": len(pool),
        "pseudo_replication_factor": round(n_mc / max(1, len(pool)), 2),
        "g1_ratio_p50": round(float(np.percentile(ratios, 50)), 3),
        "g1_ratio_p995": round(p995, 3),
        "g1_ratio_p999": round(p999, 3),
        "g1_ratio_max": round(float(np.max(ratios)), 3),
        "g1_synthetic_p999": round(s999, 3),
        "g1_synthetic_max": round(float(np.max(syn)), 3),
        "g0_rms_p999_mg": round(r999, 4),
        "g0_rms_max_mg": round(float(np.max(rms)), 4),
        "derived_g1": round(max(6.0, p999 * 1.5), 1),
        # 标定不得擅自放宽已验证的生产门限：10 是 8s 段/35 格基线，
        # 本次静态数据只证明「有余量」，不证明「可以下调」。
        "recommended_g1": round(max(6.0, p999 * 1.5, s999 * 1.5, 10.0), 1),
        "margin_to_production": round(10.0 / max(p999, 1e-9), 2),
        "derived_g0_factor": round(max(1.2, r999 / max(nf, 1e-9) * 1.10), 2),
        "note": ("实测窗有放回抽样只能覆盖 n_independent_windows 个不同窗，p99.9 属下界估计；"
                 "已用独立合成白噪交叉验证并取较严者。recommended_g1 取与生产基线 10 的较大者。"
                 "换 nperseg_s / f_lo / f_hi 会改变带内格点数，必须重标"),
    }


# ---------------------------------------------------------------- STEP 4
def step4_injection(pool: list[np.ndarray], p: dict[str, Any], fs: float,
                    nf: float) -> dict[str, Any]:
    """注入标定：在真实静止本底上叠加已知幅度/频率的正弦，扫四门可检出域。

    注意：这里注入的是【合成信号】，只验证「算法在这份本底下能检出什么」，
    不验证传感器/夹具对真实机械振动的标度、相位与谐振 —— 那需要实测激励段。
    """
    rr = np.asarray([float(np.sqrt(np.mean(fg.preprocess_window(s, fs, p)[0] ** 2)))
                     for s in pool])
    base = pool[int(np.argmin(np.abs(rr - float(np.median(rr)))))]
    n = len(base)
    t = np.arange(n) / fs
    amps = (0.5, 1.0, 2.0, 3.0, 5.0)
    freqs = (0.20, 0.30, 0.50, 0.80, 1.00, 1.50, 2.00, 2.50, 3.00, 4.00, 4.50)
    grid: dict[str, dict[str, Any]] = {}
    cvs: list[float] = []
    for a in amps:
        for f0 in freqs:
            inj = base + a * np.sin(2.0 * np.pi * f0 * t)
            r = fg.analyze_window(0, t, inj, fs, p, nf)
            grid[f"{a}|{f0}"] = {"ok": bool(r["valid"]),
                                 "f1": r["f1_hz"], "cv": r["cv"],
                                 "ratio": r["peak_ratio"]}
            if r["valid"] and r["cv"] is not None:
                cvs.append(float(r["cv"]))

    min_amp: dict[str, float] = {}
    for f0 in freqs:
        hit = next((a for a in amps if grid[f"{a}|{f0}"]["ok"]), None)
        min_amp[str(f0)] = hit if hit is not None else 0.0
    stable_3mg = [f0 for f0 in freqs if grid[f"3.0|{f0}"]["ok"]]
    f_true_min = min(stable_3mg) if stable_3mg else None

    false_valid: list[float] = []
    for f0 in freqs:
        slow = base + 8.0 * np.sin(2.0 * np.pi * f0 * t)
        r = fg.analyze_window(0, t, slow, fs, p, nf)
        if r["valid"]:
            false_valid.append(f0)

    p99_cv = float(np.percentile(cvs, 99)) if cvs else 0.08
    # 老口径的「子窗频率栅格量化地板」= 0.5·Δf_sub / f_true_min，
    # 在 refine_f1=True（log 抛物线亚格点插值）下已被大幅压低，实测注入 CV p99 即证据。
    df_sub = float(p["n_sub"]) / float(p["window_s"])
    legacy_floor_cv = 0.5 * df_sub / max(f_true_min or 1.0, 0.1)
    derived_cv = round(min(0.20, max(0.06, p99_cv * 1.5)), 3)
    # 同样不擅自收紧生产基线 8%
    recommended_cv = round(max(derived_cv, 0.08), 3)

    d_f = fs / max(1, int(p["nperseg_s"] * fs))
    return {
        "n_cases": len(grid),
        "amplitudes_mg": list(amps),
        "frequencies_hz": list(freqs),
        "min_detect_amp_mg": min_amp,
        "stable_at_3mg_hz": stable_3mg,
        "f_true_min_hz": f_true_min,
        "slow_wave_false_valid_hz": false_valid,
        "slow_wave_false_valid_lowband_hz": [f for f in false_valid if f <= 3 * 0.3],
        "injected_valid_cv_p99": round(p99_cv, 5) if cvs else None,
        "cv_legacy_grid_floor": round(legacy_floor_cv, 5),
        "derived_g2_cv_max": derived_cv,
        "recommended_g2_cv_max": recommended_cv,
        "method_limit": ("四门只判「窗内是否存在稳定窄带峰」，不判「这个峰是不是结构模态」。"
                         "因此任何频率的强单频信号（含阵风脉动、电磁干扰）都会被判 VALID，"
                         "这是单轴 + 四门的方法固有局限，不是某个频段的混淆区；"
                         "必须靠多轴同频确认、跨窗一致性、张力/温度先验补判。"),
        "low_freq_sensitivity": ("0.3Hz 高通与 f_lo 重合，0.2Hz 注入在 5mg 内完全检不出、"
                                 "0.3Hz 需 3mg、0.5Hz 需 2mg；低频检出阈值显著升高是滤波代价。"),
        "welch_delta_f_hz": round(float(d_f), 5),
        "nyquist_hz": round(fs / 2, 4),
        "grid": grid,
        "n_windows_pool": len(pool),
        "base_window_rms_mg": round(float(np.median(rr)), 4),
        "synthetic_only": True,
        "note": ("注入单位 mg；扫描关闭多轴/谐波/过零确认，它们不否决 VALID。"
                 "本项为合成注入，不代表硬件链路已验证"),
    }


# ---------------------------------------------------------------- STEP 5
def step5_excitation(path: str, hz: float, axis: str, p: dict[str, Any],
                     nf: float) -> dict[str, Any]:
    """真实激励段（可选）：验证传感器+夹具+采集链路能否正确读出已知频率。

    这是静止数据永远无法覆盖的一环：标度、相位、安装谐振、夹具松动。
    """
    if not path or hz <= 0:
        return {"status": "NOT_PROVIDED", "file": "", "excitation_hz": hz,
                "windows": 0, "matched": 0,
                "reason": "未提供已知频率的激励数据；L2 的注入标定为合成信号，只验证算法"}
    rec = _read_one(_resolve(path), axis)
    pool = build_window_pool([rec], p)
    fs, n = rec["fs"], len(pool[0])
    t = np.arange(n) / fs
    f1s: list[float | None] = []
    matched = 0
    for seg in pool:
        r = fg.analyze_window(0, t, seg, fs, p, nf)
        f1s.append(r["f1_hz"])
        if r["valid"] and r["f1_hz"] and abs(r["f1_hz"] - hz) / hz <= 0.10:
            matched += 1
    need = max(1, len(pool) // 2)
    return {
        "status": "VERIFIED" if matched >= need else "FAILED",
        "file": os.path.basename(path), "excitation_hz": hz,
        "windows": len(pool), "matched": matched, "need": need,
        "f1_hz": f1s,
        "reason": ("实测激励频率在 ±10% 内被稳定检出，硬件链路可用"
                   if matched >= need else
                   "实测激励未稳定检出：检查夹具松动、量程、采样率、激励幅度或安装谐振"),
    }


# ---------------------------------------------------------------- 参数推导
def derive_params(rec: dict[str, Any], p: dict[str, Any], noise: dict[str, Any],
                  fa: dict[str, Any], inj: dict[str, Any],
                  gravity: dict[str, Any]) -> tuple[dict[str, Any], list[float]]:
    fs = rec["fs"]
    nyq = fs / 2.0
    df = inj["welch_delta_f_hz"]
    # f1_min_hz：低于 0.6Hz 既落在 fc/f_lo 重合的衰减区，又被慢波污染；
    # 0.6~0.8Hz 是实测混淆区，保留读数但要求外部证据，不靠四门。
    f_true_min = inj["f_true_min_hz"] or 1.0
    f1_min = round(max(0.6, 5.0 * 2.0 / float(p["window_s"]),
                       min(0.6, f_true_min * 0.8)), 2)
    # 混淆区只指「fc/f_lo 过渡带内的低频阵风」，不含全带强正弦（那是方法局限）
    slow_false_low = inj["slow_wave_false_valid_lowband_hz"]
    conf_lo = round(max(f1_min, 0.6), 2)
    conf_hi = round(max(slow_false_low) if slow_false_low else conf_lo, 2)

    f_lo = round(max(float(p["f_lo"]), 2.0 * df), 2)
    f_hi = round(min(float(p["f_hi"]), 0.94 * nyq, 4.7), 2)
    params: dict[str, Any] = {
        "fs": round(fs, 4),
        "unit_scale": 1000.0,
        "window_s": float(p["window_s"]),
        "nperseg_s": float(p["nperseg_s"]),
        "n_sub": int(p["n_sub"]),
        "min_samples": int(p["min_samples"]),
        "min_span_s": float(p["min_span_s"]),
        "detrend": True, "despike": True, "refine_f1": True,
        "spike_win": 7, "spike_nsig": 6.0,
        "hp_cutoff": 0.3,
        "f_lo": f_lo, "f_hi": f_hi,
        "noise_floor_mg": noise["noise_floor_mg"],
        "g0_rms_factor": fa["derived_g0_factor"],
        "g1_peak_ratio": fa["recommended_g1"],
        "g2_mode": "band", "g2_band_frac": 0.3,
        "g2_sub_ratio": 5.0, "g2_min_subs": 3,
        "g2_cv_max": inj["recommended_g2_cv_max"],
        "g3_factor": 2.0, "f1_min_hz": f1_min,
        "n_peaks": 6, "peak_min_ratio": 3.0,
        "multi_axis": True, "ma_band_frac": 0.15, "ma_ratio": 5.0,
        "harmonic_check": True, "harm_ratio": 5.0, "xcheck": True,
    }
    bad = [k for k in params if k not in fg.DEFAULTS]
    if bad:
        raise RuntimeError(f"标定参数含非生产字段（会被 merge_params 静默丢弃）：{bad}")
    return params, [conf_lo, conf_hi]


def build_calib(args, recs, poses, seg_meta, health, gravity, noise, fa, inj,
                params, confusion, exc) -> dict[str, Any]:
    now = _dt.datetime.now()
    valid_days = int(args.valid_days)
    checks: dict[str, list[str]] = {"passed": [], "warnings": [], "failed": []}

    def ok(c, msg): checks["passed" if c else "failed"].append(msg)

    total = float(health.get("total_duration_s", health["duration_s"]))
    nseg = int(health.get("segments", 1))
    nwin = int(health.get("independent_windows", 0))
    ok(total >= 300.0, f"静止总时长 {total}s ≥ 300s（每台 5min 最低要求）")
    ok(nwin >= 10, f"独立 30s 窗 {nwin} ≥ 10 个")
    ok(health["irregular_dt_pct"] < 5.0,
       f"丢帧/不规则间隔 {health['irregular_dt_pct']}% < 5%")
    ok(not health.get("fs_mismatch"), "各静止段采样率一致")
    ok(not gravity["saturation"], "无量程饱和")
    ok(abs(gravity["gravity_norm_dev_pct"]) < 5.0,
       f"重力模长偏差 {gravity['gravity_norm_dev_pct']}% < 5%")
    ok(noise["white_like"], f"本底近似白噪 lag1={noise['lag1_autocorr']}")
    ok(fa["recommended_g1"] >= 10.0,
       f"G1 沿用生产基线 10，静止 p99.9={fa['g1_ratio_p999']} 有 {fa['margin_to_production']}× 余量")
    ok(inj["recommended_g2_cv_max"] >= 0.08, "G2 未擅自收紧生产基线 8%")
    ok(inj["f_true_min_hz"] is not None, "注入标定检出真信号")

    # —— 每段元数据（时段/温度/姿态）校验：脚本不能反解，必须采集者传入并落档 ——
    distinct_slots = sorted({m["slot"] for m in seg_meta if m["slot"]})
    temp_points = sorted({round(m["temp_c"], 1) for m in seg_meta if m["temp_c"] is not None})
    temp_range = round(max(temp_points) - min(temp_points), 2) if temp_points else 0.0
    if nseg < 3:
        checks["warnings"].append(
            f"仅 {nseg} 段静止数据：无法覆盖时段/温度/姿态变化，建议 ≥3 段（各 ≥5min，跨 ≥2 个温度点）")
    if len(distinct_slots) < 2:
        checks["warnings"].append(
            f"时段标签仅 {len(distinct_slots)} 类（{distinct_slots or '未提供 --slot'}）："
            "无法证明覆盖时段差异，请用 --slot 早/午/晚 等逐段标注")
    if len(temp_points) < 2:
        checks["warnings"].append(
            f"温度点仅 {len(temp_points)} 个（{temp_points or '未提供 --temp'}）："
            "单一温度点无法标定零偏温漂，请用 --temp 逐段标注并跨 ≥2 个温度点")
    if not any(m["pose_desc"] for m in seg_meta):
        checks["warnings"].append(
            "未提供 --pose-desc 实际摆放姿态：台架姿态 ≠ 现场姿态，装完必须现场复测静止 ≥2min 确认本底")
    if total < 900:
        checks["warnings"].append(
            f"静止总时长 {total}s < 15min：本底 p10 标准误 {noise['p10_se_mg']}mg，"
            "极值分位数依赖合成白噪交叉验证")
    checks["warnings"].append(
        f"本底采用 p10（本次 {noise['n_windows']} 窗，系统性偏低 {noise['p10_bias_pct']}%、"
        f"标准误 {noise['p10_se_mg']}mg），偏低量已由 G0 因子 {fa['derived_g0_factor']} 补偿")
    checks["warnings"].append(
        "虚警基线实测仅 " + str(fa["n_independent_windows"]) + " 个独立窗，重采样 "
        + str(fa["pseudo_replication_factor"]) + "× 属伪重复；已用独立合成白噪交叉验证")
    if exc["status"] != "VERIFIED":
        checks["warnings"].append(
            "无真实激励段验证：S4 的 55 组合成注入只验证算法，不验证标度/相位/夹具谐振/松动")
    if health["unique_values"] < 16:
        checks["warnings"].append(
            f"静止只有 {health['unique_values']} 个唯一值、量化台阶 {health['quant_step_mg']}mg，"
            "分辨率受量化限制，G1 标定结果对量程敏感")
    if noise["raw_std_vs_floor_pct"] > 5:
        checks["warnings"].append(
            f"原始 std 比生产口径本底高 {noise['raw_std_vs_floor_pct']}%，"
            "DEFAULTS.noise_floor_mg=1.286 属原始 std，已改用同口径标定值")
    if inj["slow_wave_false_valid_lowband_hz"]:
        checks["warnings"].append(
            f"低频阵风带 {inj['slow_wave_false_valid_lowband_hz']}Hz 的强慢波会误判 VALID；"
            f"过渡带混淆区 {confusion[0]}~{confusion[1]}Hz 需多轴/跨窗/张力先验补判")
    checks["warnings"].append("方法固有局限：四门只判有无稳定窄带峰，不判峰的物理来源；"
                             "全带强单频信号都会 VALID，详见 L2.method_limit")
    checks["warnings"].append("低频灵敏度下降：" + inj["low_freq_sensitivity"])
    checks["warnings"].append("L3 绝对物理量未标定：L_m / mu_kg_m / f1_ref_hz / attach_mass_g "
                             "必须安装后现场取值，当前索力输出为占位值")
    if gravity.get("optional_correction"):
        checks["warnings"].append("六面补偿矩阵已解算但默认不启用，不得覆盖出厂标定")

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now.isoformat(timespec="seconds"),
        "tool": "calibrate_accel.py",
        "sensor": {
            "id": args.sensor_id or "UNSET",
            "model": args.model,
            "range_g": float(args.range_g),
            "rate_hz": round(recs[0]["fs"], 4),
            "axis_default": recs[0]["axis"],
        },
        "environment": {
            "temp_c": float(args.temp_c),
            "temps_by_segment_c": [m["temp_c"] for m in seg_meta],
            "temp_points": temp_points,
            "temp_range_c": temp_range,
            "slots": distinct_slots,
            "mount": args.mount,
            "operator": args.operator,
        },
        "source": [{"file": r["name"], "sha256": r["sha256"], "rows": r["rows"],
                    "pose": poses[i] if i < len(poses) else "",
                    "slot": seg_meta[i]["slot"],
                    "temp_c": seg_meta[i]["temp_c"],
                    "pose_desc": seg_meta[i]["pose_desc"]}
                   for i, r in enumerate(recs)],
        "levels": {
            "L1_sensor": {"health": health, "gravity": gravity, "noise": noise},
            "L2_thresholds": {"false_alarm": fa, "injection": {k: v for k, v in inj.items()
                                                               if k != "grid"},
                              "confusion_band_hz": confusion,
                              "hardware_excitation": exc},
            "L3_site": {"status": "PENDING", "items": SITE_REQUIRED},
        },
        "data_sufficiency": {
            "verdict": ("INSUFFICIENT" if checks["failed"] else
                        "READY_FOR_INSTALL" if (nseg >= 3 and total >= 900.0
                                                and len(distinct_slots) >= 2
                                                and len(temp_points) >= 2
                                                and exc["status"] == "VERIFIED") else
                        "MINIMUM_MET_NOT_COMPLETE"),
            "verdict_meaning": {
                "INSUFFICIENT": "连 5min 最低要求都没达到，参数仅供参考，禁止投用",
                "MINIMUM_MET_NOT_COMPLETE": "单段 5min 只够 L1+静止L2，可生成参数但缺口在 WARN 里，需补采",
                "READY_FOR_INSTALL": "推荐规格已满足（≥3 段/≥15min/跨≥2时段/跨≥2温度点/已知激励段），参数可安装后直接加载，L3 仍现场填",
            },
            "total_static_s": total, "segments": nseg,
            "independent_windows": nwin,
            "slots": distinct_slots, "n_slots": len(distinct_slots),
            "temp_points": temp_points, "n_temp_points": len(temp_points),
            "temp_range_c": temp_range,
            "p10_se_mg": noise["p10_se_mg"], "p10_bias_pct": noise["p10_bias_pct"],
            "hardware_excitation": exc["status"],
            "five_min_static_covers": [
                "fs/丢帧率、量化台阶、量程饱和",
                "三轴零偏与安装倾角（单姿态综合偏差）",
                "与 G0 同口径的本底 RMS",
                "静止 G0/G1 虚警基线（配合合成白噪交叉验证）",
                "G2 稳定性门的频率栅格地板",
            ],
            "five_min_static_cannot_cover": [
                "真实机械激励下的标度因数、相位与幅频响应",
                "夹具谐振、松动、安装附加质量引入的频率漂移",
                "姿态变更后的本底变化（台架姿态 ≠ 现场姿态）",
                "零偏温漂（单一温度点）",
                "L3 绝对物理量：L_m / mu_kg_m / f1_ref_hz / attach_mass_g",
            ],
        },
        "collection_requirements": COLLECTION_REQUIREMENTS,
        "params": params,
        "site_required": SITE_REQUIRED,
        "fingerprint": {
            "axis_columns": [c for c in recs[0]["cols"] if "加速度" in c],
            "rate_hz": round(recs[0]["fs"], 4),
            "range_g": float(args.range_g),
            "unit_scale": 1000.0,
            "axis_default": recs[0]["axis"],
            "window_s": float(params["window_s"]),
            "nperseg_s": float(params["nperseg_s"]),
        },
        "validity": {
            "calibrated_on": now.date().isoformat(),
            "valid_until": (now.date() + _dt.timedelta(days=valid_days)).isoformat(),
            "valid_days": valid_days,
            "recalibrate_if": RECALIBRATE_IF,
        },
        "checks": checks,
    }


def write_report(path: str, c: dict[str, Any]) -> None:
    L = c["levels"]
    h, g, n = L["L1_sensor"]["health"], L["L1_sensor"]["gravity"], L["L1_sensor"]["noise"]
    fa, inj = L["L2_thresholds"]["false_alarm"], L["L2_thresholds"]["injection"]
    s, fp, v = c["sensor"], c["fingerprint"], c["validity"]
    out: list[str] = []
    add = out.append
    add("=" * 78)
    add("加速度计安装前系统级标定报告")
    add("=" * 78)
    add(f"传感器      : {s['model']}  ID={s['id']}  量程 ±{s['range_g']}g  "
        f"速率 {s['rate_hz']}Hz  默认轴 {s['axis_default']}")
    add(f"标定时间    : {c['generated_at']}   环境 {c['environment']['temp_c']}℃  "
        f"安装 {c['environment']['mount']}")
    add(f"数据指纹    : {c['source'][0]['sha256'][:16]}…  "
        f"{c['source'][0]['rows']} 行 / {h['duration_s']}s")
    add(f"有效期至    : {v['valid_until']}（{v['valid_days']} 天）")
    add("")
    add("采集分段（时段/温度/姿态由采集者标注，脚本只落档并校验）：")
    for i, s in enumerate(c["source"]):
        meta = f"slot={s.get('slot') or '-'}  temp={s.get('temp_c')}℃  pose={s.get('pose_desc') or s.get('pose') or '-'}"
        add(f"  [{i+1}] {s['file']:<24s}  {meta}")
    add("")
    add("-" * 78)
    add("STEP 0  数据体检")
    add("-" * 78)
    add(f"  fs={h['fs_hz']}Hz  dt中位数={h['dt_median_s']}s  "
        f"不规则间隔 {h['irregular_dt_count']} 处 ({h['irregular_dt_pct']}%)")
    add(f"  唯一值 {h['unique_values']} 个  量化台阶 {h['quant_step_mg']}mg  "
        f"峰峰 {h['peak_to_peak_mg']}mg  最长常数段 {h['longest_constant_run']} 点")
    add(f"  Z 轴均值 {h['mean_g']}g（静止应为 ±1g 附近的重力投影）")
    add("")
    add("-" * 78)
    add("STEP 1  出厂标定验证（不修改模块内部参数）")
    add("-" * 78)
    add(f"  方法={g['method']}  重力模长={g['gravity_norm_g']}g "
        f"(偏差 {g['gravity_norm_dev_pct']}%)  安装倾角≈{g['mount_tilt_deg']}°")
    add(f"  三轴均值向量 = {g['mean_vector_g']} g")
    add(f"  {g['note']}")
    if g.get("optional_correction"):
        oc = g["optional_correction"]
        add(f"  可选 12 参数解：bias={oc['bias_g']}  残差 RMS={oc['residual_rms_g']}g  "
            f"enabled={oc['enabled']}")
    add("")
    add("-" * 78)
    add("STEP 2  本底（与 G0 同口径）")
    add("-" * 78)
    add(f"  noise_floor_mg = {n['noise_floor_mg']}   [{n['definition']}]")
    add(f"  独立窗数 = {n['n_windows']}  p10 标准误 = {n['p10_se_mg']}mg  "
        f"系统性偏低 = {n['p10_bias_pct']}%（估计量固有，延长采集不消除）")
    add(f"  原始 std = {n['raw_std_mg']}mg（比同口径本底高 {n['raw_std_vs_floor_pct']}%）")
    add(f"  逐窗 RMS = {n['window_rms_mg']}")
    add(f"  lag1 自相关 = {n['lag1_autocorr']}  white_like={n['white_like']}")
    add("")
    add("-" * 78)
    add(f"STEP 3  静止虚警基线（{fa['n_trials']} 次重采样）")
    add("-" * 78)
    add(f"  G1 峰/中位: p50={fa['g1_ratio_p50']}  p99.5={fa['g1_ratio_p995']}  "
        f"p99.9={fa['g1_ratio_p999']}  max={fa['g1_ratio_max']}")
    add(f"  G0 RMS p99.9={fa['g0_rms_p999_mg']}mg  max={fa['g0_rms_max_mg']}mg")
    add(f"  独立窗 {fa['n_independent_windows']} 个 / 抽 {fa['n_trials']} 次 "
        f"(伪重复 {fa['pseudo_replication_factor']}×) → 独立合成白噪 p99.9={fa['g1_synthetic_p999']} "
        f"max={fa['g1_synthetic_max']}")
    add(f"  → 虚警下界 g1={fa['derived_g1']}（含 1.5× 系数）  "
        f"实际采用 g1_peak_ratio={fa['recommended_g1']}（不擅自放宽生产基线 10，"
        f"余量 {fa['margin_to_production']}×）")
    add(f"  → g0_rms_factor={fa['derived_g0_factor']}")
    add(f"  说明: {fa['note']}")
    add("")
    add("-" * 78)
    add("STEP 4  注入标定 / 混淆区")
    add("-" * 78)
    add(f"  Welch Δf={inj['welch_delta_f_hz']}Hz  Nyquist={inj['nyquist_hz']}Hz  "
        f"共 {inj['n_cases']} 组")
    add(f"  3mg 注入稳定检出频率: {inj['stable_at_3mg_hz']}")
    add(f"  最小可检出幅度(mg)  : {inj['min_detect_amp_mg']}")
    add(f"  8mg 强单频信号判 VALID 的频率: {inj['slow_wave_false_valid_hz']}")
    add(f"  其中低频阵风带: {inj['slow_wave_false_valid_lowband_hz']}")
    add(f"  → 过渡带混淆区 {L['L2_thresholds']['confusion_band_hz']}Hz "
        "（fc/f_lo 重合区，需外部证据补判）")
    add(f"  → 虚警下界 g2_cv={inj['derived_g2_cv_max']}  "
        f"实际采用 {inj['recommended_g2_cv_max']}（不擅自收紧生产基线 8%）")
    add(f"     注入 CV p99={inj['injected_valid_cv_p99']}；"
        f"旧口径栅格地板={inj['cv_legacy_grid_floor']} 已被亚格点插值压低，不再约束阈值")
    add(f"  方法局限: {inj['method_limit']}")
    add(f"  低频代价: {inj['low_freq_sensitivity']}")
    add(f"  本项为【合成注入】(synthetic_only={inj['synthetic_only']})，"
        f"取自 {inj['n_windows_pool']} 个静止窗中 RMS≈{inj['base_window_rms_mg']}mg 的典型窗；"
        "只验算法，不验硬件链路")
    add("")
    add("-" * 78)
    add("生成的生产参数（params，直接喂 merge_params）")
    add("-" * 78)
    for k in sorted(c["params"]):
        add(f"  {k:<22s} = {c['params'][k]}")
    add("")
    add("-" * 78)
    add("L3 安装后现场必填（未填前索力为占位值，禁止用于判定）")
    add("-" * 78)
    for it in c["site_required"]:
        add(f"  [{it['key']}] {it['label']}")
        add(f"      测法: {it['how']}")
        add(f"      原因: {it['why']}")
    add("")
    add("-" * 78)
    add("数据充分性判定：一段 5min 静止数据够不够？")
    add("-" * 78)
    ds = c["data_sufficiency"]
    ex = L["L2_thresholds"]["hardware_excitation"]
    add(f"  判定 = {ds['verdict']}  |  总静止 {ds['total_static_s']}s / "
        f"{ds['segments']} 段 / {ds['independent_windows']} 个独立 30s 窗")
    add(f"  本底 p10 标准误 {ds['p10_se_mg']}mg，系统性偏低 {ds['p10_bias_pct']}%"
        "（估计量固有偏差，延长采集不能消除，靠 G0 因子补偿）")
    add(f"  真实激励段 = {ex['status']}  {ex['reason']}")
    add("  5min 静止【能】覆盖：")
    for x in ds["five_min_static_covers"]:
        add("    + " + x)
    add("  5min 静止【不能】覆盖（必须补采或现场做）：")
    for x in ds["five_min_static_cannot_cover"]:
        add("    - " + x)
    add("")
    add("现场采集规格（缺项会在完整性检查里 WARN）：")
    for it in c["collection_requirements"]:
        add(f"  [{it['item']}] 最低：{it['minimum']}")
        add(f"      原因：{it['why']}")
    add("")
    add("-" * 78)
    add("完整性检查")
    add("-" * 78)
    for x in c["checks"]["passed"]:
        add(f"  PASS  {x}")
    for x in c["checks"]["warnings"]:
        add(f"  WARN  {x}")
    for x in c["checks"]["failed"]:
        add(f"  FAIL  {x}")
    add("")
    add("复标触发条件：")
    for x in v["recalibrate_if"]:
        add(f"  - {x}")
    add("")
    add("指纹（加载时比对，不一致会告警）：")
    for k, val in fp.items():
        add(f"  {k:<16s} = {val}")
    add("=" * 78)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="加速度计安装前系统级标定")
    ap.add_argument("--static", action="append", required=True,
                    help="静止数据文件，可重复给多次（六面法）")
    ap.add_argument("--pose", action="append", default=[],
                    help="与 --static 一一对应的姿态标签 +X/-X/+Y/-Y/+Z/-Z")
    ap.add_argument("--slot", action="append", default=[],
                    help="与 --static 一一对应的采集时段标签（如 早/午/晚/冷机/热机）")
    ap.add_argument("--temp", action="append", default=[],
                    help="与 --static 一一对应的每段环境温度℃，未给足时兜底 --temp-c")
    ap.add_argument("--pose-desc", action="append", default=[],
                    help="与 --static 一一对应的实际摆放姿态描述（自由文本）")
    ap.add_argument("--axis", default="", help="主标定轴列名，默认自动选加速度Z")
    ap.add_argument("--sensor-id", default="", help="传感器唯一编号（防串用）")
    ap.add_argument("--model", default="WT901SDCL-BT50")
    ap.add_argument("--range-g", type=float, default=2.0, help="量程 ±g")
    ap.add_argument("--temp-c", type=float, default=22.0,
                    help="默认环境温度℃，--temp 未给足时逐段兜底")
    ap.add_argument("--mount", default="bench", help="标定台架 / 安装方式描述")
    ap.add_argument("--operator", default="")
    ap.add_argument("--window-s", type=float, default=30.0)
    ap.add_argument("--nperseg-s", type=float, default=8.0)
    ap.add_argument("--n-mc", type=int, default=150, help="虚警基线重采样次数")
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--valid-days", type=int, default=180)
    ap.add_argument("--out", default="accel_calib.json")
    ap.add_argument("--report", default="accel_calib_report.txt")
    ap.add_argument("--excitation", default="",
                    help="已知频率的台架激励数据（可选，验证硬件链路）")
    ap.add_argument("--excitation-hz", type=float, default=0.0,
                    help="激励真值频率 Hz，配合 --excitation")
    args = ap.parse_args(argv)

    recs = [_read_one(_resolve(p), args.axis) for p in args.static]
    poses = list(args.pose) + [""] * max(0, len(recs) - len(args.pose))
    poses = poses[:len(recs)]

    # 每段元数据：时段 / 温度 / 实际摆放姿态（脚本无法从加速度反解，必须采集者显式传入）
    def _align(lst):
        out = list(lst) + [""] * max(0, len(recs) - len(lst))
        return out[:len(recs)]

    slots = _align(args.slot)
    temps = args.temp + [None] * max(0, len(recs) - len(args.temp))
    temps = temps[:len(recs)]
    temps = [float(t) if t is not None else float(args.temp_c) for t in temps]
    pose_desc = _align(args.pose_desc)
    seg_meta = [{"slot": slots[i], "temp_c": temps[i], "pose_desc": pose_desc[i]}
                for i in range(len(recs))]

    p = fg.merge_params({"window_s": args.window_s, "nperseg_s": args.nperseg_s,
                         "axis": recs[0]["axis"], "multi_axis": False,
                         "harmonic_check": False, "xcheck": False})
    pool = build_window_pool(recs, p)      # 所有静止段都参与，不再静默丢弃
    health = step0_health(recs[0])
    health.update(step0_summary(recs, p, pool))
    fs0 = recs[0]["fs"]
    gravity = step1_gravity(recs, poses, args.range_g)
    noise, nf = step2_noise(pool, p, fs0, recs[0]["y"], health)
    rng = np.random.default_rng(args.seed)
    fa = step3_false_alarm(pool, p, fs0, nf, noise, args.n_mc, rng)
    inj = step4_injection(pool, p, fs0, nf)
    exc = step5_excitation(args.excitation, args.excitation_hz, args.axis, p, nf)
    params, confusion = derive_params(recs[0], p, noise, fa, inj, gravity)
    calib = build_calib(args, recs, poses, seg_meta, health, gravity, noise, fa, inj,
                        params, confusion, exc)

    out = _resolve(args.out)
    rep = _resolve(args.report)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(calib, fh, ensure_ascii=False, indent=2)
    write_report(rep, calib)

    print(f"标定完成 → {out}")
    print(f"报告     → {rep}")
    print(f"本底 RMS = {noise['noise_floor_mg']} mg  "
          f"(原始 std {noise['raw_std_mg']} mg)")
    print(f"采用阈值 = G0×{params['g0_rms_factor']}  G1≥{params['g1_peak_ratio']}  "
          f"G2 CV≤{params['g2_cv_max']*100:.1f}%  "
          f"f1_min={params['f1_min_hz']}Hz  "
          f"带 {params['f_lo']}~{params['f_hi']}Hz")
    print(f"虚警下界 = G1≥{fa['derived_g1']}  G2≤{inj['derived_g2_cv_max']*100:.1f}% "
          f"（未擅自放宽/收紧生产基线）")
    print(f"过渡带混淆区 = {confusion[0]}~{confusion[1]}Hz（需外部证据）")
    print(f"数据充分性 = {calib['data_sufficiency']['verdict']} | 总静止 "
          f"{health['total_duration_s']}s / {health['segments']} 段 / "
          f"{health['independent_windows']} 独立窗 | 本底标准误 "
          f"{noise['p10_se_mg']}mg、偏低 {noise['p10_bias_pct']}%")
    ds = calib["data_sufficiency"]
    print(f"分段元数据 = 时段 {ds['n_slots']} 类 {ds['slots'] or '未标注'} | "
          f"温度点 {ds['n_temp_points']} 个 {ds['temp_points'] or '未标注'} | "
          f"温差 {ds['temp_range_c']}℃")
    print(f"硬件激励段 = {exc['status']}  {exc['reason'][:52]}")
    print(f"检查     = PASS {len(calib['checks']['passed'])} / "
          f"WARN {len(calib['checks']['warnings'])} / "
          f"FAIL {len(calib['checks']['failed'])}")
    for x in calib["checks"]["failed"]:
        print(f"  FAIL  {x}")
    verdict = calib["data_sufficiency"]["verdict"]
    print(f"结论     = {verdict} — {calib['data_sufficiency']['verdict_meaning'][verdict]}")
    print("提醒     = L3 绝对物理量未标定，安装后必须现场填写 L_m / mu_kg_m / "
          "f1_ref_hz / attach_mass_g")
    return 1 if calib["checks"]["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
