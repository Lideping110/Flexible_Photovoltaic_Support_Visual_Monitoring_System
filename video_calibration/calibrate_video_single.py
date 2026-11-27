# -*- coding: utf-8 -*-
"""视觉靶标「单段静止视频」标定 —— 用一段 ≥5min 的静止靶标视频标定四门阈值。

与 calibrate_accel_single.py 的对应关系
----------------------------------------
- 加速度版：一份 ≥5min 静止 CSV → 本底 + G0/G1/G2/G3 门限，安装后直接加载判定。
- 本脚本（视觉版）：一份静止靶标视频主链路产出的 displacement.jsonl →
  同一套 本底 + 四门阈值，供靶标振动分析（spectrum 链路）后续加载判定。

标定口径与算法完全复用 sensor_data/accelerometer/frequency_gate.py：
预处理（去尖峰→去趋势→高通）、Welch、四门、注入扫描全部直接调用 fg 的函数，
不另写一套 —— 避免「标定时一套、上线时另一套」（与 calibrate_accel 设计原则一致）。
唯一的分叉是数据入口：CSV 时间列 → displacement.jsonl 的逐帧累计位移。

信号表示（与 2026-11-21 分析决策一致）
--------------------------------------
- 输入用 cum_dx_mm / cum_dy_mm（0 阶累计位移，视觉天然量），不差分到加速度：
  f₁ 是阶次不变量，换阶只放大高频噪声（(2πf)²），视觉低频优势频段反而被打折。
- 窗内红谱/慢波问题交给预处理链解决：去趋势消温漂、0.3Hz 高通压慢波，
  与传感器拦温漂的手段同源。
- 位移/加速度两条链对标的是「天然量 + 窗内预处理 + 按本底标定的门控」框架，
  不是量纲；阈值单位为 mm（加速度版为 mg）。

输入怎么来（工作流）
-------------------
先用静止靶标视频跑一次主链路（config.yaml 中该路 marker 相机 enabled: true，
视频须为静止场景：结构不动、相机不动、光照稳定）：

    uv run python main.py --refresh-init

产物 output/marker/<cam_id>/displacement.jsonl 即本脚本的输入。

用法（项目根目录，必须用 .venv）
--------------------------------
    .venv/Scripts/python.exe video_calibration/calibrate_video_single.py \
        --static output/marker/marker_cam_01/displacement.jsonl \
        --cam-id marker_cam_01 --axis dy --video-desc "静止靶标视频 xxx.mp4"

    # 可选：已知频率激励段（如标定过的振动台/外部激励视频的主链路产物）
    ... --excitation output/marker/marker_cam_01_exc/displacement.jsonl --excitation-hz 2.0

    # 可叠加多份静止段（各 ≥5min 更佳），按段切窗不跨段
    ... --static a.jsonl --static b.jsonl

产物（默认写到本目录）：
    <cam_id>_calib.json           参数（结构兼容 frequency_gate.load_calibration）
    <cam_id>_calib_report.txt     人读报告

标定产出什么
------------
    noise_floor（mm，G0 基准）  g0_rms_factor  g1_peak_ratio  g2_cv_max  f1_min_hz
    分析频带 f_lo~f_hi（默认 0.3 ~ 0.94×Nyquist，随帧率自适应）
    + 数据体检：丢帧率、ID 跳变步数、漂移、逐靶标本底差异、白噪性
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ACCEL_DIR = os.path.normpath(os.path.join(_HERE, "..", "sensor_data", "accelerometer"))
if _ACCEL_DIR not in sys.path:
    sys.path.insert(0, _ACCEL_DIR)
import frequency_gate as fg   # noqa: E402  标定口径=生产口径：全部复用，不另写
import calibrate_accel as ca  # noqa: E402  复用切窗/汇总，保证窗结构与加速度版一致

ROOT = _HERE
_PROJECT_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
SCHEMA_VERSION = 1  # 与 frequency_gate.load_calibration 期望一致

# 注入扫描幅度（× 本底）与加速度版 (0.5,1,2,3,5)mg 同构：
# 视觉本底量级远小于传感器电噪声（亚像素 → ~0.1mm），固定 mg 幅度无意义，
# 按本底比例缩放使标定结果跨部署可迁移。
AMPS_REL = (0.5, 1.0, 2.0, 3.0, 5.0)
SLOW_AMP_REL = 8.0   # 强单频慢波（加速度版 8mg ≈ 6.8×本底）

RECALIBRATE_IF = [
    "更换相机/镜头、重新对焦或变焦、改变安装位置或支架",
    "改变分辨率、帧率、编码参数（码率/关键帧间隔/去畸变开关）",
    "更换或重训检测/亚像素模型（模型直接改变亚像素噪声分布）",
    "更换靶标或改变 marker 标定（real_diameter_mm），mm_per_px 偏离标定时 >5%",
    "环境光照/昼夜温度显著变化（热漂移会改变视觉本底）",
    "连续 7 天出现 G0 全部 FAIL 或静止段 RMS 漂移 >20%",
    "超过 valid_until，或相机/支架受过冲击、拆卸、维修",
]


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve(path: str) -> str:
    """相对路径按项目根解析（用户在项目根目录运行）；产物落在本目录。"""
    if os.path.isabs(path):
        return path
    cand = os.path.join(_PROJECT_ROOT, path)
    if os.path.isfile(cand) or not path.endswith((".json", ".txt")):
        return cand
    return os.path.join(ROOT, path)


def _f(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if x != x:  # NaN
        return None
    return x


# ---------------------------------------------------------------- 数据入口
def read_displacement(path: str, axis: str, min_samples: int = 32) -> list[dict[str, Any]]:
    """读 displacement.jsonl，按 pid 拆成「加速度版 rec 等价」的序列结构。

    - 序列 = cum_{axis}_mm（0 阶累计位移，mm），null（跟踪丢失帧）剔除；
    - 每靶标独立重采样到自身中位数 dt 的等间隔栅格（FFT/切窗前提），
      与 fg.build_series 的重采样逻辑同源；
    - 丢帧体检用 frame_id 跳号/缺失帧（视觉是「检测成功才记录」，ts 偶尔
      抖动是处理耗时，不能当丢帧）；ID 连续性用 raw_track_id + continuous
      （cum 台阶只是亚像素中心抖动，不等于 ID 切换，保留为诊断）。
    返回按 pid 排序的 rec 列表；无有效序列时抛错。
    """
    cum_key = f"cum_{axis}_mm"
    raw_key = "cy" if axis == "dy" else "cx"
    per_pid: dict[str, dict[str, list[float]]] = {}
    mmpp: dict[str, list[float]] = {}
    confs: dict[str, list[float]] = {}
    n_rows = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_rows += 1
            pid = r.get("pid")
            t = _f(r.get("ts"))
            y = _f(r.get(cum_key))
            m = _f(r.get("mm_per_px"))
            if pid is None or t is None or y is None:
                continue
            d = per_pid.setdefault(str(pid), {
                "t": [], "y": [], "raw": [], "fid": [], "rid": [], "cont": []})
            d["t"].append(t)
            d["y"].append(y)
            d["raw"].append(_f(r.get(raw_key)) or 0.0)
            d["fid"].append(int(_f(r.get("frame_id")) or 0))
            d["rid"].append(int(_f(r.get("raw_track_id")) or 0))
            d["cont"].append(bool(r.get("continuous")))
            if m is not None:
                mmpp.setdefault(str(pid), []).append(m)
            c = _f(r.get("conf"))
            if c is not None:
                confs.setdefault(str(pid), []).append(c)

    sha = _sha256(path)
    recs: list[dict[str, Any]] = []
    for pid in sorted(per_pid):
        d = per_pid[pid]
        t = np.asarray(d["t"], dtype=float)
        y = np.asarray(d["y"], dtype=float)
        raw = np.asarray(d["raw"], dtype=float)
        fid = np.asarray(d["fid"], dtype=np.int64)
        rid = np.asarray(d["rid"], dtype=np.int64)
        cont = np.asarray(d["cont"], dtype=bool)
        if len(t) < 3:
            continue
        order = np.argsort(t, kind="stable")
        t, y, raw = t[order], y[order], raw[order]
        fid, rid, cont = fid[order], rid[order], cont[order]
        keep = np.concatenate(([True], np.diff(t) > 1e-6))
        t, y, raw = t[keep], y[keep], raw[keep]
        fid, rid, cont = fid[keep], rid[keep], cont[keep]
        # 真丢帧：frame_id 跨度内缺失了多少帧（不是 ts 间隔抖动）
        fid_span = int(fid[-1] - fid[0]) + 1 if len(fid) else 0
        frame_lost = max(0, fid_span - len(fid))
        frame_jump_events = int(np.sum(np.diff(fid) > 1))
        # 真 ID 连续性：raw_track_id 跳变 + continuous 缺失
        id_switches = int(np.sum(rid[1:] != rid[:-1]))
        cont_bad = int(np.sum(~cont))
        dt = float(np.median(np.diff(t)))
        if dt <= 0:
            continue
        fs = 1.0 / dt
        n = max(8, int(round((t[-1] - t[0]) * fs)) + 1)
        tu = t[0] + np.arange(n) / fs
        yu = np.interp(tu, t, y)
        # 原始间隔（未重采样）用于丢帧体检
        diffs = np.diff(t)
        diffs = diffs[diffs > 0]
        # ID 跳变步数：差分的稳健 σ 外超限（持续台阶，非单帧尖峰）
        if len(diffs) and len(yu) > 8:
            dd = np.diff(yu)
            mad = float(np.median(np.abs(dd - np.median(dd))))
            sig = 1.4826 * mad
            n_steps = int(np.sum(dd > 5.0 * sig)) if sig > 0 else 0
            max_gap = float(np.max(diffs))
        else:
            n_steps, max_gap = 0, 0.0
        mm_pxp = float(np.median(mmpp.get(pid, [0.0]))) if mmpp.get(pid) else 0.0
        recs.append({
            "pid": pid, "t": tu, "y": yu, "raw_px": raw, "fs": float(fs),
            "dt_raw": diffs.tolist(), "mm_per_px": mm_pxp,
            "conf_mean": float(np.mean(confs.get(pid, [0.0]))),
            "n_valid": int(len(t)), "n_rows": n_rows,
            "id_steps": n_steps, "max_gap_s": round(max_gap, 4),
            "frame_span": fid_span, "frame_lost": frame_lost,
            "frame_jump_events": frame_jump_events,
            "id_switches": id_switches, "cont_bad": cont_bad,
            "drift_mm": round(float(yu[-1] - yu[0]), 4),
            "name": os.path.basename(path), "sha256": sha,
            "path": path,
        })
    if not recs:
        raise ValueError(
            f"{path} 中没有任何有效序列（cum_{axis}_mm 全为 null？"
            "请确认该次主链路运行时 marker 标定已就绪、靶标持续可见）")
    return recs


# ---------------------------------------------------------------- STEP 0
def step0_health_video(recs: list[dict[str, Any]]) -> dict[str, Any]:
    """视觉数据体检：帧率、丢帧、mm 标定、ID 跳变、漂移、逐靶标覆盖。"""
    all_dt = np.concatenate([np.asarray(r["dt_raw"], dtype=float) for r in recs])
    base = float(np.median(all_dt)) if len(all_dt) else 1.0 / recs[0]["fs"]
    irregular = int(np.sum(np.abs(all_dt - base) > 0.25 * base)) if len(all_dt) else 0
    irregular_pct = round(100.0 * irregular / max(len(all_dt), 1), 3)
    fs_all = [r["fs"] for r in recs]
    fs_med = float(np.median(fs_all))
    fs_mismatch = any(abs(f - fs_med) / fs_med > 0.05 for f in fs_all)

    mm_ok = all(r["mm_per_px"] > 0 for r in recs)
    mm_med = float(np.median([r["mm_per_px"] for r in recs]))
    # 亚像素中心是连续浮点，量化台阶按原始像素序列估计（诊断用）
    quant_px = 0.0
    for r in recs:
        u = np.unique(np.round(r["raw_px"], 6))
        s = np.diff(u)
        s = s[s > 1e-9]
        if len(s):
            quant_px = max(quant_px, float(np.median(s)))
            break

    total_steps = sum(r["id_steps"] for r in recs)
    total_valid = sum(r["n_valid"] for r in recs)
    # 真丢帧口径：frame_id 跨度内缺失帧占比。视觉是「检测成功才记录」，
    # ts 间隔受推理/解码/落盘耗时抖动（本例 0.013~0.068s），不能用它判丢帧。
    fid_span = sum(r["frame_span"] for r in recs)
    frame_lost = sum(r["frame_lost"] for r in recs)
    frame_loss_pct = round(100.0 * frame_lost / max(fid_span, 1), 3)
    frame_jump_events = sum(r["frame_jump_events"] for r in recs)
    # 真 ID 连续性口径：raw_track_id 跳变 + continuous=False，cum 台阶仅诊断。
    id_switches = sum(r["id_switches"] for r in recs)
    cont_bad = sum(r["cont_bad"] for r in recs)
    per_pid = [{
        "pid": r["pid"], "fs_hz": round(r["fs"], 4),
        "n_valid": r["n_valid"], "duration_s": round(r["n_valid"] / r["fs"], 1),
        "mm_per_px": round(r["mm_per_px"], 6),
        "conf_mean": round(r["conf_mean"], 4),
        "id_steps": r["id_steps"], "max_gap_s": r["max_gap_s"],
        "frame_span": r["frame_span"], "frame_lost": r["frame_lost"],
        "frame_jump_events": r["frame_jump_events"],
        "id_switches": r["id_switches"], "cont_bad": r["cont_bad"],
        "drift_mm": r["drift_mm"],
    } for r in recs]
    return {
        "n_pids": len(recs),
        "n_samples": int(total_valid),
        "duration_s": round(float(max(r["n_valid"] / r["fs"] for r in recs)), 3),
        "fs_hz": round(fs_med, 4),
        "dt_median_s": round(base, 6),
        # 下面是处理耗时抖动诊断，不是丢帧判据（见 frame_loss_pct）
        "irregular_dt_count": irregular,
        "irregular_dt_pct": irregular_pct,
        "frame_span_total": fid_span,
        "frame_lost_total": frame_lost,
        "frame_loss_pct": frame_loss_pct,
        "frame_jump_events": frame_jump_events,
        "id_switches_total": id_switches,
        "continuous_bad_total": cont_bad,
        "continuous_bad_pct": round(100.0 * cont_bad / max(total_valid, 1), 3),
        "fs_mismatch": bool(fs_mismatch),
        "mm_calibration_ready": bool(mm_ok),
        "mm_per_px_median": round(mm_med, 6),
        "quant_step_px": round(quant_px, 6),
        "quant_step_mm": round(quant_px * mm_med, 6) if quant_px > 0 else 0.0,
        "id_steps_total": total_steps,
        "id_steps_ratio": round(total_steps / max(total_valid, 1), 6),
        "per_pid": per_pid,
        "axis_column": "cum位移(mm)",
    }


# ---------------------------------------------------------------- STEP 2
def step2_noise_video(pool: list[np.ndarray], p: dict[str, Any], fs: float,
                      raw_concat: np.ndarray, health: dict[str, Any],
                      per_pid_pools: dict[str, list[np.ndarray]],
                      per_pid_pools_px: dict[str, list[np.ndarray]] | None = None,
                      pool_mm: list[np.ndarray] | None = None
                      ) -> tuple[dict[str, Any], float]:
    """视觉本底（核心在 px 域，与生产 gate_core 的 G0 判定域一致）。

    pool 是 px 域窗（cum_dy_mm / mm_per_px）；nf 从 px 窗 RMS 的 p10 推导，
    写入 noise_floor_mg 供生产 G0 比较。step3 的 G0 因子、step4 的注入幅度
    同用 px 域 nf —— 保证「标定 / 生产 / 验证」三者同域（2026-11-26 修复域分裂）。

    mm 域字段（noise_floor_mm / window_rms_mm 等）仅作体检与展示，不参与门控；
    per_pid_floor_px / domain_compare 保留双域差异分析。

    px 与 mm 仅差逐靶标常数 mm_per_px，预处理各步（去尖峰 σ 倍阈值、去趋势、
    高通）尺度线性，谱比值（G1/CV）域无关，故只有依赖绝对幅值的 G0 必须同域。
    """
    # 核心本底：px 域（与生产 gate_core G0 判定域一致）
    rms_win: list[float] = []
    pre_all: list[np.ndarray] = []
    for seg in pool:
        x, _ = fg.preprocess_window(seg, fs, p)
        pre_all.append(x)
        rms_win.append(float(np.sqrt(np.mean(x * x))))
    nf = float(np.percentile(rms_win, 10))  # px 本底 → noise_floor_mg
    x = np.concatenate(pre_all)
    # 白噪判据必须用一阶差分。输入是 cum_* 0 阶累计位移，白噪逐帧积分成随机游走后
    # 相邻值天然高相关（静止靶标也能到 lag1≈0.85），直接用累计残差判会误杀。
    # 一阶差分回到与传感器加速度同阶，才是"亚像素噪声是否独立"的正确判据。
    dx = np.diff(x)
    ac1_raw = float(np.corrcoef(x[:-1], x[1:])[0, 1]) if len(x) > 2 else 0.0
    ac1 = float(np.corrcoef(dx[:-1], dx[1:])[0, 1]) if len(dx) > 2 else 0.0
    # 原始 cum（未预处理）一阶差分的 lag1：典型为负相关（跟踪器阻尼/过冲），
    # 用来证明「视觉链路的差分不是独立白噪」是固有属性，不是本段数据异常。
    d0 = np.diff(raw_concat)
    ac1_raw_diff = (float(np.corrcoef(d0[:-1], d0[1:])[0, 1])
                    if len(d0) > 2 else 0.0)
    q = np.percentile(np.abs(x), [50, 95, 99, 100])

    k, npt = len(rms_win), len(pool[0])
    sigma = float(np.mean(rms_win))
    mc = sigma * np.sqrt(np.random.default_rng(0).chisquare(npt, size=(2000, k)) / npt)
    p10s = np.percentile(mc, 10, axis=1)

    per_pid_floor = {}
    for pid, segs in per_pid_pools.items():
        if not segs:
            continue
        rr = [float(np.sqrt(np.mean(fg.preprocess_window(s, fs, p)[0] ** 2)))
              for s in segs]
        per_pid_floor[pid] = round(float(np.percentile(rr, 10)), 4)
    floors = [v for v in per_pid_floor.values() if v > 0]
    spread = round(max(floors) / min(floors), 2) if len(floors) >= 2 else 1.0

    # ---- px 域本底体检：与 mm 域同函数、同窗结构，仅序列量纲不同 ----
    per_pid_floor_px: dict[str, float] = {}
    if per_pid_pools_px:
        for pid, segs in per_pid_pools_px.items():
            if not segs:
                continue
            rr = [float(np.sqrt(np.mean(fg.preprocess_window(s, fs, p)[0] ** 2)))
                  for s in segs]
            per_pid_floor_px[pid] = round(float(np.percentile(rr, 10)), 4)
    floors_px = [v for v in per_pid_floor_px.values() if v > 0]
    spread_px = round(max(floors_px) / min(floors_px), 2) if len(floors_px) >= 2 else 1.0
    ratio = round(spread / spread_px, 2) if spread_px > 0 else None
    if spread_px > 0 and spread > 1.5 * spread_px:
        conclusion = ("px 域差异显著小于 mm 域 → 差异主要来自 mm_per_px 量纲换算，"
                      "G0 可考虑改用相机级 px 本底判定（输出仍用 mm）")
    else:
        conclusion = ("px 域差异与 mm 域接近 → 差异来自靶标自身亚像素噪声，"
                      "G0 仍需靶标级本底")
    domain_compare = {
        "spread_px": spread_px,
        "spread_mm": spread,
        "spread_ratio_mm_over_px": ratio,
        "per_pid_mm_per_px": {pid: round(mm, 6) for pid, mm in
                              ((r["pid"], r["mm_per_px"]) for r in health.get("per_pid", []))
                              if mm > 0},
        "conclusion": conclusion,
        "caveat": ("仅凭本段静止视频的 px 域差异小，尚不足以证明跨光照/跨距离下 px 本底恒定；"
                   "需多相机/多光照补标后再决定是否正式改相机级"),
    }

    # mm 域体检字段（不参与门控，仅展示/漂移检查）：用 pool_mm 单独算
    rms_win_mm: list[float] = []
    for seg in (pool_mm or []):
        xm, _ = fg.preprocess_window(seg, fs, p)
        rms_win_mm.append(float(np.sqrt(np.mean(xm * xm))))
    nf_mm = float(np.percentile(rms_win_mm, 10)) if rms_win_mm else 0.0
    sigma_mm = float(np.mean(rms_win_mm)) if rms_win_mm else 0.0

    info = {
        "noise_floor_px": round(nf, 4),
        "noise_floor_mm": round(nf_mm, 4),
        "noise_floor_mg": round(nf, 4),   # fg 键名；生产 G0 在 px 域判定，故装 px 本底
        "unit": "px",
        "definition": ("预处理(去尖峰→去趋势→0.3Hz高通)后逐30s窗 RMS 的 10 分位，"
                       "px 域（与生产 gate_core G0 判定域一致）；mm 域值仅作体检"),
        "n_windows": k,
        # 核心（px 域，参与门控）
        "pre_rms_mean_px": round(sigma, 4),
        "window_rms_px": [round(v, 4) for v in rms_win],
        "abs_percentile_px": {"p50": round(float(q[0]), 4), "p95": round(float(q[1]), 4),
                              "p99": round(float(q[2]), 4), "max": round(float(q[3]), 4)},
        "p10_se_px": round(float(np.std(p10s)), 4),
        "p10_bias_pct": round((float(np.mean(p10s)) / max(sigma, 1e-9) - 1) * 100, 2),
        # mm 域体检（不参与门控）
        "pre_rms_mean_mm": round(sigma_mm, 4),
        "window_rms_mm": [round(v, 4) for v in rms_win_mm],
        "raw_std_mm": round(float(np.std(raw_concat)), 4),
        "raw_std_vs_floor_pct": round((float(np.std(raw_concat)) / max(nf_mm, 1e-9) - 1) * 100, 2),
        "lag1_autocorr": round(ac1, 4),
        "white_like": bool(abs(ac1) < 0.15),  # 诊断字段，不再作为充分性判据
        "white_like_basis": ("预处理残差的一阶差分（cum 位移积分后 lag1 天然高，不作判据）；"
                             "视觉 cum 链路该值仅诊断，不作 READY 门槛"),
        "lag1_autocorr_cumulative": round(ac1_raw, 4),
        "lag1_raw_cum_diff": round(ac1_raw_diff, 4),
        "quant_step_mm": float(health.get("quant_step_mm") or 0.0),
        "per_pid_floor_mm": per_pid_floor,
        "per_pid_floor_spread": spread,
        "per_pid_floor_px": per_pid_floor_px,
        "per_pid_floor_spread_px": spread_px,
        "domain_compare": domain_compare,
    }
    return info, nf


# ---------------------------------------------------------------- STEP 3
def _synth_peak_ratios(rng: np.random.Generator, sigma: float, n: int, t: np.ndarray,
                       fs: float, p: dict[str, Any], nf: float,
                       n_trials: int) -> list[float]:
    """合成白噪（sigma 与 pool/nf 同为 px 域）过同一 analyze_window 的 peak_ratio 分布。"""
    out: list[float] = []
    for _ in range(int(n_trials)):
        w = rng.normal(0.0, float(sigma), n)
        r = fg.analyze_window(0, t, w, fs, p, nf)
        if r["peak_ratio"] is not None:
            out.append(float(r["peak_ratio"]))
    return out or [0.0]


def step3_false_alarm_video(pool: list[np.ndarray], p: dict[str, Any], fs: float,
                            nf: float, noise: dict[str, Any], n_mc: int,
                            rng: np.random.Generator) -> dict[str, Any]:
    """静止虚警基线：数学与 ca.step3_false_alarm 逐步一致。"""
    n = len(pool[0])
    t = np.arange(n) / fs
    ratios: list[float] = []
    rms: list[float] = []
    static_f1: list[float] = []
    for _ in range(n_mc):
        seg = pool[int(rng.integers(0, len(pool)))]
        r = fg.analyze_window(0, t, seg, fs, p, nf)
        if r["peak_ratio"] is not None:
            ratios.append(float(r["peak_ratio"]))
        if r["rms_mg"] is not None:
            rms.append(float(r["rms_mg"]))
        if r["f1_hz"] is not None:
            static_f1.append(float(r["f1_hz"]))
    ratios = ratios or [0.0]
    rms = rms or [nf]
    p995, p999 = float(np.percentile(ratios, 99.5)), float(np.percentile(ratios, 99.9))
    r50 = float(np.percentile(rms, 50))
    r999 = float(np.percentile(rms, 99.9))
    # ---- G0 因子改用「典型窗 RMS」（中位），不用 p99.9 极端分位（2026-11-27）----
    # 旧式 max(1.2, r999/nf×1.1) 让最差的一个静止窗（多为慢漂段）定全局：
    # A 得 2.61、B 得 4.31，B 的 G0 阈值 0.185px 把 0.2px 真振动检出压到 8%，
    # 而同一批数据在 G0 从 1.2 到 4.3 全程静止虚警都是 0/40——高 G0 一点虚警都没省。
    # 原因是慢漂峰已由 G3 慢漂上沿拦住，白噪由 G1 拦，G0 只需做「不是死寂」的粗幅值下限。
    g0_factor = round(max(1.2, r50 / max(nf, 1e-9) * 1.15), 2)

    # ---- 慢漂频段上沿：G3 绝对下限的依据（2026-11-27）----
    # 静止窗里没有真实振动，凡是它产出「峰」都是伪峰：0.3Hz 高通后的准静态慢漂残留。
    # 实测两路静止窗 f1 全部落在 0.38~1.07Hz（A/B 一致），而 f1_min=0.4 时
    # 交叉验证出现 2/40 静止误报，误报主频 0.53/0.57Hz 正好在这个频段。
    # 所以 G3 绝对下限必须高于该频段上沿，否则慢漂峰以 VALID 放行。
    # 用上沿×1.15 留格点余量（Welch Δf=1/nperseg_s）。
    s_max = max(static_f1) if static_f1 else None
    drift_edge = round(s_max * 1.15, 2) if s_max else None

    # 独立合成白噪（视觉亚像素中心为连续值，量化不构成限制，不加量化台阶）
    # pool 与 nf 均为 px 域，sigma 必须同域，否则合成虚警基线被量纲放大/缩小
    sigma = float(noise.get("pre_rms_mean_px") or noise["pre_rms_mean_mm"])
    syn = _synth_peak_ratios(rng, sigma, n, t, fs, p, nf, max(n_mc, 300))
    s999 = float(np.percentile(syn, 99.9))

    # ---- G1 基线改用「边界白噪」，不用实测静止窗（2026-11-27 修正）----
    # 实测静止窗的 peak_ratio 被 0.3~1Hz 准静态慢漂污染：峰频中位仅 0.6~0.7Hz，
    # ratio 可达 160~430（合成白噪 p99.9 只有 4.1）。用实测 p99.9 定 G1 会把阈值
    # 推到 242/642，而静止虚警并未减少一个（静止窗本来就卡在 G0），只把真振动检出率
    # 压死（0.3px 注入由 97.5% 降到 61%）。慢漂的幅度放大已由 G0 幅度门拦住，
    # G1 只需拦住「能过 G0 的最弱非信号」= 边界白噪，其 sigma 取最终 G0 阈值。
    # 该基线只依赖谱参数(nperseg_s/带宽/帧率)，与视频、靶标数无关，故多段数据会收敛。
    sigma_bnd = g0_factor * nf
    bnd = _synth_peak_ratios(rng, sigma_bnd, n, t, fs, p, nf, max(n_mc, 300))
    b999 = float(np.percentile(bnd, 99.9))
    bmax = float(np.max(bnd))
    recommended_g1 = round(max(6.0, b999 * 2.0), 1)
    return {
        "n_trials": n_mc,
        "n_independent_windows": len(pool),
        "pseudo_replication_factor": round(n_mc / max(1, len(pool)), 2),
        # 实测静止窗分布：仅作诊断（含慢漂），不参与 G1 推导
        "g1_ratio_p50": round(float(np.percentile(ratios, 50)), 3),
        "g1_ratio_p995": round(p995, 3),
        "g1_ratio_p999": round(p999, 3),
        "g1_ratio_max": round(float(np.max(ratios)), 3),
        "g1_synthetic_p999": round(s999, 3),
        "g1_synthetic_max": round(float(np.max(syn)), 3),
        # 边界白噪（sigma = G0 阈值）：G1 的真正基线
        "g1_boundary_sigma_px": round(sigma_bnd, 4),
        "g1_boundary_p999": round(b999, 3),
        "g1_boundary_max": round(bmax, 3),
        # 慢漂频段（G3 绝对下限依据）
        "static_f1_max_hz": round(s_max, 4) if s_max else None,
        "drift_edge_hz": drift_edge,
        "g0_rms_p50_px": round(r50, 4),
        "g0_rms_p999_px": round(r999, 4),
        "g0_rms_max_px": round(float(np.max(rms)), 4),
        "g1_source": "boundary_white_noise",
        "derived_g1": recommended_g1,
        "recommended_g1": recommended_g1,
        "margin_to_production": round(10.0 / max(p999, 1e-9), 2),
        "derived_g0_factor": g0_factor,
        "note": ("G1 = max(6, 边界白噪(sigma=G0阈值) peak_ratio p99.9 × 2)。边界白噪代表"
                 "「刚好能过 G0 的最弱非信号」，G1 只需拦住它；实测静止窗 p99.9 含 0.3~1Hz "
                 "慢漂，仅作诊断。G0 = 典型窗RMS(p50)×1.15/本底(p10)，只做粗幅值下限"
                 "（旧 p99.9 极端分位会砍灵敏度）。G3 绝对下限取静止窗 f1 上沿×1.15。"
                 "换 nperseg_s / f_lo / f_hi / 帧率会改变带内格点数，必须重标"),
    }


# ---------------------------------------------------------------- STEP 4
def step4_injection_video(pool: list[np.ndarray], p: dict[str, Any], fs: float,
                          nf: float) -> dict[str, Any]:
    """注入标定（幅度按本底比例缩放）：结构、判定与 ca.step4_injection 一致。"""
    rr = np.asarray([float(np.sqrt(np.mean(fg.preprocess_window(s, fs, p)[0] ** 2)))
                     for s in pool])
    base = pool[int(np.argmin(np.abs(rr - float(np.median(rr)))))]
    n = len(base)
    t = np.arange(n) / fs
    # 注入幅度网格 = 本底倍数 ∪ G0 阈值倍数。
    # 只用本底倍数时，G0 因子一旦大于 2 倍，0.5/1/2×nf 全部低于 G0 阈值必然检不出，
    # min_detect_amp 只能跳到 3 倍以上，粗糙到看不出真实检出下限；补一组围绕
    # G0 阈值(0.8/1/1.2/1.5/2×)的档位，让扫描在检出边界上有分辨率。
    g0_thr = float(p["g0_rms_factor"]) * nf
    amps_px = tuple(sorted({round(a * nf, 4) for a in AMPS_REL}
                           | {round(k * g0_thr, 4) for k in (0.8, 1.0, 1.2, 1.5, 2.0)}))  # px 域
    nyq = fs / 2.0
    freqs = tuple(f for f in (0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0,
                              6.0, 8.0, 10.0) if f < 0.9 * nyq)
    grid: dict[str, dict[str, Any]] = {}
    cvs: list[float] = []
    for a in amps_px:
        for f0 in freqs:
            inj = base + a * np.sin(2.0 * np.pi * f0 * t)
            r = fg.analyze_window(0, t, inj, fs, p, nf)
            grid[f"{a}|{f0}"] = {"ok": bool(r["valid"]),
                                 "f1": r["f1_hz"], "cv": r["cv"],
                                 "ratio": r["peak_ratio"]}
            if r["valid"] and r["cv"] is not None:
                cvs.append(float(r["cv"]))

    min_amp: dict[str, dict[str, Any]] = {}
    for f0 in freqs:
        hit = next((a for a in amps_px if grid[f"{a}|{f0}"]["ok"]), None)
        if hit is None:
            min_amp[str(f0)] = {"rel": 0.0, "px": 0.0}
        else:
            min_amp[str(f0)] = {"rel": round(hit / nf, 2), "px": hit}
    # 参考幅度改为「G0 阈值的 1.5 倍」而不是固定的「3× 本底」。
    # 门控在最终阈值下跑之后，固定 3× 本底可能整段低于 G0 阈值（本底 p10 偏低，
    # G0 因子 2.6~4.3 时 3× 本底 = 0.13px < 阈值 0.185px），全频段必然检不出，
    # 「注入检出真信号」会假失败。取证口径应是「刚越过幅度门、有物理意义的信号」。
    amp_ref = round(1.5 * g0_thr, 4)
    stable_ref = [f0 for f0 in freqs if grid[f"{amp_ref}|{f0}"]["ok"]]
    f_true_min = min(stable_ref) if stable_ref else None

    false_valid: list[float] = []
    amp_slow = round(SLOW_AMP_REL * nf, 4)
    slow_wave_amp_px = amp_slow
    for f0 in freqs:
        slow = base + amp_slow * np.sin(2.0 * np.pi * f0 * t)
        r = fg.analyze_window(0, t, slow, fs, p, nf)
        if r["valid"]:
            false_valid.append(f0)

    p99_cv = float(np.percentile(cvs, 99)) if cvs else 0.08
    df_sub = float(p["n_sub"]) / float(p["window_s"])
    legacy_floor_cv = 0.5 * df_sub / max(f_true_min or 1.0, 0.1)
    derived_cv = round(min(0.20, max(0.06, p99_cv * 1.5)), 3)
    recommended_cv = round(max(derived_cv, 0.08), 3)

    d_f = fs / max(1, int(p["nperseg_s"] * fs))
    return {
        "n_cases": len(grid),
        "amplitudes_rel": [round(a / nf, 3) for a in amps_px],   # 折算成本底倍数
        "amplitudes_nf_grid": list(AMPS_REL),                     # 原始本底倍数档
        "amplitudes_px": list(amps_px),
        "frequencies_hz": list(freqs),
        "min_detect_amp": min_amp,
        "stable_at_3xfloor_hz": stable_ref,      # 兼容旧键名：现为 1.5×G0 阈值
        "stable_at_ref_amp_hz": stable_ref,
        "ref_amp_px": amp_ref,
        "ref_amp_note": "参考幅度 = 1.5×G0 阈值（刚越过幅度门）",
        "f_true_min_hz": f_true_min,
        "slow_wave_false_valid_hz": false_valid,
        "slow_wave_amp_px": slow_wave_amp_px,
        "slow_wave_false_valid_lowband_hz": [f for f in false_valid if f <= 3 * float(p["f_lo"])],
        "injected_valid_cv_p99": round(p99_cv, 5) if cvs else None,
        "cv_legacy_grid_floor": round(legacy_floor_cv, 5),
        "derived_g2_cv_max": derived_cv,
        "recommended_g2_cv_max": recommended_cv,
        "method_limit": ("四门只判「窗内是否存在稳定窄带峰」，不判「这个峰是不是结构模态」。"
                         "视觉链路中检测框闪烁、JPEG/H.264 块效应等也可能形成窄带峰，"
                         "必须靠跨窗一致性、报警复核等外部证据补判。"),
        "low_freq_sensitivity": ("0.3Hz 高通与 f_lo 重合，低于 f_lo 的注入会被滤波代价抬高检出阈值；"
                                 "0.2Hz 档预期检不出，属滤波设计代价而非缺陷。"),
        "welch_delta_f_hz": round(float(d_f), 5),
        "nyquist_hz": round(nyq, 4),
        "grid": grid,
        "n_windows_pool": len(pool),
        "base_window_rms_px": round(float(np.median(rr)), 4),
        "synthetic_only": True,
        "note": ("注入幅度=本底×(0.5/1/2/3/5)，叠加在真实静止窗上；"
                 "只验证算法在本视觉本底下的可检出域，不验证相机/支架对真实振动的响应"),
    }


# ---------------------------------------------------------------- STEP 5
def step5_excitation_video(path: str, hz: float, axis: str, p: dict[str, Any],
                           nf: float) -> dict[str, Any]:
    """已知频率激励段（可选）：验证「检测+亚像素+mm换算」全链路能否读出真频率。"""
    if not path or hz <= 0:
        return {"status": "NOT_PROVIDED", "file": "", "excitation_hz": hz,
                "windows": 0, "matched": 0,
                "reason": "未提供已知频率的激励数据；L2 的注入标定为合成信号，只验证算法"}
    recs = read_displacement(_resolve(path), axis)
    pool = ca.build_window_pool(recs, p)
    # nf 是 px 域；激励窗也须转 px 域（cum_mm / mm_per_px），否则 G0 量纲错配
    pool = [seg / r["mm_per_px"] for r in recs for seg in ca.build_window_pool([r], p)
            if r["mm_per_px"] > 0]
    if not pool:
        return {"status": "FAILED", "file": os.path.basename(path), "excitation_hz": hz,
                "windows": 0, "matched": 0,
                "reason": "激励数据无有效 mm_per_px，无法转 px 域验证"}
    fs, n = recs[0]["fs"], len(pool[0])
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
        "reason": ("实测激励频率在 ±10% 内被稳定检出，视觉全链路（检测/亚像素/mm换算/FFT）可用"
                   if matched >= need else
                   "实测激励未稳定检出：检查激励幅度是否低于本底、靶标是否持续可见、"
                   "帧率是否混叠、或激励频率是否超出分析带"),
    }


# ---------------------------------------------------------------- 参数推导
def derive_params_video(recs: list[dict[str, Any]], p: dict[str, Any],
                        noise: dict[str, Any], fa: dict[str, Any],
                        inj: dict[str, Any]) -> tuple[dict[str, Any], list[float]]:
    """推导生产参数。键名必须全部属于 fg.DEFAULTS（否则 merge_params 会静默丢弃）。"""
    fs = float(np.median([r["fs"] for r in recs]))
    nyq = fs / 2.0
    mm_med = float(np.median([r["mm_per_px"] for r in recs]))
    f_true_min = inj["f_true_min_hz"] or 1.0
    # f1_min：不低于 f_lo（滤波过渡带），也不低于稳定检出下限的 0.8×，
    # 更不低于静止窗 f1 上沿（慢漂频段上沿）——后者是静止误报的直接防线。
    drift_edge = float(fa.get("drift_edge_hz") or 0.0)
    f1_min = round(max(float(p["f_lo"]), f_true_min * 0.8, drift_edge), 2)
    slow_false_low = inj["slow_wave_false_valid_lowband_hz"]
    conf_lo = round(max(f1_min, float(p["f_lo"])), 2)
    conf_hi = round(max(slow_false_low) if slow_false_low else conf_lo, 2)

    f_lo = round(max(float(p["f_lo"]), 2.0 * inj["welch_delta_f_hz"]), 2)
    f_hi = round(min(float(p["f_hi"]), 0.94 * nyq), 2)
    params: dict[str, Any] = {
        "fs": round(fs, 4),
        "unit_scale": round(mm_med, 6),   # 视觉链路 px→mm 换算，语义上等价于加速度版 g→mg
        "window_s": float(p["window_s"]),
        "nperseg_s": float(p["nperseg_s"]),
        "n_sub": int(p["n_sub"]),
        "min_samples": int(p["min_samples"]),
        "min_span_s": float(p["min_span_s"]),
        "detrend": True, "despike": True, "refine_f1": True,
        "spike_win": 7, "spike_nsig": 6.0,
        "hp_cutoff": 0.3,
        "f_lo": f_lo, "f_hi": f_hi,
        "noise_floor_mg": noise["noise_floor_px"],   # fg 键名；装 px 本底（unit=px，与 gate_core G0 判定域一致）
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


# ---------------------------------------------------------------- 产物组装
def build_calib_video(args, recs: list[dict[str, Any]], health: dict[str, Any],
                      noise: dict[str, Any], fa: dict[str, Any], inj: dict[str, Any],
                      params: dict[str, Any], confusion: list[float],
                      exc: dict[str, Any]) -> dict[str, Any]:
    now = _dt.datetime.now()
    valid_days = int(args.valid_days)
    checks: dict[str, list[str]] = {"passed": [], "warnings": [], "failed": []}

    def ok(c, msg):
        checks["passed" if c else "failed"].append(msg)

    total = float(health["duration_s"])
    nwin = int(health["independent_windows"])
    ok(total >= 300.0, f"静止总时长 {total}s ≥ 300s（5min 最低要求）")
    ok(nwin >= 10, f"独立 30s 窗 {nwin} ≥ 10 个")
    ok(health["frame_loss_pct"] < 5.0,
       f"采集丢帧(frame_id 缺失) {health['frame_loss_pct']}% < 5%"
       f"（跳号 {health['frame_jump_events']} 次；ts 间隔抖动 "
       f"{health['irregular_dt_pct']}% 是处理耗时，仅诊断）")
    ok(not health["fs_mismatch"], "各靶标帧率一致")
    ok(health["mm_calibration_ready"],
       f"mm 标定就绪（mm_per_px 中位数 {health['mm_per_px_median']}）")
    # 白噪判据降级为诊断：cum 是 0 阶积分量，经跟踪器平滑后一阶差分天然负相关，
    # 传感器 ADC 的 |lag1|<0.15 不是视觉链路的必要前提（门控阈值已用实测静止窗
    # 虚警基线标定，不依赖白噪假设）。见 warnings。
    ok(fa["recommended_g1"] >= fa["g1_boundary_p999"] * 1.5,
       f"G1={fa['recommended_g1']} 覆盖边界白噪上界 "
       f"{fa['g1_boundary_p999']}（sigma={fa['g1_boundary_sigma_px']}px，"
       f"取 2× 余量；实测静止窗 p99.9={fa['g1_ratio_p999']} 含慢漂，仅诊断）")
    if fa["recommended_g1"] < 10.0:
        checks["warnings"].append(
            f"G1={fa['recommended_g1']} 低于传感器版生产基线 10：视觉链路边界白噪"
            f"p99.9 仅 {fa['g1_boundary_p999']}，10 属传感器 ADC 经验值，不适用；"
            "已按边界白噪 2× 定阈，若现场出现持续性窄带干扰需按实测补标")
    ok(inj["recommended_g2_cv_max"] >= 0.08, "G2 未擅自收紧生产基线 8%")
    _de = fa.get("drift_edge_hz")
    if _de:
        ok(float(params["f1_min_hz"]) >= _de,
           f"G3 绝对下限 {params['f1_min_hz']}Hz ≥ 慢漂频段上沿 {_de}Hz"
           f"（静止窗 f1 最大 {fa.get('static_f1_max_hz')}Hz，均属伪峰）")
        if _de >= 1.2:
            checks["warnings"].append(
                f"慢漂频段上沿 {_de}Hz 偏高：该频段以下无法区分慢漂与真实振动，"
                f"分析带实际从 {params['f1_min_hz']}Hz 起。若现场存在低于该频率的结构模态，"
                "需改用更长窗/更严去趋势或补录激励视频确认")
    ok(inj["f_true_min_hz"] is not None,
       f"注入标定检出真信号（参考幅度 {inj['ref_amp_px']}px = 1.5×G0 阈值，"
       f"最低可检出 {inj['f_true_min_hz']}Hz）")
    ok(health["id_switches_total"] == 0 and
       health["continuous_bad_pct"] < 5.0,
       f"跟踪连续性：raw_track_id 跳变 {health['id_switches_total']} 次、"
       f"continuous 缺失 {health['continuous_bad_pct']}% < 5%"
       f"（cum 台阶 {health['id_steps_ratio']*100:.3f}% 仅诊断）")

    if not args.video_desc:
        checks["warnings"].append(
            "未提供 --video-desc 静止视频来源描述：标定与数据无法互溯，建议补标")
    if health["id_steps_total"] > 0:
        checks["warnings"].append(
            f"cum 一阶差分检出 {health['id_steps_total']} 个超 5σ 台阶"
            f"（{health['id_steps_ratio']*100:.3f}%）：是亚像素中心抖动/短时遮挡，"
            f"非 ID 切换（raw_track_id 实际跳变 {health['id_switches_total']} 次）。"
            "台阶会抬高本底，建议核对该段跟踪连续性")
    if health["frame_lost_total"] > 0:
        checks["warnings"].append(
            f"frame_id 跨度内缺失 {health['frame_lost_total']} 帧"
            f"（{health['frame_loss_pct']}%，跳号 {health['frame_jump_events']} 次）："
            "视觉链路按检测成功才记录，短时丢失会削弱 FFT 等间隔前提；"
            "整段达标不代表每窗达标——丢失集中在少数时段时，单个 30s 窗缺失率"
            "可远超 5% 而被生产丢帧门拦截（该窗主频输出置空）")
    checks["warnings"].append(
        f"本底一阶差分 lag1={noise['lag1_autocorr']}：cum 是 0 阶积分量，经跟踪器"
        f"平滑后差分天然非独立白噪（原始 cum 差分 lag1={noise['lag1_raw_cum_diff']}），"
        "传感器 ADC 的 |lag1|<0.15 判据不适用；该值仅诊断不作充分性判据，"
        "G0 阈值走实测静止窗 RMS 的 p99.9、G1 阈值走边界白噪 p99.9，均不依赖白噪假设")
    max_drift = max(abs(v["drift_mm"]) for v in health["per_pid"])
    if max_drift > 5.0 * noise["noise_floor_mm"]:
        checks["warnings"].append(
            f"靶标累计漂移最大 {max_drift}mm > 5×本底：视频疑似非完全静止，"
            "本底已含慢漂成分（去趋势可压，但会抬高 G0）")
    if noise["per_pid_floor_spread"] > 2.0:
        dc = noise.get("domain_compare") or {}
        checks["warnings"].append(
            f"逐靶标本底差异(mm) {noise['per_pid_floor_spread']}× > 2，"
            f"px 域差异 {dc.get('spread_px')}×："
            + ("mm 域差异主要来自 mm_per_px 量纲换算，可评估 G0 改用相机级 px 本底"
               "（仍需多相机/多光照补标确认，勿凭单段直接切换）"
               if (dc.get("spread_ratio_mm_over_px") or 0) > 1.5 else
               "部分靶标亚像素噪声本身偏大（模糊/小目标/遮挡边），"
               "生产按靶标单独设本底或剔除该靶标"))
    checks["warnings"].append(
        f"本底采用 p10（{noise['n_windows']} 窗，系统性偏低 {noise['p10_bias_pct']}%、"
        f"标准误 {noise.get('p10_se_px', noise.get('p10_se_mm'))}px），"
        f"偏低量已由 G0 因子 {fa['derived_g0_factor']} 补偿")
    checks["warnings"].append(
        "虚警基线实测仅 " + str(fa["n_independent_windows"]) + " 个独立窗，重采样 "
        + str(fa["pseudo_replication_factor"]) + "× 属伪重复；已用独立合成白噪交叉验证")
    if exc["status"] != "VERIFIED":
        checks["warnings"].append(
            "无真实激励段：注入扫描为合成信号，只验证算法，"
            "不验证相机支架谐振/图像压缩对真实振动的畸变（可选 --excitation）")
    if health["quant_step_mm"] > 0.3 * noise["noise_floor_mm"]:
        checks["warnings"].append(
            f"像素量化台阶 {health['quant_step_mm']}mm 接近本底 "
            f"{noise['noise_floor_mm']}mm：分辨率受量化限制")
    checks["warnings"].append("方法固有局限：四门只判有无稳定窄带峰，不判峰的物理来源")
    checks["warnings"].append("低频灵敏度下降：" + inj["low_freq_sensitivity"])
    checks["warnings"].append(
        "单一静态视频无法覆盖昼夜光照/温度变化：户外部署建议跨光照条件补标")

    verdict = "READY" if not checks["failed"] else "INSUFFICIENT"
    fs_med = health["fs_hz"]
    mm_med = health["mm_per_px_median"]
    cam_id = args.cam_id or recs[0].get("cam_id") or "UNSET"

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now.isoformat(timespec="seconds"),
        "tool": "calibrate_video_single.py",
        "mode": "single_static_video",
        "unit": "mm",
        "scope": {
            "signal": "cum_dx_mm/cum_dy_mm（0 阶累计位移，视觉天然量；不做阶次变换）",
            "preprocess": "窗内 去尖峰→去趋势→0.3Hz 高通（复用 frequency_gate，与生产同口径）",
            "lighting_temperature": "未纳入标定（单一静止视频、单一光照条件）",
            "note": ("本脚本只标定频率提取可信度门控（本底+四门+频带），"
                     "不产出索力/挠度物理参数"),
        },
        "sensor": {
            "id": cam_id,
            "model": args.model or "camera",
            "rate_hz": fs_med,
            "axis_default": args.axis,
            "mm_per_px": mm_med,
        },
        "source": [{"file": r["name"], "sha256": r["sha256"], "pid": r["pid"],
                    "rows": r["n_rows"]} for r in recs],
        "video_desc": args.video_desc,
        "levels": {
            "L1_sensor": {"health": health, "noise": noise},
            "L2_thresholds": {"false_alarm": fa,
                              "injection": {k: v for k, v in inj.items() if k != "grid"},
                              "confusion_band_hz": confusion,
                              "hardware_excitation": exc},
        },
        "data_sufficiency": {
            "verdict": verdict,
            "verdict_meaning": {
                "READY": ("单段 ≥5min 且 ≥10 个完整 30s 窗，本底 + 四门阈值已标定，"
                          "可供靶标振动分析加载并判定可信度"),
                "INSUFFICIENT": ("静止不足 5min 或完整 30s 窗不足 10 个，"
                                 "参数仅供参考，禁止投用"),
            },
            "total_static_s": total,
            "segments": len({r["name"] for r in recs}),
            "independent_windows": nwin,
            "p10_se_px": noise.get("p10_se_px", noise.get("p10_se_mm", 0.0)),
            "p10_bias_pct": noise["p10_bias_pct"],
            "hardware_excitation": exc["status"],
        },
        "params": params,
        "fingerprint": {
            "cam_id": cam_id,
            "axis_default": args.axis,
            "rate_hz": fs_med,
            "unit_scale": round(mm_med, 6),
            "mm_per_px": round(mm_med, 6),
            "window_s": float(params["window_s"]),
            "nperseg_s": float(params["nperseg_s"]),
            "f_lo": float(params["f_lo"]),
            "f_hi": float(params["f_hi"]),
        },
        "validity": {
            "calibrated_on": now.date().isoformat(),
            "valid_until": (now.date() + _dt.timedelta(days=valid_days)).isoformat(),
            "valid_days": valid_days,
            "recalibrate_if": RECALIBRATE_IF,
        },
        "checks": checks,
    }


def write_report_video(path: str, c: dict[str, Any]) -> None:
    L = c["levels"]
    h, n = L["L1_sensor"]["health"], L["L1_sensor"]["noise"]
    fa, inj = L["L2_thresholds"]["false_alarm"], L["L2_thresholds"]["injection"]
    fp, v, ds = c["fingerprint"], c["validity"], c["data_sufficiency"]
    out: list[str] = []
    add = out.append
    add("=" * 78)
    add("视觉靶标单段静止视频标定报告")
    add("=" * 78)
    add(f"相机        : {c['sensor']['id']}  帧率 {c['sensor']['rate_hz']}Hz  "
        f"轴 {c['sensor']['axis_default']}  mm/px={c['sensor']['mm_per_px']}")
    add(f"标定时间    : {c['generated_at']}   模式 {c['mode']}   单位 {c['unit']}")
    add(f"视频来源    : {c.get('video_desc') or '（未标注 --video-desc）'}")
    add(f"数据指纹    : {c['source'][0]['sha256'][:16]}…  共 {len(c['source'])} 条序列")
    add(f"有效期至    : {v['valid_until']}（{v['valid_days']} 天）")
    add(f"信号口径    : {c['scope']['signal']}")
    add("")
    add("-" * 78)
    add("STEP 0  数据体检")
    add("-" * 78)
    add(f"  fs={h['fs_hz']}Hz  dt中位数={h['dt_median_s']}s  "
        f"ts间隔抖动 {h['irregular_dt_count']} 处 ({h['irregular_dt_pct']}%，诊断用)  "
        f"靶标数 {h['n_pids']}")
    add(f"  真丢帧(frame_id) = {h['frame_lost_total']}/{h['frame_span_total']} "
        f"({h['frame_loss_pct']}%, 跳号 {h['frame_jump_events']} 次)  "
        f"ID跳变 {h['id_switches_total']} 次  continuous缺失 {h['continuous_bad_pct']}%")
    add(f"  mm 标定就绪={h['mm_calibration_ready']}  mm_per_px中位数={h['mm_per_px_median']}  "
        f"量化台阶={h['quant_step_mm']}mm")
    add(f"  cum 台阶(诊断,非ID)={h['id_steps_total']}（{h['id_steps_ratio']*100:.3f}%）")
    add("  逐靶标：")
    for it in h["per_pid"]:
        add(f"    pid={it['pid']:<6s} {it['duration_s']:>7.1f}s  n={it['n_valid']:>5d}  "
            f"mm/px={it['mm_per_px']:<9.6f}  conf={it['conf_mean']:<6.3f}  "
            f"台阶={it['id_steps']:<3d}  ID跳变={it['id_switches']}  "
            f"丢帧={it['frame_lost']}/{it['frame_span']}  "
            f"max_gap={it['max_gap_s']}s  drift={it['drift_mm']}mm")
    add(f"  独立 30s 窗 {ds['independent_windows']} 个（完整窗，残窗不计）")
    add("")
    add("-" * 78)
    add("STEP 2  视觉本底（核心 px 域，与生产 G0 判定域一致；mm 域仅体检）")
    add("-" * 78)
    add(f"  noise_floor = {n['noise_floor_px']} px  [{n['definition']}]")
    add(f"  (mm 域对照 = {n['noise_floor_mm']} mm，不参与门控)")
    add(f"  逐窗 RMS(px) = {n.get('window_rms_px', n['window_rms_mm'])}")
    add(f"  逐靶标本底(mm) = {n['per_pid_floor_mm']}  差异 {n['per_pid_floor_spread']}×")
    add(f"  逐靶标本底(px) = {n['per_pid_floor_px']}  差异 {n['per_pid_floor_spread_px']}×")
    dc = n.get("domain_compare") or {}
    if dc:
        add(f"  【px/mm 双域】差异比 mm/px = {dc.get('spread_ratio_mm_over_px')}×  "
            f"mm_per_px={dc.get('per_pid_mm_per_px')}")
        add(f"    结论: {dc.get('conclusion')}")
        add(f"    注意: {dc.get('caveat')}")
    add(f"  p10 系统性偏低 {n['p10_bias_pct']}%  标准误 {n.get('p10_se_px', n.get('p10_se_mm'))}px  "
        f"原始 std {n['raw_std_mm']}mm（差 {n['raw_std_vs_floor_pct']}%，mm 域对照）")
    add(f"  白噪性 lag1(预处理残差差分)={n['lag1_autocorr']}（|lag1|<0.15 原为传感器判据，"
        f"视觉链路仅诊断）；预处理残差 lag1={n['lag1_autocorr_cumulative']}，"
        f"原始 cum 一阶差分 lag1={n['lag1_raw_cum_diff']}（负相关=跟踪器阻尼，固有属性）")
    add("")
    add("-" * 78)
    add("STEP 3  各门阈值（供靶标振动分析加载判定 VALID/INVALID）")
    add("-" * 78)
    add(f"  G0  幅度门      = px域RMS ≥ {c['params']['noise_floor_mg']}px × {c['params']['g0_rms_factor']}"
        f" = {round(c['params']['noise_floor_mg']*c['params']['g0_rms_factor'],4)}px")
    add(f"      依据：典型窗 RMS(p50)={fa.get('g0_rms_p50_px')}px × 1.15 ÷ 本底(p10)="
        f"{c['params']['noise_floor_mg']}px → ×{c['params']['g0_rms_factor']}"
        f"（旧 p99.9={fa.get('g0_rms_p999_px')}px 极端分位法会砍灵敏度，已弃用）")
    add(f"  G1  峰/中位数   ≥ {c['params']['g1_peak_ratio']}  "
        f"(边界白噪 sigma={fa['g1_boundary_sigma_px']}px 的 p99.9/max="
        f"{fa['g1_boundary_p999']}/{fa['g1_boundary_max']}，取 2× 余量)")
    add(f"      诊断：静止实测 p50/p99.9/max={fa['g1_ratio_p50']}/{fa['g1_ratio_p999']}/"
        f"{fa['g1_ratio_max']}（含 0.3~1Hz 慢漂，不参与定阈）；"
        f"合成白噪 p99.9={fa['g1_synthetic_p999']}")
    add(f"  G2  子窗 CV     ≤ {c['params']['g2_cv_max']*100:.1f}%  "
        f"(注入 CV p99={inj['injected_valid_cv_p99']}, 不擅自收紧生产基线 8%)")
    add(f"  G3  频率下限    ≥ {c['params']['f1_min_hz']}Hz  "
        f"(拦累计量钉最低 bin/慢波假峰；静止窗 f1 上沿 {fa.get('drift_edge_hz')}Hz，"
        f"{inj['ref_amp_px']}px(=1.5×G0阈值)注入稳定检出最低 {inj['f_true_min_hz']}Hz)")
    add(f"  分析带          = {c['params']['f_lo']} ~ {c['params']['f_hi']}Hz  "
        f"(Nyquist={inj['nyquist_hz']}Hz)")
    add(f"  过渡带混淆区    = {L['L2_thresholds']['confusion_band_hz']}Hz（需外部证据）")
    add("")
    add("-" * 78)
    add("STEP 4  注入扫描（幅度=本底倍数 ∪ G0阈值倍数，叠加在真实静止窗上，最终阈值已回填）")
    add("-" * 78)
    add(f"  幅度档(px) = {inj['amplitudes_px']}（× 本底 {inj['amplitudes_rel']}）")
    add(f"  3×本底稳定检出频率: {inj['stable_at_3xfloor_hz']}Hz")
    add(f"  最小可检出幅度: " + ", ".join(
        f"{f}Hz→{d['px']}px({d['rel']}×)" for f, d in inj["min_detect_amp"].items()))
    add(f"  {inj['slow_wave_amp_px']}px（{SLOW_AMP_REL}×本底）强单频误判 VALID 的频率: "
        f"{inj['slow_wave_false_valid_hz']}Hz")
    add(f"  方法局限: {inj['method_limit']}")
    add(f"  本项为【合成注入】(synthetic_only={inj['synthetic_only']})，只验算法不验硬件链路")
    add("")
    add("-" * 78)
    add("数据充分性")
    add("-" * 78)
    add(f"  结论 = {ds['verdict']}  —  {ds['verdict_meaning'][ds['verdict']]}")
    add(f"  总静止 {ds['total_static_s']}s / {ds['segments']} 段 / "
        f"{ds['independent_windows']} 独立窗 / 本底标准误 {ds.get('p10_se_px', ds.get('p10_se_mm'))}px")
    ex = L["L2_thresholds"]["hardware_excitation"]
    add(f"  真实激励段 = {ex['status']}  {ex['reason']}")
    add("")
    add("复标触发条件：")
    for x in v["recalibrate_if"]:
        add(f"  - {x}")
    add("")
    add("完整性检查")
    add("-" * 78)
    for x in c["checks"]["passed"]:
        add(f"  PASS  {x}")
    for x in c["checks"]["warnings"]:
        add(f"  WARN  {x}")
    for x in c["checks"]["failed"]:
        add(f"  FAIL  {x}")
    add("")
    add("指纹（加载时比对，不一致会告警）：")
    for k, val in fp.items():
        add(f"  {k:<16s} = {val}")
    add("=" * 78)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")


# ---------------------------------------------------------------- 主流程
def run(args) -> tuple[dict[str, Any], str, str]:
    recs: list[dict[str, Any]] = []
    for f in args.static:
        recs.extend(read_displacement(_resolve(f), args.axis))
    if not recs:
        raise ValueError("没有读到任何有效位移序列")
    fs_med = float(np.median([r["fs"] for r in recs]))
    f_hi = (round(0.94 * fs_med / 2.0, 2) if args.f_hi <= 0 else round(float(args.f_hi), 2))
    p = fg.merge_params({
        "window_s": args.window_s, "nperseg_s": args.nperseg_s,
        "f_lo": args.f_lo, "f_hi": f_hi, "axis": args.axis,
        "multi_axis": False, "harmonic_check": False, "xcheck": False,
    })

    pool = ca.build_window_pool(recs, p)          # mm 域：按段切完整 30s 窗（体检/展示用）
    per_pid_pools: dict[str, list[np.ndarray]] = {}
    for r in recs:
        per_pid_pools[r["pid"]] = ca.build_window_pool([r], p)
    # ---- px 域窗（G0 判定域，与生产 gate_core 一致）----
    # 生产 gate_core 的 G0 在 px 域算 RMS、与 noise_floor 比较；标定必须同域。
    # mm 序列 ÷ 该靶标 mm_per_px = px 位移（cum_dy_mm/mm_per_px = dy_px），
    # 窗结构与 mm 完全一致，仅量纲不同；预处理各步尺度线性，谱比值不变。
    mmpp_by_pid = {r["pid"]: r["mm_per_px"] for r in recs}
    per_pid_pools_px: dict[str, list[np.ndarray]] = {}
    for pid, segs in per_pid_pools.items():
        m = mmpp_by_pid.get(pid, 0.0)
        if m > 0:
            per_pid_pools_px[pid] = [seg / m for seg in segs]
    # 合并的 px 域 pool（顺序与 pool 一致：逐 rec 逐窗，仅换算量纲）
    pool_px: list[np.ndarray] = []
    for r in recs:
        m = r["mm_per_px"]
        for seg in ca.build_window_pool([r], p):
            pool_px.append(seg / m if m > 0 else seg)

    health = step0_health_video(recs)
    health.update(ca.step0_summary(recs, p, pool))  # 复用：独立窗/可用时长汇总
    raw_concat = np.concatenate([r["y"] for r in recs])  # mm 域原始序列（体检用）
    # nf / G0 因子 / 注入扫描全部走 px 域，写入 noise_floor_mg 的是 px 本底
    noise, nf = step2_noise_video(pool_px, p, fs_med, raw_concat, health,
                                  per_pid_pools, per_pid_pools_px, pool_mm=pool)
    rng = np.random.default_rng(args.seed)
    fa = step3_false_alarm_video(pool_px, p, fs_med, nf, noise, args.n_mc, rng)
    # 把最终 G0/G1 阈值回填到 p 再跑注入扫描。
    # 否则 step4/step5 仍在 fg.DEFAULTS（g0=1.2 / g1=10）下跑，得到的 min_detect_amp
    # 与「注入检出真信号」PASS 都不代表写入文件的阈值——本轮即为修此缺陷。
    p["g0_rms_factor"] = fa["derived_g0_factor"]
    p["g1_peak_ratio"] = fa["recommended_g1"]
    # G3 绝对下限同样要在注入扫描前生效：否则 step4 仍以 f1_min=0 判 valid，
    # 会把慢漂频段的「可检出」当成有效检出，f_true_min 报得过低。
    if fa.get("drift_edge_hz"):
        p["f1_min_hz"] = float(fa["drift_edge_hz"])
    inj = step4_injection_video(pool_px, p, fs_med, nf)
    # G2 由首轮注入结果推导；回填后再跑一轮，使检出域在最终四门阈值下成立。
    # 第二轮推导出的 G2 不再回填（否则自引用），沿用首轮值并记 g2_applied_cv_max。
    p["g2_cv_max"] = inj["recommended_g2_cv_max"]
    inj = step4_injection_video(pool_px, p, fs_med, nf)
    inj["derived_g2_cv_max"] = p["g2_cv_max"]
    inj["recommended_g2_cv_max"] = p["g2_cv_max"]
    inj["g2_applied_cv_max"] = p["g2_cv_max"]
    inj["g0_applied_factor"] = p["g0_rms_factor"]
    inj["g1_applied_peak_ratio"] = p["g1_peak_ratio"]
    exc = step5_excitation_video(args.excitation, args.excitation_hz, args.axis, p, nf)
    params, confusion = derive_params_video(recs, p, noise, fa, inj)
    calib = build_calib_video(args, recs, health, noise, fa, inj, params, confusion, exc)
    return calib, p, fs_med


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="视觉靶标单段静止视频标定（四门阈值，单位 mm）")
    ap.add_argument("--static", action="append", required=True,
                    help="静止靶标视频主链路产物 displacement.jsonl（可重复多次）")
    ap.add_argument("--axis", default="dy", choices=("dy", "dx"),
                    help="标定轴：dy=垂向（默认，与生产主频口径一致）/ dx")
    ap.add_argument("--cam-id", default="",
                    help="相机编号，默认取 displacement.jsonl 所在目录名")
    ap.add_argument("--model", default="", help="相机/镜头描述（可选）")
    ap.add_argument("--video-desc", default="",
                    help="静止视频来源描述（建议必填，用于标定与数据互溯）")
    ap.add_argument("--window-s", type=float, default=30.0)
    ap.add_argument("--nperseg-s", type=float, default=8.0)
    ap.add_argument("--f-lo", type=float, default=0.3, help="分析带下限 Hz")
    ap.add_argument("--f-hi", type=float, default=0.0,
                    help="分析带上限 Hz；0=自动取 0.94×Nyquist")
    ap.add_argument("--n-mc", type=int, default=150, help="虚警基线重采样次数")
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--valid-days", type=int, default=180)
    ap.add_argument("--excitation", default="",
                    help="已知频率激励视频的 displacement.jsonl（可选，验证全链路）")
    ap.add_argument("--excitation-hz", type=float, default=0.0,
                    help="激励真值频率 Hz，配合 --excitation")
    args = ap.parse_args(argv)

    if not args.cam_id:
        d = os.path.dirname(os.path.abspath(_resolve(args.static[0])))
        args.cam_id = os.path.basename(d) or "UNSET"
    args.out = f"{args.cam_id}_calib.json"
    args.report = f"{args.cam_id}_calib_report.txt"

    calib, p, fs = run(args)
    out = os.path.join(ROOT, args.out)
    rep = os.path.join(ROOT, args.report)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(calib, fh, ensure_ascii=False, indent=2)
    write_report_video(rep, calib)

    noise = calib["levels"]["L1_sensor"]["noise"]
    params = calib["params"]
    print(f"标定完成 → {out}")
    print(f"报告     → {rep}")
    print(f"本底 RMS = {noise['noise_floor_px']} px (G0判定域) / {noise['noise_floor_mm']} mm (对照)  "
          f"| 相机 {args.cam_id} @ {fs}Hz 轴 {args.axis}")
    print(f"采用阈值 = G0×{params['g0_rms_factor']}  G1≥{params['g1_peak_ratio']}  "
          f"G2 CV≤{params['g2_cv_max']*100:.1f}%  f1_min={params['f1_min_hz']}Hz  "
          f"带 {params['f_lo']}~{params['f_hi']}Hz")
    exc = calib["levels"]["L2_thresholds"]["hardware_excitation"]
    print(f"真实激励段 = {exc['status']}  {exc['reason'][:60]}")
    print(f"检查     = PASS {len(calib['checks']['passed'])} / "
          f"WARN {len(calib['checks']['warnings'])} / "
          f"FAIL {len(calib['checks']['failed'])}")
    for x in calib["checks"]["failed"]:
        print(f"  FAIL  {x}")
    verdict = calib["data_sufficiency"]["verdict"]
    print(f"结论     = {verdict} — {calib['data_sufficiency']['verdict_meaning'][verdict]}")
    print("目标     = 靶标振动频率提取的可信度门控（不涉索力/挠度换算）")
    return 1 if calib["checks"]["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
