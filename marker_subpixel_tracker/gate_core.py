# -*- coding: utf-8 -*-
"""靶标振动四门 + 两个视觉特有小门的公共实现（主链路 / 看板 / 界面三处同源）。

设计原则
--------
1. 复用 sensor_data/accelerometer/frequency_gate.py 的预处理与 Welch，不在本文件
   另写一套谱算法（与视觉标定件 calibrate_video_single.py 同口径）。
2. 门控是可选增强层：没有标定 JSON 时返回 legacy 口径（单段 FFT + SNR 门），
   现有主链路/看板行为不变；有标定 JSON 时才启用 G0/G1/G2/G3。
3. 判定域用 px（相机级本底），输出幅值仍用 mm（报警/物理量口径不变）。
   mm_per_px 只做换算，不参与门控阈值。
4. 丢帧率门与 ID 连续性门不依赖标定阈值，且必须用视觉链路自己的口径
   （与 calibrate_video_single.py 同源）：
   - 丢帧 = frame_id 缺失帧占比 < 5%；ts 间隔抖动只是处理耗时，仅诊断。
   - ID 切换 = raw_track_id 跳变 + continuous 缺失 < 5%；cum 一阶差分台阶
     是亚像素中心抖动/短时遮挡，不是 ID 切换，仅诊断。
   数据不足时标 SKIP 而非 FAIL。

门的定义（与 sensor/accelerometer 对齐，阈值来自标定 JSON 的 params）
---------------------------------------------------------------------
  G0 幅度门     去趋势后 RMS >= g0_rms_factor × noise_floor（px 域）
  G1 显著性门   带内 Welch 峰/中位数 >= g1_peak_ratio
  G2 稳定性门   四个子窗带内主频 CV <= g2_cv_max
  G3 频率下限   f1 > max(g3_factor / 窗跨度, f1_min_hz)
  丢帧率门      frame_id 缺失帧占比 < 5%（ts 抖动仅诊断）
  ID 连续性门   raw_track_id 不跳变 + continuous 缺失率 < 5%（cum 台阶仅诊断）

标定的 JSON 结构由 calibrate_video_single.py 产出（schema_version=1，params 键名
与 fg.DEFAULTS 一致）。加载时只取 params + fingerprint，指纹不一致给 warnings。
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ACCEL = os.path.normpath(os.path.join(_HERE, "..", "sensor_data", "accelerometer"))
if _ACCEL not in sys.path:
    sys.path.insert(0, _ACCEL)
import frequency_gate as fg  # noqa: E402

# 未标定时的固定口径（对应 sensor 生产基线，不作为判定、只作为"未启用"标记）
_LEGACY = {"snr": 5.0, "window_s": 30.0, "min_samples": 32, "min_span_s": 10.0}

DROP_MAX_PCT = 5.0          # 丢帧率门：frame_id 缺失帧占比上限（视觉链路）
ID_STEP_NSIG = 6.0          # ID 连续性门：cum 台阶诊断倍数（不参与判定）
_ID_BAD_RATE_MAX = 0.05     # ID 连续性门：continuous=False 占比上限


def load_gate_config(path: str | None) -> dict[str, Any] | None:
    """加载标定 JSON。空路径/文件不存在/格式错误 → None（保持 legacy 行为）。"""
    if not path:
        return None
    p = path if os.path.isabs(path) else os.path.normpath(os.path.join(_HERE, "..", path))
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if int(doc.get("schema_version", 0)) != 1:
        return None
    params = dict(doc.get("params") or {})
    bad = [k for k in params if k not in fg.DEFAULTS]
    if bad:
        params = {k: v for k, v in params.items() if k not in fg.DEFAULTS}
    return {
        "path": p,
        "params": params,
        "fingerprint": doc.get("fingerprint") or {},
        "verdict": ((doc.get("data_sufficiency") or {}).get("verdict") or ""),
        "cam_id": (doc.get("fingerprint") or {}).get("cam_id") or "",
    }


def check_fingerprint(cfg: dict[str, Any], fs: float, cam_id: str | None = None) -> list[str]:
    """运行期指纹校验：帧率/相机与标定不符时告警，不静默放行。"""
    if not cfg:
        return []
    fp = cfg.get("fingerprint") or {}
    w: list[str] = []
    rate = float(fp.get("rate_hz") or 0)
    if rate > 0 and abs(fs - rate) / rate > 0.05:
        w.append(f"采样率 {fs:.3f}Hz ≠ 标定 {rate:.3f}Hz（G0/G1/G2 阈值失效）")
    if cam_id and fp.get("cam_id") and str(cam_id) != str(fp["cam_id"]):
        w.append(f"相机 {cam_id} ≠ 标定相机 {fp['cam_id']}（mm_per_px/本底不适用）")
    if str(cfg.get("verdict") or "").upper() != "READY":
        w.append(f"标定充分性={cfg.get('verdict') or '未知'}，参数仅供参考，禁止作为正式判定")
    return w


def _band_peak(f, power, lo, hi):
    return fg._band_peak(f, power, lo, hi)


def _spectrum(t, y, p, noise_floor):
    """预处理器 + Welch + 峰提取。返回 (f, power, x, f1_raw, f1, ratio, med, rms)。"""
    fs = 1.0 / float(np.median(np.diff(np.asarray(t, dtype=float))))
    lo = max(0.0, float(p["f_lo"]))
    hi = min(float(p["f_hi"]), fs / 2.0 - 1e-6)
    nperseg = int(float(p["nperseg_s"]) * fs)
    x, n_spike = fg.preprocess_window(np.asarray(y, dtype=float), fs, p)
    rms = float(np.sqrt(np.mean(x * x)))
    wr = fg.welch_psd(x, fs, nperseg)
    if wr is None:
        return None
    f, power = wr
    f1_raw, ratio, med = _band_peak(f, power, lo, hi)
    i1 = int(np.argmin(np.abs(f - (f1_raw if f1_raw is not None else lo))))
    f1 = (fg.refine_logparabola(f, power, i1)
          if p.get("refine_f1", True) and f1_raw is not None else f1_raw)
    return {"f": f, "power": power, "x": x, "fs": fs, "lo": lo, "hi": hi,
            "nperseg": nperseg, "f1_raw": f1_raw, "f1": f1,
            "ratio": ratio, "med": med, "rms": rms, "n_spike": n_spike}


def _downsample(f, power, nb=200):
    if len(f) > nb:
        idx = np.linspace(0, len(f) - 1, nb).astype(int)
        return [round(float(v), 5) for v in f[idx]], [float(v) for v in power[idx]]
    return [round(float(v), 5) for v in f], [float(v) for v in power]


def evaluate_window(t, y_px, y_mm, snr_min, *, cfg=None, meta=None,
                    window_s=None, min_samples=None, min_span_s=None):
    """单窗六门判定。返回兼容 _segment_spectrum 的 dict + gates/reason 扩展。

    t/y_px/y_mm 为等长序列；y_px 用于 G0/G1/G2 判定，y_mm 只算出输出幅值。
    cfg=None → legacy：单段 FFT + snr 门，保持旧行为。
    meta 为 {"frame_id":[], "raw_track_id":[], "continuous":[]} 可选。
    """
    t = np.asarray(t, dtype=float)
    y_px = np.asarray(y_px, dtype=float)
    y_mm = np.asarray(y_mm, dtype=float)
    n = len(t)
    result = {"freq": [], "power": [], "f1_hz": None, "snr": None,
              "n": int(n), "dt": None, "gates": {}, "reason": "", "valid": False}
    if n < 4:
        result["reason"] = "样本不足"
        return result
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        result["reason"] = "时间戳异常"
        return result
    result["dt"] = dt
    span = float(t[-1] - t[0]) + dt
    gates: dict[str, Any] = {}

    if cfg is None:
        # ---- legacy：完全沿用旧 _segment_spectrum（保证无标定时零行为变化）----
        z = y_mm - y_mm.mean()
        spec = np.abs(np.fft.rfft(z)) ** 2
        freqs = np.fft.rfftfreq(n, d=dt)
        if len(spec) < 3:
            result["reason"] = "谱长度不足"
            return result
        body = spec[1:]
        k = int(np.argmax(body)) + 1
        noise = float(np.median(body))
        snr = (float(spec[k]) / noise) if noise > 0 else float("inf")
        f1 = float(freqs[k]) if not (snr < float(snr_min) or freqs[k] <= 0) else None
        fd, pd = ((freqs, spec) if len(freqs) <= 200 else
                  (freqs[np.linspace(0, len(freqs) - 1, 200).astype(int)],
                   spec[np.linspace(0, len(spec) - 1, 200).astype(int)]))
        result.update({"freq": [float(x) for x in fd], "power": [float(x) for x in pd],
                       "f1_hz": f1, "snr": snr, "valid": f1 is not None,
                       "mode": "legacy",
                       "reason": "legacy(SNR>=%.1f)%s" % (
                           snr_min, "" if f1 is not None else " 未达标")})
        return result

    # ---- 标定模式：px 域六门 ----
    p = dict(cfg["params"])
    min_n = int(min_samples if min_samples is not None else p.get("min_samples", 32))
    need = (float(min_span_s) if min_span_s is not None
            else float(p.get("min_span_s", 10.0)))
    if n < min_n or span < min(need, float(window_s or p.get("window_s", 30.0))) - 1e-9:
        result["reason"] = f"样本不足({n}<{min_n})或跨度不足({span:.1f}<{need:.1f})"
        gates["eligible"] = False
        result["gates"] = gates
        return result

    r = _spectrum(t, y_px, p, None)
    if r is None:
        result["reason"] = "Welch 无法计算"
        return result
    f, power, x, fs = r["f"], r["power"], r["x"], r["fs"]
    lo, hi = r["lo"], r["hi"]
    f1_raw, f1, ratio, med, rms = r["f1_raw"], r["f1"], r["ratio"], r["med"], r["rms"]

    nf = float(p.get("noise_floor_mg", 0.0)) or 0.0  # 键名沿用 fg，装 px 本底
    g0_thr = float(p.get("g0_rms_factor", 0.0)) * nf
    g0 = bool(nf > 0 and g0_thr > 0 and rms >= g0_thr)
    g1 = bool(ratio is not None and ratio >= float(p.get("g1_peak_ratio", 10.0)))

    # G2：四子窗带内验峰 CV（与 fg.analyze_window 的 band 模式同构）
    subs: list[float] = []
    nsub = int(p.get("n_sub", 4))
    band_frac = float(p.get("g2_band_frac", 0.3))
    sub_ratio = float(p.get("g2_sub_ratio", 5.0))
    for k in range(nsub):
        a, b = int(k * n / nsub), int((k + 1) * n / nsub)
        part = x[a:b]
        if len(part) < 8:
            continue
        sr = fg.welch_psd(part, fs, min(r["nperseg"], len(part)))
        if sr is None:
            continue
        sf, sp = sr
        if f1_raw is None:
            continue
        slo, shi = f1_raw * (1 - band_frac), f1_raw * (1 + band_frac)
        band = (sf >= slo) & (sf <= shi)
        if not band.any():
            continue
        smed = float(np.median(sp[(sf >= lo) & (sf <= hi)])) if np.any((sf >= lo) & (sf <= hi)) else 0.0
        i = int(np.argmax(np.where(band, sp, -np.inf)))
        if smed > 0 and float(sp[i]) >= sub_ratio * smed:
            subs.append(float(fg.refine_logparabola(sf, sp, i)))
    need_subs = max(2, min(int(p.get("g2_min_subs", 3)), nsub))
    cv = (float(np.std(subs) / np.mean(subs)) if len(subs) >= need_subs and np.mean(subs) > 0 else None)
    g2 = bool(cv is not None and cv <= float(p.get("g2_cv_max", 0.08)))

    g3_thr = max(float(p.get("g3_factor", 2.0)) / span, float(p.get("f1_min_hz", 0.0)))
    g3 = bool(f1 is not None and f1 > g3_thr)

    # 丢帧率门（数据不足则 SKIP）
    g_drop = _drop_gate(meta, t, dt)
    # ID 连续性门
    g_id = _id_gate(meta, y_px)

    gates.update({
        "eligible": True, "mode": "calibrated",
        "G0": {"pass": g0, "rms": round(rms, 4), "threshold": round(g0_thr, 4),
               "enabled": bool(nf > 0 and g0_thr > 0)},
        "G1": {"pass": g1, "ratio": round(float(ratio), 2) if ratio is not None else None,
               "threshold": float(p.get("g1_peak_ratio", 10.0))},
        "G2": {"pass": g2, "cv": round(cv, 5) if cv is not None else None,
               "threshold": float(p.get("g2_cv_max", 0.08)), "n_sub_valid": len(subs)},
        "G3": {"pass": g3, "f1": f1, "threshold": round(g3_thr, 5)},
        "drop": g_drop, "id": g_id,
    })
    failed = [k for k in ("G0", "G1", "G2", "G3")
              if k in gates and gates[k].get("enabled", True) and not gates[k]["pass"]]
    for k in ("drop", "id"):
        if gates[k].get("status") == "FAIL":
            failed.append(k)
    valid = bool(not failed and f1 is not None)
    result.update({
        "freq": _downsample(f, power)[0], "power": _downsample(f, power)[1],
        # 主频未通过门控时不输出（与 legacy 的 f1=None 语义一致）
        "f1_hz": float(f1) if valid else None,
        "snr": float(ratio) if ratio is not None and np.isfinite(ratio) else None,
        "max_amp_mm": round(float(np.max(np.abs(y_mm))), 3),
        "n_spike": r["n_spike"],
        "valid": valid, "gates": gates,
        "reason": ("未通过：" + "/".join(failed)) if failed else (
            "四门全过" if valid else "主频未提取"),
    })
    return result


def _drop_gate(meta, t, dt):
    """丢帧率门：frame_id 缺失帧占比。视觉链路「检测成功才记录」，ts 间隔
    受推理/解码/落盘耗时抖动（实测 0.013~0.068s，中位 0.050s），不能沿用
    传感器 ADC 的「ts 间隔偏离>25%=丢帧」口径——那会把 25% 正常抖动误判
    成丢帧，也会把 1 帧短时丢失（跳号=1）整窗作废。校准脚本同口径。"""
    out = {"status": "SKIP", "reason": "无 frame_id 元数据"}
    if not meta or not meta.get("frame_id") or len(meta["frame_id"]) < 4:
        return out
    fid = np.asarray(meta["frame_id"], dtype=np.int64)
    gaps = np.diff(fid)
    span = int(fid[-1] - fid[0]) + 1 if len(fid) else 0
    lost = int(np.sum(np.maximum(gaps - 1, 0)))
    pct = round(100.0 * lost / max(span, 1), 3)
    jump = int(np.sum(gaps > 1))
    uneven = np.diff(np.asarray(t, dtype=float))
    jitter = int(np.sum(uneven > 1.25 * dt)) if dt > 0 else 0
    jitter_pct = round(100.0 * jitter / max(len(uneven), 1), 3)
    ok = pct < DROP_MAX_PCT
    out.update({"status": "PASS" if ok else "FAIL", "frame_lost": lost,
                "frame_span": span, "frame_loss_pct": pct,
                "jump_events": jump, "ts_jitter_pct": jitter_pct,
                "dt_s": round(dt, 5),
                "reason": (f"frame_id 缺失{lost}/{span}={pct}%/{DROP_MAX_PCT}% "
                           f"跳号{jump}；ts抖动{jitter_pct}%仅诊断")})
    return out


def _id_gate(meta, y_px):
    """ID 连续性门：raw_track_id 跳变 + continuous 缺失率。cum 一阶差分
    台阶是亚像素中心抖动/短时遮挡（实测全程 raw_track_id 0 次跳变却有
    5% 台阶），沿用传感器「台阶=ID 跳变」口径会误杀绝大多数窗；台阶
    保留为诊断字段，不参与判定。校准脚本同口径。"""
    out = {"status": "SKIP", "reason": "无 ID 元数据"}
    if not meta or not meta.get("raw_track_id") or len(meta["raw_track_id"]) < 4:
        return out
    raw = np.asarray(meta["raw_track_id"])
    switches = int(np.sum(raw[1:] != raw[:-1]))
    dd = np.diff(np.asarray(y_px, dtype=float))
    mad = float(np.median(np.abs(dd - np.median(dd))))
    sigma = 1.4826 * mad
    steps = int(np.sum(np.abs(dd) > ID_STEP_NSIG * sigma)) if sigma > 0 else 0
    cont = meta.get("continuous")
    bad_rate = 0.0
    if cont and len(cont) >= 4:
        bad_rate = float(np.sum(~np.asarray(cont, dtype=bool)) / len(cont))
    ok = switches == 0 and bad_rate < _ID_BAD_RATE_MAX
    out.update({"status": "PASS" if ok else "FAIL", "raw_switches": switches,
                "cum_steps": steps,
                "continuous_bad_rate": round(bad_rate, 4),
                "reason": (f"raw跳变{switches} 断续{bad_rate*100:.1f}%/"
                           f"{_ID_BAD_RATE_MAX*100:.0f}%（cum台阶{steps}仅诊断）")})
    return out
