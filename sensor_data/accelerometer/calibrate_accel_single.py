# -*- coding: utf-8 -*-
"""加速度计「单段 5min 静止」标定 —— 固定垂直姿态，忽略温度与时段。

与 calibrate_accel.py（多段版）的区别
------------------------------------
- 多段版：要求 ≥3 段静止 + 跨时段/温度 + 已知激励，覆盖零偏温漂与时段的完整标定。
- 本脚本（单段版）：适用于「垂直固定安装、环境温度稳定」的场景，用一份 ≥5min
  的静止数据一次标定出 本底 + G0/G1/G2/G3 全部门限阈值，安装后直接加载判定。

标定口径与生产完全一致：复用 frequency_gate.py 与 calibrate_accel.py 的
预处理 / Welch / 四门 / 注入扫描 函数，不另写一套（避免「标定一套、上线另一套」）。

一份 5min 静止数据能标定出什么（安装后直接用的门）
--------------------------------------------------
  noise_floor_mg   本底（G0 的基准）
  g0_rms_factor    G0 幅度门因子
  g1_peak_ratio    G1 峰/中位数门限（沿用生产基线 10，不擅自下调）
  g2_cv_max        G2 子窗主频 CV 门限（沿用生产基线 8%，不擅自收紧）
  f1_min_hz        G3 频率下限（拦累计量/慢波假峰）
以上全部只靠「静止本底 + 注入扫描」即可标定，无需多段、无需温度、无需真实激励。

本脚本只做「频率提取正确」：标定 本底 + G0/G1/G2/G3 四门阈值 + 分析频带 + 亚格点插值，
安装后据此提取频率并判定可信度（VALID/INVALID）。不产出、不告警任何 L3/索力参数
（L_m / mu_kg_m / f1_ref_hz / attach_mass_g 属索力换算，留待后续阶段）。

用法（项目根目录，必须用 .venv）
--------------------------------
    .venv/Scripts/python.exe sensor_data/accelerometer/10-34-27-942/calibrate_accel_single.py \
        --static data_0.csv --sensor-id WT901-0001 --range-g 2

    # 可选：已知频率激励段（验证硬件链路，非门限标定所必需）
    ... --excitation exc.csv --excitation-hz 2

产物：
    accel_calib_single.json        参数（gate_server.py --calib accel_calib_single.json 或前端载入）
    accel_calib_single_report.txt  人读报告
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calibrate_accel as ca   # noqa: E402  复用全部 step 函数，保证口径一致
import frequency_gate as fg    # noqa: E402


def build_calib_single(args, rec: dict[str, Any], health: dict[str, Any],
                       gravity: dict[str, Any], noise: dict[str, Any],
                       fa: dict[str, Any], inj: dict[str, Any],
                       params: dict[str, Any], confusion: list[float],
                       exc: dict[str, Any]) -> dict[str, Any]:
    """单段静止版标定产物：去掉多段/温度/时段判定，只留「5min 够不够」。"""
    now = _dt.datetime.now()
    valid_days = int(args.valid_days)
    checks: dict[str, list[str]] = {"passed": [], "warnings": [], "failed": []}

    def ok(c, msg): checks["passed" if c else "failed"].append(msg)

    total = float(health["total_duration_s"])
    nwin = int(health["independent_windows"])
    ok(total >= 300.0, f"静止总时长 {total}s ≥ 300s（5min 最低要求）")
    ok(nwin >= 10, f"独立 30s 窗 {nwin} ≥ 10 个")
    ok(health["irregular_dt_pct"] < 5.0,
       f"丢帧/不规则间隔 {health['irregular_dt_pct']}% < 5%")
    ok(not gravity["saturation"], "无量程饱和")
    ok(abs(gravity["gravity_norm_dev_pct"]) < 5.0,
       f"重力模长偏差 {gravity['gravity_norm_dev_pct']}% < 5%")
    ok(noise["white_like"], f"本底近似白噪 lag1={noise['lag1_autocorr']}")
    ok(fa["recommended_g1"] >= 10.0,
       f"G1 沿用生产基线 10（静止 p99.9={fa['g1_ratio_p999']}，有 {fa['margin_to_production']}× 余量，不擅自下调）")
    ok(inj["recommended_g2_cv_max"] >= 0.08, "G2 未擅自收紧生产基线 8%")
    ok(inj["f_true_min_hz"] is not None, "注入标定检出真信号")

    # —— WARN：诚实记录单段 5min 的统计局限，不因简化而隐瞒 ——
    checks["warnings"].append(
        f"本底采用 p10（{noise['n_windows']} 窗，系统性偏低 {noise['p10_bias_pct']}%、"
        f"标准误 {noise['p10_se_mg']}mg），偏低量已由 G0 因子 {fa['derived_g0_factor']} 补偿")
    checks["warnings"].append(
        "虚警基线实测仅 " + str(fa["n_independent_windows"]) + " 个独立窗，重采样 "
        + str(fa["pseudo_replication_factor"]) + "× 属伪重复；已用独立合成白噪交叉验证")
    if exc["status"] != "VERIFIED":
        checks["warnings"].append(
            "无真实激励段：注入扫描为合成信号只验证算法，不验证标度/相位/夹具谐振/松动（可选 --excitation）")
    if health["unique_values"] < 16:
        checks["warnings"].append(
            f"静止只有 {health['unique_values']} 个唯一值、量化台阶 {health['quant_step_mg']}mg，"
            "分辨率受量化限制，G1 标定结果对量程敏感")
    if noise["raw_std_vs_floor_pct"] > 5:
        checks["warnings"].append(
            f"原始 std 比生产口径本底高 {noise['raw_std_vs_floor_pct']}%，已改用同口径标定值")
    checks["warnings"].append("方法固有局限：四门只判有无稳定窄带峰，不判峰的物理来源")
    checks["warnings"].append("低频灵敏度下降：" + inj["low_freq_sensitivity"])

    verdict = "READY" if not checks["failed"] else "INSUFFICIENT"

    return {
        "schema_version": ca.SCHEMA_VERSION,
        "generated_at": now.isoformat(timespec="seconds"),
        "tool": "calibrate_accel_single.py",
        "mode": "single_static",
        "scope": {
            "pose": "固定垂直安装姿态（单姿态）",
            "temperature": "未纳入标定（适用于环境温度稳定场景；温度大幅波动请改用 calibrate_accel.py 多段版）",
            "time": "未纳入标定（单段 ≥5min）",
            "note": "本脚本只做频率提取正确与可信度判定（本底 + 四门 + 频带 + 亚格点插值），不涉索力/挠度换算",
        },
        "sensor": {
            "id": args.sensor_id or "UNSET",
            "model": args.model,
            "range_g": float(args.range_g),
            "rate_hz": round(rec["fs"], 4),
            "axis_default": rec["axis"],
        },
        "source": {"file": rec["name"], "sha256": rec["sha256"], "rows": rec["rows"]},
        "levels": {
            "L1_sensor": {"health": health, "gravity": gravity, "noise": noise},
            "L2_thresholds": {"false_alarm": fa, "injection": {k: v for k, v in inj.items()
                                                               if k != "grid"},
                              "confusion_band_hz": confusion,
                              "hardware_excitation": exc},
        },
        "data_sufficiency": {
            "verdict": verdict,
            "verdict_meaning": {
                "READY": "单段 ≥5min 且 ≥10 个完整 30s 窗，本底 + 四门阈值已标定，可安装后加载提取频率并判定可信度",
                "INSUFFICIENT": "静止不足 5min 或完整 30s 窗不足 10 个，参数仅供参考，禁止投用",
            },
            "total_static_s": total,
            "segments": 1,
            "independent_windows": nwin,
            "p10_se_mg": noise["p10_se_mg"],
            "p10_bias_pct": noise["p10_bias_pct"],
            "hardware_excitation": exc["status"],
        },
        "params": params,
        "fingerprint": {
            "axis_columns": [c for c in rec["cols"] if "加速度" in c],
            "rate_hz": round(rec["fs"], 4),
            "range_g": float(args.range_g),
            "unit_scale": 1000.0,
            "axis_default": rec["axis"],
            "window_s": float(params["window_s"]),
            "nperseg_s": float(params["nperseg_s"]),
        },
        "validity": {
            "calibrated_on": now.date().isoformat(),
            "valid_until": (now.date() + _dt.timedelta(days=valid_days)).isoformat(),
            "valid_days": valid_days,
            "recalibrate_if": ca.RECALIBRATE_IF,
        },
        "checks": checks,
    }


def write_report_single(path: str, c: dict[str, Any]) -> None:
    L = c["levels"]
    h, g, n = L["L1_sensor"]["health"], L["L1_sensor"]["gravity"], L["L1_sensor"]["noise"]
    fa, inj = L["L2_thresholds"]["false_alarm"], L["L2_thresholds"]["injection"]
    s, fp, v, ds = c["sensor"], c["fingerprint"], c["validity"], c["data_sufficiency"]
    out: list[str] = []
    add = out.append
    add("=" * 78)
    add("加速度计单段 5min 静止标定报告（固定垂直姿态）")
    add("=" * 78)
    add(f"传感器      : {s['model']}  ID={s['id']}  量程 ±{s['range_g']}g  "
        f"速率 {s['rate_hz']}Hz  默认轴 {s['axis_default']}")
    add(f"标定时间    : {c['generated_at']}   模式 {c['mode']}")
    add(f"适用范围    : {c['scope']['pose']} ｜ {c['scope']['temperature']}")
    add(f"数据指纹    : {c['source']['sha256'][:16]}…  {c['source']['rows']} 行 / {ds['total_static_s']}s")
    add(f"有效期至    : {v['valid_until']}（{v['valid_days']} 天）")
    add("")
    add("-" * 78)
    add("STEP 0  数据体检")
    add("-" * 78)
    add(f"  fs={h['fs_hz']}Hz  dt中位数={h['dt_median_s']}s  "
        f"不规则间隔 {h['irregular_dt_count']} 处 ({h['irregular_dt_pct']}%)")
    add(f"  唯一值 {h['unique_values']} 个  量化台阶 {h['quant_step_mg']}mg  "
        f"峰峰 {h['peak_to_peak_mg']}mg  最长常数段 {h['longest_constant_run']} 点")
    add(f"  独立 30s 窗 {ds['independent_windows']} 个（完整窗，残窗不计）")
    add("")
    add("-" * 78)
    add("STEP 1  出厂标定验证（不修改模块内部参数）")
    add("-" * 78)
    add(f"  方法={g['method']}  重力模长={g['gravity_norm_g']}g "
        f"(偏差 {g['gravity_norm_dev_pct']}%)  安装倾角≈{g['mount_tilt_deg']}°")
    add(f"  三轴均值向量 = {g['mean_vector_g']} g")
    add(f"  {g['note']}")
    add("")
    add("-" * 78)
    add("STEP 2  本底（与生产 G0 同口径）")
    add("-" * 78)
    add(f"  noise_floor_mg = {n['noise_floor_mg']} mg  "
        f"(预处理后逐 30s 窗 RMS 的 10 分位)")
    add(f"  逐窗 RMS = {n['window_rms_mg']}")
    add(f"  p10 系统性偏低 {n['p10_bias_pct']}%  标准误 {n['p10_se_mg']}mg  "
        f"原始 std {n['raw_std_mg']}mg（差 {n['raw_std_vs_floor_pct']}%）")
    add(f"  白噪性 lag1={n['lag1_autocorr']}（<0.15 判白噪）")
    add("")
    add("-" * 78)
    add("STEP 3  各门阈值（安装后直接加载判定 VALID/INVALID）")
    add("-" * 78)
    add(f"  G0  幅度门      = RMS ≥ {c['params']['noise_floor_mg']} × {c['params']['g0_rms_factor']}")
    add(f"  G1  峰/中位数   ≥ {c['params']['g1_peak_ratio']}  "
        f"(静止实测 p50/p99.9/max={fa['g1_ratio_p50']}/{fa['g1_ratio_p999']}/{fa['g1_ratio_max']}, "
        f"合成白噪 p99.9={fa['g1_synthetic_p999']})")
    add(f"  G2  子窗 CV     ≤ {c['params']['g2_cv_max']*100:.1f}%  "
        f"(注入 CV p99={inj['injected_valid_cv_p99']}, 不擅自收紧生产基线 8%)")
    add(f"  G3  频率下限    ≥ {c['params']['f1_min_hz']}Hz  "
        f"(拦累计量/慢波假峰；注入 3mg 稳定检出最低 {inj['f_true_min_hz']}Hz)")
    add(f"  分析带          = {c['params']['f_lo']} ~ {c['params']['f_hi']}Hz")
    add(f"  过渡带混淆区    = {c['levels']['L2_thresholds']['confusion_band_hz']}Hz（需外部证据）")
    add("")
    add("-" * 78)
    add("频率提取精度")
    add("-" * 78)
    add(f"  窗 {c['params']['window_s']}s / Welch 段 {c['params']['nperseg_s']}s → 格点分辨率 Δf = {inj['welch_delta_f_hz']}Hz")
    add(f"  refine_f1 = 亚格点 log 抛物线插值（误差 ~1.8mHz，远细于格点 Δf）")
    add(f"  主频取 SNR 更高轴；跨窗稳定性由 G2 子窗 CV 把关")
    add("")
    add("-" * 78)
    add("数据充分性")
    add("-" * 78)
    add(f"  结论 = {ds['verdict']}  —  {ds['verdict_meaning'][ds['verdict']]}")
    add(f"  总静止 {ds['total_static_s']}s / {ds['segments']} 段 / {ds['independent_windows']} 独立窗"
        f" / 本底标准误 {ds['p10_se_mg']}mg、偏低 {ds['p10_bias_pct']}%")
    add(f"  硬件激励段 = {ds['hardware_excitation']}")
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
    ap = argparse.ArgumentParser(description="加速度计单段 5min 静止标定（固定垂直姿态）")
    ap.add_argument("--static", required=True, help="静止数据文件（≥5min，垂直放置）")
    ap.add_argument("--axis", default="", help="主标定轴列名，默认自动选加速度Z")
    ap.add_argument("--sensor-id", default="", help="传感器唯一编号（防串用）")
    ap.add_argument("--model", default="WT901SDCL-BT50")
    ap.add_argument("--range-g", type=float, default=2.0, help="量程 ±g")
    ap.add_argument("--window-s", type=float, default=30.0)
    ap.add_argument("--nperseg-s", type=float, default=8.0)
    ap.add_argument("--n-mc", type=int, default=150, help="虚警基线重采样次数")
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--valid-days", type=int, default=180)
    ap.add_argument("--out", default="accel_calib_single.json")
    ap.add_argument("--report", default="accel_calib_single_report.txt")
    ap.add_argument("--excitation", default="",
                    help="已知频率的台架激励数据（可选，验证硬件链路）")
    ap.add_argument("--excitation-hz", type=float, default=0.0,
                    help="激励真值频率 Hz，配合 --excitation")
    args = ap.parse_args(argv)

    rec = ca._read_one(ca._resolve(args.static), args.axis)
    p = fg.merge_params({"window_s": args.window_s, "nperseg_s": args.nperseg_s,
                         "axis": rec["axis"], "multi_axis": False,
                         "harmonic_check": False, "xcheck": False})
    pool = ca.build_window_pool([rec], p)
    health = ca.step0_health(rec)
    health.update(ca.step0_summary([rec], p, pool))
    fs = rec["fs"]
    gravity = ca.step1_gravity([rec], [""], args.range_g)
    noise, nf = ca.step2_noise(pool, p, fs, rec["y"], health)
    rng = np.random.default_rng(args.seed)
    fa = ca.step3_false_alarm(pool, p, fs, nf, noise, args.n_mc, rng)
    inj = ca.step4_injection(pool, p, fs, nf)
    exc = ca.step5_excitation(args.excitation, args.excitation_hz, args.axis, p, nf)
    params, confusion = ca.derive_params(rec, p, noise, fa, inj, gravity)
    calib = build_calib_single(args, rec, health, gravity, noise, fa, inj,
                               params, confusion, exc)

    out = ca._resolve(args.out)
    rep = ca._resolve(args.report)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(calib, fh, ensure_ascii=False, indent=2)
    write_report_single(rep, calib)

    print(f"标定完成 → {out}")
    print(f"报告     → {rep}")
    print(f"本底 RMS = {noise['noise_floor_mg']} mg  (原始 std {noise['raw_std_mg']} mg)")
    print(f"采用阈值 = G0×{params['g0_rms_factor']}  G1≥{params['g1_peak_ratio']}  "
          f"G2 CV≤{params['g2_cv_max']*100:.1f}%  f1_min={params['f1_min_hz']}Hz  "
          f"带 {params['f_lo']}~{params['f_hi']}Hz")
    print(f"数据充分性 = {calib['data_sufficiency']['verdict']} | 静止 "
          f"{health['total_duration_s']}s / {health['independent_windows']} 独立窗 | "
          f"本底标准误 {noise['p10_se_mg']}mg、偏低 {noise['p10_bias_pct']}%")
    print(f"硬件激励段 = {exc['status']}  {exc['reason'][:52]}")
    print(f"检查     = PASS {len(calib['checks']['passed'])} / "
          f"WARN {len(calib['checks']['warnings'])} / "
          f"FAIL {len(calib['checks']['failed'])}")
    for x in calib["checks"]["failed"]:
        print(f"  FAIL  {x}")
    verdict = calib["data_sufficiency"]["verdict"]
    print(f"结论     = {verdict} — {calib['data_sufficiency']['verdict_meaning'][verdict]}")
    print("目标     = 频率提取正确与可信度判定（不涉索力/挠度换算）")
    return 1 if calib["checks"]["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
