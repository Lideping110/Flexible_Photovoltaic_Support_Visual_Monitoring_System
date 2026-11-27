# -*- coding: utf-8 -*-
"""用主链新产的 displacement.jsonl 交叉验证旧标定 JSON 是否仍适用。

场景：换了靶标位置/距离后，重新跑 5min 主链（不带标定参数），
用新位移数据检验旧标定的三项前提是否还成立：
  1. 帧率指纹：新数据 fs vs 标定 rate_hz（>5% 阈值失效）
  2. 本底一致性：新静止本底 vs 标定本底（px 域与主链路判定域一致）
  3. 零虚警：新静止数据喂进旧标定六门，应全部 INVALID；
     同时给出 legacy 口径对照（旧口径会误报多少）。

mm/px 只作观察项（判定在 px 域，不影响门控；换位置后各靶标 mm/px 变化属预期）。
只复用现有算法原子：gate_core / spectrum / frequency_gate，不重写谱算法。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "sensor_data", "accelerometer"))


def _f(v, default=0.0):
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _uniform(ts, ys, dt):
    t = np.asarray(ts, dtype=float)
    y = np.asarray(ys, dtype=float)
    keep = np.concatenate(([True], np.diff(t) > 1e-9))
    t, y = t[keep], y[keep]
    if len(t) < 2:
        return t, y
    n = max(8, int(round((t[-1] - t[0]) / dt)) + 1)
    tu = t[0] + np.arange(n) * dt
    return tu, np.interp(tu, t, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="output/marker/marker_cam_01/displacement.jsonl")
    ap.add_argument("--calib", default="video_calibration/marker_cam_01_calib.json")
    ap.add_argument("--axis", default="dy", choices=("dx", "dy"))
    ap.add_argument("--save-json", default="")
    args = ap.parse_args()

    import frequency_gate as fg
    from marker_subpixel_tracker import gate_core as gc
    from marker_subpixel_tracker.spectrum import _segment_spectrum

    calib_path = args.calib if os.path.isabs(args.calib) else os.path.join(_ROOT, args.calib)
    with open(calib_path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    cfg = gc.load_gate_config(calib_path)
    if cfg is None:
        raise SystemExit(f"标定 JSON 无法加载: {calib_path}")
    p = cfg["params"]
    fs_cal = float(doc["fingerprint"].get("rate_hz") or 0)
    mmp_cal = float(doc["fingerprint"].get("mm_per_px") or 0)
    _noise = ((doc.get("levels") or {}).get("L1_sensor") or {}).get("noise") or {}
    floor_cal_mm = _f(_noise.get("noise_floor_mm"))
    floor_cal_px = float((_noise.get("per_pid_floor_px") or {}).get("M1", 0)) or None
    print(f"[标定] {os.path.basename(calib_path)} verdict={cfg['verdict']} "
          f"fs={fs_cal:.4f}Hz mm/px={mmp_cal:.6f} 本底={floor_cal_mm}mm "
          f"(px≈{floor_cal_px})")

    jpath = args.jsonl if os.path.isabs(args.jsonl) else os.path.join(_ROOT, args.jsonl)
    rows = []
    for line in open(jpath, "r", encoding="utf-8"):
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    by_pid: dict[str, list[dict]] = {}
    for r in rows:
        by_pid.setdefault(str(r.get("pid")), []).append(r)
    print(f"[数据] {jpath}  records={len(rows)}  靶标={list(by_pid)}")

    W = float(p.get("window_s", 30.0))
    axis = args.axis
    cum_mm = f"cum_{axis}_mm"
    raw_px = "cy" if axis == "dy" else "cx"
    report = {
        "jsonl": os.path.basename(jpath), "calib": os.path.basename(calib_path),
        "axis": axis, "fs_cal": fs_cal, "targets": {}, "verdict": "",
    }
    all_cal_valid = all_legacy_valid = all_windows = 0
    fs_list, floor_mm_list, floor_px_list = [], [], []

    for pid, smp in sorted(by_pid.items()):
        t_all = [float(s["ts"]) for s in smp]
        dt = float(np.median(np.diff(sorted(t_all))))
        fs = 1.0 / dt
        fs_list.append(fs)
        t_max = max(t_all)
        # px 域累计位移：用 cy/cx 减首帧（判定域），mm 域用 cum_*_mm
        # 注意 ref 不随跟踪丢失重置，但静止段首帧即锚点，与生产口径一致
        y0 = float(smp[0][raw_px])
        px_all = np.array([float(s[raw_px]) - y0 for s in smp], dtype=float)
        mm_all = np.array([float(s[cum_mm]) for s in smp], dtype=float)
        mmp = float(np.median([_f(s["mm_per_px"]) for s in smp if s.get("mm_per_px")]))

        # 本底：逐完整窗预处理后 RMS 的 p10（mm 与 px 两域）
        rms_mm, rms_px = [], []
        seg = 0
        while seg * W <= t_max + 1e-9:
            lo, hi = seg * W, (seg + 1) * W
            idx = [i for i, tt in enumerate(t_all) if lo <= tt < hi]
            if len(idx) >= 32 and hi <= t_max + 1e-9:
                tu, yu_mm = _uniform([t_all[i] for i in idx], [mm_all[i] for i in idx], dt)
                _, yu_px = _uniform([t_all[i] for i in idx], [px_all[i] for i in idx], dt)
                xm, _ = fg.preprocess_window(yu_mm, fs, p)
                xp, _ = fg.preprocess_window(yu_px, fs, p)
                rms_mm.append(float(np.sqrt(np.mean(xm * xm))))
                rms_px.append(float(np.sqrt(np.mean(xp * xp))))
            seg += 1
        floor_mm = float(np.percentile(rms_mm, 10)) if rms_mm else None
        floor_px = float(np.percentile(rms_px, 10)) if rms_px else None
        if floor_mm is not None:
            floor_mm_list.append(floor_mm)
        if floor_px is not None:
            floor_px_list.append(floor_px)

        # 逐窗六门判定 + legacy 对照
        cal_valid = legacy_valid = nwin = 0
        win_rows = []
        seg = 0
        while seg * W <= t_max + 1e-9:
            lo, hi = seg * W, (seg + 1) * W
            idx = [i for i, tt in enumerate(t_all) if lo <= tt < hi]
            if len(idx) >= 32 and hi <= t_max + 1e-9:
                tu, yu_mm = _uniform([t_all[i] for i in idx], [mm_all[i] for i in idx], dt)
                _, yu_px = _uniform([t_all[i] for i in idx], [px_all[i] for i in idx], dt)
                meta = {
                    "frame_id": [int(smp[i]["frame_id"]) for i in idx],
                    "raw_track_id": [int(smp[i]["raw_track_id"]) for i in idx],
                    "continuous": [bool(smp[i]["continuous"]) for i in idx],
                }
                need_n = int(p.get("min_samples", 32))
                need_span = min(float(p.get("min_span_s", 10.0)), W)
                r = gc.evaluate_window(tu, yu_px, yu_mm, 10.0, cfg=cfg, meta=meta,
                                       window_s=W, min_samples=need_n, min_span_s=need_span)
                lg = _segment_spectrum(tu, yu_mm, 5.0, window_s=W, min_samples=need_n,
                                       min_span_s=need_span)
                cv_ = bool(r.get("valid") and r.get("f1_hz") is not None)
                lv_ = bool(lg.get("valid") and lg.get("f1_hz") is not None)
                cal_valid += int(cv_)
                legacy_valid += int(lv_)
                nwin += 1
                g = r.get("gates") or {}
                fail = []
                for k in ("G0", "G1", "G2", "G3"):
                    v = g.get(k) or {}
                    if v.get("enabled", True) and not v.get("pass", False):
                        fail.append(k)
                for k in ("drop", "id"):
                    if (g.get(k) or {}).get("status") == "FAIL":
                        fail.append(k)
                win_rows.append({
                    "seg": seg, "t0": round(lo, 1), "cal_valid": cv_,
                    "legacy_valid": lv_, "f1_cal": r.get("f1_hz"),
                    "f1_legacy": lg.get("f1_hz"), "failed": fail,
                    "reason": r.get("reason", ""),
                })
            seg += 1
        all_cal_valid += cal_valid
        all_legacy_valid += legacy_valid
        all_windows += nwin
        mmp_dev = abs(mmp - mmp_cal) / mmp_cal * 100 if mmp_cal else 0.0
        print(f"\n[靶标 {pid}] fs={fs:.3f}Hz mm/px={mmp:.6f}(偏差{mmp_dev:.2f}%) "
              f"n={len(smp)} 本底 mm={floor_mm} px={floor_px}")
        print(f"  完整窗 {nwin}：六门 VALID {cal_valid}，legacy VALID {legacy_valid}")
        for w in win_rows[:3]:
            print(f"    窗{w['seg']} 六门={'VALID' if w['cal_valid'] else 'INVALID'} "
                  f"legacy={'VALID' if w['legacy_valid'] else 'INVALID'} "
                  f"FAIL={','.join(w['failed']) if w['failed'] else '-'}")
        report["targets"][pid] = {
            "fs_hz": round(fs, 4), "mm_per_px": round(mmp, 6),
            "mm_per_px_dev_pct": round(mmp_dev, 3),
            "floor_mm": floor_mm, "floor_px": floor_px,
            "n_windows": nwin, "cal_valid": cal_valid,
            "legacy_valid": legacy_valid, "windows": win_rows,
        }

    fs_new = float(np.median(fs_list)) if fs_list else 0.0
    fs_dev = abs(fs_new - fs_cal) / fs_cal * 100 if fs_cal else 0.0
    floor_mm_new = float(np.mean(floor_mm_list)) if floor_mm_list else None
    floor_px_new = float(np.mean(floor_px_list)) if floor_px_list else None
    ratio_mm = floor_mm_new / floor_cal_mm if floor_mm_new and floor_cal_mm else None
    ratio_px = floor_px_new / floor_cal_px if floor_px_new and floor_cal_px else None

    print(f"\n{'='*74}")
    print(f"汇总：完整窗 {all_windows} 个，新数据 fs={fs_new:.4f}Hz (标定 {fs_cal:.4f}Hz, 偏差 {fs_dev:.2f}%)")
    print(f"  六门(旧标定) VALID = {all_cal_valid}   ← 静止期望 0")
    print(f"  legacy VALID       = {all_legacy_valid}   ← 旧口径误报对照")
    print(f"  本底 mm: 新 {floor_mm_new} / 标定 {floor_cal_mm} = {round(ratio_mm,3) if ratio_mm else '-'}")
    print(f"  本底 px: 新 {floor_px_new} / 标定 {floor_cal_px} = {round(ratio_px,3) if ratio_px else '-'}")
    ok_fs = fs_dev <= 5.0
    ok_floor_px = ratio_px is not None and 0.5 <= ratio_px <= 2.0
    ok_floor_mm = ratio_mm is not None and 0.5 <= ratio_mm <= 2.0
    ok_zero = all_cal_valid == 0
    print(f"  判定：帧率={'PASS' if ok_fs else 'FAIL'}  "
          f"本底px={'PASS' if ok_floor_px else 'FAIL'}  "
          f"本底mm={'PASS' if ok_floor_mm else 'FAIL'}  "
          f"零虚警={'PASS' if ok_zero else 'FAIL'}")
    verdict = "PASS" if (ok_fs and ok_floor_px and ok_floor_mm and ok_zero) else "FAIL"
    print(f"结论 = {verdict}")
    if verdict == "FAIL":
        reasons = []
        if not ok_fs:
            reasons.append(f"帧率偏差 {fs_dev:.2f}% > 5%")
        if not ok_floor_px:
            reasons.append(f"px 本底比 {round(ratio_px,3) if ratio_px else '-'} 越界")
        if not ok_floor_mm:
            reasons.append(f"mm 本底比 {round(ratio_mm,3) if ratio_mm else '-'} 越界")
        if not ok_zero:
            reasons.append(f"静止误报 {all_cal_valid} 窗")
        print("  失效原因：" + "；".join(reasons))
        print("  建议：换靶标位置属重标触发条件（mm/px/本底/目标数全变），应重新标定")
    report.update({
        "fs_new": round(fs_new, 4), "fs_dev_pct": round(fs_dev, 3),
        "floor_new_mm": floor_mm_new, "floor_new_px": floor_px_new,
        "floor_cal_mm": floor_cal_mm, "floor_cal_px": floor_cal_px,
        "floor_ratio_mm": round(ratio_mm, 5) if ratio_mm else None,
        "floor_ratio_px": round(ratio_px, 5) if ratio_px else None,
        "n_windows": all_windows, "cal_valid": all_cal_valid,
        "legacy_valid": all_legacy_valid,
        "checks": {"frame_rate": ok_fs, "floor_px": ok_floor_px,
                   "floor_mm": ok_floor_mm, "zero_false_alarm": ok_zero},
        "verdict": verdict,
    })
    if args.save_json:
        out = args.save_json if os.path.isabs(args.save_json) else os.path.join(_ROOT, args.save_json)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        print(f"明细 → {out}")


if __name__ == "__main__":
    main()
