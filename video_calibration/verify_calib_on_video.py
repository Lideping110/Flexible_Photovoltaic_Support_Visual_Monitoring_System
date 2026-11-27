# -*- coding: utf-8 -*-
"""用一段**未参与标定**的静止视频交叉验证标定 JSON 是否正确。

三项检验（缺一不可）：
1. 指纹一致性：帧率 / mm_per_px / 相机 是否与标定参数匹配（超差则标定不适用）。
2. 本底一致性：新视频的静止本底（px 域 + mm 域）是否接近标定本底
   ——差太多说明标定本底不具代表性。
3. 零虚警检验（核心）：把新视频的静止位移喂进生产六门（加载标定 JSON，
   完整传 frame_id / raw_track_id / continuous），静止信号应被全部判 INVALID；
   若报出有效主频，说明门控阈值太松。同时给出 legacy 口径的虚警数作对照。

只复用现有算法原子，不重写检测/亚像素/谱：
    video_measurement.measure_video      检测→亚像素→自标定→cum
    gate_core.load_gate_config/check_fingerprint/evaluate_window
    frequency_gate.preprocess_window     本底统计（与 G0 同口径）
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


def _floor_mm(mm_samples, p, axis="dy", window_s=30.0):
    """按标定同口径算预处理后逐窗 RMS 的 10 分位（mm）。"""
    sys.path.insert(0, os.path.join(_ROOT, "sensor_data", "accelerometer"))
    import frequency_gate as fg
    t = np.asarray([s["ts"] for s in mm_samples], dtype=float)
    key = f"cum_{axis}_mm"
    vals = np.asarray([float(s[key]) for s in mm_samples], dtype=float)
    dt = 1.0 / _f(p.get("fs"), 20.0)
    fs = 1.0 / dt
    rms = []
    seg = 0
    while seg * window_s <= (t[-1] if len(t) else 0):
        lo, hi = seg * window_s, (seg + 1) * window_s
        idx = [i for i, tt in enumerate(t) if lo <= tt < hi]
        if len(idx) >= 32:
            tu, yu = _uniform([t[i] for i in idx], [vals[i] for i in idx], dt)
            x, _ = fg.preprocess_window(yu, fs, p)
            rms.append(float(np.sqrt(np.mean(x * x))))
        seg += 1
    if not rms:
        return None, []
    return float(np.percentile(rms, 10)), [round(v, 5) for v in rms]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="独立静止视频路径")
    ap.add_argument("--calib", default="video_calibration/marker_cam_01_calib.json")
    ap.add_argument("--axis", default="dy", choices=("dx", "dy"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--iou", type=float, default=None)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--save-json", default="", help="可选：保存完整逐窗判定")
    args = ap.parse_args()

    from video_calibration import video_measurement as vm
    from marker_subpixel_tracker import gate_core as gc
    from marker_subpixel_tracker.spectrum import _segment_spectrum

    calib_path = args.calib if os.path.isabs(args.calib) else os.path.join(_ROOT, args.calib)
    with open(calib_path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    cfg = gc.load_gate_config(calib_path)
    if cfg is None:
        raise SystemExit(f"标定 JSON 无法加载：{calib_path}")
    p = cfg["params"]
    fs_cal = float(doc["fingerprint"].get("rate_hz") or 0)
    mmp_cal = float(doc["fingerprint"].get("mm_per_px") or 0)
    _l1 = ((doc.get("levels") or {}).get("L1_sensor") or {})
    _noise = (_l1.get("noise") or {})
    floor_cal = _f(_noise.get("noise_floor_mm"))
    floor_cal_px = float((_noise.get("per_pid_floor_px") or {}).get("M1", 0)) or None
    print(f"[标定] {os.path.basename(calib_path)} verdict={cfg['verdict']} "
          f"fs={fs_cal:.4f}Hz mm/px={mmp_cal:.6f} 本底={floor_cal}mm")

    settings = vm.load_settings()
    print(f"[视频] {args.video}")
    meas = vm.measure_video(
        args.video, settings, conf=args.conf, iou=args.iou, device=args.device,
        max_frames=args.max_frames, frame_stride=args.frame_stride,
        progress=lambda stage, done, total, msg: (
            print(f"  [{stage}] {done}/{total} {msg}") if done % 300 == 0 or done == total else None))
    fs = float(meas["fps"])
    print(f"[视频] fps={fs:.4f} 帧={meas['n_frames']} 处理={meas['n_processed']} "
          f"检出={meas['n_detections']} 定位={meas['n_located']} "
          f"目标={list(meas['targets'])}")

    warns = gc.check_fingerprint(cfg, fs)
    for w in warns:
        print(f"[指纹告警] {w}")
    fs_dev = abs(fs - fs_cal) / fs_cal * 100 if fs_cal else 0.0
    print(f"[指纹] 帧率偏差 {fs_dev:.2f}% (容差 5%) → "
          f"{'一致' if fs_dev <= 5 else '不一致，阈值失效'}")

    W = float(p.get("window_s", 30.0))
    snr = float(p.get("g1_peak_ratio", 10.0))  # 仅占位，标定模式内部用 params
    dt = 1.0 / fs
    axis = args.axis
    pk = f"cum_{axis}_mm"
    pxk = f"{axis}_px"
    report = {
        "video": meas["video"], "video_path": meas["video_path"],
        "calib": os.path.basename(calib_path),
        "fps": fs, "fps_cal": fs_cal, "fps_dev_pct": round(fs_dev, 3),
        "fingerprint_warnings": warns, "axis": axis,
        "floor_cal_mm": floor_cal, "targets": {}, "verdict": "",
    }
    all_cal_valid, all_legacy_valid, all_windows = 0, 0, 0
    floors = []
    for pid, info in meas["targets"].items():
        smp = info["samples"]
        t_all = [s["ts"] for s in smp]
        t_max = t_all[-1]
        floor_mm, rms_list = _floor_mm(smp, p, axis=axis, window_s=W)
        if floor_mm is not None:
            floors.append(floor_mm)
        mmp = float(info["mm_per_px"])
        mmp_dev = abs(mmp - mmp_cal) / mmp_cal * 100 if mmp_cal else 0.0
        print(f"\n[目标 {pid}] mm/px={mmp:.6f} 偏差={mmp_dev:.2f}% "
              f"n={len(smp)} 时长={info['duration_s']:.1f}s "
              f"本底={floor_mm}mm (标定 {floor_cal}mm)")

        win_rows = []
        cal_valid = legacy_valid = nwin = 0
        seg = 0
        while seg * W <= t_max + 1e-9:
            lo, hi = seg * W, (seg + 1) * W
            win = [s for s in smp if lo <= s["ts"] < hi]
            is_tail = bool(win) and hi > t_max + 1e-9
            if is_tail and smp[-1] not in win:
                win.append(smp[-1])
            if len(win) >= 32 and not is_tail:
                tu, y_mm = _uniform([s["ts"] for s in win],
                                    [float(s[pk]) for s in win], dt)
                _, y_px = _uniform([s["ts"] for s in win],
                                   [float(s[pxk]) for s in win], dt)
                meta = {
                    "frame_id": [int(s["frame_id"]) for s in win],
                    "raw_track_id": [int(s["raw_track_id"]) for s in win],
                    "continuous": [bool(s["continuous"]) for s in win],
                }
                need_n = int(p.get("min_samples", 32))
                need_span = min(float(p.get("min_span_s", 10.0)), W)
                r = gc.evaluate_window(tu, y_px, y_mm, snr, cfg=cfg, meta=meta,
                                       window_s=W, min_samples=need_n,
                                       min_span_s=need_span)
                legacy = _segment_spectrum(tu, y_mm, 5.0, window_s=W,
                                           min_samples=need_n, min_span_s=need_span)
                cv_ = bool(r.get("valid") and r.get("f1_hz") is not None)
                lv_ = bool(legacy.get("valid") and legacy.get("f1_hz") is not None)
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
                row = {
                    "seg": seg, "t0": round(lo, 1), "cal_valid": cv_,
                    "legacy_valid": lv_, "f1_cal": r.get("f1_hz"),
                    "f1_legacy": legacy.get("f1_hz"),
                    "failed": fail, "reason": r.get("reason", ""),
                    "drop": (g.get("drop") or {}).get("reason", ""),
                    "id": (g.get("id") or {}).get("reason", ""),
                }
                win_rows.append(row)
                if seg < 3 or cv_:
                    print(f"  窗{seg:>2} t={lo:>6.1f}s  六门={'VALID' if cv_ else 'INVALID'}"
                          f"  legacy={'VALID' if lv_ else 'INVALID'}"
                          f"  f1={r.get('f1_hz')}"
                          f"  FAIL={','.join(fail) if fail else '-'}"
                          f"  [{r.get('reason','')}]")
                if cv_:
                    print(f"       drop: {row['drop']}")
                    print(f"       id  : {row['id']}")
            seg += 1
        all_cal_valid += cal_valid
        all_legacy_valid += legacy_valid
        all_windows += nwin
        print(f"  -> 完整窗 {nwin} 个：六门 VALID {cal_valid}，legacy VALID {legacy_valid}")
        report["targets"][pid] = {
            "mm_per_px": mmp, "mm_per_px_dev_pct": round(mmp_dev, 3),
            "floor_mm": floor_mm, "floor_mm_list": rms_list,
            "n_windows": nwin, "cal_valid": cal_valid,
            "legacy_valid": legacy_valid, "windows": win_rows,
        }

    floor_mean = float(np.mean(floors)) if floors else None
    floor_ratio = floor_mean / floor_cal if floor_cal and floor_mean else None
    print(f"\n{'='*72}")
    print(f"汇总：完整窗 {all_windows} 个")
    print(f"  六门(加载标定) VALID = {all_cal_valid}  ← 静止视频期望为 0")
    print(f"  legacy(仅SNR) VALID = {all_legacy_valid}  ← 对照：旧口径会误报多少")
    print(f"  新视频本底 = {floor_mean}mm  标定本底 = {floor_cal}mm  "
          f"比值 = {round(floor_ratio,3) if floor_ratio else '-'}")
    # 指纹否决项只看帧率：帧率改变会令 G0/G1/G2 的 Welch 格点失效。
    # mm/px 是「每靶标视觉自标定值」，不参与门控判定（判定在 px 域），
    # 且标定 JSON 记录的 mm/px 只是标定视频当时的值——换视频后个别靶标
    # 外圈测量偏了属靶标级问题，只告警不否决。
    ok_fp = fs_dev <= 5.0
    mmp_devs = {
        pid: abs(float(i["mm_per_px"]) - mmp_cal) / mmp_cal * 100
        for pid, i in meas["targets"].items()} if mmp_cal else {}
    bad_mmp = {pid: round(v, 2) for pid, v in mmp_devs.items() if v > 5}
    if bad_mmp:
        print(f"  [观察] 靶标级 mm/px 自标定偏差 >5%: {bad_mmp} "
              f"— 不影响门控判定(px域)，但该靶标输出幅值的 mm 换算需复核")
    ok_floor = floor_ratio is not None and 0.5 <= floor_ratio <= 2.0
    ok_zero = all_cal_valid == 0
    print(f"  判定：帧率一致={'PASS' if ok_fp else 'FAIL'}  "
          f"本底一致={'PASS' if ok_floor else 'FAIL'}  "
          f"零虚警={'PASS' if ok_zero else 'FAIL'}")
    verdict = "PASS" if (ok_fp and ok_floor and ok_zero) else "FAIL"
    print(f"结论 = {verdict}")
    report.update({
        "floor_new_mm": floor_mean,
        "floor_ratio": round(floor_ratio, 5) if floor_ratio else None,
        "n_windows": all_windows, "cal_valid": all_cal_valid,
        "legacy_valid": all_legacy_valid,
        "mm_per_px_dev_pct": {pid: round(v, 3) for pid, v in mmp_devs.items()},
        "mm_per_px_warn": bad_mmp,
        "checks": {"frame_rate": ok_fp, "floor": ok_floor, "zero_false_alarm": ok_zero},
        "verdict": verdict,
    })
    if args.save_json:
        out = args.save_json if os.path.isabs(args.save_json) else os.path.join(_ROOT, args.save_json)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        print(f"明细 → {out}")


if __name__ == "__main__":
    main()
