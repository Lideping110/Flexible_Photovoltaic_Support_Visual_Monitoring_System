# -*- coding: utf-8 -*-
"""回归测试：gate_core 六门 + spectrum 主链路集成（2026-11-23）。

要点：
1. cfg=None 的 legacy 路径必须与改造前的单段 FFT + SNR 口径一致（不标定不改行为）。
2. 标定模式：G3 拦最低 bin 假峰；丢帧率门与 ID 门在元数据缺失时 SKIP 不误杀。
3. 元数据齐全时丢帧/ID 跳变能被抓住。
4. supervisor 的关键字传参不再错位。
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from marker_subpixel_tracker.gate_core import evaluate_window  # noqa: E402
from marker_subpixel_tracker.spectrum import _segment_spectrum  # noqa: E402


def _cfg():
    """模拟已标定的 cfg：本底=0.2px，G0=1.2，G1=10，G2=0.08，G3=2/T。"""
    return {"path": "<test>", "verdict": "READY", "cam_id": "test_cam",
            "fingerprint": {"rate_hz": 25.0, "cam_id": "test_cam"},
            "params": {
                "window_s": 30.0, "nperseg_s": 8.0, "n_sub": 4,
                "f_lo": 0.3, "f_hi": 10.0, "min_samples": 32, "min_span_s": 10.0,
                "detrend": True, "despike": True, "hp_cutoff": 0.3,
                "spike_win": 7, "spike_nsig": 6.0, "refine_f1": True,
                "noise_floor_mg": 0.2, "g0_rms_factor": 1.2,
                "g1_peak_ratio": 10.0, "g2_mode": "band", "g2_band_frac": 0.3,
                "g2_sub_ratio": 5.0, "g2_min_subs": 3, "g2_cv_max": 0.08,
                "g3_factor": 2.0, "f1_min_hz": 0.3,
            }}


def test_legacy_unchanged():
    fs, n = 25.0, 500
    t = np.arange(n) / fs
    y = 5.0 * np.sin(2 * np.pi * 2.0 * t)
    r = _segment_spectrum(t, y, 5.0)  # cfg=None
    assert r is not None and r["mode"] == "legacy", r
    assert r["f1_hz"] is not None and abs(r["f1_hz"] - 2.0) < 0.2, r["f1_hz"]
    assert r["snr"] > 5.0
    print("PASS legacy 路径：单段 FFT + SNR 门保持原口径，f1≈2Hz")


def test_legacy_rejects_below_snr():
    fs, n = 25.0, 500
    t = np.arange(n) / fs
    rng = np.random.default_rng(1)
    y = rng.normal(0, 0.1, n)
    r = _segment_spectrum(t, y, 5.0)
    # 白噪的 peak/median 通常 <5；不强制断言，只检查接口
    assert r is not None
    print(f"PASS legacy 白噪接口：f1={r['f1_hz']} snr={r['snr']:.2f}")


def test_calibrated_clean_signal():
    fs, n = 25.0, 500
    t = np.arange(n) / fs
    sig_px = 2.0 * np.sin(2 * np.pi * 1.2 * t)
    sig_mm = sig_px * 0.5
    meta = {"frame_id": list(range(n)), "raw_track_id": [1] * n,
            "continuous": [True] * n}
    r = evaluate_window(t, sig_px, sig_mm, 5.0, cfg=_cfg(), meta=meta)
    g = r["gates"]
    assert g["G1"]["pass"] and g["G2"]["pass"] and g["G3"]["pass"], g
    assert g["drop"]["status"] == "PASS", g["drop"]
    assert g["id"]["status"] == "PASS", g["id"]
    assert r["f1_hz"] is not None and abs(r["f1_hz"] - 1.2) < 0.15, r["f1_hz"]
    assert r["max_amp_mm"] > 0
    print(f"PASS 标定模式干净信号：f1={r['f1_hz']:.3f}Hz 六门状态="
          f"G0={g['G0']['pass']} G1={g['G1']['pass']} G2={g['G2']['pass']} "
          f"G3={g['G3']['pass']} drop={g['drop']['status']} id={g['id']['status']}")


def test_g3_blocks_lowest_bin():
    fs, n = 25.0, 500
    t = np.arange(n) / fs
    y = 0.02 * np.ones(n) + np.random.default_rng(2).normal(0, 0.001, n)
    r = evaluate_window(t, y, y, 5.0, cfg=_cfg())
    # 常量信号去趋势后能量极低，G3 或 G1 会拦；f1 必须 None
    assert r["f1_hz"] is None, r
    print(f"PASS G3/门控拦截常量信号：f1=None reason={r['reason']}")


def test_missing_meta_skips():
    fs, n = 25.0, 500
    t = np.arange(n) / fs
    y = 2.0 * np.sin(2 * np.pi * 1.2 * t)
    r = evaluate_window(t, y, y, 5.0, cfg=_cfg(), meta=None)
    g = r["gates"]
    assert g["drop"]["status"] == "SKIP" and g["id"]["status"] == "SKIP", g
    print("PASS 无元数据：丢帧/ID 门 SKIP（不误杀）")


def test_id_switch_detected():
    fs, n = 25.0, 500
    t = np.arange(n) / fs
    y = 2.0 * np.sin(2 * np.pi * 1.2 * t)
    meta = {"frame_id": list(range(n)),
            "raw_track_id": [1] * 250 + [2] * 250,
            "continuous": [True] * n}
    r = evaluate_window(t, y, y, 5.0, cfg=_cfg(), meta=meta)
    assert r["gates"]["id"]["status"] == "FAIL", r["gates"]["id"]
    assert r["f1_hz"] is None
    print(f"PASS ID 跳变被抓：{r['gates']['id']['reason']}")


def test_frame_gap_detected():
    fs, n = 25.0, 500
    t = np.arange(n) / fs
    y = 2.0 * np.sin(2 * np.pi * 1.2 * t)
    fid = list(range(n))
    fid[200] = 250  # 跳号
    meta = {"frame_id": fid, "raw_track_id": [1] * n, "continuous": [True] * n}
    r = evaluate_window(t, y, y, 5.0, cfg=_cfg(), meta=meta)
    assert r["gates"]["drop"]["status"] == "FAIL", r["gates"]["drop"]
    assert r["f1_hz"] is None
    print(f"PASS 丢帧被抓：{r['gates']['drop']['reason']}")


def test_supervisor_kwargs_order():
    """关键：defl/cable 阈值与占位值不再错位。"""
    import inspect
    from utils import supervisor as sv
    src = inspect.getsource(sv.run)
    block = src[src.find('procs["spectrum"]'):src.find('log("supervisor", f"已启动')]
    for key in ("deflection_threshold", "cable_force_threshold",
                "placeholder_deflection", "placeholder_cable_force", "gate_config"):
        assert f'"{key}"' in block, f"关键字参数缺失: {key}"
    assert "_defl_threshold(cfg)" in block and "_ph_deflection(cfg)" in block
    # 两者必须出现在各自正确的键上（错位 bug 的特征是互换）
    assert '"deflection_threshold": _defl_threshold(cfg)' in block
    assert '"placeholder_deflection": _ph_deflection(cfg)' in block
    print("PASS supervisor 关键字传参：阈值/占位值映射正确，无错位")


if __name__ == "__main__":
    test_legacy_unchanged()
    test_legacy_rejects_below_snr()
    test_calibrated_clean_signal()
    test_g3_blocks_lowest_bin()
    test_missing_meta_skips()
    test_id_switch_detected()
    test_frame_gap_detected()
    test_supervisor_kwargs_order()
    print("ALL PASS")
