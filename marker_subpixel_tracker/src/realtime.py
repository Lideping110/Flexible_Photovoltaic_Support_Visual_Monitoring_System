"""Camera-first realtime marker tracing with video fallback.

This entry point mirrors ``marker_tracing.py``'s workflow while keeping the
implementation inside the package: ByteTrack + subpixel center localization,
automatic outer-circle calibration, cumulative displacement CSV output,
realtime overlay, displacement plot and optional segmented FFT.
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from .main import (camera_source_from_config, load_frames, load_realtime_frames,
                   is_openvino_model, open_camera, resolve_path,
                   select_model_weights)
from .pipeline import localize_one
from .tracker import ByteTrackTracker, MultiTargetTracker
from .features import measure_outer_diameter_px


def draw_realtime(frame, measured, fps=0.0, per_calib=None):
    out = frame.copy()
    for m in measured:
        x1, y1, x2, y2 = np.round(m["box"]).astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(out, f"id={m['track_id']}", (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
        if m.get("center") is not None:
            cv2.drawMarker(out, tuple(np.round(m["center"]).astype(int)),
                           (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
    cv2.putText(out, f"targets={len(measured)} FPS={fps:.1f}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
    return out


def draw_displacement_curve(hist, calib, plot_axis="dy"):
    canvas = np.full((300, 640, 3), 255, np.uint8)
    series = []
    for i, (tid, vals) in enumerate(sorted(hist.items())):
        if len(vals) < 2 or not calib.get(tid, {}).get("ready"):
            continue
        arr = np.asarray(vals, float)
        series.append((tid, arr[:, 0], arr[:, 1 if plot_axis == "dx" else 2],
                       ((0, 0, 220), (220, 80, 0), (0, 150, 0))[i % 3]))
    if not series:
        cv2.putText(canvas, "calibrating...", (220, 150), cv2.FONT_HERSHEY_SIMPLEX, .8, (80, 80, 80), 2)
        return canvas
    lo, hi = min(s[1][0] for s in series), max(s[1][-1] for s in series)
    ymin, ymax = min(float(s[2].min()) for s in series), max(float(s[2].max()) for s in series)
    if ymax <= ymin: ymax = ymin + 1.0
    for _tid, xs, ys, col in series:
        px = 45 + ((xs - lo) / max(hi - lo, 1e-6) * 580).astype(int)
        py = 280 - ((ys - ymin) / (ymax - ymin) * 250).astype(int)
        cv2.polylines(canvas, [np.column_stack([px, py])], False, col, 2)
    cv2.putText(canvas, f"displacement {plot_axis} (mm)", (50, 20), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 1)
    return canvas


def compute_segment_fft(samples, t0, t1, fmin=0.05, fmax_cap=16.0, snr_thresh=5.0):
    arr = np.asarray([(t, y) for t, y in samples if t0 <= t < t1], float)
    if len(arr) < 8:
        return None
    n = len(arr); dt = float(np.median(np.diff(arr[:, 0])))
    if not np.isfinite(dt) or dt <= 0: return None
    ys = arr[:, 1] - arr[:, 1].mean(); freqs = np.fft.rfftfreq(n, dt)
    amp = np.abs(np.fft.rfft(ys)) / n
    mask = (freqs >= fmin) & (freqs <= min(fmax_cap, 0.5 / dt))
    if not np.any(mask): return None
    noise = float(np.median(amp[mask])) + 1e-9; idx = np.where(mask)[0][np.argmax(amp[mask])]
    snr = float(amp[idx] / noise)
    return (freqs[mask], amp[mask], float(freqs[idx]), snr) if snr >= snr_thresh else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--camera", "--rtsp", dest="camera", default=None)
    ap.add_argument("--video", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--csv", default="displacement_mm.csv")
    ap.add_argument("--real-diameter-mm", type=float, default=100.0)
    ap.add_argument("--calib-frames", type=int, default=8)
    ap.add_argument("--calib-min", type=int, default=1)
    ap.add_argument("--no-calib", action="store_true")
    ap.add_argument("--mm-per-px", type=float, default=None)
    ap.add_argument("--radius-dir", default="radius")
    ap.add_argument("--fft-win", type=float, default=30.0)
    ap.add_argument("--fft-snr", type=float, default=5.0)
    ap.add_argument("--max-side", type=int, default=960)
    ap.add_argument("--display", dest="display", action="store_true")
    ap.add_argument("--no-display", dest="display", action="store_false")
    ap.set_defaults(display=None)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1].parent
    cfg_path = Path(args.config or (root / "marker_subpixel_tracker/config/config.yaml")).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    source_cfg, model_cfg = cfg["source"], cfg["model"]
    fallback = resolve_path(args.video or source_cfg["input"], cfg_path)
    camera = args.camera if args.camera is not None else camera_source_from_config(source_cfg)
    cap = open_camera(camera) if camera is not None else None
    realtime = cap is not None
    if camera is not None and not realtime:
        print(f"[realtime] camera unavailable, fallback to {fallback}")
    weights = select_model_weights(model_cfg, cfg_path, args.weights)
    if not weights.exists():
        raise FileNotFoundError(
            f"模型不存在: {weights}；请放置 OpenVINO 模型目录或 .pt 权重"
        )
    device = args.device or model_cfg.get("device", "cpu")
    print(f"[realtime] model={weights} ({'OpenVINO' if is_openvino_model(weights) else '.pt'}) device={device}")
    tracker = ByteTrackTracker(weights, model_cfg.get("confidence", .25),
                               model_cfg.get("class_id", 0), device,
                               tracker=cfg.get("tracking", {}).get("tracker", "bytetrack.yaml"),
                               persist=True)
    compute_fft = compute_segment_fft
    loc_cfg = cfg.get("localization", {})
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    radius_dir = Path(args.radius_dir)
    radius_dir.mkdir(parents=True, exist_ok=True)
    out_csv = csv_path.open("w", newline="", encoding="utf-8-sig")
    csvw = csv.writer(out_csv)
    csvw.writerow("frame,track_id,cx_px,cy_px,dx_px,dy_px,dx_mm,dy_mm,continuous,t_sec".split(","))
    multi = MultiTargetTracker()
    calib, hist = {}, {}
    frame_index = 0
    start = time.perf_counter()
    seg_len = max(1.0, args.fft_win)
    try:
        frames = load_realtime_frames(camera, cap) if realtime else load_frames(fallback)
        for key, frame, file_t in frames:
            now_t = time.perf_counter() - start if realtime else (0.0 if not np.isfinite(file_t) else file_t)
            tracks = tracker.update(frame)
            multi.prune(frame_index)
            measured = []
            for tr in tracks:
                tid, box = tr["id"], np.asarray(tr["box"], float)
                prev = multi.previous(tid)
                pc, pb = prev if prev is not None else (None, None)
                center, lm, continuous = localize_one(frame, box, loc_cfg, pc, pb)
                dx = dy = float("nan")
                if center is not None and multi.delta_ok(tid, continuous, frame_index):
                    dx, dy = map(float, center - pc)
                m = {"track_id": tid, "box": box, "center": center, "conf": tr["conf"],
                     "lambda_min": lm, "continuous": continuous, "dx": dx, "dy": dy}
                measured.append(m)
                if center is None:
                    continue
                multi.record(tid, center, box, frame_index)
                st = calib.setdefault(tid, {"tried": 0, "diams": [], "ready": bool(args.mm_per_px),
                                            "mm_per_px": args.mm_per_px, "ref": None})
                if not st["ready"] and not args.no_calib and st["tried"] < args.calib_frames:
                    st["tried"] += 1
                    res = measure_outer_diameter_px(frame, box)
                    if res is not None:
                        st["diams"].append(float(res[0] if isinstance(res, tuple) else res))
                    if len(st["diams"]) >= max(1, args.calib_min) or (st["tried"] >= args.calib_frames and st["diams"]):
                        st["mm_per_px"] = args.real_diameter_mm / float(np.mean(st["diams"]))
                        st["ready"] = True
                        print(f"[calib] id={tid} mm/px={st['mm_per_px']:.6f}")
                if st["ready"] and st["mm_per_px"]:
                    if st["ref"] is None:
                        st["ref"] = center.copy()
                    dxmm = float((center[0] - st["ref"][0]) * st["mm_per_px"])
                    dymm = float((center[1] - st["ref"][1]) * st["mm_per_px"])
                    hist.setdefault(tid, []).append((now_t, dxmm, dymm))
                    csvw.writerow([frame_index, tid, center[0], center[1], dx, dy, dxmm, dymm, int(continuous), now_t])
                    out_csv.flush()
            vis = draw_realtime(frame, measured, 0.0, calib)
            if args.max_side and max(vis.shape[:2]) > args.max_side:
                scale = args.max_side / max(vis.shape[:2]); vis = cv2.resize(vis, None, fx=scale, fy=scale)
            show = args.display if args.display is not None else (realtime or source_cfg.get("display", False))
            if show:
                cv2.imshow("marker_subpixel_tracker realtime", vis)
            if hist and show:
                cv2.imshow("displacement", draw_displacement_curve(hist, calib))
                # Compute completed non-overlapping FFT segments for each
                # calibrated track; results are printed for downstream use.
                for tid, samples in hist.items():
                    if len(samples) < 8:
                        continue
                    end_t = samples[-1][0]
                    seg_no = int(end_t // seg_len) - 1
                    if seg_no < 0:
                        continue
                    t0, t1 = seg_no * seg_len, (seg_no + 1) * seg_len
                    ys = [(t, dy) for t, _dx, dy in samples]
                    spec = compute_fft(ys, t0, t1, snr_thresh=args.fft_snr)
                    if spec is not None and not calib[tid].get("fft_reported", False):
                        calib[tid]["fft_reported"] = True
                        print(f"[FFT] id={tid} segment={t0:.0f}-{t1:.0f}s f1={spec[2]:.3f}Hz SNR={spec[3]:.1f}")
            if show and cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_index += 1
    finally:
        out_csv.close()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
    print(f"CSV: {csv_path}; Frames: {frame_index}")


if __name__ == "__main__":
    main()
