"""Camera-first realtime marker tracing with video fallback.

This entry point mirrors ``marker_tracing.py``'s workflow while keeping the
implementation inside the package: ByteTrack + subpixel center localization,
automatic outer-circle calibration, cumulative displacement CSV output,
realtime overlay, displacement plot and optional segmented FFT.
"""
import argparse
import csv
import importlib.util
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

# Keep the package entry point visually identical to the proven standalone
# marker_tracing workflow.  The standalone module only defines helpers at
# import time; its main loop remains guarded and is never started here.
def _load_visualization_reference():
    """Load the standalone visualizer without requiring cwd on sys.path."""
    try:
        import marker_tracing as module
        return module
    except ModuleNotFoundError:
        path = Path(__file__).resolve().parents[2] / "marker_tracing.py"
        spec = importlib.util.spec_from_file_location("_marker_tracing_reference", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载可视化参考模块: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


_visualization_reference = _load_visualization_reference()
_compute_segment_fft_reference = _visualization_reference.compute_segment_fft
_draw_displacement_curve_reference = _visualization_reference.draw_displacement_curve
_draw_realtime_reference = _visualization_reference.draw_realtime
_draw_spectrum_panel_reference = _visualization_reference.draw_spectrum_panel

draw_realtime = _draw_realtime_reference
draw_displacement_curve = _draw_displacement_curve_reference
draw_spectrum_panel = _draw_spectrum_panel_reference
compute_segment_fft = _compute_segment_fft_reference


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
    ap.add_argument("--plot-axis", choices=("dy", "dx", "both"), default="dy")
    ap.add_argument("--seg-max", type=int, default=6)
    ap.add_argument("--fmax", type=float, default=0.0)
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
    seg_spec = {}
    frame_index = 0
    start = time.perf_counter()
    seg_len = max(1.0, args.fft_win)
    seg_max = max(1, args.seg_max)
    seg_idx = 0
    last_tick = time.perf_counter()
    fps_ema = 0.0
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
                     "lambda_min": lm, "continuous": continuous, "dx": dx, "dy": dy,
                     "cum_dx_mm": float("nan"), "cum_dy_mm": float("nan")}
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
                    m["cum_dx_mm"], m["cum_dy_mm"] = dxmm, dymm
                    csvw.writerow([frame_index, tid, center[0], center[1], dx, dy, dxmm, dymm, int(continuous), now_t])
                    out_csv.flush()

            tick = time.perf_counter()
            dt = tick - last_tick
            last_tick = tick
            if dt > 0:
                inst_fps = 1.0 / dt
                fps_ema = inst_fps if fps_ema == 0.0 else 0.9 * fps_ema + 0.1 * inst_fps
            vis = draw_realtime(frame, measured, fps_ema, calib)
            if args.max_side and max(vis.shape[:2]) > args.max_side:
                scale = args.max_side / max(vis.shape[:2]); vis = cv2.resize(vis, None, fx=scale, fy=scale)
            show = args.display if args.display is not None else (realtime or source_cfg.get("display", False))
            if show:
                cv2.imshow("marker_subpixel_tracker realtime", vis)
            if hist and show:
                all_t = [p[0] for values in hist.values() for p in values]
                t_lo = float(min(all_t)) if all_t else 0.0
                t_hi = float(max(now_t, max(all_t))) if all_t else float(now_t)
                disp = draw_displacement_curve(
                    hist, calib, args.plot_axis, t_lo=t_lo, t_hi=t_hi
                )

                # Match marker_tracing: fixed, non-overlapping full segments.
                ready_tids = sorted(tid for tid, st in calib.items()
                                    if st.get("ready"))
                while now_t >= (seg_idx + 1) * seg_len:
                    t0, t1 = seg_idx * seg_len, (seg_idx + 1) * seg_len
                    components = []
                    if args.plot_axis in ("dy", "both"):
                        components.append(("dy", 1))
                    if args.plot_axis in ("dx", "both"):
                        components.append(("dx", 0))
                    for tid in ready_tids:
                        samples = hist.get(tid, [])
                        for component, column in components:
                            ys = [(row[0], row[column + 1]) for row in samples]
                            result = compute_segment_fft(
                                ys, t0, t1, snr_thresh=args.fft_snr
                            )
                            if result is None:
                                continue
                            freqs, amp, f1, snr = result
                            key = (tid, component)
                            seg_spec.setdefault(key, []).append({
                                "k": seg_idx, "t0": t0, "t1": t1,
                                "freqs": freqs, "amp": amp,
                                "f1": f1,
                            })
                            seg_spec[key] = seg_spec[key][-seg_max:]
                            print(f"[FFT] id={tid} segment={t0:.0f}-{t1:.0f}s "
                                  f"f1={f1:.3f}Hz SNR={snr:.1f}")
                    seg_idx += 1

                spectrum = draw_spectrum_panel(
                    seg_spec, args.plot_axis, seg_len,
                    fmin=0.05, fmax=args.fmax
                )
                cv2.imshow("位移时程 + 频谱图", np.vstack([disp, spectrum]))
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
