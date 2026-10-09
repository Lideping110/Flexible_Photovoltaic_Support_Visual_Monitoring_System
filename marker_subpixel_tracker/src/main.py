import argparse
import csv
import os
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from .pipeline import localize_one
from .tracker import ByteTrackTracker, MultiTargetTracker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent


def resolve_path(value, config_path):
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [PROJECT_ROOT / path, WORKSPACE_ROOT / path,
                  config_path.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # New output paths are project-local by default.
    return PROJECT_ROOT / path


def is_openvino_model(path):
    """Return True for an Ultralytics OpenVINO IR model directory."""
    path = Path(path)
    return path.is_dir() and any(path.glob("*.xml"))


def select_model_weights(model_cfg, config_path, explicit=None):
    """Select explicit weights, otherwise prefer bundled OpenVINO over .pt."""
    if explicit:
        return resolve_path(explicit, config_path)

    roots = (PROJECT_ROOT, WORKSPACE_ROOT, config_path.parent,
             config_path.parent.parent)
    candidates = []
    for key in ("openvino_weights", "openvino_model", "openvino_dir"):
        value = model_cfg.get(key)
        if value:
            candidates.append(resolve_path(value, config_path))
    configured = model_cfg.get("weights")
    model_stem = Path(configured).stem if configured else "marker26_det"
    names = [f"{model_stem}_openvino_model"]
    names.extend(name for name in (
        "marker26_det_openvino_model", "marker26s_det_openvino_model"
    ) if name not in names)
    for name in names:
        candidates.extend(root / name for root in roots)
    for candidate in candidates:
        if is_openvino_model(candidate):
            return candidate

    fallback = resolve_path(configured, config_path) if configured else PROJECT_ROOT / "marker26_det.pt"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(
        "未找到可用的 OpenVINO 模型目录或 .pt 权重文件。"
        f" OpenVINO 候选: {', '.join(str(p) for p in candidates)}; .pt 回退: {fallback}"
    )


def probe_source_fps(source):
    """Return the source video's FPS; 10.0 default for image sequences.

    The annotated output video must play at the source's real speed; a
    hardcoded writer FPS makes 25fps sources play 2.5x too fast.
    """
    source = Path(source)
    if any(ch in str(source) for ch in "*?[]"):
        return 10.0
    if source.is_dir():
        return 10.0
    if source.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        return 10.0
    cap = cv2.VideoCapture(str(source))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()
    return float(fps) if fps and fps > 0 else 10.0


def load_frames(source):
    source = Path(source)
    if not any(ch in str(source) for ch in "*?[]") and source.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        image = cv2.imread(str(source))
        if image is None:
            raise RuntimeError(f"无法读取图片: {source}")
        yield str(source), image, 0.0
        return
    if source.is_dir():
        paths = sorted(p for p in source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        for path in paths:
            image = cv2.imread(str(path))
            if image is not None:
                yield str(path), image, float("nan")
        return
    if any(ch in str(source) for ch in "*?[]"):
        paths = sorted(source.parent.glob(source.name))
        for path in paths:
            image = cv2.imread(str(path))
            if image is not None:
                yield str(path), image, float("nan")
        return
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开输入源: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_id = 0
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            yield frame_id, image, frame_id / fps if fps > 0 else float("nan")
            frame_id += 1
    finally:
        cap.release()


def open_camera(source):
    """Open a camera/RTSP source with low-latency capture settings."""
    value = source
    # Numeric strings address local webcams (0, 1, ...); other values are URLs.
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if isinstance(value, str) and value.lower().startswith(("rtsp://", "rtsps://")):
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            "rtsp_transport;tcp|fflags;nobuffer|probesize;32|analyzeduration;0",
        )
        cap = cv2.VideoCapture(value, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(value)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def camera_source_from_config(source_cfg):
    """Return the first configured camera URL, or None when absent."""
    for key in ("camera_url", "camera", "rtsp", "stream"):
        value = source_cfg.get(key)
        if value not in (None, "", False):
            return value
    return None


def load_realtime_frames(source, cap=None):
    """Yield frames from a live camera/RTSP source with reconnect support."""
    cap = cap or open_camera(source)
    if cap is None:
        raise RuntimeError(f"无法打开摄像头/RTSP源: {source}")
    frame_id = 0
    fail_count = 0
    started = time.perf_counter()
    try:
        while True:
            ok, image = cap.read()
            if not ok or image is None:
                fail_count += 1
                if fail_count < 30:
                    continue
                cap.release()
                time.sleep(1.0)
                cap = open_camera(source)
                fail_count = 0
                if cap is None:
                    raise RuntimeError(f"摄像头/RTSP重连失败: {source}")
                continue
            fail_count = 0
            yield frame_id, image, time.perf_counter() - started
            frame_id += 1
    finally:
        cap.release()


def draw_results(image, measured):
    """Draw every target's box, id and center onto a copy of the frame."""
    out = image.copy()
    for m in measured:
        x1, y1, x2, y2 = np.round(m["box"]).astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
        id_label = f"id={m['track_id']}"
        cv2.putText(out, id_label, (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4,
                    cv2.LINE_AA)
        cv2.putText(out, id_label, (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                    cv2.LINE_AA)
        if m["center"] is not None:
            p = tuple(np.round(m["center"]).astype(int))
            cv2.drawMarker(out, p, (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
    label = f"targets={len(measured)}"
    cv2.putText(out, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def should_compute_delta(center, previous_state, track_id, continuous,
                         last_measured_frame, frame_index):
    """dx/dy is valid only between CONSECUTIVE measurements of one track.

    All must hold:
    - this frame actually measured a center;
    - the previous measurement belongs to the SAME locked track_id;
    - that previous measurement was the IMMEDIATELY preceding frame — the
      first measurement after a lost gap must not emit a cross-gap
      displacement as if it were frame-to-frame (observed as a bogus 135px
      spike during early testing);
    - the peak came from the anchor window (``continuous``).
    """
    return (center is not None and previous_state is not None
            and previous_state[0] == track_id
            and last_measured_frame == frame_index - 1
            and continuous)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "config.yaml"))
    parser.add_argument("--camera", "--rtsp", dest="camera", default=None,
                        help="摄像头索引或 RTSP 地址；不可用时回退视频")
    parser.add_argument("--video", default=None, help="覆盖配置中的回退视频")
    parser.add_argument("--display", action="store_true", help="显示实时窗口，按 q 退出")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    source_cfg, model_cfg = cfg["source"], cfg["model"]
    localization_cfg = cfg.get("localization", {})
    tracking_cfg = cfg.get("tracking", {})
    source = resolve_path(args.video or source_cfg["input"], config_path)
    camera = args.camera if args.camera is not None else camera_source_from_config(source_cfg)
    weights = select_model_weights(model_cfg, config_path)
    print(f"[marker_subpixel_tracker] 使用推理模型: {weights}")
    output_csv = resolve_path(source_cfg["output_csv"], config_path)
    output_dir = resolve_path(source_cfg["output_dir"], config_path)
    output_video = resolve_path(source_cfg["output_video"], config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    tracker = ByteTrackTracker(weights, model_cfg["confidence"], model_cfg["class_id"],
                               model_cfg["device"], tracker=tracking_cfg.get("tracker", "bytetrack.yaml"),
                               persist=tracking_cfg.get("persist", True))
    multi = MultiTargetTracker()
    csv_file = output_csv.open("w", newline="", encoding="utf-8-sig")
    writer = csv.writer(csv_file)
    writer.writerow(["frame", "time_s", "detected", "method", "det_conf", "track_id",
                     "lambda_min", "continuous", "x_px", "y_px", "dx_px", "dy_px"])

    camera_cap = open_camera(camera) if camera is not None else None
    realtime = camera_cap is not None
    if camera is not None and not realtime:
        print(f"[marker_subpixel_tracker] 摄像头不可用，回退视频文件: {source}")
    elif realtime:
        print(f"[marker_subpixel_tracker] 使用摄像头实时流: {camera}")
    writer_fps = (float(camera_cap.get(cv2.CAP_PROP_FPS) or 25.0) if realtime
                  else probe_source_fps(source))
    display = bool(args.display or source_cfg.get("display", False))
    writer_video = None
    frame_index = 0

    def process(frame_key, frame, time_s):
        nonlocal writer_video
        tracks = tracker.update(frame)
        multi.prune(frame_index)
        measured = []
        for track in tracks:
            tid, box = track["id"], np.asarray(track["box"], dtype=float)
            prev = multi.previous(tid)
            prev_center, prev_box = prev if prev is not None else (None, None)
            center, lm, continuous = localize_one(frame, box, localization_cfg, prev_center, prev_box)
            dx = dy = float("nan")
            if center is not None and multi.delta_ok(tid, continuous, frame_index):
                dx, dy = map(float, center - prev_center)
            measured.append({"track_id": tid, "box": box, "center": center, "conf": track["conf"],
                             "lambda_min": lm, "continuous": continuous, "dx": dx, "dy": dy})
            if center is not None:
                multi.record(tid, center, box, frame_index)
        annotated = draw_results(frame, measured)
        if writer_video is None and source_cfg.get("save_video", True):
            output_video.parent.mkdir(parents=True, exist_ok=True)
            writer_video = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                           writer_fps, (frame.shape[1], frame.shape[0]))
        if writer_video is not None:
            writer_video.write(annotated)
        if source_cfg.get("save_debug", cfg.get("runtime", {}).get("save_debug", True)):
            cv2.imwrite(str(output_dir / f"frame_{frame_index:06d}.jpg"), annotated)
        rows = measured or [{"center": None, "conf": float("nan"), "track_id": "", "lambda_min": float("nan"),
                             "continuous": False, "dx": float("nan"), "dy": float("nan")}]
        for m in rows:
            writer.writerow([frame_key, time_s, m["center"] is not None,
                             "bytetrack+paper_center" if measured else "lost", m["conf"], m["track_id"],
                             m["lambda_min"], m["continuous"],
                             *(m["center"].tolist() if m["center"] is not None else [float("nan"), float("nan")]),
                             m["dx"], m["dy"]])
        return annotated

    try:
        if realtime:
            frames = load_realtime_frames(camera, camera_cap)
        else:
            frames = load_frames(source)
        for frame_key, frame, time_s in frames:
            annotated = process(frame_key, frame, time_s)
            if display:
                cv2.imshow("marker_subpixel_tracker", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_index += 1
    finally:
        csv_file.close()
        if writer_video is not None:
            writer_video.release()
        if display:
            cv2.destroyAllWindows()
    print(f"CSV: {output_csv}")
    print(f"Frames: {frame_index}")
    if writer_video is not None:
        print(f"Video: {output_video}")


if __name__ == "__main__":
    main()
