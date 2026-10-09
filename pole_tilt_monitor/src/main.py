import argparse
import os
from pathlib import Path
import cv2
import yaml
import numpy as np

from .geometry import load_intrinsics, undistort_frame

from .inference import InstanceSegmentationAdapter
from .tracker import ByteTrackTracker
from .monitor import PoleMonitor, ImageRelativePoleMonitor, create_csv, write_csv
from .vision import draw_measurement
from .diagnostics import save_pole_diagnostic


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def resolve_config_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    return cwd_path if cwd_path.exists() else PACKAGE_ROOT / path


def workspace_path(path):
    path = Path(path)
    return path if path.is_absolute() else WORKSPACE_ROOT / path


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_relative_baseline_images(video_source, monitor, output_dir, K, dist):
    """Save all completed pole baselines on one representative frame.

    Each pole may have a different nearest-to-median source frame. A common
    background frame is selected from the median of those frame ids, while
    every annotation uses that pole's final median baseline coordinates.
    """
    if isinstance(video_source, int):
        print("基准图片未保存：摄像头/直播源无法回读历史帧")
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    pole_ids = [pid for pid in sorted(monitor.baseline)
                if monitor.baseline_frame_id.get(pid) is not None]
    if not pole_ids:
        return saved
    representative_ids = [int(monitor.baseline_frame_id[pid]) for pid in pole_ids]
    median_frame_id = float(np.median(representative_ids))
    source_frame_id = min(representative_ids,
                          key=lambda value: abs(value - median_frame_id))

    baseline_cap = cv2.VideoCapture(str(video_source))
    if not baseline_cap.isOpened():
        print(f"基准图片未保存：无法重新打开视频 {video_source}")
        return saved

    try:
        baseline_cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame_id)
        ok, image = baseline_cap.read()
        if not ok:
            print(f"基准图片未保存：无法读取公共代表帧 {source_frame_id}")
            return saved
        image = undistort_frame(image, K, dist)
        palette = [(255, 255, 0), (255, 0, 255), (0, 255, 255),
                   (255, 128, 0), (128, 255, 0), (0, 128, 255)]
        title = (f"All pole baselines  background frame {source_frame_id}  "
                 f"N={monitor.baseline_frames}")
        cv2.putText(image, title, (30, 42), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, title, (30, 42), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)

        for index, pole_id in enumerate(pole_ids):
            baseline = monitor.baseline[pole_id]
            top = tuple(np.round(baseline[:2]).astype(int))
            bottom = tuple(np.round(baseline[2:4]).astype(int))
            angle = float(baseline[4])
            quality = monitor.baseline_quality.get(pole_id, float("nan"))
            pole_frame_id = monitor.baseline_frame_id[pole_id]
            color = palette[index % len(palette)]
            cv2.line(image, bottom, top, color, 3)
            cv2.circle(image, top, 7, (0, 255, 0), -1)
            cv2.circle(image, bottom, 7, (0, 0, 255), -1)
            label = (f"Pole {pole_id}  angle={angle:+.4f} deg  "
                     f"Q={quality:.2f}  rep={pole_frame_id}")
            label_at = (max(5, top[0] - 35), max(70, top[1] - 12))
            cv2.putText(image, label, label_at, cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(image, label, label_at, cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, color, 2, cv2.LINE_AA)

        path = output_dir / f"baseline_all_poles_frame_{source_frame_id}.jpg"
        if cv2.imwrite(str(path), image):
            saved.append(path)
    finally:
        baseline_cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=str(PACKAGE_ROOT / "config" / "config.yaml"),
        help="配置文件路径（默认使用项目内 config/config.yaml）"
    )
    args = parser.parse_args()
    cfg = load_config(resolve_config_path(args.config))

    vc = cfg["video"]
    mc = cfg["models"]
    geom = cfg["measurement"]
    bc = cfg["baseline"]
    fc = cfg["filter"]
    debug_cfg = cfg.get("debug", {})
    relative_mode = geom.get("mode", "world_3d") == "image_relative"
    calibration_path = PACKAGE_ROOT / "calibration" / "camera.yaml"
    relative_K = relative_dist = None
    if relative_mode:
        relative_K, relative_dist = load_intrinsics(calibration_path)

    # One-pass instance-segmentation model: its track() result supplies
    # boxes, ByteTrack IDs and masks together.
    instance_model = (mc.get("instance_segmentation") or
                      mc.get("segmenter") or mc.get("detector"))
    if not instance_model:
        raise KeyError("models.instance_segmentation must point to a YOLO-seg checkpoint")
    instance_model = workspace_path(instance_model)
    detector = InstanceSegmentationAdapter(
        instance_model, mc.get("device","cpu"),
        mc.get("confidence",0.35), mc.get("pole_class_id",0)
    )

    if geom.get("mode", "world_3d") == "image_relative":
        monitor = ImageRelativePoleMonitor(
            calibration_path,
            geom.get("endpoint_ratio", 0.10),
            geom.get("ransac_residual_px", 2.5),
            geom.get("ransac_iterations", 300),
            bc.get("relative_frames", 1), 
            fc.get("alpha", 0.25),
            geom.get("quality_threshold", 0.60),
            geom.get("quality_weights"),
            geom.get("max_centerline_residual_px", 4.0),
            geom.get("max_endpoint_spread_px", 4.0),
            geom.get("body_trim_ratio", 0.10)
        )
    else:
        monitor = PoleMonitor(
            calibration_path, geom["pole_length_m"],
            geom.get("ground_z_m", 0.0), geom.get("endpoint_ratio", 0.10),
            geom.get("ransac_residual_px", 2.5),
            geom.get("ransac_iterations", 300), bc.get("frames", 60),
            fc.get("alpha", 0.25), geom.get("quality_threshold", 0.60),
            geom.get("quality_weights"), geom.get("min_ray_plane_angle_deg", 5.0),
            geom.get("max_centerline_residual_px", 4.0),
            geom.get("max_endpoint_spread_px", 4.0),
            geom.get("max_length_error_ratio", 0.03),
            geom.get("body_trim_ratio", 0.10)
        )

    tracker = ByteTrackTracker(
        detector.model, mc.get("device", "cpu"),
        mc.get("confidence", 0.35), mc.get("pole_class_id", 0),
        mc.get("tracker", "bytetrack.yaml"), mc.get("persist", True)
    )

    video_source = vc["source"]
    if isinstance(video_source, str) and not video_source.isdigit():
        video_source = str(workspace_path(video_source))
    elif isinstance(video_source, str) and video_source.isdigit():
        video_source = int(video_source)
    output_video = workspace_path(vc["output_video"])
    output_csv = workspace_path(vc["output_csv"])

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    requested_sample_fps = float(vc.get("sample_fps", fps))
    if requested_sample_fps <= 0:
        raise ValueError("video.sample_fps 必须大于 0")
    sample_fps = min(requested_sample_fps, fps)
    sample_period_s = 1.0 / sample_fps

    out_writer = None
    if vc.get("save_video",True):
        os.makedirs(output_video.parent, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(
            str(output_video), fourcc, sample_fps, (W, H)
        )

    csv_file,csv_writer = create_csv(output_csv)

    frame_id = 0
    next_sample_time_s = 0.0
    processed_frames = 0
    diagnostic_saved = False
    try:
        while True:
            # Advance every source frame, but materialize image data only for
            # frames selected by the sampling schedule below.
            ok = cap.grab()
            if not ok:
                break

            source_time_s = frame_id / fps if fps > 0 else 0.0
            if source_time_s + 1e-9 < next_sample_time_s:
                frame_id += 1
                continue
            next_sample_time_s += sample_period_s
            processed_frames += 1

            ok, frame = cap.retrieve()
            if not ok:
                break

            # Keep the entire inference chain in one rectified image space:
            # detection, ByteTrack, segmentation and line fitting all see the
            # same undistorted pixels.
            if relative_mode:
                frame = undistort_frame(frame, relative_K, relative_dist)

            # Detection and identity association are performed in one
            # Ultralytics call using the maintained ByteTrack implementation.
            dets = tracker.update(frame)

            if (debug_cfg.get("enabled", False)
                    and not diagnostic_saved
                    and frame_id == int(debug_cfg.get("frame_id", 0))
                    and dets):
                target = min(dets, key=lambda d: float(d["box"][0]))
                diag_dir = workspace_path(
                    debug_cfg.get("output_dir", "pole_tilt_monitor/output/diagnostics")
                )
                metrics = save_pole_diagnostic(
                    frame, target["mask"], target["box"], diag_dir,
                    pole_id=target.get("id"), frame_id=frame_id,
                    endpoint_ratio=geom.get("endpoint_ratio", 0.10),
                    ransac_residual_px=geom.get("ransac_residual_px", 2.5),
                    ransac_iterations=geom.get("ransac_iterations", 300),
                    body_trim_ratio=geom.get("body_trim_ratio", 0.10)
                )
                print("Diagnostic:", diag_dir)
                print("Diagnostic metrics:", metrics)
                diagnostic_saved = True

            for det in dets:
                pid = det["id"]
                result = monitor.process(pid, det["mask"], frame_id=frame_id)

                if result is None:
                    continue

                draw_measurement(frame,pid,det["box"],result)
                write_csv(csv_writer,frame_id,fps,pid,result)

                if result.get("status") == "INVALID":
                    x1,y1,_,_ = map(int,det["box"])
                    cv2.putText(frame, "INVALID", (x1, max(40, y1-28)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)
                    continue

                alarm = (
                    abs(result["delta_lr"]) >= cfg["alarm"]["left_right_deg"]
                    or abs(result["delta_fb"]) >= cfg["alarm"]["front_back_deg"]
                    or result["delta_total"] >= cfg["alarm"]["total_deg"]
                )

                if alarm:
                    x1,y1,_,_ = map(int,det["box"])
                    cv2.putText(
                        frame,"ALARM",(x1,max(40,y1-28)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2
                    )

            cv2.putText(
                frame,
                f"Frame {frame_id}  Baseline poles: {len(monitor.baseline)}",
                (20,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2
            )

            if out_writer is not None:
                out_writer.write(frame)

            if vc.get("display",True):
                cv2.imshow("Pole Tilt Monitor",frame)
                key = cv2.waitKey(1)&0xff
                if key in (27,ord("q")):
                    break

            frame_id += 1

    finally:
        cap.release()
        if out_writer is not None:
            out_writer.release()
        csv_file.close()
        cv2.destroyAllWindows()

    if relative_mode:
        saved = save_relative_baseline_images(
            video_source, monitor, output_video.parent, relative_K, relative_dist
        )
        print("Baseline images:", len(saved))

    print("处理完成")
    print(f"抽帧处理: {processed_frames} 帧, {sample_fps:g} FPS")
    print("CSV:", output_csv)
    if out_writer is not None:
        print("Video:", output_video)


if __name__ == "__main__":
    main()
