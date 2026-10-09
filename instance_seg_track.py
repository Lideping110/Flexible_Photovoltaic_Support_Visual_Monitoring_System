"""
实例分割与跟踪 (Instance Segmentation and Tracking)
参考: https://docs.ultralytics.com/zh/guides/instance-segmentation-and-tracking

基于 YOLO11 分割模型, 对视频/RTSP流/摄像头进行实例分割 + 多目标跟踪,
绘制掩膜、检测框和跟踪 ID, 并统计每个跟踪 ID 的类别。

用法:
    uv run python instance_seg_track.py                          # 默认摄像头 0
    uv run python instance_seg_track.py --source rtsp://xxx      # RTSP 流
    uv run python instance_seg_track.py --source video.mp4 -o out.mp4  # 视频文件并保存
    uv run python instance_seg_track.py --model yolo11s-seg.pt --tracker bytetrack.yaml
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO 实例分割与跟踪")
    parser.add_argument("--source", default="D:/摄像头拉流/1080/capture_20260908_151742.mp4", help="视频路径 / RTSP地址 / 摄像头序号(默认0)")
    parser.add_argument("--model", default="exp-7.pt", help="分割模型权重")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="跟踪器: botsort.yaml / bytetrack.yaml")
    parser.add_argument("--conf", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU 阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="推理输入尺寸")
    parser.add_argument("--classes", nargs="+", type=int, default=None, help="只跟踪指定类别ID, 如 0 2")
    parser.add_argument("--output", "-o", default='out_1080.mp4', help="保存结果视频路径")
    parser.add_argument("--csv", default=None, help="保存跟踪记录 CSV 路径")
    parser.add_argument("--show", action="store_true", help="实时显示窗口")
    parser.add_argument("--line-width", type=int, default=2, help="绘制线宽")
    return parser.parse_args()


def resolve_source(source: str):
    """摄像头序号转 int, 其余原样返回 (路径/RTSP URL)。"""
    try:
        return int(source)
    except ValueError:
        return source


def main():
    args = parse_args()

    # 加载实例分割模型 (首次运行会自动下载权重)
    model = YOLO(args.model)
    names = model.names  # {class_id: class_name}

    source = resolve_source(args.source)

    # 历史记录: track_id -> [(frame_idx, class_name), ...]
    track_history = defaultdict(list)

    # 可选: 视频写出
    writer = None
    # 可选: CSV 写出
    csv_file = csv.writer(open(args.csv, "w", newline="", encoding="utf-8")) if args.csv else None
    if csv_file:
        csv_file.writerow(["frame", "track_id", "class", "conf"])

    frame_idx = 0
    # stream=True 生成器, 逐帧处理, track 开启 persist 保持跨帧 ID 连续
    results = model.track(
        source=source,
        stream=True,
        persist=True,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        classes=args.classes,
        tracker=args.tracker,
        verbose=False,
    )

    for result in results:
        frame_idx += 1
        annotated = result.plot(line_width=args.line_width)  # 绘制掩膜+框+标签+ID

        if result.boxes is not None and result.boxes.is_track:
            ids = result.boxes.id.int().tolist()
            classes = result.boxes.cls.int().tolist()
            confs = result.boxes.conf.tolist()

            for tid, cid, conf in zip(ids, classes, confs):
                cname = names[cid]
                track_history[tid].append((frame_idx, cname))
                if csv_file:
                    csv_file.writerow([frame_idx, tid, cname, f"{conf:.3f}"])

            # 左上角汇总: 当前帧出现的唯一目标
            summary = " | ".join(f"{names[c]}#{t}" for t, c in zip(ids, classes))
            cv2.putText(annotated, f"frame {frame_idx}: {summary}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if args.output:
            if writer is None:
                h, w = annotated.shape[:2]
                writer = cv2.VideoWriter(
                    args.output, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h)
                )
            writer.write(annotated)

        if args.show:
            cv2.imshow("Instance Segmentation & Tracking", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if writer:
        writer.release()
    if csv_file:
        print(f"跟踪记录已保存: {args.csv}")

    # 打印统计: 每个 track_id 出现的帧数和类别
    print(f"\n处理完成, 共 {frame_idx} 帧, {len(track_history)} 个跟踪目标:")
    for tid, records in sorted(track_history.items()):
        cnames = {c for _, c in records}
        print(f"  ID {tid}: 类别={','.join(cnames)}, 出现 {len(records)} 帧")

    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
