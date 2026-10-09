"""
视频去畸变 (Undistort Video)
从标定文件读取相机内参 K 和畸变系数 dist, 对视频逐帧去畸变并保存。

用法:
    uv run python undistort_video.py                                        # 默认参数
    uv run python undistort_video.py --input xxx.mp4 --output yyy.mp4       # 自定义输入输出
    uv run python undistort_video.py --alpha 0.0                            # 裁剪黑边
    uv run python undistort_video.py --keep-k                               # new_K=K, 与 pole_tilt_monitor 流程一致
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="视频去畸变")
    parser.add_argument("--input", default=r"D:\摄像头拉流\move_babiao.mp4", help="输入视频路径")
    parser.add_argument("--output", default=None, help="输出视频路径 (默认与输入同目录, _undistort 后缀)")
    parser.add_argument("--calib", default=r"D:\摄像头拉流\pole_tilt_monitor\calibration\camera.yaml", help="标定文件路径")
    parser.add_argument("--alpha", type=float, default=0.0, help="getOptimalNewCameraMatrix 的 alpha, 1.0保留全部像素, 0.0裁剪黑边")
    parser.add_argument("--keep-k", action="store_true", help="直接使用原内参 K 作为 new_K (与 pole_tilt_monitor 的 undistort_frame 完全一致)")
    return parser.parse_args()


def main():
    args = parse_args()

    # 读取标定参数
    with open(args.calib, "r", encoding="utf-8") as f:
        calib = yaml.safe_load(f)
    K = np.array(calib["K"], dtype=np.float64)
    dist = np.array(calib["dist"], dtype=np.float64)
    print(f"标定文件: {args.calib}")
    print(f"K =\n{K}")
    print(f"dist = {dist.ravel()}")

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"输入: {args.input} ({w}x{h} @ {fps:.2f}fps, 共 {total} 帧)")

    # --keep-k: 严格复刻 pole_tilt_monitor 的去畸变方式
    # (geometry.py: undistort_frame -> cv2.undistort(frame, K, dist), new_K=K, 每帧调用)
    # 否则: getOptimalNewCameraMatrix + 预生成映射表逐帧 remap (更快)
    if args.keep_k:
        print("模式: --keep-k, 每帧调用 cv2.undistort(frame, K, dist), 与 monitor 严格一致")
    else:
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), args.alpha, (w, h))
        print(f"new_K =\n{new_K}")
        map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, new_K, (w, h), cv2.CV_16SC2)

    # 输出路径: 默认与输入同目录, 文件名加 _undistort 后缀
    output = args.output or str(Path(args.input).with_name(Path(args.input).stem + "_undistort.mp4"))
    writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise SystemExit(f"无法创建输出视频: {output}")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if args.keep_k:
            undistorted = cv2.undistort(frame, K, dist)
        else:
            # remap 比逐帧 cv2.undistort 更快
            undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        writer.write(undistorted)
        if frame_idx % 100 == 0:
            print(f"已处理 {frame_idx}/{total} 帧")

    cap.release()
    writer.release()
    print(f"处理完成, 共 {frame_idx} 帧, 已保存: {output}")


if __name__ == "__main__":
    main()
