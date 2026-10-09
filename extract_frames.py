# -*- coding: utf-8 -*-
"""
提取 8mp 和 1080 文件夹中视频的帧
每秒提取 3 帧(间隔 1/3 秒), 保存到 images/<来源文件夹>/ 下
用法:
    python extract_frames.py
"""
import os
import sys

import cv2
import numpy as np


def imwrite_unicode(path: str, frame) -> bool:
    """cv2.imwrite 在 Windows 上不支持含中文的路径, 改用编码后写文件"""
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        return False
    buf.tofile(path)
    return True

SRC_DIRS = ["9m"]   # 待处理的视频文件夹
OUT_DIR = "images"           # 输出根目录
FPS_OUT = 25                  # 每秒提取帧数


def extract_from_video(video_path: str, out_subdir: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[错误] 无法打开: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = fps / FPS_OUT          # 每隔 interval 帧取一帧
    name = os.path.splitext(os.path.basename(video_path))[0]

    saved = 0
    next_frame = 0.0
    while True:
        frame_idx = int(round(next_frame))
        if frame_idx >= total and total > 0:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        # 时间戳: 秒.毫秒, 用于命名
        # ts = frame_idx / fps
        out_name = f"{name}_{frame_idx}.jpg"
        if not imwrite_unicode(os.path.join(out_subdir, out_name), frame):
            print(f"[警告] 写图失败: {out_name}")
            continue
        saved += 1
        next_frame += interval

    cap.release()
    return saved


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    total_saved = 0
    for src in SRC_DIRS:
        src_path = os.path.join(root, src)
        if not os.path.isdir(src_path):
            print(f"[跳过] 文件夹不存在: {src}")
            continue
        out_subdir = os.path.join(root, OUT_DIR, src)
        os.makedirs(out_subdir, exist_ok=True)

        videos = sorted(f for f in os.listdir(src_path)
                        if f.lower().endswith((".mp4", ".avi", ".mkv", ".mov")))
        print(f"[处理] {src}: {len(videos)} 个视频")
        for v in videos:
            n = extract_from_video(os.path.join(src_path, v), out_subdir)
            print(f"  {v}: 提取 {n} 帧")
            total_saved += n

    print(f"[完成] 共保存 {total_saved} 张图片 -> {OUT_DIR}/")


if __name__ == "__main__":
    sys.exit(main())
