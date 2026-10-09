# -*- coding: utf-8 -*-
"""
RTSP 拉流并保存本地视频
用法:
    python rtsp_capture.py                    # 默认持续录制, Ctrl+C 停止
    python rtsp_capture.py --duration 60      # 录制 60 秒后自动停止
    python rtsp_capture.py --out myvideo.mp4  # 指定输出文件
"""
import argparse
import os
import sys
import time

import cv2

# RTSP_URL = "rtsp://admin:hhjt220220@192.168.1.65:554/streaming/Channels/101"
RTSP_URL = "rtsp://admin:hhjt110110@192.168.1.64:554/streaming/Channels/101"
FOURCC = "mp4v"          # 编码器; 备选 "avc1"(H264) / "XVID"
TCP_TRANSPORT = True     # True 用 TCP 取流, 丢包更少更稳; False 用 UDP(延迟更低)
CONNECT_TIMEOUT = 10     # 连接超时(秒)
RECONNECT_DELAY = 3      # 断线重连间隔(秒)
MAX_RETRIES = 5          # 最大重连次数, -1 表示无限重试


def open_stream(url: str):
    """打开 RTSP 流, 失败返回 None"""
    if TCP_TRANSPORT:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        return None
    # 减小内部缓冲, 降低延迟
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    return cap


def main():
    parser = argparse.ArgumentParser(description="RTSP 拉流保存本地")
    parser.add_argument("--url", default=RTSP_URL, help="取流地址")
    parser.add_argument("--out", default=None, help="输出文件路径(默认按时间命名)")
    parser.add_argument("--duration", type=float, default=0, help="录制时长(秒), 0=不限")
    parser.add_argument("--max-frames", type=int, default=0, help="最大帧数, 0=不限")
    args = parser.parse_args()

    out_path = args.out or time.strftime("capture_%Y%m%d_%H%M%S.mp4")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    # ---- 连接 ----
    cap = None
    for attempt in range(1, MAX_RETRIES + 1 if MAX_RETRIES > 0 else 10**9):
        print(f"[连接] 第 {attempt} 次尝试: {args.url}")
        cap = open_stream(args.url)
        if cap:
            break
        print(f"[连接] 失败, {RECONNECT_DELAY}s 后重试...")
        time.sleep(RECONNECT_DELAY)
    if not cap:
        print("[错误] 无法连接视频流, 退出")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"[成功] 分辨率 {width}x{height}, FPS {fps:.1f}")

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*FOURCC), fps, (width, height))
    if not writer.isOpened():
        print("[错误] VideoWriter 初始化失败, 退出")
        cap.release()
        sys.exit(1)

    # ---- 录制循环 ----
    frame_count = 0
    start = time.time()
    consecutive_fail = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                consecutive_fail += 1
                print(f"[警告] 读帧失败 x{consecutive_fail}, 尝试重连...")
                cap.release()
                cap = open_stream(args.url)
                if cap:
                    consecutive_fail = 0
                    continue
                if consecutive_fail >= 3:
                    print("[错误] 连续 3 次重连失败, 结束录制")
                    break
                time.sleep(RECONNECT_DELAY)
                continue

            writer.write(frame)
            frame_count += 1

            if frame_count % (int(fps) * 10 or 100) == 0:
                elapsed = time.time() - start
                print(f"[录制] 已写 {frame_count} 帧 / {elapsed:.0f}s -> {out_path}")

            if args.duration and (time.time() - start) >= args.duration:
                print(f"[停止] 达到设定时长 {args.duration}s")
                break
            if args.max_frames and frame_count >= args.max_frames:
                print(f"[停止] 达到设定帧数 {args.max_frames}")
                break
    except KeyboardInterrupt:
        print("\n[停止] 手动中断 (Ctrl+C)")

    # ---- 收尾 ----
    writer.release()
    cap.release()
    elapsed = time.time() - start
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"[完成] {out_path} | {frame_count} 帧 | {elapsed:.1f}s | {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
