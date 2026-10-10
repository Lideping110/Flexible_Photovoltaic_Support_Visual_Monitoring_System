"""离线结果可视化包（方案 A：结果回贴原始视频）。

只消费 output/ 下的 JSONL + 原始视频，不 import 推理栈，不依赖模型文件。
坐标系对齐规则（与 capture.py 完全一致）：
- marker 分支：帧不去畸变，box/cx/cy 是原始像素坐标 → 直接叠加。
- pole 分支 image_relative 模式：capture 已对帧做 cv2.undistort，
  box/top/bottom 是去畸变帧坐标 → 叠加前必须对原始帧做同参数 undistort。
"""
