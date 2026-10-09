# marker_subpixel_tracker

靶标定位与亚像素跟踪验证项目，与 `pole_tilt_monitor` 共用工作区根目录的 Python 环境。

## 方法

1. 每帧使用 YOLO26 `marker26_det.pt` 通过 Ultralytics 内置 ByteTrack（`model.track(persist=True)`）定位靶标 ROI，得到带持久轨迹 ID 的检测框；
2. 目标锁定器（`TargetLock`）把测量固定在单一轨迹 ID 上：首次自动锁定置信度最高的轨迹，此后逐帧只测该 ID 的框；可配置丢失超过 N 帧后自动重锁（`tracking.reacquire_after_frames`）；
3. 直接裁剪锁定框得到 ROI（不扩边；扩边会把框外背景裁进 ROI，在裁剪边界产生比真实角点更强的截断伪峰），按论文 DAVIM 的 Harris/结构张量流程计算 `lambda_min`；
4. 选峰策略（锚点窗口）：在 `localization.continuity_radius_px` 半径内选 `lambda_min` 峰。锚点分两种——**同轨迹续帧**用「框位移预测锚点」（`predict_anchor`：当前框中心 + 上一帧测量点相对框中心的固定偏移，即锚点跟随 ByteTrack 框走，框每帧独立回归不滞后，故快速移动时窗口不会被 20px 半径限速、测量点不会拖后）；**首帧/换轨迹**用 ROI 中心（对应靶标板中心圆斑，确定性种子）。全局 argmax 会在板内多个等强角点间逐帧翻转（实测 ±63px）且可能锁到 ROI 边界截断峰，锚点窗口两者皆免疫；设 `continuity_radius_px: 0` 可退回论文原始全局 argmax 行为。所选峰的 λ 须 ≥ `localization.min_lambda_min`（默认 5000，快速场景真纹理 λ ≥ 5089、纯噪声 < 1000，中间有清晰分界）：低于阈值的帧诚实报丢失（`detected=False`），而非硬测垃圾纹理；
5. 将选中峰作为靶标中心，使用亚像素迭代精化得到中心点坐标；
6. 仅当相邻两次测量属于同一轨迹 ID、**帧号严格相邻**（跨丢失段的重捕获帧不产出位移，防止把跨段总位移混入帧间数据）、且本次峰来自锚点窗口时，用中心点坐标差计算像素位移，否则 dx/dy 记 NaN。

跟踪实现与 `pole_tilt_monitor/src/tracker.py` 同构：一次前向同时产出检测框与 ByteTrack ID，无 ID 时拒绝输出（绝不退化为逐帧检测冒充跟踪）。这里不使用金字塔 Lucas-Kanade 光流，也不使用 RANSAC 仿射变换。论文测量的是每帧 ROI 内的单个靶标中心点，而不是多角点的帧间跟踪；因此去畸变后靶标发生椭圆形变也不会被假设为正圆。

CSV 列：`frame, time_s, detected, method, det_conf, track_id, lambda_min, continuous, x_px, y_px, dx_px, dy_px`（`continuous=False` 表示该帧峰来自全局 argmax 回退而非锚点窗口，其 dx/dy 必为 NaN）。

输出视频帧率自动跟随输入视频源帧率（图片序列默认 10fps），不会出现变速播放。

## 运行

推理模型按以下顺序选择：先查找配置项 `model.openvino_dir`（也接受
`openvino_model` 或 `openvino_weights`），再扫描工作区和项目目录下的
`marker26s_det_openvino_model`、`marker26_det_openvino_model`；找到包含 `.xml`
的 OpenVINO 导出目录后直接使用。没有可用 OpenVINO 目录时，回退到
`model.weights` 指定的 `.pt` 文件。可通过命令行 `--weights` 显式指定模型以覆盖自动选择。

修改 `config/config.yaml` 的输入源后运行：

```powershell
uv run marker_subpixel_tracker --config marker_subpixel_tracker/config/config.yaml
```

The command now starts the camera-first realtime pipeline. Use `--camera` (or
`--rtsp`) for a camera URL/index; when it cannot be opened, `source.input` is
used as the video fallback. Add `--no-display` for headless processing, or
`--display` to force the preview window. The same pipeline can be launched as
`uv run python -m marker_subpixel_tracker.src.realtime`.

实时摄像头优先：在配置的 `source.camera_url` 填入 RTSP 地址（或运行时传入
`--camera rtsp://...` / `--camera 0`）。摄像头无法打开时会自动回退到
`source.input` 视频文件；也可用 `--video path/to/file.mp4` 临时覆盖回退视频。
加 `--display` 可打开实时标注窗口，按 `q` 退出。

每次启动会打印最终选中的推理模型路径。

也可以直接使用配置文件中的默认路径运行：

```powershell
uv run marker_subpixel_tracker
```
