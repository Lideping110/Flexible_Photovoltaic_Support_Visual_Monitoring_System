# camera-pull — RTSP 拉流保存本地

基于 OpenCV 拉取海康摄像头 RTSP 视频流并保存为本地 MP4。

## 取流地址

```
rtsp://admin:hhjt110110@192.168.1.64:554/streaming/Channels/101
```

| 通道 | 说明 |
|---|---|
| `Channels/101` | 主码流（3840×2160 @ 20fps，约 90MB/分钟） |
| `Channels/102` | 子码流（更低分辨率，适合长时间存档） |

## 环境

- Python ≥ 3.13
- 依赖：`opencv-python`、`numpy`（由 uv 管理，见 `pyproject.toml`）
- 包索引：清华镜像（已写入 `pyproject.toml`，规避本机代理拦截）

```bash
# 首次同步环境
uv sync

# 运行（自动使用项目 .venv）
uv run python rtsp_capture.py
```

## 用法

```bash
# 持续录制，Ctrl+C 停止（默认按时间命名）
uv run python rtsp_capture.py

# 限时录制（秒）
uv run python rtsp_capture.py --duration 300

# 指定输出文件
uv run python rtsp_capture.py --out 2026-09-08.mp4

# 指定帧数上限
uv run python rtsp_capture.py --max-frames 1000

# 更换取流地址
uv run python rtsp_capture.py --url "rtsp://admin:pwd@192.168.1.64:554/streaming/Channels/102"
```

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--url` | 见上 | 取流地址 |
| `--out` | `capture_年月日_时分秒.mp4` | 输出文件 |
| `--duration` | `0`（不限） | 录制时长（秒） |
| `--max-frames` | `0`（不限） | 最大帧数 |

## 实现要点

- FFMPEG 后端 + TCP 取流（`rtsp_transport;tcp`），抗丢包更稳
- 自动断线重连（最多 5 次），断线续录
- 默认编码 `mp4v`；4K 存档建议换 `avc1`（H264）或改用子码流
