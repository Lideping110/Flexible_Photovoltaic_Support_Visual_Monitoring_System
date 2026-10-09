# 柔性光伏支架可视化监测系统

统一入口多进程监测系统：**立柱倾角监测**（分割模型）+ **靶标亚像素跟踪与主频提取**（检测模型），一条命令拉起全部摄像头，输出结构化 JSONL 供可视化消费。

## 启动

```bash
uv run python main.py
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--config` | `config.yaml` | 统一配置文件（唯一入口） |
| `--refresh-init` | 关 | 忽略已保存的初始基准，强制重建 init_state.json |

## 目录结构

```
main.py                       # 统一入口
config.yaml                   # 唯一配置文件（摄像头/模型/分支参数/运行参数）
utils/                        # 公共工具（跨分支通用，模块顶层绝不 import 推理框架）
  ├── common.py               #   日志 / 相机 / 队列背压 / 数值清洗 / 模型选择 / JSON 序列化
  ├── config.py               #   配置加载与校验
  ├── persist.py              #   初始基准原子持久化（init_frame.jpg + init_state.json）
  ├── anchor.py               #   持久 ID 锚点匹配层（遮挡换轨纠正）
  ├── inference.py            #   批推理基类（BatchInferenceBase/Seg/Det）+ ByteTrack 关联
  ├── capture.py              #   拉流进程（每分支 1 个，内部每摄像头 1 个读流线程）
  ├── recorder.py             #   Recorder 进程（单一写入者，JSONL + latest.json）
  └── supervisor.py           #   Supervisor 主进程：spawn + 健康监控 + 优雅退出
pole_tilt_monitor/            # 立柱分支（分割模型 + 倾角测量算法）
  ├── worker.py               #   立柱推理进程（整秒攒批 + PoleCameraSession 会话类）
  └── monitor.py ...          #   倾角测量算法（RANSAC/质量门控/EMA/基准）
marker_subpixel_tracker/      # 靶标分支（检测模型 + 亚像素跟踪算法）
  ├── worker.py               #   靶标推理进程（滑动攒批 + MarkerCameraSession 会话类）
  ├── spectrum.py             #   主频分析进程（30s 窗口 FFT）
  └── pipeline.py ...         #   亚像素定位 / 特征 / 跟踪算法
convert_to_openvino.py        # 模型 .pt/.onnx → OpenVINO IR 导出工具
output/                       # 结构化输出（JSONL + latest.json 快照）
docs/architecture/            # 架构设计图（5 张 SVG + README）
```

## 架构要点

- **进程总数恒为 7**，与摄像头数量无关：Supervisor + PoleCapture + PoleInference + MarkerCapture + MarkerInference + SpectrumAnalyzer + Recorder；
- **批推理**：每分支推理进程只加载一份模型，跨摄像头攒批前向；跟踪关联按 cam_id 隔离（每摄像头独立 ByteTrack），与 JRY_BAA_121422「批推理 + 按流关联」同构；
- **批推理基类 + 每摄像头会话类**：`utils/inference.py` 提供 `BatchInferenceBase`（`SegInference`/`DetInference` 继承），两分支复用批预测与自动退化；推理进程按摄像头实例化 `PoleCameraSession`/`MarkerCameraSession`，封装 ByteTrack + 锚点 + 测量 + 基准持久化，替代 dict 状态 + 闭包；
- **单一配置文件**：`cameras[]` 是唯一摄像头清单（id/url/task/内联标定 K·dist·R·t），`task: pole|marker` 指定摄像头角色；
- **初始基准持久化**：首帧保存 `init_frame.jpg` + `init_state.json`（原子写），重启自动加载、delta 延续；
- **持久 ID（两分支语义不同）**：立柱固定 → 锚点框匹配（遮挡换轨如 id2→id9 按初始框几何先验纠正回原 ID）；靶标晃动 → 靠 ByteTrack raw id 持续跟踪维持身份，初始框仅存档、不参与匹配；

## 输出结构

```
output/
├── events.jsonl                                  # 全生命周期事件
├── pole/{cam_id}/
│   ├── measurements.jsonl                        # 逐杆逐帧倾角（含 delta/质量/报警）
│   ├── latest.json                               # 实时快照（可视化轮询）
│   ├── calibration.yaml                          # 落盘标定
│   └── init/{init_frame.jpg, init_state.json}    # 初始基准
└── marker/{cam_id}/
    ├── displacement.jsonl                        # 逐帧位移（px/mm）
    ├── spectrum.jsonl                            # 每 30s 主频记录
    ├── latest.json
    └── init/...
```

## 取流地址（现场部署参考）

```
rtsp://admin:xxx@192.168.1.64:554/streaming/Channels/101   # 主码流
rtsp://admin:xxx@192.168.1.64:554/streaming/Channels/102   # 子码流
```

将 `config.yaml` 中对应摄像头的 `url` 改为 RTSP 地址即可切换实时流（本地视频文件仅用于联调）。
