# 柔性光伏支架可视化监测系统

统一入口多进程监测系统：**立柱倾角监测**（分割模型）+ **靶标亚像素跟踪与主频提取**（检测模型），一条命令拉起全部摄像头，输出结构化 JSONL 供可视化消费。

## 启动

```bash
uv run python main.py --refresh-init   # 推理 + Web 大屏（默认 8023）
# 只跑推理时加 --no-dashboard
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--config` | `config.yaml` | 统一配置文件（唯一入口） |
| `--refresh-init` | 关 | 忽略已保存的初始基准，强制重建 init_state.json |
| `--no-dashboard` | 关 | 仅运行采集/推理，不启动 Web 大屏 |
| `--dashboard-port` | `8023` | 统一入口启动的大屏端口 |

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
│   ├── {cam_id}_annotated.mp4                    # main 收尾自动生成：标注视频（检测框+拟合直线，H.264）
│   └── init/{init_frame.jpg, init_state.json}    # 初始基准
└── marker/{cam_id}/
    ├── displacement.jsonl                        # 逐帧位移（px/mm）
    ├── spectrum.jsonl                            # 每 30s 主频记录
    ├── latest.json
    ├── {cam_id}_annotated.mp4                    # main 收尾自动生成（仅 file 源相机）
    └── init/...
└── viz/                                          # 方案A离线标注视频（viz_overlay.py 手动产物）
    ├── {cam_id}_annotated.mp4                    # 结果回贴原始视频
    ├── {cam_id}_baseline_frame_N.jpg             # pole 基线参考图（对齐 pole_tilt_monitor）
    └── frames/{cam_id}/frame_XXXXXX.jpg          # 标注帧 jpg（视觉 QA）
```

> 产物归属约定：**运行 main** 的内容全部落在 `output/<branch>/<cam_id>/` 下（含该相机的标注视频，由 supervisor 收尾阶段对 file 源相机自动生成，实时流源无本地文件则跳过）；**运行 viz_overlay** 的视频与基线图落在 `output/viz/`。Web 大屏实时画面优先播放每相机文件夹下的标注视频。
>
> **先清后跑**：每个相机的 `{measurements,displacement,spectrum}.jsonl` 会在本次运行的**首次写入时截断**（Recorder 以 `"w"` 开文件，同一次运行内后续写入仍是追加）。原因是 `ts` 为「相对源视频/流的秒」，每次运行都从 0 重新计时——若沿用追加，同一相机的文件里会叠出多段各自从 0 开始的时序，采样间隔中位数被拉小后 FFT 频率轴随之错乱（实测把 0.06Hz 主频算成 25.9Hz、索力算成千万级）。`latest.json` 仍覆盖写，`init/` 仍按 `--refresh-init` 语义持久化。
> 换摄像头/换源后必须重跑 main，否则旧数据的时间基准与新媒体文件对不上（例如把 RTSP 的 48s 数据配到 11.9s 的视频上）。

> **实时画面与 main 解耦**：main 进程只做推理，把位移/频谱/测量结果写入 jsonl（**不发布任何图片**）；Web 大屏的实时画面由 **dashboard 后端自行 cv2 直连 RTSP 子码流**（默认通道 102，由相机 URL 的 `Channels/101` 派生为 `Channels/102`）拉流、以 MJPEG 推给浏览器，纯原始画面、无叠加。main 走主码流 101，两进程是相机的两个独立消费者，互不通信、互不抢带宽。两路视频源的差异见下表：

| 相机源 | 大屏实时画面 | 说明 |
|---|---|---|
| RTSP（实时流） | `GET /video/<cam>` MJPEG（dashboard 直连子码流 102） | `<img>` 直接播放，main 不参与 |
| 本地文件 | `<video>` 播 `output/<branch>/<cam>/<cam>_annotated.mp4` | 由 supervisor 对 file 源自动生成 |

> 注：RTSP 相机目前**没有落盘的「处理好的录像」**（supervisor 仅对 file 源生成 `_annotated.mp4`，RTSP 源跳过）。非实时/录像回放形态待后续接入「持续录制 + 离线标注」能力后再补齐。

## Web 大屏（dashboard.py）

```bash
uv run python dashboard.py --config config.yaml --port 8023
# 浏览器打开 http://127.0.0.1:8023
```

运行 `uv run python main.py` 会自动以子进程启动同一个大屏，并在 main 结束时回收；
上面的命令适合需要独立运行/调试大屏的场景。

左右分屏：**左＝立柱倾角监测**、**右＝视频测振**；每侧含「初始状态画面 / 实时展示(分析)界面」两个视图，视图内可按相机下拉切换（多路摄像头各自独立）。

| 路由 | 说明 |
|---|---|
| `GET /` | 大屏页面 |
| `GET /api/state` | 左（立柱 初始基准 + 当前快照 + 时间线）/ 右（测振 最新 30s 窗口） |
| `GET /api/marker_window?cam=<id>&t=<sec>` | 测振**播放联动**：返回「以 t 结尾的最近 30s」窗口的时移曲线 + **同一窗口**的振动分析 + 频谱出图门限（`fft_ready` / `fft_cover_s` / `fft_need_s` / `fft_reason`）；`t` 省略 = 最新 |
| `GET /video/<cam>` | RTSP 相机 MJPEG 实时帧流（dashboard 直连子码流，纯原始画面） |
| `GET /media/...` | 原始视频、标注视频、初始帧图（支持 Range，`<video>` 可直接播） |

### 视频文件源的播放联动

`cameras[].url` 为本地视频文件时，实时界面用 `<video>` 播放 `output/<branch>/<cam_id>/<cam_id>_annotated.mp4`（不存在则回退原视频），并且**画面播到哪，下面的表格/曲线就是哪一段**：

- **左**：`timeupdate` → 在时间线里取 `ts <= currentTime` 的最近快照，驱动「当前状态」表；
- **右**：`timeupdate`（节流 0.4s，拖动进度条由 `seeked` 强制重拉）→ 请求 `/api/marker_window`，把时移曲线窗口、频谱图与振动分析一起切到 `[t-30, t]`；三者取自**同一段数据**，避免"曲线是这 30s、主频却来自另一 30s"的错配；
- 实时流源（RTSP）没有 `currentTime`，退回「最新窗口」展示。

#### 频谱图为什么要等窗口攒够才出图

频谱是"一整段稳态数据"的统计量，不是逐帧量。**数据不够时画出来的只是趋势/噪声形状，不是谱**。

此前 `chartSpectrum` 由前端本地实时算 FFT，判据是"曲线有 8 个采样点就画"——25fps 下约 **0.32s** 就冒出一条曲线，与主链路"每 `window_s` 秒一窗"的语义完全背离，这就是"刚开始就有频率曲线"的原因。

现在改为**只读后端结论**（`_fft_gate`，与振动分析表里"主频"同一条判据），未满窗时只显示「等待首窗数据… 累积 X / 需 Y s」，不画曲线。因此：

| 场景 | 行为 |
|---|---|
| 播放到 0 ~ 门限前 | 频谱区显示累积进度，**不出曲线**；表格里主频为 `—`（两者永远同步） |
| 跨过门限后 | 出曲线，图上标注的主频**直接用后端值**（不再前端自己找峰，避免"图上 3.12Hz、表格 `—`"） |
| 页面刚打开且未播放 | 走「最新窗口」语义（与左侧一致），显示当前完整窗口的结果 |

#### 播放末尾为什么要强制刷新一次

`timeupdate` 有 0.4s 节流（避免每帧都发请求），但**末尾几次更新的增量往往小于门限，会被一直跳过**；又因为标注视频带 `loop`，循环回绕时浏览器**不触发 `ended`**。两者叠加就会出现"视频播完了、曲线还停在中途窗口"：

> 实测停在 `t=11.36s`，而数据实际到 `11.84s` —— 界面显示 `M1 0.088Hz / SNR 2540.1`、`M2 0.128Hz / SNR 301.8`，而文件里是 `M1 0.0842Hz / SNR 3741.4`、`M2 0.1202Hz / SNR 389.6`。**这正是"界面 SNR 和 spectrum.jsonl 对不上"的真正原因**（反查 SNR 逐位吻合，可确证窗口终点是 11.36s 而非 11.84s）。

处理：**离末尾不足 0.5s 时逐帧强制刷新**（绕过节流），回绕前必落在数据末端；`pause`（暂停点就是用户要看的时刻）与 `ended`（非 loop 场景）同样强制重拉；请求撞车时不丢弃，改为**排队补跑**。

> 注：循环回绕到 0 之后，窗口会重新从 `[0,0]` 开始累积（回到「等待首窗数据」，约 10s 后再出频谱）。这是循环播放的固有行为。若希望短视频（源短于 `marker.fft.window_s`）**始终显示整段分析结果**、不做重复累积，需要改窗口定义，属**待定项**。

### 联动逻辑改完怎么自检

```bash
uv run python dashboard.py --config config.yaml --port 8023   # 先起大屏
node tools/frontend_probe.js                                  # 再跑探针
```

探针从运行中的大屏拉首页 HTML，抽出内嵌的真实脚本，在最小 DOM 沙箱里执行，打印各时刻的窗口范围、曲线点数、**频谱门限与频谱图实际画了什么**、分析表数值。它还会**模拟真实 DOM 事件**（`addEventListener` 挂上的 `timeupdate` / `pause` / `ended` 由 `fire()` 触发，并带上 `duration`），因此能直接复现"末尾被节流吞掉"这类只在事件序列里才暴露的问题。

**判定要点**：曲线点数随 t 增长并在一个窗口后保持有界；`t<10s` 频谱区只出现「等待首窗数据…」且主频为 `—`；`t>=10s` 才出频谱且图上主频与表格主频一致；**播放末尾必须强制刷新到数据末端**（不得停在中途窗口，最终值应与 `spectrum.jsonl` 的 `dy` 行逐位一致）；挠度/索力恒为占位值。改动联动逻辑后跑一次，可确认前后端口径没有分叉。

### 挠度与索力：目前是占位值，不做计算

界面上「初始挠度/索力」「挠度/索力」两列显示的是 **`config.marker.placeholder` 里的占位值**（当前 `0.0mm` / `100.0`），并在表头标注「占位」——**不是测量结果**。

原因是二者尚无可靠的物理模型（影响因素没梳理清楚），而此前看板是拿
「窗口内位移幅值冒充挠度、主频平方比 `(f1/f1_ref)²×100` 冒充索力」顶上的，数字看着像真的却无法解释。推理主链路 `marker_subpixel_tracker` 本来就**只产出** `displacement.jsonl`（位移）与 `spectrum.jsonl`（主频），**不产出挠度/索力**。

同表其余两列仍是真实计算：

| 列 | 来源 | 性质 |
|---|---|---|
| 最大振幅(mm) | 窗口内位移模长的最大值 × `mm_per_px` | 真实计算 |
| 主频率(Hz) | 窗口 FFT（复用主链路 `_segment_fft`，门限取 `marker.fft`）| 真实计算 |
| 挠度(mm) | `config.marker.placeholder.deflection_mm` | **占位** |
| 索力(相对) | `config.marker.placeholder.cable_force_idx` | **占位** |

将来真实输出就绪，把 `placeholder.enabled` 置 `false` 走真实计算，或直接改这两个数值即可（前端两处的「占位」标注同步去掉）。

**主频 / 频谱的可信度门限**（与主链路 `SpectrumAnalyzer` 同一套判定，不是另一套标准）：

- 三个参数全部取自 config 的 `marker.fft`：`min_samples`（窗口内最少样本）、`snr`（最小信噪比）、`min_span_s`（窗口内样本**实际跨度**下限，秒）；
- 跨度不足时**不出频谱、不给主频**——此时 FFT 主峰只是运动趋势的伪峰，画出来/算出来都没有意义；
- **主频与频谱图共用这一条判据**（后端 `_fft_gate` 单一来源，`/api/state` 与 `/api/marker_window` 各调一次同一个函数），所以"表格里有主频"与"频谱图出现曲线"永远同刻发生，不会一边有一边空；
- 源总长不足 `window_s` 时，跨度门限取 `min(min_span_s, window_s)`——否则拿名义 30s 去判会**永远判未满窗**（11.9s 的短视频频谱永不出图）。主链路冲刷末窗时同样按实际样本成窗，口径一致；
- 想严格"满 30s 才出图"，把 `marker.fft.min_span_s` 改成 `30.0` 即可，**不需要改代码**；
- 不满足时主频显示 `—`、频谱区显示累积进度，并在标题处标注「窗口 Xs 样本不足，主频需 ≥Ns 窗口」。可信度指标（样本数 / 跨度 / SNR）随每次响应返回。

（挠度与索力是占位值，不参与上述门限，见上一节。另：界面标题里的"最近 30s"与 `marker.fft.window_s` 是同一个数，若改窗口长度需同步改标题文案。）

## 结果可视化（方案 A：回贴原始视频）

```bash
uv run python viz_overlay.py                  # 全部摄像头
uv run python viz_overlay.py --cam marker_cam_01 --dump-frames 6
```

- 原理：按 `frame_id`（源视频 0 基真帧序）顺序遍历原始视频逐帧对位叠加，规避 H.264 seek 漂移；
- **坐标系对齐**：marker 分支不去畸变 → 直接叠加；pole 分支 `image_relative` 模式
  在 capture 阶段已 `cv2.undistort` → 渲染前对每帧做同参数去畸变，保证
  `box/top/bottom` 严丝合缝；
- **渲染对齐**：pole 分支复刻 `D:\摄像头拉流 pole_tilt_monitor` 的实时渲染风格
  （`vision.draw_measurement`）：白框、绿顶/红底端点圆、青色中心线、白字
  `ID..Q..OK LR..FB..T..deg`、INVALID 橙字、ALARM 红字、顶部 HUD
  `Frame N  Baseline poles: M`；离线回放与实时监测画面像素级一致，可直接对比；
- **基线参考图**：pole 分支额外产出 `{cam_id}_baseline_frame_N.jpg`——回读首有效帧、
  去畸变后绘制各杆中位基线（对齐 `save_relative_baseline_images`）；
- 叠加内容：立柱框 + 端点圆 + 中心线 + 倾角标签；靶标框 + 亚像素中心十字
  + 累计位移(mm) + mm/px；
- **亚像素中心十字的尺寸/线宽只在 `utils/common.py` 定义一次**
  （`CROSS_SIZE_PX` / `CROSS_THICKNESS` / `CROSS_DOT_RADIUS`，绘制函数
  `draw_subpixel_cross`），`viz/render.py`（标注视频 + 实时叠加）与
  `utils/persist.py`（init_frame.jpg）都调它，**改一处三处同时生效**。
  此前两处各抄了一份 `size = max(18, min(bw,bh)*0.9)` 的自适应公式，结果是靶标框
  一大十字就跟着暴涨（框 110px 时算出 99px 跨度、线宽 3+黑边 5，几乎盖住整个靶标）。
  两个注意点：① `CROSS_THICKNESS` **不要设成 1**——1px 细线经 JPEG / H.264 的色度
  子采样后会被压掉大半（实测只剩 4×4 像素的红点、十字几乎不可见）；② 十字跨度有
  上限，不再随大框无限放大，小框才按比例收缩。效果对比见 `docs/cross_marker_compare.jpg`。
- 只消费 `output/` + 源视频，不 import 推理栈（无模型文件也能跑）；RTSP 实时源
  不支持离线回放，请先录制成文件。

## 取流地址（现场部署参考）

```
rtsp://admin:xxx@192.168.1.64:554/streaming/Channels/101   # 主码流
rtsp://admin:xxx@192.168.1.64:554/streaming/Channels/102   # 子码流
```

将 `config.yaml` 中对应摄像头的 `url` 改为 RTSP 地址即可切换实时流（本地视频文件仅用于联调）。
