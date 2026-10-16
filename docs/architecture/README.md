# 架构设计图（柔性光伏支架可视化监测系统重构）

六张架构图定义了项目的目标架构，与 `main.py`（`uv run python main.py`）实现一一对应。

| 文件 | 内容 |
|---|---|
| `01_overall_multiprocess_architecture.svg` | 总体多进程架构（批推理模式）：Supervisor + 每分支 1 个拉流进程（线程×N 摄像头）+ 1 个推理进程（模型仅加载一次、跨摄像头攒批）+ SpectrumAnalyzer + Recorder，进程总数恒为 7。重点区分 **文件源 / RTSP 实时流双链路**：采集层文件源 put_backpressure 背压不丢、RTSP 用 put_latest 丢旧保新+低延迟；消费层文件源由 supervisor 经 viz/export 生成带标注的 `_annotated.mp4` 供大屏 `<video>` 回放，RTSP 源由 dashboard 后端 RtspStreamManager 直连子码流 MJPEG 推浏览器（纯原始画面，main 不发布图片）。**报警与数据留存**也并入本图（见下节）：立柱倾角报警由 PoleInference **内联触发、直调 alarm_store 落盘**（不经 result_q/Recorder，原始帧+重编码帧成对 + manifest.jsonl）；振动报警由 SpectrumAnalyzer 触发、经 events.jsonl 留存原始片段+分析数据 |
| `02_unified_config_structure.svg` | 单一配置文件 `config.yaml` 结构：cameras[]（实际 4 路：pole_cam_01/02 + marker_cam_01/02，pole 挂内联 K/dist/R/t、marker 不挂标定）、models、pole、marker（fft 用 window_s/snr/min_samples/min_span_s；alarm 含 amplitude_mm/deflection_mm/cable_force_idx 三类振动报警阈值；placeholder 挠度/索力占位）、dashboard（stream_channel/stream_fps/reconnect_s）、runtime（capture 含 rtsp_rolling_s/rtsp_seg_s 滚动缓冲、batch/queue/init） |
| `03_branch_pipelines.svg` | 两条分支的进程内数据流水线：立柱（1fps 抽帧→去畸变→分割+ByteTrack→持久 ID 锚点匹配→几何观测双路径：行中心线快路径 / 骨架回退→RANSAC 拟合+质量→delta+EMA→measurements.jsonl）；靶标（逐帧→检测+ByteTrack 持续跟踪→λmin 选峰+亚像素→位移 mm/px→displacement.jsonl），位移样本经 spectrum_q 送给**独立 SpectrumAnalyzer 进程**做 30s FFT，并按振幅/索力/挠度三类阈值触发振动报警留存（详见 06 图） |
| `04_init_baseline_persistence.svg` | 初始基准持久化：首张成功帧保存 init_frame.jpg + init_state.json（原子写 tmp→os.replace，WinError5 退避重试×6，头部带 version=1 + created_at）；重启检测到已存在则直接加载、跳过基准建立期、delta 延续；存储于 output/{branch}/{cam_id}/init/（calibration.yaml 在 init/ 同级，仅 pole） |
| `05_persistent_id_anchor_matching.svg` | 持久 ID 匹配层：ByteTrack 原始 ID 仅作临时身份，持久 ID 由锚点（初始帧检测框 + 初始数据）经匈牙利指派（1−IoU，IoU≥iou_min=0.3）分配，遮挡换轨（如 id2→id9）可纠正回原 ID；配置：pole iou_min=0.3 / release_after_s=10 / ema_alpha=0.05，marker relock_max_px=200 |
| `06_alarm_logic_architecture.svg` | **报警逻辑架构（重点）**：上半部详述振动测振链路——SpectrumAnalyzer._finalize 每 30s 窗按 (cam_id,pid) 结算，振幅(真实)/索力(占位)/挠度(占位) 三类阈值任一命中即留存；三者共享同一份 analysis 与同一视频片段、RTSP 切窗每窗仅一次；按源类型分流留存（RTSP 滚动缓冲切 `{当前时间}_original.mp4` H.264 / 文件源 ffmpeg -c copy 切 `{t0}_{t1}_original.mp4` 原生编码）。**双产物**：原视频文件始终保留不删（侧车 `clip_native`），另存 `{原视频文件名}_h264.mp4`（侧车 `clip`）供 web `<video>` 播放——因 `-c copy` 保留的 mp4v/H.265 浏览器解不了码，故须先探测源编码、仅当确为 H.264 才流拷贝，否则转码，并校验副本确为 h264 才交付。底部对照条给出**另一条链路**——立柱倾角报警在 PoleInference 内联触发、直调 `alarm_store` 逐帧落盘（原始帧+重编码帧+manifest.jsonl，不经 result_q/Recorder，带 cooldown 5s 节流）。两条链路的形状差异与 dashboard 聚合方式一并呈现 |

## 设计原则（对齐 E:\JRY_BAA_121422 多进程架构）

1. `spawn` 启动，模型只在子进程内加载，Supervisor 绝不 import 推理框架；
2. 有界队列 + 差异化丢弃策略（帧队列按摄像头丢旧保序，结果队列阻塞写）；
3. 单一职责进程，队列单向 DAG；
4. 主进程 `is_alive()` 僵尸检测 + 统一 terminate；子进程错峰启动；
5. 检测模型批推理（攒批 predict），跟踪关联按 cam_id 隔离（每摄像头独立 BYTETracker 实例），与 JRY「批推理 + 按流关联」同构；
6. 代码分层：批推理能力沉淀为 `utils/inference.py` 的 `BatchInferenceBase`（`SegInference`/`DetInference` 继承，批量前向 + 自动退化）；推理进程按摄像头实例化会话类（`PoleCameraSession`/`MarkerCameraSession`），封装 ByteTrack + 锚点 + 测量 + 基准持久化，替代 dict 状态 + 闭包。

## 报警与数据留存（并入主流程，非独立旁路）

两条留存链路由主链路在运行时直接写盘，大屏「报警记录」页统一聚合展示：

### 1. 立柱检测报警 → 原始帧 + 重新编码帧
- **触发**：`pole_tilt_monitor/worker.py` 每帧算 `alarm=bool(...)`（阈值 `pole.alarm.left_right/front_back/total_deg`），为 True 时由 `utils/alarm_store.save_pole_alarm_snapshot` 落盘（受 `pole.alarm.cooldown_s` 节流，同一根立柱最短间隔才再写，避免持续报警刷屏）。
- **命名**（按源类型区分，对齐用户要求）：
  - 实时视频流（RTSP）：用 **当前时间** 命名 —— `<当前时间>_annotated.jpg`（重新编码帧）+ `<当前时间>_original.jpg`（原始帧）；
  - 视频数据（文件源）：用 **frame_id** 命名 —— `<frame_id>_annotated.jpg` + `<frame_id>_original.jpg`。
  - `<当前时间>` 格式 `YYYYMMDD_HHMMSS_ffffff`（微秒防重名）。
- **产物**（`output/alarms/pole/`）：上述成对 jpg + `manifest.jsonl`（每帧一条明细：snapshot=重新编码帧访问路径、original=原始帧访问路径、source=realtime/file，以及 pid/level/detail/各方向 delta/阈值）。
- **展示**：dashboard `build_records` 直接逐条读 `manifest.jsonl` 生成立柱报警记录（成对展示「重新编码帧」与「原始帧」）。

### 2. 视频测振 → 原始视频片段 + 振动分析数据
- **触发**：`marker_subpixel_tracker/spectrum.py` 每个 30s 固定窗结算，三类条件任一超过阈值（均沿用 `>0 启用、<=0 关闭`约定）即在该窗口留存报警数据，且**共享同一窗口的视频片段**（每窗口切窗请求只发一次）：
  - **振幅** `marker.alarm.amplitude_mm`：窗口内累计位移模长最大值（真实量，mm），语义同「最大振幅」；
  - **挠度** `marker.alarm.deflection_mm`：与 `marker.placeholder.deflection_mm`（占位值）比较；
  - **索力** `marker.alarm.cable_force_idx`：与 `marker.placeholder.cable_force_idx`（占位值）比较。
  - 分别发 `amplitude_alarm` / `deflection_alarm` / `cable_force_alarm` 事件（写入 `events.jsonl`），各自带独立 `id`（`{cam_id}_{pid}_w{seg}` / `..._defl` / `..._cable`），事件内嵌同一份 `analysis`（频率取 SNR 更高轴的 `f1_hz`、振幅、挠度/索力占位值并显式标注 `placeholder`，及相应阈值列）。
- **命名 + 产物**（按源类型区分）：
  - **文件源（视频数据）**：收尾阶段 supervisor 调 `viz/clip_retention.retain_vibration_clips` 从源视频按 `[t0,t1]` 切出 **`<t0>_<t1>_original.mp4`**，用 ffmpeg 流拷贝(-c copy) **保持原编码（不转 web 可播）**（ffmpeg 缺失时回退 OpenCV 重封装，仍不改 H.264）；
  - **RTSP 实时源**：`capture` 进程对实时 marker 相机维护 **滚动缓冲**（`output/rtsp_buffer/<cam_id>/seg_*.mp4`，分段时长 `runtime.capture.rtsp_seg_s`、保留 `rtsp_rolling_s` 秒）；振动报警发生时 `spectrum` 向 `clip_q` 发切窗请求，capture 从中切出 **`<当前时间>_original.mp4`** 并转 H.264（web 可播）。
  - 两类源最终都写出 `output/vibration/<id>.json` 侧车（`<id>` 即事件 id `{cam_id}_{pid}_w{seg}` / `..._defl` / `..._cable`），内含完整 analysis + `source` + **`clip`（web 可播副本）+ `clip_native`（原视频文件）**。同一窗口被多条件同时触发时三者**共享同一视频片段**（`vibration/<id>.json` 各自一份、片段文件同路径），不重复抽取。
- **双产物：原视频文件保留 + web 可播副本**（用户要求：原视频文件不能删，但 web 要能播）：
  - 原视频文件**始终保留、不修改不删除**，侧车记为 `clip_native`；文件源那份是 `-c copy` 的原生编码（实测多为 mp4v/MPEG-4 Part 2 或 H.265，**浏览器解不了码**），故不能直接给 `<video>`。
  - 另存一份 **`{原视频文件名}_h264.mp4`**（如 `0.0_2.0_original_h264.mp4`），侧车记为 `clip`，供 web 播放。
  - 由 `viz/clip_retention.ensure_web_h264` 生成：**先 ffprobe 探测源编码，仅当确为 `h264` 才用 `-c copy` 无损重封装，其余一律转码 libx264（veryfast/crf23/yuv420p/+faststart）**；转码后再校验副本确为 h264 才交付。⚠ 不探测就 copy 会把 mp4v 原样复制成"可播副本"，浏览器依然播不了（首次实现即踩此坑，测试当场暴露）。
  - 转码失败（ffmpeg 缺失等）时 `clip=None`、状态记 `ok_no_web_copy`，**如实置空，绝不把不可播原文件当成可播副本**。
- **保留上限**（`marker.alarm.keep_clips`，默认 **10**）：只保留最近 N 个**报警窗口**，超出的由 `prune_vibration_clips` 按视频 mtime 倒序整组删除（`_original.mp4` + `_h264.mp4` + 引用它们的侧车 json），避免磁盘无限增长。配成 `0` / `-1` / `all` / `inf` 表示"无穷大"——全部保留不删。同窗口的 3 个事件共享片段，按窗口整组同进同出，不留"有侧车没视频"的悬空引用。
- **展示**：dashboard `build_records` 读 `vibration/<id>.json` 侧车取 `clip` + `clip_native` + analysis，与三类振动报警事件按 `id` 关联（无重复记录）。大屏报警记录页顶部筛选器含「立柱超限 / 索力报警 / 振幅报警 / 挠度报警」四类；振动类（振幅/索力/挠度）详情统一渲染「`<video>` 播 `clip`（_h264 可播副本）+ 原视频文件下载入口（`clip_native`，注明浏览器可能无法直接播放）+ 主频/位移幅值/挠度/索力（均含阈值列，占位值标注「占位」）/窗口」。若切窗文件缺失或 web 副本生成失败则 `<video>` 不渲染，界面如实提示仅保留振动分析数据。
