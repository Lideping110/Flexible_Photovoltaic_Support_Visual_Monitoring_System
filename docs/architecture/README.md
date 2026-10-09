# 架构设计图（柔性光伏支架可视化监测系统重构）

五张架构图定义了项目的目标架构，与 `main.py`（`uv run python main.py`）实现一一对应。

| 文件 | 内容 |
|---|---|
| `01_overall_multiprocess_architecture.svg` | 总体多进程架构（批推理模式）：Supervisor + 每分支 1 个拉流进程（线程×N 摄像头）+ 1 个推理进程（模型仅加载一次、跨摄像头攒批）+ SpectrumAnalyzer + Recorder，进程总数恒为 7 |
| `02_unified_config_structure.svg` | 单一配置文件 `config.yaml` 结构：cameras[]（id/url/task/内联标定）、models、pole、marker、runtime |
| `03_branch_pipelines.svg` | 两条分支的进程内数据流水线：立柱（1fps 抽帧→去畸变→分割+ByteTrack→RANSAC 倾角）；靶标（逐帧→检测锁定→λmin 亚像素→位移 mm→30s FFT） |
| `04_init_baseline_persistence.svg` | 初始基准持久化：首张成功帧保存 init_frame.jpg + init_state.json（原子写）；重启检测到已存在则直接加载、跳过基准建立期、delta 延续 |
| `05_persistent_id_anchor_matching.svg` | 持久 ID 匹配层：ByteTrack 原始 ID 仅作临时身份，持久 ID 由锚点（初始帧检测框 + 初始数据）经匈牙利指派（1−IoU）分配，遮挡换轨（如 id2→id9）可纠正回原 ID |

## 设计原则（对齐 E:\JRY_BAA_121422 多进程架构）

1. `spawn` 启动，模型只在子进程内加载，Supervisor 绝不 import 推理框架；
2. 有界队列 + 差异化丢弃策略（帧队列按摄像头丢旧保序，结果队列阻塞写）；
3. 单一职责进程，队列单向 DAG；
4. 主进程 `is_alive()` 僵尸检测 + 统一 terminate；子进程错峰启动；
5. 检测模型批推理（攒批 predict），跟踪关联按 cam_id 隔离（每摄像头独立 BYTETracker 实例），与 JRY「批推理 + 按流关联」同构；
6. 代码分层：批推理能力沉淀为 `utils/inference.py` 的 `BatchInferenceBase`（`SegInference`/`DetInference` 继承，批量前向 + 自动退化）；推理进程按摄像头实例化会话类（`PoleCameraSession`/`MarkerCameraSession`），封装 ByteTrack + 锚点 + 测量 + 基准持久化，替代 dict 状态 + 闭包。
