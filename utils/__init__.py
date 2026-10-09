"""公共工具包：配置、相机、队列背压、持久化、锚点匹配、批推理、进程编排。

仅包含跨分支（pole / marker）共用的通用能力；各分支专属算法
（倾角测量 / 靶标亚像素）留在各自的包内。本包模块顶层绝不 import 推理框架
（ultralytics/OpenVINO 只在推理子进程内、调用到相关函数时才延迟加载）。

模块：
- common.py     日志、相机打开、队列背压、数值清洗、模型路径选择、JSON 序列化
- config.py     统一配置加载与校验（config.yaml）
- persist.py    初始基准持久化（init_state.json / init_frame.jpg）
- anchor.py     持久 ID 锚点匹配（AnchorRegistry）
- inference.py  批推理基类（BatchInferenceBase/Seg/Det）与 ByteTrack 关联
- capture.py    拉流进程（每分支一个，内部多线程）
- recorder.py   记录进程（全系统唯一写入者）
- supervisor.py 主进程编排（spawn 子进程 + 僵尸检测 + 优雅退出）
"""
