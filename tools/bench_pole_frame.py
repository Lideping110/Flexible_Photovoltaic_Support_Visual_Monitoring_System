"""基准：pole worker 单帧全链路耗时拆解（predict / tracker_update / monitor.process）。

用真实 1080p 帧（含 capture 去畸变），复刻 run_pole_inference 内循环，
三段分别计时，定位 ~3s/帧 的真正元凶。
"""
import sys, time, tempfile
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import yaml
from utils.common import select_model_weights
from utils.inference import SegInference, make_byte_tracker, tracker_update
from utils.anchor import AnchorRegistry
from pole_tilt_monitor.monitor import ImageRelativePoleMonitor
from utils.persist import write_calibration_yaml

cfg = yaml.safe_load(Path(ROOT / "config.yaml").read_text(encoding="utf-8"))
m_pole = cfg["models"]["pole_seg"]
pole_cfg = cfg["pole"]
geom = pole_cfg.get("measurement", {})
bc = pole_cfg.get("baseline", {})
fc = pole_cfg.get("filter", {})
anchor_cfg = cfg["runtime"]["init"]["anchor"]

# 构造 monitor（image_relative 模式）
calib_dir = Path(tempfile.mkdtemp(prefix="cal_"))
calib_path = write_calibration_yaml(
    calib_dir / "calibration.yaml",
    {"K": [[2410.0, 0, 951.0], [0, 2413.0, 484.0], [0, 0, 1.0]], "dist": []})
monitor = ImageRelativePoleMonitor(
    str(calib_path), baseline_frames=bc.get("relative_frames", 1),
    endpoint_ratio=geom.get("endpoint_ratio", 0.10),
    ransac_residual_px=geom.get("ransac_residual_px", 2.5),
    ransac_iterations=geom.get("ransac_iterations", 300),
    ema_alpha=fc.get("alpha", 0.25),
    quality_threshold=geom.get("quality_threshold", 0.60))
tracker = make_byte_tracker(m_pole.get("tracker", "bytetrack.yaml"))
anchors = AnchorRegistry("bench", "pole", iou_min=anchor_cfg["iou_min"],
                         release_after_s=anchor_cfg["release_after_s"],
                         ema_alpha=anchor_cfg["ema_alpha"], prefix="P")
seg = SegInference(select_model_weights(m_pole, ROOT), m_pole["confidence"],
                   m_pole["class_id"], m_pole["device"], m_pole["tracker"], "bench")
seg.warmup()


def und(frame):
    K = np.array([[2410.0, 0, 951.0], [0, 2413.0, 484.0], [0, 0, 1.0]])
    return cv2.undistort(frame, K, np.zeros((4, 1)))


cap = cv2.VideoCapture(str(ROOT / "1080" / "1080.mp4"))
frames = []
while len(frames) < 20:
    ok, fr = cap.read()
    if not ok:
        break
    frames.append(und(fr))
cap.release()
print(f"[帧数] {len(frames)}, 尺寸 {frames[0].shape}")

t_pred = t_track = t_meas = 0.0
N = 0
ts = 0.0
for i, frame in enumerate(frames):
    t0 = time.perf_counter()
    result = seg.predict([frame])[0]
    t_pred += time.perf_counter() - t0

    t0 = time.perf_counter()
    dets = tracker_update(tracker, result, frame, with_mask=True)
    t_track += time.perf_counter() - t0

    # 首帧建锚点
    if i == 0:
        anchors.ensure_init(dets, ts)
    mapping = anchors.assign(dets, ts)

    t0 = time.perf_counter()
    for det in dets:
        pid = mapping.get(det["id"])
        if pid is None:
            continue
        monitor.process(pid, det["mask"], i)
    t_meas += time.perf_counter() - t0
    N += 1
    ts += 1.0

print(f"\n===== pole 单帧全链路（{N} 帧）=====")
print(f"predict(seg)   : {t_pred/N*1000:8.1f} ms/帧")
print(f"tracker_update : {t_track/N*1000:8.1f} ms/帧")
print(f"monitor.process: {t_meas/N*1000:8.1f} ms/帧")
print(f"合计          : {(t_pred+t_track+t_meas)/N*1000:8.1f} ms/帧")
print(f"→ 与真实 ~3s/帧 的差距若仍大，说明瓶颈在 worker 主循环其它处（如 json 写盘/队列等待）")
