"""head-to-head：pole 分割模型 vs marker 检测模型，同帧同硬件计时。

两模型骨干同为 yolo26s，差异仅在任务头：
 - pole  : YOLO26s-seg  (retina_masks=True，输出逐像素 mask)
 - marker: YOLO26s-det  (仅包围盒)
同帧、同 GPU.0、同 OpenVINO LATENCY 模式下各跑 N 次，量化"分割头"带来的额外开销。
"""
import sys, time
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import yaml
from utils.common import select_model_weights
from utils.inference import SegInference, DetInference

cfg = yaml.safe_load(Path(ROOT / "config.yaml").read_text(encoding="utf-8"))
m_pole = cfg["models"]["pole_seg"]
m_marker = cfg["models"]["marker_det"]

seg_w = select_model_weights(m_pole, ROOT)
det_w = select_model_weights(m_marker, ROOT)
print(f"[seg 权重] {seg_w}")
print(f"[det 权重] {det_w}")

seg = SegInference(seg_w, m_pole["confidence"], m_pole["class_id"],
                  m_pole["device"], m_pole["tracker"], "bench-seg")
det = DetInference(det_w, m_marker["confidence"], m_marker["class_id"],
                   m_marker["device"], m_marker["tracker"], "bench-det")

# 同一帧（1080p），喂给两个模型以隔离"模型本身"差异（去畸变是模型外 cv2 操作，不影响模型耗时）
cap = cv2.VideoCapture(str(ROOT / "1080" / "1080.mp4"))
ok, frame = cap.read()
for _ in range(10):
    ok, frame = cap.read()
cap.release()
assert ok and frame is not None, "未取到帧"
print(f"[帧尺寸] {frame.shape}")

seg.predict([frame]); det.predict([frame])  # 预热（OpenVINO 编译）

def avg(fn, n=20):
    for _ in range(3):
        fn([frame])
    t0 = time.perf_counter()
    for _ in range(n):
        fn([frame])
    return (time.perf_counter() - t0) / n * 1000.0

t_seg = avg(seg.predict)
t_det = avg(det.predict)

print("\n===== 同帧同硬件：分割 vs 检测（yolo26s 骨干相同）=====")
print(f"pole  seg (YOLO26s-seg, retina_masks): {t_seg:8.1f} ms/帧")
print(f"marker det (YOLO26s-det)            : {t_det:8.1f} ms/帧")
print(f"差距倍数 : {t_seg/max(t_det,1e-6):.1f}×   (纯分割头开销，与 26s 骨干无关)")
print("注：生产环境 pole 还额外做 cv2.undistort（~数 ms，模型外），此处未计，仅隔离模型本体。")
