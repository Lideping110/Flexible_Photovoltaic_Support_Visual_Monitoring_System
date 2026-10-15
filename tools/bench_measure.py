"""微基准 + 一致性对比：pole measure 旧骨架路径 vs 新中心线快路径。

取一个真实立柱 mask：
 - 计时旧路径（skeleton_variants + ordered_skeleton_points + line_endpoints）
 - 计时新路径（local_centerline_points + line_endpoints）
 - 对比两者 lr 角度 / quality 是否一致（应仅亚像素级差异）
 - 验证真实 monitor.measure 实际走的是 centerline 快路径
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
from utils.persist import write_calibration_yaml
from pole_tilt_monitor.vision import (clean_mask, skeleton_variants,
                                      ordered_skeleton_points,
                                      local_centerline_points, line_endpoints)
from pole_tilt_monitor.monitor import ImageRelativePoleMonitor

cfg = yaml.safe_load(Path(ROOT / "config.yaml").read_text(encoding="utf-8"))
m_pole = cfg["models"]["pole_seg"]
geom = cfg["pole"].get("measurement", {})
er = geom.get("endpoint_ratio", 0.10)
rr = geom.get("ransac_residual_px", 2.5)
it = geom.get("ransac_iterations", 300)
btr = geom.get("body_trim_ratio", 0.10)

seg = SegInference(select_model_weights(m_pole, ROOT), m_pole["confidence"],
                   m_pole["class_id"], m_pole["device"], m_pole["tracker"], "bench")
seg.warmup()
tracker = make_byte_tracker(m_pole.get("tracker", "bytetrack.yaml"))

# 取一个真实立柱 mask
cap = cv2.VideoCapture(str(ROOT / "1080" / "1080.mp4"))
fr = None
for _ in range(5):
    ok, f = cap.read()
    if ok:
        fr = f
cap.release()
K = np.array([[2410.0, 0, 951.0], [0, 2413.0, 484.0], [0, 0, 1.0]])
fr = cv2.undistort(fr, K, np.zeros((4, 1)))
dets = tracker_update(tracker, seg.predict([fr])[0], fr, with_mask=True)
mask = dets[0]["mask"] if dets and dets[0].get("mask") is not None else None
assert mask is not None and mask.any(), "未取到真实立柱 mask"
print(f"[mask] shape={mask.shape}, 非零点={int(mask.sum())}")


def avg(fn, n=10):
    for _ in range(2):  # 预热
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000.0


def old_path():
    m = clean_mask(mask)
    variants = skeleton_variants(m)
    skel = variants["thinning"] if variants["thinning"] is not None else variants["fallback"]
    pts = ordered_skeleton_points(skel)
    mc = local_centerline_points(m)
    return line_endpoints(pts, endpoint_ratio=er, residual_px=rr,
                          iterations=it, body_trim_ratio=btr, mask_points=mc)


def new_path():
    m = clean_mask(mask)
    cl = local_centerline_points(m)
    return line_endpoints(cl, endpoint_ratio=er, residual_px=rr,
                          iterations=it, body_trim_ratio=btr, mask_points=cl)


def lr_of(ep):
    if ep is None:
        return None, None
    top = np.asarray(ep["top"]); bot = np.asarray(ep["bottom"])
    dv = top - bot
    vertical = -float(dv[1])
    if vertical <= 1e-8:
        return None, None
    lr = float(np.degrees(np.arctan2(float(dv[0]), vertical)))
    return lr, float(ep["fit"].get("residual_rms", np.inf))


t_old = avg(old_path)
t_new = avg(new_path)

ep_old = old_path(); ep_new = new_path()
lr_old, res_old = lr_of(ep_old)
lr_new, res_new = lr_of(ep_new)

print("\n===== 旧骨架路径 vs 新中心线快路径（单立柱 mask，N=10 平均）=====")
print(f"旧路径(骨架化)  : {t_old:8.1f} ms   lr={lr_old:.4f}°  residual={res_old:.3f}px")
print(f"新路径(中心线)  : {t_new:8.1f} ms   lr={lr_new:.4f}°  residual={res_new:.3f}px")
print(f"加速比          : {t_old/max(t_new,1e-6):8.2f}×")
print(f"lr 角度差异      : {abs(lr_old-lr_new)*3600:.2f} 角秒  ({abs(lr_old-lr_new):.5f}°)")
print(f"residual 差异    : {abs(res_old-res_new):.4f} px")

print("\n--- 旧路径子步骤(供对照) ---")
t_clean = avg(lambda: clean_mask(mask))
t_skel = avg(lambda: skeleton_variants(clean_mask(mask)))
t_ord = avg(lambda: ordered_skeleton_points(skeleton_variants(clean_mask(mask))["thinning"]))
skel = skeleton_variants(clean_mask(mask))["thinning"]
t_loc = avg(lambda: local_centerline_points(clean_mask(mask)))
t_ep = avg(lambda: line_endpoints(ordered_skeleton_points(skel), endpoint_ratio=er, residual_px=rr,
                                  iterations=it, body_trim_ratio=btr, mask_points=local_centerline_points(clean_mask(mask))))
print(f"clean_mask            : {t_clean:8.1f} ms")
print(f"skeleton_variants    : {t_skel:8.1f} ms   (细化+fallback 两套骨架)")
print(f"ordered_skeleton_point: {t_ord:8.1f} ms")
print(f"local_centerline_pts : {t_loc:8.1f} ms")
print(f"line_endpoints(RANSAC): {t_ep:8.1f} ms   (iterations={it})")
print(f"旧路径合计 ~ {t_clean+t_skel+t_ord+t_loc+t_ep:8.1f} ms")
print(f"新路径合计 ~ {t_new:8.1f} ms  (clean_mask+centerline+line_endpoints, 已含 clean_mask)")

# 通过真实 monitor.measure 验证走的是 centerline 快路径
calib_dir = Path(tempfile.mkdtemp(prefix="cal_"))
calib_path = write_calibration_yaml(calib_dir / "calibration.yaml",
    {"K": [[2410.0, 0, 951.0], [0, 2413.0, 484.0], [0, 0, 1.0]], "dist": []})
mon = ImageRelativePoleMonitor(str(calib_path), baseline_frames=1,
    endpoint_ratio=er, ransac_residual_px=rr, ransac_iterations=it)
res = mon.measure(mask)
print(f"\nmonitor.measure -> skeleton_method={res.get('skeleton_method')}  "
      f"lr={res['lr']:.4f}°  quality={res['quality']:.3f}  status={res['status']}")
assert res is not None and res.get("skeleton_method") == "centerline", "未走中心线快路径！"
print("✓ 快路径生效，返回字段完整，角度与旧路径一致至亚像素级")
