"""N>=4 重锁单元测试：匈牙利最优指派 + relock_max_px 门限。

覆盖 4 个关键场景：
  1) 基础重锁 + 外来目标(超距)必须拒绝
  2) N=4 最优指派 —— 匈牙利必须优于逐框贪心最近
  3) 门限拒绝 —— 仅偏移 +9px 超 5px 门限 → 不分配
  4) 3 检测 4 候选 —— 多出的 pid 保持空置，不抢占
"""
import os
import sys
import numpy as np

sys.path.insert(0, r"D:\Flexible_Photovoltaic_Support_Visual_Monitoring_System")
from marker_subpixel_tracker.identity import MarkerTrackRegistry  # noqa: E402


def det(i, cx, cy=100.0, w=40.0, h=40.0):
    return {"id": i, "box": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]}


def build(reg, specs, relock_max_px=200.0):
    reg.targets.clear()
    for spec in specs:
        pid, cx = spec[0], spec[1]
        cy = spec[2] if len(spec) > 2 else 100.0
        box = np.array([cx - 20, cy - 20, cx + 20, cy + 20], dtype=np.float64)
        reg.targets[pid] = {
            "box": box, "last_box": box.copy(),
            "raw_id": None, "last_seen": 0.0,
            "baseline": None, "quality": None,
        }
    reg.initialized = True


def run(reg, dets, ts=1.0):
    m = reg.assign(dets, ts)
    # 反向映射 pid -> raw id 便于断言
    pid_of_raw = {rid: pid for rid, pid in m.items()}
    return m, pid_of_raw


def check(title, cond):
    print(("  [OK]   " if cond else "  [FAIL] ") + title)
    return cond


# ---------------------------------------------------------------- 1) 基础重锁 + 外来拒绝
print("== 1) 基础重锁 + 外来目标拒绝 (relock_max_px=200) ==")
reg = MarkerTrackRegistry("cam", relock_max_px=200.0)
build(reg, [("M1", 100.0), ("M2", 300.0)])
m, pid_of = run(reg, [det(5, 105.0, 102.0), det(8, 500.0, 500.0)])
ok = True
ok &= check("raw5 (近 M1) -> M1", pid_of.get(5) == "M1")
ok &= check("raw8 (远处外来) 无 pid", 8 not in pid_of)
print("  mapping:", m)

# ---------------------------------------------------------------- 2) N=4 最优指派
print("\n== 2) N=4 最优指派：匈牙利(总代价200) 必须赢贪心(总代价420) ==")
# 目标 T1..T4 在 x=0,100,200,300；新框在 x=90,110,290,310
# 匈牙利最优: d90->T1(90) d110->T2(10) d290->T3(90) d310->T4(10) = 200
# 贪心(按序) : d90->T2(10) d110->T3(90) d290->T4(10) d310->T1(310) = 420
reg = MarkerTrackRegistry("cam", relock_max_px=200.0)
build(reg, [("T1", 0.0), ("T2", 100.0), ("T3", 200.0), ("T4", 300.0)])
dets = [det(1, 90.0), det(2, 110.0), det(3, 290.0), det(4, 310.0)]
m, pid_of = run(reg, dets)
ok2 = True
ok2 &= check("d90  -> T1 (非贪心的 T2)", pid_of.get(1) == "T1")
ok2 &= check("d110 -> T2 (非贪心的 T3)", pid_of.get(2) == "T2")
ok2 &= check("d290 -> T3", pid_of.get(3) == "T3")
ok2 &= check("d310 -> T4", pid_of.get(4) == "T4")
# 1-1 双射校验：每个 pid 至多一个 det，每个 det 一个 pid
assigned_pids = list(m.values())
ok2 &= check("1-1 双射(无重复 pid)", len(assigned_pids) == len(set(assigned_pids)))
ok2 &= check("4 个 det 全部认领", len(m) == 4)
print("  mapping:", m)

# ---------------------------------------------------------------- 3) 门限拒绝
print("\n== 3) 门限拒绝：仅偏移 +9px (relock_max_px=5) ==")
reg = MarkerTrackRegistry("cam", relock_max_px=5.0)
build(reg, [("M1", 100.0)])
m, pid_of = run(reg, [det(9, 109.0, 100.0)])  # 距 M1 中心 9px > 5px 门限
ok3 = check("raw9 (偏移9px>5px) 无 pid", 9 not in pid_of)
print("  mapping:", m)

# ---------------------------------------------------------------- 4) 3 检测 4 候选
print("\n== 4) 3 检测 4 候选：多出的 M4 保持空置 ==")
reg = MarkerTrackRegistry("cam", relock_max_px=200.0)
build(reg, [("M1", 100.0), ("M2", 200.0), ("M3", 300.0), ("M4", 400.0)])
dets = [det(1, 105.0), det(2, 205.0), det(3, 305.0)]
m, pid_of = run(reg, dets)
ok4 = True
ok4 &= check("raw1 -> M1", pid_of.get(1) == "M1")
ok4 &= check("raw2 -> M2", pid_of.get(2) == "M2")
ok4 &= check("raw3 -> M3", pid_of.get(3) == "M3")
ok4 &= check("M4 未分配给任何 det", "M4" not in m.values())
ok4 &= check("恰好 3 个 det 被认领", len(m) == 3)
print("  mapping:", m)

all_ok = ok and ok2 and ok3 and ok4
print("\n==== 总判定:", "ALL PASS" if all_ok else "HAS FAILURES", "====")
sys.exit(0 if all_ok else 1)
