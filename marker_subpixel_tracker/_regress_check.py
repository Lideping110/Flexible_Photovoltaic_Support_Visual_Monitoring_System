"""回归验证：统计 marker 跟踪结果。

- 总帧数、M1/M2 各自覆盖帧数（M2 目标覆盖率 >= 200/297 视为无退步）
- 是否存在同一帧同时出现 M1 与 M2（双框，证明重锁两个目标都在场）
- 各 frame 出现过的 pid 集合
"""
import json
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else \
    r"D:\Flexible_Photovoltaic_Support_Visual_Monitoring_System\output\marker\marker_cam_01\displacement.jsonl"

frames_pid = defaultdict(set)
pid_count = defaultdict(int)
total_frames = set()
with open(path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        fid = r.get("frame_id")
        pid = r.get("pid")
        if fid is None or pid is None:
            continue
        total_frames.add(fid)
        frames_pid[fid].add(pid)
        pid_count[pid] += 1

n_frames = len(total_frames)
print(f"结果文件: {path}")
print(f"总帧数(含 M1/M2 任一带位移的帧): {n_frames}")
for pid in sorted(pid_count):
    print(f"  {pid}: 出现 {pid_count[pid]} 条记录")

# 双框帧：同帧同时含 M1 与 M2
dual = [fid for fid, ps in frames_pid.items() if "M1" in ps and "M2" in ps]
print(f"\n双框帧(M1&M2 同帧在场): {len(dual)} 个")
if dual:
    print("  示例 frame_id:", sorted(dual)[:8], "..." if len(dual) > 8 else "")

# 覆盖率（以位移记录帧为分母）
for pid in ("M1", "M2"):
    cov = pid_count.get(pid, 0)
    print(f"  {pid} 覆盖记录数: {cov} / 总帧 {n_frames} = {100.0*cov/n_frames:.1f}%")

# 判定：M2 覆盖 >= 200 视为无退步（先验 210/297）
m2 = pid_count.get("M2", 0)
ok = m2 >= 200 and len(dual) >= 1
print("\n==== 判定:", "REGRESSION OK" if ok else "CHECK NEEDED", "====")
sys.exit(0 if ok else 1)
