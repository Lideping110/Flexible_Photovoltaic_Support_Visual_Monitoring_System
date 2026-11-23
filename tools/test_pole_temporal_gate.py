"""回归测试：pole 时间稳定性门控 = 锚固端（bottom）一阶差分 median。

背景（2026-11-20 实测 bug，pole_cam_01 / heng.mp4 82 帧）：
    旧版门控用"两端点绝对位置在 10 帧窗口内的 std(spread)"一起惩罚，
    杆真实倾斜（top 快速移动）与 bottom 检测乱跳混在同一 spread 里，
    导致倾角从 1.7° 涨到 5°+ 后整段 INVALID——监测系统在最该出数的
    时刻把数据丢了（ts19-25 全 INVALID，delta 全 NaN）。

新版行为（bottom_jitter_score）：
    1) top 的移动不参与时间惩罚（被测量本身，单帧几何质量照旧把关）
    2) bottom（锚固端）帧间移动中位步长 > tol 才扣分
    3) 单帧尖峰被 median 忽略，不会像 std 那样污染整个 10 帧窗口

跑法（项目根目录）：
    .venv/Scripts/python.exe tools/test_pole_temporal_gate.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pole_tilt_monitor.monitor import bottom_jitter_score  # noqa: E402

TOL, SCALE = 2.0, 4.0


def _seq(points):
    return [np.asarray(p, dtype=float) for p in points]


def main() -> int:
    failures = []

    # 场景1：bottom 稳定（亚像素抖动）→ 不惩罚（真实倾斜时 bottom 就该这样）
    hist = _seq([(620.0, 969.0 + 0.3 * i % 2 - 0.5) for i in range(10)])
    s, j = bottom_jitter_score(hist, TOL, SCALE)
    if s < 0.99:
        failures.append(f"稳定序列被惩罚: score={s:.3f} jitter={j:.2f}")

    # 场景2：bottom 持续乱跳（每帧 ~13px，实测 ts26-45 段）→ 重罚
    rng = np.random.RandomState(7)
    hist = _seq([(620.0 + 13.0 * (-1) ** i, 969.0 + rng.uniform(0, 5))
                 for i in range(10)])
    s, j = bottom_jitter_score(hist, TOL, SCALE)
    if s > 0.10 or j < 10.0:
        failures.append(f"乱跳序列未重罚: score={s:.3f} jitter={j:.2f}")

    # 场景3：单帧尖峰（一步 40px，其余 ~0.5px）→ median 忽略，不惩罚
    #        （旧版 std 会被这一个尖峰拉高整个窗口 → 长期误伤）
    hist = _seq([(620.0, 969.0)] * 5 + [(620.0, 1009.0)] + [(620.0, 969.0)] * 4)
    s, j = bottom_jitter_score(hist, TOL, SCALE)
    if s < 0.99:
        failures.append(f"单帧尖峰未被 median 忽略: score={s:.3f} jitter={j:.2f}")

    # 场景4：历史不足 3 帧 → 不惩罚
    s, j = bottom_jitter_score(_seq([(1.0, 1.0), (2.0, 2.0)]), TOL, SCALE)
    if s != 1.0:
        failures.append(f"历史不足仍惩罚: score={s}")

    # 场景5：回放实测——真实倾斜段（ts10-25 完整 10 帧窗口：前段 bottom
    # 稳定，后段 top 台阶倾斜 + bottom 偶发尖峰）。窗口内尖峰只占少数，
    # median 稀释后真实倾斜帧恢复可用（这正是 2026-11-20 bug 的案发现场）。
    replay_bottom = [
        (621.1, 969.0), (621.1, 968.9), (621.1, 967.9), (621.1, 969.0),
        (621.4, 969.5), (621.0, 969.9), (621.0, 969.5), (621.3, 969.4),
        (620.8, 969.9),  # ts10-18 稳定段
        (622.8, 969.7), (619.7, 1003.8), (623.0, 969.7), (622.9, 969.2),
        (622.8, 969.2), (621.6, 982.2), (619.3, 999.3),
    ]  # pole_cam_01 measurements.jsonl ts10-25 的 bottom 实测值
    s, j = bottom_jitter_score(_seq(replay_bottom), TOL, SCALE)
    if s < 0.9:
        failures.append(f"实测真实倾斜段仍被惩罚: score={s:.3f} jitter={j:.2f}")

    # 场景6：实测持续乱跳段（ts26-33 摘录，步长 6~8px 为主 + 尖峰）→
    # 压低 bottom_stability 后足以把 quality 拉破 0.60（端到端模拟：
    # 稳定期典型 quality=0.65、bottom_stability=0.9、归一化权重 0.152）
    replay_bad = [
        (626.8, 964.7), (626.8, 961.1), (627.3, 954.1), (620.3, 1002.2),
        (621.1, 996.1), (627.2, 957.6), (628.1, 949.6), (628.3, 942.0),
    ]
    s, j = bottom_jitter_score(_seq(replay_bad), TOL, SCALE)
    quality = 0.65 - 0.152 * (0.9 - min(0.9, s))
    if quality >= 0.60:
        failures.append(f"实测乱跳段未能翻转 INVALID: score={s:.3f} "
                        f"jitter={j:.2f} quality={quality:.3f}")

    if failures:
        print("FAIL")
        for f in failures:
            print("  -", f)
        return 1
    print("PASS: 锚固端门控 6 场景全部符合预期（真实倾斜放过 / "
          "bottom 乱跳重罚 / 单帧尖峰免疫）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
