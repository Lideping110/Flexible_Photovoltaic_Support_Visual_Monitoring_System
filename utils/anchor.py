"""持久 ID 锚点匹配层。

ByteTrack 的原始 ID 只作"临时身份"，对外的持久 ID（P1/P2/… 或 M1/M2/…）
由本层依据初始帧锚点（检测框 + 初始数据）指派：

- 所有权延续：锚点被某条活跃轨迹占用时，新轨迹不许抢；
  占用轨迹连续丢失超过 release_after_s 秒后锚点才释放。
- 匹配规则：新轨迹框 vs 各空置锚点框，代价 1−IoU 匈牙利全局最优指派；
  IoU >= iou_min 接受；兜底条件为中心距 <= 锚点框短边。
- 锚点框 EMA 缓慢跟随占用轨迹的检测框（缓慢形变免疫，快移不跟）。

依赖"相机固定"这一既有硬性约束（见 pole_tilt_monitor/README）。
"""
import numpy as np


def iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def center(box):
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], float)


def short_side(box) -> float:
    return max(1.0, min(float(box[2] - box[0]), float(box[3] - box[1])))


def _hungarian(cost: np.ndarray):
    """返回 [(row, col, cost)]；scipy 可用时用匈牙利，否则贪心。"""
    if cost.size == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(cost)
        return [(int(r), int(c), float(cost[r, c])) for r, c in zip(rows, cols)]
    except Exception:
        # 贪心回退：每次取全局最小代价
        cost = cost.copy()
        pairs = []
        rows, cols = cost.shape
        used_r, used_c = set(), set()
        order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
        for r, c in order:
            if int(r) in used_r or int(c) in used_c:
                continue
            used_r.add(int(r))
            used_c.add(int(c))
            pairs.append((int(r), int(c), float(cost[r, c])))
            if len(used_r) == rows or len(used_c) == cols:
                break
        return pairs


class AnchorRegistry:
    """一个摄像头一个实例；锚点可来自首帧（首次运行）或 init_state.json（重启）。"""

    def __init__(self, cam_id, branch, iou_min=0.3, release_after_s=10.0,
                 ema_alpha=0.05, prefix="P"):
        self.cam_id = cam_id
        self.branch = branch
        self.iou_min = float(iou_min)
        self.release_after_s = float(release_after_s)
        self.ema_alpha = float(ema_alpha)
        self.prefix = prefix
        # pid -> {"box": np(4), "owner": raw_id|None, "last_seen": ts,
        #         "baseline": Any, "quality": float|None}
        self.anchors: dict = {}
        self.loaded_from_disk = False
        self.initialized = False

    # ---------- 初始化（首帧或重启加载） ----------

    def ensure_init(self, detections, ts, boxes_only=False):
        """首帧带检测时建立锚点（首次运行）。已初始化则直接返回。

        detections: [{"box": [...], "conf": float, ...}]（按检测框 x 中心排序编号）
        """
        if self.initialized or not detections:
            return
        ordered = sorted(detections, key=lambda d: ((d["box"][0] + d["box"][2]) / 2.0))
        for i, det in enumerate(ordered):
            pid = f"{self.prefix}{i + 1}"
            self.anchors[pid] = {
                "box": np.asarray(det["box"], dtype=np.float64),
                "owner": None,
                "last_seen": float(ts),
                "baseline": None,
                "quality": None,
            }
        self.initialized = True

    def load_anchors(self, targets: list, ts: float) -> None:
        """从 init_state.json 恢复锚点（项目重启）。"""
        for t in targets:
            self.anchors[t["pid"]] = {
                "box": np.asarray(t["box"], dtype=np.float64),
                "owner": None,
                "last_seen": float(ts),
                "baseline": t.get("baseline"),
                "quality": t.get("quality"),
            }
        self.initialized = True
        self.loaded_from_disk = True

    # ---------- 每帧指派 ----------

    def assign(self, tracks, ts) -> dict:
        """把本帧轨迹（raw id）指派到持久 ID。

        tracks: [{"id": raw_id, "box": [...], "conf": ...}]
        返回 {raw_id: pid}；未被任何锚点接受的轨迹不在结果里。
        """
        if not self.initialized:
            return {}
        mapping = {}
        present_ids = {t["id"] for t in tracks}

        # 1) 所有权延续：已是锚点 owner 的轨迹直接保留
        claimed = set()
        for pid, anchor in self.anchors.items():
            if anchor["owner"] is not None and anchor["owner"] in present_ids:
                mapping[anchor["owner"]] = pid
                claimed.add(pid)

        # 2) 释放超时空置的锚点（占用者长期不在场）
        for pid, anchor in self.anchors.items():
            if (anchor["owner"] is not None
                    and anchor["owner"] not in present_ids
                    and ts - anchor["last_seen"] > self.release_after_s):
                anchor["owner"] = None

        # 3) 剩余轨迹 vs 空置锚点：匈牙利指派
        free_pids = [pid for pid, a in self.anchors.items()
                     if pid not in claimed and a["owner"] is None]
        rest = [t for t in tracks if t["id"] not in mapping]
        if free_pids and rest:
            cost = np.ones((len(rest), len(free_pids)), dtype=np.float64)
            for i, t in enumerate(rest):
                for j, pid in enumerate(free_pids):
                    cost[i, j] = 1.0 - iou(t["box"], self.anchors[pid]["box"])
            for i, j, c in _hungarian(cost):
                t, pid = rest[i], free_pids[j]
                anchor = self.anchors[pid]
                # 接受条件：IoU 达标，或中心距不超过锚点框短边（兜底）
                ok = (1.0 - c) >= self.iou_min or (
                    float(np.linalg.norm(center(t["box"]) - center(anchor["box"])))
                    <= short_side(anchor["box"])
                )
                if ok:
                    mapping[t["id"]] = pid
                    anchor["owner"] = t["id"]
                    anchor["last_seen"] = float(ts)
                    self._ema_follow(pid, t["box"])

        # 4) 刷新存活锚点的 last_seen
        for raw_id, pid in mapping.items():
            anchor = self.anchors[pid]
            anchor["last_seen"] = float(ts)
            track = next(t for t in tracks if t["id"] == raw_id)
            self._ema_follow(pid, track["box"])
        return mapping

    def _ema_follow(self, pid, box) -> None:
        anchor = self.anchors[pid]
        if anchor["owner"] is None:
            return
        a = self.ema_alpha
        anchor["box"] = a * np.asarray(box, dtype=np.float64) + (1 - a) * anchor["box"]

    # ---------- 初始数据 ----------

    def set_baseline(self, pid, baseline, quality=None) -> None:
        if pid in self.anchors:
            self.anchors[pid]["baseline"] = baseline
            self.anchors[pid]["quality"] = quality

    def get_baseline(self, pid):
        anchor = self.anchors.get(pid)
        return None if anchor is None else anchor.get("baseline")

    def to_targets(self) -> list:
        """导出为 init_state.json 的 targets 列表。"""
        out = []
        for pid in sorted(self.anchors, key=lambda p: int(p[1:]) if p[1:].isdigit() else 0):
            a = self.anchors[pid]
            out.append({
                "pid": pid,
                "box": [float(v) for v in a["box"]],
                "baseline": a.get("baseline"),
                "quality": a.get("quality"),
            })
        return out
