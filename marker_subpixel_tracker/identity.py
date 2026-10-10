"""靶标持久 ID 注册表：靠 ByteTrack 持续跟踪维持身份，不靠框匹配。

与立柱 AnchorRegistry（utils/anchor.py）的关键区别：

- 立柱固定 → 初始框 = 画面中的固定位置先验 → 遮挡换轨后靠初始框 IoU
  匹配纠正回原 ID 可靠；
- 靶标挂在拉索上会晃动 → 初始框不能作为身份先验（晃出初始框即匹配失败）。
  靶标身份靠 ByteTrack 每帧关联的 raw id 维持（owner 延续），raw id 丢失重发
  时才用「最近已知位置」（last_box）重锁，而非初始框。

初始框 box 仍持久化存储（存档 / init_frame 标注），但不参与身份匹配。
"""
import lap
import numpy as np


def _center(box) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], dtype=np.float64)


class MarkerTrackRegistry:
    """一个摄像头一个实例：raw id → 持久 ID（M1/M2…）的持续跟踪映射。

    targets: pid -> {"box": 初始框(存档), "last_box": 最近位置(重锁用),
                     "raw_id": 当前绑定的 ByteTrack id | None,
                     "last_seen": ts, "baseline": Any, "quality": float|None}
    """

    def __init__(self, cam_id, prefix="M", release_after_s=10.0,
                 relock_max_px=200.0):
        self.cam_id = cam_id
        self.prefix = prefix
        self.release_after_s = float(release_after_s)
        # 重锁门限：未认领新检测框中心距某失主 last_box 超过此值(px)则不抢身份，
        # 视为外来/新目标跳过——挡掉 N≥4 多目标张冠李戴与跨屏乱绑。
        self.relock_max_px = float(relock_max_px)
        self.targets: dict = {}
        self.initialized = False
        self.loaded_from_disk = False

    # ---------- 初始化（首帧锁定 / 重启加载） ----------

    def ensure_init(self, dets, ts):
        """首帧锁定：检测框按 x 中心排序编号，直接绑定 raw id（不靠框匹配）。"""
        if self.initialized or not dets:
            return
        ordered = sorted(dets, key=lambda d: ((d["box"][0] + d["box"][2]) / 2.0))
        for i, det in enumerate(ordered):
            pid = f"{self.prefix}{i + 1}"
            box = np.asarray(det["box"], dtype=np.float64)
            self.targets[pid] = {
                "box": box,                          # 初始框：存档/可视化
                "last_box": box.copy(),              # 最近位置：重锁用
                "raw_id": int(det["id"]),
                "last_seen": float(ts),
                "baseline": None,
                "quality": None,
            }
        self.initialized = True

    def load_targets(self, targets: list, ts: float):
        """重启恢复：恢复初始框与 baseline，raw_id 置空等下一帧重绑定。"""
        for t in targets:
            box = np.asarray(t["box"], dtype=np.float64)
            self.targets[t["pid"]] = {
                "box": box,
                "last_box": box.copy(),
                "raw_id": None,
                "last_seen": float(ts),
                "baseline": t.get("baseline"),
                "quality": t.get("quality"),
            }
        self.initialized = True
        self.loaded_from_disk = True

    # ---------- 每帧指派（持续跟踪） ----------

    def assign(self, dets, ts) -> dict:
        """raw id → pid：owner 延续优先；丢失重发用最近位置重锁。"""
        if not self.initialized:
            return {}
        mapping = {}
        present_ids = {d["id"] for d in dets}
        det_by_id = {d["id"]: d for d in dets}

        # 1) 持续跟踪：仍在场的 raw id 直接延续，并刷新最近位置
        for pid, t in self.targets.items():
            if t["raw_id"] is not None and t["raw_id"] in present_ids:
                mapping[t["raw_id"]] = pid
                t["last_seen"] = float(ts)
                t["last_box"] = np.asarray(
                    det_by_id[t["raw_id"]]["box"], dtype=np.float64)

        # 2) 释放长期丢失的绑定
        for pid, t in self.targets.items():
            if (t["raw_id"] is not None
                    and t["raw_id"] not in present_ids
                    and ts - t["last_seen"] > self.release_after_s):
                t["raw_id"] = None

        # 3) 新 raw id → 空置/失主 pid：匈牙利最优指派 + 距离门限重锁
        #    失主 = 绑定的 raw id 本帧不在场（ByteTrack 丢失后换了新 id）。
        #    候选 = 空置 pid 或失主 pid（同池）；未认领 det = 还没拿到 pid 的 raw id。
        #    用全距离矩阵 + Hungarian（lap.lapjv，与 ByteTrack 同款求解器）求全局
        #    最优 1-1 配对，而非逐 det 贪心取最近——否则 N≥4 多目标同帧重现时，
        #    局部最近会累积成“张冠李戴”（贪心把每个新框绑到邻居）。
        #    门限实现（满方阵，自管虚拟行/列，不依赖 lap 的 extend_cost/cost_limit 黑箱）：
        #      方阵 N = n_det + n_cand；行=真实 det + n_cand 个虚拟 det；
        #      列=真实候选 + n_det 个虚拟候选。真实 det→真实候选=距离(超门限设 INF)；
        #      每个真实 det 留一列虚拟候选(代价=门限+1)；虚拟 det 只认虚拟候选(代价0)。
        #      → 合法近距配对(≤门限) 代价 < 虚拟候选，正常认领；外来/超距目标所有边
        #        为 INF，只能落虚拟候选(门限+1) → 视为未配对，worker 侧 pid is None 跳过，
        #        且绝不抢占真主人的 pid。
        candidates = [pid for pid, t in self.targets.items()
                      if t["raw_id"] is None or t["raw_id"] not in present_ids]
        rest = [d for d in dets if d["id"] not in mapping]
        if candidates and rest:
            det_centers = [_center(d["box"]) for d in rest]
            cand_centers = [_center(self.targets[pid]["last_box"]) for pid in candidates]
            limit = float(self.relock_max_px)
            INF = 1e6
            n_det = len(det_centers)
            n_cand = len(cand_centers)
            N = n_det + n_cand                       # 方阵
            cost = np.full((N, N), INF, dtype=np.float32)
            for i in range(n_det):
                for j in range(n_cand):
                    d = float(np.linalg.norm(det_centers[i] - cand_centers[j]))
                    cost[i, j] = d if d <= limit else INF
                cost[i, n_cand + i] = limit + 1.0    # 真实 det i 的虚拟候选列
            for r in range(n_det, N):                # 虚拟 det 行只认虚拟候选(0)
                for k in range(n_cand, N):
                    cost[r, k] = 0.0
            _, colsol, _ = lap.lapjv(cost)           # 满方阵纯最优指派
            colsol = np.atleast_1d(colsol)           # 方阵时 lap 可能返回标量，统一成数组
            # colsol[j] = 候选列 j 分到的 det 行；遍历候选（列），反查 det。
            for j, pid in enumerate(candidates):
                di = int(colsol[j])
                if di < 0 or di >= n_det:            # 分到虚拟 det（外来/超距）→ 不分配
                    continue
                if cost[di, j] >= INF:
                    continue
                det = rest[di]
                mapping[det["id"]] = pid
                self.targets[pid]["raw_id"] = int(det["id"])
                self.targets[pid]["last_seen"] = float(ts)
                self.targets[pid]["last_box"] = np.asarray(
                    det["box"], dtype=np.float64)

        return mapping

    # ---------- 存储 ----------

    @property
    def anchors(self) -> dict:
        """供 save_anchor_frame 标注的存档框视图 {pid: {"box": ...}}。"""
        return {pid: {"box": t["box"]} for pid, t in self.targets.items()}

    def set_baseline(self, pid, baseline, quality=None):
        if pid in self.targets:
            self.targets[pid]["baseline"] = baseline
            self.targets[pid]["quality"] = quality

    def to_targets(self) -> list:
        """导出为 init_state.json 的 targets 列表（box 为存档初始框）。"""
        out = []
        for pid in sorted(self.targets,
                          key=lambda p: int(p[1:]) if p[1:].isdigit() else 0):
            t = self.targets[pid]
            out.append({
                "pid": pid,
                "box": [float(v) for v in t["box"]],
                "baseline": t.get("baseline"),
                "quality": t.get("quality"),
            })
        return out
