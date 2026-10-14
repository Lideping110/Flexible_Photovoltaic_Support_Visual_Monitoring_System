"""Recorder 进程：全系统唯一写入者。

消费 result_q 的结构化消息，按 (branch, cam_id, kind) 分文件追加 JSONL：
- pole     → output/pole/{cam}/measurements.jsonl
- marker   → output/marker/{cam}/displacement.jsonl · spectrum.jsonl
- event    → output/events.jsonl（全局）
并维护每个摄像头的 latest.json 实时快照（供可视化低延迟轮询）。

单一写入者，无文件竞争；收到 SENTINEL（由 Supervisor 在所有生产者结束后发出）即退出。
"""
import json
import os
import time
from pathlib import Path

from .common import SENTINEL, jsonable, log, now_ts

# latest.json 快照的最小写入间隔（秒）。marker 逐帧推送时若不节流，每秒要
# os.replace 上百次：既浪费 IO，又成倍放大与"大屏每秒轮询 latest.json"的
# Windows 文件占用冲突。前端本就 1s 轮询，0.2s 的快照新鲜度完全够用。
_LATEST_MIN_INTERVAL = 0.2


def _replace_retry(tmp, path, attempts=10, delay=0.03):
    """Windows 上 os.replace 会因目标文件正被读取而报 WinError 5（拒绝访问）。

    根因：Python 的 open() 打开文件时默认不共享删除权限，只要读取方（大屏
    /api/state 每秒读 latest.json）的句柄还在，MoveFileEx 替换就会被拒。
    读取方持句柄只有毫秒级，短重试即可稳定通过（Windows 写-替换的标准做法）。
    """
    err = None
    for _ in range(attempts):
        try:
            os.replace(tmp, path)
            return True
        except PermissionError as exc:
            err = exc
            time.sleep(delay)
    raise err


class _LatestState:
    def __init__(self):
        self.ts = None
        self.branch = None
        self.poles = {}     # pole: pid -> 最新测量
        self.targets = {}   # marker: pid -> 最新位移
        self.spectrum = {}  # marker: (pid, axis) -> 最新主频
        self.last_flush = 0.0
        self.dirty = False  # 因节流被跳过、尚未落盘的最新快照

    def to_dict(self):
        return {
            "ts": self.ts,
            "branch": self.branch,
            "poles": jsonable(self.poles) if self.branch == "pole" else None,
            "targets": jsonable(self.targets) if self.branch == "marker" else None,
            "spectrum": jsonable(self.spectrum) if self.branch == "marker" else None,
        }


def run_recorder(result_q, output_dir, stop_event):
    tag = "recorder"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    handles = {}          # (branch, cam_id, kind) -> file handle
    latest = {}           # cam_id -> _LatestState
    events_path = output_dir / "events.jsonl"
    events_f = events_path.open("a", encoding="utf-8")
    log(tag, f"启动，输出目录 {output_dir}")

    def _handle(branch, cam_id, kind):
        key = (branch, cam_id, kind)
        if key not in handles:
            d = output_dir / branch / cam_id
            d.mkdir(parents=True, exist_ok=True)
            path = d / f"{kind}.jsonl"
            # 本次运行首次写入即截断（"w"）：ts 是"相对源视频/流的秒"，每次运行都从 0
            # 重新计时；若沿用 "a" 追加，同一相机的文件里会叠出多段各自从 ~0 开始的
            # 时序，采样间隔中位数被拉小 → FFT 频率轴错乱（实测把 0.06Hz 主频算成
            # 25.9Hz、索力算成千万级）。同一 run 内 handle 有缓存，后续写入仍是追加。
            handles[key] = path.open("w", encoding="utf-8")
            log(tag, f"[{branch}/{cam_id}] 重置 {kind}.jsonl（本次运行结果，覆盖上次遗留）")
        return handles[key]

    def _latest_state(cam_id, branch):
        st = latest.get(cam_id)
        if st is None:
            st = _LatestState()
            st.branch = branch
            latest[cam_id] = st
        return st

    def _flush_latest(cam_id, branch, force=False):
        st = latest.get(cam_id)
        if st is None:
            return
        now = time.monotonic()
        if not force:
            if (now - st.last_flush) < _LATEST_MIN_INTERVAL:
                st.dirty = True          # 攒着，到点或收尾时补写
                return
        path = output_dir / branch / cam_id / "latest.json"
        tmp = path.with_suffix(".json.tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(st.to_dict(), f, ensure_ascii=False)
            _replace_retry(tmp, path)
            st.last_flush = now
            st.dirty = False
        except Exception as exc:
            # 快照写失败绝不能拖垮监测链路（原先会抛出去 → recorder 退出 →
            # supervisor 判定关键进程异常 → 全系统终止）。记警告，下条消息再试。
            log(tag, f"[{branch}/{cam_id}] latest.json 写入失败，已跳过本次快照: {exc}")

    try:
        while not stop_event.is_set():
            try:
                msg = result_q.get(timeout=1.0)
            except Exception:
                continue
            if msg is SENTINEL:
                break
            branch = msg.get("branch")
            kind = msg.get("kind")
            cam_id = msg.get("cam_id")
            data = msg.get("data")
            ts = msg.get("ts")

            if kind == "frame_snapshot":
                # 推理端在每个输入帧（包括“本帧无目标”）后发布当前目标集合。
                # latest.json 只保留这一帧仍存在的目标，避免目标离开画面后
                # 永久显示最后一次测量/报警。
                st = _latest_state(cam_id, branch)
                pids = set((data or {}).get("pids") or [])
                st.ts = ts
                if branch == "pole":
                    st.poles = {pid: value for pid, value in st.poles.items()
                                if pid in pids}
                elif branch == "marker":
                    st.targets = {pid: value for pid, value in st.targets.items()
                                  if pid in pids}
                    st.spectrum = {(pid, axis): value
                                   for (pid, axis), value in st.spectrum.items()
                                   if pid in pids}
                _flush_latest(cam_id, branch)
                continue

            if kind == "event":
                events_f.write(json.dumps(jsonable(msg), ensure_ascii=False) + "\n")
                events_f.flush()
                continue

            if kind == "camera_done":
                # 该路文件源推理结束，且其全部记录都排在 camera_done 之前（result_q
                # 是 FIFO）已写完并 flush → 此刻 jsonl 已完整。完成标记由 **Recorder
                # 这一最后一环**亲自发出，是"数据确实落盘"的确定性事件，取代
                # "jsonl 不增长了"的猜测判据（无目标时段不写行会让那个判据误判）。
                try:
                    rec_d = output_dir / branch / cam_id
                    rec_d.mkdir(parents=True, exist_ok=True)
                    rec_d.joinpath(".rec_done").write_text(
                        json.dumps({"cam_id": cam_id, "ts": ts}, ensure_ascii=False),
                        encoding="utf-8")
                    log(tag, f"[{branch}/{cam_id}] 已写 .rec_done（该路结果全部落盘）")
                except Exception as exc:
                    log(tag, f"[{branch}/{cam_id}] .rec_done 写入失败: {exc}")
                # 关闭该路 jsonl 句柄：避免导出进程读取时被 Windows 文件占用挡住
                for k in [k for k in handles if k[0] == branch and k[1] == cam_id]:
                    try:
                        handles[k].close()
                    except Exception:
                        pass
                    handles.pop(k, None)
                continue

            if kind == "measurement" and branch == "pole":
                f = _handle("pole", cam_id, "measurements")
                f.write(json.dumps(jsonable({"ts": ts, **data}),
                                   ensure_ascii=False) + "\n")
                f.flush()
                st = _latest_state(cam_id, "pole")
                st.ts = ts
                st.poles[data["pid"]] = data

            elif kind == "displacement" and branch == "marker":
                f = _handle("marker", cam_id, "displacement")
                f.write(json.dumps(jsonable({"ts": ts, **data}),
                                   ensure_ascii=False) + "\n")
                f.flush()
                st = _latest_state(cam_id, "marker")
                st.ts = ts
                st.targets[data["pid"]] = data

            elif kind == "spectrum" and branch == "marker":
                f = _handle("marker", cam_id, "spectrum")
                f.write(json.dumps(jsonable({"ts": ts, **data}),
                                   ensure_ascii=False) + "\n")
                f.flush()
                st = _latest_state(cam_id, "marker")
                st.ts = ts
                st.spectrum[(data["pid"], data["axis"])] = data

            _flush_latest(cam_id, branch)

    finally:
        for f in handles.values():
            try:
                f.close()
            except Exception:
                pass
        # 收尾兜底：节流可能把最后一条快照攒着没写，这里强制补齐，
        # 否则大屏在数据静态化后会一直显示倒数第二条状态。
        for cam_id, st in list(latest.items()):
            if st.branch:
                _flush_latest(cam_id, st.branch, force=True)
        try:
            events_f.close()
        except Exception:
            pass
        log(tag, "退出")
