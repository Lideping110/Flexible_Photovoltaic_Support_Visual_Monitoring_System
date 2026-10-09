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
from pathlib import Path

from .common import SENTINEL, jsonable, log, now_ts


class _LatestState:
    def __init__(self):
        self.ts = None
        self.branch = None
        self.poles = {}     # pole: pid -> 最新测量
        self.targets = {}   # marker: pid -> 最新位移
        self.spectrum = {}  # marker: (pid, axis) -> 最新主频

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
            handles[key] = (d / f"{kind}.jsonl").open("a", encoding="utf-8")
        return handles[key]

    def _latest_state(cam_id, branch):
        st = latest.get(cam_id)
        if st is None:
            st = _LatestState()
            st.branch = branch
            latest[cam_id] = st
        return st

    def _flush_latest(cam_id, branch):
        st = latest[cam_id]
        path = output_dir / branch / cam_id / "latest.json"
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(st.to_dict(), f, ensure_ascii=False)
        os.replace(tmp, path)

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

            if kind == "event":
                events_f.write(json.dumps(jsonable(msg), ensure_ascii=False) + "\n")
                events_f.flush()
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
        try:
            events_f.close()
        except Exception:
            pass
        log(tag, "退出")
