"""JSONL 读取与按帧分组。"""
import json
from collections import defaultdict
from pathlib import Path


def load_jsonl(path) -> list:
    """逐行解析 JSONL，跳过空行与坏行。"""
    rows = []
    p = Path(path)
    if not p.is_file():
        return rows
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def group_by_frame(rows: list) -> dict:
    """按 frame_id 分组，每组内按 pid 稳定排序。

    返回 {frame_id(int): [record, ...]}
    """
    d = defaultdict(list)
    for r in rows:
        fid = r.get("frame_id")
        if fid is None:
            continue
        d[int(fid)].append(r)
    for fid in d:
        d[fid].sort(key=lambda r: str(r.get("pid")))
    return dict(d)
