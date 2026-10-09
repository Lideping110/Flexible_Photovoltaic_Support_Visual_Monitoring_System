"""初始基准持久化：init_state.json / init_frame.jpg 的原子读写。

存储位置：{output_dir}/{branch}/{cam_id}/init/
- init_frame.jpg  首张成功帧（带锚点标注）
- init_state.json 每个目标（立柱/靶标）的持久 ID、检测框与初始数据

原子写：先写 tmp 再 os.replace，崩溃时不会留下半写文件。
"""
import json
import os
from datetime import datetime
from pathlib import Path

STATE_VERSION = 1


def init_paths(output_dir, branch: str, cam_id: str):
    """返回 (init_dir, init_state.json 路径, init_frame.jpg 路径)。"""
    init_dir = Path(output_dir) / branch / cam_id / "init"
    init_dir.mkdir(parents=True, exist_ok=True)
    return init_dir, init_dir / "init_state.json", init_dir / "init_frame.jpg"


def atomic_write_json(path, obj: dict) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_json(path):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def save_anchor_frame(frame, anchor_boxes: dict, frame_path) -> None:
    """把锚点框标注到首帧并保存为 init_frame.jpg（pole / marker 共用）。"""
    import cv2

    vis = frame.copy()
    for pid, a in anchor_boxes.items():
        x1, y1, x2, y2 = (int(round(v)) for v in a["box"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(vis, pid, (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(vis, pid, (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imwrite(str(frame_path), vis)


def build_init_state(cam_id: str, branch: str, targets: list) -> dict:
    """构造 init_state.json 内容。

    targets: [{pid, box:[x1,y1,x2,y2], baseline: dict|list|None, quality: float|None}]
    """
    return {
        "version": STATE_VERSION,
        "cam_id": cam_id,
        "branch": branch,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "init_frame": "init_frame.jpg",
        "targets": targets,
    }


def write_calibration_yaml(path, calibration: dict) -> Path:
    """把内联标定参数落盘为 camera.yaml（供 pole monitor 原构造函数复用）。"""
    import yaml

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "K": calibration["K"],
                "dist": calibration.get("dist", []),
                "R": calibration.get(
                    "R", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                ),
                "t": calibration.get("t", [0.0, 0.0, 5.0]),
            },
            f,
            allow_unicode=True,
        )
    return path
