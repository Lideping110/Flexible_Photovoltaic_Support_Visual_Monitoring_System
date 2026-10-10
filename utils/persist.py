"""初始基准持久化：init_state.json / init_frame.jpg 的原子读写。

存储位置：{output_dir}/{branch}/{cam_id}/init/
- init_frame.jpg  首张成功帧（带锚点标注）
- init_state.json 每个目标（立柱/靶标）的持久 ID、检测框与初始数据

原子写：先写 tmp 再 os.replace，崩溃时不会留下半写文件。
"""
import json
import os
import time
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
    # Windows 上 os.replace 可能被杀毒/索引服务对新建文件的瞬时锁打断
    #（实测多进程流水线中偶发 WinError 5，单进程压测 300 次零失败）。
    # 原子替换遇 PermissionError 短退避重试是 Windows 下的标准做法。
    delay = 0.02
    for attempt in range(6):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(delay)
            delay *= 2


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
    """把锚点标注到首帧并保存为 init_frame.jpg（pole / marker 共用）。

    pole 分支：若锚点已建立基线，则复刻基线参考图风格绘制（调色板中线 +
    绿顶/红底端点圆 + `Pole Px angle=.. Q=..` 标签），与 viz 基线图、实时
    监测画面一致；基线未就绪时退化为仅画框 + pid 文字。
    """
    import cv2

    vis = frame.copy()
    palette = [(255, 255, 0), (255, 0, 255), (0, 255, 255),
               (255, 128, 0), (128, 255, 0), (0, 128, 255)]
    for i, (pid, a) in enumerate(anchor_boxes.items()):
        box = a.get("box")
        # 注意：marker 分支传入的 box 是 numpy 数组，不能用 `if not box` 判空
        #（多元素数组真值歧义会抛 ValueError），必须显式判 None + 长度。
        if box is None or len(box) < 4:
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 2)

        bl = a.get("baseline")
        values = bl.get("values") if isinstance(bl, dict) else bl
        if values and len(values) >= 5:
            top = (int(round(values[0])), int(round(values[1])))
            bottom = (int(round(values[2])), int(round(values[3])))
            angle = float(values[4])
            color = palette[i % len(palette)]
            cv2.line(vis, bottom, top, color, 3)
            cv2.circle(vis, top, 7, (0, 255, 0), -1)
            cv2.circle(vis, bottom, 7, (0, 0, 255), -1)
            q = a.get("quality")
            label = f"Pole {pid}  angle={angle:+.4f} deg  Q={q:.2f}"
            label_at = (max(5, top[0] - 35), max(70, top[1] - 12))
            cv2.putText(vis, label, label_at, cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 4)
            cv2.putText(vis, label, label_at, cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, color, 2)
        else:
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
