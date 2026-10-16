"""报警留存：把"报警发生时刻"的关键证据落盘，供大屏报警记录页回溯。

两条留存链路（对齐用户需求）：
- 立柱检测报警：保存 *原始帧*（无任何叠加）与 *重新编码帧*（画好检测框 + 拟合直线
  + ALARM 标签）；两者成对落盘到 output/alarms/pole/，并追加 manifest.jsonl 明细。
- 振动报警（靶标位移幅值超限）：原始视频片段 + 振动分析数据的留存见 viz/clip_retention
  （由 supervisor 在收尾时对文件源抽取 / 对 RTSP 实时源从滚动缓冲切窗），本模块不直接处理视频。

命名规范（按源类型区分，见用户要求）：
- 实时视频流（RTSP）：用 *当前时间* 命名 —— <当前时间>_original.jpg / <当前时间>_annotated.jpg
  （振动视频片段同理 <当前时间>_original.mp4，且需 web 可播）。
- 视频数据（文件源）：用 *frame_id*（图片）或 *起止时间*（视频）命名 —— <frame_id>_original.jpg
  / <frame_id>_annotated.jpg（振动视频 <t0>_<t1>_original.mp4，保持原编码、不转 web）。

落盘只用标准库 + cv2，不引入推理依赖；大屏侧由 dashboard.build_records 读取展示。
"""
import json
import threading
from datetime import datetime
from pathlib import Path

from .common import log

_POLE_LOCK = threading.Lock()


def _now_stamp():
    """实时流报警留存命名：YYYYMMDD_HHMMSS_ffffff（微秒防重名）。"""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def save_pole_alarm_snapshot(output_dir, cam_id, ts, frame, record, alarm_cfg,
                             cooldowns, cooldown_s=5.0, is_realtime=False,
                             frame_id=None):
    """立柱报警：落盘 原始帧(<stem>_original.jpg) + 重新编码帧(<stem>_annotated.jpg)。

    命名：实时流用当前时间，文件源用 frame_id。返回 True 表示实际写盘（受 cooldown
    节流），False 表示被节流跳过。
    - frame：BGR numpy 数组（已去畸变，与检测/测量同坐标系）。
    - record：与 measurements.jsonl 同一份测量记录（含 box/top/bottom/status/alarm）。
    重新编码帧用 viz.render._draw_pole 在副本上绘制，绝不污染原始帧。
    manifest 同时记录 snapshot(重编码) 与 original(原始) 的访问路径，供大屏展示。
    """
    ts = float(ts)
    pid = record.get("pid")

    # cooldown 节流：同一 (cam_id, pid) 最短间隔 cooldown_s 才再写，避免每秒一张刷屏
    key = (cam_id, pid)
    last = cooldowns.get(key)
    if last is not None and (ts - last) < cooldown_s:
        return False
    cooldowns[key] = ts

    alarms_dir = Path(output_dir) / "alarms" / "pole"
    alarms_dir.mkdir(parents=True, exist_ok=True)

    import cv2
    from viz.render import _draw_pole

    h, w = frame.shape[:2]
    # 命名：实时流 → 当前时间；视频数据(文件源) → frame_id
    if is_realtime:
        stem = _now_stamp()
    else:
        stem = str(int(frame_id)) if frame_id is not None else f"{ts:.3f}"
    ann_name = f"{stem}_annotated.jpg"
    raw_name = f"{stem}_original.jpg"
    ann_path = alarms_dir / ann_name
    raw_path = alarms_dir / raw_name

    # 1) 原始帧：裸存，不画任何东西
    try:
        ok_raw = bool(cv2.imwrite(str(raw_path), frame))
    except Exception as exc:
        log("alarm-store", f"立柱报警原始帧写盘失败 {cam_id}/{pid}: {exc}")
        ok_raw = False

    # 2) 重新编码帧：在 *副本* 上画框 + 拟合直线 + 标签，保留原始帧
    annotated = frame.copy()
    try:
        _draw_pole(annotated, record, w, h)
    except Exception as exc:
        log("alarm-store", f"立柱报警重编码绘制失败 {cam_id}/{pid}: {exc}")
    try:
        ok_ann = bool(cv2.imwrite(str(ann_path), annotated))
    except Exception as exc:
        log("alarm-store", f"立柱报警重编码帧写盘失败 {cam_id}/{pid}: {exc}")
        ok_ann = False

    if not (ok_raw and ok_ann):
        return False

    level = 1
    detail = (f"立柱 {pid} 倾角超限：LR={record.get('delta_lr_deg')} "
              f"FB={record.get('delta_fb_deg')} "
              f"T={record.get('delta_total_deg')} deg")
    entry = {
        "ts": ts,
        "cam_id": cam_id,
        "pid": pid,
        "level": level,
        "detail": detail,
        "source": "realtime" if is_realtime else "file",
        # 重新编码帧（画框+拟合直线）即快照；原始帧单独留存
        "snapshot": f"/alarms/pole/{ann_name}",
        "original": f"/alarms/pole/{raw_name}",
        "delta_lr_deg": record.get("delta_lr_deg"),
        "delta_fb_deg": record.get("delta_fb_deg"),
        "delta_total_deg": record.get("delta_total_deg"),
        "left_right_deg": alarm_cfg.get("left_right_deg", 2.0),
        "front_back_deg": alarm_cfg.get("front_back_deg", 2.0),
        "total_deg": alarm_cfg.get("total_deg", 2.5),
    }
    try:
        with _POLE_LOCK:
            with (alarms_dir / "manifest.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        log("alarm-store", f"manifest 写盘失败: {exc}")
    return True
