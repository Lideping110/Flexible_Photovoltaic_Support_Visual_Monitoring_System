"""标注视频渲染：顺序遍历原始视频，按 frame_id 把 JSONL 结果回贴。

设计要点：
- 顺序遍历（不用 seek），避免 H.264 跳帧漂移，保证 frame_id 与源帧严格对应。
- pole 分支若处于 image_relative 模式，渲染前对每一帧做与 capture 完全一致的
  cv2.undistort（同 K/dist），使 box/top/bottom（去畸变坐标）严丝合缝。
- marker 分支不去做畸变，直接叠加原始帧。
- 叠加文字只用 ASCII / 数字（cv2 字体不支持中文，避免乱码框）。
- 可选 dump：把前 N 个有标注的帧另存 jpg，供人工视觉 QA。

渲染对齐（与 D:\\摄像头拉流 pole_tilt_monitor 实时输出像素级一致）：
- pole 逐帧渲染复刻 vision.draw_measurement：白框、绿顶/红底端点圆、青色中线、
  白字 "ID..Q..OK LR..FB..T..deg"（INVALID 仅画橙色文字），ALARM 红字，顶部 HUD。
- 额外生成基线参考图（复刻 main.save_relative_baseline_images）：回读代表帧、
  去畸变、绘各杆中位基线。
"""
from pathlib import Path

import cv2
import json
import numpy as np

LINE_AA = cv2.LINE_AA
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _make_undistort(calibration: dict):
    """返回 undistort(frame)->frame，无标定则返回 None。"""
    if not calibration:
        return None
    K = np.asarray(calibration["K"], dtype=np.float64)
    dist = np.asarray(calibration.get("dist", []), dtype=np.float64).reshape(-1, 1)
    if dist.size == 0:
        return None

    def _f(frame):
        try:
            return cv2.undistort(frame, K, dist)
        except cv2.error:
            return frame

    return _f


def _clip(pt, w, h):
    x = max(0, min(w - 1, int(round(pt[0]))))
    y = max(0, min(h - 1, int(round(pt[1]))))
    return x, y


def _fmt_deg(v):
    """格式化角度/位移，nan 安全（对齐摄像头拉流 draw_measurement 的 :+.3f）。"""
    try:
        if v is None or (isinstance(v, float) and v != v):  # NaN
            return "nan"
        return f"{float(v):+.3f}"
    except (TypeError, ValueError):
        return "nan"


def _draw_pole(frame, rec, w, h):
    """pole 逐帧渲染——对齐摄像头拉流 vision.draw_measurement 的像素风格。

    - 白框；绿顶/红底端点圆；青色中心线；
    - 白字标签 "ID..Q..OK LR..FB..T..deg"，INVALID 仅画 "..INVALID" 文字；
    - ALARM 红字（独立绘制，对齐 main.py）。
    """
    box = rec.get("box")
    if not box:
        return
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)  # 白框

    top = rec.get("top")
    bottom = rec.get("bottom")
    if top and bottom and len(top) >= 2 and len(bottom) >= 2:
        t = (int(round(top[0])), int(round(top[1])))
        b = (int(round(bottom[0])), int(round(bottom[1])))
        cv2.circle(frame, t, 5, (0, 255, 0), -1)    # 绿顶
        cv2.circle(frame, b, 5, (0, 0, 255), -1)    # 红底
        cv2.line(frame, t, b, (255, 255, 0), 2)     # 青中线

    pid = rec.get("pid", "?")
    q = rec.get("quality")
    status = rec.get("status")
    if status == "INVALID":
        text = f"ID {pid} Q {q if q is not None else 0.0:.2f} INVALID"
    else:
        text = (f"ID {pid} Q {q:.2f} OK "
                f"LR {_fmt_deg(rec.get('delta_lr_deg'))} "
                f"FB {_fmt_deg(rec.get('delta_fb_deg'))} "
                f"T {_fmt_deg(rec.get('delta_total_deg'))} deg")
    cv2.putText(frame, text, (x1, max(20, y1 - 8)),
                FONT, 0.52, (255, 255, 255), 2)

    if status == "INVALID":
        # 对齐摄像头拉流 main.py：INVALID 额外画橙色文字（与 ALARM 同位、互斥）
        cv2.putText(frame, "INVALID", (x1, max(40, y1 - 28)),
                    FONT, 0.7, (0, 165, 255), 2)
    elif bool(rec.get("alarm")):
        cv2.putText(frame, "ALARM", (x1, max(40, y1 - 28)),
                    FONT, 0.8, (0, 0, 255), 2)


def _draw_marker(frame, rec, w, h):
    box = rec.get("box")
    if not box:
        return
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    color = (0, 255, 255)  # 黄色框
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    cx, cy = rec.get("cx"), rec.get("cy")
    if cx is not None and cy is not None:
        px, py = _clip((cx, cy), w, h)
        cv2.drawMarker(frame, (px, py), (0, 0, 255),  # 红色十字（BGR）
                       cv2.MARKER_CROSS, 12, 2)

    pid = rec.get("pid", "?")
    raw = rec.get("raw_track_id")          # ByteTrack 原始跟踪 id（底层会跳变）
    cdx = rec.get("cum_dx_mm")
    cdy = rec.get("cum_dy_mm")
    mmpp = rec.get("mm_per_px")
    parts = [pid]
    if raw is not None:
        parts.append(f"raw = {raw}")       # 例: M2 raw = 1
    if cdx is not None and cdy is not None:
        parts.append(f"d=({cdx:+.1f},{cdy:+.1f})mm")
    if mmpp is not None:
        parts.append(f"{mmpp:.3f}mm/px")
    txt = " ".join(parts)
    cv2.putText(frame, txt, (x1, max(16, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
    cv2.putText(frame, txt, (x1, max(16, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def _draw_hud(frame, frame_id, baseline_poles):
    """顶部 HUD——对齐摄像头拉流 main.py 的 'Frame X  Baseline poles: N'。"""
    cv2.putText(frame, f"Frame {frame_id}  Baseline poles: {baseline_poles}",
                (20, 30), FONT, 0.7, (0, 0, 0), 4, LINE_AA)
    cv2.putText(frame, f"Frame {frame_id}  Baseline poles: {baseline_poles}",
                (20, 30), FONT, 0.7, (255, 255, 255), 2, LINE_AA)


def render_annotated_video(src_path, branch, frame_map, out_path,
                           calibration=None, undistort=False, fps=None,
                           dump_dir=None, dump_max=0, hud=True,
                           baseline_poles=0):
    """渲染标注视频。

    参数:
        src_path   原始视频（本地 mp4；RTSP 暂不支持，需先录制成文件）
        branch     "pole" | "marker"
        frame_map  {frame_id: [record,...]}
        out_path   输出 mp4 路径
        calibration pole 标定（K/dist），undistort=True 时必填
        undistort  True 时对每帧做 cv2.undistort（pole image_relative）
        fps        输出帧率（默认取源视频）
        dump_dir   若给定，另存前 dump_max 个有标注帧为 jpg
        dump_max   最多 dump 帧数
        baseline_poles 顶部 HUD 显示的已建基准杆数（对齐摄像头拉流）

    返回: 统计 dict
    """
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {src_path}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out_fps = fps or src_fps

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                         out_fps, (w, h))

    und = _make_undistort(calibration) if undistort else None
    if undistort and und is None:
        # 指定去畸变但无标定：退化为不去畸变，避免静默错位
        undistort = False

    dump_paths = []
    if dump_dir:
        dump_dir = Path(dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)

    fid = 0
    annotated = 0
    dumped = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if undistort:
            frame = und(frame)

        recs = frame_map.get(fid)
        n_t = len(recs) if recs else 0
        if recs:
            for r in recs:
                if branch == "pole":
                    _draw_pole(frame, r, w, h)
                else:
                    _draw_marker(frame, r, w, h)
            annotated += 1

        ts = recs[0].get("ts") if recs else fid / src_fps if src_fps else 0.0
        if hud:
            _draw_hud(frame, fid, baseline_poles)

        vw.write(frame)

        if dump_dir and recs and dumped < dump_max:
            dp = dump_dir / f"frame_{fid:06d}.jpg"
            cv2.imwrite(str(dp), frame)
            dump_paths.append(str(dp))
            dumped += 1

        fid += 1

    cap.release()
    vw.release()
    return {
        "src": str(src_path),
        "out": str(out_path),
        "src_fps": round(src_fps, 3),
        "frames_total": fid,
        "frames_annotated": annotated,
        "first_annotated": min(frame_map) if frame_map else None,
        "last_annotated": max(frame_map) if frame_map else None,
        "undistorted": bool(undistort),
        "dump_frames": dump_paths,
        "video_src_frames": n_total,
        "baseline_poles": baseline_poles,
    }


def load_init_baselines(output_dir, cam_id):
    """从本项目的 init_state.json 读取 pole 各杆基线（image_relative 模式）。

    返回 list[dict]: {pid, top:[x,y], bottom:[x,y], angle:deg, quality:float}
    与摄像头拉流 monitor.baseline 字段对齐（values=[topx,topy,botx,boty,angle]）。
    """
    p = Path(output_dir) / "pole" / cam_id / "init" / "init_state.json"
    if not p.is_file():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    out = []
    for t in data.get("targets", []) or []:
        bl = t.get("baseline") or {}
        v = bl.get("values")
        if not v or len(v) < 5:
            continue
        out.append({
            "pid": t.get("pid"),
            "top": [float(v[0]), float(v[1])],
            "bottom": [float(v[2]), float(v[3])],
            "angle": float(v[4]),
            "quality": float(t.get("quality", float("nan"))),
        })
    return out


def render_baseline_image(src_path, branch, calibration, baselines, out_path,
                          frame_id=0):
    """生成基线参考图——对齐摄像头拉流 main.save_relative_baseline_images。

    回读代表帧（默认 frame_id=0，即首有效帧）、去畸变，把各杆中位基线
    （top→bottom）以调色板色绘制，并标注杆号/角度/质量/代表帧号。
    仅 pole 分支有意义；无基线则返回 None。
    """
    if branch != "pole" or not baselines:
        return None
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
    ok, image = cap.read()
    cap.release()
    if not ok or image is None:
        return None

    if calibration:
        und = _make_undistort(calibration)
        if und is not None:
            image = und(image)

    palette = [(255, 255, 0), (255, 0, 255), (0, 255, 255),
               (255, 128, 0), (128, 255, 0), (0, 128, 255)]
    title = (f"All pole baselines  background frame {frame_id}  "
             f"N={len(baselines)}")
    cv2.putText(image, title, (30, 42), FONT, 0.8, (0, 0, 0), 4, LINE_AA)
    cv2.putText(image, title, (30, 42), FONT, 0.8, (255, 255, 255), 2, LINE_AA)

    for i, bl in enumerate(baselines):
        top = (int(round(bl["top"][0])), int(round(bl["top"][1])))
        bottom = (int(round(bl["bottom"][0])), int(round(bl["bottom"][1])))
        angle = float(bl["angle"])
        quality = float(bl.get("quality", float("nan")))
        color = palette[i % len(palette)]
        cv2.line(image, bottom, top, color, 3)
        cv2.circle(image, top, 7, (0, 255, 0), -1)
        cv2.circle(image, bottom, 7, (0, 0, 255), -1)
        label = (f"Pole {bl['pid']}  angle={angle:+.4f} deg  "
                 f"Q={quality:.2f}  rep={frame_id}")
        label_at = (max(5, top[0] - 35), max(70, top[1] - 12))
        cv2.putText(image, label, label_at, FONT, 0.55, (0, 0, 0), 4, LINE_AA)
        cv2.putText(image, label, label_at, FONT, 0.55, color, 2, LINE_AA)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if cv2.imwrite(str(out_path), image):
        return str(out_path)
    return None
