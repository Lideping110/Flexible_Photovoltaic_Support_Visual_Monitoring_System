"""标注视频导出：把 JSONL 结果回贴到源视频，输出浏览器可播的 H.264 标注视频。

供两处复用（同一套渲染逻辑，产物目录不同）：
- supervisor  （main 跑完自动生成 → <output>/<branch>/<cam_id>/<cam_id>_annotated.mp4）
- viz_overlay  （离线手动 → <output>/viz/<cam_id>_annotated.mp4）

只消费 output/ + 源视频，不触碰推理主链路；实时流源（rtsp/http…）无本地文件，
明确跳过并提示先录制成文件。
"""
import os
import subprocess
from pathlib import Path

from .readers import group_by_frame, load_jsonl
from .render import render_annotated_video

JSONL_KIND = {"pole": "measurements", "marker": "displacement"}


def _to_h264(path: Path) -> bool:
    """把 OpenCV 写的 mp4v 标注视频转成浏览器可播的 H.264 + faststart。

    OpenCV 的 VideoWriter 默认 mp4v（MPEG-4 Part 2），现代浏览器 <video> 不解码；
    用 ffmpeg 二次封装为 libx264 + faststart。ffmpeg 不可用则保留 mp4v。
    """
    tmp = path.with_suffix(".h264.tmp.mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(path), "-c:v", "libx264",
             "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", "-an", str(tmp)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.replace(str(tmp), str(path))  # 原子替换
        return True
    except Exception:
        if tmp.is_file():
            try:
                tmp.unlink()
            except Exception:
                pass
        return False


def export_cam_video(cam: dict, output_dir, pole_mode: str, out_dir,
                     dump_frames: int = 0) -> dict:
    """为单个摄像头生成标注视频到 out_dir/<cam_id>_annotated.mp4。

    参数:
        cam         {"id","task","url"(绝对路径或 rtsp/http),"calibration"|None}
        output_dir  结果目录（读 {branch}/{cam_id}/{kind}.jsonl）
        pole_mode   "image_relative" → 渲染前逐帧去畸变
        out_dir     输出目录
    返回:
        {"cam","status":"ok","out",...} 或 {"cam","status":"skip","reason"}
    """
    branch = cam.get("task")
    kind = JSONL_KIND.get(branch)
    if kind is None:
        return {"cam": cam["id"], "status": "skip", "reason": f"未知分支 {branch}"}

    src = Path(cam["url"])
    if not src.exists():
        if str(cam["url"]).lower().startswith(("rtsp://", "rtmp://", "http://", "https://")):
            return {"cam": cam["id"], "status": "skip",
                    "reason": "实时流源不在线下回放模式，请先录制成文件"}
        return {"cam": cam["id"], "status": "skip", "reason": f"源视频不存在: {src}"}

    jsonl = Path(output_dir) / branch / cam["id"] / f"{kind}.jsonl"
    if not jsonl.is_file():
        return {"cam": cam["id"], "status": "skip", "reason": f"无结果文件: {jsonl}"}

    rows = load_jsonl(jsonl)
    if not rows:
        return {"cam": cam["id"], "status": "skip", "reason": "结果文件为空"}

    frame_map = group_by_frame(rows)
    undistort = (branch == "pole" and pole_mode == "image_relative")
    pids = {r.get("pid") for recs in frame_map.values()
            for r in recs if r.get("pid") is not None}
    baseline_poles = len(pids)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cam['id']}_annotated.mp4"
    dump_dir = out_dir / "frames" / cam["id"] if dump_frames > 0 else None

    stats = render_annotated_video(
        src, branch, frame_map, out_path,
        calibration=cam.get("calibration"), undistort=undistort,
        dump_dir=dump_dir, dump_max=dump_frames,
        baseline_poles=baseline_poles)
    stats["cam"] = cam["id"]
    stats["status"] = "ok"
    stats["h264"] = _to_h264(out_path)
    return stats
