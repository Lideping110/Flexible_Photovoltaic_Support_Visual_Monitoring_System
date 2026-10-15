"""标注视频导出：把 JSONL 结果回贴到源视频，输出浏览器可播的 H.264 标注视频。

供两处复用（同一套渲染逻辑，产物目录不同）：
- supervisor  （main 跑完自动生成 → <output>/<branch>/<cam_id>/<cam_id>_annotated.mp4）
- viz_overlay  （离线手动 → <output>/viz/<cam_id>_annotated.mp4）

只消费 output/ + 源视频，不触碰推理主链路；实时流源（rtsp/http…）无本地文件，
明确跳过并提示先录制成文件。
"""
import os
import shutil
import subprocess
import time
from pathlib import Path

from utils.common import log
from .readers import group_by_frame, load_jsonl
from .render import render_annotated_video

JSONL_KIND = {"pole": "measurements", "marker": "displacement"}


def _to_h264(path: Path, attempts: int = 3) -> bool:
    """把 OpenCV 写的 mp4v 标注视频转成浏览器可播的 H.264 + faststart。

    OpenCV 的 VideoWriter 默认 mp4v（MPEG-4 Part 2），现代浏览器 <video> 不解码；
    用 ffmpeg 二次封装为 libx264 + faststart。
    ffmpeg 偶发在进程启动压力下 fork 失败 / 临时文件被占用 → 重试若干次，
    全部失败才返回 False（由调用方标记转码失败，绝不把不可播 mp4v 当成成功）。
    ffmpeg 本身不可用则直接失败（重试无意义）。
    """
    if shutil.which("ffmpeg") is None:
        log("export", "ffmpeg 不可用，无法转码为浏览器可播的 H.264（保留不可播的 mp4v）")
        return False
    last_err = None
    for attempt in range(1, attempts + 1):
        tmp = path.with_suffix(f".h264.tmp{attempt}.mp4")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(path), "-c:v", "libx264",
                 "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", "-an", str(tmp)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # 原子替换；若目标正被浏览器读取（WinError 5）则稍等重试替换
            try:
                os.replace(str(tmp), str(path))
            except OSError:
                time.sleep(0.5)
                os.replace(str(tmp), str(path))
            return True
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if tmp.is_file():
                try:
                    tmp.unlink()
                except Exception:
                    pass
            if attempt < attempts:
                time.sleep(0.4 * attempt)  # 退避，错开进程启动压力
    log("export", f"_to_h264 重试 {attempts} 次仍失败: {last_err}")
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
    h264 = _to_h264(out_path)
    stats["h264"] = h264
    fail_marker = out_dir / ".export_failed"
    if h264:
        stats["status"] = "ok"
        # 本轮成功，清除可能存在的失败标记
        try:
            fail_marker.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass
    else:
        # 绝不把不可播的 mp4v（MPEG-4 Part2）当成成功：大屏据此显示"转码失败"
        # 而非静默喂给浏览器一个解码不了的 video。
        stats["status"] = "failed"
        stats["reason"] = ("转码为浏览器可播的 H.264 失败：浏览器无法解码 MPEG-4 "
                           "Part2，请用本地播放器查看或重跑本相机")
        try:
            fail_marker.write_text(stats["reason"], encoding="utf-8")
        except Exception:
            pass
        log("export", f"{cam['id']} 标注视频转码失败（mp4v 不可播），已写 .export_failed")
    return stats
