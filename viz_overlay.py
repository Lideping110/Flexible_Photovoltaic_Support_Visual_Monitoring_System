"""方案 A：离线标注视频生成器（结果回贴原始视频）。

用法：
    uv run python viz_overlay.py                      # 处理 config.yaml 全部摄像头
    uv run python viz_overlay.py --cam pole_cam_01    # 只处理某摄像头
    uv run python viz_overlay.py --dump-frames 6      # 额外 dump 6 张标注帧 jpg

特性：
- 自动从 config.yaml 发现 pole / marker 摄像头及其源视频与标定。
- pole(image_relative) 分支渲染前对每帧做同参数去畸变，保证 box 严丝合缝。
- marker 分支直接叠原始帧。
- 产物：<output_dir>/viz/<cam_id>_annotated.mp4
         <output_dir>/viz/frames/<cam_id>/frame_XXXXXX.jpg（视觉 QA）
- 只消费 output/ + 源视频，不触碰推理主链路。
"""
import argparse
from pathlib import Path

from viz.config_io import load_viz_config
from viz.readers import group_by_frame, load_jsonl
from viz.render import (render_annotated_video, load_init_baselines,
                        render_baseline_image)

JSONL_KIND = {"pole": "measurements", "marker": "displacement"}


def _run_cam(cam, output_dir, pole_mode, dump_frames):
    branch = cam["task"]
    kind = JSONL_KIND.get(branch)
    if kind is None:
        return {"cam": cam["id"], "status": "skip", "reason": f"未知分支 {branch}"}

    src = Path(cam["url"])
    if not src.exists():
        # RTSP 等实时源不做帧级回放（frame_id 对齐需现场抓取），明确跳过
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

    # 已建基准杆数（顶部 HUD 用，对齐摄像头拉流 "Baseline poles: N"）
    pids = set()
    for recs in frame_map.values():
        for r in recs:
            pid = r.get("pid")
            if pid is not None:
                pids.add(pid)
    baseline_poles = len(pids)
    baseline_frame_id = min(frame_map) if frame_map else 0

    out_dir = Path(output_dir) / "viz"
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

    # 统一基线参考图（对齐摄像头拉流 save_relative_baseline_images）
    if branch == "pole" and pole_mode == "image_relative":
        baselines = load_init_baselines(output_dir, cam["id"])
        if baselines:
            bl_path = out_dir / f"{cam['id']}_baseline_frame_{baseline_frame_id}.jpg"
            saved = render_baseline_image(
                src, branch, cam.get("calibration"), baselines,
                bl_path, frame_id=baseline_frame_id)
            if saved:
                stats["baseline_image"] = saved
            # 同步统一渲染到持久化的 init_frame.jpg：用 init_state 的全部基线
            # 重绘首帧，规避"首帧落盘时其余杆基线尚未建立"的时序缺漏。
            init_path = (Path(output_dir) / "pole" / cam["id"]
                         / "init" / "init_frame.jpg")
            saved_init = render_baseline_image(
                src, branch, cam.get("calibration"), baselines,
                init_path, frame_id=baseline_frame_id)
            if saved_init:
                stats["init_frame_unified"] = saved_init
    return stats


def main():
    ap = argparse.ArgumentParser(description="方案A：结果回贴原始视频标注生成器")
    ap.add_argument("--config", default="config.yaml", help="统一配置文件路径")
    ap.add_argument("--cam", default=None, help="只处理指定摄像头 id（默认全部）")
    ap.add_argument("--dump-frames", type=int, default=4,
                    help="额外 dump 的标注帧数量（默认 4，0=不 dump）")
    args = ap.parse_args()

    cfg = load_viz_config(args.config)
    cams = cfg["cameras"]
    if args.cam:
        cams = [c for c in cams if c["id"] == args.cam]
        if not cams:
            print(f"[错误] 未找到摄像头 id={args.cam}")
            return 1

    print(f"配置文件: {Path(args.config).resolve()}")
    print(f"输出目录: {Path(cfg['output_dir'])/'viz'}")
    print(f"pole 测量模式: {cfg['pole_mode']} "
          f"(image_relative → 渲染前对每帧去畸变)")
    print("-" * 64)

    for cam in cams:
        st = _run_cam(cam, cfg["output_dir"], cfg["pole_mode"], args.dump_frames)
        if st["status"] == "ok":
            print(f"[{cam['task']:6}] {st['cam']:<14} ✅ 标注视频: {st['out']}")
            print(f"        源帧数={st['video_src_frames']}  "
                  f"已标注帧={st['frames_annotated']}  "
                  f"覆盖帧号 {st['first_annotated']}..{st['last_annotated']}  "
                  f"去畸变={st['undistorted']}  fps={st['src_fps']}  "
                  f"基准杆数={st.get('baseline_poles', 0)}")
            if st.get("dump_frames"):
                print(f"        QA 帧: {len(st['dump_frames'])} 张 → "
                      f"{Path(st['dump_frames'][0]).parent}")
            if st.get("baseline_image"):
                print(f"        基线参考图: {st['baseline_image']}")
            if st.get("init_frame_unified"):
                print(f"        统一 init_frame: {st['init_frame_unified']}")
        else:
            print(f"[{cam['task']:6}] {st['cam']:<14} ⚠️  跳过: {st['reason']}")

    print("-" * 64)
    print("完成。用播放器打开 <output_dir>/viz/<cam_id>_annotated.mp4 查看。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
