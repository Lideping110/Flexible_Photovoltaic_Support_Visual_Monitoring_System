"""纯视频靶标检测脚本（离线验证用，不启动主链路）。

只做一件事：按 config.yaml 的 models.marker_det 口径加载**靶标检测模型**，
对一段视频逐帧（或批量）跑检测，画出检测框并给出检出率 / 置信度 / 耗时统计。
不启动 supervisor、不做亚像素定位、不做 ByteTrack、不做 FFT、不写
output/<branch>/<cam>/ 下的任何主链路产物。

模型选择与推理参数沿用主链路，保证口径一致：
  utils.common.select_model_weights  → OpenVINO 目录优先，回退 .pt
  utils.inference.DetInference       → conf / iou / classes / device / rect=True / 批量退化
注意：脚本 --conf 默认 0.4（只影响本脚本，主链路仍用 config 的 0.25）；
      --iou 默认取 config 的 models.marker_det.iou=0.6。

用法（在仓库根目录执行）：
    uv run python detect_marker_video.py --source babiao                      # 整个文件夹
    uv run python detect_marker_video.py --source babiao --recursive          # 含子目录
    uv run python detect_marker_video.py --source babiao/fast_move_shorter.mp4
    uv run python detect_marker_video.py --cam marker_cam_02 --max-frames 100
    uv run python detect_marker_video.py --source babiao/static_video.mp4 --batch 8 --device cpu
    uv run python detect_marker_video.py --source babiao/static_video.mp4 --conf 0.4 --stride 2
    uv run python detect_marker_video.py --source babiao/static_video.mp4 --no-video  # 只跑统计
    uv run python detect_marker_video.py --source rtsp://... --max-frames 300 --show

输入 / 输出规则：
  - --source 传文件夹：处理其下全部视频（默认不递归子目录，--recursive 递归并镜像层级）；
    默认输出到与输入目录**同级**的 <输入目录名>_det/，例如 babiao/ → babiao_det/。
  - --source 传单个视频：默认输出到输入视频所在目录；--out 可指定完整输出路径。
  - 命名统一为 <视频原名>-det.mp4。默认只落盘这一个标注视频文件；
    检出率/置信度/耗时等统计只在控制台打印，需要逐帧记录与汇总 json 时
    加 --save-report（写 <原名>-det.jsonl 与 <原名>-det_summary.json）。
  - 多个视频共用一次模型加载；--max-frames 对每个视频分别生效。产物已存在时
    拒绝覆盖（整批事先检查），需 --overwrite。

耗时口径（summary 里三个数字不要混用）：
  - 预热用**源视频真实分辨率**，否则 rect=True 下 640x640 dummy 与实际 640x384
    形状不同，OpenVINO 会在首帧重新编译，把 1~2s 算进推理耗时；
  - --batch>1 时不同批大小（尤其尾批）会触发重编译，因此 summary 同时给
    「含首批」「稳态：剔除首批且只看主流批大小」「按批大小分组」三组数字；
  - 测单帧性能用默认 --batch 1，且让 --max-frames 足够大（>=50）再取稳态。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.common import (WORKSPACE_ROOT, jsonable, log, open_camera,
                          select_model_weights)
from utils.inference import DetInference

TAG = "marker-det"
BOX_COLOR = (0, 255, 255)   # 黄色，与 viz/render.py::_draw_marker 一致
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _resolve_source(value: str, config_dir: Path):
    """把 config 里的相对视频路径解析为绝对路径；RTSP/URL 原样返回。"""
    text = str(value).strip()
    if text.lower().startswith(("rtsp://", "rtsps://", "rtmp://",
                                "http://", "https://")) or text.isdigit():
        return text
    p = Path(text)
    if p.is_absolute():
        return p
    for base in (ROOT, config_dir):
        candidate = base / p
        if candidate.exists():
            return candidate
    return ROOT / p


def _pick_source(args, raw: dict, config_dir: Path):
    """--source > --cam > config 里第一个 marker 摄像头。"""
    cams = raw.get("cameras") or []
    if args.source:
        if args.cam:
            log(TAG, "--cam 与 --source 同时给出，以 --source 为准")
        return _resolve_source(args.source, config_dir), args.source
    if args.cam:
        for cam in cams:
            if str(cam.get("id")) == args.cam:
                if cam.get("task") != "marker":
                    log(TAG, f"警告: 摄像头 {args.cam} 的 task={cam.get('task')}，"
                             "不是 marker 源，仍按检测模型跑")
                return _resolve_source(cam["url"], config_dir), args.cam
        raise SystemExit(f"[错误] config 中未找到摄像头 id={args.cam}")
    for cam in cams:
        if cam.get("task") == "marker" and cam.get("url"):
            cid = str(cam.get("id"))
            log(TAG, f"未指定视频源，使用 config 首个 marker 摄像头: {cid}")
            return _resolve_source(cam["url"], config_dir), cid
    raise SystemExit("[错误] 未指定 --source，且 config 中没有可用的 marker 摄像头")


def _stem(source, cam_label: str) -> str:
    if isinstance(source, Path):
        return source.stem
    if cam_label and not str(cam_label).startswith(("rtsp://", "http://")):
        return str(cam_label)
    return "stream"


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v",
              ".mpg", ".mpeg", ".ts", ".webm", ".vob", ".3gp"}


def _collect_tasks(args, raw: dict, config_dir: Path, default_out_dir: Path) -> list:
    """把 --source / --cam 解析成待处理任务列表。

    文件夹模式：扫描该目录下所有视频（--recursive 时含子目录），默认输出到
    与输入目录**同级**的 `<输入目录名>_det/`，文件名 `<原名>-det.mp4`；
    单文件模式：默认输出到输入文件旁边，同名规则；--out 可显式指定完整路径。
    """
    explicit_out_dir = Path(args.out_dir).resolve() if args.out_dir else None

    if args.source:
        src = _resolve_source(args.source, config_dir)
        if isinstance(src, Path) and src.is_dir():
            if args.cam:
                raise SystemExit("[错误] 文件夹模式下不要同时给 --cam")
            if args.out:
                raise SystemExit("[错误] 文件夹模式不支持 --out（单文件输出路径），"
                                 "请用 --out-dir 指定输出根目录")
            paths = src.rglob("*") if args.recursive else src.glob("*")
            files = sorted((p for p in paths
                            if p.is_file() and p.suffix.lower() in VIDEO_EXTS),
                           key=lambda p: str(p).lower())
            if not files:
                raise SystemExit(f"[错误] 文件夹下没有视频文件: {src}"
                                 f"（识别扩展名: {', '.join(sorted(VIDEO_EXTS))}）")
            out_root = explicit_out_dir or (src.parent / f"{src.name}_det")
            log(TAG, f"文件夹模式: {src}  共 {len(files)} 个视频"
                     f"{'（含子目录）' if args.recursive else ''}"
                     f"  输出根目录: {out_root}")
            tasks = []
            for f in files:
                # 递归时保留子目录层级，避免不同子目录里的同名视频互相覆盖
                rel = f.relative_to(src).parent if args.recursive else Path(".")
                tasks.append({"source": f, "cam": f.stem,
                              "out_dir": out_root / rel, "stem": f.stem,
                              "explicit_video": None})
            return tasks

    source, cam_label = _pick_source(args, raw, config_dir)
    if isinstance(source, Path) and not source.is_file():
        raise SystemExit(f"[错误] 视频文件不存在: {source}")
    stem = _stem(source, cam_label)
    explicit_video = Path(args.out).resolve() if args.out else None
    if explicit_video is not None:
        out_dir = explicit_video.parent
    elif explicit_out_dir is not None:
        out_dir = explicit_out_dir
    elif isinstance(source, Path):
        out_dir = source.parent
    else:
        out_dir = default_out_dir   # RTSP 没有本地同级目录，回退到配置的输出目录
    return [{"source": source, "cam": cam_label, "out_dir": out_dir,
             "stem": stem, "explicit_video": explicit_video}]


def _target_paths(task: dict, args) -> tuple:
    """任务的三个产物路径。视频名统一为 <原名>-det.mp4。"""
    video = task["explicit_video"] or (task["out_dir"] /
                                       f"{task['stem']}-det.mp4").resolve()
    jsonl = video.with_name(f"{video.stem}.jsonl")
    summary = video.with_name(f"{video.stem}_summary.json")
    targets = ([] if args.no_video else [video]) + \
              ([jsonl, summary] if args.save_report else [])
    return video, jsonl, summary, targets


def _parse_boxes(result) -> list:
    """ultralytics Boxes → [{box, conf, cls}]，无检测返回空列表。"""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    data = np.asarray(boxes.data.cpu().numpy(), dtype=float)
    out = []
    for row in data:
        if len(row) < 6:
            continue
        out.append({
            "box": [round(float(v), 2) for v in row[:4]],
            "conf": round(float(row[4]), 4),
            "cls": int(row[5]),
        })
    return out


def _text(frame, txt, org, color=(255, 255, 255), scale=0.6, thickness=2):
    cv2.putText(frame, txt, org, FONT, scale, (0, 0, 0), thickness + 2,
                cv2.LINE_AA)
    cv2.putText(frame, txt, org, FONT, scale, color, thickness, cv2.LINE_AA)


def _label(frame, txt, x, y):
    """框顶标签：实心黑底 + 黄字。浅色墙面上纯描边小字会被 H.264 压到看不见。"""
    scale, thick = 0.55, 1
    (tw, th), _ = cv2.getTextSize(txt, FONT, scale, thick)
    y0 = max(th + 6, y)
    cv2.rectangle(frame, (x, y0 - th - 6), (x + tw + 10, y0 + 2), (0, 0, 0), -1)
    cv2.putText(frame, txt, (x + 5, y0 - 3), FONT, scale, BOX_COLOR, thick,
                cv2.LINE_AA)


def _draw(frame, dets: list, frame_id: int, time_s: float, infer_ms: float,
          conf_thr: float, model_name: str):
    """黄框 + 置信度标签 + 左上 HUD；漏检帧在画面中央显式提示。"""
    for d in dets:
        x1, y1, x2, y2 = (int(round(v)) for v in d["box"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
        _label(frame, f"c{d['cls']} {d['conf']:.2f}", x1, y1 - 6)
    h, w = frame.shape[:2]
    _text(frame, f"Frame {frame_id}  t={time_s:.2f}s  det={len(dets)}",
          (12, 30), (255, 255, 255), 0.65, 2)
    _text(frame, f"conf>={conf_thr:.2f}  infer={infer_ms:.1f}ms  {model_name}",
          (12, 58), (200, 255, 200), 0.55, 1)
    if not dets:
        _text(frame, "NO DETECTION", (max(12, w // 2 - 140), h // 2),
              (0, 140, 255), 1.0, 2)
    return frame


def _percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[idx]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="纯视频靶标检测：加载 marker_det 模型对视频逐帧推理，出标注视频 + 检出统计",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml", help="配置文件（默认 config.yaml）")
    ap.add_argument("--source", default=None,
                    help="视频文件 / 文件夹 / RTSP 地址。传文件夹时处理其下全部视频，"
                         "输出到同级 <目录名>_det/；省略时用 config 首个 marker 摄像头")
    ap.add_argument("--cam", default=None, help="从 config 取指定摄像头 id 的 url 作为视频源")
    ap.add_argument("--recursive", action="store_true",
                    help="文件夹模式下递归处理子目录（输出目录镜像子目录层级）")
    ap.add_argument("--weights", default=None,
                    help="覆盖模型路径（.pt 或 OpenVINO 目录）；默认按 config 自动选择")
    ap.add_argument("--device", default=None,
                    help="覆盖推理设备，如 cpu / 0 / intel:gpu（默认取 config）")
    ap.add_argument("--conf", type=float, default=0.4,
                    help="置信度阈值（默认 0.4，只影响本脚本；主链路仍取 config）")
    ap.add_argument("--iou", type=float, default=None,
                    help="NMS IoU 阈值（默认取 config 的 models.marker_det.iou）")
    ap.add_argument("--out", default=None,
                    help="单视频模式下指定标注视频完整路径（默认输出到输入文件旁边，<原名>-det.mp4）")
    ap.add_argument("--out-dir", default=None,
                    help="产物根目录（文件夹模式默认 <输入目录>_det；单文件默认输入文件所在目录）")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="每个视频最多推理多少帧，0=全部（默认 0）")
    ap.add_argument("--stride", type=int, default=1, help="每隔 N 帧检测一帧（默认 1，逐帧）")
    ap.add_argument("--batch", type=int, default=1,
                    help="每次前向的帧数，>1 走 DetInference 批量路径（默认 1）")
    ap.add_argument("--show", action="store_true", help="本机弹窗预览，按 q 退出")
    ap.add_argument("--no-video", action="store_true", help="不写标注视频，只跑检测与统计")
    ap.add_argument("--save-report", action="store_true",
                    help="额外落盘逐帧记录 jsonl 与汇总 summary.json（默认只保存标注视频）")
    ap.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的同名产物")
    ap.add_argument("--no-h264", action="store_true",
                    help="跳过 ffmpeg H.264 转码，保留 OpenCV 的 mp4v（本地播放器可播）")
    args = ap.parse_args()

    if args.stride < 1 or args.batch < 1 or args.max_frames < 0:
        raise SystemExit("[错误] --stride / --batch 必须 >=1，--max-frames 必须 >=0")

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg_path = cfg_path.resolve()
    if not cfg_path.is_file():
        raise SystemExit(f"[错误] 配置文件不存在: {cfg_path}")
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    model_cfg = dict((raw.get("models") or {}).get("marker_det") or {})
    if not model_cfg and args.weights is None:
        raise SystemExit("[错误] config 缺少 models.marker_det，且未用 --weights 指定模型")

    configured = (raw.get("runtime") or {}).get("output_dir", "output")
    base = Path(configured)
    default_out_dir = (base if base.is_absolute() else ROOT / base) / "detect"
    tasks = _collect_tasks(args, raw, cfg_path.parent, default_out_dir)

    # 模型：与 marker worker 完全相同的选择口径；多视频只加载一次
    if args.weights:
        w = Path(args.weights)
        weights = w if w.is_absolute() else ROOT / w
    else:
        weights = select_model_weights(model_cfg, cfg_path.parent)
    if not weights.exists():
        raise SystemExit(f"[错误] 模型路径不存在: {weights}")

    device = args.device or model_cfg.get("device", "cpu")
    conf = float(args.conf if args.conf is not None
                 else model_cfg.get("confidence", 0.25))
    iou = float(args.iou if args.iou is not None
                else model_cfg.get("iou", 0.7))
    cls_id = int(model_cfg.get("class_id", 0))
    tracker = model_cfg.get("tracker", "bytetrack.yaml")
    # .pt 走 PyTorch 后端，不认 intel:gpu / intel:cpu 这类 OpenVINO 设备串
    if weights.suffix.lower() == ".pt" and str(device).startswith("intel:"):
        log(TAG, f"警告: .pt 权重不支持 device={device}，自动改用 cpu"
                 "（OpenVINO IR 目录才用 intel:gpu）")
        device = "cpu"

    # 产物路径与防误覆盖：整批任务事先一次性检查，避免跑到一半才发现冲突。
    # 绝不写主链路 output/<branch>/<cam>/，也不写 output/viz/。
    all_existing = []
    for task in tasks:
        _, _, _, targets = _target_paths(task, args)
        all_existing.extend(p for p in targets if p.exists())
    if all_existing and not args.overwrite:
        raise SystemExit("[错误] 以下产物已存在，拒绝覆盖（如需覆盖请加 --overwrite）：\n  "
                         + "\n  ".join(str(p) for p in all_existing))
    for task in tasks:
        task["out_dir"].mkdir(parents=True, exist_ok=True)

    log(TAG, f"模型    : {weights}")
    log(TAG, f"参数    : conf={conf} iou={iou} class_id={cls_id} device={device} "
             f"stride={args.stride} batch={args.batch} max_frames={args.max_frames or 'all'}")
    log(TAG, f"任务数  : {len(tasks)}")
    try:
        engine = DetInference(weights, conf, cls_id, device, tracker, TAG, iou=iou)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"[错误] 模型加载失败: {exc}\n"
                         "若 intel:gpu 不可用，改用 --device cpu 重试") from exc

    warm_sizes = set()
    stop_flag = {"v": False}

    def run_one(task: dict, idx: int, total: int) -> dict:
        """处理单个视频；内部循环沿用原有逐帧/批量口径。"""
        prefix = f"[{idx}/{total}]"
        source, cam_label = task["source"], task["cam"]
        video_path, jsonl_path, summary_path, _ = _target_paths(task, args)
        is_stream = isinstance(source, str)
        if is_stream and args.max_frames == 0:
            log(TAG, f"{prefix} 实时流源未指定 --max-frames，"
                     "将持续运行直到 Ctrl+C 或流断开")
        out_desc = (str(video_path) if not args.no_video else
                    ("仅统计，不落盘" if not args.save_report
                     else str(jsonl_path)))
        log(TAG, f"{prefix} 视频源  : {source}")
        log(TAG, f"{prefix} 产物    : {out_desc}")

        cap = open_camera(str(source), low_latency=True) if is_stream \
            else cv2.VideoCapture(str(source))
        if cap is None or not cap.isOpened():
            log(TAG, f"{prefix} [失败] 无法打开视频源: {source}")
            return {"ok": False, "reason": f"无法打开视频源: {source}"}
        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not (0 < src_fps < 240):
            src_fps = 25.0
            log(TAG, f"{prefix} 源 FPS 不可用，按 25 fps 计时")
        out_fps = src_fps / args.stride
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        # 不同视频分辨率不同：每种尺寸预热一次，避免把 OpenVINO 二次编译
        # 的 1~2s 算进耗时统计（640x640 dummy 与 1080p 的 640x384 形状不同）。
        size_key = (frame_h or 640, frame_w or 640)
        if size_key not in warm_sizes:
            log(TAG, f"{prefix} 预热中（真实帧尺寸 {frame_w}x{frame_h}，不计入耗时统计）...")
            engine.warmup(h=size_key[0], w=size_key[1])
            warm_sizes.add(size_key)

        writer = None
        det_records = []
        infer_ms_list = []
        calls = []              # 批次级：每次前向的帧数与耗时，便于溯源冷启动
        batch_count = 0
        no_det_frames = []
        conf_values = []
        frames_read = 0
        frames_inferred = 0
        boxes_total = 0
        wall_start = time.perf_counter()
        interrupted = False

        def flush(pending):
            """对攒好的一批帧前向 + 绘制 + 记录，返回本批推理帧数。"""
            nonlocal writer, frames_inferred, boxes_total, batch_count
            if not pending:
                return 0
            frames = [f for _, f in pending]
            t0 = time.perf_counter()
            results = engine.predict(frames)
            call_ms = (time.perf_counter() - t0) * 1000.0
            per_frame_ms = call_ms / len(frames)
            batch_count += 1
            calls.append({"index": batch_count, "batch": len(frames),
                          "call_ms": round(call_ms, 2),
                          "per_frame_ms": round(per_frame_ms, 3)})
            h, w = frames[0].shape[:2]
            if writer is None and not args.no_video:
                writer = cv2.VideoWriter(str(video_path),
                                        cv2.VideoWriter_fourcc(*"mp4v"),
                                        out_fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"无法创建视频写入器: {video_path}")
            for (frame_id, frame), result in zip(pending, results):
                dets = _parse_boxes(result)
                time_s = frame_id / src_fps
                infer_ms_list.append(per_frame_ms)
                boxes_total += len(dets)
                conf_values.extend(d["conf"] for d in dets)
                if not dets:
                    no_det_frames.append(frame_id)
                if args.save_report:
                    det_records.append({
                        "frame": frame_id,
                        "time_s": round(time_s, 4),
                        "det": len(dets),
                        "boxes": [d["box"] for d in dets],
                        "confs": [d["conf"] for d in dets],
                        "classes": [d["cls"] for d in dets],
                        "infer_ms": round(per_frame_ms, 3),
                    })
                if not args.no_video or args.show:
                    _draw(frame, dets, frame_id, time_s, per_frame_ms, conf,
                          weights.name if weights.is_dir() else weights.stem)
                    if writer is not None:
                        writer.write(frame)
                    if args.show:
                        cv2.imshow("marker_det", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            raise KeyboardInterrupt
                frames_inferred += 1
            return len(pending)

        try:
            pending = []
            while True:
                if args.max_frames and frames_inferred + len(pending) >= args.max_frames:
                    break
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if frames_read % args.stride == 0:
                    pending.append((frames_read, frame))
                frames_read += 1
                # 攒满 batch 或到达 max_frames 才前向；剩余额度按「本次运行总上限」
                # 计算，不能扣掉 pending 自身，否则尾段会被切成一堆 1 帧的小批。
                limit = (min(args.batch, args.max_frames - frames_inferred)
                         if args.max_frames else args.batch)
                if len(pending) >= max(1, limit):
                    flush(pending)
                    pending = []
                    if frames_inferred % (args.batch * 50) < args.batch:
                        log(TAG, f"{prefix} 进度: 推理 {frames_inferred} 帧, "
                                 f"检出 {boxes_total} 框, "
                                 f"最近 {infer_ms_list[-1]:.1f} ms/帧")
            flush(pending)  # 尾批（不足 batch 也要跑完）
        except KeyboardInterrupt:
            interrupted = True
            stop_flag["v"] = True   # 通知外层批量循环停止，不再开下一个视频
            log(TAG, f"{prefix} 收到中断，收尾并写出已完成部分")
        finally:
            if writer is not None:
                writer.release()
            cap.release()
            if args.show:
                cv2.destroyAllWindows()

        wall_s = time.perf_counter() - wall_start
        infer_s = sum(infer_ms_list) / 1000.0
        det_frames = frames_inferred - len(no_det_frames)
        # 稳态口径：OpenVINO 对不同 batch size（含尾批）会重新编译，混在一起统计
        # 会把几百 ms 的编译耗时算成"每帧推理"。只取剔除首批后、与主流批大小
        # 一致的批次，样本量与批大小一并写进 summary，数字可自行复核。
        common_batch = (Counter(c["batch"] for c in calls).most_common(1)[0][0]
                        if calls else None)
        steady = [c["per_frame_ms"] for c in calls[1:]
                  if c["batch"] == common_batch]
        by_batch = {}
        for c in calls:
            by_batch.setdefault(c["batch"], []).append(c["call_ms"])
        call_ms_by_batch = {
            str(size): {
                "calls": len(vals),
                "mean_ms": round(statistics.fmean(vals), 2),
                "p50_ms": round(_percentile(vals, 0.5), 2),
                "max_ms": round(max(vals), 2),
            } for size, vals in by_batch.items()
        }
        summary = {
            "scope": {
                "purpose": "纯检测验证：不做跟踪 / 亚像素定位 / FFT / 主链路落盘",
                "config": str(cfg_path),
                "source": str(source),
                "cam": cam_label,
                "weights": str(weights),
                "device": device,
                "conf": conf,
                "iou": iou,
                "class_id": cls_id,
                "src_fps": round(src_fps, 4),
                "out_fps": round(out_fps, 4),
                "stride": args.stride,
                "batch": args.batch,
                "max_frames": args.max_frames,
                "interrupted": interrupted,
            },
            "stats": {
                "frames_read": frames_read,
                "frames_inferred": frames_inferred,
                "frames_with_det": det_frames,
                "detection_rate": round(det_frames / frames_inferred, 4) if frames_inferred else None,
                "boxes_total": boxes_total,
                "boxes_per_inferred_frame": round(boxes_total / frames_inferred, 3) if frames_inferred else None,
                "conf_min": round(min(conf_values), 4) if conf_values else None,
                "conf_mean": round(statistics.fmean(conf_values), 4) if conf_values else None,
                "conf_max": round(max(conf_values), 4) if conf_values else None,
                "infer_ms_mean": round(statistics.fmean(infer_ms_list), 2) if infer_ms_list else None,
                "infer_ms_p50": round(_percentile(infer_ms_list, 0.5), 2) if infer_ms_list else None,
                "infer_ms_p95": round(_percentile(infer_ms_list, 0.95), 2) if infer_ms_list else None,
                # 稳态口径：剔除首批 + 只取主流批大小，避免 OpenVINO 重编译污染。
                "steady_batch_size": common_batch,
                "steady_calls": len(steady),
                "steady_ms_mean": round(statistics.fmean(steady), 2) if steady else None,
                "steady_ms_p50": round(_percentile(steady, 0.5), 2) if steady else None,
                "steady_ms_p95": round(_percentile(steady, 0.95), 2) if steady else None,
                "steady_fps": round(1000.0 / statistics.fmean(steady), 2) if steady else None,
                "batch_sizes": sorted(str(k) for k in by_batch),
                "call_ms_by_batch": call_ms_by_batch,
                "calls_total": len(calls),
                "calls_head": calls[:20],
                "infer_only_fps": round(frames_inferred / infer_s, 2) if infer_s > 0 else None,
                "end_to_end_fps": round(frames_inferred / wall_s, 2) if wall_s > 0 else None,
                "wall_s": round(wall_s, 2),
                "no_det_frames_count": len(no_det_frames),
                "no_det_frames_head": no_det_frames[:20],
            },
            "outputs": {
                "video": None if args.no_video else str(video_path),
                "jsonl": str(jsonl_path) if args.save_report else None,
                "summary": str(summary_path) if args.save_report else None,
            },
        }

        h264 = None
        if not args.no_video and frames_inferred and writer is not None and not args.no_h264:
            try:
                from viz.export import _to_h264  # 复用项目统一转码口径，保证浏览器可播
                h264 = _to_h264(video_path)
                if not h264:
                    log(TAG, "H.264 转码失败，保留 mp4v（可用本地播放器查看）")
            except Exception as exc:  # noqa: BLE001
                h264 = False
                log(TAG, f"H.264 转码不可用，保留 mp4v: {exc}")
        summary["outputs"]["h264"] = h264

        if args.save_report:
            with jsonl_path.open("w", encoding="utf-8") as f:
                for row in det_records:
                    f.write(json.dumps(jsonable(row), ensure_ascii=False) + "\n")
            summary_path.write_text(
                json.dumps(jsonable(summary), ensure_ascii=False, indent=2),
                encoding="utf-8")

        st = summary["stats"]
        print("-" * 68)
        print(f"{prefix} 视频: {source}")
        print(f"[汇总] 推理帧 {st['frames_inferred']} / 读取帧 {st['frames_read']}"
              f"  检出框 {st['boxes_total']}")
        print(f"[汇总] 有检出帧 {st['frames_with_det']} "
              f"({(st['detection_rate'] or 0) * 100:.1f}%)  "
              f"无检出帧 {st['no_det_frames_count']}"
              + (f"  前几个帧号 {no_det_frames[:10]}" if no_det_frames else ""))
        print(f"[汇总] 置信度 min/mean/max = {st['conf_min']} / "
              f"{st['conf_mean']} / {st['conf_max']}")
        print(f"[汇总] 前向耗时(含首批) mean/p50/p95 = {st['infer_ms_mean']} / "
              f"{st['infer_ms_p50']} / {st['infer_ms_p95']} ms")
        if st["steady_calls"]:
            print(f"[汇总] 稳态(剔除首批 + 仅 batch={st['steady_batch_size']}, "
                  f"n={st['steady_calls']} 次前向) mean/p50/p95 = "
                  f"{st['steady_ms_mean']} / {st['steady_ms_p50']} / "
                  f"{st['steady_ms_p95']} ms → {st['steady_fps']} fps")
        else:
            print("[汇总] 稳态样本不足（只跑了首批），耗时含冷启动，不用于性能判断")
        print(f"[汇总] 端到端 {st['end_to_end_fps']} fps（含解码/绘制/落盘）")
        if not args.no_video and writer is not None:
            print(f"[产物] 标注视频: {video_path}"
                  + (" (H.264)" if h264 else " (mp4v，本地播放器可播)"))
        if args.save_report:
            print(f"[产物] 逐帧记录: {jsonl_path}")
            print(f"[产物] 汇总统计: {summary_path}")
        print("-" * 68)
        return {
            "ok": frames_inferred > 0,
            "summary": summary,
            "frames": frames_inferred,
            "detection_rate": st["detection_rate"],
            "steady_ms": st["steady_ms_mean"],
            "interrupted": interrupted,
        }

    results = []
    for i, task in enumerate(tasks, 1):
        if stop_flag["v"]:
            break
        try:
            results.append((task, run_one(task, i, len(tasks))))
        except KeyboardInterrupt:
            stop_flag["v"] = True
            log(TAG, "收到中断，停止后续视频")
            break
        except Exception as exc:  # noqa: BLE001  单个视频失败不影响整批
            name = Path(str(task["source"])).name
            log(TAG, f"[{i}/{len(tasks)}] [失败] {name}: {exc}")
            results.append((task, {"ok": False, "reason": str(exc)}))

    skipped = len(tasks) - len(results)
    ok_n = sum(1 for _, r in results if r["ok"])
    fail_n = len(results) - ok_n
    if len(tasks) > 1:
        print("=" * 68)
        print(f"[批量完成] 成功 {ok_n} / 失败 {fail_n} / 跳过 {skipped}"
              f"（共 {len(tasks)} 个视频）")
        for task, r in results:
            name = Path(str(task["source"])).name
            if not r["ok"]:
                print(f"  [失败] {name}: {r.get('reason', '未推理到任何帧')}")
                continue
            st = r["summary"]["stats"]
            print(f"  [成功] {name}: 帧 {r['frames']}  检出率 "
                  f"{(r['detection_rate'] or 0) * 100:.1f}%  "
                  f"稳态 {r['steady_ms']} ms/帧"
                  + ("  （已中断）" if r["interrupted"] else ""))
        if skipped:
            print(f"  [跳过] 剩余 {skipped} 个视频未处理")
        print("=" * 68)
    return 1 if (fail_n or skipped) else 0


if __name__ == "__main__":
    raise SystemExit(main())
