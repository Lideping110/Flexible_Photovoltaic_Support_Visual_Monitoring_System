# -*- coding: utf-8 -*-
"""单帧标定：抓取一张清晰图 → 检测 → 分割找外圆 → 求 mm/px → 输出参数。

为什么需要它：实时 marker_tracing 的自动标定(前10帧)在压缩噪声下容易取到
错误的大圆(套到柱子/背景/棋盘格外环)。本脚本改为对**一张静态清晰图**做标定，
把得到的比例直接作为 --mm-per-px 传给 marker_tracing，彻底跳过实时10帧标定。

用法：
    uv run python calibrate_scale.py                      # 拉流抓最清晰帧
    uv run python calibrate_scale.py --image 一张清晰图.jpg  # 直接读图
    uv run python calibrate_scale.py --real-diameter-mm 100 --topk 40
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from marker_subpixel_tracker.src.tracker import ByteTrackTracker
from marker_subpixel_tracker.src.features import measure_outer_diameter_px

PALETTE = [(0, 255, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255),
           (0, 165, 255), (255, 255, 0), (128, 0, 255), (0, 128, 255)]

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RTSP = "rtsp://admin:hhjt220220@192.168.1.65:554/streaming/Channels/101"
DEFAULT_CONFIG = SCRIPT_DIR / "marker_subpixel_tracker" / "config" / "config.yaml"
OPENVINO_DIR = SCRIPT_DIR / "marker26_det_openvino_model"


def open_stream(rtsp):
    """打开 RTSP 流并尽可能压低缓冲延迟。"""
    cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开 RTSP 流: {rtsp}")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def resolve_weights(cfg_path, weights):
    """按 config 文件的权重相对路径解析到真实文件。"""
    p = Path(weights)
    if p.is_absolute():
        return p
    for base in (SCRIPT_DIR, cfg_path.parent, cfg_path.parent.parent):
        cand = base / p
        if cand.exists():
            return cand
    return SCRIPT_DIR / p


def is_openvino_dir(p):
    """判断路径是否为 OpenVINO 模型目录(含 .xml 拓扑文件)。"""
    p = Path(p)
    return p.is_dir() and any(p.glob("*.xml"))


def sharpness(gray):
    """拉普拉斯方差 = 图像清晰度指标(越大越清晰)。"""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rtsp", default=DEFAULT_RTSP)
    ap.add_argument("--image", default=None,
                    help="直接读一张图片做标定(不拉流)")
    ap.add_argument("--real-diameter-mm", type=float, default=100.0,
                    help="靶标外圆真实直径(mm), 默认100=10cm")
    ap.add_argument("--out", default="radius/99_final_circles.jpg",
                    help="保存的清晰图(含绿圈+直径标注)路径")
    ap.add_argument("--radius-dir", default="radius",
                    help="逐步中间图保存目录(默认 radius)")
    ap.add_argument("--topk", type=int, default=30,
                    help="拉流时抓取的帧数, 取最清晰的一帧(默认30)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--weights", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--conf", type=float, default=None,
                    help="检测置信度阈值(覆盖 config)")
    args = ap.parse_args()

    radius_dir = Path(args.radius_dir)
    radius_dir.mkdir(parents=True, exist_ok=True)

    def save(step, img, tag=""):
        """保存一步的中间图到 radius/ 目录(文件名全英文, 规避中文路径坑)。"""
        name = f"{step}{('_' + tag) if tag else ''}.jpg"
        path = radius_dir / name
        cv2.imwrite(str(path), img)
        print(f"[标定] 保存 -> {path}")
        return path

    # ---- 1. 取清晰帧 ----
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            sys.exit(f"[标定] 无法读取图片: {args.image}")
        print(f"[标定] 使用图片 {args.image}, 清晰度="
              f"{sharpness(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)):.1f}")
    else:
        cap = open_stream(args.rtsp)
        best_var, best_frame = -1.0, None
        for i in range(args.topk):
            ok, f = cap.read()
            if not ok or f is None:
                continue
            v = sharpness(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
            if v > best_var:
                best_var, best_frame = v, f.copy()
        cap.release()
        if best_frame is None:
            sys.exit("[标定] 拉流抓帧失败")
        frame = best_frame
        print(f"[标定] 抓取 {args.topk} 帧, 选用最清晰帧 清晰度={best_var:.1f}")

    # 步骤1图: 原始清晰帧
    save("00_original", frame)

    # ---- 2. 检测 ----
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_cfg = cfg["model"]
    if args.weights:
        weights = Path(args.weights).resolve()
    elif OPENVINO_DIR.is_dir():
        weights = OPENVINO_DIR
    else:
        weights = resolve_weights(Path(args.config), model_cfg["weights"])
    if not Path(weights).exists():
        sys.exit(f"[标定] 模型不存在: {weights}")
    device = args.device or ("intel:gpu" if is_openvino_dir(weights)
                              else model_cfg["device"])
    conf = args.conf if args.conf is not None else model_cfg["confidence"]
    tracker = ByteTrackTracker(
        str(weights), conf, model_cfg["class_id"], device,
        tracker=cfg.get("tracking", {}).get("tracker", "bytetrack.yaml"),
        persist=True)
    tracks = tracker.update(frame)
    if not tracks:
        sys.exit("[标定] 未检测到靶标, 请确认靶标在画面内/调整角度后重试")
    print(f"[标定] 检测到 {len(tracks)} 个目标")

    # 步骤2图: 检测框 + id + conf
    detect_img = frame.copy()
    for idx, t in enumerate(tracks):
        x1, y1, x2, y2 = np.round(t["box"]).astype(int)
        col = PALETTE[idx % len(PALETTE)]
        cv2.rectangle(detect_img, (x1, y1), (x2, y2), col, 2)
        cv2.putText(detect_img, f"id={t['id']} conf={t['conf']:.2f}",
                    (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, col, 2, cv2.LINE_AA)
    save("01_detect", detect_img)

    # ---- 3. 逐个目标: 找外圆 + 比例(含逐步中间图) ----
    real_d = args.real_diameter_mm
    results = []
    for idx, t in enumerate(tracks):
        box = np.asarray(t["box"], dtype=float)
        tag = f"id{t['id']}"
        # 与 marker_tracing.py 保持一致的 margin 逻辑: 检测框短边 / 10
        x1, y1, x2, y2 = np.round(box).astype(int)
        sbox = min(x2 - x1, y2 - y1)
        margin_px = max(1, int(sbox / 10))
        res = measure_outer_diameter_px(frame, box, margin=margin_px,
                                        return_debug=True)
        circle, dbg = res
        # 逐步存图(只存非 None 的)
        if dbg.get("crop") is not None:
            save(f"02_{tag}_crop", dbg["crop"])
        if dbg.get("gray") is not None:
            save(f"03_{tag}_gray", dbg["gray"])
        if dbg.get("blurred") is not None:
            save(f"04_{tag}_blur", dbg["blurred"])
        if dbg.get("binary") is not None:
            save(f"05_{tag}_binary", dbg["binary"])
        if dbg.get("candidates") is not None:
            save(f"06_{tag}_candidates", dbg["candidates"])
        if circle is None:
            print(f"[标定] id={t['id']} 外圆提取失败, 跳过(已存中间图核对)")
            continue
        d, (ccx, ccy), r = circle
        # 步骤7图: 最终选中圆叠加(绿圈已在 chosen 内, 仅补直径文字)
        chosen = dbg.get("chosen")
        final_crop = (chosen if chosen is not None else dbg["crop"]).copy()
        cv2.putText(final_crop, f"D={d:.2f}px r={r:.2f}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2, cv2.LINE_AA)
        save(f"07_{tag}_circle", final_crop)

        mm_per_px = real_d / d
        results.append((t["id"], d, mm_per_px))
        cv2.circle(frame, (int(ccx), int(ccy)), int(r), (0, 255, 0), 3)
        cv2.putText(frame, f"id{t['id']} D={d:.2f}px r={r:.2f}",
                    (int(ccx - r), max(20, int(ccy - r) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                    cv2.LINE_AA)

    if not results:
        sys.exit("[标定] 所有目标外圆提取均失败(请查看 radius/ 中间图定位原因)")

    # ---- 4. 保存带圆清晰图 ----
    cv2.imwrite(args.out, frame)
    print(f"[标定] 已保存带圆清晰图 -> {args.out}")

    # ---- 5. 输出比例与推荐参数 ----
    print("\n==== 各目标像素→mm 比例 ====")
    for tid, d, mpp in results:
        print(f"  id={tid}: 直径={d:.2f}px  真实={real_d}mm  →  mm/px={mpp:.5f}")
    if len(results) == 1:
        rec = results[0][2]
        print(f"\n[标定] 推荐参数(单靶标)：")
        print(f"  uv run python marker_tracing.py --mm-per-px {rec:.5f}")
    else:
        rec = results[0][2]
        print(f"\n[标定] 检测到多个目标: marker_tracing --mm-per-px 为所有目标共用比例")
        print(f"       (假设各目标到相机距离相同; 距离不同请分别标定或后续扩展)")
        print(f"  uv run python marker_tracing.py --mm-per-px {rec:.5f}  # 取 id={results[0][0]}")


if __name__ == "__main__":
    main()
