#!/usr/bin/env python3
"""convert_to_openvino.py — 将检测/分割模型导出为 OpenVINO IR。

与本项目推理后端严格对齐:
  - 推理后端: ultralytics YOLO, 加载含 .xml 的目录即自动走 OpenVINO(Intel GPU)。
  - 依赖: ultralytics>=8.4.147, openvino>=2026.4.0 (见 pyproject.toml)。
  - 项目所有模型均为 YOLO (.pt): marker26_det.pt / yolo11n-seg.pt / yolo26s-seg.pt / pole26_seg.pt。

重要 — 默认加载名硬编码:
  marker_tracing.py / calibrate_scale.py 里的 OPENVINO_DIR 写死为
  "marker26_det_openvino_model"。因此新模型转完后, 推理必须显式指定:
      uv run python marker_tracing.py --weights <新目录>
  或对本脚本加 --link-default 把新目录复制为该默认名(会自动备份旧模型)。

用法:
  uv run python convert_to_openvino.py model.pt
  uv run python convert_to_openvino.py model.pt --imgsz 640 --half --dynamic
  uv run python convert_to_openvino.py model.pt --out-dir mymodel_openvino_model
  uv run python convert_to_openvino.py model.pt --link-default
  uv run python convert_to_openvino.py model.onnx --out-dir model_openvino_model

INT8 量化未在此脚本实现: 需要校准数据集 + data yaml, 属独立流程, 避免写出不可靠代码。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="将模型导出为 OpenVINO IR (与本项目 ultralytics 推理后端对齐)")
    p.add_argument("input", help="输入模型: .pt(YOLO) / .onnx / 已是 OpenVINO 的目录")
    p.add_argument("--out-dir", default=None,
                   help="输出目录 (默认 <stem>_openvino_model, 与推理识别一致)")
    p.add_argument("--imgsz", type=int, default=640,
                   help="输入尺寸(正方形), 默认 640; 需与推理 imgsz 一致")
    p.add_argument("--half", action="store_true",
                   help="导出 FP16 IR (Intel GPU 推荐: 更快、体积更小)")
    p.add_argument("--dynamic", action="store_true",
                   help="动态输入形状 (允许推理时变分辨率; 略增体积)")
    p.add_argument("--device", default="cpu",
                   help="导出用设备 cpu / cuda:0, 默认 cpu")
    p.add_argument("--link-default", action="store_true",
                   help="导出后复制为 marker26_det_openvino_model/ 供 marker_tracing 等默认加载"
                        " (会先备份旧模型为 .bak)")
    return p.parse_args()


def is_openvino_dir(p: Path) -> bool:
    return p.is_dir() and any(p.glob("*.xml"))


def export_yolo(src: Path, out_dir: Path, args: argparse.Namespace) -> Path:
    """YOLO .pt -> OpenVINO IR, 用 ultralytics 原生 export。"""
    from ultralytics import YOLO
    print(f"[export] 加载 YOLO 模型: {src}")
    model = YOLO(str(src))
    # ultralytics 默认把 openvino 导出到 <src_stem>_openvino_model (与 src 同目录)
    exported = model.export(
        format="openvino",
        imgsz=args.imgsz,
        half=args.half,
        dynamic=args.dynamic,
        device=args.device,
        int8=False,
    )
    ep = Path(exported)
    if ep.resolve() != out_dir.resolve() and ep.exists():
        if out_dir.exists():
            shutil.rmtree(out_dir)
        shutil.move(str(ep), str(out_dir))
    return out_dir


def export_onnx(src: Path, out_dir: Path, args: argparse.Namespace) -> Path:
    """ONNX -> OpenVINO IR, 用 openvino.convert_model。"""
    from openvino import convert_model, serialize
    print(f"[export] ONNX -> OpenVINO: {src}")
    ov_model = convert_model(str(src))
    out_dir.mkdir(parents=True, exist_ok=True)
    xml_path = out_dir / (src.stem + ".xml")
    serialize(ov_model, str(xml_path))
    return out_dir


def link_default(out_dir: Path, script_dir: Path) -> None:
    target = script_dir / "marker26_det_openvino_model"
    if target.exists():
        bak = script_dir / "marker26_det_openvino_model.bak"
        if bak.exists():
            shutil.rmtree(bak)
        shutil.move(str(target), str(bak))
        print(f"[link] 已备份旧默认模型 -> {bak.name}")
    shutil.copytree(out_dir, target)
    print(f"[link] 已复制为默认加载名 {target.name} "
          f"(marker_tracing / calibrate_scale 将自动使用)")


def main() -> int:
    args = parse_args()
    src = Path(args.input).resolve()
    if not src.exists():
        print(f"[error] 输入不存在: {src}", file=sys.stderr)
        return 2

    script_dir = Path(__file__).resolve().parent
    # 输出目录默认: 与 src 同目录、同名 + _openvino_model
    default_out = src.parent / (src.stem + "_openvino_model")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else default_out

    # 输入已是 OpenVINO 目录 -> 直接复用/重命名
    if src.is_dir() and is_openvino_dir(src):
        print(f"[export] 输入已是 OpenVINO 目录, 复制到: {out_dir}")
        if out_dir.exists():
            shutil.rmtree(out_dir)
        shutil.copytree(src, out_dir)
    elif src.suffix.lower() == ".pt":
        out_dir = export_yolo(src, out_dir, args)
    elif src.suffix.lower() == ".onnx":
        out_dir = export_onnx(src, out_dir, args)
    else:
        print(f"[error] 不支持的输入类型: {src.suffix} "
              f"(支持 .pt / .onnx / OpenVINO 目录)", file=sys.stderr)
        return 2

    if not is_openvino_dir(out_dir):
        print(f"[error] 导出后未找到 .xml, 导出可能失败: {out_dir}",
              file=sys.stderr)
        return 1

    print(f"[ok] OpenVINO IR 已生成: {out_dir}")
    for f in sorted(out_dir.glob("*.xml")):
        print(f"      - {f.name}")
    if args.half:
        print("[note] 已启用 FP16; 确认推理设备支持 FP16 (Intel GPU 一般支持)")
    if args.dynamic:
        print("[note] 已启用动态形状; 推理可传任意分辨率 (建议与 --imgsz 接近以保精度)")

    if args.link_default:
        link_default(out_dir, script_dir)

    print("\n[使用] 推理时指定权重:")
    print(f"  uv run python marker_tracing.py --weights {out_dir}")
    print(f"  uv run python calibrate_scale.py --weights {out_dir}")
    if not args.link_default:
        print("  (或加 --link-default 设为默认加载名 marker26_det_openvino_model)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
