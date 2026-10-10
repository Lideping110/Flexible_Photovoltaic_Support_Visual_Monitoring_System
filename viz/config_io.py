"""极简配置读取：只为可视化提供 cameras / output_dir / pole 测量模式。

刻意不调用 utils.config.load_config（它会校验并加载模型路径），
让 viz 完全独立于推理栈——没有模型文件也能跑。
"""
from pathlib import Path

import yaml


def load_viz_config(config_path: str) -> dict:
    """读取可视化所需的少量配置。

    返回：
        {
          "output_dir": <绝对路径>,
          "pole_mode": "image_relative" | "world_3d",
          "cameras": [{"id","task","url"(已解析绝对), "calibration"|None}, ...]
        }
    """
    p = Path(config_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"配置文件不存在: {p}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    base = p.parent
    rt = cfg.get("runtime", {}) or {}
    out = rt.get("output_dir", "output")
    out = out if Path(out).is_absolute() else str(base / out)

    pole_mode = (((cfg.get("pole", {}) or {}).get("measurement", {}) or {})
                 .get("mode", "image_relative"))

    cameras = []
    for c in cfg.get("cameras", []) or []:
        url = c.get("url", "")
        src = url if Path(url).is_absolute() else str(base / url)
        cameras.append({
            "id": c["id"],
            "task": c["task"],
            "url": src,
            "calibration": c.get("calibration"),
        })

    return {"output_dir": out, "pole_mode": pole_mode, "cameras": cameras}
