"""柔性光伏支架可视化监测系统 — 统一入口。

启动：
    uv run python main.py

可选参数：
    --config PATH      统一配置文件路径（默认根目录 config.yaml）
    --refresh-init     忽略已保存的初始基准（init_state.json），强制重建。
"""
import argparse
import multiprocessing as mp


def main() -> int:
    parser = argparse.ArgumentParser(description="柔性光伏支架可视化监测系统")
    parser.add_argument("--config", default="config.yaml",
                        help="统一配置文件路径（默认 config.yaml）")
    parser.add_argument("--refresh-init", action="store_true",
                        help="忽略已保存的初始基准，强制重建 init_state.json")
    args = parser.parse_args()

    from utils.supervisor import run

    return run(args.config, refresh_init=args.refresh_init)


if __name__ == "__main__":
    mp.freeze_support()  # Windows spawn 必需
    raise SystemExit(main())
