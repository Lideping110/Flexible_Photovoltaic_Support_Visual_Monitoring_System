"""柔性光伏支架可视化监测系统 — 统一入口。

启动：
    uv run python main.py                 # 同时启动推理和 Web 大屏（8023）
    uv run python main.py --no-dashboard  # 仅启动推理

可选参数：
    --config PATH      统一配置文件路径（默认根目录 config.yaml）
    --refresh-init     忽略已保存的初始基准（init_state.json），强制重建。
    --dashboard-port   Web 大屏端口（默认 8023）
"""
import argparse
import multiprocessing as mp
import subprocess
import sys
from pathlib import Path


def _start_dashboard(config_path, host, port):
    """启动与 main 同生命周期的 Web 大屏子进程。"""
    script = Path(__file__).with_name("dashboard.py").resolve()
    cfg = Path(config_path).resolve()
    return subprocess.Popen(
        [sys.executable, str(script), "--config", str(cfg),
         "--host", str(host), "--port", str(int(port))],
        cwd=str(script.parent),
    )


def _stop_dashboard(proc):
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser(description="柔性光伏支架可视化监测系统")
    parser.add_argument("--config", default="config.yaml",
                        help="统一配置文件路径（默认 config.yaml）")
    parser.add_argument("--refresh-init", action="store_true",
                        help="忽略已保存的初始基准，强制重建 init_state.json")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="只运行采集/推理，不启动 Web 大屏")
    parser.add_argument("--dashboard-host", default="127.0.0.1",
                        help="Web 大屏监听地址（默认 127.0.0.1）")
    parser.add_argument("--dashboard-port", type=int, default=8023,
                        help="Web 大屏监听端口（默认 8023）")
    args = parser.parse_args()

    from utils.config import load_config
    from utils.supervisor import clear_run_outputs, run

    # Dashboard 是独立子进程且先于 Supervisor 启动。refresh-init 时先在这里
    # 清掉上一轮快照/分析/初始帧，避免 Dashboard 在推理新一轮尚未产出数据前
    # 继续显示旧结果。
    if args.refresh_init:
        clear_run_outputs(load_config(args.config), refresh_init=True)

    dashboard_proc = None
    try:
        if not args.no_dashboard:
            dashboard_proc = _start_dashboard(
                args.config, args.dashboard_host, args.dashboard_port)
            print(f"[main] Web 大屏已启动 → http://{args.dashboard_host}:"
                  f"{args.dashboard_port}", flush=True)
        return run(args.config, refresh_init=args.refresh_init)
    finally:
        _stop_dashboard(dashboard_proc)


if __name__ == "__main__":
    mp.freeze_support()  # Windows spawn 必需
    raise SystemExit(main())
