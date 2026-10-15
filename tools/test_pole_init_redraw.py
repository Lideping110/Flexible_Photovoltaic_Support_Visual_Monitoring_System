"""回归测试：pole 初始帧应随每根柱子基线就绪而逐步重绘，而非只在第一个基线冻结。

直接驱动 PoleCameraSession.persist_init / finalize_init 的真实代码路径，
验证 init_frame.jpg 在 P1..P4 各自基线就绪时被重绘、最终 4 根全部带拟合线。
"""
import sys, types, tempfile, os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 避免 worker 构造时真实加载 bytetrack 配置（与本次修复无关）
import utils.inference as inf
_dummy_tracker = types.SimpleNamespace(update=lambda *a, **k: None)
inf.make_byte_tracker = lambda *a, **k: _dummy_tracker

import pole_tilt_monitor.worker as w

# 捕获 save_anchor_frame 的每次调用，记录背景 id 与"已带基线条数"
calls = []
_orig_save = w.save_anchor_frame
def _spy_save(frame, anchors, path):
    bg_id = id(frame)
    n_fit = 0
    for a in anchors.values():
        bl = a.get("baseline")
        vals = bl.get("values") if isinstance(bl, dict) else bl
        if vals and len(vals) >= 5:
            n_fit += 1
    calls.append({"bg_id": bg_id, "n_fit": n_fit, "path": str(path)})
    # 仍真正落盘，验证产物存在
    _orig_save(frame, anchors, path)
w.save_anchor_frame = _spy_save


def build_session(tmp):
    cam = {
        "id": "test_pole",
        "calibration": {
            "K": [[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]],
            "dist": [],
        },
    }
    model_cfg = {"tracker": "bytetrack.yaml"}
    pole_cfg = {
        "baseline": {"relative_frames": 1},
        "filter": {"alpha": 0.25},
        "measurement": {"mode": "image_relative", "quality_threshold": 0.0},
    }
    anchor_cfg = {"iou_min": 0.3, "release_after_s": 10.0, "ema_alpha": 0.05}
    runtime = {
        "init": {"persist": True, "anchor": anchor_cfg},
        "output_dir": str(tmp),
    }
    import queue
    return w.PoleCameraSession(cam, model_cfg, pole_cfg, anchor_cfg,
                               runtime, True, queue.Queue(), "t")


def main():
    tmp = Path(tempfile.mkdtemp(prefix="pole_init_"))
    sess = build_session(tmp)

    frame = (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)
    # 4 根柱子，x 中心递增；首帧全部可见 → ensure_init 建立 P1..P4
    dets = [
        {"id": i, "box": [100 + i * 250, 50, 180 + i * 250, 700], "conf": 0.9}
        for i in range(4)
    ]
    sess.handle_init(dets, ts=0.0)
    assert sess.anchors.initialized, "ensure_init 未建立锚点"
    assert len(sess.anchors.anchors) == 4, "锚点数应为 4"

    # 逐根让基线就绪（模拟 P1 先、P2/P3/P4 后，跨多帧）
    for i in range(4):
        pid = f"P{i+1}"
        sess.monitor.baseline[pid] = np.array([0.0, 0.0, 0.0, 0.0, -1.0, 0.0])
        sess.maybe_persist_baseline(pid, 0.9, frame)

    # finalize 再收尾一次
    sess.finalize_init()

    # ---- 断言 ----
    # 1) 每次新增基线都触发一次重绘（4 次），finalize 再 1 次 = 5
    assert len(calls) == 5, f"save_anchor_frame 调用次数={len(calls)}，期望 5"
    # 2) 背景对象在整个过程中保持同一份（首帧采一次）
    bg_ids = {c["bg_id"] for c in calls}
    assert len(bg_ids) == 1, f"背景帧对象不统一：{bg_ids}"
    # 3) 第 1 次重绘（仅 P1 就绪）应只有 1 条拟合线
    assert calls[0]["n_fit"] == 1, f"首次重绘拟合线数={calls[0]['n_fit']}，期望 1"
    # 4) 最终（calls[-1]）应 4 根全部带拟合线
    assert calls[-1]["n_fit"] == 4, f"最终拟合线数={calls[-1]['n_fit']}，期望 4"
    # 5) 产物存在
    fpath = Path(tmp) / "pole" / "test_pole" / "init" / "init_frame.jpg"
    assert fpath.is_file(), "init_frame.jpg 未生成"
    spath = Path(tmp) / "pole" / "test_pole" / "init" / "init_state.json"
    assert spath.is_file(), "init_state.json 未生成"

    print("PASS: 初始帧随基线逐步重绘；首帧背景稳定；最终 4 根全部拟合；产物齐全")
    print(f"  save_anchor_frame 调用={len(calls)} 次，背景对象唯一={len(bg_ids)==1}")
    print(f"  首次拟合线={calls[0]['n_fit']}，最终拟合线={calls[-1]['n_fit']}")


if __name__ == "__main__":
    main()
