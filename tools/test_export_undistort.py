"""回归测试：相机级 undistort 开关必须一路透传到标注视频导出。

背景（2026-11-20 实测 bug）：
    supervisor 构造导出参数时用白名单只挑 id/task/url/calibration，
    把 undistort 丢了 → camera_undistort() 收到 None → 回退 pole mode
    默认（image_relative=True）→ 明明配置写 false，标注视频仍整帧去畸变，
    连源视频烧录的时间戳都被扭斜。

跑法（项目根目录）：
    .venv/Scripts/python.exe tools/test_export_undistort.py
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import viz.export as vexp  # noqa: E402
import utils.supervisor as sup  # noqa: E402
from utils.config import camera_undistort  # noqa: E402


def _cfg(tmp: Path, undistort) -> dict:
    cam = {
        "id": "pole_cam_01",
        "task": "pole",
        "url": "fake.mp4",          # 导出函数被 mock，文件无需真实存在
        "calibration": {"K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "dist": [0, 0, 0, 0, 0]},
    }
    if undistort is not None:      # None = 故意省略该字段，测缺省回退
        cam["undistort"] = undistort
    rec = tmp / "pole" / "pole_cam_01"
    rec.mkdir(parents=True, exist_ok=True)
    (rec / ".rec_done").write_text("done", encoding="utf-8")
    return {
        "runtime": {"output_dir": str(tmp)},
        "cameras": [cam],
        "pole": {"measurement": {"mode": "image_relative"}},
        "_config_dir": str(ROOT),
    }


def _capture(cfg) -> list:
    seen = []

    def fake(cam, output_dir, pole_mode, out_dir, **kwargs):
        seen.append(dict(cam))
        return {"status": "ok", "out": "fake"}

    original = vexp.export_cam_video
    vexp.export_cam_video = fake
    try:
        sup._export_annotated_videos(cfg)
        sup._poll_done_exports(cfg, set())
    finally:
        vexp.export_cam_video = original
    return seen


def main() -> int:
    failures = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for value in (False, True):
            seen = _capture(_cfg(tmp, value))
            if len(seen) != 2:
                failures.append(f"undistort={value}: 期望捕获 2 次导出，实际 {len(seen)}")
            for c in seen:
                if c.get("undistort") is not value:
                    failures.append(
                        f"undistort={value}: 导出参数没透传开关 → {c.get('undistort')!r}")
                final = camera_undistort(c, "image_relative")
                if final is not value:
                    failures.append(f"undistort={value}: 最终判定错误 → {final}")

        # 省略字段时确实应回退到 pole mode 默认（这就是 bug 发生时的行为）
        seen = _capture(_cfg(tmp, None))
        for c in seen:
            if camera_undistort(c, "image_relative") is not True:
                failures.append("缺省字段应回退 image_relative=True")

    # world_3d 下显式 true 必须被忽略（避免重复去畸变）
    if camera_undistort({"id": "x", "undistort": True}, "world_3d") is not False:
        failures.append("world_3d 下 undistort=true 未被忽略")

    if failures:
        print("FAIL")
        for f in failures:
            print("  -", f)
        return 1
    print("PASS: supervisor 两条导出路径均完整透传 undistort；"
          "camera_undistort 判定符合预期")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
