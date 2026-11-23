# -*- coding: utf-8 -*-
"""靶标视频主频提取 · 本地 Web 服务（标准库 + 项目已有算法）。

启动（项目根目录，必须用 .venv）：
    .venv/Scripts/python.exe video_calibration/video_gate_server.py --port 8767
浏览器打开 http://127.0.0.1:8767

接口：
    GET  /                        界面
    GET  /api/defaults            参数默认值 / 可用标定 JSON / 配置摘要
    GET  /api/health              健康检查
    GET  /api/job?id=...          异步任务状态、进度、结果
    GET  /api/download?id=...     下载该任务的 displacement.jsonl
    POST /api/calib/upload        上传标定 JSON（原始 body），保存并返回校验摘要
    POST /api/calib/validate      {path} 校验已有标定 JSON
    POST /api/analyze_jsonl       {content, axis, ...} 直接对 displacement.jsonl 提频
    POST /api/video/start         上传视频（原始 body，query 带参数）启动异步测量+提频

安全：只绑定 127.0.0.1；上传文件按 basename + 扩展名白名单落在本目录 _uploads
下，不执行上传内容；结果只在本机可读。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
for p in (_HERE, _PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import video_measurement as vm  # noqa: E402
from marker_subpixel_tracker.gate_core import (  # noqa: E402
    check_fingerprint, load_gate_config,
)

ROOT = _HERE
UPLOADS = os.path.join(ROOT, "_uploads")
CALIBS = os.path.join(UPLOADS, "calibs")
for d in (UPLOADS, CALIBS):
    os.makedirs(d, exist_ok=True)

JOBS: dict[str, dict] = {}
JLOCK = threading.Lock()
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
MAX_VIDEO_BYTES = 800 * 1024 * 1024


def _safe_name(name: str, exts: set[str] | None = None) -> str:
    base = os.path.basename(str(name or "upload")).replace("\\", "/")
    base = os.path.basename(base)
    base = re.sub(r"[^0-9A-Za-z._\-\u4e00-\u9fff]", "_", base)[:120] or "upload"
    if exts and os.path.splitext(base)[1].lower() not in exts:
        base += ".mp4"
    return base


def _settings() -> dict:
    return vm.load_settings()


def _calib_options() -> list[dict]:
    """config 指定 + 标定产物 + 上传目录，去重。"""
    out: list[dict] = []
    seen: set[str] = set()
    cfg = _settings()
    configured = str((cfg.get("fft") or {}).get("calib") or "")
    candidates: list[tuple[str, str]] = []
    if configured:
        candidates.append((configured, "config.marker.fft.calib"))
    for name in sorted(os.listdir(ROOT)):
        if name.endswith("_calib.json"):
            candidates.append((os.path.join(ROOT, name), "标定产物"))
    for name in sorted(os.listdir(CALIBS)):
        if name.endswith(".json"):
            candidates.append((os.path.join(CALIBS, name), "界面上传"))
    for path, src in candidates:
        full = path if os.path.isabs(path) else os.path.normpath(os.path.join(_PROJECT_ROOT, path))
        if not os.path.isfile(full) or full in seen:
            continue
        seen.add(full)
        cfgd = load_gate_config(full)
        out.append({
            "path": full, "name": os.path.basename(full), "source": src,
            "verdict": (cfgd or {}).get("verdict", ""),
            "cam_id": (cfgd or {}).get("cam_id", ""),
            "params": (cfgd or {}).get("params") or {},
            "fingerprint": (cfgd or {}).get("fingerprint") or {},
        })
    return out


def _summarize_result(freq: dict) -> dict:
    """裁掉每窗的 freq/power 明细与位移样本，避免接口响应过大。"""
    out = {"axis": freq["axis"], "window_s": freq["window_s"],
           "nyquist_hz": freq["nyquist_hz"], "mode": freq["mode"], "targets": {}}
    for pid, t in freq["targets"].items():
        windows = []
        for w in t["windows"]:
            windows.append({k: v for k, v in w.items() if k not in ("freq", "power")})
        out["targets"][pid] = {
            **{k: v for k, v in t.items() if k != "windows"},
            "windows": windows,
        }
    return out


def _spectrum_payload(freq: dict) -> dict:
    """只给每窗第一条谱曲线，供前端画图。"""
    ret = {}
    for pid, t in freq["targets"].items():
        ws = [w for w in t["windows"] if w.get("freq") and w.get("power")]
        if not ws:
            continue
        w = ws[-1]
        ret[pid] = {"freq": w["freq"], "power": w["power"],
                    "seg": w["seg"], "f1_hz": w.get("f1_hz")}
    return ret


def _jsonl_to_measurement(content: str, axis: str = "dy") -> dict:
    """把 displacement.jsonl 文本转成 video_measurement 的测量结构。"""
    import numpy as np

    per_pid: dict[str, list[dict]] = {}
    pk = f"cum_{axis}_mm"
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        pid, ts = r.get("pid"), r.get("ts")
        y = r.get(pk)
        if pid is None or ts is None or y is None:
            continue
        try:
            ts, y = float(ts), float(y)
        except (TypeError, ValueError):
            continue
        mmp = 0.0
        try:
            mmp = float(r.get("mm_per_px") or 0.0)
        except (TypeError, ValueError):
            pass
        per_pid.setdefault(str(pid), []).append({
            "ts": ts, pk: y,
            f"{axis}_px": y / mmp if mmp > 0 else y,
            "frame_id": r.get("frame_id"),
            "raw_track_id": r.get("raw_track_id") or r.get("id"),
            "continuous": r.get("continuous"),
            "mm_per_px": mmp,
        })
    if not per_pid:
        raise ValueError(f"jsonl 中没有有效字段 pid/ts/{pk}")
    all_dt = np.concatenate([
        np.diff([s["ts"] for s in v]) for v in per_pid.values()
        if len(v) > 1
    ]) if any(len(v) > 1 for v in per_pid.values()) else np.asarray([1 / 25.0])
    dt = float(np.median(all_dt)) if len(all_dt) else 1 / 25.0
    if dt <= 0:
        dt = 1 / 25.0
    targets = {}
    for pid, v in sorted(per_pid.items()):
        v.sort(key=lambda s: s["ts"])
        mmp = next((s["mm_per_px"] for s in reversed(v) if s["mm_per_px"] > 0), 0.0)
        targets[pid] = {"mm_per_px": round(mmp, 6), "n_valid": len(v), "samples": v}
    return {"fps": round(1.0 / dt, 5), "targets": targets,
            "video": "uploaded displacement.jsonl", "source": "jsonl"}


def _one(src, key, default=None):
    """同时兼容 parse_qs 的 list 值与 JSON body 的标量值。"""
    v = src.get(key, default)
    if isinstance(v, (list, tuple)):
        return v[0] if v else default
    return v


def _parse_video_args(src) -> dict:
    return {
        "conf": float(_one(src, "conf", 0) or 0) or None,
        "iou": float(_one(src, "iou", 0) or 0) or None,
        "device": _one(src, "device") or None,
        "axis": _one(src, "axis") or "dy",
        "window_s": float(_one(src, "window_s", 30) or 30),
        "snr": float(_one(src, "snr", 5) or 5),
        "min_samples": int(_one(src, "min_samples", 32) or 32),
        "min_span_s": float(_one(src, "min_span_s", 10) or 10),
        "max_frames": int(_one(src, "max_frames", 0) or 0),
        "frame_stride": int(_one(src, "frame_stride", 1) or 1),
        "calib": _one(src, "calib") or None,
    }


def _launch_job(video_path: str, args: dict, name: str, size_mb: float) -> str:
    """建任务目录并启动后台线程，返回 job_id。"""
    jid = uuid.uuid4().hex[:12]
    os.makedirs(os.path.join(UPLOADS, jid), exist_ok=True)
    with JLOCK:
        JOBS[jid] = {
            "id": jid, "status": "running", "stage": "prepare",
            "done": 0, "total": 1, "message": "排队中",
            "created_at": time.time(),
            "video": name, "size_mb": round(size_mb, 2), "args": args,
        }
    threading.Thread(target=_run_job, args=(jid, video_path, args),
                     daemon=True).start()
    return jid


def _run_job(job_id: str, video_path: str, args: dict) -> None:
    def prog(stage: str, done: int, total: int, msg: str) -> None:
        with JLOCK:
            j = JOBS[job_id]
            j["stage"], j["done"], j["total"], j["message"] = stage, done, total, msg

    try:
        prog("prepare", 0, 1, "准备参数")
        cfg = _settings()
        gate_cfg = load_gate_config(args.get("calib") or "") if args.get("calib") else None
        measurement = vm.measure_video(
            video_path, cfg,
            conf=args.get("conf"), iou=args.get("iou"), device=args.get("device"),
            max_frames=int(args.get("max_frames") or 0),
            frame_stride=max(1, int(args.get("frame_stride") or 1)),
            progress=prog)
        # 落位移 jsonl，供复核与离线再分析
        out_jsonl = os.path.join(UPLOADS, job_id, "displacement.jsonl")
        with open(out_jsonl, "w", encoding="utf-8") as fh:
            for pid, info in measurement["targets"].items():
                for s in info["samples"]:
                    fh.write(json.dumps({"pid": pid, **s}, ensure_ascii=False) + "\n")
        prog("spectrum", 0, 1, "提取主频")
        freq = vm.extract_frequencies(
            measurement, axis=args.get("axis") or "dy",
            window_s=float(args.get("window_s") or 30.0),
            snr=float(args.get("snr") or 5.0),
            min_samples=int(args.get("min_samples") or 32),
            min_span_s=float(args.get("min_span_s") or 10.0),
            gate_cfg=gate_cfg)
        warnings = check_fingerprint(gate_cfg, measurement["fps"])
        with JLOCK:
            j = JOBS[job_id]
            j.update({
                "status": "done", "stage": "done", "done": 1, "total": 1,
                "message": "完成", "finished_at": time.time(),
                "result": _summarize_result(freq),
                "spectrum": _spectrum_payload(freq),
                "measurement": {
                    "video": measurement["video"], "fps": measurement["fps"],
                    "n_frames": measurement["n_frames"],
                    "n_processed": measurement["n_processed"],
                    "n_detections": measurement["n_detections"],
                    "n_located": measurement["n_located"],
                    "targets": {pid: {"mm_per_px": t["mm_per_px"],
                                      "n_valid": t["n_valid"],
                                      "duration_s": t["duration_s"]}
                                for pid, t in measurement["targets"].items()},
                    "settings": measurement["settings"],
                },
                "warnings": warnings,
                "jsonl": os.path.basename(out_jsonl),
            })
    except Exception as exc:  # noqa: BLE001
        with JLOCK:
            JOBS[job_id].update({
                "status": "error", "stage": "error", "message": str(exc),
                "trace": traceback.format_exc(limit=8), "finished_at": time.time()})
        print(f"[job {job_id}] 失败: {exc}")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "VideoGate/1.0"

    def _send(self, code: int, payload, ctype="application/json; charset=utf-8"):
        if isinstance(payload, str):
            data = payload.encode("utf-8")
        elif isinstance(payload, bytes):
            data = payload
        else:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _body_bytes(self, limit: int) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return b""
        if length > limit:
            raise ValueError(f"上传内容超过 {limit // (1024*1024)}MB 上限")
        return self.rfile.read(length)

    def _json_body(self) -> dict:
        raw = self._body_bytes(16 * 1024 * 1024).decode("utf-8", errors="replace")
        try:
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            raise ValueError(f"请求体不是合法 JSON：{exc}") from exc

    def _error(self, exc: Exception):
        self._send(500, {"error": str(exc), "trace": traceback.format_exc(limit=6)})

    def log_message(self, fmt, *args):
        if len(args) > 1 and str(args[1]).startswith((" 4", " 5")):
            super().log_message(fmt, *args)

    # ------------------------------------------------------------- GET
    def do_GET(self):
        u = urlparse(self.path)
        try:
            if u.path in ("/", "/index.html"):
                with open(os.path.join(ROOT, "video_gate_ui.html"), "r",
                          encoding="utf-8") as fh:
                    self._send(200, fh.read(), "text/html; charset=utf-8")
                return
            if u.path == "/api/health":
                self._send(200, {"ok": True, "root": ROOT})
                return
            if u.path == "/api/defaults":
                cfg = _settings()
                self._send(200, {
                    "defaults": {
                        "conf": float(cfg["det"].get("confidence", 0.25)),
                        "iou": float(cfg["det"].get("iou", 0.6)),
                        "device": cfg["det"].get("device", "cpu"),
                        "axis": "dy",
                        "window_s": float((cfg.get("fft") or {}).get("window_s", 30.0)),
                        "snr": float((cfg.get("fft") or {}).get("snr", 5.0)),
                        "min_samples": int((cfg.get("fft") or {}).get("min_samples", 32)),
                        "min_span_s": float((cfg.get("fft") or {}).get("min_span_s", 10.0)),
                        "max_frames": 0, "frame_stride": 1,
                    },
                    "calibs": _calib_options(),
                    "config_path": cfg["config_path"],
                    "real_diameter_mm": float(
                        (cfg.get("calibration") or {}).get("real_diameter_mm", 100.0)),
                })
                return
            if u.path == "/api/job":
                jid = (parse_qs(u.query).get("id") or [""])[0]
                with JLOCK:
                    j = JOBS.get(jid)
                if not j:
                    self._send(404, {"error": "任务不存在"})
                    return
                self._send(200, {k: v for k, v in j.items() if k != "trace"})
                return
            if u.path == "/api/download":
                jid = (parse_qs(u.query).get("id") or [""])[0]
                path = os.path.join(UPLOADS, jid, "displacement.jsonl")
                if not os.path.isfile(path):
                    self._send(404, {"error": "该任务尚无位移文件"})
                    return
                with open(path, "rb") as fh:
                    self._send(200, fh.read(), "application/octet-stream; charset=utf-8")
                return
            self._send(404, {"error": "not found"})
        except Exception as exc:  # pragma: no cover
            self._error(exc)

    # ------------------------------------------------------------ POST
    def do_POST(self):
        u = urlparse(self.path)
        try:
            if u.path == "/api/calib/upload":
                raw = self._body_bytes(8 * 1024 * 1024)
                if not raw:
                    raise ValueError("标定文件内容为空")
                try:
                    doc = json.loads(raw.decode("utf-8", errors="replace"))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"标定文件不是合法 JSON：{exc}") from exc
                name = _safe_name((parse_qs(u.query).get("name") or ["upload.json"])[0])
                if not name.endswith(".json"):
                    name += ".json"
                path = os.path.join(CALIBS, name)
                with open(path, "wb") as fh:
                    fh.write(raw)
                cfgd = load_gate_config(path)
                self._send(200, {
                    "path": path, "name": name,
                    "loaded": bool(cfgd),
                    "verdict": (cfgd or {}).get("verdict", ""),
                    "cam_id": (cfgd or {}).get("cam_id", ""),
                    "fingerprint": (cfgd or {}).get("fingerprint") or {},
                    "params": (cfgd or {}).get("params") or {},
                    "warning": ("标定未通过 READY 校验，仅供试验，勿作正式判定"
                                if (cfgd or {}).get("verdict") != "READY" else ""),
                })
                return

            # 二进制上传必须先于 _json_body()，否则视频字节会被当 JSON 解析
            if u.path == "/api/video/start":
                raw = self._body_bytes(MAX_VIDEO_BYTES)
                if not raw:
                    raise ValueError("视频文件为空")
                q = parse_qs(u.query)
                name = _safe_name((q.get("name") or ["upload.mp4"])[0], VIDEO_EXT)
                jid = uuid.uuid4().hex[:12]
                jdir = os.path.join(UPLOADS, jid)
                os.makedirs(jdir, exist_ok=True)
                video_path = os.path.join(jdir, name)
                with open(video_path, "wb") as fh:
                    fh.write(raw)
                args = _parse_video_args(q)
                with JLOCK:
                    JOBS[jid] = {
                        "id": jid, "status": "running", "stage": "prepare",
                        "done": 0, "total": 1, "message": "已上传，排队中",
                        "created_at": time.time(),
                        "video": name, "size_mb": round(len(raw) / 1048576, 2),
                        "args": args,
                    }
                threading.Thread(target=_run_job, args=(jid, video_path, args),
                                 daemon=True).start()
                self._send(200, {"id": jid, "status": "running"})
                return

            body = self._json_body()

            # 本机路径直读：本地工具的核心入口，避免大视频走 HTTP 上传受限
            if u.path == "/api/video/start_path":
                path = str(body.get("path") or "").strip().strip('"')
                if not path:
                    raise ValueError("请填写本机视频路径")
                if not os.path.isabs(path):
                    path = os.path.join(_PROJECT_ROOT, path)
                path = os.path.normpath(path)
                if not os.path.isfile(path):
                    raise ValueError(f"视频文件不存在：{path}")
                if os.path.splitext(path)[1].lower() not in VIDEO_EXT:
                    raise ValueError(f"不支持的视频格式：{os.path.splitext(path)[1]}")
                low = path.lower()
                if any(low.startswith(b) for b in ("c:\\windows", "c:\\program files",
                                                   "c:\\users\\all users", "/system", "/etc")):
                    raise ValueError("不允许读取系统目录")
                args = _parse_video_args(body)
                mb = os.path.getsize(path) / 1048576
                jid = _launch_job(path, args, os.path.basename(path), mb)
                self._send(200, {"id": jid, "status": "running",
                                 "path": path, "size_mb": round(mb, 2)})
                return

            if u.path == "/api/calib/validate":
                path = str(body.get("path") or "")
                cfgd = load_gate_config(path)
                if not cfgd:
                    self._send(200, {"loaded": False,
                                     "error": "文件不存在/格式不符/schema_version≠1"})
                    return
                self._send(200, {
                    "loaded": True, "verdict": cfgd.get("verdict", ""),
                    "cam_id": cfgd.get("cam_id", ""),
                    "fingerprint": cfgd.get("fingerprint") or {},
                    "params": cfgd.get("params") or {},
                    "warnings": check_fingerprint(cfgd, float(body.get("fps") or 0)),
                })
                return

            if u.path == "/api/analyze_jsonl":
                content = str(body.get("content") or "")
                if not content.strip():
                    raise ValueError("请先上传 displacement.jsonl")
                meas = _jsonl_to_measurement(content, str(body.get("axis") or "dy"))
                gate_cfg = load_gate_config(str(body.get("calib") or "")) if body.get("calib") else None
                freq = vm.extract_frequencies(
                    meas, axis=str(body.get("axis") or "dy"),
                    window_s=float(body.get("window_s") or 30.0),
                    snr=float(body.get("snr") or 5.0),
                    min_samples=int(body.get("min_samples") or 32),
                    min_span_s=float(body.get("min_span_s") or 10.0),
                    gate_cfg=gate_cfg)
                self._send(200, {
                    "result": _summarize_result(freq),
                    "spectrum": _spectrum_payload(freq),
                    "measurement": {"fps": meas["fps"], "targets": {
                        pid: {"n_valid": t["n_valid"], "mm_per_px": t["mm_per_px"]}
                        for pid, t in meas["targets"].items()}},
                    "warnings": check_fingerprint(gate_cfg, meas["fps"]),
                })
                return

            self._send(404, {"error": "not found"})
        except Exception as exc:  # pragma: no cover
            self._error(exc)


def main() -> int:
    ap = argparse.ArgumentParser(description="靶标视频主频提取本地服务")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8767)
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"靶标视频主频提取服务： http://{args.host}:{args.port}")
    print(f"数据与上传目录：{ROOT}")
    print("按 Ctrl+C 停止")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
