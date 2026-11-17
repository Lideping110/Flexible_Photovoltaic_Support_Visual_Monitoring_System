# -*- coding: utf-8 -*-
"""四门控频率分析 — 本地 Web 服务（只用标准库 + numpy）。

启动（在项目根目录执行，用项目 .venv）：
    .venv/Scripts/python.exe sensor_data/accelerometer/10-34-27-942/gate_server.py --port 8766
    # 标定参数自动生效：同目录存在 accel_calib.json 时自动加载；也可显式指定
    ... --calib accel_calib.json
然后浏览器打开 http://127.0.0.1:8766

接口：
    GET  /                       前端页面
    GET  /api/defaults           默认参数（已按标定覆盖） + 标定元信息 + 示例文件
    GET  /api/calib              当前完整标定文件
    GET  /api/sample?name=...    读取目录内白名单示例（data_0.csv / data_0.txt）
    POST /api/columns            {content} -> 可解析的数值列
    POST /api/calib/validate     {content(JSON文本)} -> 校验标定并返回可回填参数
    POST /api/analyze            {content, filename, axis, params, calib?} -> 完整分析结果
只绑定 127.0.0.1；上传内容不落盘、不执行；示例文件走白名单，禁止路径穿越。
标定加载只做校验与告警，不因 L3 未填而拒启动（现场不能因少填一项整链停摆）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import frequency_gate as fg  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLE_WHITELIST = {"data_0.csv", "data_0.txt"}

# 服务进程只读的标定状态；ThreadingHTTPServer 下仅并发读取，不写入
ACTIVE: dict = {"calib_path": None, "calib": None, "defaults": None, "meta": None}


def setup_calibration(path: str | None) -> None:
    """加载标定文件；未指定时自动探测同目录 accel_calib.json。"""
    if not path:
        auto = os.path.join(ROOT, "accel_calib.json")
        path = auto if os.path.isfile(auto) else None
    if not path:
        print("未发现标定文件：使用内置 DEFAULTS（建议先跑 calibrate_accel.py）")
        return
    full = path if os.path.isabs(path) else os.path.join(ROOT, path)
    with open(full, "r", encoding="utf-8") as fh:
        calib = json.load(fh)
    _, meta = fg.load_calibration(calib)
    defaults = fg.merge_params({"calib": calib})
    defaults.pop("_calib", None)          # 前端 fillForm 只认 PARAM_KEYS
    ACTIVE.update({"calib_path": full, "calib": calib,
                   "defaults": defaults, "meta": meta})
    s = meta.get("sensor") or {}
    print(f"已加载标定：{s.get('id')} / {s.get('model')} | "
          f"本底 {meta.get('noise_floor_mg')}mg | 有效至 {meta.get('valid_until')}")
    for w in meta.get("warnings") or []:
        print(f"  [标定告警] {w}")


def _effective_defaults() -> dict:
    return dict(ACTIVE["defaults"] or fg.DEFAULTS)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "FrequencyGate/1.0"

    # ------------------------------------------------------------ helpers
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

    def _json_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"请求体不是合法 JSON：{exc}") from exc

    def _error(self, exc: Exception):
        self._send(500, {"error": str(exc), "trace": traceback.format_exc(limit=6)})

    def log_message(self, fmt, *args):
        # 只打印错误与关键请求，避免刷屏
        if " 4" in (args[1] if len(args) > 1 else "") or " 5" in (args[1] if len(args) > 1 else ""):
            super().log_message(fmt, *args)

    # ------------------------------------------------------------ GET
    def do_GET(self):
        u = urlparse(self.path)
        try:
            if u.path in ("/", "/index.html"):
                with open(os.path.join(ROOT, "gate_ui.html"), "r", encoding="utf-8") as fh:
                    self._send(200, fh.read(), "text/html; charset=utf-8")
                return
            if u.path == "/api/defaults":
                samples = [n for n in SAMPLE_WHITELIST
                           if os.path.isfile(os.path.join(ROOT, n))]
                self._send(200, {"defaults": _effective_defaults(),
                                 "param_keys": list(fg.PARAM_KEYS),
                                 "samples": sorted(samples), "root": ROOT,
                                 "calib": ACTIVE["meta"],
                                 "calib_path": ACTIVE["calib_path"]})
                return
            if u.path == "/api/calib":
                if not ACTIVE["calib"]:
                    self._send(404, {"error": "服务未加载标定文件"})
                    return
                self._send(200, {"calib": ACTIVE["calib"], "meta": ACTIVE["meta"],
                                 "path": ACTIVE["calib_path"]})
                return
            if u.path == "/api/sample":
                name = (parse_qs(u.query).get("name") or [""])[0]
                if name not in SAMPLE_WHITELIST:
                    self._send(403, {"error": "只允许读取目录内的示例文件"})
                    return
                path = os.path.join(ROOT, name)
                if not os.path.isfile(path):
                    self._send(404, {"error": f"示例文件不存在：{name}"})
                    return
                with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
                    self._send(200, {"filename": name, "content": fh.read()})
                return
            if u.path == "/api/health":
                self._send(200, {"ok": True})
                return
            self._send(404, {"error": "not found"})
        except Exception as exc:  # pragma: no cover
            self._error(exc)

    # ------------------------------------------------------------ POST
    def do_POST(self):
        u = urlparse(self.path)
        try:
            body = self._json_body()
            content = body.get("content") or ""
            if u.path == "/api/calib/validate":
                if not content.strip():
                    raise ValueError("标定文件内容为空")
                try:
                    doc = json.loads(content)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"标定文件不是合法 JSON：{exc}") from exc
                cparams, meta = fg.load_calibration(doc)
                defaults = fg.merge_params({"calib": doc})
                defaults.pop("_calib", None)
                self._send(200, {"params": cparams, "defaults": defaults,
                                 "meta": meta, "calib": doc})
                return
            if u.path == "/api/columns":
                if not content.strip():
                    raise ValueError("文件内容为空")
                header, data = fg.parse_table(content)
                cols = fg.numeric_columns(header, data)
                default = next((c for c in cols if "加速度Z" in c), cols[0] if cols else "")
                self._send(200, {"axes": cols, "default": default,
                                 "columns": header, "rows": len(data)})
                return
            if u.path == "/api/analyze":
                if not content.strip():
                    raise ValueError("请先上传 csv / txt 文件")
                params = dict(body.get("params") or {})
                params["axis"] = body.get("axis") or ""
                # 标定作为基底注入，令运行时指纹校验生效；前端手填值优先级更高
                calib = body.get("calib") or ACTIVE["calib"]
                if calib and body.get("use_calib", True):
                    params["calib"] = calib
                res = fg.analyze_text(content, str(body.get("filename") or "upload.csv"), params)
                self._send(200, res)
                return
            self._send(404, {"error": "not found"})
        except Exception as exc:  # pragma: no cover
            self._error(exc)


def main() -> int:
    ap = argparse.ArgumentParser(description="四门控频率分析本地服务")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--calib", default="",
                    help="标定 JSON 路径；留空则自动探测同目录 accel_calib.json")
    args = ap.parse_args()
    try:
        setup_calibration(args.calib or None)
    except Exception as exc:
        print(f"[标定加载失败] {exc}；回退到内置 DEFAULTS")
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"四门控频率分析服务已启动： http://{args.host}:{args.port}")
    print(f"数据目录：{ROOT}")
    if ACTIVE["calib_path"]:
        print(f"标定文件：{ACTIVE['calib_path']}")
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
