"""实时监控大屏（方案 C：dashboard 直连 RTSP 子码流）：左侧立柱倾角监测 + 右侧视频测振。

零第三方依赖（仅 cv2 用于拉流/编码，标准库 http.server 承载）。与推理进程彻底解耦：
main 进程只做推理、把位移/频谱/测量结果写 jsonl，不发布任何图片；本服务对 RTSP 相机
自行 cv2 直连子码流、以 MJPEG 把原始画面推给浏览器（见 RtspStreamManager），对文件源
相机仍用 <video> 裸播 main 收尾生成的标注视频。两进程是相机的两个独立消费者，互不通信。

启动：
    uv run python dashboard.py --config config.yaml --port 8000
    # 浏览器打开 http://127.0.0.1:8000

路由：
    GET  /                  大屏页面（左右分屏）
    GET  /api/state         JSON：左(立柱 初始+当前1s维持) / 右(测振 30s更新)
    GET  /api/marker_window?cam=<id>&t=<sec>
                            视频文件播放联动：当前固定段曲线（横轴 0~30s）
                            + 最近完成/EOF 尾段频谱（t 省略 → 最新窗口）
    GET  /video/<cam_id>   RTSP 相机 MJPEG 实时帧流（dashboard 直连子码流，纯原始画面）
"""
import argparse
import json
import math
import os
import re
import struct
import threading
import time
from pathlib import Path

from utils.config import load_config

try:
    from utils.persist import load_json
except Exception:  # 兜底：本地实现
    def load_json(path):
        p = Path(path)
        if not p.is_file():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None


def to_f(x):
    """转 float，NaN/None/空 → None（JSON 不允许 NaN）。"""
    if x is None:
        return None
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def cameras_of(cfg):
    pole = next((c for c in cfg.get("cameras", []) if c.get("task") == "pole"), None)
    marker = next((c for c in cfg.get("cameras", []) if c.get("task") == "marker"), None)
    return (pole["id"] if pole else None, marker["id"] if marker else None)


def _tail_monotonic(pairs, tol: float = 1e-9):
    """从尾部截出「最后一段 ts 单调不降」的连续记录。

    recorder 以追加方式写 jsonl 且不截断，同一相机多次运行（改 config/重跑）会让
    文件里叠出多段各自从 ~0 开始的时序；若把多段混在一起，采样间隔中位数会被拉小，
    FFT 频率轴随之错乱（实测会把 0.06Hz 主频算成 25.9Hz、索力算出千万级）。
    故只保留最后一段——它才是"本次运行"的结果。
    """
    if not pairs:
        return pairs
    i = len(pairs) - 1
    while i > 0 and pairs[i - 1][0] <= pairs[i][0] + tol:
        i -= 1
    return pairs[i:]


def load_recent_displacement(path: Path, window_s: float = 30.0, max_lines: int = 4000):
    """返回 {pid: [{"ts", "dx", "dy"}]} —— 最近 window_s 秒、每个目标一条序列。"""
    if not path.is_file():
        return {}
    seq = {}
    try:
        with path.open("r", encoding="utf-8") as f:
            lines = f.readlines()[-max_lines:]
    except Exception:
        return {}
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
        except Exception:
            continue
        pid = r.get("pid")
        ts = to_f(r.get("ts"))
        if pid is None or ts is None:
            continue
        seq.setdefault(pid, []).append(
            (ts, to_f(r.get("cum_dx_mm")) or 0.0, to_f(r.get("cum_dy_mm")) or 0.0))
    if not seq:
        return {}
    # 每目标只保留"最后一段"（本次运行），再按全局最新 ts 裁到最近 window_s
    tail = {pid: _tail_monotonic(v) for pid, v in seq.items()}
    tail = {pid: v for pid, v in tail.items() if v}
    if not tail:
        return {}
    now_ts = max(t for v in tail.values() for t, _, _ in v)
    out = {}
    for pid, v in tail.items():
        keep = [{"ts": t, "dx": dx, "dy": dy}
                for t, dx, dy in v if now_ts - t <= window_s]
        if keep:
            out[pid] = keep
    return out


def load_spectrum_rows(path: Path, max_lines: int = None):
    if not path.is_file():
        return []
    rows = []
    try:
        with path.open("r", encoding="utf-8") as f:
            lines = f if max_lines is None else f.readlines()[-max_lines:]
            for ln in lines:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rows.append(json.loads(ln))
                except Exception:
                    continue
    except Exception:
        return []
    return rows


def _cam_info(cfg):
    """把 config 的 cameras 汇总成 {cam_id: {branch, source_type, source_url}}。

    source_type:
      - "stream"：RTSP/HTTP 等实时拉流 → 大屏由 dashboard 直连子码流、MJPEG 裸播
                 （/video/<cam>，纯原始画面）；
      - "file"  ：本地视频文件 → 大屏用 <video> 原生裸播（/media/<相对路径>）。
    """
    info = {}
    for c in cfg.get("cameras", []):
        cid = c.get("id")
        if not cid:
            continue
        url = c.get("url", "") or ""
        stype = "stream" if re.match(r"^(rtsp|rtsps|http|https)://", url, re.I) else "file"
        info[cid] = {"branch": c.get("task"), "source_type": stype, "source_url": url}
    return info


def _cam_live_source(cid, cam_info, output_dir=None):
    """由 camera 配置推导其实时播放源 {type, media}，供大屏实时视图下拉框按路切换。

    - file  ：优先 main 收尾生成的每相机标注视频
              /media/output/<branch>/<cid>/<cid>_annotated.mp4；
              其次 viz_overlay 产物 /media/output/viz/<cid>_annotated.mp4；
              最后回退原视频 /media/<相对路径>；前端用 <video> 播放。
    - stream：dashboard 后端 cv2 直连子码流、以 MJPEG 推送的原始画面
              /video/<cid>；浏览器用 <img> 直接播放，纯原始、无叠加。实时分析面板
              仍由 /api/state 每秒驱动，与视频元素解耦。main 进程不发布任何图片，
              实时画面完全由 dashboard 自行拉流（与 main 解耦，见 RtspStreamManager）。
    """
    info = (cam_info or {}).get(cid) or {}
    stype = info.get("source_type", "stream")
    url = info.get("source_url")
    branch = info.get("branch") or ""
    if stype == "file" and url:
        cam_dir = (Path(output_dir) / branch / cid) if output_dir else None
        annotated = None
        if cam_dir is not None:
            # 按优先级找已编码的标注视频（检测框+拟合直线已画进画面）。
            # 注意 output_dir 本身已是 .../output，故文件路径不再拼 output/ 前缀，
            # 而对外 URL 仍带 output/（/media 相对项目根）。
            for url_rel, fs_rel in (
                (f"output/{branch}/{cid}/{cid}_annotated.mp4",
                 f"{branch}/{cid}/{cid}_annotated.mp4"),
                (f"output/viz/{cid}_annotated.mp4",
                 f"viz/{cid}_annotated.mp4"),
            ):
                if (Path(output_dir) / fs_rel).is_file():
                    annotated = f"/media/{url_rel}"
                    break
        if annotated:
            # 标注视频已生成 → 直接展示处理结果（循环播放）
            return {"type": "file", "media": annotated}
        # 尚未生成：按处理阶段给占位，绝不回退播放原始录像（处理中看原片无意义）
        if cam_dir is not None:
            if (cam_dir / ".export_failed").is_file():
                return {"type": "file_failed", "media": None}
            if (cam_dir / ".rec_done").is_file():
                # Recorder 已确认该路结果全部落盘，supervisor 正在导出标注视频
                return {"type": "file_processing", "media": None,
                        "hint": "标注视频生成中…"}
        return {"type": "file_processing", "media": None,
                "hint": "录像文件处理中，跑完后自动展示处理结果"}
    elif stype == "stream":
        # 实时流：dashboard 后端 cv2 直连子码流，MJPEG 推原始画面（零重编码/无叠加）。
        # 浏览器 <img src="/video/<cid>"> 播放；main 不发布图片，两进程解耦。
        return {"type": "stream", "media": f"/video/{cid}"}
    else:
        return {"type": stype, "media": None}


def _pole_timeline(pdir: Path):
    """读 measurements.jsonl，按 frame_id 分组成按秒快照时间线。

    返回 [{frame_id, ts, poles:{pid:{lr_deg,fb_deg,total_deg,delta_lr_deg,delta_fb_deg,status,alarm}}}]
    按 frame_id 升序，供前端 <video> 播放时按 currentTime 联动"当前状态"表。
    """
    p = Path(pdir) / "measurements.jsonl"
    if not p.is_file():
        return []
    by_frame = {}
    try:
        with p.open("r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    r = json.loads(ln)
                except Exception:
                    continue
                fid = r.get("frame_id")
                if fid is None:
                    continue
                by_frame.setdefault(int(fid), []).append(r)
    except Exception:
        return []
    out = []
    for fid in sorted(by_frame):
        ts = None
        poles = {}
        for d in by_frame[fid]:
            if ts is None:
                ts = to_f(d.get("ts"))
            pid = d.get("pid")
            if pid is None:
                continue
            poles[pid] = {
                "lr_deg": round(to_f(d.get("lr_deg")) or 0.0, 3),
                "fb_deg": round(to_f(d.get("fb_deg")) or 0.0, 3),
                "total_deg": round(to_f(d.get("total_deg")) or 0.0, 3),
                "delta_lr_deg": round(to_f(d.get("delta_lr_deg")) or 0.0, 3),
                "delta_fb_deg": round(to_f(d.get("delta_fb_deg")) or 0.0, 3),
                "status": d.get("status"),
                "alarm": bool(d.get("alarm")),
            }
        out.append({"frame_id": fid, "ts": ts, "poles": poles})
    return out


# ---- 全量位移序列（带 mtime 缓存）----
# 供"视频播放联动"按 currentTime 取窗口用：displacement.jsonl 是逐帧结果，
# 文件源的视频总长可能几百上千秒，不能把全量塞进 /api/state 每秒轮询，
# 因此单独开 /api/marker_window 按需查询，这里做进程内缓存避免重复解析。
_DISP_CACHE = {}
_DISP_LOCK = threading.Lock()


def _load_displacement_cached(path: Path, max_lines: int = 200000):
    """读取全量位移序列 → ({pid: [(ts, dx, dy), ...] 按 ts 升序}, t_max, t_min)。

    以 (mtime, size) 为键缓存：推理进程追加写文件后会变更 mtime，自动失效重读。
    """
    try:
        st = path.stat()
    except OSError:
        return {}, None, None
    key = str(path)
    sig = (st.st_mtime_ns, st.st_size)
    with _DISP_LOCK:
        hit = _DISP_CACHE.get(key)
        if hit and hit["sig"] == sig:
            return hit["series"], hit["t_max"], hit["t_min"]
    seq = {}
    try:
        with path.open("r", encoding="utf-8") as f:
            for ln in f.readlines()[-max_lines:]:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    r = json.loads(ln)
                except Exception:
                    continue
                pid = r.get("pid")
                ts = to_f(r.get("ts"))
                if pid is None or ts is None:
                    continue
                seq.setdefault(pid, []).append(
                    (ts, to_f(r.get("cum_dx_mm")) or 0.0,
                     to_f(r.get("cum_dy_mm")) or 0.0))
    except Exception:
        return {}, None, None
    # 只保留每目标"最后一段"（本次运行）——不排序！排序会把跨运行的残留段
    # 按 ts 交错掺进来，破坏采样间隔，直接毁掉 FFT 结果。
    rows = {pid: _tail_monotonic(v) for pid, v in seq.items()}
    rows = {pid: v for pid, v in rows.items() if v}
    if not rows:
        return {}, None, None
    t_max = max(s[0] for v in rows.values() for s in v)
    t_min = min(s[0] for v in rows.values() for s in v)
    with _DISP_LOCK:
        _DISP_CACHE[key] = {"sig": sig, "series": rows,
                            "t_max": t_max, "t_min": t_min}
    return rows, t_max, t_min


def _window_fft(t, y, snr_min):
    """单轴窗口主频：复用推理链路的同一套 FFT（保证与 spectrum.jsonl 数值同源）。"""
    try:
        from marker_subpixel_tracker.spectrum import _segment_fft
    except Exception:
        return None
    if len(t) < 4:
        return None
    try:
        return _segment_fft(t, y, snr_min)
    except Exception:
        return None


def _window_spectrum(t, y, snr_min):
    """单轴功率谱（含主频/功率数组）：复用推理链路的同一套 FFT。

    与 _window_fft 同源，只是额外把功率谱降采样下发，供前端直接画曲线——
    这样前端画的形状、标的主频，与表格主频、spectrum.jsonl 都来自同一算法，
    不存在"第三套口径"。
    """
    try:
        from marker_subpixel_tracker.spectrum import _segment_spectrum
    except Exception:
        return None
    if len(t) < 4:
        return None
    try:
        return _segment_spectrum(t, y, snr_min)
    except Exception:
        return None


def _display_window(t, t_max, window_s, min_span_s, flush):
    """固定不重叠窗口：给定播放时刻 t，决定应展示哪个窗口 [t0, t1)。

    分段方式与主链路 SpectrumAnalyzer 完全一致（seg = int(t//window_s)），
    但本函数用于"播放联动展示"——展示的是「最近一个已完成窗口」的结果，
    下一窗口完成时再覆盖前者（即 0-30 算一次、30-60 再算一次将前面的更新掉）。
    源不足一个窗长时（flush=末尾），用全部数据算一次。

    返回 (t0, t1) 或 None（仍在累积、不出图）。
    """
    if t_max is None or t_max <= 0:
        return None
    W = float(window_s)
    last_k = int(t_max // W)
    at_end = bool(flush)
    if at_end:
        # t_max 恰好落在边界 → 末数据窗是 last_k-1（避免空窗 [30,30.1]
        # 被当成独立窗）；非 flush 的实时 latest 路径不走这里。
        if last_k * W >= t_max - 1e-9:
            last_k -= 1
        k = last_k
        # EOF 尾段不能因不足 min_span_s 回退到前窗；按实际剩余样本分析。
    else:
        if t is None:
            # 实时源没有“播放末尾”语义：t=None 时只展示已经完整结束的
            # 固定窗口。当前正在累积的窗口必须继续显示为“等待首窗/窗口累积”，
            # 不能用 flush 把部分数据提前当成最终频谱。
            k = int(t_max // W) - 1
        else:
            k = int(t // W) - 1
            if k > last_k:
                k = last_k
    if k < 0:
        return None
    t0 = k * W
    t1 = min((k + 1) * W, t_max)
    return (t0, t1)


def _marker_f1_ref(spec_rows):
    """读取每个目标首个垂向(dy)频谱，避免水平轴影响垂向分析基准。"""
    # spectrum.jsonl 仍记录 dx/dy 两轴；看板的主频口径固定为 dy。
    by_pid = {}
    for r in spec_rows:
        pid, axis = r.get("pid"), r.get("axis")
        if pid and axis == "dy":
            by_pid.setdefault(pid, []).append(r)
    best = {}   # pid -> (ts, snr, f1)
    for pid, rows in by_pid.items():
        times = [to_f(r.get("ts")) for r in rows]
        pairs = [(t, i) for i, t in enumerate(times) if t is not None]
        if not pairs:
            continue
        i = pairs[-1][1]
        while i > 0 and times[i - 1] is not None and \
                times[i - 1] <= times[i] + 1e-9:
            i -= 1
        for r in rows[i:]:
            f1 = to_f(r.get("f1_hz"))
            if not f1:
                continue
            ts = to_f(r.get("ts")) or 0.0
            snr = to_f(r.get("snr")) or 0.0
            cur = best.get(pid)
            if (cur is None or ts < cur[0] - 1e-9
                    or (abs(ts - cur[0]) <= 1e-9 and snr > cur[1])):
                best[pid] = (ts, snr, f1)
    return {pid: v[2] for pid, v in best.items()}


# 出主频 / 出频谱所需的最短样本跨度（秒）。与 marker.fft 的 min_samples、snr 一起
# 构成频谱类结果的"可出图门限"。集中在此单一来源，避免 /api/state 与
# /api/marker_window 各写一套判据后分叉；实际取值由 config.marker.fft.min_span_s 覆盖。
_MARKER_MIN_SPAN_S = 10.0


def _fft_gate(t_lists, window_s: float = 30.0, min_samples: int = 32,
              min_span_s: float = _MARKER_MIN_SPAN_S):
    """频谱图是否已具备出图条件——判据与主频计算**完全同一条**。

    频谱图（前端画）与主频（后端算）若各用一套门限，就会出现"表格里已经有主频、
    频谱图却还是空白"这种新的前后端分叉。故统一：窗口内任一目标的样本数 ≥
    min_samples 且样本实际跨度 ≥ min(min_span_s, window_s)，即可出图。

    为什么跨度门限取 min(min_span_s, window_s) 而不是死守 window_s：源比 window_s
    短时（如 11.9s 的短视频），拿名义 30s 去判会**永远判未满窗**，频谱永不出图。
    主链路 SpectrumAnalyzer 在冲刷末窗时同样按实际样本成窗，口径一致。

    返回 (ready, cover_s, need_s)：cover_s 为当前各目标里最长的样本跨度（秒）。
    """
    need = min(float(min_span_s), float(window_s))
    cover = 0.0
    ready = False
    for t in (t_lists or []):
        if not t or len(t) < 2:
            continue
        span = float(t[-1]) - float(t[0])
        if span > cover:
            cover = span
        if len(t) >= int(min_samples) and span >= need - 1e-9:
            ready = True
    return ready, round(cover, 2), round(need, 1)


def _marker_window(mdir: Path, t=None, window_s: float = 30.0,
                   snr_min: float = 5.0, min_samples: int = 32,
                   min_span_s: float = _MARKER_MIN_SPAN_S, f1_ref=None,
                   flush: bool = False):
    """固定不重叠 window_s 窗口：按播放时刻 t 取「当前已完成/末尾窗口」的曲线与分析。

    分段方式与主链路 SpectrumAnalyzer 完全一致（seg = int(t//window_s)），但本函数
    用于"播放联动展示"：展示的是**最近一个已完成窗口**的结果，下一窗口完成时再覆盖前者
    （即 0-30 算一次、30-60 算一次将前面的更新掉）。源不足一个窗长时（flush=末尾），
    用全部数据算一次。绝不做滑动窗重算——否则会跟 spectrum.jsonl 的口径分叉。

    返回结构供前端播放联动直接消费：
      {"t","t0","t1","t_max","window_s","series":{pid:{t,dx,dy}},
       "spectra":{pid:{axis,freq,power}}, "analysis":{pid:{...}}, ...}
    """
    series_all, t_max, t_min = _load_displacement_cached(mdir / "displacement.jsonl")
    if not series_all or t_max is None:
        return {"t": None, "t_max": None, "series": {}, "spectra": {},
                "analysis": {}, "window_s": window_s, "analysis_updated": None,
                "curve_t0": None, "curve_t1": None,
                "fft_ready": False, "fft_cover_s": 0.0,
                "fft_need_s": round(min(float(min_span_s), float(window_s)), 1),
                "fft_reason": "该时刻无数据", "note": "该时刻无数据"}
    W = float(window_s)
    # t 越界保护：负值/超长都夹到数据范围内，避免前端 currentTime 回绕时窗口空
    cur = t_max if t is None or flush else max(0.0, min(float(t), t_max))
    # ---- 时移曲线：只展示当前固定窗口，并把时间重置为 0~W ----
    # 频谱按 [0,W)、[W,2W) ... 固定分段；曲线也必须在进入下一段时从 0
    # 重新开始，不能用 [cur-W, cur] 的滑动窗口把相邻两段拼在一起。
    # 文件末尾 flush 时，曲线与尾部短窗口共用同一个起点。
    display_t = cur if t is not None else None
    bounds = _display_window(display_t, t_max, W, min_span_s, flush)
    if flush and bounds is not None:
        curve_start = bounds[0]
    else:
        curve_start = min(math.floor(cur / W) * W, max(0.0, t_max))
    curve_end = min(cur, curve_start + W)
    series = {}
    for pid, samples in series_all.items():
        win_s = [s for s in samples
                 if curve_start - 1e-9 <= s[0] <= curve_end + 1e-9]
        if not win_s:
            continue
        series[pid] = {
            # 前端横轴使用窗口内相对秒，因此每个新窗口都从 0 开始。
            "t": [round(s[0] - curve_start, 2) for s in win_s],
            "dx": [round(s[1], 3) for s in win_s],
            "dy": [round(s[2], 3) for s in win_s],
        }
    # ---- 频谱/主频：固定不重叠窗口，需窗口就绪（flush 或已完成窗）才出图 ----
    # t=None 表示“最新实时状态”，不是视频播放到 t_max；只有 file 源收尾
    # flush=True 才允许把当前未满固定窗的尾段作为一次短窗计算。
    if bounds is None:
        # 还没到任何窗口的完成点：时移曲线照常逐帧显示，仅频谱提示"累积中"
        need = round(min(float(min_span_s), W), 1)
        cover = 0.0
        for pid, sm in series.items():
            if sm["t"]:
                cover = max(cover, sm["t"][-1] - sm["t"][0])
        return {
            "t": round(cur, 2), "t0": None, "t1": None, "t_max": round(t_max, 2),
            "window_s": W, "series": series, "spectra": {}, "analysis": {},
            "curve_t0": round(curve_start, 2), "curve_t1": round(curve_end, 2),
            "analysis_updated": None,
            "fft_ready": False, "fft_cover_s": round(cover, 2),
            "fft_need_s": need,
            "fft_reason": f"窗口累积中 {min(cover, need):.1f} / 需 {need:.1f}s",
            "note": f"窗口累积中 {min(cover, need):.1f} / 需 {need:.1f}s",
        }
    t0, t1 = bounds
    need_s = min(float(min_span_s), W)
    spectra = {}
    analysis = {}
    partial_tail = bool(flush and (t1 - t0) < W - 1e-9)
    effective_min_samples = 4 if partial_tail else int(min_samples)
    effective_need_s = 0.0 if partial_tail else need_s
    def in_fft_window(sample):
        # 完整窗严格右开，只有文件尾窗包含实际最后一个采样点。
        return sample[0] >= t0 and (sample[0] <= t1 if partial_tail else sample[0] < t1)

    gate_ready, gate_cover, gate_need = _fft_gate(
        [[s[0] for s in samples if in_fft_window(s)]
         for samples in series_all.values()],
        window_s=W, min_samples=effective_min_samples, min_span_s=effective_need_s)
    for pid, samples in series_all.items():
        # 取该固定窗内的样本（与主链路 _finalize 的 [seg*W, (seg+1)*W) 同口径）
        win = [s for s in samples if in_fft_window(s)]
        if not win:
            continue
        ts = [s[0] for s in win]
        # “最大振幅”与主频使用同一物理方向：靶标画面 y 轴（上下振动）。
        amps = [abs(s[2]) for s in win]
        amp = round(max(amps), 3) if amps else 0.0
        span = (ts[-1] - ts[0]) if len(ts) > 1 else 0.0
        # 样本数和实际时间跨度是独立门限，必须与主链路同时满足；
        # _segment_spectrum 本身只负责 FFT/SNR，不能替代这两个门控条件。
        eligible = (gate_ready and len(win) >= effective_min_samples
                    and span >= effective_need_s - 1e-9)
        # 本看板评估靶标上下振动，主频与频谱统一只使用垂向 dy。
        sp = (_window_spectrum(ts, [s[2] for s in win], snr_min)
              if eligible else None)
        ref = (f1_ref or {}).get(pid)
        if sp is not None:
            spectra[pid] = {"axis": "dy", "freq": sp["freq"], "power": sp["power"]}
            analysis[pid] = {
                "max_amp_mm": amp,             # 真实：窗口内最大垂向位移幅值
                "f1_hz": sp["f1_hz"],          # 真实：窗口 FFT 主频（同主链路算法）
                "snr": (round(sp["snr"], 1)
                        if sp["snr"] != float("inf") else None),
                "n_samples": len(win),
                "span_s": round(span, 2),
                "axis": "dy",
                "f1_ref_hz": round(ref, 4) if ref else None,
                # 占位（未计算），见 _MARKER_PH —— 与 /api/state 的 analysis 同源
                "deflection_mm": _MARKER_PH["deflection_mm"],
                "cable_force_idx": _MARKER_PH["cable_force_idx"],
                "placeholder": True,
                "reliable": bool(sp["f1_hz"] is not None),
            }
        else:
            analysis[pid] = {
                "max_amp_mm": amp, "f1_hz": None, "snr": None,
                "n_samples": len(win), "span_s": round(span, 2), "axis": None,
                "f1_ref_hz": round(ref, 4) if ref else None,
                "deflection_mm": _MARKER_PH["deflection_mm"],
                "cable_force_idx": _MARKER_PH["cable_force_idx"],
                "placeholder": True, "reliable": False,
            }
    fft_ready = bool(spectra)
    floor_s = round(need_s, 1)
    if not analysis:
        note = "该窗口无数据"
        fft_reason = "该窗口无数据"
    elif not fft_ready:
        note = f"窗口样本不足或 SNR 未达标（需 ≥{floor_s}s 窗口）"
        fft_reason = note
    else:
        note = ""
        fft_reason = ""
    return {
        "t": round(cur, 2), "t0": round(t0, 2), "t1": round(t1, 2),
        "t_max": round(t_max, 2), "window_s": W,
        "series": series, "spectra": spectra, "analysis": analysis,
        "curve_t0": round(curve_start, 2), "curve_t1": round(curve_end, 2),
        # 标记实际完成的固定窗口终点，而不是当前流最新帧 t_max；两者在
        # RTSP 窗口累积期间通常不同。
        "analysis_updated": round(t1, 1),
        # 频谱图门限（前端只读，不自行判定）：与上面主频同一条判据
        "fft_ready": fft_ready,
        "fft_cover_s": gate_cover, "fft_need_s": gate_need,
        "fft_reason": fft_reason, "note": note,
    }


# ---- 挠度 / 索力占位值 ----
# 这两个量的物理模型尚未确定（影响因素没梳理清楚），主链路 marker_subpixel_tracker
# 也**不产出**它们（只出 displacement / spectrum）。此前看板是用"窗口内位移幅值
# 冒充挠度、用主频平方比 (f1/f1_ref)²×100 冒充索力"，数字看着像真的却无法解释。
# 现改为直接显示配置里的占位值，不做任何计算；真实输出就绪后再接进来。
# 由 apply_marker_placeholder(cfg) 在启动时按 config.marker.placeholder 覆盖。
_MARKER_PH = {"enabled": True, "deflection_mm": 0.0, "cable_force_idx": 100.0}


def apply_marker_placeholder(cfg):
    """按配置刷新挠度/索力占位值（dashboard 启动时调用一次）。"""
    ph = ((cfg or {}).get("marker", {}) or {}).get("placeholder") or {}
    for k in ("enabled", "deflection_mm", "cable_force_idx"):
        if k in ph and ph[k] is not None:
            _MARKER_PH[k] = ph[k]
    _MARKER_PH["enabled"] = bool(_MARKER_PH["enabled"])
    return dict(_MARKER_PH)


def _marker_analysis(spec_rows, recent, f1_ref):
    """由频谱行 + 时移序列汇总每目标的分析数据。

    既用于实时（最新窗口），也用于初始（首窗）分析——同源计算，初始即首窗结果。
    其中 max_amp_mm / f1_hz 是真实的垂向位移幅值与 dy 主频；deflection_mm / cable_force_idx 是
    占位值（见 _MARKER_PH 说明），不做计算。
    """
    latest_spec = {}
    for r in spec_rows:
        pid = r.get("pid")
        ax = r.get("axis")
        if pid and ax == "dy":
            latest_spec[pid] = r
    analysis = {}
    for pid in set(latest_spec) | set(recent):
        best = latest_spec.get(pid)
        f1_value = to_f(best.get("f1_hz")) if best else None
        f1 = round(f1_value, 4) if f1_value is not None else None
        ref = f1_ref.get(pid)
        amps = [abs(s["dy"]) for s in recent.get(pid, [])]
        amp = round(max(amps), 3) if amps else 0.0
        analysis[pid] = {
            "max_amp_mm": amp,          # 真实：窗口内最大垂向位移幅值
            "f1_hz": f1,                # 真实：FFT 主频（原样保留）
            "axis": "dy" if best else None,
            "snr": round(float(best["snr"]), 1) if best else None,
            "f1_ref_hz": round(ref, 4) if ref else None,
            # 下面两项为占位（未计算），见 _MARKER_PH
            "deflection_mm": _MARKER_PH["deflection_mm"],
            "cable_force_idx": _MARKER_PH["cable_force_idx"],
            "placeholder": True,
        }
    return analysis


# 初始（首窗）振动分析缓存：满窗后冻结，避免"初始状态"随实时数据滑动而刷新
_INIT_ANALYSIS_CACHE = {}


def _first_window_recent(path: Path, window_s: float = 30.0, max_lines=None):
    """返回本次运行最早 window_s 秒的位移序列和时间范围。

    不能只读文件末尾：长录像的最后 4000 行不包含首窗，初始基准会被
    错当成末尾窗口。每个 pid 先截取自己的最后一段单调时间序列，避免
    历史运行追加内容污染本次运行，再从该段的开头取首窗。
    """
    if not path.is_file():
        return {}, None, None
    rows = []
    try:
        with path.open("r", encoding="utf-8") as f:
            lines = f if max_lines is None else f.readlines()[-max_lines:]
            for ln in lines:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    r = json.loads(ln)
                except Exception:
                    continue
                pid = r.get("pid")
                ts = to_f(r.get("ts"))
                if pid is None or ts is None:
                    continue
                rows.append((ts, pid, to_f(r.get("cum_dx_mm")) or 0.0, to_f(r.get("cum_dy_mm")) or 0.0))
    except Exception:
        return {}, None, None
    if not rows:
        return {}, None, None
    by_pid = {}
    for row in rows:
        by_pid.setdefault(row[1], []).append((row[0], row[2], row[3]))
    by_pid = {pid: _tail_monotonic(v) for pid, v in by_pid.items()}
    by_pid = {pid: v for pid, v in by_pid.items() if v}
    if not by_pid:
        return {}, None, None
    min_ts = min(v[0][0] for v in by_pid.values())
    max_ts = max(v[-1][0] for v in by_pid.values())
    out = {}
    for pid, samples in by_pid.items():
        pid_start = samples[0][0]
        for ts, dx, dy in samples:
            if pid_start <= ts <= pid_start + window_s:
                out.setdefault(pid, []).append({"ts": ts, "dx": dx, "dy": dy})
    return out, min_ts, max_ts


def _build_init_analysis(mdir: Path, spec_rows, f1_ref, window_s: float = 30.0):
    """初始分析取本次运行首个 window_s 窗口，并在数据完整后冻结。"""
    key = str(mdir)
    disp_path = mdir / "displacement.jsonl"
    try:
        st = disp_path.stat()
        cache_sig = (st.st_mtime_ns, st.st_size, float(window_s))
    except OSError:
        return {}
    cached = _INIT_ANALYSIS_CACHE.get(key)
    if cached and cached[0] == cache_sig:
        return cached[1]
    fw_recent, min_ts, max_ts = _first_window_recent(disp_path, window_s)
    if not fw_recent:
        return {}
    fw_spec = [r for r in spec_rows
               if (to_f(r.get("ts")) or 1e9) <= (min_ts or 0) + window_s]
    res = _marker_analysis(fw_spec, fw_recent, f1_ref)
    if min_ts is not None and max_ts is not None and (max_ts - min_ts) >= window_s:
        _INIT_ANALYSIS_CACHE[key] = (cache_sig, res)
    return res


def _pole_init_block(pdir: Path):
    """读取某立柱相机初始基准（init_state.json），返回 {pid: {lr_init, fb_init, total_init, quality}}。"""
    init = load_json(pdir / "init" / "init_state.json") or {}
    init_targets = {t["pid"]: t for t in init.get("targets", []) if "pid" in t}
    fb_supported = True
    init_block = {}
    for pid, t in init_targets.items():
        bl = t.get("baseline") or {}
        mode = bl.get("mode") if isinstance(bl, dict) else None
        if mode == "image_relative":
            fb_supported = False
        vals = bl.get("values") if isinstance(bl, dict) else None
        lr0 = round(float(vals[4]), 4) if vals and len(vals) >= 5 else 0.0
        init_block[pid] = {
            "lr_init": lr0,
            "fb_init": 0.0,
            "total_init": round(abs(lr0), 4),
            "quality": round(to_f(t.get("quality")) or 0.0, 3),
        }
    return init_block, fb_supported


def _marker_init_analysis(mdir: Path, window_s: float = 30.0):
    """某测振相机的首窗分析，与实时分析使用同一 f1_ref 口径。"""
    spec_rows = load_spectrum_rows(mdir / "spectrum.jsonl")
    f1_ref = _marker_f1_ref(spec_rows)
    return _build_init_analysis(mdir, spec_rows, f1_ref, window_s=window_s)


def _fft_values(fft_cfg):
    cfg = fft_cfg or {}
    return (float(to_f(cfg.get("window_s")) or 30.0),
            to_f(cfg.get("snr")) or 5.0,
            int(to_f(cfg.get("min_samples")) or 32),
            to_f(cfg.get("min_span_s")) or _MARKER_MIN_SPAN_S)


def _pole_camera_state(output_dir, cid, cam_info):
    pdir = Path(output_dir) / "pole" / cid
    init = load_json(pdir / "init" / "init_state.json") or {}
    latest = load_json(pdir / "latest.json") or {}
    poles_cur = latest.get("poles") or {}
    init_targets = {t["pid"]: t for t in init.get("targets", [])
                    if isinstance(t, dict) and "pid" in t}
    init_block, fb_supported = _pole_init_block(pdir)
    current = {}
    alarms = []
    for pid, d in poles_cur.items():
        current[pid] = {
            "frame_id": d.get("frame_id"),
            "lr_deg": round(to_f(d.get("lr_deg")) or 0.0, 3),
            "fb_deg": round(to_f(d.get("fb_deg")) or 0.0, 3),
            "total_deg": round(to_f(d.get("total_deg")) or 0.0, 3),
            "delta_lr_deg": round(to_f(d.get("delta_lr_deg")) or 0.0, 3),
            "delta_fb_deg": round(to_f(d.get("delta_fb_deg")) or 0.0, 3),
            "status": d.get("status"),
            "alarm": bool(d.get("alarm")),
            "quality": round(to_f(d.get("quality")) or 0.0, 3),
        }
        if d.get("alarm"):
            alarms.append(pid)
    first = next(iter(poles_cur.values()), {})
    info = (cam_info or {}).get(cid) or {}
    branch = info.get("branch") or "pole"
    init_frame = pdir / "init" / "init_frame.jpg"
    return {
        "cam_id": cid,
        "branch": branch,
        "initialized": bool(init_targets),
        "frame_id": first.get("frame_id") if poles_cur else None,
        "frame_ts": first.get("ts"),
        "fb_supported": fb_supported,
        "source": _cam_live_source(cid, cam_info, output_dir),
        # 文件尚未生成时返回 null，前端显示初始化空态而不是破图。
        "init_frame_url": (f"/media/output/{branch}/{cid}/init/init_frame.jpg"
                           if init_frame.is_file() else None),
        "init": init_block,
        "current": current,
        "timeline": _pole_timeline(pdir),
        "alarm_any": bool(alarms),
        "alarms": alarms,
    }


def _marker_camera_state(output_dir, cid, cam_info, fft_cfg):
    mdir = Path(output_dir) / "marker" / cid
    info = (cam_info or {}).get(cid) or {}
    # 持续 RTSP 源没有“播放末尾”；只能展示已完成的固定 FFT 窗口。
    # 本地文件源在 /api/state 中允许 flush，以便短视频收尾时展示整段结果。
    flush_latest = (info.get("source_type") == "file"
                    and (mdir / ".rec_done").is_file())
    latest = load_json(mdir / "latest.json") or {}
    targets = latest.get("targets") or {}
    spec_rows = load_spectrum_rows(mdir / "spectrum.jsonl")
    f1_ref = _marker_f1_ref(spec_rows)
    window_s, snr, min_samples, min_span_s = _fft_values(fft_cfg)
    win = _marker_window(mdir, t=None, window_s=window_s, snr_min=snr,
                         min_samples=min_samples, min_span_s=min_span_s,
                         f1_ref=f1_ref, flush=flush_latest)
    series = win["series"]
    ready = bool(win["fft_ready"] and series)
    branch = info.get("branch") or "marker"
    init_frame = mdir / "init" / "init_frame.jpg"
    return {
        "cam_id": cid,
        "branch": branch,
        "frame_id": next(iter(targets.values()), {}).get("frame_id"),
        "window_s": window_s,
        "targets": {
            pid: {
                "cum_dx_mm": round(to_f(t.get("cum_dx_mm")) or 0.0, 3),
                "cum_dy_mm": round(to_f(t.get("cum_dy_mm")) or 0.0, 3),
                "mm_per_px": round(to_f(t.get("mm_per_px")) or 0.0, 4),
                "conf": round(to_f(t.get("conf")) or 0.0, 3),
            }
            for pid, t in targets.items()
        },
        "analysis": win["analysis"],
        "init_analysis": _build_init_analysis(mdir, spec_rows, f1_ref,
                                               window_s=window_s),
        "analysis_updated": win["analysis_updated"],
        "series": series,
        "curve_t0": win.get("curve_t0"), "curve_t1": win.get("curve_t1"),
        "t0": win.get("t0"), "t1": win.get("t1"),
        "spectra": win.get("spectra", {}),
        "fft_ready": ready,
        "fft_cover_s": win["fft_cover_s"],
        "fft_need_s": win["fft_need_s"],
        "fft_reason": ("" if ready else win["fft_reason"]),
        "source": _cam_live_source(cid, cam_info, output_dir),
        "init_frame_url": (f"/media/output/{branch}/{cid}/init/init_frame.jpg"
                           if init_frame.is_file() else None),
    }


def build_state(output_dir, pole_cam, marker_cam, cam_info=None, fft_cfg=None):
    out = {"server_time": time.time(), "pole": None, "marker": None}
    pole_ids = [cid for cid, info in (cam_info or {}).items()
                if (info.get("branch") or "") == "pole"] or ([pole_cam] if pole_cam else [])
    marker_ids = [cid for cid, info in (cam_info or {}).items()
                  if (info.get("branch") or "") == "marker"] or ([marker_cam] if marker_cam else [])

    if pole_cam:
        pole_cams = {cid: _pole_camera_state(output_dir, cid, cam_info)
                     for cid in pole_ids}
        primary_pole = pole_cams.get(pole_cam) or _pole_camera_state(
            output_dir, pole_cam, cam_info)
        out["pole"] = dict(primary_pole)
        out["pole"]["cams"] = pole_cams

    if marker_cam:
        marker_cams = {cid: _marker_camera_state(output_dir, cid, cam_info,
                                                  fft_cfg)
                       for cid in marker_ids}
        primary_marker = marker_cams.get(marker_cam) or _marker_camera_state(
            output_dir, marker_cam, cam_info, fft_cfg)
        out["marker"] = dict(primary_marker)
        out["marker"]["cams"] = marker_cams

    return out


# ----------------------------- 报警子页面数据 -----------------------------

def load_alarm_events(output_dir):
    """读取 events.jsonl，筛出报警类事件。

    约定事件字段（后续接入真实检测时遵循）：
        event : "pole_alarm" | "cable_force_alarm" | "amplitude_alarm"
              | "deflection_alarm"
        ts    : 报警触发时的相对时间（秒）
        pid   : 目标/立柱编号
        cam_id: 摄像头编号
        level : 报警等级（>=1，越大越严重）
        detail: 文字说明
    当前没有真实报警数据时返回空列表 —— 子页面需优雅处理空态。
    """
    p = Path(output_dir) / "events.jsonl"
    events = []
    if not p.is_file():
        return events
    for ln in p.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if r.get("event") in ("pole_alarm", "cable_force_alarm", "amplitude_alarm",
                               "deflection_alarm"):
            events.append(r)
    events.sort(key=lambda e: to_f(e.get("ts")) or 0.0)
    return events


def build_records(output_dir, pole_cam, marker_cam):
    """汇总所有报警记录，供"报警记录"页以列表/详情方式呈现（不做左右分屏）。

    记录来源：
      1) 立柱超限快照：output/alarms/pole/<ts>.jpg（可选同目录 manifest.jsonl
         提供该时刻的 pid/level/detail/倾角等明细）。文件名即触发时刻 ts。
      2) events.jsonl 中的报警事件：pole_alarm / cable_force_alarm /
         amplitude_alarm / deflection_alarm。

    当前无数据时 records 为空、has_data=False —— 页面如实显示空态，不编造。
    """
    records = []

    # 1) 立柱超限快照：output/alarms/pole/manifest.jsonl 逐条即一次报警留存，
    #    每条含 snapshot(重新编码帧=画框+拟合直线) / original(原始帧) / source(实时/文件)。
    #    命名已改为 {当前时间}_ 或 {frame_id}_ 前缀，无法 float 解析，故直接按 manifest 逐条建记录。
    pd = Path(output_dir) / "alarms" / "pole"
    mfile = pd / "manifest.jsonl"
    if mfile.is_file():
        for ln in mfile.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            records.append({
                "id": f"pole_{r.get('cam_id')}_{r.get('ts')}",
                "ts": to_f(r.get("ts")),
                "type": "pole",
                "pid": r.get("pid") or "—",
                "level": r.get("level", 1),
                "detail": r.get("detail") or "立柱倾斜角超过阈值",
                # snapshot = 重新编码帧（画框+拟合直线）；original = 裸原始帧
                "snapshot": r.get("snapshot"),
                "original": r.get("original"),
                "source": r.get("source"),
                "data": r,
            })

    # 2) events.jsonl 报警事件
    TYPE_MAP = {
        "pole_alarm": "pole",
        "cable_force_alarm": "cable",
        "amplitude_alarm": "amplitude",
        "deflection_alarm": "deflection",
    }
    vib_dir = Path(output_dir) / "vibration"
    for e in load_alarm_events(output_dir):
        ev = e.get("event")
        if ev not in TYPE_MAP:
            continue
        aid = e.get("id")
        analysis = e.get("analysis")
        # 振动报警：收尾阶段 supervisor 已为文件源从源文件抽取片段、为 RTSP 实时源
        # 引用 capture 滚动缓冲切出的片段；两者都把分析数据 + clip 路径写入
        # output/vibration/<aid>.json 侧车（clip 仅在片段确实落盘时非空）。
        # clip = web 可播副本（{原名}_h264.mp4）；clip_native = 原视频文件（保留不删）。
        # build_records 优先读侧车，回退事件内嵌分析。
        clip = None
        clip_native = None
        if ev in ("amplitude_alarm", "deflection_alarm", "cable_force_alarm") and aid:
            js = vib_dir / f"{aid}.json"
            if js.is_file():
                try:
                    analysis = json.loads(js.read_text(encoding="utf-8"))
                except Exception:
                    pass
            if isinstance(analysis, dict):
                clip = analysis.get("clip")
                clip_native = analysis.get("clip_native")
        records.append({
            "id": aid or f"{ev}_{e.get('ts')}",
            "ts": to_f(e.get("ts")),
            "type": TYPE_MAP[ev],
            "pid": e.get("pid") or "—",
            "level": e.get("level", 1),
            "detail": e.get("detail") or "",
            "snapshot": None,
            "original": None,
            "clip": clip,
            "clip_native": clip_native,
            "source": (analysis or {}).get("source") if isinstance(analysis, dict) else None,
            "analysis": analysis,
            "data": e,
        })

    records.sort(key=lambda r: (r["ts"] is None, r["ts"] if r["ts"] is not None else 0))
    return {
        "server_time": time.time(),
        "has_data": bool(records),
        "records": records,
    }


# ----------------------------- HTTP 服务 -----------------------------

def _placeholder_jpg(text):
    """生成一张带文字的占位 JPEG（无数据时显示）。"""
    try:
        import cv2
        import numpy as np
        img = np.full((360, 640, 3), (30, 30, 30), dtype=np.uint8)
        cv2.putText(img, text, (40, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (200, 200, 200), 2)
        ok, buf = cv2.imencode(".jpg", img)
        if ok:
            return buf.tobytes()
    except Exception:
        pass
    # 极简兜底：1x1 灰图
    return b"\xff\xd8\xff\xdb\x00\x43\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07" \
           b"\x09\x09\x08\x0a\x0c\x14\x0d\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d" \
           b"\x1a\x1f\x1e\x1d\x1a\x1c\x1c\x20\x24\x2e\x27\x20\x22\x2c\x23\x1c\x1c" \
           b"\x28\x37\x29\x2c\x30\x31\x34\x34\x34\x1f\x27\x39\x3d\x38\x32\x3c\x2e" \
           b"\x33\x34\x32\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff" \
           b"\xd9"


def _derive_stream_url(url, channel):
    """由主码流 URL 派生显示用的子码流 URL。

    海康等约定：主码流 Channels/101、子码流 Channels/102。把 URL 末尾的 3 位
    通道号替换为配置的子码流通道（默认 "102"），其余保持不变，使 dashboard 走
    子码流、与 main 走主码流互不抢带宽。无法识别时原样返回（退化为同一条流）。
    """
    if not url:
        return url
    # 海康 URL 通常是 /Streaming/Channels/101 或 /Channels/101；匹配时
    # 忽略大小写并允许后续 query/path，避免未识别后错误地回退到主码流。
    m = re.search(r"(/channels/)(\d+)(?=[/?&#]|$)", url, re.I)
    if m:
        return url[:m.start(2)] + str(channel) + url[m.end(2):]
    # 少数设备把码流号放在 query 中；仅替换明确的 channel/stream/substream
    # 参数，避免误改用户名、密码或其他数字参数。
    q = re.search(r"([?&](?:channel|stream|substream)=)(\d+)(?=[&#]|$)",
                  url, re.I)
    if q:
        return url[:q.start(2)] + str(channel) + url[q.end(2):]
    return url


class RtspStreamManager:
    """每个 RTSP 相机一个后台拉流线程：cv2 直连子码流、持续抓取最新帧、编码 JPEG
    存入线程安全缓冲；WSGI 请求进来时把缓冲帧以 MJPEG 推给浏览器。

    与 main 进程彻底解耦：本进程是相机的*第二个独立消费者*（走子码流），main
    走主码流，两进程互不通信、互不抢带宽。相机离线/连接失败自动重连（后台线程
    周期性重试），恢复后无缝续传。浏览器侧用 ``<img src="/video/<cam>">`` 直接播放。
    """

    def __init__(self, encode_fps=15, reconnect_s=3.0):
        self._lock = threading.Lock()
        # cid -> {url, cap, latest_jpg, last_err, stop, thread}
        self._streams = {}
        self._encode_fps = max(1, int(encode_fps))
        self._reconnect_s = max(0.5, float(reconnect_s))

    def register(self, cid, url):
        """注册一个相机并开始后台拉流（幂等：已存在则仅更新 URL）。"""
        with self._lock:
            s = self._streams.get(cid)
            if s is None:
                s = {"url": url, "cap": None, "latest_jpg": None,
                     "last_err": "", "stop": False, "thread": None}
                self._streams[cid] = s
            else:
                s["url"] = url
                return  # 已在运行，无需重复起线程
        t = threading.Thread(target=self._worker, args=(cid,), daemon=True)
        s["thread"] = t
        t.start()

    def _worker(self, cid):
        try:
            import cv2
        except Exception as exc:  # cv2 缺失：明确报错，不再静默卡死
            with self._lock:
                self._streams[cid]["last_err"] = f"cv2 不可用: {exc}"
            return
        cap = None
        last_encode = 0.0
        min_interval = 1.0 / self._encode_fps
        while True:
            with self._lock:
                if self._streams[cid]["stop"]:
                    break
                url = self._streams[cid]["url"]
            if cap is None or not cap.isOpened():
                try:
                    # 复用 main 的 RTSP/TCP/nobuffer 初始化，确保 Dashboard
                    # 与推理进程都使用低延迟连接和最小 OpenCV 缓冲。
                    from utils.common import open_camera
                    cap = open_camera(url)
                    if cap is None:
                        raise RuntimeError("VideoCapture.open 失败")
                except Exception as exc:
                    with self._lock:
                        self._streams[cid]["last_err"] = \
                            f"连接失败，{self._reconnect_s:.0f}s 后重连: {exc}"
                        self._streams[cid]["latest_jpg"] = None
                    time.sleep(self._reconnect_s)
                    continue
            ok, frame = cap.read()
            if not ok or frame is None:
                with self._lock:
                    self._streams[cid]["last_err"] = "读帧失败，尝试重连"
                    self._streams[cid]["latest_jpg"] = None
                try:
                    cap.release()
                except Exception:
                    pass
                cap = None
                time.sleep(self._reconnect_s)
                continue
            now = time.time()
            if now - last_encode >= min_interval:
                last_encode = now
                try:
                    ok2, buf = cv2.imencode(
                        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if ok2:
                        with self._lock:
                            self._streams[cid]["latest_jpg"] = buf.tobytes()
                            self._streams[cid]["last_err"] = ""
                except Exception:
                    pass
            # 继续抓下一帧（保持缓冲新鲜），无额外 sleep：抓帧本身即节流

    def get_frame(self, cid):
        with self._lock:
            s = self._streams.get(cid)
            return (s or {}).get("latest_jpg")

    def get_error(self, cid):
        with self._lock:
            s = self._streams.get(cid)
            return (s or {}).get("last_err") or ""

    def has_stream(self, cid):
        with self._lock:
            return cid in self._streams

    def shutdown(self):
        with self._lock:
            for s in self._streams.values():
                s["stop"] = True


class Handler:
    def __init__(self, output_dir, pole_cam, marker_cam, html, records_html,
                 cam_info=None, fft_cfg=None, stream_mgr=None):
        self.output_dir = Path(output_dir)
        self.pole_cam = pole_cam
        self.marker_cam = marker_cam
        self.html = html
        self.records_html = records_html
        self.cam_info = cam_info or {}
        self.fft_cfg = fft_cfg or {}
        # 实时流管理器：RTSP 相机由本进程 cv2 直连子码流，MJPEG 推浏览器。
        # 为 None 时 /video/<cam> 退化为占位图（不依赖 main 发布任何图片）。
        self.stream_mgr = stream_mgr
        # 播放联动窗口长度：与主链路 SpectrumAnalyzer 读**同一个** config
        # （marker.fft.window_s）。界面上说的"30s 窗口"必须与 spectrum.jsonl 的
        # 分段长度同源，否则曲线窗口和频谱窗口是两个数。
        self.marker_window_s = to_f(self.fft_cfg.get("window_s")) or 30.0

    def __call__(self, environ, start_response):
        from urllib.parse import urlparse
        self.environ = environ
        path = urlparse(environ["PATH_INFO"]).path
        method = environ.get("REQUEST_METHOD", "GET")

        if path in ("/", "/index.html"):
            body = self.html.encode("utf-8")
            start_response("200 OK", [
                ("Content-Type", "text/html; charset=utf-8"),
                ("Content-Length", str(len(body))),
                # 不缓存：页面脚本是内嵌的，改完代码刷新即可生效，
                # 免得浏览器拿旧脚本造成"明明修了却还是老行为"的误判
                ("Cache-Control", "no-store"),
            ])
            return [body]

        if path == "/api/state":
            payload = build_state(self.output_dir, self.pole_cam, self.marker_cam,
                                  self.cam_info, fft_cfg=self.fft_cfg)
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            start_response("200 OK", [
                ("Content-Type", "application/json; charset=utf-8"),
                ("Cache-Control", "no-store"),
                ("Content-Length", str(len(body))),
            ])
            return [body]

        if path == "/api/marker_window":
            # 视频文件播放联动：按 currentTime=t 取固定不重叠窗口的曲线+振动分析。
            # t 省略 → 最新（等同 /api/state 中 marker.series/analysis 的口径）。
            # flush=1 表示视频已到末尾/全部数据可用（短视频即用全部数据算一次）。
            from urllib.parse import parse_qs
            q = parse_qs(environ.get("QUERY_STRING", ""))
            cam = (q.get("cam") or [self.marker_cam])[0] or self.marker_cam
            tval = (q.get("t") or [None])[0]
            try:
                t = float(tval) if tval not in (None, "") else None
            except (TypeError, ValueError):
                t = None
            flush = (q.get("flush") or ["0"])[0] in ("1", "true", "True")
            payload = self._marker_window_payload(cam, t, flush=flush)
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            start_response("200 OK", [
                ("Content-Type", "application/json; charset=utf-8"),
                ("Cache-Control", "no-store"),
                ("Content-Length", str(len(body))),
            ])
            return [body]

        if path.startswith("/video/"):
            cam = path[len("/video/"):]
            return self._stream_video(cam, start_response)

        if path.startswith("/media/"):
            return self._serve_media(path[len("/media/"):], start_response)

        if path in ("/records", "/records.html"):
            body = self.records_html.encode("utf-8")
            start_response("200 OK", [
                ("Content-Type", "text/html; charset=utf-8"),
                ("Content-Length", str(len(body))),
                # 不缓存：页面脚本是内嵌的，改完代码刷新即可生效，
                # 免得浏览器拿旧脚本造成"明明修了却还是老行为"的误判
                ("Cache-Control", "no-store"),
            ])
            return [body]

        if path == "/api/records":
            payload = build_records(self.output_dir, self.pole_cam, self.marker_cam)
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            start_response("200 OK", [
                ("Content-Type", "application/json; charset=utf-8"),
                ("Cache-Control", "no-store"),
                ("Content-Length", str(len(body))),
            ])
            return [body]

        if path.startswith("/alarms/"):
            return self._serve_static(path[len("/alarms/"):], start_response)

        start_response("404 Not Found", [("Content-Type", "text/plain; charset=utf-8")])
        return [b"not found"]

    def _marker_window_payload(self, cam, t, flush=False):
        """按相机 + 时间点组装测振窗口数据。

        索力基准 f1_ref 与 /api/state 用同一函数提取，保证"播放联动"看到的
        索力相对值与"最新窗口"完全同一口径（否则同一时刻两处数值会对不上）。

        flush=True 表示「视频已到末尾/全部数据可用」：源不足一个窗长时即用全部
        数据算一次固定窗（与"0-30 算一次、不足 30s 末尾用全部数据"的口径一致）。
        """
        mdir = self.output_dir / "marker" / cam
        info = (self.cam_info or {}).get(cam) or {}
        # 未结束文件/流源不能通过 flush 查询参数提前发布尾段。
        flush = bool(flush and info.get("source_type") == "file"
                     and (mdir / ".rec_done").is_file())
        f1_ref = _marker_f1_ref(load_spectrum_rows(mdir / "spectrum.jsonl"))
        win = _marker_window(
            mdir, t=t, window_s=self.marker_window_s,
            snr_min=to_f(self.fft_cfg.get("snr")) or 5.0,
            min_samples=int(to_f(self.fft_cfg.get("min_samples")) or 32),
            min_span_s=to_f(self.fft_cfg.get("min_span_s")) or _MARKER_MIN_SPAN_S,
            f1_ref=f1_ref, flush=bool(flush))
        win["cam_id"] = cam
        return win

    def _stream_video(self, cam, start_response):
        """RTSP 直连 MJPEG 流：后端 cv2 拉子码流，把最新帧以 MJPEG 推给浏览器。

        与 main 进程解耦（main 走主码流，本端走子码流），实时画面为纯原始画面，
        不做任何叠加。浏览器用 ``<img src="/video/<cam>">`` 直接播放。相机离线/连接中
        时下发占位图，避免前端闪烁。未登记为 RTSP 流的相机（如文件源）返回 404。
        """
        mgr = self.stream_mgr
        if mgr is None or not mgr.has_stream(cam):
            start_response("404 Not Found", [
                ("Content-Type", "text/plain; charset=utf-8")])
            return [b"not a stream camera"]
        boundary = "frame-boundary"
        # 注意：WSGI 规范禁止应用下发 hop-by-hop 头（Connection / Keep-Alive /
        # Transfer-Encoding 等），否则 wsgiref 直接 AssertionError → 500。
        # 连接保活由服务器层负责，这里只下发端到端头。
        start_response("200 OK", [
            ("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}"),
            ("Cache-Control", "no-store"),
        ])

        def gen():
            boundary_b = f"--{boundary}\r\n".encode()
            end_b = b"\r\n"
            sent_ph = False
            try:
                while True:
                    data = mgr.get_frame(cam) if mgr else None
                    if not data:
                        err = mgr.get_error(cam) if mgr else ""
                        if not sent_ph:
                            msg = "等待相机画面…" + (f" {err}" if err else "")
                            data = _placeholder_jpg(msg)
                            sent_ph = True
                        else:
                            time.sleep(0.2)
                            continue
                    else:
                        sent_ph = False
                    yield boundary_b
                    yield b"Content-Type: image/jpeg\r\n"
                    yield f"Content-Length: {len(data)}\r\n\r\n".encode()
                    yield data
                    yield end_b
                    time.sleep(0.03)
            except GeneratorExit:
                return
        return gen()

    def _serve_media(self, rel, start_response):
        """安全下发项目根目录下的媒体/图像文件（录制视频、初始帧等）。

        仅允许特定扩展名，且必须落在项目根目录内（防目录穿越）。用于"实时画面"
        对本地录制文件做 <video> 原生裸播，以及初始帧可视化图片展示。

        支持 HTTP Range 分段（206 Partial Content）：浏览器播 MP4 必须先发 Range
        探测，服务端不回 206 会拒绝播放；同时实现拖动/流式播放。
        """
        root = Path.cwd().resolve()
        full = (root / rel).resolve()
        try:
            full.relative_to(root)
        except ValueError:
            start_response("403 Forbidden", [("Content-Type", "text/plain; charset=utf-8")])
            return [b"forbidden"]
        if not full.is_file():
            start_response("404 Not Found", [("Content-Type", "text/plain; charset=utf-8")])
            return [b"not found"]
        ctype = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
            ".mp4": "video/mp4", ".webm": "video/webm", ".avi": "video/x-msvideo",
            ".mov": "video/quicktime", ".mkv": "video/x-matroska",
        }.get(full.suffix.lower())
        if ctype is None:
            start_response("403 Forbidden", [("Content-Type", "text/plain; charset=utf-8")])
            return [b"forbidden"]
        try:
            size = full.stat().st_size
            with open(full, "rb") as f:
                rng = (self.environ or {}).get("HTTP_RANGE")
                m = re.match(r"bytes=(\d*)-(\d*)$", rng) if rng else None
                if m:
                    start_s = int(m.group(1)) if m.group(1) else 0
                    end_s = int(m.group(2)) if m.group(2) else size - 1
                    if start_s >= size:
                        start_response("416 Range Not Satisfiable", [
                            ("Content-Range", f"bytes */{size}")])
                        return [b""]
                    if end_s >= size:
                        end_s = size - 1
                    length = end_s - start_s + 1
                    f.seek(start_s)
                    data = f.read(length)
                    start_response("206 Partial Content", [
                        ("Content-Type", ctype),
                        ("Content-Length", str(len(data))),
                        ("Accept-Ranges", "bytes"),
                        ("Content-Range", f"bytes {start_s}-{end_s}/{size}"),
                        ("Cache-Control", "no-store"),
                    ])
                    return [data]
                # 无 Range 或解析失败：回退完整文件
                data = f.read()
            start_response("200 OK", [
                ("Content-Type", ctype),
                ("Content-Length", str(len(data))),
                ("Accept-Ranges", "bytes"),
                ("Cache-Control", "no-store"),
            ])
            return [data]
        except OSError:
            start_response("404 Not Found", [("Content-Type", "text/plain; charset=utf-8")])
            return [b"not found"]

    def _serve_static(self, rel, start_response):
        """安全地下发 output_dir/alarms/ 下的图片/清单（防目录穿越）。"""
        base = (self.output_dir / "alarms").resolve()
        full = (base / rel).resolve()
        try:
            full.relative_to(base)  # 只允许落在 alarms 目录内
        except ValueError:
            start_response("403 Forbidden", [("Content-Type", "text/plain; charset=utf-8")])
            return [b"forbidden"]
        if not full.is_file():
            start_response("404 Not Found", [("Content-Type", "text/plain; charset=utf-8")])
            return [b"not found"]
        ctype = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".json": "application/json",
        }.get(full.suffix.lower(), "application/octet-stream")
        data = full.read_bytes()
        start_response("200 OK", [
            ("Content-Type", ctype),
            ("Content-Length", str(len(data))),
            ("Cache-Control", "no-store"),
        ])
        return [data]


def run(config_path="config.yaml", host="127.0.0.1", port=8000):
    cfg = load_config(config_path)
    output_dir = cfg["runtime"]["output_dir"]
    pole_cam, marker_cam = cameras_of(cfg)
    records_html = build_records_html(pole_cam, marker_cam)
    cam_info = _cam_info(cfg)
    # 挠度/索力占位值（物理模型未定前不计算）按配置生效
    ph = apply_marker_placeholder(cfg)
    print(f"[dashboard] 挠度/索力：{'占位值' if ph['enabled'] else '按配置计算'}"
          f"（挠度={ph['deflection_mm']}mm，索力={ph['cable_force_idx']}）", flush=True)
    # 播放联动窗口分析沿用主链路 marker.fft 的门限（SNR / 最少样本），
    # 保证界面上的主频与 spectrum.jsonl 是同一套判定标准。
    mfft = ((cfg.get("marker", {}) or {}).get("fft", {}) or {})
    marker_window_s = float(to_f(mfft.get("window_s")) or 30.0)
    html = build_html(pole_cam, marker_cam, marker_window_s)

    # 实时流管理器：对每个 RTSP 相机自行 cv2 直连子码流（由主码流 URL 派生，
    # 默认通道 102），后台线程抓帧编码、MJPEG 推浏览器。main 走主码流 101，两进程解耦。
    dashboard_cfg = (cfg.get("dashboard") or {}) or {}
    stream_channel = str(dashboard_cfg.get("stream_channel", "102"))
    stream_mgr = RtspStreamManager(
        encode_fps=int(dashboard_cfg.get("stream_fps", 15)),
        reconnect_s=float(dashboard_cfg.get("reconnect_s", 3.0)))
    n_stream = 0
    for cid, info in cam_info.items():
        if (info.get("source_type") == "stream") and info.get("source_url"):
            stream_url = _derive_stream_url(info["source_url"], stream_channel)
            stream_mgr.register(cid, stream_url)
            n_stream += 1
            if stream_url != info["source_url"]:
                print(f"[dashboard] 相机 {cid} 走子码流(显示): {stream_url}", flush=True)
            else:
                print(f"[dashboard] 相机 {cid} 走原始 URL(未识别子码道，已回退): "
                      f"{stream_url}", flush=True)
    if n_stream:
        print(f"[dashboard] 已为 {n_stream} 路 RTSP 相机启动子码流拉流（显示用）",
              flush=True)

    handler = Handler(output_dir, pole_cam, marker_cam, html, records_html, cam_info,
                      fft_cfg=mfft, stream_mgr=stream_mgr)

    # 用标准库 wsgiref 承载：Handler 是标准 WSGI 应用（environ + start_response）。
    # 注意：不能用 BaseHTTPRequestHandler —— 它既没有 start_response，也没有带
    # PATH_INFO 的 environ，两者都会抛 AttributeError 并被 except 吞掉，导致
    # 连接被接受却一个字节都不发（客户端表现为挂起直至超时）。
    from wsgiref.simple_server import WSGIServer, WSGIRequestHandler
    from socketserver import ThreadingMixIn

    class _QuietWSGIHandler(WSGIRequestHandler):
        # HTTP/1.0：MJPEG 是无限流、无 Content-Length，靠连接关闭结束，浏览器兼容最好
        protocol_version = "HTTP/1.0"

        def log_message(self, *args):
            pass  # 静默

    class _ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
        daemon_threads = True    # MJPEG 长连接线程随主进程退出
        allow_reuse_address = True

    srv = _ThreadingWSGIServer((host, port), _QuietWSGIHandler)
    srv.set_app(handler)

    url = f"http://{host}:{port}"
    print(f"[dashboard] 输出目录: {Path(output_dir).resolve()}", flush=True)
    print(f"[dashboard] 立柱摄像头: {pole_cam} | 测振摄像头: {marker_cam}", flush=True)
    print(f"[dashboard] 大屏已启动 → {url}  (Ctrl+C 退出)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[dashboard] 已停止", flush=True)
    finally:
        try:
            stream_mgr.shutdown()
        except Exception:
            pass


# ----------------------------- 前端页面 -----------------------------

def build_html(pole_cam, marker_cam, marker_window_s=30.0):
    pc = pole_cam or ""
    mc = marker_cam or ""
    ws = str(int(marker_window_s)) if float(marker_window_s).is_integer() else str(marker_window_s)
    return (HTML_TEMPLATE.replace("__POLE_CAM__", pc)
            .replace("__MARKER_CAM__", mc)
            .replace("__MARKER_WINDOW_S__", ws))


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>柔性光伏支架视觉监测系统</title>
<style>
  * { box-sizing: border-box; }
  html,body { margin:0; height:100%; background:#0b0f17; color:#e6edf3;
    font-family: "Segoe UI", "Microsoft YaHei", system-ui, sans-serif; }
  #clock { position:fixed; top:8px; right:18px; font-size:14px; color:#7d8794; z-index:9; }
  header { padding:10px 18px; font-size:18px; font-weight:600; letter-spacing:1px;
    border-bottom:1px solid #1c2430; display:flex; align-items:center; gap:12px; }
  header .dot { width:10px; height:10px; border-radius:50%; background:#3fb950; box-shadow:0 0 8px #3fb950; }
  .nav { margin-left:auto; display:flex; gap:10px; }
  .nav a { color:#9fd0ff; text-decoration:none; font-size:13px; font-weight:600; padding:5px 14px; border:1px solid #1c2430; border-radius:8px; background:#111824; }
  .nav a.active { background:#16202e; border-color:#3fb950; color:#7ee787; }
  .nav a:hover { border-color:#3fb950; }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; padding:12px; height:calc(100% - 52px); }
  .panel { background:#111824; border:1px solid #1c2430; border-radius:10px; display:flex; flex-direction:column; overflow:hidden; }
  .panel > h2 { margin:0; padding:8px 12px; font-size:14px; background:#0e1521; border-bottom:1px solid #1c2430; color:#9fd0ff; }
  .panel.alarm > h2 { color:#ff7b72; }
  .content { flex:1 1 0; min-height:0; overflow:auto; padding:8px 10px; }
  .media-box { width:100%; height:260px; background:#000; display:flex; align-items:center; justify-content:center; border-radius:8px; overflow:hidden; margin-bottom:8px; }
  .media-box img, .media-box video, .media-box canvas { max-width:100%; max-height:100%; object-fit:contain; background:#000; }
  table { width:100%; border-collapse:collapse; font-size:12px; }
  th,td { padding:4px 6px; border-bottom:1px solid #1a2230; text-align:center; }
  th { color:#7d8794; font-weight:500; }
  .ok { color:#3fb950; } .bad { color:#ff7b72; } .warn { color:#d29922; }
  .sec-title { font-size:12px; color:#7d8794; margin:6px 0 3px; }
  canvas { width:100%; display:block; background:#0c121c; border-radius:6px; }
  .charts { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
  .badge { display:inline-block; padding:1px 7px; border-radius:10px; font-size:11px; }
  .viewbar { display:flex; gap:8px; padding:6px 10px; background:#0e1521; border-bottom:1px solid #1c2430; flex-wrap:wrap; }
  .vbtn { background:#111824; color:#c9d4e0; border:1px solid #1c2430; border-radius:8px; padding:4px 12px; font-size:12px; cursor:pointer; }
  .vbtn.active { background:#16202e; border-color:#3fb950; color:#7ee787; }
  .vbtn:hover { border-color:#3fb950; }
  .media-box { width:100%; height:100%; display:flex; align-items:center; justify-content:center; }
  .media-box img, .media-box video, .media-box canvas { max-width:100%; max-height:100%; object-fit:contain; background:#000; }
  .viewpane { padding-top:4px; }
  .sec-bar { display:flex; align-items:center; gap:8px; margin:2px 0 6px; }
  .sec-bar label { font-size:12px; color:#7d8794; }
  .camsel { background:#111824; color:#c9d4e0; border:1px solid #1c2430; border-radius:8px; padding:3px 8px; font-size:12px; min-width:170px; }
  .camsel:focus { outline:none; border-color:#3fb950; }
  .empty { color:#5b6776; font-size:12px; text-align:center; padding:10px; }
  .wininfo { color:#3fb950; font-size:11px; margin-left:6px; }
  /* 占位标记：挠度/索力尚无物理模型，界面显示占位值，明确标注避免被当成实测值 */
  .ph { color:#d29922; font-size:10px; font-weight:400; margin-left:4px; }
</style>
</head>
<body>
<div id="clock"></div>
<header><span class="dot"></span>柔性光伏支架可视化监测
  <nav class="nav"><a href="/" class="active">实时监控</a><a href="/records">报警记录</a></nav>
</header>
<div class="grid">
  <!-- 左：立柱倾角监测 -->
  <div class="panel" id="polePanel">
    <h2>◧ 立柱倾角监测　<span id="poleStatus" class="badge"></span></h2>
    <div class="viewbar">
      <button class="vbtn active" data-panel="pole" data-view="init">初始状态画面</button>
      <button class="vbtn" data-panel="pole" data-view="live">实时展示画面</button>
    </div>
    <div class="content">
      <!-- 初始状态画面：初始帧可视化 + 初始基准状态 -->
      <div class="viewpane" data-panel="pole" data-view="init">
        <div class="sec-bar"><label>选择相机：</label><select id="poleInitSel" class="camsel"></select></div>
        <div class="media-box"><img id="poleInitFrame" alt="初始帧可视化"></div>
        <div class="sec-title">初始基准状态（左右/前后倾斜 · 约定：负=左/后，正=右/前）</div>
        <table id="poleInitTbl"><thead><tr><th>立柱</th><th>左右倾斜(°)</th><th>前后倾斜(°)</th><th>总倾角(°)</th></tr></thead><tbody></tbody></table>
      </div>
      <!-- 实时展示画面：裸播录制/拉流 + 初始基准 + 当前状态 -->
      <div class="viewpane" data-panel="pole" data-view="live" style="display:none">
        <div class="sec-bar"><label>选择相机：</label><select id="poleLiveSel" class="camsel"></select></div>
        <div class="media-box" id="poleLiveMedia"></div>
        <div class="sec-title">初始基准状态</div>
        <table id="poleInitTblLive"><thead><tr><th>立柱</th><th>左右倾斜(°)</th><th>前后倾斜(°)</th><th>总倾角(°)</th></tr></thead><tbody></tbody></table>
        <div class="sec-title">当前状态（每帧即该秒快照 · <span id="poleFrame"></span>）</div>
        <table id="poleCurTbl"><thead><tr><th>立柱</th><th>左右倾斜(°)</th><th>前后倾斜(°)</th><th>总倾角(°)</th><th>相对初始变化(°)</th><th>状态</th><th>报警</th></tr></thead><tbody></tbody></table>
        <div id="poleLiveTime" class="sec-title"></div>
      </div>
    </div>
  </div>
  <!-- 右：视频测振 -->
  <div class="panel" id="markerPanel">
    <h2>◨ 视频测振　<span id="markerStatus" class="badge"></span></h2>
    <div class="viewbar">
      <button class="vbtn active" data-panel="marker" data-view="init">初始状态画面</button>
      <button class="vbtn" data-panel="marker" data-view="live">实时分析界面</button>
    </div>
    <div class="content">
      <!-- 初始状态画面：仅各目标初始挠度 + 索力 -->
      <div class="viewpane" data-panel="marker" data-view="init">
        <div class="sec-bar"><label>选择相机：</label><select id="markerInitSel" class="camsel"></select></div>
        <div class="media-box"><img id="markerInitFrame" alt="初始帧可视化"></div>
        <div class="sec-title">各目标初始挠度与索力（<span class="ph">占位值 · 模型未定，暂不计算</span>）</div>
        <table id="markerInitTbl"><thead><tr><th>目标</th><th>初始挠度(mm)<span class="ph">占位</span></th><th>初始索力(相对)<span class="ph">占位</span></th></tr></thead><tbody></tbody></table>
      </div>
      <!-- 实时分析界面：裸播录制/拉流 + 时移曲线 + 振动分析 -->
      <div class="viewpane" data-panel="marker" data-view="live" style="display:none">
        <div class="sec-bar"><label>选择相机：</label><select id="markerLiveSel" class="camsel"></select></div>
        <div class="media-box" id="markerLiveMedia"></div>
        <div class="sec-title">垂向位移曲线 dy（当前 0~__MARKER_WINDOW_S__s 窗口，mm）<span id="markerWinInfo" class="wininfo"></span></div>
        <div class="charts">
          <canvas id="chartShift" height="120"></canvas>
          <canvas id="chartSpectrum" height="120"></canvas>
        </div>
        <div class="sec-title">垂向频谱 dy（每__MARKER_WINDOW_S__s更新）</div>
        <div class="sec-title">上下振动分析（主频基于 dy · 每__MARKER_WINDOW_S__s更新 · <span id="anaTime"></span>）</div>
        <table id="markerAnaTbl"><thead><tr><th>目标</th><th>最大振幅(mm)</th><th>上下振动主频率 dy (Hz)</th><th>挠度(mm)<span class="ph">占位</span></th><th>索力(相对)<span class="ph">占位</span></th></tr></thead><tbody></tbody></table>
      </div>
    </div>
  </div>
</div>

<script>
const PAD = {l:38,r:10,t:10,b:18};
const MARKER_WINDOW_S = Number("__MARKER_WINDOW_S__") || 30;
function drawAxes(ctx,w,h,ymin,ymax,label){ ctx.strokeStyle="#26303f"; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(PAD.l, PAD.t); ctx.lineTo(PAD.l,h-PAD.b); ctx.lineTo(w-PAD.r,h-PAD.b); ctx.stroke();
  ctx.fillStyle="#5b6776"; ctx.font="10px sans-serif"; ctx.textAlign="right";
  for(let i=0;i<=2;i++){ const y=PAD.t+(h-PAD.t-PAD.b)*i/2; const v=ymax-(ymax-ymin)*i/2;
    ctx.fillText(v.toFixed(1), PAD.l-4, y+3); ctx.beginPath(); ctx.moveTo(PAD.l,y); ctx.lineTo(w-PAD.r,y); ctx.strokeStyle="#1a2230"; ctx.stroke(); }
  ctx.textAlign="left"; ctx.fillText(label, PAD.l+2, PAD.t+10); }

// 多目标配色：按目标名在 pidOrder 中的下标固定分配，曲线/频谱/表格共用同一映射
const PALETTE=["#58a6ff","#3fb950","#d29922","#ff7b72","#bc8cff","#39c5cf"];
function colorOf(pid, order){ const i=(order||[]).indexOf(pid); return PALETTE[(i<0?0:i)%PALETTE.length]; }

function drawLegend(ctx,w,h,items){
  const fs=10; ctx.font=fs+"px sans-serif"; ctx.textAlign="left";
  let x=w-PAD.r-2;
  for(let i=items.length-1;i>=0;i--){
    const t=String(items[i].name); const tw=ctx.measureText(t).width;
    x-=(10+4+tw+8);
    ctx.fillStyle=items[i].color; ctx.fillRect(x, PAD.t+4, 10, 3);
    ctx.fillStyle="#8b949e"; ctx.fillText(t, x+14, PAD.t+fs);
  }
}

function dtOf(t){
  if(!t||t.length<2) return 0.05;
  const d=(t[t.length-1]-t[0])/(t.length-1);
  return (isFinite(d)&&d>0)?d:0.05;
}

function drawSeries(canvas, list){
  const ctx=canvas.getContext("2d"); const w=canvas.width=canvas.clientWidth, h=canvas.height=canvas.clientHeight;
  ctx.clearRect(0,0,w,h);
  const valid=(list||[]).filter(s=>s&&s.data&&s.data.length>0);
  let lo=Infinity, hi=-Infinity;
  valid.forEach(s=>s.data.forEach(v=>{ if(v<lo)lo=v; if(v>hi)hi=v; }));
  if(!valid.length){ lo=-1; hi=1; }
  if(hi-lo<1e-6){ hi+=1; lo-=1; }
  const pad=(hi-lo)*0.1; lo-=pad; hi+=pad;
  drawAxes(ctx,w,h,lo,hi,"mm · 时间(s)");
  ctx.fillStyle="#3f4a58"; ctx.font="9px sans-serif"; ctx.textAlign="center";
  for(let i=0;i<=2;i++){
    const sec=MARKER_WINDOW_S*i/2;
    ctx.fillText(sec.toFixed(sec%1?1:0), PAD.l+(w-PAD.l-PAD.r)*i/2, h-PAD.b+13);
  }
  if(!valid.length){ ctx.fillStyle="#5b6776"; ctx.font="11px sans-serif"; ctx.textAlign="left"; ctx.fillText("等待数据…",PAD.l+30,h/2); return; }
  valid.forEach(s=>{
    ctx.strokeStyle=s.color; ctx.lineWidth=1.6; ctx.beginPath();
    const n=s.data.length;
    const x=i=>{
      const sec=(s.t&&s.t.length===n)?Number(s.t[i]):MARKER_WINDOW_S*i/(n-1||1);
      const ratio=Math.max(0,Math.min(1,sec/MARKER_WINDOW_S));
      return PAD.l+(w-PAD.l-PAD.r)*ratio;
    };
    s.data.forEach((v,i)=>{ const y=PAD.t+(h-PAD.t-PAD.b)*(1-(v-lo)/(hi-lo||1)); i?ctx.lineTo(x(i),y):ctx.moveTo(x(i),y); });
    ctx.stroke();
  });
  drawLegend(ctx,w,h,valid);
}

function drawSpectrum(canvas, specs, gate, ana){
  const ctx=canvas.getContext("2d"); const w=canvas.width=canvas.clientWidth, h=canvas.height=canvas.clientHeight;
  ctx.clearRect(0,0,w,h);
  // 未满窗不出图。频谱是"一整段稳态数据"的统计量：数据不够时画出来的只是趋势/噪声
  // 形状，不是谱。此图直接画后端下发的功率谱（freq/power），与振动分析表里的主频
  // 来自同一套固定窗 FFT 算法——前端不再本地做 FFT，杜绝"第三套口径"。
  const g=gate||{ready:true};
  if(!g.ready || !specs || !specs.length){
    ctx.fillStyle="#5b6776"; ctx.font="11px sans-serif"; ctx.textAlign="left";
    ctx.fillText("等待窗口累积…",PAD.l+24,h/2-4);
    if(g.reason){ ctx.fillStyle="#4a5563"; ctx.font="10px sans-serif"; ctx.fillText(g.reason,PAD.l,h/2+14); }
    return;
  }
  const FMAX=5.0;
  const valid=specs.filter(s=>s&&s.freq&&s.power&&s.freq.length);
  if(valid.length===0){ ctx.fillStyle="#5b6776"; ctx.font="11px sans-serif"; ctx.fillText("等待数据…",PAD.l+30,h/2); return; }
  let hi=0; valid.forEach(o=>o.power.forEach(p=>{ if(p>hi) hi=p; })); hi=hi||1;
  drawAxes(ctx,w,h,0,hi,"dy功率 · Hz");
  valid.forEach(o=>{
    // 只画 0~FMAX 区间（结构振动主频在低段，超出的高频细节无意义）
    const pts=[];
    for(let i=0;i<o.freq.length;i++){ if(o.freq[i]<=FMAX) pts.push([o.freq[i], o.power[i]]); else break; }
    ctx.strokeStyle=o.color; ctx.lineWidth=1.4; ctx.beginPath();
    pts.forEach(([f,p],i)=>{ const x=PAD.l+(w-PAD.l-PAD.r)*(f/FMAX); const y=PAD.t+(h-PAD.t-PAD.b)*(1-p/hi); i?ctx.lineTo(x,y):ctx.moveTo(x,y); });
    ctx.stroke();
  });
  // 横轴频率刻度（原图横轴无刻度，看不出主频落在哪）
  ctx.fillStyle="#3f4a58"; ctx.font="9px sans-serif"; ctx.textAlign="center";
  for(let i=0;i<=5;i++){ ctx.fillText((FMAX*i/5).toFixed(2), PAD.l+(w-PAD.l-PAD.r)*i/5, h-PAD.b+13); }
  ctx.textAlign="left"; ctx.font="10px sans-serif";
  valid.forEach((o,i)=>{
    // 主频直接用后端（主链路同一套 FFT + SNR 门控）的值，前端不再自己找峰——
    // 否则会出现"图上标着 3.12Hz、表格里是 —"这种两个口径的场面。
    const a=(ana||{})[o.name]||{};
    const txt=(a.f1_hz!==null&&a.f1_hz!==undefined)
      ? ("主频 "+Number(a.f1_hz).toFixed(3)+"Hz"+(a.snr!=null?("（SNR "+Number(a.snr).toFixed(1)+"）"):""))
      : "主频 —（SNR 未达标）";
    ctx.fillStyle=o.color;
    ctx.fillText(o.name+"  "+txt, PAD.l+2, h-PAD.b-4-i*11);
  });
  drawLegend(ctx,w,h,valid);
}

function fmt(v,d=2){ return (v===null||v===undefined)?"—":Number(v).toFixed(d); }

function tiltDeg(deg, axis, fbSupported){
  const a=Number(deg)||0;
  if(axis==='fb' && !fbSupported && Math.abs(a)<1e-6) return '0.00° (单目无深度)';
  const dir = axis==='lr' ? (a<0?'左':'右') : (a<0?'后':'前');
  return (a===0?'':dir)+Math.abs(a).toFixed(2)+'°';
}

// ---- 视图切换（同页面内切换，不跳转） ----
function isViewActive(panel, view){
  const b=document.querySelector(`.vbtn[data-panel="${panel}"][data-view="${view}"]`);
  return !!(b && b.classList.contains("active"));
}

// 初始帧可视化图片：固定展示初始画面（src 不变则不重设，避免每秒重载）
function setInitImg(imgId, url, version){
  const img=document.getElementById(imgId);
  if(!img) return;
  if(url){
    img.style.display="block";
    const src=version ? (url+(url.includes("?")?"&":"?")+"v="+encodeURIComponent(version)) : url;
    if(img.getAttribute("src")!==src) img.src=src;
  }
  else {
    // refresh-init 后初始图尚未生成时不请求一个必然 404 的 URL，避免破图。
    img.removeAttribute("src");
    img.style.display="none";
    img.alt="初始帧尚未生成，等待初始化完成";
  }
}

// 初始状态画面（左）：按下拉框选中的某一路相机，渲染其初始帧 + 初始基准
function renderPoleInitView(){
  const st=window.__state; if(!st||!st.pole) return;
  const P=st.pole; const cams=Object.keys(P.cams||{});
  if(!cams.length) return;
  const sel=document.getElementById("poleInitSel");
  let cur=(sel && sel.value && cams.includes(sel.value)) ? sel.value : (P.cam_id||cams[0]);
  if(sel){
    const optsKey=cams.join(",");
    if(sel.dataset.opts!==optsKey){ sel.dataset.opts=optsKey; sel.innerHTML=cams.map(c=>`<option value="${c}">${c}</option>`).join(""); }
    sel.value=cur;
  }
  const ci=P.cams[cur];
  const sig="P:"+cur+":"+JSON.stringify(ci?ci.init:null);
  if(sig!==poleInitViewSig){
    poleInitViewSig=sig;
    setInitImg("poleInitFrame", ci?ci.init_frame_url:null, sig);
    const it=document.querySelector("#poleInitTbl tbody"); it.innerHTML="";
    for(const [pid,v] of Object.entries((ci&&ci.init)||{})){
      it.innerHTML+=`<tr><td>${pid}</td>
        <td>${tiltDeg(v.lr_init,'lr',ci.fb_supported!==undefined?ci.fb_supported:P.fb_supported)}</td>
        <td>${tiltDeg(v.fb_init,'fb',ci.fb_supported!==undefined?ci.fb_supported:P.fb_supported)}</td>
        <td class="ok">${fmt(v.total_init,3)}</td></tr>`;
    }
  }
}

// 初始状态画面（右）：按下拉框选中的某一路相机，渲染其初始帧 + 初始挠度/索力
function renderMarkerInitView(){
  const st=window.__state; if(!st||!st.marker) return;
  const M=st.marker; const cams=Object.keys(M.cams||{});
  if(!cams.length) return;
  const sel=document.getElementById("markerInitSel");
  let cur=(sel && sel.value && cams.includes(sel.value)) ? sel.value : (M.cam_id||cams[0]);
  if(sel){
    const optsKey=cams.join(",");
    if(sel.dataset.opts!==optsKey){ sel.dataset.opts=optsKey; sel.innerHTML=cams.map(c=>`<option value="${c}">${c}</option>`).join(""); }
    sel.value=cur;
  }
  const ci=M.cams[cur];
  const initAna=(ci&&ci.init_analysis)||{};
  const initPids=Object.keys(initAna).sort();
  const sig="M:"+cur+":"+JSON.stringify(initAna);
  if(sig!==markerInitViewSig){
    markerInitViewSig=sig;
    setInitImg("markerInitFrame", ci?ci.init_frame_url:null, sig);
    const iit=document.querySelector("#markerInitTbl tbody");
    if(initPids.length===0){ iit.innerHTML='<tr><td colspan="3" class="empty">暂无初始分析数据</td></tr>'; return; }
    let h="";
    for(const pid of initPids){
      const v=initAna[pid]; const c=colorOf(pid,initPids);
      // 同上：占位值，不加真实口径后缀
      h+=`<tr><td><span style="display:inline-block;width:10px;height:3px;background:${c};vertical-align:middle;margin-right:5px"></span>${pid}</td>`+
        `<td class="ph">${fmt(v.deflection_mm,3)}</td>`+
        `<td class="ph">${fmt(v.cable_force_idx,1)}</td></tr>`;
    }
    iit.innerHTML=h;
  }
}

// 实时媒体：文件 → <video> 原生裸播；流 → 裸 MJPEG（不重绘标注，正常播放）
function renderLiveMedia(panel, st, force){
  const box=document.getElementById(panel==="pole"?"poleLiveMedia":"markerLiveMedia");
  if(!box) return;
  const root = (panel==="pole") ? st.pole : st.marker;
  if(!root) return;
  const cams = root.cams || {};
  const sel = document.getElementById(panel==="pole"?"poleLiveSel":"markerLiveSel");
  let cid = root.cam_id;
  if(sel && Object.keys(cams).length){
    const keys = Object.keys(cams).sort();
    if(sel.dataset.opts !== JSON.stringify(keys)){   // 选项只填一次
      sel.innerHTML="";
      for(const k of keys){
        const o=document.createElement("option");
        o.value=k; o.textContent=(cams[k]&&cams[k].cam_id)||k;
        sel.appendChild(o);
      }
      sel.dataset.opts=JSON.stringify(keys);
    }
    cid = sel.value || root.cam_id || keys[0];
    if(!cams[cid]) cid = (cams[root.cam_id] ? root.cam_id : keys[0]);  // 非法选择回退
    sel.value = cid;
  }
  // The selected camera owns both the media source and its live analysis state.
  // Keep this identity updated even while a file source is still being exported.
  if(panel==="marker") markerLiveCam=cid;
  let src=null;
  if(cams[cid] && cams[cid].source) src=cams[cid].source;   // 按选中相机取源
  else if(root.source) src=root.source;                     // 兜底：单相机/兼容旧结构
  // 真实「无媒体源」才提前返回；处理中/生成失败虽 media 为空，也要渲染占位提示
  if(!src || ((src.type==="file"||src.type==="stream") && !src.media)){
    if(box.dataset.state!=="empty"){ box.innerHTML='<div class="empty">无媒体源（未配置视频/流）</div>'; box.dataset.state="empty"; }
    return;
  }
  // 同一路、同类型、同地址已渲染且不强制 → 绝不重建，否则 <video>/<img> 每 tick 被重置
  if(box.dataset.state==="media" && box.dataset.cam===cid
     && box.dataset.srctype===src.type && (box.dataset.media||"")===(src.media||"")
     && !force) return;
  const cameraChanged = box.dataset.cam !== cid;
  if(panel==="pole" && (force || cameraChanged)) poleSyncing=false;
  let html;
  if(src.type==="file"){
    html = `<video src="${src.media}" autoplay muted loop playsinline controls></video>`;
  } else if(src.type==="file_processing"){
    // 录像文件处理中：不播放原片（无意义），显示占位，跑完自动切到标注视频
    html = `<div class="empty">${src.hint||"录像文件处理中，跑完后自动展示处理结果"}</div>`;
  } else if(src.type==="file_failed"){
    html = `<div class="empty bad">标注视频生成失败，详见后端日志</div>`;
  } else {
    // 流源：dashboard 后端 cv2 直连子码流、MJPEG 推原始画面，浏览器 <img> 直接播放
    // （纯原始、无叠加）。main 不发布图片，实时画面完全由 dashboard 自行拉流。
    html = `<img src="${src.media}" alt="实时画面">`;
  }
  box.innerHTML=html;
  box.dataset.state="media";
  box.dataset.cam=cid;
  box.dataset.srctype=src.type;
  box.dataset.media=src.media||"";
  if(panel==="pole"){
    // 记录当前相机时间线，并给文件视频挂事件做"当前状态随播放联动"
    poleTimeline = (cams[cid] && cams[cid].timeline) || root.timeline || [];
    const v=box.querySelector("video");
    if(v){
      // 播放/暂停/拖动进度条/回绕都按"当前时间"取快照并锁定表格；
      // 暂停点即用户要看的时刻，绝不回退到最新（与 marker 的 pause/seeked/ended 处理对齐）。
      v.addEventListener("timeupdate", ()=>onPoleVideoTime(v));
      v.addEventListener("seeked", ()=>onPoleVideoTime(v, true));
      v.addEventListener("pause", ()=>onPoleVideoTime(v, true));
      v.addEventListener("ended", ()=>onPoleVideoTime(v, true));
    }
  } else {
    // 右侧测振：文件源由播放时间驱动时移曲线 + 振动分析（流源无 currentTime，保持最新）
    markerIsFile = (src.type==="file");
    markerSyncing = false;
    markerWinT = -1;
    const v=box.querySelector("video");
    if(v){
      // 固定不重叠窗口：曲线/分析随"最近一个已完成窗口"切换（0-30 算一次、30-60
      // 再算一次覆盖前者）。末尾 0.5s 内或 ended 时**强制 flush**——用全部数据算
      // 一次当前窗口（短视频即 [0,t_max]），避免"播完了、曲线还停在中途窗口"。
      // 注意 loop 回绕不触发 ended，故靠 duration-currentTime<0.5 自判 nearEnd。
      v.addEventListener("timeupdate", ()=>{
        const d=Number(v.duration);
        const nearEnd = isFinite(d) && d>0 && (d-Number(v.currentTime))<0.5;
        onMarkerVideoTime(v, nearEnd, nearEnd);
      });
      // 拖动进度条后 timeupdate 会被节流挡住，seeked 强制重拉确保曲线立刻跟上
      // （flush 由随后的 timeupdate 按是否 nearEnd 自行判定，这里只强制刷新）
      v.addEventListener("seeked", ()=>{ markerWinT=-1; onMarkerVideoTime(v, true, false); });
      // 暂停必须强制重拉：暂停点就是用户要看的时刻；仅当 ended 时才 flush
      v.addEventListener("pause", ()=>onMarkerVideoTime(v, true, !!v.ended));
      // 非 loop 场景的收尾（loop 时该事件不触发，靠上面的 nearEnd 兜底）
      v.addEventListener("ended", ()=>{ markerWinT=-1; onMarkerVideoTime(v, true, true); });
    }
  }
}

function activePanelState(panel, root){
  if(!root) return {cid:null, data:null};
  const sel=document.getElementById(panel==="pole"?"poleLiveSel":"markerLiveSel");
  const cams=root.cams||{};
  const keys=Object.keys(cams);
  let cid=(sel&&sel.value)||root.cam_id;
  if(!cams[cid] && keys.length) cid=keys[0];
  return {cid:cid, data:cams[cid]||root};
}

// 振动分析表统一渲染入口——"最新窗口"与"播放联动窗口"共用，避免两处表格式漂移
function renderMarkerAna(ana, label, pids){
  const at=document.querySelector("#markerAnaTbl tbody");
  if(!at) return;
  at.innerHTML="";
  const keys=(pids&&pids.length)?pids:Object.keys(ana||{}).sort();
  if(!keys.length){ at.innerHTML='<tr><td colspan="5" class="empty">等待数据…</td></tr>';
    const e0=document.getElementById("anaTime"); if(e0) e0.textContent="—"; return; }
  for(const pid of keys){
    const c=colorOf(pid,pidOrder);
    const dot=`<span style="display:inline-block;width:10px;height:3px;background:${c};vertical-align:middle;margin-right:5px"></span>`;
    const v=(ana||{})[pid];
    if(!v){ at.innerHTML+=`<tr><td>${dot}${pid}</td><td colspan="4" class="empty">—</td></tr>`; continue; }
    // 挠度/索力为占位值（未计算），只显示数值，不再缀"(首帧=100)"这类真实口径说明
    at.innerHTML+=`<tr><td>${dot}${pid}</td><td>${fmt(v.max_amp_mm,3)}</td>
      <td>${fmt(v.f1_hz,4)}</td><td class="ph">${fmt(v.deflection_mm,3)}</td>
      <td class="ph">${fmt(v.cable_force_idx,1)}</td></tr>`;
  }
  const el=document.getElementById("anaTime");
  if(el) el.textContent = label||"—";
}

// 视频播放时间 → 拉取固定窗口（最近一个已完成窗口 / 末尾 flush 用全部数据）的曲线与振动分析并重绘
async function onMarkerVideoTime(v, force, flush){
  if(!markerIsFile || !markerLiveCam) return;
  const t=Number(v.currentTime)||0;
  if(!force && Math.abs(t-markerWinT)<0.4) return;   // 节流：间隔太小不重复请求
  // 忙时不能把 force 请求直接丢掉：pause/ended 常与上一次请求撞车，丢了就永远停在
  // 中途窗口。改为排队，待本次请求结束后补跑（连同 flush 标记一起带过去）。
  if(markerBusy){ if(force) markerQueued={t:t, paused:!!(v.paused||v.ended), flush:!!flush}; return; }
  markerWinT=t;
  const cam=markerLiveCam;
  markerBusy=true;
  try{
    const url=`/api/marker_window?cam=${encodeURIComponent(cam)}&t=${t.toFixed(2)}${flush?"&flush=1":""}`;
    const r=await fetch(url);
    const w=await r.json();
    if(!w || w.cam_id!==markerLiveCam) return;       // 请求途中切了相机 → 丢弃
    applyMarkerWindow(w, v.paused||v.ended);
  }catch(e){
    // 不能完全静默：请求持续失败时曲线会莫名停住而无任何线索。
    // 只警告前 3 次，避免每 0.4s 刷屏（下一帧 timeupdate 会自动重试）。
    if(markerWinErr++ < 3) console.warn("[marker_window] 窗口数据拉取失败，曲线将停在上一次：", e);
  }
  finally{
    markerBusy=false;
    if(markerQueued){
      const q=markerQueued; markerQueued=null;
      onMarkerVideoTime({currentTime:q.t, paused:q.paused, ended:q.paused}, true, q.flush);
    }
  }
}

// 用窗口数据刷新曲线 + 分析表（左侧"当前状态"联动的右侧对应物）
function applyMarkerWindow(w, paused){
  const sM=w.series||{};
  const sPids=Object.keys(sM);
  if(sPids.length) pidOrder=[...new Set([...pidOrder,...sPids,...Object.keys(w.analysis||{})])];
  // 时移曲线：固定窗口内的累计位移（dy）
  const list=sPids.map(p=>({name:p,t:sM[p].t,data:sM[p].dy,color:colorOf(p,pidOrder)}));
  lastShiftList=list;
  lastDt=dtOf(sPids.length? sM[sPids[0]].t : null);
  // 频谱曲线：直接画后端下发的功率谱（freq/power）——不再本地 FFT，保证与主链路同源
  const sp=w.spectra||{};
  lastSpectra=Object.keys(sp).map(p=>({name:p, color:colorOf(p,pidOrder),
                                        freq:sp[p].freq, power:sp[p].power}));
  chartGate={ready:!!w.fft_ready, reason:w.fft_reason||"", cover:w.fft_cover_s||0, need:w.fft_need_s||0};
  lastAna=w.analysis||{};
  if(isViewActive("marker","live")) drawCharts();
  const info=document.getElementById("markerWinInfo");
  if(info){
    const curveInfo=(w.curve_t0!=null&&w.curve_t1!=null)
      ? `曲线 ${w.curve_t0}~${w.curve_t1}s` : "曲线累积中";
    const fftInfo=(w.t0!=null&&w.t1!=null)
      ? `频谱 ${w.t0}~${w.t1}s` : "频谱累积中";
    info.textContent=`${curveInfo} · ${fftInfo}${paused?" ⏸已暂停":""}`;
  }
  // "为什么没值"的提示语由后端给（它才知道是样本不足还是首窗未满），前端不再另算
  const label=(w.t1!=null)
    ? `ts ${w.t1}s · 窗口 ${w.t0}~${w.t1}s${w.note?"（"+w.note+"）":""}`
    : "—";
  renderMarkerAna(w.analysis, label, [...new Set([...sPids,...Object.keys(w.analysis||{})])]);
  markerSyncing=true;   // 自此由播放驱动，tick 不得用"最新窗口"覆盖
}

// 按给定快照渲染左面板"当前状态"表 + 帧秒标题（播放联动与 latest 共用）
function renderPoleCurrent(frame_id, ts, poles, P, paused){
  const ct=document.querySelector("#poleCurTbl tbody");
  if(ct){ ct.innerHTML="";
    if(!Object.keys(poles||{}).length){
      ct.innerHTML='<tr><td colspan="7" class="empty">暂无立柱数据（该相机未检测到立柱）</td></tr>';
    }
    for(const [pid,v] of Object.entries(poles||{})){
      const cls=v.alarm?"bad":(Math.abs(v.delta_lr_deg)>1?"warn":"ok");
      ct.innerHTML+=`<tr><td>${pid}</td>
        <td>${tiltDeg(v.lr_deg,'lr',P.fb_supported)}</td>
        <td>${tiltDeg(v.fb_deg,'fb',P.fb_supported)}</td>
        <td>${fmt(v.total_deg,3)}</td>
        <td class="${cls}">${tiltDeg(v.delta_lr_deg,'lr',P.fb_supported)}</td>
        <td>${v.status||"—"}</td>
        <td class="${v.alarm?'bad':'ok'}">${v.alarm?"是":"否"}</td></tr>`;
    }
  }
  const el=document.getElementById("poleFrame");
  if(el){
    if(!Object.keys(poles||{}).length){
      el.textContent="未检测到立柱（该相机画面中暂无立柱，或标定不匹配）";
      return;
    }
    const sec = (ts!=null)? Math.round(Number(ts)) : ((frame_id!=null)? frame_id+1 : 0);
    el.textContent =
      "frame_id "+(frame_id!=null?frame_id:"?")+" · t="+(ts!=null?Number(ts).toFixed(1):"?")+"s（第 "+sec+" 秒快照）"+(paused?" · ⏸已暂停":"");
  }
}

// 视频播放时间 → 在时间线里取 ts<=currentTime 的最近快照，驱动"当前状态"表。
// 播放/暂停/拖动进度条/回绕都按"当前时间"取快照，并锁定表格不被 tick 的"最新"覆盖；
// 暂停点即用户要看的时刻，绝不回退到最新（修复：旧逻辑在 timeupdate 的 else 分支
// 把表格重置为 PA.current，导致点击暂停后"当前状态"直接跳到最新状态）。
function onPoleVideoTime(video, force){
  const st=window.__state; if(!st || !st.pole) return;
  const P=st.pole;
  const sel=document.getElementById("poleLiveSel");
  const cid=(sel && sel.value) || P.cam_id;
  // 播放/暂停都必须使用下拉框当前相机的状态；否则多相机时画面是 A、表格却跳到 B。
  const PA=(P.cams && P.cams[cid]) || P;
  const tl=poleTimeline;
  if(!tl || !tl.length) return;
  const paused = !!video.paused || !!video.ended;
  const t=Number(video.currentTime)||0;
  let snap=tl[0];
  for(const e of tl){ if(e.ts!=null && e.ts <= t+1e-6) snap=e; else break; }
  renderPoleCurrent(snap.frame_id, snap.ts, snap.poles, PA, paused);
  // 锁定：文件视频播放/暂停联动期间，tick 不得用"最新"覆盖（流源无 timeupdate，
  // 该值保持 false，表格自动跟随 latest）。
  poleSyncing=true;
}

function setupViewSwitch(){
  document.querySelectorAll(".vbtn").forEach(btn=>{
    btn.addEventListener("click", ()=>{
      const panel=btn.dataset.panel, view=btn.dataset.view;
      document.querySelectorAll(`.vbtn[data-panel="${panel}"]`).forEach(b=>b.classList.toggle("active", b===btn));
      document.querySelectorAll(`.viewpane[data-panel="${panel}"]`).forEach(v=>{
        v.style.display = (v.dataset.view===view)?"block":"none";
      });
      // 离开实时视图时暂停文件视频，避免后台空播并重置播放联动
      if(view!=="live"){
        const mb=document.getElementById(panel==="pole"?"poleLiveMedia":"markerLiveMedia");
        if(mb){ const v=mb.querySelector("video"); if(v) v.pause(); }
        if(panel==="pole") poleSyncing=false;
      }
      const st=window.__state;
      if(st){
        if(view==="init"){ if(panel==="pole") renderPoleInitView(); else renderMarkerInitView(); }
        if(view==="live"){ renderLiveMedia(panel, st); if(panel==="marker") drawCharts(); }
      }
    });
  });
  // 下拉框切换时立即重绘初始状态（不等下一轮 tick）
  const psi=document.getElementById("poleInitSel");
  if(psi) psi.addEventListener("change", renderPoleInitView);
  const msi=document.getElementById("markerInitSel");
  if(msi) msi.addEventListener("change", renderMarkerInitView);
  // 实时视图下拉框：切换相机后强制重建媒体（force=true），立即切到该路视频/流
  const pls=document.getElementById("poleLiveSel");
  if(pls) pls.addEventListener("change", ()=>{
    const st=window.__state; if(st && st.pole) renderLiveMedia("pole", st, true);
  });
  const mls=document.getElementById("markerLiveSel");
  if(mls) mls.addEventListener("change", ()=>{
    const st=window.__state; if(st && st.marker) renderLiveMedia("marker", st, true);
  });
}

// 最近一次曲线数据（切换回实时视图时复用，避免空窗）
let lastShiftList=[], lastDt=0.05;
// 频谱出图门限 + 主频表：均由后端下发，前端只读。两条数据通路（/api/state 的
// "最新窗口"与 /api/marker_window 的"播放联动窗口"）共用这两个全局量，谁后写谁生效，
// 保证任一时刻屏幕上"频谱图有没有出图"与"表格里有没有主频"永远同步。
// lastSpectra：后端下发的功率谱（freq/power），直接画，不再本地 FFT。
let chartGate={ready:true, reason:"", cover:0, need:0};
let lastAna={};
let lastSpectra=[];
function drawCharts(){
  drawSeries(document.getElementById("chartShift"), lastShiftList);
  drawSpectrum(document.getElementById("chartSpectrum"), lastSpectra, chartGate, lastAna);
}

// ---- 状态保持 ----
let poleLastFrame=null, markerAnaUpdated=null, pidOrder=[];
let poleInitViewSig=null, poleInitLiveSig=null, markerInitViewSig=null;  // 初始表只在数据真正变化时重绘，避免每秒闪烁
let poleTimeline=[];      // 当前选中立柱相机的时间线（播放联动用）
let poleSyncing=false;    // true=视频播放中，"当前状态"表由播放时间驱动，不随 latest 覆盖
// ---- 右侧测振：视频文件播放联动 ----
// 与左侧同一思路：由视频 currentTime 决定"看哪一段"。左侧只联动状态表，右侧把
// 时移曲线窗口与振动分析一起切到 [t-30, t]，做到"画面播到哪，曲线与分析就是哪段"。
let markerSyncing=false;  // true=播放联动中，tick 不得用"最新窗口"覆盖曲线/分析表
let markerWinT=-1;        // 上一次已加载的窗口终点（节流）；-1 表示待重拉
let markerBusy=false;     // 联动请求进行中，防重入（拖进度条会密集触发 timeupdate）
let markerQueued=null;    // busy 期间被挡下的"强制刷新"（pause/ended 撞车时产生），本次结束后补跑
let markerWinErr=0;       // 联动请求失败计数（只告警前几次，避免刷屏）
let markerIsFile=false;   // 当前测振源是否为视频文件（流没有 currentTime，无法联动）
let markerLiveCam=null;   // 当前选中的测振相机

// 离开 live 视图时清空媒体框：移除 <img>/<video>，浏览器随之停止拉流/播放，
// 避免隐藏期后台空跑 MJPEG 或视频解码。清空 dataset.state 使返回 live 时强制重建。
function pauseLiveMedia(panel){
  const box=document.getElementById(panel==="pole"?"poleLiveMedia":"markerLiveMedia");
  if(box && box.dataset.state==="media"){ box.innerHTML=""; box.dataset.state=""; }
}

async function tick(){
  let st; try{ st=await (await fetch("/api/state")).json(); }catch(e){ return; }
  window.__state=st;
  document.getElementById("clock").textContent=new Date().toLocaleString();

  // 初始状态画面（下拉框选中的某一路相机）按需重绘
  if(st.pole) renderPoleInitView();
  if(st.marker) renderMarkerInitView();
  // 实时媒体：仅当 live 视图激活时渲染；离开 live 视图则清空媒体框，停止后台拉流/播放
  if(st.pole && isViewActive("pole","live")) renderLiveMedia("pole", st);
  else pauseLiveMedia("pole");
  if(st.marker && isViewActive("marker","live")) renderLiveMedia("marker", st);
  else pauseLiveMedia("marker");

  // ---------- 左：立柱 ----------
  const P=st.pole;
  const pp=document.getElementById("polePanel");
  if(!P){ document.getElementById("poleStatus").textContent="无数据"; pp.classList.remove("alarm"); }
  else{
    const PA=isViewActive("pole","live") ? activePanelState("pole",P).data : P;
    pp.classList.toggle("alarm", !!PA.alarm_any);
    // 无立柱数据时绝不能显示"正常"（那是假值）：如实标注未检测到立柱，
    // 并提示可能原因（相机画面里没有立柱 / 标定不匹配 / 首帧未就绪）
    const poleHasData = !!(PA.initialized && Object.keys(PA.current||{}).length);
    document.getElementById("poleStatus").innerHTML = PA.alarm_any
      ? '<span class="badge bad">报警 '+PA.alarms.join(",")+'</span>'
      : (poleHasData ? '<span class="badge ok">正常</span>'
                     : '<span class="badge warn">未检测到立柱</span>');
    // 实时视图内的初始基准表（当前选中相机）：仅在数据变化时重绘
    const sigPL=JSON.stringify([PA.cam_id,PA.init||{}]);
    if(sigPL!==poleInitLiveSig){
      poleInitLiveSig=sigPL;
      const rowOf=(pid,v)=>`<tr><td>${pid}</td>
        <td>${tiltDeg(v.lr_init,'lr',PA.fb_supported)}</td>
        <td>${tiltDeg(v.fb_init,'fb',PA.fb_supported)}</td>
        <td class="ok">${fmt(v.total_init,3)}</td></tr>`;
      const itL=document.querySelector("#poleInitTblLive tbody"); itL.innerHTML="";
      for(const [pid,v] of Object.entries(PA.init||{})){
        itL.innerHTML+=rowOf(pid,v);
      }
      if(!Object.keys(PA.init||{}).length){
        itL.innerHTML='<tr><td colspan="4" class="empty">暂无初始基准（未检测到立柱）</td></tr>';
      }
    }
    // 当前表：frame_id 变化才刷新；视频播放联动期间（poleSyncing）不覆盖
    const poleFrameKey=String(PA.cam_id||"")+":"+String(PA.frame_id==null?"":PA.frame_id);
    if(poleFrameKey!==poleLastFrame){
      poleLastFrame=poleFrameKey;
      if(!poleSyncing){
        renderPoleCurrent(PA.frame_id, PA.frame_ts, PA.current, PA);
      }
      // 实时视频流：当前状态后直线显示当前时间 年-月-日-分-时-秒；录像文件沿用上方格式
      const lt=document.getElementById("poleLiveTime");
      if(PA.source && PA.source.type==="stream"){
        const d=new Date(); const p=n=>String(n).padStart(2,"0");
        lt.textContent="当前时间 "+d.getFullYear()+"-"+p(d.getMonth()+1)+"-"+p(d.getDate())+"-"+p(d.getHours())+"-"+p(d.getMinutes())+"-"+p(d.getSeconds());
      } else { lt.textContent=""; }
    }
  }

  // ---------- 右：测振 ----------
  const M=st.marker;
  if(!M){ document.getElementById("markerStatus").textContent="无数据"; }
  else{
    const MA=isViewActive("marker","live") ? activePanelState("marker",M).data : M;
    document.getElementById("markerStatus").innerHTML='<span class="badge ok">分析中</span>';
    // 初始挠度/索力表（init 视图，由 renderMarkerInitView 按下拉选中的相机渲染）
    // 时移曲线 + 频谱 + 振动分析：仅在"未进入播放联动"时用最新窗口渲染
    // （视频文件播放中由 onMarkerVideoTime 驱动，tick 覆盖会把曲线拽回末尾）
    if(!markerSyncing){
      const sM=MA.series||{};
      pidOrder=[...new Set([...Object.keys(sM),...Object.keys(MA.analysis||{})])].sort();
      const shiftList=pidOrder.filter(p=>sM[p]).map(p=>({name:p,t:sM[p].t,data:sM[p].dy,color:colorOf(p,pidOrder)}));
      const dt=dtOf(shiftList.length? sM[shiftList[0].name].t : null);
      lastShiftList=shiftList; lastDt=dt;
      const info=document.getElementById("markerWinInfo");
      if(info) info.textContent=(MA.curve_t0!=null)
        ? `曲线 ${MA.curve_t0}~${MA.curve_t1}s · ${MA.t0!=null ? `频谱 ${MA.t0}~${MA.t1}s` : "频谱累积中"}`
        : "等待数据…";
      chartGate={ready:!!MA.fft_ready, reason:MA.fft_reason||"", cover:MA.fft_cover_s||0, need:MA.fft_need_s||0};
      lastAna=MA.analysis||{};
      // 频谱曲线：直接画后端下发的功率谱（freq/power），不再本地 FFT，保证与主链路同源。
      // 必须与 applyMarkerWindow 共用同一段赋值——否则 RTSP 流源没有 <video> 的 timeupdate，
      // 这条赋值永远不执行，频谱图就一直停在"等待窗口累积"，而振动分析表却已出主频（前后端分叉）。
      const sp=MA.spectra||{};
      lastSpectra=Object.keys(sp).map(p=>({name:p, color:colorOf(p,pidOrder), freq:sp[p].freq, power:sp[p].power}));
      if(isViewActive("marker","live")) drawCharts();
      const markerAnalysisKey=String(MA.cam_id||"")+":"+String(MA.analysis_updated==null?"":MA.analysis_updated);
      if(markerAnalysisKey!==markerAnaUpdated){
        markerAnaUpdated=markerAnalysisKey;
        const anaPids=Object.keys(MA.analysis||{}).sort((a,b)=>pidOrder.indexOf(a)-pidOrder.indexOf(b));
        renderMarkerAna(MA.analysis,
          MA.analysis_updated?("最新窗口 ts "+MA.analysis_updated+"s"):"—", anaPids);
      }
    }
  }
}
setupViewSwitch();
setInterval(tick, 1000); tick();
</script>
</body>
</html>
"""


def build_records_html(pole_cam, marker_cam):
    pc = pole_cam or ""
    mc = marker_cam or ""
    return RECORDS_HTML_TEMPLATE.replace("__POLE_CAM__", pc).replace("__MARKER_CAM__", mc)


RECORDS_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>柔性光伏支架视觉监测系统 · 报警记录</title>
<style>
  * { box-sizing: border-box; }
  html,body { margin:0; min-height:100%; background:#0b0f17; color:#e6edf3;
    font-family: "Segoe UI", "Microsoft YaHei", system-ui, sans-serif; }
  #clock { position:fixed; top:8px; right:18px; font-size:14px; color:#7d8794; z-index:9; }
  header { padding:10px 18px; font-size:18px; font-weight:600; letter-spacing:1px;
    border-bottom:1px solid #1c2430; display:flex; align-items:center; gap:14px; }
  header .dot { width:10px; height:10px; border-radius:50%; background:#3fb950; box-shadow:0 0 8px #3fb950; }
  .nav { display:flex; gap:10px; }
  .nav a { color:#9fd0ff; text-decoration:none; font-size:13px; font-weight:600; padding:5px 14px; border:1px solid #1c2430; border-radius:8px; background:#111824; }
  .nav a.active { background:#16202e; border-color:#3fb950; color:#7ee787; }
  .nav a:hover { border-color:#3fb950; }
  .wrap { padding:14px 18px; max-width:1100px; margin:0 auto; }
  .bar { display:flex; align-items:center; gap:10px; margin-bottom:10px; flex-wrap:wrap; }
  .bar h2 { margin:0; font-size:15px; color:#9fd0ff; font-weight:600; }
  .filters { display:flex; gap:8px; }
  .filters button { background:#111824; color:#c9d4e0; border:1px solid #1c2430; border-radius:8px; padding:4px 12px; font-size:12px; cursor:pointer; }
  .filters button.active { background:#16202e; border-color:#3fb950; color:#7ee787; }
  .count { color:#7d8794; font-size:13px; margin-left:auto; }
  table { width:100%; border-collapse:collapse; font-size:13px; background:#111824; border:1px solid #1c2430; border-radius:10px; overflow:hidden; }
  th,td { padding:9px 12px; border-bottom:1px solid #1a2230; text-align:left; }
  th { color:#7d8794; font-weight:500; background:#0e1521; }
  tbody tr { cursor:pointer; }
  tbody tr:hover { background:#16202e; }
  tbody tr.sel { background:#1c2a1c; }
  .badge { display:inline-block; padding:1px 8px; border-radius:10px; font-size:11px; }
  .t-pole { background:#3a2230; color:#ff9d96; } .t-cable { background:#2a2340; color:#bc8cff; } .t-amp { background:#23323a; color:#39c5cf; } .t-defl { background:#3a3320; color:#ffd479; }
  .lvl { color:#ff7b72; font-weight:600; }
  .empty { color:#5b6776; font-size:14px; padding:40px; text-align:center; background:#111824; border:1px solid #1c2430; border-radius:10px; }
  .detail { margin-top:14px; background:#111824; border:1px solid #1c2430; border-radius:10px; padding:14px; display:none; }
  .detail.show { display:block; }
  .detail h3 { margin:0 0 10px; font-size:14px; color:#9fd0ff; }
  .detail img { max-width:100%; max-height:420px; object-fit:contain; background:#000; border-radius:6px; display:block; margin-top:8px; }
  .kv { display:grid; grid-template-columns:110px 1fr; gap:4px 10px; font-size:13px; }
  .kv > div:nth-child(odd) { color:#7d8794; }
  .hint { color:#5b6776; font-size:12px; margin-top:8px; }
</style>
</head>
<body>
<div id="clock"></div>
<header><span class="dot"></span>柔性光伏支架可视化监测
  <nav class="nav"><a href="/">实时监控</a><a href="/records" class="active">报警记录</a></nav>
</header>
<div class="wrap">
  <div class="bar">
    <h2>报警记录 · 历史回放</h2>
    <div class="filters" id="filters">
      <button data-f="all" class="active">全部</button>
      <button data-f="pole">立柱超限</button>
      <button data-f="cable">索力报警</button>
      <button data-f="amplitude">振幅报警</button>
      <button data-f="deflection">挠度报警</button>
    </div>
    <span class="count" id="count"></span>
  </div>
  <div id="listBox"></div>
  <div class="detail" id="detail"></div>
</div>

<script>
let ALL=[], curFilter="all";
function fmt(v,d=2){ return (v===null||v===undefined)?"—":Number(v).toFixed(d); }
function typeCls(t){ return t==='pole'?'t-pole':(t==='cable'?'t-cable':(t==='amplitude'?'t-amp':'t-defl')); }
function typeLabel(t){ return t==='pole'?'立柱超限':(t==='cable'?'索力报警':(t==='amplitude'?'振幅报警':'挠度报警')); }

async function load(){
  document.getElementById("clock").textContent=new Date().toLocaleString();
  let R; try{ R=await (await fetch("/api/records")).json(); }catch(e){ return; }
  ALL=R.records||[];
  renderRows(curFilter);
}
function renderRows(filter){
  const box=document.getElementById("listBox");
  const rows=ALL.filter(r=>filter==="all"||r.type===filter);
  document.getElementById("count").textContent="共 "+rows.length+" 条";
  if(!rows.length){ box.innerHTML='<div class="empty">暂无报警记录（output/alarms/ 下无快照、events.jsonl 无报警事件）</div>'; return; }
  let h='<table><thead><tr><th>时间(s)</th><th>类型</th><th>目标</th><th>等级</th><th>说明</th></tr></thead><tbody>';
  rows.forEach(r=>{
    h+=`<tr data-id="${r.id}"><td>${r.ts!=null?fmt(r.ts,1):"—"}</td>`+
       `<td><span class="badge ${typeCls(r.type)}">${typeLabel(r.type)}</span></td>`+
       `<td>${r.pid||"—"}</td><td class="lvl">${r.level!=null?r.level:"—"}</td><td>${r.detail||""}</td></tr>`;
  });
  h+='</tbody></table>';
  box.innerHTML=h;
  box.querySelectorAll("tbody tr").forEach(tr=>{
    tr.onclick=()=>{ box.querySelectorAll("tr").forEach(x=>x.classList.remove("sel")); tr.classList.add("sel"); showDetail(tr.dataset.id); };
  });
}
function showDetail(id){
  const r=ALL.find(x=>x.id===id); const d=document.getElementById("detail");
  if(!r){ d.classList.remove("show"); return; }
  let h=`<h3>${typeLabel(r.type)} · ${r.ts!=null?("t="+fmt(r.ts,1)+"s"):""} · 目标 ${r.pid||"—"}</h3>`;
  h+='<div class="kv">';
  h+=`<div>时间</div><div>${r.ts!=null?fmt(r.ts,3)+" s":"—"}</div>`;
  h+=`<div>类型</div><div>${typeLabel(r.type)}</div>`;
  h+=`<div>目标</div><div>${r.pid||"—"}</div>`;
  h+=`<div>等级</div><div class="lvl">${r.level!=null?r.level:"—"}</div>`;
  h+=`<div>说明</div><div>${r.detail||"—"}</div>`;
  h+='</div>';

  if(r.type==='pole'){
    // 立柱报警：重新编码帧（画框+拟合直线）与原始帧（无叠加）并列展示
    if(r.snapshot){ h+=`<div style="margin-top:10px;color:#7d8794;font-size:12px">重新编码帧（画框+拟合直线）</div><img src="${r.snapshot}?t=${Date.now()}" alt="重新编码帧">`; }
    if(r.original){ h+=`<div style="margin-top:10px;color:#7d8794;font-size:12px">原始帧（无叠加）</div><img src="${r.original}?t=${Date.now()}" alt="原始帧">`; }
  } else if(r.type==='amplitude' || r.type==='cable' || r.type==='deflection'){
    // 振动类报警（振幅/索力/挠度）：同一窗口共享原始视频片段（<video> 原生播）+ 振动分析数据
    const a=r.analysis||(r.data&&r.data.analysis)||{};
    // r.clip = web 可播副本（{原名}_h264.mp4）；r.clip_native = 原视频文件（保留不删）
    if(r.clip){
      h+=`<div style="margin-top:10px;color:#7d8794;font-size:12px">原始视频片段（报警窗口 · web 转码副本 H.264）</div><video src="${r.clip}" controls preload="metadata" style="max-width:100%;max-height:420px;background:#000;border-radius:6px;display:block"></video>`;
    }
    else { h+=`<div class="hint">原始视频片段：当前源为实时流（无落地文件）、尚未抽取，或 web 转码副本生成失败（详见"原视频文件"行），仅保留振动分析数据。</div>`; }
    if(r.clip_native){
      h+=`<div style="margin-top:8px;color:#7d8794;font-size:12px">原视频文件（保持原始编码 · 浏览器可能无法直接播放，可下载留存）</div>`+
         `<div><a href="${r.clip_native}" download style="color:#39c5cf;font-size:12px">⬇ 下载 ${r.clip_native.split('/').pop()}</a></div>`;
    }
    h+='<div class="kv" style="margin-top:10px">';
    h+=`<div>主频</div><div>${a.frequency_hz!=null?fmt(a.frequency_hz,3)+" Hz":"—"}${a.frequency_axis?(" ("+a.frequency_axis+")"):""}</div>`;
    h+=`<div>位移幅值</div><div>${a.amplitude_mm!=null?fmt(a.amplitude_mm,3)+" mm":"—"}（阈值 ${a.amplitude_threshold_mm!=null?fmt(a.amplitude_threshold_mm,2):"?"} mm）</div>`;
    h+=`<div>挠度</div><div>${a.deflection_mm!=null?fmt(a.deflection_mm,2)+" mm":"—"}${a.deflection_placeholder?"（占位）":""}（阈值 ${a.deflection_threshold_mm!=null?fmt(a.deflection_threshold_mm,2):"?"} mm）</div>`;
    h+=`<div>索力</div><div>${a.cable_force_idx!=null?fmt(a.cable_force_idx,2):"—"}${a.cable_force_placeholder?"（占位）":""}（阈值 ${a.cable_force_threshold_idx!=null?fmt(a.cable_force_threshold_idx,2):"?"}）</div>`;
    h+=`<div>窗口</div><div>${a.t0!=null?fmt(a.t0,1):"?"} ~ ${a.t1!=null?fmt(a.t1,1):"?"} s（#${a.window!=null?a.window:"?"}）</div>`;
    h+='</div>';
  } else {
    // 其它未知类型：回退通用渲染
    if(r.snapshot){ h+=`<img src="${r.snapshot}?t=${Date.now()}" alt="快照">`; }
    const skip=new Set(["id","ts","type","type_label","pid","level","detail","snapshot","event","analysis","clip","original"]);
    const extra=r.data||{};
    const keys=Object.keys(extra).filter(k=>!skip.has(k));
    if(keys.length){ h+='<div class="kv" style="margin-top:10px">'; keys.forEach(k=>{ h+=`<div>${k}</div><div>${JSON.stringify(extra[k])}</div>`; }); h+='</div>'; }
  }
  h+='<div class="hint">点击上方其他记录可切换。</div>';
  d.innerHTML=h; d.classList.add("show");
  d.scrollIntoView({behavior:"smooth", block:"nearest"});
}
document.getElementById("filters").addEventListener("click", e=>{
  const b=e.target.closest("button"); if(!b||!b.dataset.f) return;
  curFilter=b.dataset.f;
  document.querySelectorAll("#filters button").forEach(x=>x.classList.remove("active"));
  b.classList.add("active"); renderRows(curFilter);
  document.getElementById("detail").classList.remove("show");
});
load(); setInterval(load, 5000);
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description="柔性光伏支架实时监控大屏")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    run(args.config, args.host, args.port)


if __name__ == "__main__":
    main()
