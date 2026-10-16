"""振动报警留存：为合并后的 vibration_alarm 事件（含触发条件 triggered 列表）窗口抽取
*原始* 视频片段 + 写入振动分析数据。

两条留存路径（按源类型区分，对齐用户命名规范）：

- **文件源（视频数据）**：从源视频文件按 [t0, t1] 切出
  ``output/vibration/{t0}_{t1}_original.mp4``，并 **保持原编码（不转 web 可播）**——
  用户明确要求文件源片段"不需要转成 web 端可播放的形式"。优先用 ffmpeg 流拷贝(-c copy)
  精确切片；ffmpeg 缺失时回退 OpenCV 重封装（仍为原始帧、不改 H.264）。

- **RTSP 实时源**：main 不落地源文件，片段由 capture 进程「滚动缓冲」在报警发生时
  实时切出 ``output/vibration/{当前时间}_original.mp4``（H.264，web 可播）。本模块在
  收尾阶段只 *引用* 该片段（校验文件存在），并把分析数据写入侧车 json——
  命名与切割均在 capture 侧完成（见 utils/capture.RollingBuffer）。

每个报警窗口最终都写出 ``output/vibration/<id>.json`` 侧车（id 即事件的 id：
``{cam_id}_{pid}_w{seg}``），内含完整 analysis（含 triggered 列表）+ clip 路径 + source。
同一窗口被多个条件（振幅/挠度/索力）同时触发时，合并为同一条 vibration_alarm 事件、
共享同一视频片段（文件源：同一 ``{t0}_{t1}_original.mp4``；RTSP：同一 ``{当前时间}_original.mp4``）。
大屏 ``build_records`` 直接读该侧车，按 id 关联，不重复造记录。已存在则跳过（幂等）。

**双产物（用户要求：原视频文件不能删，但 web 端要能播）**：
原视频文件（``clip_native``）始终保留、不修改不删除；同时另存一份
``{原视频文件名}_h264.mp4``（``clip``）供 web ``<video>`` 播放。二者都写进侧车，
大屏用 ``clip`` 播放、并单独给出原视频文件的下载入口。转码失败时 ``clip=None``，
如实置空——绝不把浏览器解不了码的原文件当成"可播"给出去。

**保留上限（``marker.alarm.keep_clips``，默认 10）**：只保留最近 N 个报警窗口，
超出的整组删除（见 `prune_vibration_clips`）；配成 0 / -1 / ``all`` / ``inf`` 表示
"无穷大"——全部保留不删。同窗口的 3 个事件（振幅/挠度/索力）共享片段，整组同进同出，
不留"有侧车没视频"的悬空引用。
"""
import json
from pathlib import Path

from utils.common import is_realtime_source, log


def extract_clip(src_path, t0, t1, out_path, fourcc="mp4v", transcode=True):
    """从源视频抽取 [t0, t1] 秒片段写入 out_path（原始帧，不做标注）。

    参数：
        transcode  切完后是否用 ffmpeg 转 H.264（web 可播）。RTSP 滚动缓冲切出的片段
                    需要；文件源保持原编码时传 False，仅 OpenCV 重封装、不改码。
    返回 {"status","out","reason","h264"}。
    """
    import cv2

    src_path = Path(src_path)
    if not src_path.is_file():
        return {"status": "skip", "reason": f"源视频不存在: {src_path}"}
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        return {"status": "skip", "reason": "无法打开源视频"}

    try:
        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*fourcc),
                             src_fps, (w, h))

        start_frame = int(t0 * src_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame))
        written = 0
        dur = max(0.0, t1 - t0)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # 用已写帧数推算更稳（POS_FRAMES 在部分封装下不可靠）
            if written / src_fps >= dur:
                break
            vw.write(frame)
            written += 1
        cap.release()
        vw.release()
        if written == 0:
            try:
                out_path.unlink(missing_ok=True)
            except Exception:
                pass
            return {"status": "skip", "reason": "片段为空（区间无帧）"}

        # 仅当 transcode=True（RTSP）才转 H.264；文件源保持原编码。
        h264 = False
        if transcode:
            try:
                from .export import _to_h264
                h264 = bool(_to_h264(out_path))
            except Exception as exc:
                log("clip-retention", f"H.264 转码失败（保留 mp4v）: {exc}")
        return {"status": "ok", "out": str(out_path), "h264": h264,
                "frames": written}
    finally:
        try:
            cap.release()
        except Exception:
            pass
        try:
            vw.release()
        except Exception:
            pass


def extract_clip_native(src_path, t0, t1, out_path):
    """文件源片段抽取：尽量 ffmpeg 流拷贝(-c copy)保持原编码（不转 web 可播）。

    失败回退 OpenCV 重封装（仍为原始帧、不改用 H.264）。返回 {"status","out","reason"}。
    """
    import shutil
    import subprocess

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg") is not None:
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-ss", f"{t0:.3f}", "-to", f"{t1:.3f}",
                 "-i", str(src_path), "-c", "copy", "-an", str(out_path)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if out_path.is_file() and out_path.stat().st_size > 0:
                return {"status": "ok", "out": str(out_path),
                        "reason": "ffmpeg stream-copy（原编码）"}
        except Exception as exc:
            log("clip-retention", f"ffmpeg 流拷贝失败，回退 OpenCV: {exc}")
    # 回退：OpenCV 重封装（mp4v），仍不转 H.264
    return extract_clip(src_path, t0, t1, out_path, fourcc="mp4v", transcode=False)


def _h264_copy_name(src_path):
    """由原视频文件名推出 web 可播副本名：``{stem}_h264.mp4``。

    例：``0.0_2.0_original.mp4`` → ``0.0_2.0_original_h264.mp4``。
    与用户命名要求一致：可展示文件 = 原文件名 + ``_h264``。
    """
    p = Path(src_path)
    return f"{p.stem}_h264.mp4"


def _probe_codec(path):
    """探测视频编码名（如 h264 / mpeg4 / hevc）；探测不到返回 None。

    ffprobe 不可用时返回 None → 调用方按"未知即转码"处理（宁可转码也不给
    浏览器解不了的副本）。
    """
    import shutil
    import subprocess
    if shutil.which("ffprobe") is None:
        return None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0",
             str(path)],
            check=False, capture_output=True, text=True, timeout=30)
        name = (out.stdout or "").strip().lower()
        return name or None
    except Exception:
        return None


def ensure_web_h264(src_path, out_dir=None):
    """把原视频文件**另存**为浏览器可播的 ``{stem}_h264.mp4``（不动源文件）。

    为什么需要：`extract_clip_native` 对文件源刻意 **流拷贝(-c copy) 保持原编码**
    （实测常为 mp4v/MPEG-4 Part 2、H.265 等浏览器**解不了码**的编码），用户要求
    "原视频文件不能删"，因此不能像 `viz.export._to_h264` 那样**原地覆盖**，
    只能另存一份 H.264 副本。

    关键点：必须**先探测源编码**，仅当确为 H.264 时才用 ``-c copy``（无损快）；
    否则一律转码 libx264。若不探测就copy，mpeg4/mp4v 会被原样复制出来，
    副本仍不可播 —— 这正是本函数要避免的问题。

    参数:
        src_path  原视频文件（只读，绝不修改/删除）
        out_dir   副本输出目录，默认与源同目录
    返回 ``{"status","out","reason","copied","src_codec"}``：
        status=ok    副本就绪（copied=True 表示本次真的转码/重封装出来）
        status=reuse 副本已存在，直接复用（幂等）
        status=skip  ffmpeg 不可用或转码失败 —— 调用方须如实告知"web 不可播"，
                     **不得**把原文件当成可播副本（宁可空缺也不给假值）。
    """
    import shutil
    import subprocess

    src_path = Path(src_path)
    if not src_path.is_file():
        return {"status": "skip", "reason": f"原视频不存在: {src_path}"}
    out_dir = Path(out_dir) if out_dir else src_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / _h264_copy_name(src_path.name)
    if dst.is_file() and dst.stat().st_size > 0:
        return {"status": "reuse", "out": str(dst), "copied": False,
                "src_codec": _probe_codec(src_path)}

    if shutil.which("ffmpeg") is None:
        return {"status": "skip", "out": str(dst), "copied": False,
                "reason": "ffmpeg 不可用，无法生成 web 可播副本"}

    # 只有源确为 H.264 才流拷贝重封装；其余（mpeg4/mp4v/hevc/探测失败）一律转码。
    src_codec = _probe_codec(src_path)
    if src_codec == "h264":
        attempts = [
            ["-c:v", "copy", "-movflags", "+faststart"],
        ]
    else:
        attempts = [
            ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            # 极少数编码 libx264 失败时，回退 mpeg4→H.264 的通用转码参数
            ["-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p"],
        ]

    for idx, args in enumerate(attempts):
        tmp = dst.with_suffix(f".tmp{idx}.mp4")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(src_path), *args, "-an", str(tmp)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if not (tmp.is_file() and tmp.stat().st_size > 0):
                raise RuntimeError("ffmpeg 产出空文件")
            # 校验副本确实是 H.264：宁可判失败，也不能把不可播文件交给 web
            if _probe_codec(tmp) != "h264":
                raise RuntimeError(f"副本编码非 H.264（{_probe_codec(tmp)}）")
            # 只替换 dst（本次新建的副本），绝不触碰源文件
            tmp.replace(dst)
            return {"status": "ok", "out": str(dst), "copied": True,
                    "src_codec": src_codec}
        except Exception as exc:  # noqa: BLE001
            try:
                if tmp.is_file():
                    tmp.unlink()
            except Exception:
                pass
            log("clip-retention",
                f"生成 web 可播副本失败（{src_codec or 'unknown'} → h264）"
                f"{src_path.name}: {exc}")
    return {"status": "skip", "out": str(dst), "copied": False,
            "src_codec": src_codec, "reason": "H.264 转码失败"}


def prune_vibration_clips(out_dir, keep):
    """报警片段保留上限：只保留**最近 keep 个报警窗口**，超出删除旧的。

    以**视频文件本身**为清理单元（而非侧车 json），原因：侧车 ``<id>.json`` 是**收尾
    阶段**才写入的，而 RTSP 长时间运行时不存在收尾，目录里只有 mp4；按视频清理
    在两种场景下都成立。

    **整组删除**：同一报警窗口可能被振幅/挠度/索力多个条件同时触发，它们共享同一
    对文件（``{stem}_original.mp4`` + ``{stem}_h264.mp4``，共 3 个事件 id）。故按
    ``{stem}_original.mp4`` 归组，整组同进同出——否则会留下"有侧车没视频"的悬空
    引用（大屏点开变坏链）。同时清理引用该视频的侧车 json 与遗留的临时文件。

    keep 语义：
      - 正整数 N：保留最近 N 个报警窗口；
      - ``<= 0`` 或非数值（含 ``None`` / ``all`` / ``inf``）：**不限制**，全部保留。

    返回 ``{"kept":n, "dropped":[stem...]}``（dropped 为被删的窗口名，便于日志核对）。
    """
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return {"kept": 0, "dropped": []}

    # 不限制：<=0 / 非数值 / inf / nan 一律视为"全部保留"
    # 注意 float('inf')/float('nan') 转 int 会抛 OverflowError/ValueError，必须一并捕获
    try:
        fk = float(keep)
    except (TypeError, ValueError):
        return {"kept": -1, "dropped": []}
    # inf / nan 明确判定为"无穷大 = 全保留"（int() 无法表示）
    if fk != fk or fk == float("inf"):
        return {"kept": -1, "dropped": []}
    try:
        k = int(fk)
    except (TypeError, ValueError, OverflowError):
        return {"kept": -1, "dropped": []}
    if k <= 0:
        return {"kept": -1, "dropped": []}

    # 以 {stem}_original.mp4 为窗口代表；RTSP/文件源命名都以此结尾
    originals = [p for p in out_dir.glob("*_original.mp4") if p.is_file()]
    if len(originals) <= k:
        return {"kept": len(originals), "dropped": []}

    # 按 mtime 倒序：最新的在最前，保留前 k 个，其余删除
    originals.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    drop = originals[k:]
    keep_names = {p.name for p in originals[:k]}

    dropped, freed = [], 0
    for orig in drop:
        stem = orig.name[:-len("_original.mp4")]
        # 整组：原文件 + web 可播副本 + 可能的临时文件
        targets = [orig,
                   out_dir / f"{stem}_original_h264.mp4",
                   *out_dir.glob(f"{stem}_original*.tmp*.mp4")]
        for t in targets:
            try:
                if t.is_file():
                    freed += t.stat().st_size
                    t.unlink()
            except Exception:
                pass
        dropped.append(stem)

    # 清理引用了被删视频的侧车 json（analysis.clip / analysis.clip_native 命中即可）
    # 判定用"该侧车引用的视频是否还在磁盘上"，而不是拿名字比对 keep_names——
    # 侧车可能引用 _h264.mp4（不在 keep_names 里），按名字比会误删正常侧车。
    for js in out_dir.glob("*.json"):
        try:
            a = json.loads(js.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(a, dict):
            continue
        refs = [str(a.get(k)) for k in ("clip", "clip_native") if a.get(k)]
        if not refs:
            continue
        # 任一引用文件在磁盘上不存在 → 侧车一并删除，避免大屏出现坏链
        if any(not (out_dir / Path(v).name).is_file() for v in refs):
            try:
                js.unlink()
            except Exception:
                pass

    if dropped:
        log("clip-retention",
            f"[振动留存] 保留上限 {k}：删除 {len(dropped)} 个旧报警窗口"
            f"（{dropped[0]} … {dropped[-1]}），释放 {freed / 1048576:.1f} MiB")
    return {"kept": k, "dropped": dropped}


def _keep_clips_cfg(cfg):
    """报警片段保留上限（marker.alarm.keep_clips）；<=0 或非数值 = 不限制。

    默认 10（只保留最近 10 个报警窗口）。配成 0 / -1 / ``all`` / ``inf`` 等
    表示"无穷大"——全部保留，不做任何删除。
    """
    try:
        v = ((cfg.get("marker", {}) or {}).get("alarm", {}) or {}).get(
            "keep_clips", 10)
    except Exception:
        return 10
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return v  # 交给 prune 判定为"不限制"


def retain_vibration_clips(cfg):
    """收尾阶段：为 vibration_alarm（合并振幅/挠度/索力触发）窗口
    抽取/引用原始视频片段 + 振动分析数据。

    每个报警窗口写出 ``output/vibration/<id>.json`` 侧车（id 即事件 id）：
      - 文件源：抽取 ``{t0}_{t1}_original.mp4``（保持原编码，不转 web）；
      - RTSP 实时源：引用 capture 滚动缓冲已切出的 ``{当前时间}_original.mp4``
        （已转 H.264，web 可播）；若切窗文件缺失则 clip=None、状态 missing。

    **双产物（用户要求：原视频文件不能删，但 web 要能播）**：
      原视频文件始终保留（``clip_native``），并额外另存一份
      ``{原视频文件名}_h264.mp4``（``clip``）供 web `<video>` 播放；
      转码失败时 ``clip=None``（如实置空，绝不把不可播的原文件当可播副本）。
    同一窗口被多条件触发时片段文件共享（已存在则复用），不重复抽取。片段路径与 source
    一并写入侧车，供大屏 build_records 直接读取。幂等可重复调用。
    """
    output_dir = Path(cfg["runtime"]["output_dir"])
    config_dir = cfg.get("_config_dir") or "."
    ev_path = output_dir / "events.jsonl"
    if not ev_path.is_file():
        return []
    out_dir = output_dir / "vibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    produced = []
    for ln in ev_path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            e = json.loads(ln)
        except Exception:
            continue
        if e.get("event") != "vibration_alarm":
            continue
        aid = e.get("id")
        if not aid:
            continue
        js = out_dir / f"{aid}.json"
        if js.is_file():
            continue  # 已抽取，幂等跳过

        analysis = dict(e.get("analysis", {}))
        cid = e.get("cam_id")
        cam = next((c for c in cfg.get("cameras", []) if c.get("id") == cid),
                   None)
        # source 优先取事件内嵌（RTSP 实时流由 spectrum 标注），否则按相机 url 推断
        source = analysis.get("source")
        if source is None and cam is not None:
            source = ("realtime" if is_realtime_source(cam.get("url", ""))
                      else "file")

        clip_url = None        # web 可播副本（_h264.mp4），给 <video> 播放
        clip_native = None     # 原视频文件（保持原编码/原始切片），**始终保留不删**
        clip_status = "pending"
        if source == "realtime":
            # 引用 capture 滚动缓冲已切出的片段（命名 {当前时间}_original.mp4）
            cand = analysis.get("clip")
            if cand:
                mp4 = out_dir / Path(cand).name
                if mp4.is_file():
                    clip_native = f"/media/output/vibration/{mp4.name}"
                    # 该片段 capture 侧已 _to_h264 成 H.264；再另存 web 副本，
                    # 使「原文件 + _h264 副本」双产物在两类源上口径一致。
                    r = ensure_web_h264(mp4, out_dir)
                    if r.get("status") in ("ok", "reuse"):
                        hname = _h264_copy_name(mp4.name)
                        clip_url = f"/media/output/vibration/{hname}"
                    clip_status = "ok" if clip_url else (
                        "ok_no_web_copy" if clip_native else "missing")
                else:
                    clip_status = "missing"
        else:
            # 文件源：从源文件抽取 {t0}_{t1}_original.mp4，保持原编码（不转 web）
            src = None
            if cam is not None:
                s = Path(cam.get("url", ""))
                src = s if s.is_absolute() else Path(config_dir) / s
            if src and src.is_file():
                t0 = float(analysis.get("t0", 0.0))
                t1 = float(analysis.get("t1", 0.0))
                stamp = f"{t0:.1f}_{t1:.1f}"
                cname = f"{stamp}_original.mp4"
                cpath = out_dir / cname
                # 同一窗口可能被振幅/挠度/索力多个条件同时触发，片段文件共享；
                # 已存在则直接复用，不重复抽取（幂等）。
                if not cpath.is_file():
                    extract_clip_native(src, t0, t1, cpath)
                if cpath.is_file():
                    # 原视频文件：保留（web 端不解码，仅作证据留存/下载）
                    clip_native = f"/media/output/vibration/{cname}"
                    # 另存 web 可播副本 {原视频文件名}_h264.mp4（不动原文件）
                    r = ensure_web_h264(cpath, out_dir)
                    if r.get("status") in ("ok", "reuse"):
                        clip_url = (f"/media/output/vibration/"
                                    f"{_h264_copy_name(cname)}")
                    clip_status = "ok" if clip_url else (
                        "ok_no_web_copy" if clip_native else "extract_failed")
                else:
                    clip_status = "extract_failed"
            else:
                clip_status = "skip_no_source"

        analysis = dict(analysis)
        analysis["cam_id"] = cid
        analysis["source"] = source
        # clip = web 可播副本；clip_native = 原视频文件（保留、不删）
        analysis["clip"] = clip_url
        analysis["clip_native"] = clip_native
        analysis["clip_status"] = clip_status
        try:
            js.write_text(json.dumps(analysis, ensure_ascii=False, indent=2),
                          encoding="utf-8")
            produced.append(aid)
            log("clip-retention",
                f"[振动留存] {cid} {aid} 源={source} 原文件={clip_native} "
                f"可播副本={clip_url} 状态={clip_status}")
        except Exception as exc:
            log("clip-retention", f"[振动留存] 写 JSON 失败 {aid}: {exc}")

    # 保留上限：只保留最近 keep 个报警窗口，超出删除旧的（默认 10；<=0/非数值=不限制）
    keep = _keep_clips_cfg(cfg)
    try:
        pr = prune_vibration_clips(out_dir, keep)
        if pr.get("kept") == -1:
            log("clip-retention", f"[振动留存] 保留上限=不限制（keep_clips={keep}），全部保留")
        elif pr.get("dropped"):
            log("clip-retention",
                f"[振动留存] 保留最近 {pr['kept']} 个报警窗口，"
                f"已清理 {len(pr['dropped'])} 个")
    except Exception as exc:
        log("clip-retention", f"[振动留存] 清理旧片段异常: {exc}")
    return produced
