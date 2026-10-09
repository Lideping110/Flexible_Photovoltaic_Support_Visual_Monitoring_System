import cv2
import numpy as np


def structure_tensor_lambda_min(gray, sigma=1.5):
    """Compute the paper's Gaussian-weighted Shi-Tomasi response."""
    gray = np.asarray(gray, dtype=np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    ksize = max(3, int(round(6 * float(sigma) + 1)) | 1)
    a = cv2.GaussianBlur(gx * gx, (ksize, ksize), sigma)
    b = cv2.GaussianBlur(gx * gy, (ksize, ksize), sigma)
    c = cv2.GaussianBlur(gy * gy, (ksize, ksize), sigma)
    trace = a + c
    discriminant = np.sqrt(np.maximum((a - c) ** 2 + 4.0 * b * b, 0.0))
    return 0.5 * (trace - discriminant)


def refine_subpixel(gray, points, window=5):
    """Refine a paper-selected integer point to subpixel precision."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    if len(points) == 0:
        return points.reshape(0, 2)
    half = max(2, int(window) // 2)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        40,
        1e-3,
    )
    refined = points.copy()
    cv2.cornerSubPix(
        gray,
        refined,
        (half, half),
        (-1, -1),
        criteria,
    )
    return refined.reshape(-1, 2)


def predict_anchor(prev_box, prev_center, cur_box):
    """Predict the peak-selection anchor for the current frame (scheme A).

    The measured point is assumed to keep a fixed offset from the detection
    box center across consecutive frames.  ByteTrack regresses the box
    independently every frame, so the box does NOT lag under fast motion;
    anchoring to it lets the search window follow a fast-moving target instead
    of capping the measured point's travel at ``continuity_radius_px`` (the
    root cause of the slow-motion-lag bug).

    Returns the anchor in GLOBAL image coordinates; the caller subtracts the
    current ROI origin to get ROI-local coordinates for ``detect_marker_center``.
    """
    prev_center = np.asarray(prev_center, dtype=np.float32)
    prev_box = np.asarray(prev_box, dtype=np.float32)
    cur_box = np.asarray(cur_box, dtype=np.float32)
    prev_box_center = np.array([(prev_box[0] + prev_box[2]) / 2.0,
                                (prev_box[1] + prev_box[3]) / 2.0],
                               dtype=np.float32)
    cur_box_center = np.array([(cur_box[0] + cur_box[2]) / 2.0,
                               (cur_box[1] + cur_box[3]) / 2.0],
                              dtype=np.float32)
    return cur_box_center + (prev_center - prev_box_center)


def local_peaks(response, min_val=1000.0, min_dist=5):
    """Return local maxima of ``response`` (value, x, y) sorted by value desc.

    A pixel qualifies when it is strictly the maximum inside its 3x3
    neighbourhood and above ``min_val``; ``min_dist`` thins clusters so only
    the strongest peak within that radius is kept.
    """
    h, w = response.shape
    pad = np.pad(response, 1, mode="constant", constant_values=-np.inf)
    raw = []
    for y in range(h):
        for x in range(w):
            v = pad[y + 1, x + 1]
            if v < min_val:
                continue
            if v == pad[y:y + 3, x:x + 3].max():
                raw.append((float(v), x, y))
    raw.sort(reverse=True)
    kept = []
    for v, x, y in raw:
        if all(np.hypot(x - kx, y - ky) > min_dist for _, kx, ky in kept):
            kept.append((v, x, y))
    return kept


def detect_marker_center(gray, cfg, anchor=None):
    """Locate the single marker center used by the paper.

    The paper evaluates the smaller eigenvalue in a sliding window and uses
    the largest valid lambda_min as the marker's central point.  The integer
    maximum is then refined to subpixel precision.

    Window radius (anchor present):
      * ``continuity_radius_px`` > 0  -> legacy FIXED radius (deprecated; this
        fixed window is what caused the far-distance flip, because at 9m the
        ROI (38x37px) is smaller than the 41x41px window and the window
        degenerates into a global argmax over the whole target).
      * otherwise (the new default) -> PROPORTIONAL radius
        ``clamp(win_ratio * ROI_short_side, 3, win_max)``.  At far distance the
        ROI shrinks, so the window shrinks with it and stays a small patch
        around the central dot instead of covering the whole board.

    Peak selection inside the window: the peak NEAREST the anchor is chosen.
    The 8 marker corners always sit farther from the (predicted) centre than
    the central dot, so locking onto the nearest peak removes corner flips
    entirely -- even when a corner is momentarily the strongest response.

    Contract preserved for the regression tests: when ``anchor`` is None, the
    window is disabled (``continuity_radius_px == 0``), or the windowed peak
    is below ``min_lambda_min``, selection falls back to the global argmax
    (and reports honest loss / None when even that is below threshold).

    Returns ``(point, response, continuous)``: point may be None,
    ``response`` is the selected lambda_min, and ``continuous`` is True only
    when the point came from the anchor window.
    """
    response = structure_tensor_lambda_min(
        gray, sigma=float(cfg.get("structure_sigma", 1.5))
    )
    if response.size == 0:
        return None, float("nan"), False
    threshold = float(cfg.get("min_lambda_min", 5000.0))
    win_ratio = float(cfg.get("win_ratio", 0.25))
    win_max = float(cfg.get("win_max", 20))
    min_spacing = int(cfg.get("min_spacing", 5))
    h, w = response.shape
    cy = cx = None
    selected = float("nan")
    continuous = False
    radius_cfg = cfg.get("continuity_radius_px", None)
    if anchor is not None and radius_cfg != 0:
        if radius_cfg is not None and radius_cfg > 0:
            radius = int(radius_cfg)               # legacy fixed radius
        else:
            sbox = min(w, h)
            radius = int(min(win_max, max(3, win_ratio * sbox)))  # NEW proportional
        px, py = float(anchor[0]), float(anchor[1])
        x1, x2 = max(0, int(px) - radius), min(w, int(px) + radius + 1)
        y1, y2 = max(0, int(py) - radius), min(h, int(py) + radius + 1)
        if x2 > x1 and y2 > y1:
            window = response[y1:y2, x1:x2]
            if float(np.max(window)) >= threshold:
                peaks = local_peaks(window, min_val=threshold, min_dist=min_spacing)
                if peaks:
                    # nearest the anchor == the central dot (robust to flips)
                    best = min(peaks, key=lambda pv: np.hypot(
                        (pv[1] + x1) - px, (pv[2] + y1) - py))
                    cx, cy = best[1] + x1, best[2] + y1
                    selected = best[0]
                else:
                    wy, wx = np.unravel_index(int(np.argmax(window)), window.shape)
                    cy, cx = wy + y1, wx + x1
                    selected = float(np.max(window))
                continuous = True
    if not continuous:
        # anchor None / window disabled / windowed peak too weak -> global argmax
        # (legacy contract; NOT the preferred path at far distance).
        max_response = float(np.max(response))
        if not np.isfinite(max_response) or max_response < threshold:
            return None, max_response, False
        cy, cx = np.unravel_index(int(np.argmax(response)), response.shape)
        selected = max_response
    point = refine_subpixel(
        gray,
        np.array([[float(cx), float(cy)]], dtype=np.float32),
        int(cfg.get("subpixel_window", 5)),
    )
    return point[0], selected, continuous


def measure_outer_diameter_px(frame, box, margin=30, rel_lo=0.25, rel_hi=0.6,
                               return_debug=False):
    """对检测框裁剪区域二值分割，提取靶标**最外圈圆**直径(像素)。

    关键改进（相比旧版"取裁剪区最大圆"）：半径范围用**检测框本身的短边**
    sbox 限定在 [rel_lo*sbox, rel_hi*sbox]，从而排除靶标附近更大的闭合边缘
    （柱子/背景框/棋盘格外环）被误套成大圆甚至超出检测框。范围内取最大半径
    圆即外圆。HoughCircles 失败才退化为轮廓外接圆，并同样按面积过滤。

    返回：
      默认: (直径px, 圆心(原帧坐标), 半径px) 或 None
      return_debug=True: ((直径..., 圆心..., 半径...), debug_dict) 或 (None, debug_dict)
        debug_dict 含 crop/gray/blurred/binary/candidates/chosen/method，
        供标定脚本逐步存图核对。
    """
    debug = {"crop": None, "gray": None, "blurred": None,
             "binary": None, "candidates": None, "chosen": None,
             "method": None}
    bx1, by1, bx2, by2 = np.round(box).astype(int)
    box_w0, box_h0 = bx2 - bx1, by2 - by1
    if box_w0 <= 0 or box_h0 <= 0:
        return (None, debug) if return_debug else None
    sbox = min(box_w0, box_h0)
    h, w = frame.shape[:2]
    x1 = max(0, bx1 - margin); y1 = max(0, by1 - margin)
    x2 = min(w - 1, bx2 + margin); y2 = min(h - 1, by2 + margin)
    if x2 <= x1 or y2 <= y1:
        return (None, debug) if return_debug else None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return (None, debug) if return_debug else None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    debug["crop"] = crop.copy()
    debug["gray"] = gray.copy()
    debug["blurred"] = g.copy()
    hh, ww = g.shape
    r_lo = max(3, int(sbox * rel_lo))
    r_hi = int(sbox * rel_hi)
    # 优先 HoughCircles：半径限定在 [r_lo, r_hi]，范围内取最大者 = 最外圈
    circ = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2,
                            minDist=max(ww, hh) // 2, param1=80, param2=18,
                            minRadius=r_lo, maxRadius=r_hi)
    if circ is not None:
        biggest = max(circ[0], key=lambda c: c[2])
        hough_img = crop.copy()
        for c in circ[0]:
            cv2.circle(hough_img, (int(c[0]), int(c[1])), int(c[2]),
                       (120, 120, 120), 1)
        cv2.circle(hough_img, (int(biggest[0]), int(biggest[1])),
                   int(biggest[2]), (0, 255, 0), 2)
        debug["candidates"] = hough_img
        debug["chosen"] = hough_img.copy()
        debug["method"] = "hough"
        res = (2.0 * float(biggest[2]),
               (x1 + float(biggest[0]), y1 + float(biggest[1])),
               float(biggest[2]))
        return (res, debug) if return_debug else res
    # 退化：Otsu 二值 + 轮廓外接圆，仅保留面积落在合理环带内的轮廓
    amin = 3.14159265 * r_lo * r_lo
    amax = 3.14159265 * r_hi * r_hi
    last_img = crop.copy()
    for inv in (cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY):
        _, th = cv2.threshold(g, 0, 255, inv + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cand = [c for c in cnts if amin <= cv2.contourArea(c) <= amax]
        vis = crop.copy()
        cv2.drawContours(vis, cand, -1, (120, 120, 120), 1)
        if cand:
            c = max(cand, key=cv2.contourArea)
            (cx, cy), r = cv2.minEnclosingCircle(c)
            cv2.circle(vis, (int(cx), int(cy)), int(r), (0, 255, 0), 2)
            debug["binary"] = th.copy()
            debug["candidates"] = vis
            debug["chosen"] = vis.copy()
            debug["method"] = "contour"
            res = (2.0 * float(r), (x1 + float(cx), y1 + float(cy)), float(r))
            return (res, debug) if return_debug else res
        last_img = vis
    debug["candidates"] = last_img
    return (None, debug) if return_debug else None
