import cv2
import numpy as np


def _largest_component(binary):
    """Keep only the largest 8-connected component of a binary image."""
    binary = np.asarray(binary, dtype=bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), 8
    )
    if n <= 1:
        return binary
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == idx


def prune_skeleton(skeleton, min_branch_length=8):
    """Remove short endpoint branches (毛刺) from a one-pixel skeleton."""
    sk = np.asarray(skeleton, dtype=bool).copy()
    if min_branch_length <= 0:
        return sk
    h, w = sk.shape

    def neighbors(y, x):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (dy or dx) and 0 <= y + dy < h and 0 <= x + dx < w:
                    if sk[y + dy, x + dx]:
                        yield y + dy, x + dx

    # Iterative endpoint tracing: a short path ending at a junction is a spur.
    for _ in range(3):
        endpoints = [(y, x) for y, x in zip(*np.where(sk))
                     if sum(1 for _ in neighbors(y, x)) == 1]
        remove = []
        for start in endpoints:
            path, prev, cur = [start], None, start
            for _step in range(min_branch_length + 1):
                nxt = [p for p in neighbors(*cur) if p != prev]
                if len(nxt) != 1:
                    break
                prev, cur = cur, nxt[0]
                path.append(cur)
                degree = sum(1 for _ in neighbors(*cur))
                if degree != 2:
                    break
            degree = sum(1 for _ in neighbors(*cur))
            if degree >= 3 and len(path) <= min_branch_length + 1:
                remove.extend(path[:-1])
        for p in remove:
            sk[p] = False
        sk = _largest_component(sk)
    return sk


def _prepare_skeleton_mask(mask):
    """Apply the common mask cleanup used by both skeleton implementations."""
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    # Opening removes isolated burrs; closing fills tiny segmentation gaps.
    kernel = np.ones((3, 3), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    return (_largest_component(mask_u8 > 0).astype(np.uint8) * 255)


def _row_centerline_skeleton(mask_u8):
    """Build the historical fallback: one horizontal midpoint per occupied row."""
    skel = np.zeros_like(mask_u8, dtype=bool)
    ys = np.where(mask_u8 > 0)[0]
    for y in np.unique(ys):
        xs = np.where(mask_u8[y] > 0)[0]
        if len(xs):
            skel[y, int(round((xs[0] + xs[-1]) / 2))] = True
    return skel


def skeleton_variants(mask, spur_length=8):
    """Return the comparable cleaned thinning and row-midpoint skeletons.

    The returned variants include the same largest-component and spur-pruning
    post-processing used by the production measurement path.  ``thinning`` is
    ``None`` only when the OpenCV contrib implementation is unavailable.
    """
    mask_u8 = _prepare_skeleton_mask(mask)
    thinning = None

    try:
        import cv2.ximgproc as xip
        skel = xip.thinning(
            mask_u8, thinningType=xip.THINNING_ZHANGSUEN
        ) > 0
        thinning = prune_skeleton(_largest_component(skel), spur_length)
    except Exception:
        pass

    fallback = prune_skeleton(
        _largest_component(_row_centerline_skeleton(mask_u8)), spur_length
    )
    return {"thinning": thinning, "fallback": fallback}


def skeleton_from_mask(mask, spur_length=8):
    """Create a cleaned skeleton using contrib thinning when available."""
    variants = skeleton_variants(mask, spur_length=spur_length)
    return variants["thinning"] if variants["thinning"] is not None else variants["fallback"]


def skeleton_points(mask):
    skel = skeleton_from_mask(mask)
    y, x = np.where(skel)
    return np.column_stack([x, y]).astype(np.float64)


def ordered_skeleton_points(skeleton):
    """Return the longest ordered path through an 8-connected skeleton."""
    sk = np.asarray(skeleton, dtype=bool)
    ys, xs = np.where(sk)
    if len(xs) < 2:
        return np.column_stack([xs, ys]).astype(np.float64)

    coords = list(zip(ys.tolist(), xs.tolist()))
    index = {p: i for i, p in enumerate(coords)}
    adjacency = [[] for _ in coords]
    for i, (y, x) in enumerate(coords):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                j = index.get((y + dy, x + dx))
                if j is not None:
                    adjacency[i].append(j)

    endpoints = [i for i, nbs in enumerate(adjacency) if len(nbs) <= 1]
    if not endpoints:
        endpoints = [0]

    def bfs(start):
        parent = [-1] * len(coords)
        dist = [-1] * len(coords)
        queue = [start]
        dist[start] = 0
        for cur in queue:
            for nxt in adjacency[cur]:
                if dist[nxt] < 0:
                    dist[nxt] = dist[cur] + 1
                    parent[nxt] = cur
                    queue.append(nxt)
        return dist, parent

    best_pair = None
    best_distance = -1
    for start in endpoints:
        dist, _ = bfs(start)
        for end in endpoints:
            if dist[end] > best_distance:
                best_distance = dist[end]
                best_pair = (start, end)
    if best_pair is None:
        return np.column_stack([xs, ys]).astype(np.float64)

    start, end = best_pair
    _, parent = bfs(start)
    path = []
    cur = end
    while cur >= 0:
        path.append(cur)
        if cur == start:
            break
        cur = parent[cur]
    path.reverse()
    return np.asarray([[coords[i][1], coords[i][0]] for i in path],
                      dtype=np.float64)


def local_centerline_points(mask, skeleton=None, window=5):
    """Estimate a locally fitted, sub-pixel centerline from mask scanlines.

    For each occupied row, the midpoint of the robust 5--95% span is used;
    a rolling median suppresses jagged mask edges and retains fractional x.
    """
    m = np.asarray(mask, dtype=bool)
    ys = np.where(m.any(axis=1))[0]
    if len(ys) == 0:
        return np.empty((0, 2), dtype=np.float64)
    rows = []
    for y in ys:
        xs = np.where(m[y])[0].astype(np.float64)
        if len(xs) < 2:
            continue
        lo, hi = np.percentile(xs, [5, 95])
        inside = xs[(xs >= lo) & (xs <= hi)]
        if len(inside) == 0:
            inside = xs
        rows.append((float(y), float((inside.min() + inside.max()) / 2.0)))
    if not rows:
        return np.empty((0, 2), dtype=np.float64)
    arr = np.asarray(rows, dtype=np.float64)
    # Median filter x while preserving y and fractional coordinates.
    k = max(1, int(window) // 2)
    x = arr[:, 1].copy()
    for i in range(len(x)):
        x[i] = np.median(x[max(0, i-k):min(len(x), i+k+1)])
    return np.column_stack([x, arr[:, 0]])


def endpoint_band_candidates(points, fit, endpoint_ratio=0.10,
                             max_candidates=200):
    """Return endpoint bands and residual-qualified subsets."""
    points = np.asarray(points, dtype=np.float64)
    empty = np.empty((0, 2), dtype=np.float64)
    if fit is None or len(points) == 0:
        return {"top_band": empty, "bottom_band": empty,
                "top_good": empty, "bottom_good": empty,
                "gate_px": float("inf")}
    p0, d = fit["point"], fit["direction"]
    s = (points - p0) @ d
    lo, hi = np.percentile(s, [0, 100])
    span = max(float(hi - lo), 1e-9)
    ratio = min(0.25, max(0.02, float(endpoint_ratio)))
    residual = float(fit.get("residual_rms", 2.5))
    if not np.isfinite(residual):
        residual = 2.5
    perp = np.abs((points - p0)[:, 0] * d[1] -
                  (points - p0)[:, 1] * d[0])
    gate = max(2.5, 2.5 * residual)
    top_mask = s <= lo + ratio * span
    bottom_mask = s >= hi - ratio * span
    good = perp <= gate

    def limit(arr):
        if len(arr) <= max_candidates:
            return arr
        return arr[np.linspace(0, len(arr)-1, max_candidates).astype(int)]

    return {
        "top_band": limit(points[top_mask]),
        "bottom_band": limit(points[bottom_mask]),
        "top_good": limit(points[top_mask & good]),
        "bottom_good": limit(points[bottom_mask & good]),
        "top_perp": perp[top_mask],
        "bottom_perp": perp[bottom_mask],
        "gate_px": float(gate), "span": span,
        "lo": float(lo), "hi": float(hi), "s": s, "perp": perp,
    }


def filter_endpoint_candidates(points, fit, endpoint_ratio=0.10,
                                max_candidates=200):
    """Select residual-qualified endpoint neighborhoods."""
    band = endpoint_band_candidates(points, fit, endpoint_ratio,
                                    max_candidates)
    return band["top_good"], band["bottom_good"]


def _fit_line_ransac_raw(points, residual_px=2.5, iterations=300):
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 5:
        return None

    rng = np.random.default_rng(12345)
    best = None

    for _ in range(iterations):
        i, j = rng.choice(len(points), 2, replace=False)
        p1, p2 = points[i], points[j]
        d = p2 - p1
        n = np.linalg.norm(d)
        if n < 1e-8:
            continue
        d = d / n

        diff = points - p1
        dist = np.abs(diff[:, 0] * d[1] - diff[:, 1] * d[0])
        inliers = np.where(dist <= residual_px)[0]

        if best is None or len(inliers) > len(best):
            best = inliers

    if best is None or len(best) < 5:
        return None

    P = points[best]
    center = P.mean(axis=0)
    _, _, vh = np.linalg.svd(P - center)
    d = vh[0]
    d = d / np.linalg.norm(d)

    # 图像中v向下为正，统一d指向下方
    if d[1] < 0:
        d = -d

    diff = P - center
    residuals = np.abs(diff[:, 0] * d[1] - diff[:, 1] * d[0])
    return {
        "point": center, "direction": d, "inliers": best,
        "inlier_ratio": float(len(best) / max(1, len(points))),
        "residual_rms": float(np.sqrt(np.mean(residuals ** 2))),
        "residual_median": float(np.median(residuals)),
    }


def fit_line_ransac(points, residual_px=2.5, iterations=300,
                    body_trim_ratio=0.10):
    """Fit the shaft from the ordered skeleton middle, then score all points.

    Endpoint observations are excluded before RANSAC so a widened, hooked,
    or incomplete end cannot rotate the main shaft line.
    """
    points = np.asarray(points, dtype=np.float64)
    trim = min(0.30, max(0.0, float(body_trim_ratio)))
    trim_count = int(np.floor(len(points) * trim))
    if trim_count > 0 and len(points) - 2 * trim_count >= 5:
        body_points = points[trim_count:len(points) - trim_count]
    else:
        body_points = points
        trim_count = 0
    shaft = _fit_line_ransac_raw(body_points, residual_px, iterations)
    if shaft is None:
        return None
    p0, d = shaft["point"], shaft["direction"]
    diff = points - p0
    perp = np.abs(diff[:, 0] * d[1] - diff[:, 1] * d[0])
    all_inliers = np.where(perp <= residual_px)[0]
    if len(all_inliers):
        rms = float(np.sqrt(np.mean(perp[all_inliers] ** 2)))
        median = float(np.median(perp[all_inliers]))
    else:
        rms = float("inf")
        median = float("inf")
    return {
        "point": p0, "direction": d, "inliers": all_inliers,
        "inlier_ratio": float(len(all_inliers) / max(1, len(points))),
        "residual_rms": rms, "residual_median": median,
        "body_trim_ratio": trim,
        "body_point_count": int(len(body_points)),
        "body_start_index": int(trim_count),
        "body_end_index": int(len(points) - trim_count),
    }


def line_endpoints(points, endpoint_ratio=0.10,
                    residual_px=2.5, iterations=300,
                    body_trim_ratio=0.10, mask_points=None,
                    coverage_tolerance_ratio=0.03):
    """Fit a skeleton-only shaft and return projected endpoints.

    Mask centers only supplement an end when skeleton height coverage is
    incomplete. Their perpendicular error affects quality, never line fitting.
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 5:
        return None
    mask_points = (np.empty((0, 2), dtype=np.float64) if mask_points is None
                   else np.asarray(mask_points, dtype=np.float64))
    fit = fit_line_ransac(
        points, residual_px=residual_px, iterations=iterations,
        body_trim_ratio=body_trim_ratio
    )
    if fit is None:
        return None

    p0, d = fit["point"], fit["direction"]
    s = (points - p0) @ d
    smin, smax = np.percentile(s, [0, 100])
    length = smax - smin

    if length < 1:
        return None

    ratio = min(0.25, max(0.02, float(endpoint_ratio)))
    skeleton_band = endpoint_band_candidates(points, fit, endpoint_ratio=ratio)
    mask_band = endpoint_band_candidates(mask_points, fit, endpoint_ratio=ratio)

    skeleton_y_min = float(points[:, 1].min())
    skeleton_y_max = float(points[:, 1].max())
    if len(mask_points) >= 4:
        mask_y_min = float(mask_points[:, 1].min())
        mask_y_max = float(mask_points[:, 1].max())
        mask_height = max(mask_y_max - mask_y_min, 1.0)
        tolerance_px = max(2.0, float(coverage_tolerance_ratio) * mask_height)
        top_missing = skeleton_y_min > mask_y_min + tolerance_px
        bottom_missing = skeleton_y_max < mask_y_max - tolerance_px
        overlap = max(0.0, min(skeleton_y_max, mask_y_max) -
                      max(skeleton_y_min, mask_y_min))
        coverage_ratio = float(np.clip(overlap / mask_height, 0.0, 1.0))
    else:
        mask_y_min = mask_y_max = float("nan")
        tolerance_px = float("nan")
        top_missing = bottom_missing = False
        coverage_ratio = 1.0

    def endpoint_observation(which, use_mask):
        source_band = mask_band if use_mask else skeleton_band
        candidates = source_band[f"{which}_band"]
        good = source_band[f"{which}_good"]
        # Keep mask candidates unfiltered so deviation remains quality evidence.
        fallback = use_mask or len(good) < 2
        selected = candidates if fallback else good
        if len(selected) < 2:
            return None
        raw = np.median(selected, axis=0)
        projected = p0 + float((raw - p0) @ d) * d
        perpendicular = np.abs(
            (candidates - p0)[:, 0] * d[1] -
            (candidates - p0)[:, 1] * d[0]
        )
        deviation = abs((raw - p0)[0] * d[1] - (raw - p0)[1] * d[0])
        spread = float(np.sqrt(np.mean(perpendicular ** 2)))
        return {
            "point": projected, "raw": raw,
            "spread": max(spread, float(deviation)),
            "deviation": float(deviation),
            "count": int(len(candidates)), "gate_count": int(len(good)),
            "fallback": bool(fallback),
            "source": "mask" if use_mask else "skeleton",
        }

    top_obs = endpoint_observation("top", top_missing)
    bottom_obs = endpoint_observation("bottom", bottom_missing)
    if top_obs is None or bottom_obs is None:
        return None
    top, bottom = top_obs["point"], bottom_obs["point"]

    # 图像中v小的一端是顶部
    if top[1] > bottom[1]:
        top, bottom = bottom, top
        top_obs, bottom_obs = bottom_obs, top_obs

    # Stability is measured perpendicular to the fitted centerline; extent
    # along the pole is expected and must not be penalized as noise.
    return {
        "top": top, "bottom": bottom, "fit": fit,
        "top_spread_px": top_obs["spread"],
        "bottom_spread_px": bottom_obs["spread"],
        "top_endpoint_deviation_px": top_obs["deviation"],
        "bottom_endpoint_deviation_px": bottom_obs["deviation"],
        "top_raw": top_obs["raw"], "bottom_raw": bottom_obs["raw"],
        "top_count": top_obs["count"], "bottom_count": bottom_obs["count"],
        "top_gate_count": top_obs["gate_count"],
        "bottom_gate_count": bottom_obs["gate_count"],
        "top_fallback": top_obs["fallback"],
        "bottom_fallback": bottom_obs["fallback"],
        "top_source": top_obs["source"],
        "bottom_source": bottom_obs["source"],
        "top_skeleton_missing": bool(top_missing),
        "bottom_skeleton_missing": bool(bottom_missing),
        "skeleton_mask_height_coverage": coverage_ratio,
        "coverage_tolerance_px": tolerance_px,
        "skeleton_y_range": (skeleton_y_min, skeleton_y_max),
        "mask_y_range": (mask_y_min, mask_y_max),
        "endpoint_gate_px": float(skeleton_band["gate_px"]),
        "body_trim_ratio": float(body_trim_ratio),
    }


def clean_mask(mask, min_area=200):
    m = (mask > 0).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return m > 0

    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    if stats[idx, cv2.CC_STAT_AREA] < min_area:
        return np.zeros_like(m, dtype=bool)

    return labels == idx


def draw_measurement(frame, pole_id, box, result):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,255), 2)

    if result is None:
        return

    top = tuple(np.round(result["top"]).astype(int))
    bottom = tuple(np.round(result["bottom"]).astype(int))

    cv2.circle(frame, top, 5, (0,255,0), -1)
    cv2.circle(frame, bottom, 5, (0,0,255), -1)
    cv2.line(frame, top, bottom, (255,255,0), 2)

    if result.get("status") == "INVALID":
        text = f"ID {pole_id} Q {result.get('quality', 0.0):.2f} INVALID"
    else:
        text = (
            f"ID {pole_id} Q {result.get('quality', 0.0):.2f} OK "
            f"LR {result['delta_lr']:+.3f} "
            f"FB {result['delta_fb']:+.3f} "
            f"T {result['delta_total']:+.3f} deg"
        )
    cv2.putText(
        frame, text, (x1, max(20, y1-8)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 2
    )
