"""Visual diagnostics for one pole's mask-to-endpoint pipeline."""
import json
from pathlib import Path

import cv2
import numpy as np

from .vision import (clean_mask, skeleton_from_mask, local_centerline_points,
                     ordered_skeleton_points, fit_line_ransac,
                     endpoint_band_candidates, line_endpoints)


def _save(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def _points(image, points, color, radius=2):
    out = image.copy()
    for x, y in np.asarray(points, dtype=np.float64):
        cv2.circle(out, (int(round(x)), int(round(y))), radius, color, -1)
    return out


def save_pole_diagnostic(frame, mask, box, output_dir, pole_id=None,
                         frame_id=None, endpoint_ratio=0.10,
                         ransac_residual_px=2.5, ransac_iterations=300,
                         body_trim_ratio=0.10):
    """Save step-by-step images and machine-readable metrics for one mask."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prefix = f"pole_{pole_id}_frame_{frame_id}" if pole_id is not None else f"frame_{frame_id}"
    h, w = frame.shape[:2]
    raw = np.asarray(mask, dtype=bool)
    cleaned = clean_mask(raw)
    skeleton = skeleton_from_mask(cleaned)
    local = local_centerline_points(cleaned, skeleton)
    points = ordered_skeleton_points(skeleton)

    annotated = frame.copy()
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 255), 2)
    _save(out / f"{prefix}_01_undistorted_box.jpg", annotated)

    raw_overlay = frame.copy()
    raw_overlay[raw] = (0.55 * raw_overlay[raw] + 0.45 * np.array([0, 0, 255])).astype(np.uint8)
    _save(out / f"{prefix}_02_raw_mask_overlay.jpg", raw_overlay)
    _save(out / f"{prefix}_03_clean_mask.png", cleaned.astype(np.uint8) * 255)

    skel_img = frame.copy()
    yy, xx = np.where(skeleton)
    skel_img[yy, xx] = (0, 255, 0)
    _save(out / f"{prefix}_04_skeleton.jpg", skel_img)

    center_img = frame.copy()
    center_img = _points(center_img, points, (255, 0, 255), 2)
    _save(out / f"{prefix}_05_local_centerline.jpg", center_img)

    metrics = {
        "pole_id": pole_id, "frame_id": frame_id,
        "box": [float(v) for v in box],
        "raw_mask_area_px": int(raw.sum()),
        "clean_mask_area_px": int(cleaned.sum()),
        "skeleton_pixels": int(skeleton.sum()),
        "centerline_points": int(len(points)),
    }
    fit = fit_line_ransac(points, residual_px=ransac_residual_px,
                          iterations=ransac_iterations,
                          body_trim_ratio=body_trim_ratio) if len(points) >= 5 else None
    if fit is None:
        metrics.update({"failure_stage": "ransac", "reason": "RANSAC 无有效模型"})
        _save(out / f"{prefix}_06_ransac_failed.jpg", center_img)
        (out / f"{prefix}_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        return metrics

    p0, direction = fit["point"], fit["direction"]
    inlier_set = set(int(i) for i in fit["inliers"])
    ransac_img = frame.copy()
    for i, p in enumerate(points):
        color = (0, 255, 0) if i in inlier_set else (0, 0, 255)
        cv2.circle(ransac_img, tuple(np.round(p).astype(int)), 2, color, -1)
    # Draw the fitted line over the observed projection range.
    s = (points - p0) @ direction
    for sval in (float(s.min()), float(s.max())):
        q = p0 + sval * direction
        if sval == float(s.min()):
            q0 = q
        else:
            q1 = q
    cv2.line(ransac_img, tuple(np.round(q0).astype(int)), tuple(np.round(q1).astype(int)), (255, 255, 0), 3)
    _save(out / f"{prefix}_06_ransac_inliers_outliers.jpg", ransac_img)

    lo, hi = np.percentile(s, [0, 100])
    span = max(float(hi - lo), 1e-9)
    ratio = min(0.25, max(0.02, float(endpoint_ratio)))
    bands = endpoint_band_candidates(points, fit, ratio)
    in_band_top = s <= lo + ratio * span
    in_band_bottom = s >= hi - ratio * span
    good = bands["perp"] <= bands["gate_px"]

    endpoint_img = frame.copy()
    # Candidate bands are shown with translucent red/blue horizontal-ish
    # projections; accepted points are green (top) and orange (bottom).
    for p in points[in_band_top]:
        cv2.circle(endpoint_img, tuple(np.round(p).astype(int)), 2, (255, 0, 0), -1)
    for p in points[in_band_bottom]:
        cv2.circle(endpoint_img, tuple(np.round(p).astype(int)), 2, (0, 165, 255), -1)
    for p in points[in_band_top & good]:
        cv2.circle(endpoint_img, tuple(np.round(p).astype(int)), 3, (0, 255, 0), -1)
    for p in points[in_band_bottom & good]:
        cv2.circle(endpoint_img, tuple(np.round(p).astype(int)), 3, (0, 255, 0), -1)
    cv2.line(endpoint_img, tuple(np.round(q0).astype(int)), tuple(np.round(q1).astype(int)), (255, 255, 0), 2)
    _save(out / f"{prefix}_07_endpoint_bands_candidates.jpg", endpoint_img)

    metrics.update({
        "ransac_inlier_count": int(len(fit["inliers"])),
        "ransac_inlier_ratio": float(fit["inlier_ratio"]),
        "ransac_residual_rms_px": float(fit["residual_rms"]),
        "projection_min": float(lo), "projection_max": float(hi),
        "endpoint_span_px": span, "endpoint_ratio": ratio,
        "endpoint_perpendicular_gate_px": float(bands["gate_px"]),
        "top_band_count": int(in_band_top.sum()),
        "bottom_band_count": int(in_band_bottom.sum()),
        "top_candidate_count": int(len(bands["top_good"])),
        "bottom_candidate_count": int(len(bands["bottom_good"])),
        "top_fallback": bool(len(bands["top_good"]) < 2),
        "bottom_fallback": bool(len(bands["bottom_good"]) < 2),
        "body_trim_ratio": float(fit.get("body_trim_ratio", 0.10)),
        "body_point_count": int(fit.get("body_point_count", 0)),
    })
    ep = line_endpoints(points, endpoint_ratio=ratio,
                        residual_px=ransac_residual_px,
                        iterations=ransac_iterations,
                        body_trim_ratio=body_trim_ratio,
                        mask_points=local)
    if ep is None:
        metrics.update({"failure_stage": "endpoint_geometry",
                        "reason": "端部观测范围不足，无法形成有效端点带"})
        _save(out / f"{prefix}_08_final_INVALID.jpg", endpoint_img)
    else:
        final_img = frame.copy()
        top = tuple(np.round(ep["top"]).astype(int))
        bottom = tuple(np.round(ep["bottom"]).astype(int))
        cv2.line(final_img, bottom, top, (255, 255, 0), 3)
        cv2.circle(final_img, top, 7, (0, 255, 0), -1)
        cv2.circle(final_img, bottom, 7, (0, 0, 255), -1)
        metrics.update({"failure_stage": None, "reason": "通过端点筛选",
                        "top": [float(v) for v in ep["top"]],
                        "bottom": [float(v) for v in ep["bottom"]]})
        metrics.update({
            "top_endpoint_source": ep["top_source"],
            "bottom_endpoint_source": ep["bottom_source"],
            "top_endpoint_deviation_px": ep["top_endpoint_deviation_px"],
            "bottom_endpoint_deviation_px": ep["bottom_endpoint_deviation_px"],
            "skeleton_mask_height_coverage": ep["skeleton_mask_height_coverage"],
            "coverage_tolerance_px": ep["coverage_tolerance_px"],
        })
        _save(out / f"{prefix}_08_final_OK.jpg", final_img)

    (out / f"{prefix}_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics
