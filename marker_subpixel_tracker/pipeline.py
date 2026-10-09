"""Per-box localization orchestration shared by single- and multi-target loops.

The single-target and multi-target entry points (main.py and marker_tracing.py)
both need the same "crop -> grayscale -> anchor -> peak-select -> subpixel"
unit per detection box.  This module keeps that unit in one place so the two
loops stay identical in behaviour and never drift apart.
"""
import cv2
import numpy as np

from .detector import crop_box
from .features import detect_marker_center, predict_anchor


def localize_one(frame, box, cfg, prev_center=None, prev_box=None):
    """Locate the marker center inside one detection box (full-frame coords).

    Replicates the per-frame body of ``main.py`` exactly: crop ROI (no expand)
    -> grayscale -> anchor (scheme A: box-displacement prediction when this
    target has a previous measurement, else ROI center) -> detect_marker_center
    (lambda_min peak within the anchor window + cornerSubPix refinement).

    Returns ``(center, lambda_min, continuous)`` where ``center`` is in FULL
    FRAME coordinates (None when lost), ``lambda_min`` the selected response,
    and ``continuous`` True only for an anchor-window-selected peak.
    """
    box = np.asarray(box, dtype=np.float32)
    roi, (ox, oy) = crop_box(frame, box)
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    predicted_global = None
    if prev_center is not None and prev_box is not None:
        predicted_global = predict_anchor(prev_box, prev_center, box)
        anchor = predicted_global - np.array([ox, oy], dtype=np.float32)
    else:
        anchor = np.array([roi.shape[1] / 2.0, roi.shape[0] / 2.0],
                          dtype=np.float32)
    local_center, lambda_min, continuous = detect_marker_center(
        roi_gray, cfg, anchor)
    if local_center is None:
        # No usable peak in the ROI at all. With tracking history we FALL BACK
        # to the box-predicted location (it follows the detection box every
        # frame) so a fast-moving / motion-blurred target still shows a marker
        # that tracks the box instead of the point silently vanishing. Without
        # history there is nothing to predict from, so we report honest loss.
        if predicted_global is not None:
            return predicted_global, float("nan"), False
        return None, float("nan"), False
    center_global = local_center + np.array([ox, oy], dtype=np.float32)
    # Reliability gate (NOT a freeze): the measured subpixel center must stay
    # near its BOX-PREDICTED location ``predicted_global``.  ``predict_anchor``
    # already follows the detection box each frame, so a genuinely moving target
    # shifts both the box and the predicted anchor and the residual stays small
    # -> never gated (real motion is preserved).  When the residual is large the
    # subpixel peak is unreliable -- a corner/background flip at far distance, or
    # a collapsed lambda_min response under motion blur. In BOTH cases the
    # box-predicted location is the safer estimate: it tracks the box and never
    # freezes. We fall back to it (non-continuous) instead of the bad peak or the
    # frozen previous point.
    # NOTE: the old bug froze ``prev_center`` here, which dead-locked the track
    # whenever the target moved > max_jump_px/frame (box moved, point stuck).
    max_jump = float(cfg.get("max_jump_px", 3.0))
    if predicted_global is not None:
        residual = float(np.hypot(center_global[0] - predicted_global[0],
                                  center_global[1] - predicted_global[1]))
        if residual > max_jump:
            return predicted_global, lambda_min, False
    return center_global, lambda_min, continuous
