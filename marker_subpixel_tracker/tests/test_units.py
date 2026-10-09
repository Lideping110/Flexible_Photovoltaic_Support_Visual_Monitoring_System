"""Regression tests for tracking and peak-selection contracts.

Plain asserts, no pytest dependency:

    python -m marker_subpixel_tracker.tests.test_units

Covers the contracts locked in on 2026-09-30:
- detect_marker_center anchor window (defeats global argmax flips and
  ROI-boundary peaks), fallbacks, and the radius=0 paper mode;
- TargetLock acquisition / loss counting / threshold reacquire / never-
  reacquire / explicit target_id;
- probe_source_fps fallbacks (the VideoWriter speed bug fix).
"""
import cv2
import numpy as np

from marker_subpixel_tracker.src.features import (
    detect_marker_center, predict_anchor,
)
from marker_subpixel_tracker.src.main import probe_source_fps, should_compute_delta
from marker_subpixel_tracker.src.tracker import MultiTargetTracker, TargetLock


CFG = {
    "structure_sigma": 1.5,
    "min_lambda_min": 1.0,
    "subpixel_window": 5,
    "continuity_radius_px": 10,
}


def _synthetic_frame():
    """White frame: 10x10 black square at center, checkerboard patch at the
    right boundary with a much stronger texture response."""
    img = np.full((100, 100), 255, np.uint8)
    img[45:55, 45:55] = 0
    yy, xx = np.mgrid[35:65, 80:100]
    img[35:65, 80:100] = np.where(((yy // 2) + (xx // 2)) % 2 == 0, 0, 255)
    return img


def _tr(tid, conf):
    return {"id": tid, "conf": conf, "box": [0.0, 0.0, 10.0, 10.0], "cls": 0}


def test_anchor_window_beats_global_max():
    """Peak nearest the anchor wins even though the boundary checkerboard is
    globally stronger — this is what kills the ±63px flip / boundary peak."""
    pt, resp, cont = detect_marker_center(_synthetic_frame(), CFG, (50, 50))
    assert cont is True
    assert np.hypot(float(pt[0]) - 50, float(pt[1]) - 50) < 15
    assert np.isfinite(resp) and resp > 0


def test_predict_anchor_static_box_keeps_measurement():
    """Scheme A: a stationary box must not move the anchor (fast-motion lag
    fix must be a no-op when the target is still)."""
    prev_box = [100.0, 200.0, 200.0, 300.0]   # center (150, 250)
    prev_center = [140.0, 260.0]              # point offset (-10, +10)
    cur_box = [100.0, 200.0, 200.0, 300.0]    # identical box
    anchor = predict_anchor(prev_box, prev_center, cur_box)
    assert np.allclose(anchor, [140.0, 260.0], atol=1e-4)


def test_predict_anchor_tracks_box_displacement():
    """Scheme A core: when the box translates, the anchor follows by the same
    amount instead of capping travel at the continuity window radius."""
    prev_box = [100.0, 200.0, 200.0, 300.0]   # center (150, 250)
    prev_center = [140.0, 260.0]              # offset (-10, +10)
    cur_box = [140.0, 200.0, 240.0, 300.0]    # center (190, 250): +40 x
    anchor = predict_anchor(prev_box, prev_center, cur_box)
    # cur_box_center + (prev_center - prev_box_center) = (190,250)+(-10,+10)
    assert np.allclose(anchor, [180.0, 260.0], atol=1e-4)


def test_predict_anchor_preserves_point_offset_from_box():
    """The measured point keeps a fixed offset from the box center across
    frames, so the anchor's offset from the new box center is unchanged."""
    prev_box = [0.0, 0.0, 10.0, 10.0]        # center (5, 5)
    prev_center = [7.0, 3.0]                 # offset (+2, -2)
    cur_box = [100.0, 100.0, 120.0, 120.0]   # center (110, 110)
    anchor = predict_anchor(prev_box, prev_center, cur_box)
    offset = anchor - np.array([110.0, 110.0], dtype=np.float32)
    assert np.allclose(offset, [2.0, -2.0], atol=1e-4)


def test_no_anchor_falls_back_to_global():
    pt, resp, cont = detect_marker_center(_synthetic_frame(), CFG, None)
    assert cont is False
    assert float(pt[0]) > 70  # global argmax lands in the checkerboard


def test_radius_zero_disables_window():
    cfg = dict(CFG, continuity_radius_px=0)
    pt, resp, cont = detect_marker_center(_synthetic_frame(), cfg, (50, 50))
    assert cont is False
    assert float(pt[0]) > 70


def test_weak_window_falls_back_to_global():
    """Anchor far from any feature: window peak below threshold -> global."""
    pt, resp, cont = detect_marker_center(_synthetic_frame(), CFG, (5, 5))
    assert cont is False
    assert float(pt[0]) > 70


def test_blank_image_returns_none():
    pt, resp, cont = detect_marker_center(
        np.full((50, 50), 255, np.uint8), CFG, None)
    assert pt is None
    assert cont is False


def test_below_default_threshold_reports_loss():
    """Scheme D: a weak texture whose peak lambda (~78) is above the legacy
    1.0 threshold but below the 5000 default must report None (honest loss).
    Guards against regressing the default back to 1.0, which would silently
    measure garbage textures again."""
    yy, xx = np.mgrid[0:60, 0:60]
    weak = np.where(((yy // 6) + (xx // 6)) % 2 == 0, 128, 132).astype(np.uint8)
    # default threshold (5000): peak ~78 < 5000 -> honest loss
    pt, resp, cont = detect_marker_center(weak, {}, None)
    assert pt is None
    assert cont is False
    # legacy 1.0 threshold would still (incorrectly) measure it
    pt2, _, _ = detect_marker_center(weak, {"min_lambda_min": 1.0}, None)
    assert pt2 is not None


def test_delta_consecutive_frames_valid():
    """Same track, immediately consecutive frames, continuous peak -> valid."""
    prev = (7, np.array([100.0, 100.0]), np.array([50, 50, 150, 150]))
    assert should_compute_delta(
        np.array([101.0, 100.0]), prev, 7, True, 40, 41) is True


def test_delta_cross_gap_is_nan():
    """The bogus-135px-spike guard: after a lost gap the first measurement
    must NOT emit a displacement (cross-gap delta is not frame-to-frame)."""
    prev = (7, np.array([100.0, 100.0]), np.array([50, 50, 150, 150]))
    # last measured at f40, gap f41-43, remeasured at f44
    assert should_compute_delta(
        np.array([110.0, 100.0]), prev, 7, True, 40, 44) is False
    # also no previous measurement at all (first frame)
    assert should_compute_delta(
        np.array([100.0, 100.0]), None, 7, True, -1, 0) is False


def test_delta_cross_track_or_noncontinuous_is_nan():
    prev = (7, np.array([100.0, 100.0]), np.array([50, 50, 150, 150]))
    # different track id (relock / switch)
    assert should_compute_delta(
        np.array([101.0, 100.0]), prev, 9, True, 40, 41) is False
    # structure-tensor fallback peak (continuous=False)
    assert should_compute_delta(
        np.array([101.0, 100.0]), prev, 7, False, 40, 41) is False
    # no center measured this frame
    assert should_compute_delta(None, prev, 7, True, 40, 41) is False


def test_targetlock_reacquire():
    lock = TargetLock(target_id=None, reacquire_after=3)
    assert lock.select([_tr(1, 0.9), _tr(2, 0.5)], 0)["id"] == 1
    assert lock.select([], 1) is None and lock.lost_frames == 1
    assert lock.select([], 3) is None and lock.lost_frames == 3
    assert lock.select([_tr(2, 0.5)], 4)["id"] == 2  # reacquired
    assert lock.target_id == 2
    assert lock.select([_tr(2, 0.5)], 5)["id"] == 2  # stays


def test_targetlock_never_reacquire():
    lock = TargetLock(target_id=1, reacquire_after=0)
    assert lock.select([_tr(2, 0.9)], 10) is None  # never relocks
    assert lock.select([_tr(1, 0.3)], 11)["id"] == 1  # original returns


def test_targetlock_explicit_id():
    lock = TargetLock(target_id=7, reacquire_after=0)
    assert lock.select([_tr(2, 0.9)], 0) is None
    assert lock.select([_tr(2, 0.9), _tr(7, 0.4)], 1)["id"] == 7


def test_multitrack_delta_gating():
    """MultiTargetTracker: dx/dy valid only between consecutive same-track
    measurements with a continuity-selected peak."""
    mt = MultiTargetTracker()
    assert mt.delta_ok(1, True, 0) is False          # no prior state
    mt.record(1, np.array([100.0, 100.0]), np.array([0, 0, 10, 10]), 5)
    assert mt.delta_ok(1, True, 6) is True           # same track, consecutive
    assert mt.delta_ok(1, False, 6) is False         # non-continuous peak
    assert mt.delta_ok(1, True, 8) is False          # cross-gap (5 -> 8)
    assert mt.delta_ok(99, True, 6) is False         # unknown track


def test_multitrack_independent_state():
    """Each track keeps its own previous state with no cross-talk."""
    mt = MultiTargetTracker()
    mt.record(1, np.array([10.0, 10.0]), np.array([0, 0, 20, 20]), 3)
    mt.record(2, np.array([200.0, 200.0]), np.array([180, 180, 220, 220]), 3)
    c1, _ = mt.previous(1)
    c2, _ = mt.previous(2)
    assert np.allclose(c1, [10.0, 10.0])
    assert np.allclose(c2, [200.0, 200.0])
    # updating track 1 must not disturb track 2
    mt.record(1, np.array([11.0, 11.0]), np.array([0, 0, 20, 20]), 4)
    c1_new, _ = mt.previous(1)
    c2_after, _ = mt.previous(2)
    assert np.allclose(c1_new, [11.0, 11.0])
    assert np.allclose(c2_after, [200.0, 200.0])


def test_multitrack_prune_drops_stale():
    """State for a track unseen beyond max_age is dropped."""
    mt = MultiTargetTracker(max_age=5)
    mt.record(1, np.array([0.0, 0.0]), np.array([0, 0, 1, 1]), 10)
    mt.record(2, np.array([0.0, 0.0]), np.array([0, 0, 1, 1]), 30)
    mt.prune(35)   # track 1 stale (35-10=25 > 5), track 2 fresh (35-30=5)
    assert mt.previous(1) is None
    assert mt.previous(2) is not None


def test_probe_fps_defaults():
    # glob patterns / image suffixes -> default 10 fps
    assert probe_source_fps("some_dir/*.jpg") == 10.0
    assert probe_source_fps("any/path.jpg") == 10.0
    # unreadable video source -> graceful 10 fps default (fps bug guard)
    assert probe_source_fps("nonexistent_dir_xyz/nope.mp4") == 10.0


def main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {t.__name__}: {exc}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
