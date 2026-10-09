"""ByteTrack tracking for marker ROI boxes.

Unlike ``pole_tilt_monitor`` (which inherits ``model.track()``'s Kalman-smoothed
boxes), this adapter returns the YOLO RAW detection boxes and uses ByteTrack
ONLY for persistent ID association.  Rationale: under fast motion ByteTrack's
Kalman filter lags the target (observed as a ~22px box offset at 1/1000s
shutter), and that lagged box feeds ``predict_anchor``, pushing the peak-search
window off the true corner and locking onto a weaker competing peak.  The raw
detection box is regressed independently every frame and does not lag, so the
anchor (and therefore the measured center) stays correct.
"""
import numpy as np
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml


class ByteTrackTracker:
    """YOLO raw detection + ByteTrack ID association for marker ROI boxes.

    We deliberately do NOT use ``model.track()``.  ``model.track()`` returns
    ``STrack.xyxy`` which is the Kalman-filtered ``self.mean[:4]`` — a smoothed
    box that lags under fast motion.  Instead we call ``model.predict()`` for
    the raw per-frame boxes, feed them to a bare ``BYTETracker`` to obtain the
    association (whose output array's LAST column is the index into the raw
    detection set), then index back into the raw boxes.  The result is
    persistent IDs + un-smoothed (non-lagging) boxes.
    """

    def __init__(self, weights, confidence=0.25, class_id=0, device="cpu",
                 tracker="bytetrack.yaml", persist=True):
        self.model = YOLO(str(weights))
        self.confidence = float(confidence)
        self.class_id = int(class_id)
        self.device = device
        self.tracker = tracker
        self.persist = persist
        # Bare ByteTrack instance for ID association only (not its Kalman boxes).
        cfg = IterableSimpleNamespace(**YAML.load(check_yaml(tracker)))
        self._bt = BYTETracker(args=cfg)

    def update(self, frame):
        if not self.persist:
            self._bt.reset()
        results = self.model.predict(
            source=frame, conf=self.confidence, classes=[self.class_id],
            device=self.device, verbose=False,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []
        det = results[0].boxes.cpu().numpy()  # raw detections (xyxy/conf/cls)
        # tracks: [x1, y1, x2, y2, id, score, cls, idx] where idx indexes back
        # into the raw detection set.  We take the raw box via that idx.
        tracks = self._bt.update(det, frame)
        if len(tracks) == 0:
            return []
        idx = tracks[:, -1].astype(int)
        raw_boxes = det.xyxy[idx]           # raw boxes (NOT Kalman-smoothed)
        ids = tracks[:, 4].astype(int)
        confs = tracks[:, 5]
        clss = tracks[:, 6].astype(int)
        return [
            {"box": [float(v) for v in raw_boxes[i]], "conf": float(confs[i]),
             "cls": int(clss[i]), "id": int(ids[i])}
            for i in range(len(ids))
        ]


class TargetLock:
    """Pin the subpixel pipeline to one ByteTrack ID across frames.

    ByteTrack may emit several candidates per frame (false positives,
    neighbouring markers) while this pipeline measures a single target.
    The lock guarantees that frame-to-frame dx/dy always refer to the same
    physical target, which is what makes the deltas meaningful.
    """

    def __init__(self, target_id=None, reacquire_after=0):
        self.target_id = None if target_id is None else int(target_id)
        self.reacquire_after = int(reacquire_after)
        self.last_seen_frame = 0
        self.lost_frames = 0

    def select(self, tracks, frame_index):
        """Return the locked track for this frame, or None when lost."""
        if self.target_id is not None:
            for t in tracks:
                if t["id"] == self.target_id:
                    self.last_seen_frame = frame_index
                    self.lost_frames = 0
                    return t
            self.lost_frames = frame_index - self.last_seen_frame
            # Optionally re-lock onto the best available track after the
            # locked one has been missing for too long (e.g. the target
            # left and re-entered the frame, or ByteTrack re-issued IDs).
            if (self.reacquire_after > 0
                    and self.lost_frames >= self.reacquire_after
                    and tracks):
                best = max(tracks, key=lambda t: t["conf"])
                self.target_id = best["id"]
                self.last_seen_frame = frame_index
                self.lost_frames = 0
                return best
            return None
        # No lock yet: acquire the most confident candidate.
        if tracks:
            best = max(tracks, key=lambda t: t["conf"])
            self.target_id = best["id"]
            self.last_seen_frame = frame_index
            self.lost_frames = 0
            return best
        return None


class MultiTargetTracker:
    """Per-target measurement state for MULTI-target localization.

    Unlike ``TargetLock`` (which pins the whole pipeline to one ID), this keeps
    an independent state per ``track_id`` so every detected target gets its own
    anchor prediction and frame-to-frame displacement, with zero cross-talk
    between targets.  Frame-consecutivity gating is enforced per track with the
    same rule as the single-target path: dx/dy is only valid between CONSECUTIVE
    frames of the SAME track, and only for a continuity-selected peak — the
    first measurement after a lost gap must not emit a cross-gap displacement.
    """

    def __init__(self, max_age=30):
        # track_id -> {"center": np.ndarray, "box": np.ndarray, "frame": int}
        self.states = {}
        self.max_age = int(max_age)

    def previous(self, track_id):
        """Return ``(prev_center, prev_box)`` or ``None`` when unknown."""
        st = self.states.get(track_id)
        if st is None:
            return None
        return st["center"], st["box"]

    def delta_ok(self, track_id, continuous, frame_index):
        """True when this frame's dx/dy is valid for ``track_id``.

        Requires a prior measurement of the same track on the IMMEDIATELY
        preceding frame and a continuity-selected peak.  The caller already
        guarantees a center was measured this frame.
        """
        st = self.states.get(track_id)
        if st is None or not continuous:
            return False
        return st["frame"] == frame_index - 1

    def record(self, track_id, center, box, frame_index):
        """Store this frame's measurement as the new previous state."""
        self.states[track_id] = {
            "center": np.asarray(center, dtype=np.float32).copy(),
            "box": np.asarray(box, dtype=np.float32).copy(),
            "frame": int(frame_index),
        }

    def prune(self, frame_index):
        """Drop state for tracks not seen for ``max_age`` frames."""
        stale = [t for t, s in self.states.items()
                 if frame_index - s["frame"] > self.max_age]
        for t in stale:
            del self.states[t]
