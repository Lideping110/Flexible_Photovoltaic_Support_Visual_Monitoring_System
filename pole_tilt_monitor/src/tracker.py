"""Tracking adapter using Ultralytics' native ByteTrack implementation."""
import cv2
import numpy as np


class ByteTrackTracker:
    def __init__(self, model, device="cpu", conf=0.35, cls_id=0,
                 tracker="bytetrack.yaml", persist=True):
        self.model = model
        self.device, self.conf, self.cls_id = device, conf, cls_id
        self.tracker, self.persist = tracker, persist

    def update(self, frame):
        results = self.model.track(
            source=frame, persist=self.persist, tracker=self.tracker,
            conf=self.conf, classes=[self.cls_id], device=self.device,
            verbose=False, retina_masks=True)
        if not results or results[0].boxes is None:
            return []
        b = results[0].boxes
        boxes, confs = b.xyxy.cpu().numpy(), b.conf.cpu().numpy()
        clss = b.cls.cpu().numpy().astype(int)
        # ByteTrack must provide persistent IDs.  Do not silently fall back to
        # a different nearest-neighbour tracker when tracking metadata is
        # unavailable; such IDs would break per-pole baselines.
        if b.id is None:
            return []
        ids = b.id.int().cpu().numpy()
        if results[0].masks is None:
            # This adapter is intentionally segmentation-only: a detection
            # checkpoint cannot satisfy the one-pass boxes+mask contract.
            return []
        masks = results[0].masks.data.cpu().numpy()
        h, w = frame.shape[:2]
        out = []
        for i, (box, conf, cls) in enumerate(zip(boxes, confs, clss)):
            tid = int(ids[i])
            mask = (masks[i] > 0.5).astype(np.uint8)
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            out.append({"box": box.tolist(), "conf": float(conf),
                        "cls": int(cls), "id": tid, "mask": mask.astype(bool)})
        return out
