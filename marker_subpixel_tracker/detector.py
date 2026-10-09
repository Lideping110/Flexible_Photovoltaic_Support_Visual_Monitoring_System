from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


class MarkerDetector:
    """YOLO marker detector used only for ROI localization/recovery."""

    def __init__(self, weights, confidence=0.25, class_id=0, device="cpu"):
        self.model = YOLO(str(weights))
        self.confidence = float(confidence)
        self.class_id = int(class_id)
        self.device = device

    def detect(self, image):
        results = self.model.predict(
            source=image,
            conf=self.confidence,
            classes=[self.class_id],
            device=self.device,
            verbose=False,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return None
        boxes = results[0].boxes
        scores = boxes.conf.detach().cpu().numpy()
        index = int(np.argmax(scores))
        box = boxes.xyxy[index].detach().cpu().numpy().astype(float)
        return {
            "box": box,
            "confidence": float(scores[index]),
            "class_id": int(boxes.cls[index].detach().cpu().item()),
        }


def crop_box(image, box):
    x1, y1, x2, y2 = np.round(box).astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    return image[y1:y2, x1:x2], (x1, y1)

