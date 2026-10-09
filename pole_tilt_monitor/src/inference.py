def _load_yolo(model_path):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Ultralytics is required for YOLO26 detection/segmentation. "
            "Install it with: pip install ultralytics"
        ) from exc
    return YOLO(model_path)


class InstanceSegmentationAdapter:
    def __init__(self, model_path, device="cpu", conf=0.35, cls_id=0):
        self.model = _load_yolo(model_path)
        self.device = device
        self.conf = conf
        self.cls_id = cls_id
