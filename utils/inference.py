"""批推理公共件：YOLO 模型单份加载 + 批量 predict + ByteTrack 关联。

只被推理子进程（pole / marker worker）import。ultralytics / cv2 全部延迟
import，模块顶层不加载任何推理框架，Supervisor 的 spawn 纪律不受影响。

模块：
- BatchInferenceBase  批量预测基类（list 输入失败自动退化为逐帧）
- SegInference        分割模型（pole 分支，predict 加 retina_masks）
- DetInference        检测模型（marker 分支）
- make_byte_tracker   按 tracker 名构造 BYTETracker
- tracker_update      BYTETracker 关联，返回带 raw id 的检测（可选掩码）
"""
import numpy as np

from .common import log


class BatchInferenceBase:
    """YOLO 模型单份 + 批量 predict + 批量失败自动退化为逐帧前向。

    子类覆盖 `_predict_kwargs()` 提供差异化的 predict 参数
    （如分割模型的 retina_masks）。模型仍只加载一份。
    """

    def __init__(self, weights, conf, cls_id, device, tracker_name, tag):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.conf = float(conf)
        self.cls_id = int(cls_id)
        self.device = device
        self.tracker_name = tracker_name
        self.tag = tag
        self._batch_ok = True

    def _predict_kwargs(self) -> dict:
        return dict(conf=self.conf, classes=[self.cls_id],
                    device=self.device, verbose=False)

    def predict(self, frames):
        """批量前向；list 输入失败时自动退化为逐帧前向（模型仍只加载一份）。"""
        kwargs = self._predict_kwargs()
        if self._batch_ok and len(frames) > 1:
            try:
                return self.model.predict(source=frames, **kwargs)
            except Exception as exc:  # noqa: BLE001
                log(self.tag, f"批量前向失败，退化为逐帧前向: {exc}")
                self._batch_ok = False
        return [self.model.predict(source=f, **kwargs)[0] for f in frames]


class SegInference(BatchInferenceBase):
    """分割模型（pole 分支）：predict 额外加 retina_masks。"""

    def _predict_kwargs(self) -> dict:
        kwargs = super()._predict_kwargs()
        kwargs["retina_masks"] = True
        return kwargs


class DetInference(BatchInferenceBase):
    """检测模型（marker 分支）：无额外参数。"""


def make_byte_tracker(tracker_name: str):
    """按 tracker 配置名构造一个 BYTETracker 实例。"""
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import IterableSimpleNamespace, YAML
    from ultralytics.utils.checks import check_yaml

    cfg = IterableSimpleNamespace(**YAML.load(check_yaml(tracker_name)))
    return BYTETracker(args=cfg)


def tracker_update(bt, result, frame, with_mask: bool = False):
    """一次 BYTETracker.update，返回带 raw id 的检测列表。

    输入 result.boxes（Boxes 对象）；输出 [{box, conf, cls, id}...]，
    with_mask=True 时额外附带二值 mask（分割模型）。使用原始检测框，
    不做 Kalman 平滑。
    """
    if result.boxes is None:
        return []
    boxes = result.boxes.cpu()
    tracks = bt.update(boxes, frame)
    if tracks is None or len(tracks) == 0:
        return []
    tracks = np.asarray(tracks)
    idx = tracks[:, -1].astype(int)
    raw = boxes.data.cpu().numpy()  # (n,6): xyxy/conf/cls
    masks = None
    if with_mask and result.masks is not None:
        masks = result.masks.data.cpu().numpy()
    h, w = frame.shape[:2]
    out = []
    for i in range(len(idx)):
        j = int(idx[i])
        if j < 0 or j >= len(raw):
            continue
        item = {
            "box": raw[j, :4].tolist(),
            "conf": float(raw[j, 4]),
            "cls": int(raw[j, 5]),
            "id": int(tracks[i, 4]),
        }
        if with_mask:
            mask = None
            if masks is not None and j < len(masks):
                mask = (masks[j] > 0.5).astype(np.uint8)
                if mask.shape != (h, w):
                    import cv2

                    mask = cv2.resize(mask, (w, h),
                                      interpolation=cv2.INTER_NEAREST)
            item["mask"] = mask.astype(bool) if mask is not None else None
        out.append(item)
    return out
