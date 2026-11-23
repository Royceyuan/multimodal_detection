"""Normalized detection boxes and intersection-over-union."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
@dataclass(frozen=True)
class Row:
    class_id: int
    x: float
    y: float
    w: float
    h: float
    confidence: float

    @property
    def xyxy(self) -> np.ndarray:
        return np.asarray(
            (self.x - self.w / 2, self.y - self.h / 2, self.x + self.w / 2, self.y + self.h / 2),
            dtype=np.float64,
        )

def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.empty((0,), dtype=np.float64)
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return np.divide(intersection, box_area + boxes_area - intersection, out=np.zeros_like(intersection), where=box_area + boxes_area - intersection > 0)
