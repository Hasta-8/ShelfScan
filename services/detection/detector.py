"""Product detection: DETR ResNet-50 fine-tuned on SKU-110K, then IoU-based NMS.

``iou`` and ``nms`` are pure functions (unit-tested without a model). ``Detector`` holds
the model; it is built once at application startup, never at import.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image


def iou(box1, box2) -> float:
    """Intersection over Union of two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0


def nms(detections: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    """Greedy NMS: keep the most confident box, drop others overlapping it above the threshold.

    The sort is stable, so equal confidences keep their original order (deterministic).
    """
    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    kept: list[dict] = []
    for det in detections:
        if all(iou(det["bbox"], k["bbox"]) <= iou_threshold for k in kept):
            kept.append(det)
    return kept


class Detector:
    """DETR detector; ``detect`` returns boxes in the given image's pixel coordinates."""

    def __init__(self, model_name: str, revision: str, cache_dir: Path) -> None:
        from transformers import DetrForObjectDetection, DetrImageProcessor

        self.processor = DetrImageProcessor.from_pretrained(
            model_name, revision=revision, cache_dir=cache_dir
        )
        self.model = DetrForObjectDetection.from_pretrained(
            model_name, revision=revision, cache_dir=cache_dir
        )
        self.model.eval()

    def detect(
        self, image: Image.Image, confidence_threshold: float = 0.8, nms_iou: float = 0.5
    ) -> list[dict]:
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([image.size[::-1]]),
            threshold=confidence_threshold,
        )[0]
        detections = [
            {
                "bbox": [int(v) for v in box.tolist()],
                "confidence": round(score.item(), 3),
                "label": "product",  # SKU-110K is class-agnostic
            }
            for score, box in zip(results["scores"], results["boxes"], strict=True)
        ]
        return nms(detections, iou_threshold=nms_iou)
