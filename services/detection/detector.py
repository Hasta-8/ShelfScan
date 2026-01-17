from transformers import DetrForObjectDetection, DetrImageProcessor
from PIL import Image
import torch

# -------------------------------------------------
# MODEL SETUP (loaded once for performance)
# -------------------------------------------------
MODEL_NAME = "isalia99/detr-resnet-50-sku110k"

processor = DetrImageProcessor.from_pretrained(MODEL_NAME)
model = DetrForObjectDetection.from_pretrained(MODEL_NAME)
model.eval()


# -------------------------------------------------
# IOU COMPUTATION
# -------------------------------------------------
def iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two boxes.
    Boxes are in [x1, y1, x2, y2] format.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0


# -------------------------------------------------
# LIGHTWEIGHT NMS
# -------------------------------------------------
def nms(detections, iou_threshold=0.5):
    """
    Apply simple IoU-based Non-Maximum Suppression.
    """
    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    kept = []

    for det in detections:
        keep = True
        for kept_det in kept:
            if iou(det["bbox"], kept_det["bbox"]) > iou_threshold:
                keep = False
                break
        if keep:
            kept.append(det)

    return kept


# -------------------------------------------------
# MAIN DETECTION FUNCTION
# -------------------------------------------------
def detect_products(image_path: str) -> list:
    """
    Detect products in a retail shelf image using a DETR model
    trained on the SKU-110K dataset.

    Args:
        image_path (str): Path to input image.

    Returns:
        list: List of detections with bbox, confidence, and label.
    """
    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])

    results = processor.post_process_object_detection(
        outputs,
        target_sizes=target_sizes,
        threshold=0.8  # confidence threshold
    )[0]

    detections = []

    for score, box in zip(results["scores"], results["boxes"]):
        x1, y1, x2, y2 = box.tolist()

        detections.append({
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "confidence": round(score.item(), 3),
            "label": "product"  # SKU-110K is class-agnostic
        })

    # Apply lightweight NMS to remove near-duplicate boxes
    detections = nms(detections, iou_threshold=0.5)

    return detections
