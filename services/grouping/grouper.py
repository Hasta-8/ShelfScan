from PIL import Image
import torch
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from transformers import AutoImageProcessor, AutoModel

# -------------------------------------------------
# DINOv2 SMALL MODEL (CPU FRIENDLY)
# -------------------------------------------------
MODEL_NAME = "facebook/dinov2-small"

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()


# -------------------------------------------------
# GROUPING FUNCTION
# -------------------------------------------------
def group_products(detections: list, image_path: str, similarity_threshold=0.7) -> list:
    """
    Group products using DINOv2 visual embeddings.

    Args:
        detections (list): List of detection dicts with bbox.
        image_path (str): Path to original image.
        similarity_threshold (float): Cosine similarity threshold.

    Returns:
        list: Detections with assigned group_id.
    """

    if not detections:
        return detections

    image = Image.open(image_path).convert("RGB")

    crops = []
    valid_detections = []

    # Crop detected products
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]

        # Filter very small boxes (noise)
        if (x2 - x1) < 20 or (y2 - y1) < 20:
            continue

        crop = image.crop((x1, y1, x2, y2))
        crops.append(crop)
        valid_detections.append(det)

    if not crops:
        return detections

    # Preprocess images
    inputs = processor(images=crops, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # Use CLS token embedding
    embeddings = outputs.last_hidden_state[:, 0, :]

    # Normalize embeddings
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    embeddings = embeddings.cpu().numpy()

    # Agglomerative clustering with cosine distance
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=1 - similarity_threshold
    )

    labels = clustering.fit_predict(embeddings)

    # Assign group IDs
    for det, group_id in zip(valid_detections, labels):
        det["group_id"] = int(group_id) + 1

    # Any skipped detections go to group 0
    for det in detections:
        if "group_id" not in det:
            det["group_id"] = 0

    return detections
