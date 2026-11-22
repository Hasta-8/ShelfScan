"""Visual-similarity grouping: DINOv2-small CLS embeddings, agglomerative clustering.

Groups are visual similarity, not brand or SKU identity. ``cluster`` is a pure function
(unit-tested without a model); ``Grouper`` holds the model and is built at startup.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import AgglomerativeClustering


def cluster(embeddings: np.ndarray, similarity_threshold: float) -> np.ndarray:
    """Group L2-normalised embeddings: average linkage on cosine distance, merging while the
    average similarity is ≥ the threshold. Returns 1-based group IDs, one per row."""
    if len(embeddings) == 0:
        return np.zeros(0, dtype=int)
    if len(embeddings) == 1:  # scikit-learn needs ≥ 2 samples to cluster
        return np.ones(1, dtype=int)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=1 - similarity_threshold,
    )
    return clustering.fit_predict(embeddings) + 1


class Grouper:
    def __init__(self, model_name: str, revision: str, cache_dir: Path) -> None:
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(
            model_name, revision=revision, cache_dir=cache_dir
        )
        self.model = AutoModel.from_pretrained(model_name, revision=revision, cache_dir=cache_dir)
        self.model.eval()

    def embed(self, crops: list[Image.Image]) -> np.ndarray:
        """CLS-token embeddings, L2-normalised, one row per crop (one batch)."""
        inputs = self.processor(images=crops, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :]
        emb = emb / emb.norm(dim=1, keepdim=True)
        return emb.cpu().numpy()

    def group(
        self,
        image: Image.Image,
        detections: list[dict],
        similarity_threshold: float = 0.7,
        min_crop_px: int = 20,
    ) -> list[dict]:
        """Add ``group_id`` to each detection; boxes too small to embed get group 0."""
        crops, valid = [], []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            if (x2 - x1) < min_crop_px or (y2 - y1) < min_crop_px:
                continue
            crops.append(image.crop((x1, y1, x2, y2)))
            valid.append(det)
        for det in detections:
            det["group_id"] = 0
        if crops:
            labels = cluster(self.embed(crops), similarity_threshold)
            for det, gid in zip(valid, labels, strict=True):
                det["group_id"] = int(gid)
        return detections
