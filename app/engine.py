"""The pipeline with its models, built once at startup (tests build one from fakes)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Protocol

from PIL import Image

from app.config import (
    DETECTOR_MODEL,
    DETECTOR_REVISION,
    EMBEDDER_MODEL,
    EMBEDDER_REVISION,
    Settings,
)


class DetectorLike(Protocol):
    def detect(self, image: Image.Image, confidence_threshold: float, nms_iou: float) -> list: ...


class GrouperLike(Protocol):
    def group(
        self, image: Image.Image, detections: list, similarity_threshold: float, min_crop_px: int
    ) -> list: ...


class Busy(Exception):
    """Another inference held the CPU for longer than the wait limit."""


@dataclass
class Engine:
    detector: DetectorLike
    grouper: GrouperLike
    settings: Settings
    models: dict = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def run(self, image: Image.Image) -> tuple[list[dict], dict[str, float]]:
        """Detect then group, one request at a time. Returns detections and timings (ms)."""
        s = self.settings
        if not self._lock.acquire(timeout=s.busy_timeout_s):
            raise Busy
        try:
            t0 = time.perf_counter()
            dets = self.detector.detect(image, s.confidence_threshold, s.nms_iou)
            t1 = time.perf_counter()
            dets = self.grouper.group(image, dets, s.similarity_threshold, s.min_crop_px)
            t2 = time.perf_counter()
        finally:
            self._lock.release()
        return dets, {"detect_ms": (t1 - t0) * 1000, "group_ms": (t2 - t1) * 1000}


def build_engine(settings: Settings) -> Engine:
    """Load both models from the shared folder at their pinned revisions."""
    import torch

    from services.detection.detector import Detector
    from services.grouping.grouper import Grouper

    torch.set_num_threads(settings.torch_threads)
    return Engine(
        detector=Detector(DETECTOR_MODEL, DETECTOR_REVISION, settings.models_dir),
        grouper=Grouper(EMBEDDER_MODEL, EMBEDDER_REVISION, settings.models_dir),
        settings=settings,
        models={
            "detector": f"{DETECTOR_MODEL}@{DETECTOR_REVISION[:7]}",
            "embedder": f"{EMBEDDER_MODEL}@{EMBEDDER_REVISION[:7]}",
        },
    )
