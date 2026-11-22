"""Settings. Model files live in a shared folder (SHELFSCAN_MODELS_DIR, Hugging Face cache
layout), so several projects can reuse one copy; revisions are pinned so the weights
can't change underneath the code."""

import os
from pathlib import Path

DEBUG = False
UPLOAD_FOLDER = "data/uploads"
OUTPUT_FOLDER = "data/outputs"

MODELS_DIR = Path(os.environ.get("SHELFSCAN_MODELS_DIR", ".cache/models")).expanduser()

# DETR ResNet-50 fine-tuned on SKU-110K. Originally published as
# isalia99/detr-resnet-50-sku110k; that name now redirects here (account renamed).
DETECTOR_MODEL = "is36e/detr-resnet-50-sku110k"
DETECTOR_REVISION = "79e15d3223c36872f4ba9328d1a427ca787e70bd"

EMBEDDER_MODEL = "facebook/dinov2-small"
EMBEDDER_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
