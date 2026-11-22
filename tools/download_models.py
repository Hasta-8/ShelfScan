"""Download the two models at their pinned revisions into SHELFSCAN_MODELS_DIR.

    uv run python tools/download_models.py

Weights only as safetensors (about 245 MB in all); files already present are reused.
"""

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import (  # noqa: E402
    DETECTOR_MODEL,
    DETECTOR_REVISION,
    EMBEDDER_MODEL,
    EMBEDDER_REVISION,
    get_settings,
)

MODELS_DIR = get_settings().models_dir

for repo, revision in [(DETECTOR_MODEL, DETECTOR_REVISION), (EMBEDDER_MODEL, EMBEDDER_REVISION)]:
    path = snapshot_download(
        repo,
        revision=revision,
        cache_dir=MODELS_DIR,
        allow_patterns=["*.json", "*.safetensors", "README.md"],
    )
    print(f"ok  {repo}@{revision[:7]} -> {path}")
