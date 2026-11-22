"""Settings, loaded once from environment variables (prefix SHELFSCAN_) and `.env`.

Model files live in a shared folder (SHELFSCAN_MODELS_DIR, Hugging Face cache layout)
so several projects can reuse one copy; revisions are pinned so the weights can't
change underneath the code.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# DETR ResNet-50 fine-tuned on SKU-110K. Originally published as
# isalia99/detr-resnet-50-sku110k; that name now redirects here (account renamed).
DETECTOR_MODEL = "is36e/detr-resnet-50-sku110k"
DETECTOR_REVISION = "79e15d3223c36872f4ba9328d1a427ca787e70bd"

EMBEDDER_MODEL = "facebook/dinov2-small"
EMBEDDER_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="SHELFSCAN_", extra="ignore")

    models_dir: Path = Path(".cache/models")
    output_dir: Path = Path("data/outputs")

    # Pipeline (the January defaults; not tuned yet, see the tracker's step 3)
    confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    nms_iou: float = Field(default=0.5, ge=0.0, le=1.0)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    min_crop_px: int = Field(default=20, ge=1)  # smaller boxes aren't embedded (group 0)

    # Results depend slightly on the thread count (floating-point order), so it is fixed.
    # 4 = the laptop's physical cores, which measured faster than 8 hyperthreads.
    torch_threads: int = Field(default=4, ge=1)

    # Upload limits
    max_upload_mb: float = Field(default=15.0, gt=0)
    max_pixels: int = Field(default=40_000_000, gt=0)  # ~40 MP; also the decompression-bomb guard
    min_side_px: int = Field(default=64, ge=1)

    # One inference at a time (CPU-bound, ~2.3 GB peak); later requests wait this long
    # for the slot before getting 503 busy.
    busy_timeout_s: float = Field(default=60.0, ge=0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
