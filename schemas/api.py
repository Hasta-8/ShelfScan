"""API contracts: what /v1/infer returns, and the one error shape."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

BBox = tuple[int, int, int, int]


class Detection(BaseModel):
    bbox: BBox = Field(
        description="[left, top, right, bottom] in pixels of the upright image (EXIF applied)"
    )
    confidence: float = Field(ge=0, le=1)
    label: Literal["product"] = "product"
    group_id: int = Field(ge=0, description="0 = too small to embed; groups are visual similarity")


class GroupSummary(BaseModel):
    group_id: int
    count: int


class ImageInfo(BaseModel):
    format: str
    width: int
    height: int
    exif_orientation: int = Field(description="1 = stored upright; 2-8 = rotated/flipped upright")


class Timings(BaseModel):
    decode_ms: float
    detect_ms: float
    group_ms: float
    annotate_ms: float
    total_ms: float


class InferResponse(BaseModel):
    request_id: str
    image: ImageInfo
    detections: list[Detection]
    products: int
    groups: list[GroupSummary] = Field(description="Largest first; group 0 listed if present")
    annotated_image_url: str
    timings: Timings
    models: dict[str, str]
    settings: dict[str, float | int]


class ErrorResponse(BaseModel):
    error: str = Field(description="Stable machine-readable code, e.g. unsupported_type")
    detail: str
    request_id: str
