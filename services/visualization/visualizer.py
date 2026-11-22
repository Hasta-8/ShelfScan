"""Annotated output: one box colour per group, derived from the group ID (deterministic)."""

from __future__ import annotations

import colorsys

from PIL import Image, ImageDraw

UNGROUPED = (128, 128, 128)  # group 0: too small to embed
_GOLDEN = 0.618033988749895


def group_colour(group_id: int) -> tuple[int, int, int]:
    """A stable, well-spread colour per group: golden-ratio steps around the hue circle."""
    if group_id <= 0:
        return UNGROUPED
    hue = (group_id * _GOLDEN) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def annotate(image: Image.Image, detections: list[dict], width: int = 3) -> Image.Image:
    """Return a copy of the image with each detection's box drawn in its group's colour."""
    out = image.copy()
    draw = ImageDraw.Draw(out)
    for det in detections:
        draw.rectangle(det["bbox"], outline=group_colour(det.get("group_id", 0)), width=width)
    return out
