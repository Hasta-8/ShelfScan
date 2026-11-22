"""Test images built in memory, and stand-ins for the two models."""

import io

from PIL import Image, ImageDraw


def shelf_image(width=640, height=480) -> Image.Image:
    """A synthetic 'shelf': coloured blocks on grey, with an asymmetric marker."""
    img = Image.new("RGB", (width, height), (200, 200, 200))
    d = ImageDraw.Draw(img)
    for i, colour in enumerate([(220, 40, 40), (40, 40, 220), (220, 40, 40), (40, 160, 40)]):
        x = 40 + i * 140
        d.rectangle([x, 100, x + 100, 300], fill=colour)
    d.rectangle([0, 0, 30, 15], fill=(0, 0, 0))  # top-left marker: shows the orientation
    return img


def encode(img: Image.Image, fmt="JPEG", exif_orientation: int | None = None) -> bytes:
    buf = io.BytesIO()
    kwargs = {}
    if exif_orientation is not None:
        exif = Image.Exif()
        exif[0x0112] = exif_orientation
        kwargs["exif"] = exif.tobytes()
    img.save(buf, fmt, **kwargs)
    return buf.getvalue()


class FakeDetector:
    """Returns fixed boxes, so API tests need no model. Records the images it saw."""

    def __init__(self, boxes=((40, 100, 140, 300), (180, 100, 280, 300), (5, 5, 15, 15))):
        self.boxes = boxes
        self.seen = []

    def detect(self, image, confidence_threshold=0.8, nms_iou=0.5):
        self.seen.append(image.size)
        return [{"bbox": list(b), "confidence": 0.9, "label": "product"} for b in self.boxes]


class FakeGrouper:
    def group(self, image, detections, similarity_threshold=0.7, min_crop_px=20):
        for i, d in enumerate(detections):
            x1, y1, x2, y2 = d["bbox"]
            small = (x2 - x1) < min_crop_px or (y2 - y1) < min_crop_px
            d["group_id"] = 0 if small else 1 + i % 2
        return detections
