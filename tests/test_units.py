"""Pure functions: geometry, clustering, colours, upload decoding. No models needed."""

import numpy as np
import pytest
from helpers import encode, shelf_image
from PIL import Image

from services.detection.detector import iou, nms
from services.grouping.grouper import cluster
from services.images import ImageRejected, decode_upload
from services.visualization.visualizer import UNGROUPED, annotate, group_colour


def test_iou():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [10, 10, 20, 20]) == 0.0
    assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)
    assert iou([0, 0, 0, 0], [0, 0, 0, 0]) == 0.0  # degenerate boxes don't divide by zero


def test_nms_keeps_the_most_confident_of_overlapping_boxes():
    dets = [
        {"bbox": [0, 0, 10, 10], "confidence": 0.7},
        {"bbox": [1, 0, 11, 10], "confidence": 0.9},  # overlaps the first (IoU ~0.82)
        {"bbox": [50, 50, 60, 60], "confidence": 0.8},
    ]
    kept = nms(dets, iou_threshold=0.5)
    assert [d["confidence"] for d in kept] == [0.9, 0.8]


def test_nms_threshold_is_exclusive_and_ties_keep_input_order():
    a = {"bbox": [0, 0, 10, 10], "confidence": 0.9, "id": "a"}
    b = {"bbox": [5, 0, 15, 10], "confidence": 0.9, "id": "b"}  # IoU exactly 1/3
    assert [d["id"] for d in nms([a, b], iou_threshold=1 / 3)] == ["a", "b"]
    assert [d["id"] for d in nms([a, b], iou_threshold=0.3)] == ["a"]


def test_cluster_groups_similar_vectors_and_handles_tiny_inputs():
    e = np.array([[1, 0], [0.99, 0.141], [0, 1], [0.1, 0.995]], dtype=float)
    e = e / np.linalg.norm(e, axis=1, keepdims=True)
    labels = cluster(e, similarity_threshold=0.9)
    assert labels[0] == labels[1] and labels[2] == labels[3] and labels[0] != labels[2]
    assert labels.min() == 1  # 1-based; 0 is reserved for "too small to embed"
    assert cluster(e[:1], 0.9).tolist() == [1]  # used to crash: sklearn needs >= 2 samples
    assert cluster(e[:0], 0.9).tolist() == []


def test_group_colours_are_stable_and_distinct():
    assert group_colour(0) == UNGROUPED and group_colour(-1) == UNGROUPED
    assert group_colour(3) == group_colour(3)
    assert len({group_colour(g) for g in range(1, 50)}) == 49


def test_annotate_is_deterministic_and_leaves_the_input_alone():
    img = shelf_image()
    dets = [{"bbox": [40, 100, 140, 300], "group_id": 1}, {"bbox": [5, 5, 15, 15], "group_id": 0}]
    a, b = annotate(img, dets), annotate(img, dets)
    assert a.tobytes() == b.tobytes()
    assert img.getpixel((40, 100)) != a.getpixel((40, 100))  # a copy was drawn on


def test_decode_accepts_jpeg_png_webp():
    for fmt in ("JPEG", "PNG", "WEBP"):
        d = decode_upload(encode(shelf_image(), fmt), max_pixels=10**7, min_side_px=64)
        assert (d.format, d.width, d.height, d.image.mode) == (fmt, 640, 480, "RGB")


def test_decode_applies_exif_orientation():
    """A photo stored sideways with EXIF orientation 6 must come out upright."""
    upright = shelf_image()
    stored = upright.transpose(Image.Transpose.ROTATE_90)  # what a phone writes
    d = decode_upload(encode(stored, exif_orientation=6), max_pixels=10**7, min_side_px=64)
    assert d.exif_orientation == 6 and (d.width, d.height) == upright.size
    assert sum(d.image.getpixel((10, 5))) < 150  # the black marker is back at the top-left


@pytest.mark.parametrize(
    ("data", "status", "code"),
    [
        (b"", 422, "empty_file"),
        (b"%PDF-1.4 not an image", 415, "unsupported_type"),
        (encode(shelf_image(), "GIF"), 415, "unsupported_type"),
        (encode(shelf_image(60, 60)), 422, "too_small"),
        (encode(shelf_image())[:400], 422, "corrupt_image"),
    ],
)
def test_decode_rejects_bad_uploads(data, status, code):
    with pytest.raises(ImageRejected) as err:
        decode_upload(data, max_pixels=10**7, min_side_px=64)
    assert (err.value.status, err.value.code) == (status, code)


def test_decode_refuses_too_many_pixels_before_decompressing():
    with pytest.raises(ImageRejected) as err:
        decode_upload(encode(shelf_image(2000, 2000)), max_pixels=1_000_000, min_side_px=64)
    assert (err.value.status, err.value.code) == (413, "too_many_pixels")
