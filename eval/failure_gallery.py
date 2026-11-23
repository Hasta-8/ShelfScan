"""Rebuild docs/failures/: the pipeline (current defaults) on openly licensed Wikimedia
Commons shelf photos, drawn with thicker boxes and cropped where noted.

    uv run python eval/failure_gallery.py

Downloads each photo's 1920 px Commons thumbnail (the gallery was made from these), runs
detection + grouping exactly as the service does, and writes the annotated JPEGs. Credits
and licences are in docs/failures.md; images from CC BY-SA photos stay CC BY-SA.
"""

from __future__ import annotations

import sys
import time
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.config import get_settings  # noqa: E402
from app.engine import build_engine  # noqa: E402
from services.images import decode_upload  # noqa: E402
from services.visualization.visualizer import annotate  # noqa: E402
from tools.commons import thumbnail  # noqa: E402

# output name -> (Commons file title, crop box in the 1920 px image or None)
GALLERY = {
    "01-merged-different-products": ("File:Honey selection at supermarket.jpeg", None),
    "02-price-tags-and-merged-crackers": (
        "File:Crackers trader Joe's grocery store United States.jpg",
        None,
    ),
    "03-merged-same-bottle-shape": ("File:New World fizzy drinks.jpg", (0, 0, 1360, 1100)),
    "04-transparent-jugs-split": (
        "File:Rows of gallons of drinking water and spring water on a supermarket shelf in"
        " Clearwater, Florida, US, on 12 October 2022 (01).jpg",
        None,
    ),
    "05-printed-icons-detected": (
        'File:Boxes of instant noodles on a supermarket shelf, with the words "First In First'
        ' Out - Retain Freshness" written on them.jpg',
        None,
    ),
    "06-steep-angle-missed": (
        "File:Retail grocery store shelf display of broad range of food bars, meal bars, and"
        " snack bars- 2013-04-19 14-39.jpg",
        None,
    ),
    "07-loose-products-fragmented": (
        "File:Meringues on a supermarket shelf.jpg",
        (0, 0, 930, 1279),
    ),
}


def main() -> None:
    settings = get_settings()
    engine = build_engine(settings)
    out_dir = ROOT / "docs" / "failures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, (title, crop) in GALLERY.items():
        data = thumbnail(title)
        image = decode_upload(data, settings.max_pixels, settings.min_side_px).image
        dets, _ = engine.run(image)
        out = annotate(image, dets, width=6)
        if crop:
            out = out.crop(crop)
        out.thumbnail((1200, 1200))
        buf = BytesIO()
        out.save(buf, "JPEG", quality=82, optimize=True)
        (out_dir / f"{name}.jpg").write_bytes(buf.getvalue())
        print(f"{name}: {len(dets)} products, {len({d['group_id'] for d in dets})} groups")
        time.sleep(4)  # Commons rate limits


if __name__ == "__main__":
    main()
