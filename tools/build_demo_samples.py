"""Build the demo's three fixed samples: photo, precomputed API response, annotated image.

    uv run python tools/build_demo_samples.py

Downloads each photo from Wikimedia Commons (openly licensed; credits in
demo/samples/samples.json and the README) and sends it through the real API
(/v1/infer?explain=true) in-process, so the precomputed results are exactly what the
service returns. The demo page shows these instantly and can re-run them live.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fastapi.testclient import TestClient  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.main import create_app  # noqa: E402
from tools.commons import thumbnail  # noqa: E402

OUT = ROOT / "demo" / "samples"
SAMPLES = [
    {
        "id": "easy",
        "label": "Easy",
        "title": "Boxed snack cakes, face-on",
        "look_at": "Face-on boxes with clear outlines: nearly every pack is found, and runs "
        "of the same box (Honey Buns, Marshmallow Pies) mostly share a colour.",
        "commons": "File:Grocery store shelves of junk food.jpg",
        "author": "Cory Doctorow",
        "licence": "CC BY-SA 2.0",
        "licence_url": "https://creativecommons.org/licenses/by-sa/2.0",
    },
    {
        "id": "crowded",
        "label": "Crowded",
        "title": "A full aisle of soup cans",
        "look_at": "About 300 products in one photo (about 45 s on a laptop CPU, as every crop "
        "is embedded). Small, distant cans at the far end are where detection thins out.",
        "commons": "File:Faced products on a supermarket shelf.JPG",
        "author": "Amnesiac86",
        "licence": "CC BY-SA 3.0",
        "licence_url": "https://creativecommons.org/licenses/by-sa/3.0",
    },
    {
        "id": "similar",
        "label": "Similar packaging",
        "title": "Honey and syrup in lookalike bottles",
        "look_at": "Different products in the same bottle shape end up in one group: six "
        "Dutch Gold honey varieties, and syrups from two brands. Click one to see why.",
        "commons": "File:Honey selection at supermarket.jpeg",
        "author": "culinary123",
        "licence": "CC BY-SA 3.0",
        "licence_url": "https://creativecommons.org/licenses/by-sa/3.0",
    },
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    settings = get_settings().model_copy(update={"output_dir": Path(tempfile.mkdtemp())})
    with TestClient(create_app(settings)) as client:
        while client.get("/ready").status_code != 200:
            time.sleep(0.5)
        manifest = []
        for s in SAMPLES:
            data = thumbnail(s["commons"])
            (OUT / f"{s['id']}.jpg").write_bytes(data)
            r = client.post(
                "/v1/infer",
                params={"explain": "true"},
                files={"file": (f"{s['id']}.jpg", data, "image/jpeg")},
            )
            r.raise_for_status()
            body = r.json()
            (OUT / f"{s['id']}-annotated.jpg").write_bytes(
                client.get(body["annotated_image_url"]).content
            )
            body["annotated_image_url"] = f"samples/{s['id']}-annotated.jpg"
            body["request_id"] = "precomputed"
            (OUT / f"{s['id']}.json").write_text(json.dumps(body, indent=1))
            page = "https://commons.wikimedia.org/wiki/" + s["commons"].replace(" ", "_")
            manifest.append(
                {
                    **{k: s[k] for k in ("id", "label", "title", "look_at")},
                    "image": f"samples/{s['id']}.jpg",
                    "result": f"samples/{s['id']}.json",
                    "credit": {
                        "title": s["commons"].removeprefix("File:"),
                        "page": page,
                        "author": s["author"],
                        "licence": s["licence"],
                        "licence_url": s["licence_url"],
                    },
                    "products": body["products"],
                    "groups": len(body["groups"]),
                    "total_s": round(body["timings"]["total_ms"] / 1000, 1),
                }
            )
            print(
                f"{s['id']}: {body['products']} products, {len(body['groups'])} groups, "
                f"{body['timings']['total_ms'] / 1000:.1f} s"
            )
            time.sleep(4)  # Commons rate limits
        (OUT / "samples.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
