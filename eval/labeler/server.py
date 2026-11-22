"""Grouping labeller for experiment S4 (local tool; nothing here is part of the service).

    uv run --group eval python eval/labeler/server.py <test-shard.parquet>
    → open http://127.0.0.1:8002

Shows each selected SKU-110K test image with its ground-truth boxes; you group the boxes
into products, tag each product's brand, and mark boxes you can't judge as unsure.
Labels go to eval/labels/<image>.json (git-ignored: the dataset's licence forbids
redistribution). No model output is shown, so the labels can't be anchored on it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
LABELS = EVAL / "labels"


class BoxLabel(BaseModel):
    id: int
    product: int | None = Field(description="product group id, null if unsure")
    unsure: bool = False


class ImageLabels(BaseModel):
    boxes: list[BoxLabel]
    brands: dict[str, str]  # product group id -> brand name
    seconds: float = 0
    note: str = ""


def create_app(parquet: Path) -> FastAPI:
    selection = json.loads((EVAL / "selection.json").read_text())
    order = selection["tune"] + selection["test"]
    split = {n: "tune" for n in selection["tune"]} | {n: "test" for n in selection["test"]}
    table = pq.read_table(parquet).to_pylist()
    rows = {r["image"]["path"]: r for r in table if r["image"]["path"] in split}
    missing = set(order) - set(rows)
    if missing:
        raise SystemExit(f"images not in {parquet}: {sorted(missing)}")
    LABELS.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="ShelfScan grouping labeller")

    def label_path(name: str) -> Path:
        if name not in rows:
            raise HTTPException(404, "unknown image")
        return LABELS / f"{name}.json"

    def boxes_of(name: str) -> tuple[list[list[float]], int, int]:
        from io import BytesIO

        from PIL import Image

        r = rows[name]
        w, h = Image.open(BytesIO(r["image"]["bytes"])).size
        boxes = [
            [max(0.0, x), max(0.0, y), min(float(w), x + bw), min(float(h), y + bh)]
            for x, y, bw, bh in r["objects"]["bbox"]
        ]
        return boxes, w, h

    @app.get("/")
    def index():
        return FileResponse(HERE / "index.html")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return Response(status_code=204)

    @app.get("/api/images")
    def images():
        out = []
        for n in order:
            p = LABELS / f"{n}.json"
            saved = json.loads(p.read_text()) if p.exists() else None
            out.append(
                {
                    "name": n,
                    "split": split[n],
                    "boxes": len(rows[n]["objects"]["bbox"]),
                    "done": saved is not None,
                    "seconds": saved["seconds"] if saved else 0,
                }
            )
        return out

    @app.get("/api/images/{name}")
    def image(name: str):
        p = label_path(name)
        boxes, w, h = boxes_of(name)
        return {
            "name": name,
            "width": w,
            "height": h,
            "boxes": boxes,
            "labels": json.loads(p.read_text()) if p.exists() else None,
        }

    @app.get("/api/images/{name}/jpeg")
    def jpeg(name: str):
        label_path(name)
        return Response(rows[name]["image"]["bytes"], media_type="image/jpeg")

    @app.get("/api/brands")
    def brands():
        found = set()
        for p in LABELS.glob("*.json"):
            found |= set(json.loads(p.read_text())["brands"].values())
        return sorted(found, key=str.lower)

    @app.post("/api/images/{name}/labels")
    def save(name: str, labels: ImageLabels):
        p = label_path(name)
        boxes, _, _ = boxes_of(name)
        ids = sorted(b.id for b in labels.boxes)
        if ids != list(range(len(boxes))):
            raise HTTPException(422, "every box must be labelled exactly once")
        used = {str(b.product) for b in labels.boxes if b.product is not None}
        if any(b.product is None and not b.unsure for b in labels.boxes):
            raise HTTPException(422, "some boxes are neither in a product nor unsure")
        missing = sorted(used - {k for k, v in labels.brands.items() if v.strip()})
        if missing:
            raise HTTPException(422, f"products without a brand: {missing}")
        data = labels.model_dump()
        data["brands"] = {k: v.strip() for k, v in labels.brands.items() if k in used}
        data["image"] = name
        data["split"] = split[name]
        p.write_text(json.dumps(data, indent=1))
        return {"saved": name}

    return app


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    uvicorn.run(create_app(Path(sys.argv[1])), host="127.0.0.1", port=8002)
