# ShelfScan

Detects the products in a retail shelf photo and groups the ones that look alike.
The output is JSON (boxes, confidences, group IDs) plus the photo annotated with one
colour per group. It runs on a laptop CPU.

**What "grouping" means here:** products whose packaging looks similar to a
self-supervised vision model end up in the same group. That is a visual-similarity
approximation of brand, **not** brand or SKU recognition: lookalike packages from
different brands can share a group, and variants of one product can be split.
There is no OCR and no product catalogue.

> **Status:** the pipeline works end to end. It has **not yet been evaluated**:
> there are no detection or grouping metrics yet, so nothing below claims accuracy.
> Hardening, evaluation, tests and a demo are in progress (see the status table).

## How it works

```
shelf image
  → detection: DETR ResNet-50 fine-tuned on SKU-110K (400 object queries)
      keep boxes with confidence ≥ 0.8, then IoU-based NMS (IoU > 0.5 removed)
  → grouping: crop each box (crops under 20 px wide or tall are skipped → group 0)
      DINOv2-small CLS embedding per crop, L2-normalised
      agglomerative clustering, cosine distance, average linkage,
      merge while similarity ≥ 0.7 (distance ≤ 0.3)
  → visualisation: one box colour per group
  → JSON response + annotated image
```

| Part | Model | Revision (pinned) | Licence |
|---|---|---|---|
| Detector | [`is36e/detr-resnet-50-sku110k`](https://huggingface.co/is36e/detr-resnet-50-sku110k) (originally published as `isalia99/…`, which now redirects) | `79e15d3` | Apache-2.0 |
| Embedder | [`facebook/dinov2-small`](https://huggingface.co/facebook/dinov2-small) | `ed25f3a` | Apache-2.0 |

The thresholds (0.8 confidence, 0.5 NMS IoU, 0.7 similarity) are the original
defaults. They have not been tuned or measured yet; a sensitivity study is planned.

## Status

| Capability | Status |
|---|---|
| Detection + NMS, grouping, annotated output, `POST /infer` (Flask) | IMPLEMENTED |
| Reproducible CPU-only setup: uv lock file, pinned model revisions, shared model folder | IMPLEMENTED |
| Lint and format checks in CI | IMPLEMENTED |
| Request/response schemas, upload validation, `/health` and `/ready`, models loaded at startup | PLANNED |
| Deterministic output (fixed thread count, stable group colours), EXIF orientation | PLANNED |
| Detection metrics (precision, recall, mAP) and grouping metrics (pairwise F1, ARI) on labelled images | PLANNED |
| Threshold sensitivity study, failure gallery | PLANNED |
| Unit and integration tests, Docker, demo | PLANNED |

## Quick start

Needs Python 3.12 and [uv](https://docs.astral.sh/uv/). Everything runs on the CPU;
no GPU or CUDA is used.

```bash
uv sync                                    # CPU-only torch, versions from uv.lock
export SHELFSCAN_MODELS_DIR=~/models/hf    # where model files go (default: .cache/models)
uv run python tools/download_models.py     # both models at their pinned revisions (~245 MB)
uv run python run.py                       # Flask on http://0.0.0.0:5000
```

```bash
curl -X POST http://127.0.0.1:5000/infer -F "file=@path/to/shelf.jpg"
```

The models load when the app's modules are imported, before the first request
(an explicit startup step with `/ready` is planned).

## API

`POST /infer`, multipart form with a `file` field (the shelf image).

```json
{
  "status": "success",
  "input_image": "data/uploads/<uuid>_shelf.jpg",
  "output_image": "data/outputs/annotated_<uuid>_shelf.jpg",
  "detections": [
    {"bbox": [x1, y1, x2, y2], "confidence": 0.92, "label": "product", "group_id": 2}
  ]
}
```

Boxes are integer pixel coordinates in the uploaded image, `[left, top, right, bottom]`.
`group_id` 0 means the box was too small to embed. `label` is always `"product"`
(SKU-110K has one class). Errors currently return `{"status": "error", "message": …}`
with 400 for a missing file; other failures are not yet handled (planned).

## Performance (measured)

On a 4-core Intel i5-10210U laptop CPU, 10 distinct sample shelf photos:

| | |
|---|---|
| Detection | 3.2–4.7 s per image, roughly constant |
| Grouping | about 0.17 s per detected product (every crop is embedded) |
| Total per image | 9 s (31 products) to 36–47 s (254 products) |
| Model loading | under 1 s from a warm disk cache |
| Memory (the Flask app) | 0.8 GB at rest, 2.3 GB peak on the 254-product image |

The range for the densest image comes from the thread count: 36 s through the
app with torch's default of 4 threads (one per core), 47 s with 8 threads, so the
CPU's extra hyperthreads don't help here.

## Reproducibility

The environment was rebuilt in November 2026: the original `requirements.txt`
installed the CUDA build of torch plus four libraries the code never imports (a
7.5 GB environment); the uv lock file installs the CPU build of the same versions
(1.1 GB). On all 12 sample images the new environment reproduces the old one
**exactly**: same boxes to the pixel, same confidences, same groups.

One caveat: results can differ by up to 1 pixel when torch uses a different number
of CPU threads (floating-point sums run in a different order). The thread count
will become a setting.

## Limitations

- **No evaluation yet:** detection and grouping quality are unmeasured.
- **Grouping looks over-split** on dense shelves: the busiest sample has 254
  products in 121 groups. Whether that is wrong depends on labels that don't exist
  yet.
- **Visual similarity is not identity:** lookalike packaging from different brands
  can be grouped together, and variants of one product (flavours, sizes) can be split.
- **EXIF orientation is ignored:** a phone photo stored sideways is analysed as
  stored, so boxes won't line up with the photo as it is displayed.
- **No input validation:** no checks on file type, size or pixel dimensions yet.
- **Latency grows with shelf density**, because every product crop is embedded.
- Price tags and shelf labels are ignored.

## Repository

```
app/          Flask app, settings (app/config.py), routes
services/     detection/, grouping/, visualization/
schemas/      response schema placeholder (typed schemas are planned)
tools/        download_models.py, create_grid.py (before/after image)
docs/original-submission/   the January 2026 write-up and JSON format notes
```

The two PDFs in `docs/original-submission/` describe the January 2026 version as
submitted. Their claims (for example "high recall on dense shelves") were not
measured; this README is the current description.

## Licence

The code is not yet licensed. The models are Apache-2.0.
