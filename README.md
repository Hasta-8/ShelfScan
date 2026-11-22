# ShelfScan

Detects the products in a retail shelf photo and groups the ones that look alike.
The output is JSON (boxes, confidences, group IDs) plus the photo annotated with one
colour per group, served by a FastAPI service. It runs on a laptop CPU.

**What "grouping" means here:** products whose packaging looks similar to a
self-supervised vision model end up in the same group. That is a visual-similarity
approximation of brand, **not** brand or SKU recognition: lookalike packages from
different brands can share a group, and variants of one product can be split.
There is no OCR and no product catalogue.

> **Status:** the pipeline runs end to end behind a hardened FastAPI service, with
> tests in CI. It has **not yet been evaluated**: there are no detection or grouping
> metrics yet, so nothing below claims accuracy. Evaluation, Docker and a demo are
> next (see the status table).

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
| Detection + NMS, grouping, annotated output | IMPLEMENTED |
| FastAPI service: typed request/response schemas, one error shape, OpenAPI docs at `/docs` | IMPLEMENTED |
| Upload validation: format sniffed from the bytes (JPEG/PNG/WebP), size and pixel limits (also a decompression-bomb guard), corrupt files | IMPLEMENTED |
| Models loaded at startup; `/health` (process up) and `/ready` (models loaded) | IMPLEMENTED |
| Deterministic output: fixed torch thread count, group colours derived from the group ID | IMPLEMENTED |
| EXIF orientation: phone photos stored sideways are processed upright | IMPLEMENTED |
| Unit and API tests in CI (stand-in models), lint, lock-file check | IMPLEMENTED |
| Reproducible CPU-only setup: uv lock file, pinned model revisions, shared model folder | IMPLEMENTED |
| Detection metrics (precision, recall, mAP) and grouping metrics (pairwise F1, ARI) on labelled images | PLANNED |
| Threshold sensitivity study, failure gallery | PLANNED |
| Docker, demo, recorded fallback | PLANNED |
## Quick start

Needs Python 3.12 and [uv](https://docs.astral.sh/uv/). Everything runs on the CPU;
no GPU or CUDA is used.

```bash
uv sync                                    # CPU-only torch, versions from uv.lock
export SHELFSCAN_MODELS_DIR=~/models/hf    # where model files go (default: .cache/models)
uv run python tools/download_models.py     # both models at their pinned revisions (~245 MB)
uv run python run.py                       # http://127.0.0.1:8001, docs at /docs
```

```bash
curl http://127.0.0.1:8001/ready           # 503 while the models load, then 200
curl -F "file=@path/to/shelf.jpg" http://127.0.0.1:8001/v1/infer
```

The server binds to 127.0.0.1 by default (`run.py --host 0.0.0.0` to expose it).
Settings come from environment variables with the `SHELFSCAN_` prefix or a `.env`
file: thresholds, `TORCH_THREADS` (default 4), upload limits (`MAX_UPLOAD_MB` 15,
`MAX_PIXELS` 40 million, `MIN_SIDE_PX` 64). See `app/config.py`.

Tests need no models (the API tests use stand-ins): `uv run pytest`.
## API

| Method and path | Purpose |
|---|---|
| `POST /v1/infer` | Multipart form with a `file` field (the shelf photo). Returns the JSON below |
| `GET /v1/outputs/{name}` | The annotated JPEG from a previous call (`annotated_image_url`) |
| `GET /health`, `GET /ready` | Liveness; readiness (503 until both models are loaded) |
| `GET /docs` | Interactive OpenAPI documentation |
| `POST /infer` | The January path, kept as a deprecated alias of `/v1/infer` |

```json
{
  "request_id": "3f2a…",
  "image": {"format": "JPEG", "width": 1620, "height": 2880, "exif_orientation": 1},
  "detections": [
    {"bbox": [412, 96, 588, 410], "confidence": 0.97, "label": "product", "group_id": 2}
  ],
  "products": 31,
  "groups": [{"group_id": 2, "count": 9}, {"group_id": 1, "count": 6}],
  "annotated_image_url": "/v1/outputs/7c1e….jpg",
  "timings": {"decode_ms": 61, "detect_ms": 1990, "group_ms": 3098, "annotate_ms": 57, "total_ms": 5206},
  "models": {"detector": "is36e/detr-resnet-50-sku110k@79e15d3", "embedder": "facebook/dinov2-small@ed25f3a"},
  "settings": {"confidence_threshold": 0.8, "nms_iou": 0.5, "similarity_threshold": 0.7, "min_crop_px": 20, "torch_threads": 4}
}
```

- Boxes are integer pixels, `[left, top, right, bottom]`, in the **upright** image
  (after applying EXIF orientation), so they match the photo as it is displayed.
- `group_id` 0 means the box was too small to embed; `label` is always `"product"`
  (SKU-110K has one class). `groups` is sorted largest first.
- Uploads are decoded in memory and **not stored**; only the annotated image is
  written, under a random name.
- One image is processed at a time (CPU-bound, up to ~2.3 GB). A second request
  waits up to 60 s for its turn, then gets 503 `busy` with `Retry-After`.

**Errors** always have the same shape, `{"error", "detail", "request_id"}`, and never
include a stack trace or a file path:

| Status | `error` | When |
|---|---|---|
| 413 | `too_large`, `too_many_pixels` | Over the upload size or pixel limit |
| 415 | `unsupported_type` | Not an image, or not JPEG/PNG/WebP (checked from the bytes, not the file name) |
| 422 | `corrupt_image`, `too_small`, `empty_file`, `invalid_request` | Truncated or malformed file, a side under 64 px, empty upload, no `file` field |
| 503 | `not_ready`, `busy` | Models still loading; another image is being processed |
## Performance (measured)

On a 4-core Intel i5-10210U laptop CPU, 10 distinct sample shelf photos:

| | |
|---|---|
| Detection | 2–4 s per image, roughly constant |
| Grouping | about 0.13–0.15 s per detected product (every crop is embedded) |
| Total per image (default 4 threads) | 5–9 s (31–46 products), ~13 s (65–74), 18 s (99), 29 s (181), 35–40 s (254) |
| Startup (model loading) | about 5 s from a warm disk cache; `/ready` turns 200 then |
| Memory | 0.8 GB at rest, 2.3 GB peak on the 254-product image |

The default of 4 torch threads (the physical cores) is faster than 8 on this CPU:
the 254-product image takes 35–40 s at 4 threads and 45–47 s at 8.
## Reproducibility

The environment was rebuilt in November 2026: the original `requirements.txt`
installed the CUDA build of torch plus four libraries the code never imports (a
7.5 GB environment); the uv lock file installs the CPU build of the same versions
(1.1 GB). On all 12 sample images the new environment reproduces the old one
**exactly**: same boxes to the pixel, same confidences, same groups. The move to
FastAPI changed nothing either: the same 12 images give exactly the same output
through the new API.

- **Thread count:** results can differ by up to 1 pixel between thread counts
  (floating-point sums run in a different order), so the count is a fixed setting
  and reported in every response.
- **Same input, same output:** the same upload twice gives a byte-identical
  annotated image (group colours are derived from the group ID).
## Limitations

- **No evaluation yet:** detection and grouping quality are unmeasured.
- **Grouping looks over-split** on dense shelves: the busiest sample has 254
  products in 121 groups. Whether that is wrong depends on labels that don't exist
  yet.
- **Visual similarity is not identity:** lookalike packaging from different brands
  can be grouped together, and variants of one product (flavours, sizes) can be split.
- **Sensitive to re-compression:** re-saving the same photo as JPEG (quality 95)
  moved one box by up to 41 px and changed one grouping decision on a 31-product
  image, while a lossless copy gives identical results. Small pixel changes can
  flip borderline detections and groups.
- **Latency grows with shelf density**, because every product crop is embedded.
- Price tags and shelf labels are ignored.
## Repository

```
app/          FastAPI app (main.py), settings (config.py), the pipeline with its models (engine.py)
services/     detection/, grouping/, visualization/, images.py (upload decoding)
schemas/      API request/response models
tests/        unit tests and API tests (stand-in models, no downloads)
tools/        download_models.py, create_grid.py (before/after image)
docs/original-submission/   the January 2026 write-up and JSON format notes
```

The two PDFs in `docs/original-submission/` describe the January 2026 version as
submitted (a Flask app, a different response format). Their claims (for example
"high recall on dense shelves") were not measured; this README is the current
description.
## Licence

The code is under the [MIT licence](LICENSE). The models are Apache-2.0.
