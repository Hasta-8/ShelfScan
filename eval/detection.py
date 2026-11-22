"""S3: detection quality on SKU-110K test images, by the standard COCO protocol.

    uv run --group eval python eval/detection.py <test-shard.parquet> [--limit N]

The detector is the deployed one (same preprocessing, NMS IoU 0.5). Ranking metrics
(AP) use every query's score (confidence threshold 0); precision/recall/F1 use the
deployed threshold (0.8). maxDets is 400, not COCO's default 100, because SKU-110K
shelves average ~150 products. Writes eval/results/s3_detection.json (metrics only:
the dataset's licence forbids redistributing images or annotations).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import DETECTOR_MODEL, DETECTOR_REVISION, get_settings  # noqa: E402
from services.detection.detector import Detector  # noqa: E402
from services.images import decode_upload  # noqa: E402


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a (N,4) and b (M,4), xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter)


def match_counts(dets: list[dict], gt: np.ndarray, min_conf: float, iou_thr: float = 0.5):
    """Greedy one-to-one matching, most confident detection first: (TP, FP, FN)."""
    kept = sorted((d for d in dets if d["confidence"] >= min_conf), key=lambda d: -d["confidence"])
    if not kept:
        return 0, 0, len(gt)
    ious = iou_matrix(np.array([d["bbox"] for d in kept], dtype=float), gt)
    used = np.zeros(len(gt), dtype=bool)
    tp = 0
    for row in ious:
        row = np.where(used, -1, row)
        j = int(row.argmax()) if len(row) else -1
        if j >= 0 and row[j] >= iou_thr:
            used[j] = True
            tp += 1
    return tp, len(kept) - tp, len(gt) - tp


def coco_ap(gt_by_image: dict, dets_by_image: dict, max_dets: list[int]) -> dict:
    images, anns, results = [], [], []
    aid = 1
    for i, (name, (gt, w, h)) in enumerate(sorted(gt_by_image.items()), start=1):
        images.append({"id": i, "file_name": name, "width": w, "height": h})
        for x1, y1, x2, y2 in gt:
            anns.append(
                {
                    "id": aid,
                    "image_id": i,
                    "category_id": 1,
                    "iscrowd": 0,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                }
            )
            aid += 1
        for d in dets_by_image[name]:
            x1, y1, x2, y2 = d["bbox"]
            results.append(
                {
                    "image_id": i,
                    "category_id": 1,
                    "score": d["confidence"],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                }
            )
    with contextlib.redirect_stdout(io.StringIO()):
        gt_coco = COCO()
        gt_coco.dataset = {
            "images": images,
            "annotations": anns,
            "categories": [{"id": 1, "name": "product"}],
        }
        gt_coco.createIndex()
        dt_coco = gt_coco.loadRes(results) if results else COCO()
        ev = COCOeval(gt_coco, dt_coco, "bbox")
        ev.params.maxDets = max_dets
        ev.evaluate()
        ev.accumulate()
    # Read the metrics from the accumulated arrays. COCOeval.summarize() hard-codes
    # maxDets=100 for its headline AP (stats[0]) whatever params.maxDets says, so its
    # "AP" can't be trusted for a 400 cap.
    prec = ev.eval["precision"]  # [IoU thresholds, recall points, classes, areas, maxDets]
    rec = ev.eval["recall"]  # [IoU thresholds, classes, areas, maxDets]
    m = len(max_dets) - 1  # the largest cap
    iou_thrs = list(np.round(ev.params.iouThrs, 2))

    def mean_valid(x):
        x = x[x > -1]
        return float(x.mean()) if x.size else float("nan")

    return {
        "AP": mean_valid(prec[:, :, 0, 0, m]),
        "AP50": mean_valid(prec[iou_thrs.index(0.5), :, 0, 0, m]),
        "AP75": mean_valid(prec[iou_thrs.index(0.75), :, 0, 0, m]),
        f"AR@{max_dets[m]}": mean_valid(rec[:, 0, 0, m]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--robustness-n", type=int, default=50, help="images re-encoded as JPEG q85")
    args = ap.parse_args()

    settings = get_settings()
    import torch

    torch.set_num_threads(settings.torch_threads)
    detector = Detector(DETECTOR_MODEL, DETECTOR_REVISION, settings.models_dir)
    rows = pq.read_table(args.parquet).to_pylist()
    rows.sort(key=lambda r: r["image"]["path"])
    if args.limit:
        rows = rows[: args.limit]

    def run(image_bytes: bytes):
        decoded = decode_upload(image_bytes, max_pixels=10**9, min_side_px=1)
        t0 = time.perf_counter()
        dets = detector.detect(decoded.image, confidence_threshold=0.0, nms_iou=settings.nms_iou)
        return decoded, dets, time.perf_counter() - t0

    gt_by_image, dets_by_image, per_image = {}, {}, []
    totals = {"tp": 0, "fp": 0, "fn": 0}
    for k, r in enumerate(rows, start=1):
        name = r["image"]["path"]
        decoded, dets, secs = run(r["image"]["bytes"])
        w, h = decoded.width, decoded.height
        gt = np.array(
            [
                [max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)]
                for x, y, bw, bh in r["objects"]["bbox"]
            ],
            dtype=float,
        )
        gt_by_image[name] = (gt.tolist(), w, h)
        dets_by_image[name] = dets
        tp, fp, fn = match_counts(dets, gt, settings.confidence_threshold)
        for key, v in zip(("tp", "fp", "fn"), (tp, fp, fn), strict=True):
            totals[key] += v
        per_image.append(
            {
                "image": name,
                "gt": len(gt),
                "kept_at_threshold": tp + fp,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "detect_s": round(secs, 2),
            }
        )
        if k % 25 == 0 or k == len(rows):
            print(f"{k}/{len(rows)} images", flush=True)

    p = totals["tp"] / max(1, totals["tp"] + totals["fp"])
    rcl = totals["tp"] / max(1, totals["tp"] + totals["fn"])
    secs = sorted(x["detect_s"] for x in per_image)
    result = {
        "experiment": "S3",
        "date": date.today().isoformat(),
        "data": {
            "source": "SKU-110K test, benjamintli/sku110k test-00000-of-00007.parquet",
            "images": len(rows),
            "gt_boxes": int(sum(x["gt"] for x in per_image)),
        },
        "model": f"{DETECTOR_MODEL}@{DETECTOR_REVISION[:7]}",
        "settings": {
            "nms_iou": settings.nms_iou,
            "torch_threads": settings.torch_threads,
            "operating_confidence": settings.confidence_threshold,
        },
        "coco_maxdets_400": coco_ap(gt_by_image, dets_by_image, [1, 100, 400]),
        "coco_maxdets_100": coco_ap(gt_by_image, dets_by_image, [1, 10, 100]),
        "at_operating_threshold": {
            **totals,
            "precision": p,
            "recall": rcl,
            "f1": 2 * p * rcl / max(1e-9, p + rcl),
        },
        "latency_s": {
            "median": secs[len(secs) // 2],
            "p95": secs[int(0.95 * (len(secs) - 1))],
            "max": secs[-1],
        },
        "per_image": per_image,
    }

    # Robustness (P11): the same images re-encoded as JPEG quality 85.
    if args.robustness_n:
        sub = rows[: args.robustness_n]
        gt_sub = {r["image"]["path"]: gt_by_image[r["image"]["path"]] for r in sub}
        orig = coco_ap(gt_sub, {n: dets_by_image[n] for n in gt_sub}, [1, 100, 400])
        reenc = {}
        for r in sub:
            buf = io.BytesIO()
            Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB").save(buf, "JPEG", quality=85)
            reenc[r["image"]["path"]] = run(buf.getvalue())[1]
        rec = coco_ap(gt_sub, reenc, [1, 100, 400])
        result["robustness_jpeg_q85"] = {
            "images": len(sub),
            "original": orig,
            "reencoded": rec,
            "delta_AP": rec["AP"] - orig["AP"],
        }

    out = Path(__file__).resolve().parent / "results" / "s3_detection.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    summary = {k: v for k, v in result.items() if k != "per_image"}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
