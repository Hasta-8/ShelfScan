"""S5 (detection): precision/recall/F1 over confidence and NMS IoU thresholds.

    uv run --group eval python eval/detection_sweep.py <test-shard.parquet> [--limit N]

One detector pass per image at the lowest confidence (0.3) with NMS off (IoU 1.0 keeps
every box), then NMS and the confidence cut are applied offline. Greedy NMS never lets
a lower-confidence box suppress a higher one, so "NMS then threshold" equals the
service's "threshold then NMS". Matching is S3's (greedy one-to-one, IoU 0.5). The grid
point (0.8, 0.5) must reproduce S3's counts. Writes eval/results/s5_detection.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from detection import iou_matrix, match_counts  # noqa: E402

from app.config import DETECTOR_MODEL, DETECTOR_REVISION, get_settings  # noqa: E402
from services.detection.detector import Detector  # noqa: E402
from services.images import decode_upload  # noqa: E402

CONFS = [round(0.3 + 0.05 * k, 2) for k in range(14)]  # 0.30 … 0.95
NMS_IOUS = [0.3, 0.4, 0.5, 0.6, 0.7]


def nms_fast(dets: list[dict], iou_thr: float) -> list[dict]:
    """Same result as services.detection.detector.nms (stable sort, keep if every IoU with
    a kept box is ≤ the threshold), vectorised."""
    dets = sorted(dets, key=lambda d: d["confidence"], reverse=True)
    if not dets:
        return []
    ious = iou_matrix(*(np.array([d["bbox"] for d in dets], dtype=float),) * 2)
    kept: list[int] = []
    for i in range(len(dets)):
        if not kept or (ious[i, kept] <= iou_thr).all():
            kept.append(i)
    return [dets[i] for i in kept]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    settings = get_settings()
    import torch

    torch.set_num_threads(settings.torch_threads)
    detector = Detector(DETECTOR_MODEL, DETECTOR_REVISION, settings.models_dir)
    rows = sorted(pq.read_table(args.parquet).to_pylist(), key=lambda r: r["image"]["path"])
    if args.limit:
        rows = rows[: args.limit]

    totals = {(c, n): [0, 0, 0] for c in CONFS for n in NMS_IOUS}
    for k, r in enumerate(rows, start=1):
        decoded = decode_upload(r["image"]["bytes"], max_pixels=10**9, min_side_px=1)
        w, h = decoded.width, decoded.height
        gt = np.array(
            [
                [max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)]
                for x, y, bw, bh in r["objects"]["bbox"]
            ],
            dtype=float,
        )
        raw = detector.detect(decoded.image, confidence_threshold=min(CONFS), nms_iou=1.0)
        for n in NMS_IOUS:
            kept = nms_fast(raw, n)
            for c in CONFS:
                for j, v in enumerate(match_counts(kept, gt, c)):
                    totals[(c, n)][j] += v
        if k % 25 == 0 or k == len(rows):
            print(f"{k}/{len(rows)} images", flush=True)

    grid = []
    for (c, n), (tp, fp, fn) in totals.items():
        p, rc = tp / max(1, tp + fp), tp / max(1, tp + fn)
        grid.append(
            {
                "confidence": c,
                "nms_iou": n,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": p,
                "recall": rc,
                "f1": 2 * p * rc / max(1e-9, p + rc),
            }
        )
    at = {(g["confidence"], g["nms_iou"]): g for g in grid}
    deployed = at[(settings.confidence_threshold, settings.nms_iou)]
    best = max(grid, key=lambda g: g["f1"])
    eligible = [
        g for g in grid if g["f1"] >= deployed["f1"] + 0.02 and g["recall"] >= deployed["recall"]
    ]
    result = {
        "experiment": "S5 (detection)",
        "date": date.today().isoformat(),
        "images": len(rows),
        "model": f"{DETECTOR_MODEL}@{DETECTOR_REVISION[:7]}",
        "deployed": deployed,
        "best_f1": best,
        "rule": "change only if F1 improves by >= 0.02 and recall doesn't drop",
        "eligible": sorted(eligible, key=lambda g: -g["f1"]),
        "adopt": bool(eligible),
        "grid": grid,
    }
    out = Path(__file__).resolve().parent / "results" / "s5_detection.json"
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k not in ("grid", "eligible")}, indent=1))
    print(f"{len(eligible)} eligible settings")
    print("conf  " + "  ".join(f"NMS {n}: P/R/F1        " for n in NMS_IOUS))
    for c in CONFS:
        cells = [at[(c, n)] for n in NMS_IOUS]
        print(
            f"{c:.2f}  "
            + "  ".join(f"{g['precision']:.3f}/{g['recall']:.3f}/{g['f1']:.3f}" for g in cells)
        )


if __name__ == "__main__":
    main()
