"""S4 + S5 (grouping): the deployed grouper on ground-truth boxes vs the user's labels.

    uv run --group eval python eval/grouping.py <test-shard.parquet>

Every ground-truth box of a labelled image is embedded and clustered exactly as the
service does (DINOv2-small CLS, average-linkage cosine, crops < min_crop_px left out as
group 0); boxes the labeller marked unsure are clustered too but not scored. Group 0
boxes count as singletons. Metrics are pairwise precision/recall/F1 pooled over images
(pairs of boxes in the same group), mean adjusted Rand index per image, and predicted
groups per true product. The similarity threshold is swept (S5); a new default is picked
on the tune split only and reported on the test split (`eval/selection.json`,
`labelled_resplit`). Writes eval/results/s4_grouping.json (metrics only: no images,
boxes or labels, per the dataset's licence).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import EMBEDDER_MODEL, EMBEDDER_REVISION, get_settings  # noqa: E402
from services.grouping.grouper import Grouper, cluster  # noqa: E402

EVAL = Path(__file__).resolve().parent
THRESHOLDS = [round(0.4 + 0.05 * k, 2) for k in range(12)]  # 0.40 … 0.95


def pair_counts(true: list[int], pred: list[int]) -> dict:
    """Pairs of items grouped together: by both, by the prediction, by the truth."""
    t, p = np.asarray(true), np.asarray(pred)
    if len(t) == 0:
        return {"both": 0, "pred": 0, "true": 0}
    _, ti = np.unique(t, return_inverse=True)
    _, pi = np.unique(p, return_inverse=True)
    table = np.zeros((ti.max() + 1, pi.max() + 1), dtype=np.int64)
    np.add.at(table, (ti, pi), 1)

    def pairs(x):
        return int((x * (x - 1) // 2).sum())

    return {"both": pairs(table), "pred": pairs(table.sum(0)), "true": pairs(table.sum(1))}


def prf(c: dict) -> dict:
    p = c["both"] / c["pred"] if c["pred"] else 1.0
    r = c["both"] / c["true"] if c["true"] else 1.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def singletonise(pred: np.ndarray) -> np.ndarray:
    """Group 0 (crop too small to embed) means "ungrouped": give each its own id."""
    out = pred.copy()
    zeros = np.where(out == 0)[0]
    out[zeros] = out.max(initial=0) + 1 + np.arange(len(zeros))
    return out


def score(items: list[dict], threshold: float) -> dict:
    """items: per image {truth (scored boxes), emb (all embeddable boxes), ...}."""
    total = {"both": 0, "pred": 0, "true": 0}
    aris, pred_groups, true_groups, boxes = [], 0, 0, 0
    for it in items:
        pred = np.zeros(it["n"], dtype=int)
        if len(it["valid"]):
            pred[it["valid"]] = cluster(it["emb"], threshold)
        pred = singletonise(pred)[it["scored"]]
        truth = it["truth"]
        c = pair_counts(truth, pred)
        for k in total:
            total[k] += c[k]
        aris.append(adjusted_rand_score(truth, pred))
        pred_groups += len(set(pred.tolist()))
        true_groups += len(set(truth))
        boxes += len(truth)
    return {
        "threshold": threshold,
        "images": len(items),
        "boxes": boxes,
        **prf(total),
        "pairs": total,
        "ari_mean": float(np.mean(aris)) if aris else float("nan"),
        "pred_groups": pred_groups,
        "true_products": true_groups,
        "groups_per_product": pred_groups / true_groups if true_groups else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    args = ap.parse_args()

    settings = get_settings()
    import torch

    torch.set_num_threads(settings.torch_threads)
    sel = json.loads((EVAL / "selection.json").read_text())["labelled_resplit"]
    split = {n: "tune" for n in sel["tune"]} | {n: "test" for n in sel["test"]}
    rows = {
        r["image"]["path"]: r
        for r in pq.read_table(args.parquet).to_pylist()
        if r["image"]["path"] in split
    }
    grouper = Grouper(EMBEDDER_MODEL, EMBEDDER_REVISION, settings.models_dir)

    items = []
    for name in sorted(split):
        labels = json.loads((EVAL / "labels" / f"{name}.json").read_text())
        r = rows[name]
        image = Image.open(BytesIO(r["image"]["bytes"])).convert("RGB")
        w, h = image.size
        boxes = [
            [max(0.0, x), max(0.0, y), min(float(w), x + bw), min(float(h), y + bh)]
            for x, y, bw, bh in r["objects"]["bbox"]
        ]
        by_id = {b["id"]: b for b in labels["boxes"]}
        assert sorted(by_id) == list(range(len(boxes))), name
        valid = [
            i
            for i, (x1, y1, x2, y2) in enumerate(boxes)
            if (x2 - x1) >= settings.min_crop_px and (y2 - y1) >= settings.min_crop_px
        ]
        t0 = time.perf_counter()
        emb = grouper.embed([image.crop(tuple(boxes[i])) for i in valid]) if valid else None
        scored = [i for i in range(len(boxes)) if not by_id[i]["unsure"]]
        items.append(
            {
                "name": name,
                "split": split[name],
                "n": len(boxes),
                "valid": np.array(valid, dtype=int),
                "emb": emb,
                "scored": np.array(scored, dtype=int),
                "truth": [by_id[i]["product"] for i in scored],
                "embed_s": time.perf_counter() - t0,
                "too_small": len(boxes) - len(valid),
            }
        )
        print(f"{name}: {len(boxes)} boxes embedded in {items[-1]['embed_s']:.1f} s", flush=True)

    def subset(s):
        return [it for it in items if it["split"] == s] if s else items

    sweep = {s or "all": [score(subset(s), t) for t in THRESHOLDS] for s in ("tune", "test", None)}
    deployed = settings.similarity_threshold
    best = max(sweep["tune"], key=lambda m: (m["f1"], -abs(m["threshold"] - deployed)))
    chosen = best["threshold"]
    test_at = {m["threshold"]: m for m in sweep["test"]}
    test_deployed = score(subset("test"), deployed)
    test_chosen = score(subset("test"), chosen)
    gain = test_chosen["f1"] - test_deployed["f1"]
    per_image = []
    for it in items:
        m = score([it], deployed)
        per_image.append(
            {
                "image": it["name"],
                "split": it["split"],
                "boxes_scored": m["boxes"],
                "too_small": it["too_small"],
                "true_products": m["true_products"],
                "pred_groups": m["pred_groups"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "ari": m["ari_mean"],
            }
        )

    result = {
        "experiments": ["S4", "S5 (grouping)"],
        "date": date.today().isoformat(),
        "data": {
            "images": len(items),
            "tune": sel["tune"],
            "test": sel["test"],
            "boxes_scored": sum(len(it["truth"]) for it in items),
            "unsure_excluded": sum(it["n"] - len(it["scored"]) for it in items),
        },
        "model": f"{EMBEDDER_MODEL}@{EMBEDDER_REVISION[:7]}",
        "settings": {"min_crop_px": settings.min_crop_px, "deployed_threshold": deployed},
        "s4_all_at_deployed": score(items, deployed),
        "s5_sweep": sweep,
        "s5_rule": {
            "chosen_on_tune": chosen,
            "tune_f1_at_chosen": best["f1"],
            "test_at_deployed": test_deployed,
            "test_at_chosen": test_chosen,
            "test_f1_gain": gain,
            "adopt": bool(gain >= 0.05),
        },
        "per_image_at_deployed": per_image,
    }
    assert test_at[round(deployed, 2)]["f1"] == test_deployed["f1"]
    out = EVAL / "results" / "s4_grouping.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("data", "s4_all_at_deployed", "s5_rule")}, indent=1))
    print("threshold  tune F1  test F1  all P   all R   all F1  groups/product")
    for a, b, c in zip(sweep["tune"], sweep["test"], sweep["all"], strict=True):
        print(
            f"  {a['threshold']:.2f}     {a['f1']:.3f}    {b['f1']:.3f}   {c['precision']:.3f}  "
            f"{c['recall']:.3f}  {c['f1']:.3f}   {c['groups_per_product']:.2f}"
        )


if __name__ == "__main__":
    main()
