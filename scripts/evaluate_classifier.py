"""Evaluate a trained classifier **once** on the held-out classification test split.

    python scripts/evaluate_classifier.py --config runs/cls_mask_guided/config.yaml --checkpoint runs/cls_mask_guided/best.pt

The split is chosen explicitly (default ``test``) and printed with its size, so a training loader
cannot be mistaken for the test set. Uses the featurizers fitted at training time.
"""
import torch  # noqa: F401  (load before pandas; see src/glioma_mtl/__init__.py)
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from glioma_mtl.cli import start_run
from glioma_mtl.data.loaders import make_dataset, make_loader
from glioma_mtl.data.metadata import TASKS
from glioma_mtl.data.transforms import cls_eval_transforms
from glioma_mtl.engine import predict, task_arrays
from glioma_mtl.evaluation import classification_metrics
from glioma_mtl.experiment import build_classifier, classification_data, load_context
from glioma_mtl.utils import LOG, load_weights


def _args(parser):
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--out-dir", default=None)


def main() -> None:
    cfg, out_dir, device, args = start_run(__doc__, _args)
    with open(Path(args.checkpoint).parent / "featurizers.pkl", "rb") as f:
        featurizers = pickle.load(f)
    data = classification_data(cfg, load_context(cfg), featurizers=featurizers)
    records = data.records[args.split]
    LOG.info("Evaluating on classification/%s: %d patients", args.split, len(records))

    d = cfg["data"]
    loader = make_loader(make_dataset(records, cls_eval_transforms(d, d.get("crop_mask_key", "label"))), d, train=False)
    model = build_classifier(cfg, data)
    load_weights(model, args.checkpoint, min_fraction=1.0)
    model.to(device)

    preds, _ = predict(model, loader, device, mask_key=d.get("model_mask_key", "label"))
    preds.to_csv(out_dir / f"predictions_{args.split}.csv", index=False)
    tables = []
    for task in data.tasks:
        y, prob = task_arrays(preds, task)
        table, cm = classification_metrics(y, prob, [str(c) for c in TASKS[task][1]], seed=cfg.get("seed", 0))
        table.insert(0, "task", task)
        tables.append(table)
        LOG.info("%s confusion matrix (rows = truth):\n%s", task, np.array2string(cm))
    summary = pd.concat(tables)
    summary.to_csv(out_dir / f"metrics_{args.split}.csv", index=False)
    LOG.info("\n%s", summary.to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
