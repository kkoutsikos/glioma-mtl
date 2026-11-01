"""Explain a trained classifier on one split: fusion attention, token occlusion, 3D Grad-CAM per patient.

    python scripts/explain_classifier.py --config runs/cls_mask_guided_s42/config.yaml \
        --checkpoint runs/cls_mask_guided_s42/best.pt --split test --out-dir runs/cls_mask_guided_s42/explain

Writes ``attention.csv``, ``occlusion.csv`` (+ summaries and plots) and ``gradcam/<patient>_<task>_<map>.png``.
"""
import torch  # noqa: F401  (load before pandas; see src/glioma_mtl/__init__.py)
import pickle
from pathlib import Path

from glioma_mtl.cli import start_run
from glioma_mtl.data.loaders import make_dataset, make_loader
from glioma_mtl.data.metadata import TASKS
from glioma_mtl.data.transforms import cls_eval_transforms
from glioma_mtl.experiment import build_classifier, classification_data, load_context
from glioma_mtl.explain import explain_classifier
from glioma_mtl.utils import LOG, load_weights


def _args(parser):
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--no-gradcam", action="store_true", help="only attention + occlusion (fast)")
    parser.add_argument("--out-dir", default=None)


def main() -> None:
    cfg, out_dir, device, args = start_run(__doc__, _args)
    with open(Path(args.checkpoint).parent / "featurizers.pkl", "rb") as f:
        featurizers = pickle.load(f)
    data = classification_data(cfg, load_context(cfg), featurizers=featurizers)
    records = data.records[args.split]
    LOG.info("Explaining classification/%s: %d patients", args.split, len(records))

    d = cfg["data"]
    loader = make_loader(make_dataset(records, cls_eval_transforms(d, d.get("crop_mask_key", "label"))), d,
                         train=False, batch_size=1)
    model = build_classifier(cfg, data)
    load_weights(model, args.checkpoint, min_fraction=1.0)
    results = explain_classifier(model, loader, device, {t: [str(c) for c in TASKS[t][1]] for t in data.tasks},
                                 out_dir, mask_key=d.get("model_mask_key", "label"), gradcam=not args.no_gradcam)
    if "occlusion" in results:
        summary = results["occlusion"].groupby(["task", "token"])["importance"].mean().unstack()
        LOG.info("Mean drop in true-class probability when a token is removed:\n%s", summary.round(3).to_string())


if __name__ == "__main__":
    main()
