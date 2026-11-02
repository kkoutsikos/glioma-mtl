"""Evaluate a segmentation checkpoint on one split; writes per-case metrics and a CI summary.

    python scripts/evaluate_segmentation.py --config configs/seg_segresnet.yaml --checkpoint runs/seg_segresnet/best.pt
    # predicted masks for test-time classification crops (classification val/test patients):
    python scripts/evaluate_segmentation.py --config configs/seg_swin_unetr_ssl.yaml \
        --checkpoint runs/seg_swin_unetr_ssl/best.pt --stage classification --split test --save-predictions data/predictions/swin \
        --largest-component

``--largest-component`` keeps only the largest connected component of each predicted mask, so a stray
false-positive blob cannot pull the classification crop off the tumour or enter the radiomics. The
segmentation benchmark (``--stage segmentation``) is reported on the raw predictions.
"""
from pathlib import Path

import torch
import nibabel as nib
import numpy as np
from monai.data import decollate_batch
from monai.transforms import Invertd

from glioma_mtl.cli import start_run
from glioma_mtl.data.loaders import make_dataset, make_loader
from glioma_mtl.data.transforms import seg_eval_transforms
from glioma_mtl.evaluation import evaluate_segmentation, summarize_segmentation
from glioma_mtl.experiment import load_context, stage_records
from glioma_mtl.models import build_segmentation_model
from glioma_mtl.utils import LOG, load_weights


def _args(parser):
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stage", default="segmentation", choices=["segmentation", "classification"])
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--save-predictions", default=None, help="folder for <patient_id>.nii.gz masks")
    parser.add_argument("--largest-component", action="store_true",
                        help="keep only the largest connected component of each prediction (metrics and saved masks)")
    parser.add_argument("--out-dir", default=None)


def main() -> None:
    cfg, out_dir, device, args = start_run(__doc__, _args)
    ctx = load_context(cfg)
    d = cfg["data"]
    transform = seg_eval_transforms(d)
    loader = make_loader(make_dataset(stage_records(ctx, args.stage, args.split), transform), d, train=False)

    model = build_segmentation_model(cfg["model"], in_channels=len(d.get("modalities", ["T2"])))
    load_weights(model, args.checkpoint, min_fraction=1.0)
    model.to(device)

    on_prediction = None
    if args.save_predictions:
        save_dir = Path(args.save_predictions)
        save_dir.mkdir(parents=True, exist_ok=True)
        # Undo Orientation/Spacing so masks align with the original NIfTI grid.
        invert = Invertd(keys="pred", transform=transform, orig_keys="image", nearest_interp=True)

        def on_prediction(batch, pred):
            item = invert(decollate_batch({"image": batch["image"], "pred": pred.cpu()})[0])
            mask = (item["pred"][0].numpy() > 0.5).astype(np.uint8)
            affine = np.asarray(item["pred"].affine, dtype=float)
            nib.save(nib.Nifti1Image(mask, affine), save_dir / f"{batch['patient_id'][0]}.nii.gz")

    per_case = evaluate_segmentation(model, loader, d["roi_size"], device, on_prediction=on_prediction,
                                     largest_component=args.largest_component)
    if args.largest_component:
        cleaned = per_case[per_case["components"] > 1]
        LOG.info("Largest component kept: %d/%d cases had extra components, %.1f mL removed in total (max %.1f mL)",
                 len(cleaned), len(per_case), cleaned["removed_ml"].sum(), per_case["removed_ml"].max())
    tag = f"{args.stage}_{args.split}"
    per_case.to_csv(out_dir / f"per_case_{tag}.csv", index=False)
    summary = summarize_segmentation(per_case)
    summary.to_csv(out_dir / f"summary_{tag}.csv", index=False)
    LOG.info("\n%s", summary.to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    with torch.inference_mode():
        main()
