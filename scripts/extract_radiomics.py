"""Extract pyradiomics features for the classification cohort.

    python scripts/extract_radiomics.py --config configs/cls_mask_guided.yaml
    # realistic setting: features from *predicted* masks
    python scripts/extract_radiomics.py --config configs/cls_mask_guided.yaml --mask-source pred --pred-dir data/predictions/swin

GT masks give "oracle" radiomics; at deployment only predicted masks exist, so report both if you can.
Feature *selection* is not done here; it happens inside training / CV on the training data only.
"""
from pathlib import Path

from glioma_mtl.cli import start_run
from glioma_mtl.experiment import load_context
from glioma_mtl.radiomics import extract_radiomics
from glioma_mtl.utils import LOG


def _args(parser):
    parser.add_argument("--mask-source", choices=["label", "pred"], default="label")
    parser.add_argument("--pred-dir", default=None)
    parser.add_argument("--out-dir", default="runs/radiomics")


def main() -> None:
    cfg, out_dir, _, args = start_run(__doc__, _args)
    ctx = load_context(cfg)
    ids = sorted(set().union(*ctx.splits["classification"].values()))
    cases = []
    for pid in ids:
        if args.mask_source == "pred":
            if not args.pred_dir:
                raise SystemExit("--mask-source pred needs --pred-dir")
            mask = Path(args.pred_dir) / f"{pid}.nii.gz"
        else:
            mask = ctx.files[pid]["label"]
        cases.append((pid, ctx.files[pid]["image"][0], mask))
    df = extract_radiomics(cases)
    path = out_dir / f"radiomics_{args.mask_source}.csv"
    df.to_csv(path)
    LOG.info("Saved %d x %d features to %s", *df.shape, path)


if __name__ == "__main__":
    main()
