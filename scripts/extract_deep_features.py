"""Extract label-free tumour features from a trained segmentation SwinUNETR for the classification cohort.

    python scripts/extract_deep_features.py --config configs/cls_mask_guided.yaml \
        --checkpoint runs/seg_swin_unetr_ssl_s42/best.pt --mask-source pred --pred-dir runs/predictions/seg_swin_unetr_ssl_s42

``--mask-source pred`` centres the crop on, and pools inside, the *predicted* tumour (needs predicted masks
for all cohort patients, see scripts/evaluate_segmentation.py --save-predictions). ``label`` uses the
ground-truth mask (oracle localisation).
"""
from glioma_mtl.cli import start_run
from glioma_mtl.data.files import build_records
from glioma_mtl.data.loaders import make_dataset, make_loader
from glioma_mtl.data.transforms import cls_eval_transforms
from glioma_mtl.experiment import load_context
from glioma_mtl.features import extract_deep_features
from glioma_mtl.models import build_swin_unetr
from glioma_mtl.utils import LOG, load_weights


def _args(parser):
    parser.add_argument("--checkpoint", required=True, help="segmentation SwinUNETR best.pt")
    parser.add_argument("--mask-source", choices=["label", "pred"], default="pred")
    parser.add_argument("--pred-dir", default=None)
    parser.add_argument("--stages", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--out-dir", default="runs/deep_features")


def main() -> None:
    cfg, out_dir, device, args = start_run(__doc__, _args)
    ctx = load_context(cfg)
    ids = sorted(set().union(*ctx.splits["classification"].values()))
    if args.mask_source == "pred" and not args.pred_dir:
        raise SystemExit("--mask-source pred needs --pred-dir")
    records = build_records(ids, ctx.files, pred_mask_dir=args.pred_dir if args.mask_source == "pred" else None)

    d = cfg["data"]
    crop_key = "pred_mask" if args.mask_source == "pred" else "label"
    loader = make_loader(make_dataset(records, cls_eval_transforms(d, crop_key)), d, train=False)
    m = cfg["model"]
    model = build_swin_unetr(m.get("in_channels", 1), 1, m["feature_size"], use_checkpoint=False)
    load_weights(model, args.checkpoint, min_fraction=1.0)

    df = extract_deep_features(model, loader, device, args.stages, mask_source=args.mask_source)
    path = out_dir / f"deep_features_{args.mask_source}.csv"
    df.to_csv(path)
    LOG.info("Saved %d patients x %d features to %s", *df.shape, path)


if __name__ == "__main__":
    main()
