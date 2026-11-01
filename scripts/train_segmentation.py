"""Train a segmentation model on the shared split; selects the checkpoint on validation Dice.

    python scripts/train_segmentation.py --config configs/seg_segresnet.yaml
    python scripts/train_segmentation.py --config configs/seg_swin_unetr_ssl.yaml --set seed=1 name=seg_swin_ssl_s1
"""
from glioma_mtl.cli import start_run
from glioma_mtl.data.loaders import make_dataset, make_loader
from glioma_mtl.data.transforms import seg_eval_transforms, seg_train_transforms
from glioma_mtl.engine import train_segmentation
from glioma_mtl.experiment import load_context, stage_records
from glioma_mtl.models import build_segmentation_model, initialise


def main() -> None:
    cfg, out_dir, device, _ = start_run(__doc__)
    ctx = load_context(cfg)
    d = cfg["data"]
    train_ds = make_dataset(stage_records(ctx, "segmentation", "train"), seg_train_transforms(d),
                            d["cache"], d.get("cache_dir"), "seg_train")
    val_ds = make_dataset(stage_records(ctx, "segmentation", "val"), seg_eval_transforms(d),
                          d["cache"], d.get("cache_dir"), "seg_val")
    model = build_segmentation_model(cfg["model"], in_channels=len(d.get("modalities", ["T2"])))
    initialise(model, cfg["model"])
    train_segmentation(model, make_loader(train_ds, d, train=True), make_loader(val_ds, d, train=False),
                       cfg, out_dir, device)


if __name__ == "__main__":
    main()
