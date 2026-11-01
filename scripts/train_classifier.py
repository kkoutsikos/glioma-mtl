"""Train the multi-task classifier (grade / diagnosis / IDH) on the classification split.

    python scripts/train_classifier.py --config configs/cls_mask_guided.yaml
    python scripts/train_classifier.py --config configs/cls_image_only.yaml
"""
from glioma_mtl.cli import start_run
from glioma_mtl.data.loaders import make_dataset, make_loader
from glioma_mtl.data.transforms import cls_eval_transforms, cls_train_transforms
from glioma_mtl.engine import train_classifier
from glioma_mtl.experiment import build_classification_loss, build_classifier, classification_data, load_context
from glioma_mtl.utils import LOG, load_weights


def main() -> None:
    cfg, out_dir, device, _ = start_run(__doc__)
    ctx = load_context(cfg)
    data = classification_data(cfg, ctx)
    data.save_featurizers(out_dir / "featurizers.pkl")
    LOG.info("tasks=%s tabular=%s class_weights=%s", data.tasks, data.tabular_dims,
             {t: w.round(2).tolist() for t, w in data.class_weights.items()})

    d = cfg["data"]
    train_ds = make_dataset(data.records["train"], cls_train_transforms(d), d["cache"], d.get("cache_dir"), "cls_train")
    val_ds = make_dataset(data.records["val"], cls_eval_transforms(d, d.get("crop_mask_key", "label")),
                          d["cache"], d.get("cache_dir"), "cls_val")

    model = build_classifier(cfg, data)
    if cfg["model"].get("backbone_checkpoint"):
        load_weights(model.backbone, cfg["model"]["backbone_checkpoint"], key="model_state")
    train_classifier(model, make_loader(train_ds, d, train=True), make_loader(val_ds, d, train=False),
                     build_classification_loss(cfg, data), cfg, out_dir, device)


if __name__ == "__main__":
    main()
