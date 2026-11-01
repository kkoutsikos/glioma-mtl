"""Self-supervised pretraining of the SwinUNETR encoder on SSL train/val patients (never test).

    python scripts/pretrain_ssl.py --config configs/ssl.yaml
"""
from glioma_mtl.cli import start_run
from glioma_mtl.data.loaders import make_dataset, make_loader
from glioma_mtl.data.transforms import ssl_transforms
from glioma_mtl.engine import pretrain_ssl
from glioma_mtl.experiment import load_context, stage_records
from glioma_mtl.models import HybridSSLPretrainer


def main() -> None:
    cfg, out_dir, device, _ = start_run(__doc__)
    ctx = load_context(cfg)
    d = cfg["data"]
    tf = ssl_transforms(d)
    loaders = {part: make_loader(make_dataset(stage_records(ctx, "ssl", part, require_label=False), tf,
                                              d["cache"], d.get("cache_dir"), f"ssl_{part}"),
                                 d, train=part == "train", batch_size=d["batch_size"])
               for part in ("train", "val")}
    m = cfg["model"]
    model = HybridSSLPretrainer(len(d.get("modalities", ["T2"])), m["feature_size"], m["projection_dim"],
                                m["mask_ratio"], m.get("mask_patch_size"), m.get("use_checkpoint", True))
    pretrain_ssl(model, loaders["train"], loaders["val"], cfg, out_dir, device)


if __name__ == "__main__":
    main()
