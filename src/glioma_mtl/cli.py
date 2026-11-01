"""Run setup shared by the scripts: parse config, make the output dir, log, seed."""
from __future__ import annotations

from pathlib import Path

import torch

from .config import base_argparser, load_config, save_config
from .utils import LOG, get_device, seed_everything, setup_logging


def start_run(description: str, extra_args=None) -> tuple[dict, Path, torch.device, object]:
    parser = base_argparser(description)
    if extra_args:
        extra_args(parser)
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    out_dir = Path(getattr(args, "out_dir", None) or Path(cfg.get("output_root", "runs")) / cfg["name"])
    setup_logging(out_dir)
    save_config(cfg, out_dir / "config.yaml")
    seed_everything(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "auto"))
    LOG.info("Run %s -> %s on %s", cfg["name"], out_dir, device)
    return cfg, out_dir, device, args
