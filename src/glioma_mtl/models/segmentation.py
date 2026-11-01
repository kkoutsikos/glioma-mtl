"""Segmentation model factory (binary whole-tumour masks)."""
from __future__ import annotations

import sys
from typing import Mapping

import torch.nn as nn
from monai.networks.nets import SegResNet, UNet

from ..utils import load_weights
from .backbone import build_swin_unetr


def build_segmentation_model(cfg: Mapping, in_channels: int = 1, out_channels: int = 1) -> nn.Module:
    name = cfg["name"]
    if name == "swin_unetr":
        return build_swin_unetr(in_channels, out_channels, cfg.get("feature_size", 48),
                                cfg.get("use_checkpoint", True), cfg.get("use_v2", False))
    if name == "segresnet":
        return SegResNet(spatial_dims=3, in_channels=in_channels, out_channels=out_channels,
                         init_filters=cfg.get("init_filters", 16), dropout_prob=cfg.get("dropout", 0.2))
    if name == "unet":
        return UNet(spatial_dims=3, in_channels=in_channels, out_channels=out_channels,
                    channels=tuple(cfg.get("channels", (32, 64, 128, 256, 320))),
                    strides=tuple(cfg.get("strides", (2, 2, 2, 2))),
                    num_res_units=cfg.get("num_res_units", 2), norm=cfg.get("norm", "instance"),
                    dropout=cfg.get("dropout", 0.0))
    if name == "segmamba":
        # SegMamba is not on PyPI: clone https://github.com/ge-xing/SegMamba and build its bundled
        # mamba / causal-conv1d (CUDA only), then set model.segmamba_repo to the clone.
        repo = cfg.get("segmamba_repo")
        if repo and repo not in sys.path:
            sys.path.append(repo)
        try:
            from model_segmamba.segmamba import SegMamba
        except ImportError as e:
            raise ImportError("SegMamba not importable; see the comment in build_segmentation_model") from e
        return SegMamba(in_chans=in_channels, out_chans=out_channels, depths=list(cfg.get("depths", [2, 2, 2, 2])),
                        feat_size=list(cfg.get("feat_size", [48, 96, 192, 384])),
                        hidden_size=cfg.get("hidden_size", 768))
    raise ValueError(f"Unknown segmentation model {name!r}")


def initialise(model: nn.Module, cfg: Mapping) -> None:
    """``init.checkpoint`` loads a full model; ``init.encoder`` loads SSL weights into ``model.swinViT``."""
    init = cfg.get("init") or {}
    if init.get("checkpoint"):
        load_weights(model, init["checkpoint"], key="model_state")
    if init.get("encoder"):
        if not hasattr(model, "swinViT"):
            raise ValueError("init.encoder is only supported for swin_unetr")
        load_weights(model.swinViT, init["encoder"], key="encoder_state")
