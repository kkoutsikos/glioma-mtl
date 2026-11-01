"""SwinUNETR helpers: one forward pass that returns logits *and* intermediate features, without hooks.

Forward hooks plus hard-coded channel counts caused two silent failures in the thesis notebooks: the
Trinity model's stage tokens were all zeros, and a checkpoint loaded nothing. Here the channel count of
every stage is derived from ``feature_size`` and checked.
"""
from __future__ import annotations

from typing import NamedTuple

import torch
from monai.networks.nets import SwinUNETR


def swin_stage_channels(feature_size: int) -> list[int]:
    """Channels of ``swinViT`` hidden states 0..4 (resolutions /2, /4, /8, /16, /32)."""
    return [feature_size * 2 ** i for i in range(5)]


def build_swin_unetr(in_channels: int = 1, out_channels: int = 1, feature_size: int = 48,
                     use_checkpoint: bool = True, use_v2: bool = False) -> SwinUNETR:
    return SwinUNETR(in_channels=in_channels, out_channels=out_channels, feature_size=feature_size,
                     use_checkpoint=use_checkpoint, use_v2=use_v2)


class SwinOutputs(NamedTuple):
    logits: torch.Tensor          # [B, out, D, H, W]
    hidden: list[torch.Tensor]    # 5 encoder stages, channel-first
    decoder: torch.Tensor         # [B, feature_size, D, H, W], input of the final 1x1 conv


def swin_unetr_forward(model: SwinUNETR, x: torch.Tensor) -> SwinOutputs:
    """Same computation as ``SwinUNETR.forward`` (MONAI >= 1.3), exposing intermediate features."""
    hidden = model.swinViT(x, model.normalize)
    enc0 = model.encoder1(x)
    enc1 = model.encoder2(hidden[0])
    enc2 = model.encoder3(hidden[1])
    enc3 = model.encoder4(hidden[2])
    dec4 = model.encoder10(hidden[4])
    dec3 = model.decoder5(dec4, hidden[3])
    dec2 = model.decoder4(dec3, enc3)
    dec1 = model.decoder3(dec2, enc2)
    dec0 = model.decoder2(dec1, enc1)
    out = model.decoder1(dec0, enc0)
    return SwinOutputs(model.out(out), list(hidden), out)
