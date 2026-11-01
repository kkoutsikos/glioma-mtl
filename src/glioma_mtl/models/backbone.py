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


class SegMambaFeatures:
    """Feature access for SegMamba (github.com/ge-xing/SegMamba) through forward hooks.

    SegMamba is third-party code whose forward we cannot safely re-implement, so its features are read
    from the modules the thesis used: ``vit`` (Mamba encoder, 4 stages at /2../16), ``encoder5`` (the
    ``hidden_size`` bottleneck) and the input of ``out`` (decoder features). Returned in the same
    layout as :func:`swin_unetr_forward`: hidden = [stage0, stage1, stage2, stage3, bottleneck]. Any
    missing module or unexpected output raises; channel counts are checked by the classifier.
    """

    MODULES = ("vit", "encoder5", "out")

    def __init__(self, model: torch.nn.Module):
        missing = [n for n in self.MODULES if not hasattr(model, n)]
        if missing:
            raise RuntimeError(f"SegMamba model has no {missing}; expected the ge-xing/SegMamba module layout")
        self.store: dict = {}
        model.vit.register_forward_hook(lambda m, i, o: self.store.__setitem__("vit", o))
        model.encoder5.register_forward_hook(lambda m, i, o: self.store.__setitem__("bottleneck", o))
        model.out.register_forward_hook(lambda m, i, o: self.store.__setitem__("decoder", i[0]))

    def __call__(self, model: torch.nn.Module, x: torch.Tensor) -> "SwinOutputs":
        self.store.clear()
        logits = model(x)
        stages = self.store.get("vit")
        if not isinstance(stages, (list, tuple)) or len(stages) < 4 or "bottleneck" not in self.store:
            raise RuntimeError(f"unexpected SegMamba encoder output: {type(stages).__name__} "
                               f"of length {len(stages) if isinstance(stages, (list, tuple)) else '-'}")
        return SwinOutputs(logits, [*stages[:4], self.store["bottleneck"]], self.store["decoder"])


def segmamba_stage_channels(feat_size, hidden_size: int = 768) -> list[int]:
    return [*list(feat_size)[:4], hidden_size]


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
