"""Hybrid self-supervised pretraining of the SwinUNETR encoder.

Branches (each can be switched off with its loss weight):

* masked reconstruction through the full SwinUNETR (MAE-style);
* rotation prediction (0/90/180/270 degrees in-plane);
* SimCLR-style contrastive loss between two augmented views;
* a lightweight decoder that reconstructs the input from the deepest features only.

Changes vs. the thesis: ``mask_patch_size`` masks **cubes** instead of independent voxels (voxel
masking is solved by interpolation and teaches little), and validation is run in ``eval`` mode with
a fixed random seed (the notebook validated in train mode with fresh random masks each epoch).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_swin_unetr, swin_stage_channels


class HybridSSLPretrainer(nn.Module):
    def __init__(self, in_channels: int = 1, feature_size: int = 48, projection_dim: int = 128,
                 mask_ratio: float = 0.6, mask_patch_size: int | None = 16, use_checkpoint: bool = True):
        super().__init__()
        self.backbone = build_swin_unetr(in_channels, in_channels, feature_size, use_checkpoint)
        self.mask_ratio = mask_ratio
        self.mask_patch_size = mask_patch_size

        deep = swin_stage_channels(feature_size)[-1]  # 16 * feature_size
        self.projector = nn.Sequential(nn.Linear(deep, deep // 4), nn.ReLU(inplace=True),
                                       nn.Linear(deep // 4, projection_dim))
        self.rot_head = nn.Linear(deep, 4)

        # /32 -> full resolution in five x2 upsampling steps
        chans = [deep, 8 * feature_size, 4 * feature_size, 2 * feature_size, feature_size, feature_size]
        layers: list[nn.Module] = []
        for c_in, c_out in zip(chans[:-1], chans[1:]):
            layers += [nn.Conv3d(c_in, c_out, 3, padding=1), nn.InstanceNorm3d(c_out), nn.LeakyReLU(inplace=True),
                       nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)]
        layers.append(nn.Conv3d(feature_size, in_channels, 1))
        self.light_decoder = nn.Sequential(*layers)

    # ------------------------------------------------------------------ input corruption helpers
    def random_mask(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(masked_input, mask)`` where ``mask == 1`` marks hidden voxels."""
        if self.mask_patch_size:
            p = self.mask_patch_size
            grid = [s // p for s in x.shape[2:]]
            coarse = (torch.rand(x.shape[0], 1, *grid, device=x.device) < self.mask_ratio).float()
            mask = F.interpolate(coarse, size=x.shape[2:], mode="nearest")
        else:
            mask = (torch.rand_like(x[:, :1]) < self.mask_ratio).float()
        return x * (1.0 - mask), mask

    @staticmethod
    def rotate(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k = torch.randint(0, 4, (x.shape[0],), device=x.device)
        rotated = torch.stack([torch.rot90(xi, int(ki), dims=(-2, -1)) for xi, ki in zip(x, k)])
        return rotated, k

    def _pooled(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        deepest = self.backbone.swinViT(x.contiguous(), self.backbone.normalize)[-1]
        return deepest, F.adaptive_avg_pool3d(deepest, 1).flatten(1)

    def forward(self, x_masked: torch.Tensor, x_con1: torch.Tensor, x_con2: torch.Tensor | None = None,
                x_rot: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        out = {"recon": self.backbone(x_masked.contiguous())}
        deep1, pooled1 = self._pooled(x_con1)
        out["light_recon"] = self.light_decoder(deep1)
        if x_con2 is not None:
            _, pooled2 = self._pooled(x_con2)
            out["z1"] = F.normalize(self.projector(pooled1), dim=1)
            out["z2"] = F.normalize(self.projector(pooled2), dim=1)
        if x_rot is not None:
            _, pooled_rot = self._pooled(x_rot)
            out["rot_logits"] = self.rot_head(pooled_rot)
        return out

    def encoder_state_dict(self) -> dict:
        return self.backbone.swinViT.state_dict()
