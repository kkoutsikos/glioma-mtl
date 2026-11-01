"""Multi-task glioma classifier on a SwinUNETR backbone.

One configurable module replaces the ~10 copy-pasted variants in ``classification_runs.ipynb``:

==================  ============================================================================
thesis variant      configuration
==================  ============================================================================
Cascade             ``fusion=concat, mask_mode=none, freeze_backbone=True, encoder_stages=[4]``
Image only          ``tabular_dims={}``
Transformer fusion  ``fusion=transformer, mask_mode=none``
Mask-guided         ``fusion=transformer, mask_mode=hard, encoder_stages=[4], use_decoder_token=True``
Multi-scale         ``encoder_stages=[2, 3, 4]``
V3 soft masking     ``mask_mode=soft``
Trinity             ``fusion=cross_attention, encoder_stages=[1, 2, 3]`` (here the stage tokens are
                    real features, not zeros)
==================  ============================================================================

Masked pooling is a per-sample **masked mean** (``sum(f*m) / sum(m)``). The notebooks used
``avg_pool(f * m)``, which divides by the whole crop and so mixes tumour size into every feature.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_swin_unetr, swin_stage_channels, swin_unetr_forward

MASK_MODES = ("soft", "hard", "input", "none")
FUSIONS = ("transformer", "cross_attention", "concat")


def masked_mean(feat: torch.Tensor, mask: torch.Tensor | None, eps: float = 1e-6) -> torch.Tensor:
    """[B, C, *S] features, [B, 1, *S'] mask -> [B, C]. Samples with an empty mask use the global mean."""
    global_mean = feat.flatten(2).mean(-1)
    if mask is None:
        return global_mean
    m = mask.float()
    if m.shape[2:] != feat.shape[2:]:
        # Downsampling: tumour fraction per cell. Upsampling (rare): nearest.
        m = (F.adaptive_avg_pool3d(m, feat.shape[2:]) if m.shape[2] >= feat.shape[2]
             else F.interpolate(m, size=feat.shape[2:], mode="nearest"))
    weight = m.flatten(2).sum(-1)                                     # [B, 1]
    pooled = (feat * m).flatten(2).sum(-1) / weight.clamp_min(eps)
    return torch.where(weight > eps, pooled, global_mean)


class MultiTaskGliomaNet(nn.Module):
    def __init__(self, tasks: Mapping[str, int], tabular_dims: Mapping[str, int] | None = None,
                 in_channels: int = 1, feature_size: int = 48, use_checkpoint: bool = True,
                 encoder_stages: Sequence[int] = (4,), use_decoder_token: bool = True,
                 mask_mode: str = "soft", fusion: str = "transformer", d_model: int = 128,
                 n_heads: int = 4, num_layers: int = 2, dropout: float = 0.3,
                 modality_dropout: float = 0.0, aux_heads: bool = True, freeze_backbone: bool = False):
        super().__init__()
        if mask_mode not in MASK_MODES:
            raise ValueError(f"mask_mode must be one of {MASK_MODES}")
        if fusion not in FUSIONS:
            raise ValueError(f"fusion must be one of {FUSIONS}")
        if not tasks:
            raise ValueError("at least one task is required")

        self.tasks = dict(tasks)
        self.tabular_dims = {k: v for k, v in (tabular_dims or {}).items() if v > 0}
        self.encoder_stages = list(encoder_stages)
        self.use_decoder_token = use_decoder_token
        self.mask_mode, self.fusion = mask_mode, fusion
        self.modality_dropout = modality_dropout
        self.freeze_backbone = freeze_backbone

        self.backbone = build_swin_unetr(in_channels, 1, feature_size, use_checkpoint)
        if freeze_backbone:
            self.backbone.requires_grad_(False)

        self.stage_channels = swin_stage_channels(feature_size)
        bad = [s for s in self.encoder_stages if not 0 <= s < len(self.stage_channels)]
        if bad:
            raise ValueError(f"encoder_stages {bad} out of range 0..4")

        def proj(c_in: int) -> nn.Module:
            return nn.Sequential(nn.Linear(c_in, d_model), nn.LayerNorm(d_model))

        self.image_proj = nn.ModuleList([proj(self.stage_channels[s]) for s in self.encoder_stages])
        if use_decoder_token:
            self.image_proj.append(proj(feature_size))
        self.tab_proj = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
            for name, dim in self.tabular_dims.items()})

        n_ctx = len(self.image_proj) + len(self.tab_proj)
        self.type_embed = nn.Parameter(torch.zeros(1, n_ctx, d_model))
        self.task_tokens = nn.Parameter(torch.randn(1, len(self.tasks), d_model) * 0.02)
        nn.init.trunc_normal_(self.type_embed, std=0.02)

        if fusion == "transformer":
            layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True, norm_first=True)
            self.fuser = nn.TransformerEncoder(layer, num_layers)
            head_in = d_model
        elif fusion == "cross_attention":
            layer = nn.TransformerDecoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True, norm_first=True)
            self.fuser = nn.TransformerDecoder(layer, num_layers)
            head_in = d_model
        else:
            self.fuser = nn.Sequential(nn.Linear(n_ctx * d_model, 2 * d_model), nn.LayerNorm(2 * d_model),
                                       nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d_model, d_model))
            head_in = d_model

        self.heads = nn.ModuleDict({
            t: nn.Sequential(nn.Linear(head_in, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, n))
            for t, n in self.tasks.items()})
        self.aux = nn.ModuleDict({
            t: nn.ModuleList([nn.Linear(d_model, n) for _ in self.image_proj]) for t, n in self.tasks.items()
        }) if aux_heads else None

    # ------------------------------------------------------------------------------------ helpers
    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def backbone_parameters(self):
        return self.backbone.parameters()

    def head_parameters(self):
        backbone_ids = {id(p) for p in self.backbone.parameters()}
        return (p for p in self.parameters() if id(p) not in backbone_ids)

    def _mask(self, seg_logits: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor | None:
        if self.mask_mode == "none":
            return None
        if self.mask_mode == "input":
            if mask is None:
                raise ValueError("mask_mode='input' needs a mask in forward()")
            return mask
        prob = torch.sigmoid(seg_logits.detach() if self.freeze_backbone else seg_logits)
        return (prob > 0.5).float() if self.mask_mode == "hard" else prob

    def _tabular_tokens(self, tabular: Mapping[str, torch.Tensor] | None) -> list[torch.Tensor]:
        tokens = []
        for name, proj in self.tab_proj.items():
            if tabular is None or name not in tabular:
                raise KeyError(f"model expects tabular input {name!r}")
            tok = proj(tabular[name].float())
            if self.training and self.modality_dropout > 0:
                keep = (torch.rand(tok.shape[0], 1, device=tok.device) >= self.modality_dropout).float()
                tok = tok * keep
            tokens.append(tok)
        return tokens

    # ------------------------------------------------------------------------------------ forward
    def forward(self, image: torch.Tensor, tabular: Mapping[str, torch.Tensor] | None = None,
                mask: torch.Tensor | None = None) -> dict:
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_backbone):
            feats = swin_unetr_forward(self.backbone, image)
        attn_mask = self._mask(feats.logits, mask)

        maps = [feats.hidden[s] for s in self.encoder_stages]
        if self.use_decoder_token:
            maps.append(feats.decoder)
        image_tokens = []
        for fmap, proj in zip(maps, self.image_proj):
            expected = proj[0].in_features
            if fmap.shape[1] != expected:  # fail loudly, never substitute zeros
                raise RuntimeError(f"feature map has {fmap.shape[1]} channels, projection expects {expected}")
            image_tokens.append(proj(masked_mean(fmap, attn_mask)))

        context = torch.stack(image_tokens + self._tabular_tokens(tabular), dim=1) + self.type_embed
        B = image.shape[0]
        queries = self.task_tokens.expand(B, -1, -1)

        if self.fusion == "transformer":
            fused = self.fuser(torch.cat([queries, context], dim=1))[:, : len(self.tasks)]
        elif self.fusion == "cross_attention":
            fused = self.fuser(tgt=queries, memory=context)
        else:
            fused = self.fuser(context.flatten(1)).unsqueeze(1).expand(-1, len(self.tasks), -1)

        out = {
            "seg_logits": feats.logits,
            "logits": {t: self.heads[t](fused[:, i]) for i, t in enumerate(self.tasks)},
            "embedding": fused.mean(1),
        }
        if self.aux is not None:
            out["aux"] = {t: [head(tok) for head, tok in zip(self.aux[t], image_tokens)] for t in self.tasks}
        return out
