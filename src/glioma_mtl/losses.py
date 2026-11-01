"""Losses for SSL pretraining and multi-task classification."""
from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import ContrastiveLoss, DiceCELoss

IGNORE = -1  # missing label


# ----------------------------------------------------------------------------------------- SSL
def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean absolute error over hidden voxels only (``mask == 1``)."""
    return ((pred - target).abs() * mask).sum() / mask.sum().clamp_min(1.0)


class SSLLoss(nn.Module):
    """Weighted sum of the four SSL objectives. A weight of 0 disables (and skips) a branch."""

    def __init__(self, weights: Mapping[str, float], temperature: float = 0.5):
        super().__init__()
        self.w = {k: float(weights.get(k, 0.0)) for k in ("mae", "rotation", "contrastive", "light")}
        self.contrastive = ContrastiveLoss(temperature=temperature)

    def forward(self, out: Mapping[str, torch.Tensor], target: torch.Tensor, mask: torch.Tensor,
                rot_labels: torch.Tensor | None) -> tuple[torch.Tensor, dict[str, float]]:
        parts: dict[str, torch.Tensor] = {"mae": masked_l1(out["recon"], target, mask),
                                          "light": F.smooth_l1_loss(out["light_recon"], target)}
        if "z1" in out and self.w["contrastive"] > 0:
            parts["contrastive"] = self.contrastive(out["z1"], out["z2"])
        if "rot_logits" in out and self.w["rotation"] > 0:
            parts["rotation"] = F.cross_entropy(out["rot_logits"], rot_labels)
        total = sum(self.w[k] * v for k, v in parts.items())
        logs = {k: v.item() for k, v in parts.items()}
        if "rot_logits" in out and rot_labels is not None:
            logs["rot_acc"] = (out["rot_logits"].argmax(1) == rot_labels).float().mean().item()
        logs["total"] = total.item()
        return total, logs


# ------------------------------------------------------------------------------ classification
class FocalLoss(nn.Module):
    """Multi-class focal loss with optional class weights; targets equal to ``IGNORE`` are skipped."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight, persistent=False)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target != IGNORE
        if not valid.any():
            return logits.sum() * 0.0
        logits, target = logits[valid], target[valid]
        log_p = F.log_softmax(logits, dim=1)
        log_pt = log_p.gather(1, target[:, None]).squeeze(1)
        loss = -((1 - log_pt.exp()) ** self.gamma) * log_pt
        if self.weight is not None:
            w = self.weight[target]
            return (loss * w).sum() / w.sum()
        return loss.mean()


class SupConLoss(nn.Module):
    """Supervised contrastive loss (Khosla et al., 2020) on L2-normalised embeddings."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.t = temperature

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        valid = labels != IGNORE
        emb, labels = F.normalize(emb[valid], dim=1), labels[valid]
        n = emb.shape[0]
        if n < 2:
            return emb.sum() * 0.0
        sim = emb @ emb.T / self.t
        self_mask = torch.eye(n, dtype=torch.bool, device=emb.device)
        pos = (labels[:, None] == labels[None, :]) & ~self_mask
        log_prob = sim - torch.logsumexp(sim.masked_fill(self_mask, float("-inf")), dim=1, keepdim=True)
        has_pos = pos.any(1)
        if not has_pos.any():
            return emb.sum() * 0.0
        mean_log_pos = (log_prob * pos).sum(1)[has_pos] / pos.sum(1)[has_pos]
        return -mean_log_pos.mean()


class MultiTaskLoss(nn.Module):
    """``seg_weight * DiceCE + sum_t w_t * L_t + aux_weight * mean(aux) + supcon_weight * SupCon``."""

    def __init__(self, task_weights: Mapping[str, float], class_weights: Mapping[str, torch.Tensor] | None = None,
                 kind: str = "focal", gamma: float = 2.0, label_smoothing: float = 0.0,
                 seg_weight: float = 1.0, aux_weight: float = 0.3, supcon_weight: float = 0.0,
                 supcon_task: str | None = None):
        super().__init__()
        class_weights = class_weights or {}
        self.task_weights = dict(task_weights)
        if kind == "focal":
            self.cls = nn.ModuleDict({t: FocalLoss(gamma, class_weights.get(t)) for t in task_weights})
        elif kind == "ce":
            self.cls = nn.ModuleDict({t: nn.CrossEntropyLoss(weight=class_weights.get(t), ignore_index=IGNORE,
                                                             label_smoothing=label_smoothing)
                                      for t in task_weights})
        else:
            raise ValueError(f"loss kind {kind!r}")
        self.seg = DiceCELoss(sigmoid=True) if seg_weight > 0 else None
        self.seg_weight, self.aux_weight = seg_weight, aux_weight
        self.supcon = SupConLoss() if supcon_weight > 0 else None
        self.supcon_weight = supcon_weight
        self.supcon_task = supcon_task or next(iter(task_weights))

    def _cls(self, task: str, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if (y == IGNORE).all():  # nn.CrossEntropyLoss returns NaN for an all-ignored batch
            return logits.sum() * 0.0
        return self.cls[task](logits, y)

    def forward(self, out: Mapping, targets: Mapping[str, torch.Tensor],
                seg_target: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, float]]:
        parts: dict[str, torch.Tensor] = {}
        for t, w in self.task_weights.items():
            parts[t] = w * self._cls(t, out["logits"][t], targets[t])
            if self.aux_weight > 0 and "aux" in out:
                aux = torch.stack([self._cls(t, a, targets[t]) for a in out["aux"][t]]).mean()
                parts[f"aux_{t}"] = self.aux_weight * w * aux
        if self.seg is not None and seg_target is not None:
            parts["seg"] = self.seg_weight * self.seg(out["seg_logits"], seg_target)
        if self.supcon is not None:
            parts["supcon"] = self.supcon_weight * self.supcon(out["embedding"], targets[self.supcon_task])
        total = sum(parts.values())
        return total, {k: v.item() for k, v in parts.items()} | {"total": total.item()}
