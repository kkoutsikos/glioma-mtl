"""Shared training utilities: optimiser/scheduler factories, AMP step, best-model tracking, cost logging."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Iterable, Mapping

import torch

from ..utils import LOG, save_checkpoint


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    params = list(model.parameters())
    return {"params": sum(p.numel() for p in params),
            "trainable_params": sum(p.numel() for p in params if p.requires_grad)}


class EpochCost:
    """Compute cost for the paper: wall time and peak GPU memory of each epoch (training + validation).

    The parameter count and GPU are written once to ``model_info.json``. Each epoch row also records the
    GPU, so a run that resumed on a different GPU type is visible in ``history.csv``.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device, out_dir: str | Path):
        self.device = device
        self.gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        info = {**count_parameters(model), "gpu": self.gpu}
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "model_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        LOG.info("%s: %.2fM parameters (%.2fM trainable) on %s", model.__class__.__name__,
                 info["params"] / 1e6, info["trainable_params"] / 1e6, self.gpu)
        self._start = time.perf_counter()

    def start(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self._start = time.perf_counter()

    def stop(self) -> dict[str, float | str]:
        out: dict[str, float | str] = {"epoch_seconds": time.perf_counter() - self._start}
        if self.device.type == "cuda":
            out["peak_mem_gb"] = torch.cuda.max_memory_allocated(self.device) / 1024 ** 3
        out["gpu"] = self.gpu
        return out


def make_optimizer(param_groups: Iterable, cfg: Mapping) -> torch.optim.Optimizer:
    return torch.optim.AdamW(param_groups, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-5))


def make_scheduler(optimizer: torch.optim.Optimizer, cfg: Mapping, epochs: int):
    """Linear warm-up then cosine decay to ``min_lr_ratio * lr``, stepped once per epoch."""
    warmup = cfg.get("warmup_epochs", 0)
    floor = cfg.get("min_lr_ratio", 0.01)

    def factor(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(epochs - warmup, 1)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class AmpStepper:
    """Mixed-precision backward with gradient accumulation and clipping."""

    def __init__(self, optimizer: torch.optim.Optimizer, params, device: torch.device, amp: bool = True,
                 accumulation: int = 1, clip_norm: float | None = 1.0):
        self.optimizer, self.params = optimizer, list(params)
        self.enabled = amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.enabled)
        self.accumulation, self.clip_norm = max(accumulation, 1), clip_norm
        self._count = 0

    def backward(self, loss: torch.Tensor) -> None:
        self.scaler.scale(loss / self.accumulation).backward()
        self._count += 1
        if self._count % self.accumulation == 0:
            self.step()

    def step(self) -> None:
        if self.clip_norm:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.params, self.clip_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

    def flush(self) -> None:
        """Apply leftover accumulated gradients at the end of an epoch."""
        if self._count % self.accumulation:
            self.step()
        self._count = 0


class BestTracker:
    """Track the best value of a (optionally EMA-smoothed) validation score, with early stopping.

    Smoothing reduces the optimism of picking the single best of many noisy epochs on a small
    validation set.
    """

    def __init__(self, mode: str = "max", ema: float = 0.0, patience: int | None = None):
        self.sign = 1.0 if mode == "max" else -1.0
        self.ema, self.patience = ema, patience
        self.smoothed: float | None = None
        self.best = -math.inf
        self.best_epoch = -1
        self.bad_epochs = 0

    def update(self, value: float, epoch: int) -> bool:
        self.smoothed = value if self.smoothed is None or not self.ema else \
            self.ema * self.smoothed + (1 - self.ema) * value
        score = self.sign * self.smoothed
        if score > self.best:
            self.best, self.best_epoch, self.bad_epochs = score, epoch, 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.patience is not None and self.bad_epochs >= self.patience


def save_last(path: str | Path, model, optimizer, scheduler, stepper: AmpStepper, tracker: BestTracker,
              epoch: int, **extra) -> None:
    """Everything needed to continue a run after a crash or a Colab disconnect."""
    save_checkpoint(path, model, optimizer=optimizer, scheduler=scheduler, scaler=stepper.scaler,
                    epoch=epoch, tracker=dict(tracker.__dict__), **extra)


def try_resume(path: str | Path, model, optimizer, scheduler, stepper: AmpStepper, tracker: BestTracker,
               enabled: bool = True) -> int:
    """Restore a checkpoint written by :func:`save_last`; returns the index of the next epoch (0 = fresh)."""
    path = Path(path)
    if not enabled or not path.exists():
        return 0
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as err:  # truncated by a session that died mid-write (before saves were atomic)
        raise RuntimeError(f"{path} is incomplete or damaged and cannot be resumed. Delete its run folder "
                           f"{path.parent} to start this run from scratch.\n{err}") from err
    if "optimizer" not in state:
        LOG.warning("%s has no optimizer state; starting from scratch", path)
        return 0
    try:
        model.load_state_dict(state["model_state"])
    except RuntimeError as err:
        raise RuntimeError(f"{path} was written by a different model version and cannot be resumed. Delete "
                           f"its run folder {path.parent} to start this run from scratch.\n{err}") from err
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    stepper.scaler.load_state_dict(state["scaler"])
    tracker.__dict__.update(state["tracker"])
    LOG.info("Resumed from %s: continuing with epoch %d", path, state["epoch"] + 2)
    return state["epoch"] + 1
