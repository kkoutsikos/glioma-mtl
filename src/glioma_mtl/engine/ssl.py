"""SSL pretraining loop. Validation runs in eval mode with a fixed seed, so epochs are comparable."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Mapping

import torch
from tqdm import tqdm

from ..losses import SSLLoss
from ..models.ssl import HybridSSLPretrainer
from ..utils import LOG, CSVLogger, format_row, save_checkpoint
from .common import AmpStepper, BestTracker, make_optimizer, make_scheduler, save_last, try_resume


def _ssl_step(model: HybridSSLPretrainer, loss_fn: SSLLoss, batch: Mapping, device: torch.device,
              amp: bool) -> tuple[torch.Tensor, dict]:
    x_con1 = batch["view_con1"].to(device)
    x_con2 = batch["view_con2"].to(device)
    x_rot, k = model.rotate(batch["view_rot"].to(device))
    x_masked, mask = model.random_mask(x_con1)
    with torch.autocast(device.type, enabled=amp):
        out = model(x_masked, x_con1,
                    x_con2 if loss_fn.w["contrastive"] > 0 else None,
                    x_rot if loss_fn.w["rotation"] > 0 else None)
        return loss_fn(out, x_con1, mask, k)


@torch.no_grad()
def validate_ssl(model, loss_fn, loader, device, amp: bool, seed: int = 1234) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = defaultdict(float)
    n = 0
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(seed)  # identical masks / rotations every epoch
        for batch in loader:
            _, logs = _ssl_step(model, loss_fn, batch, device, amp)
            for key, value in logs.items():
                totals[key] += value
            n += 1
    return {f"val_{k}": v / max(n, 1) for k, v in totals.items()}


def pretrain_ssl(model: HybridSSLPretrainer, train_loader, val_loader, cfg: Mapping, out_dir: str | Path,
                 device: torch.device) -> Path:
    out_dir = Path(out_dir)
    tcfg = cfg["train"]
    epochs = tcfg["epochs"]
    model.to(device)
    loss_fn = SSLLoss(cfg["loss"]["weights"], cfg["loss"].get("temperature", 0.5)).to(device)
    optimizer = make_optimizer(model.parameters(), tcfg)
    scheduler = make_scheduler(optimizer, tcfg, epochs)
    stepper = AmpStepper(optimizer, model.parameters(), device, tcfg.get("amp", True),
                         tcfg.get("accumulation", 1), tcfg.get("clip_norm", 1.0))
    tracker = BestTracker("min", tcfg.get("selection_ema", 0.0), tcfg.get("patience"))
    history = CSVLogger(out_dir / "history.csv")
    last = out_dir / "last.pt"
    start_epoch = try_resume(last, model, optimizer, scheduler, stepper, tracker, tcfg.get("resume", True))

    for epoch in range(start_epoch, epochs):
        if tracker.should_stop:
            LOG.info("Early stopping at epoch %d (best epoch %d)", epoch + 1, tracker.best_epoch + 1)
            break
        model.train()
        totals: dict[str, float] = defaultdict(float)
        n = 0
        for batch in tqdm(train_loader, desc=f"ssl {epoch + 1}/{epochs}", leave=False):
            loss, logs = _ssl_step(model, loss_fn, batch, device, stepper.enabled)
            stepper.backward(loss)
            for key, value in logs.items():
                totals[key] += value
            n += 1
        stepper.flush()
        scheduler.step()

        row = {"epoch": epoch + 1, "lr": optimizer.param_groups[0]["lr"],
               **{f"train_{k}": v / max(n, 1) for k, v in totals.items()},
               **validate_ssl(model, loss_fn, val_loader, device, stepper.enabled)}
        if tracker.update(row["val_total"], epoch):
            save_checkpoint(out_dir / "best.pt", model, encoder_state=model.encoder_state_dict(), epoch=epoch,
                            metrics=row)
        save_last(last, model, optimizer, scheduler, stepper, tracker, epoch, metrics=row)
        history.log(row)
        LOG.info(format_row(row))
    return out_dir / "best.pt"
