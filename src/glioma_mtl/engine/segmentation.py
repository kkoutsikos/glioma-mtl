"""Supervised segmentation training with resumable checkpoints."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from monai.losses import DiceCELoss
from tqdm import tqdm

from ..evaluation.segmentation import evaluate_segmentation
from ..utils import LOG, CSVLogger, format_row, save_checkpoint
from .common import AmpStepper, BestTracker, EpochCost, make_optimizer, make_scheduler, save_last, try_resume


def train_segmentation(model: torch.nn.Module, train_loader, val_loader, cfg: Mapping, out_dir: str | Path,
                       device: torch.device) -> Path:
    out_dir = Path(out_dir)
    tcfg = cfg["train"]
    epochs = tcfg["epochs"]
    model.to(device)
    loss_fn = DiceCELoss(sigmoid=True)
    optimizer = make_optimizer(model.parameters(), tcfg)
    scheduler = make_scheduler(optimizer, tcfg, epochs)
    stepper = AmpStepper(optimizer, model.parameters(), device, tcfg.get("amp", True),
                         tcfg.get("accumulation", 1), tcfg.get("clip_norm", 1.0))
    tracker = BestTracker("max", tcfg.get("selection_ema", 0.0), tcfg.get("patience"))
    history = CSVLogger(out_dir / "history.csv")
    last = out_dir / "last.pt"
    start_epoch = try_resume(last, model, optimizer, scheduler, stepper, tracker, tcfg.get("resume", True))
    cost = EpochCost(model, device, out_dir)

    for epoch in range(start_epoch, epochs):
        if tracker.should_stop:
            LOG.info("Early stopping at epoch %d (best epoch %d)", epoch + 1, tracker.best_epoch + 1)
            break
        cost.start()
        model.train()
        running, steps = 0.0, 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}", leave=False):
            image, label = batch["image"].to(device), batch["label"].to(device)
            with torch.autocast(device.type, enabled=stepper.enabled):
                loss = loss_fn(model(image), label)
            stepper.backward(loss)
            running += loss.item()
            steps += 1
        stepper.flush()
        scheduler.step()

        row = {"epoch": epoch + 1, "lr": optimizer.param_groups[0]["lr"], "train_loss": running / max(steps, 1)}
        if (epoch + 1) % tcfg.get("val_every", 1) == 0 or epoch + 1 == epochs:
            metrics = evaluate_segmentation(model, val_loader, cfg["data"]["roi_size"], device,
                                            amp=tcfg.get("amp", True))
            row["val_dice"] = metrics["dice"].mean()
            row["val_hd95"] = metrics["hd95"].replace(float("inf"), float("nan")).mean()
            if tracker.update(row["val_dice"], epoch):
                save_checkpoint(out_dir / "best.pt", model, epoch=epoch, metrics=row)
                LOG.info("epoch %d: new best val Dice %.4f", epoch + 1, row["val_dice"])
        row.update(cost.stop())
        save_last(last, model, optimizer, scheduler, stepper, tracker, epoch)
        history.log(row)
        LOG.info(format_row(row))
    return out_dir / "best.pt"
