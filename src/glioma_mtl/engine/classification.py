"""Multi-task classification training / inference."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from tqdm import tqdm

from ..losses import MultiTaskLoss
from ..models.classification import MultiTaskGliomaNet
from ..utils import LOG, CSVLogger, format_row, save_checkpoint
from .common import AmpStepper, BestTracker, EpochCost, make_optimizer, make_scheduler, save_last, try_resume


def _inputs(batch: Mapping, model: MultiTaskGliomaNet, device: torch.device, mask_key: str):
    image = batch["image"].to(device)
    tabular = {name: batch[f"tab_{name}"].to(device) for name in model.tabular_dims}
    mask = batch[mask_key].to(device) if model.mask_mode == "input" else None
    return image, tabular, mask


def _targets(batch: Mapping, model: MultiTaskGliomaNet, device: torch.device) -> dict[str, torch.Tensor]:
    return {t: torch.as_tensor(batch[f"y_{t}"]).long().to(device) for t in model.tasks}


@torch.no_grad()
def predict(model: MultiTaskGliomaNet, loader, device: torch.device, loss_fn: MultiTaskLoss | None = None,
            mask_key: str = "label", amp: bool = True) -> tuple[pd.DataFrame, float]:
    """One row per sample: ``patient_id``, ``y_<task>`` and ``p_<task>_<k>`` columns; plus mean loss."""
    model.eval()
    rows, losses = [], []
    for batch in loader:
        image, tabular, mask = _inputs(batch, model, device, mask_key)
        targets = _targets(batch, model, device)
        with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
            out = model(image, tabular, mask)
        if loss_fn is not None:
            seg = batch["label"].to(device) if "label" in batch else None
            losses.append(loss_fn(out, targets, seg)[0].item())
        for i in range(image.shape[0]):
            row = {"patient_id": batch["patient_id"][i]}
            for t, logits in out["logits"].items():
                row[f"y_{t}"] = int(targets[t][i])
                for k, p in enumerate(torch.softmax(logits[i].float(), 0).tolist()):
                    row[f"p_{t}_{k}"] = p
            rows.append(row)
    return pd.DataFrame(rows), float(np.mean(losses)) if losses else float("nan")


def add_derived_lgg_hgg(df: pd.DataFrame) -> pd.DataFrame:
    """LGG (grade 2-3) vs HGG (grade 4) from the 3-way grade head, for models without a dedicated head.

    P(HGG) = P(grade 4). Models trained with an ``lgg_hgg`` task keep their own head's output.
    """
    if "y_grade" not in df or "y_lgg_hgg" in df:
        return df
    out = df.copy()
    out["y_lgg_hgg"] = np.where(out["y_grade"] >= 0, (out["y_grade"] == 2).astype(int), -1)
    out["p_lgg_hgg_0"] = out["p_grade_0"] + out["p_grade_1"]
    out["p_lgg_hgg_1"] = out["p_grade_2"]
    return out


def task_arrays(df: pd.DataFrame, task: str) -> tuple[np.ndarray, np.ndarray]:
    probs = df[sorted((c for c in df if c.startswith(f"p_{task}_")), key=lambda c: int(c.rsplit("_", 1)[1]))]
    return df[f"y_{task}"].to_numpy(), probs.to_numpy()


def score(df: pd.DataFrame, tasks) -> dict[str, float]:
    out = {}
    for t in tasks:
        y, p = task_arrays(df, t)
        keep = y >= 0
        out[f"{t}_macro_f1"] = f1_score(y[keep], p[keep].argmax(1), average="macro", zero_division=0)
        out[f"{t}_bal_acc"] = balanced_accuracy_score(y[keep], p[keep].argmax(1))
    out["mean_macro_f1"] = float(np.mean([out[f"{t}_macro_f1"] for t in tasks]))
    return out


def train_classifier(model: MultiTaskGliomaNet, train_loader, val_loader, loss_fn: MultiTaskLoss, cfg: Mapping,
                     out_dir: str | Path, device: torch.device) -> Path:
    out_dir = Path(out_dir)
    tcfg = cfg["train"]
    epochs = tcfg["epochs"]
    mask_key = cfg["data"].get("model_mask_key", "label")
    model.to(device)
    loss_fn.to(device)

    groups = [{"params": list(model.head_parameters()), "lr": tcfg["lr"]}]
    if not model.freeze_backbone:
        groups.append({"params": list(model.backbone_parameters()), "lr": tcfg.get("lr_backbone", tcfg["lr"] * 0.1)})
    optimizer = make_optimizer(groups, tcfg)
    scheduler = make_scheduler(optimizer, tcfg, epochs)
    params = [p for g in groups for p in g["params"]]
    stepper = AmpStepper(optimizer, params, device, tcfg.get("amp", True), tcfg.get("accumulation", 1),
                         tcfg.get("clip_norm", 1.0))

    sel = cfg.get("selection", {})
    sel_metric = sel.get("metric", "val_loss")
    tracker = BestTracker("min" if sel_metric == "val_loss" else "max", sel.get("ema", 0.0), sel.get("patience"))
    history = CSVLogger(out_dir / "history.csv")
    last = out_dir / "last.pt"
    start_epoch = try_resume(last, model, optimizer, scheduler, stepper, tracker, tcfg.get("resume", True))
    cost = EpochCost(model, device, out_dir)

    for epoch in range(start_epoch, epochs):
        if tracker.should_stop:
            LOG.info("Early stopping at epoch %d (best %d)", epoch + 1, tracker.best_epoch + 1)
            break
        cost.start()
        model.train()
        totals: dict[str, float] = defaultdict(float)
        n = 0
        for batch in tqdm(train_loader, desc=f"cls {epoch + 1}/{epochs}", leave=False):
            image, tabular, mask = _inputs(batch, model, device, mask_key)
            targets = _targets(batch, model, device)
            with torch.autocast(device.type, enabled=stepper.enabled):
                out = model(image, tabular, mask)
                loss, logs = loss_fn(out, targets, batch["label"].to(device))
            stepper.backward(loss)
            for k, v in logs.items():
                totals[k] += v
            n += 1
        stepper.flush()
        scheduler.step()

        val_df, val_loss = predict(model, val_loader, device, loss_fn, mask_key, tcfg.get("amp", True))
        row = {"epoch": epoch + 1, **{f"train_{k}": v / max(n, 1) for k, v in totals.items()},
               "val_loss": val_loss, **{f"val_{k}": v for k, v in score(val_df, model.tasks).items()}}
        value = row[sel_metric] if sel_metric in row else row[f"val_{sel_metric}"]
        if tracker.update(value, epoch):
            save_checkpoint(out_dir / "best.pt", model, epoch=epoch, metrics=row)
            val_df.to_csv(out_dir / "val_predictions_best.csv", index=False)
        row.update(cost.stop())
        save_last(last, model, optimizer, scheduler, stepper, tracker, epoch, metrics=row)
        history.log(row)
        LOG.info(format_row(row))
    return out_dir / "best.pt"
