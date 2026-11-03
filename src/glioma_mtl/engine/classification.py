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


# Every combination of flips of the three spatial axes of [B, C, D, H, W]; training flips each axis at random.
FLIPS = [(), (2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)]


@torch.no_grad()
def predict(model: MultiTaskGliomaNet, loader, device: torch.device, loss_fn: MultiTaskLoss | None = None,
            mask_key: str = "label", amp: bool = True, tta: bool = False) -> tuple[pd.DataFrame, float]:
    """One row per sample: ``patient_id``, ``y_<task>`` and ``p_<task>_<k>`` columns; plus mean loss.

    ``tta`` averages the class probabilities over the 8 flipped copies of each crop; the loss is that of
    the unflipped crop.
    """
    model.eval()
    rows, losses = [], []
    for batch in loader:
        image, tabular, mask = _inputs(batch, model, device, mask_key)
        targets = _targets(batch, model, device)
        probs: dict[str, torch.Tensor] = defaultdict(float)
        for dims in FLIPS if tta else FLIPS[:1]:
            flip = (lambda x: x.flip(dims)) if dims else (lambda x: x)
            with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
                out = model(flip(image), tabular, None if mask is None else flip(mask))
            if not dims and loss_fn is not None:
                seg = batch["label"].to(device) if "label" in batch else None
                losses.append(loss_fn(out, targets, seg)[0].item())
            for t, logits in out["logits"].items():
                probs[t] = probs[t] + torch.softmax(logits.float(), 1) / (len(FLIPS) if tta else 1)
        for i in range(image.shape[0]):
            row = {"patient_id": batch["patient_id"][i]}
            for t, p in probs.items():
                row[f"y_{t}"] = int(targets[t][i])
                row.update({f"p_{t}_{k}": v for k, v in enumerate(p[i].tolist())})
            rows.append(row)
    return pd.DataFrame(rows), float(np.mean(losses)) if losses else float("nan")


def add_derived_lgg_hgg(df: pd.DataFrame) -> pd.DataFrame:
    """LGG (grade 2-3) vs HGG (grade 4) from the 3-way grade head, for models without a dedicated head.

    P(HGG) = P(grade 4), used for the AUC; the call is the grade head's own call (see ``task_calls``).
    Models trained with an ``lgg_hgg`` task keep their own head's output.
    """
    if "y_grade" not in df or "y_lgg_hgg" in df:
        return df
    out = df.copy()
    out["y_lgg_hgg"] = np.where(out["y_grade"] >= 0, (out["y_grade"] == 2).astype(int), -1)
    out["p_lgg_hgg_0"] = out["p_grade_0"] + out["p_grade_1"]
    out["p_lgg_hgg_1"] = out["p_grade_2"]
    out["lgg_hgg_from_grade"] = 1
    return out


def task_arrays(df: pd.DataFrame, task: str) -> tuple[np.ndarray, np.ndarray]:
    probs = df[sorted((c for c in df if c.startswith(f"p_{task}_")), key=lambda c: int(c.rsplit("_", 1)[1]))]
    return df[f"y_{task}"].to_numpy(), probs.to_numpy()


def task_calls(df: pd.DataFrame, task: str) -> np.ndarray:
    """Predicted class per patient: the most probable one, except for LGG vs HGG derived from the grade head.

    That one is HGG exactly when the grade call is grade 4. Grade 4 is about a third of the cohort, so a
    cautious model rarely puts P(grade 4) above 0.5 and the most probable binary class would be LGG even for
    patients the same model calls grade 4.
    """
    if task == "lgg_hgg" and "lgg_hgg_from_grade" in df:
        return (task_arrays(df, "grade")[1].argmax(1) == 2).astype(int)
    return task_arrays(df, task)[1].argmax(1)


def score(df: pd.DataFrame, tasks, weights: Mapping[str, float] | None = None) -> dict[str, float]:
    """Macro-F1 and balanced accuracy per task, and ``cls_nll``: the task-weighted mean negative
    log-likelihood of the true class, a smooth score of the classification heads alone (no
    segmentation, auxiliary or contrastive terms), used to pick the checkpoint."""
    out = {}
    weights = weights or {}
    for t in tasks:
        y, p = task_arrays(df, t)
        keep = y >= 0
        out[f"{t}_macro_f1"] = f1_score(y[keep], p[keep].argmax(1), average="macro", zero_division=0)
        out[f"{t}_bal_acc"] = balanced_accuracy_score(y[keep], p[keep].argmax(1))
        out[f"{t}_nll"] = float(-np.log(np.clip(p[keep][np.arange(keep.sum()), y[keep]], 1e-7, 1.0)).mean())
    out["mean_macro_f1"] = float(np.mean([out[f"{t}_macro_f1"] for t in tasks]))
    w = {t: weights.get(t, 1.0) for t in tasks}
    out["cls_nll"] = sum(w[t] * out[f"{t}_nll"] for t in tasks) / sum(w.values())
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
    minimize = sel_metric.endswith(("loss", "nll"))
    tracker = BestTracker("min" if minimize else "max", sel.get("ema", 0.0), sel.get("patience"))
    tta = cfg.get("inference", {}).get("tta_flips", False)
    history = CSVLogger(out_dir / "history.csv")
    last = out_dir / "last.pt"
    start_epoch = try_resume(last, model, optimizer, scheduler, stepper, tracker, tcfg.get("resume", True))
    cost = EpochCost(model, device, out_dir)
    # Heads first: the backbone stays frozen for the first epochs so that the randomly initialised heads'
    # early gradients do not disturb the pretrained features, then everything is fine-tuned.
    always_frozen = model.freeze_backbone
    head_epochs = 0 if always_frozen else tcfg.get("freeze_backbone_epochs", 0)

    for epoch in range(start_epoch, epochs):
        if tracker.should_stop:
            LOG.info("Early stopping at epoch %d (best %d)", epoch + 1, tracker.best_epoch + 1)
            break
        if head_epochs:
            model.freeze_backbone = epoch < head_epochs
            if epoch == head_epochs:
                LOG.info("epoch %d: backbone unfrozen after %d head-only epochs", epoch + 1, head_epochs)
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

        val_df, val_loss = predict(model, val_loader, device, loss_fn, mask_key, tcfg.get("amp", True), tta)
        row = {"epoch": epoch + 1, **{f"train_{k}": v / max(n, 1) for k, v in totals.items()}, "val_loss": val_loss,
               **{f"val_{k}": v for k, v in score(val_df, model.tasks, loss_fn.task_weights).items()}}
        value = row[sel_metric] if sel_metric in row else row[f"val_{sel_metric}"]
        if tracker.update(value, epoch):
            save_checkpoint(out_dir / "best.pt", model, epoch=epoch, metrics=row)
            val_df.to_csv(out_dir / "val_predictions_best.csv", index=False)
        row.update(cost.stop())
        save_last(last, model, optimizer, scheduler, stepper, tracker, epoch, metrics=row)
        history.log(row)
        LOG.info(format_row(row))
    model.freeze_backbone = always_frozen
    return out_dir / "best.pt"
