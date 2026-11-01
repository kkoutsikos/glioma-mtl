"""Per-case segmentation metrics (binary whole-tumour masks, 1 mm isotropic spacing)."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from monai.inferers import sliding_window_inference
from monai.metrics import (compute_dice, compute_hausdorff_distance, compute_iou,
                           compute_surface_dice)
from tqdm import tqdm


def case_metrics(pred: torch.Tensor, gt: torch.Tensor, nsd_tolerance_mm: float = 1.0) -> dict[str, float]:
    """``pred``/``gt``: binary [1, 1, D, H, W]. Single channel, so the background is not involved."""
    tp = float((pred * gt).sum())
    fp = float((pred * (1 - gt)).sum())
    fn = float(((1 - pred) * gt).sum())
    return {
        "dice": compute_dice(pred, gt, include_background=True).item(),
        "iou": compute_iou(pred, gt, include_background=True).item(),
        "hd95": compute_hausdorff_distance(pred, gt, include_background=True, percentile=95).item(),
        "nsd": compute_surface_dice(pred, gt, class_thresholds=[nsd_tolerance_mm], include_background=True).item(),
        "sensitivity": tp / (tp + fn) if tp + fn > 0 else np.nan,
        "precision": tp / (tp + fp) if tp + fp > 0 else np.nan,
        "gt_volume_ml": float(gt.sum()) / 1000.0,
        "pred_volume_ml": float(pred.sum()) / 1000.0,
    }


@torch.no_grad()
def evaluate_segmentation(model: torch.nn.Module, loader, roi_size, device: torch.device,
                          sw_batch_size: int = 2, overlap: float = 0.5, amp: bool = True,
                          on_prediction=None) -> pd.DataFrame:
    """Sliding-window inference on whole volumes; one row per patient.

    ``on_prediction(batch, binary_pred)`` is called per case (used to save predicted masks).
    """
    model.eval()
    rows = []
    for batch in tqdm(loader, desc="evaluate", leave=False):
        image = batch["image"].to(device)
        start = time.perf_counter()
        with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
            logits = sliding_window_inference(image, roi_size, sw_batch_size, model, overlap=overlap, mode="gaussian")
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        pred = (torch.sigmoid(logits.float()) > 0.5).float()
        row = {"patient_id": batch["patient_id"][0], "seconds": elapsed}
        if "label" in batch:
            row.update(case_metrics(pred.cpu(), batch["label"].cpu().float()))
        rows.append(row)
        if on_prediction is not None:
            on_prediction(batch, pred)
    return pd.DataFrame(rows)


def summarize_segmentation(df: pd.DataFrame, metrics=("dice", "iou", "hd95", "nsd", "sensitivity", "precision")) -> pd.DataFrame:
    from .stats import bootstrap_ci

    rows = []
    for m in metrics:
        if m in df:
            point, low, high = bootstrap_ci(df[m].replace([np.inf, -np.inf], np.nan).values)
            rows.append({"metric": m, "mean": point, "ci_low": low, "ci_high": high,
                         "sd": df[m].std(), "median": df[m].median(), "n": int(df[m].notna().sum())})
    return pd.DataFrame(rows)
