"""Classification metrics with bootstrap confidence intervals."""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score)

from .stats import bootstrap_ci


def macro_auc(y: np.ndarray, prob: np.ndarray) -> float:
    present = np.unique(y)
    if len(present) < 2:
        raise ValueError("AUC needs at least two classes")
    if prob.shape[1] == 2:
        return roc_auc_score(y, prob[:, 1])
    # One-vs-rest over the classes present in y; renormalise the probabilities over those classes.
    p = prob[:, present]
    p = p / p.sum(1, keepdims=True)
    return roc_auc_score(y, p, multi_class="ovr", average="macro", labels=present)


def classification_metrics(y: Sequence[int], prob: np.ndarray, class_names: Sequence, ci: bool = True,
                           seed: int = 0, pred: Sequence[int] | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Summary table (with 95% bootstrap CI) and confusion matrix. Rows with ``y == -1`` are dropped.

    ``pred`` are the predicted classes (default: the most probable one); AUC always uses ``prob``.
    """
    y = np.asarray(y)
    prob = np.asarray(prob)
    pred = prob.argmax(1) if pred is None else np.asarray(pred)
    keep = y >= 0
    y, prob, pred = y[keep], prob[keep], pred[keep]
    labels = list(range(len(class_names)))

    def call(scored):
        return scored[:, -1].astype(int)

    # The calls ride along as the last column, so the bootstrap resamples them with the probabilities.
    scored = np.column_stack([prob, pred])
    metrics = {
        "accuracy": lambda a, s: accuracy_score(a, call(s)),
        "balanced_accuracy": lambda a, s: balanced_accuracy_score(a, call(s)),
        "macro_f1": lambda a, s: f1_score(a, call(s), labels=labels, average="macro", zero_division=0),
        "macro_auc": lambda a, s: macro_auc(a, s[:, :-1]),
    }
    rows = []
    for name, fn in metrics.items():
        try:
            # A bootstrap resample of a small test set often lacks a rare class that the model still predicts;
            # balanced accuracy then averages recall over the classes present and warns once per resample.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
                point, low, high = (bootstrap_ci(y_true=y, y_score=scored, metric=fn, seed=seed) if ci
                                    else (fn(y, scored), np.nan, np.nan))
        except ValueError:
            point = low = high = np.nan
        rows.append({"metric": name, "value": point, "ci_low": low, "ci_high": high, "n": len(y)})
    for i, cname in enumerate(class_names):
        support = int((y == i).sum())
        recall = float(((pred == i) & (y == i)).sum() / support) if support else np.nan
        rows.append({"metric": f"recall[{cname}]", "value": recall, "ci_low": np.nan, "ci_high": np.nan, "n": support})
    return pd.DataFrame(rows), confusion_matrix(y, pred, labels=labels)
