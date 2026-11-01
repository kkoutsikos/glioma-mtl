"""Classification metrics with bootstrap confidence intervals."""
from __future__ import annotations

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
                           seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Summary table (with 95% bootstrap CI) and confusion matrix. Rows with ``y == -1`` are dropped."""
    y = np.asarray(y)
    prob = np.asarray(prob)
    keep = y >= 0
    y, prob = y[keep], prob[keep]
    pred = prob.argmax(1)
    labels = list(range(len(class_names)))

    metrics = {
        "accuracy": lambda a, b: accuracy_score(a, b.argmax(1)),
        "balanced_accuracy": lambda a, b: balanced_accuracy_score(a, b.argmax(1)),
        "macro_f1": lambda a, b: f1_score(a, b.argmax(1), labels=labels, average="macro", zero_division=0),
        "macro_auc": macro_auc,
    }
    rows = []
    for name, fn in metrics.items():
        try:
            point, low, high = (bootstrap_ci(y_true=y, y_score=prob, metric=fn, seed=seed) if ci
                                else (fn(y, prob), np.nan, np.nan))
        except ValueError:
            point = low = high = np.nan
        rows.append({"metric": name, "value": point, "ci_low": low, "ci_high": high, "n": len(y)})
    for i, cname in enumerate(class_names):
        support = int((y == i).sum())
        recall = float(((pred == i) & (y == i)).sum() / support) if support else np.nan
        rows.append({"metric": f"recall[{cname}]", "value": recall, "ci_low": np.nan, "ci_high": np.nan, "n": support})
    return pd.DataFrame(rows), confusion_matrix(y, pred, labels=labels)
