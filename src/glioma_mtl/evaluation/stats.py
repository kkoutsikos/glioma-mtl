"""Uncertainty and significance: bootstrap CIs and paired tests on per-patient results.

With 25 test patients a single accuracy number has a 95% CI of roughly +/-0.19, and with 74 test
volumes and a per-case Dice SD of ~0.1, a 1-point Dice difference is noise. Report these.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def bootstrap_ci(values: Sequence[float] | None = None, *, y_true=None, y_score=None,
                 metric: Callable | None = None, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """``(point, low, high)``. Pass per-case ``values`` (mean is used), or ``y_true``/``y_score`` + ``metric``.

    Resampling is over patients. Resamples where the metric fails (e.g. a class missing for AUC) are dropped.
    """
    rng = np.random.default_rng(seed)
    if values is not None:
        v = np.asarray(values, dtype=float)
        v = v[~np.isnan(v)]
        point = float(v.mean())
        stats = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)]
    else:
        y_true, y_score = np.asarray(y_true), np.asarray(y_score)
        point = float(metric(y_true, y_score))
        stats = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(y_true), len(y_true))
            try:
                stats.append(metric(y_true[idx], y_score[idx]))
            except ValueError:
                continue
    low, high = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return point, float(low), float(high)


def paired_wilcoxon(a: pd.DataFrame, b: pd.DataFrame, metric: str, on: str = "patient_id") -> dict:
    """Wilcoxon signed-rank test on per-patient ``metric`` of two models evaluated on the same cases."""
    merged = a[[on, metric]].merge(b[[on, metric]], on=on, suffixes=("_a", "_b")).dropna()
    diff = merged[f"{metric}_a"] - merged[f"{metric}_b"]
    stat, p = wilcoxon(merged[f"{metric}_a"], merged[f"{metric}_b"])
    return {"n": len(merged), "mean_a": merged[f"{metric}_a"].mean(), "mean_b": merged[f"{metric}_b"].mean(),
            "mean_diff": diff.mean(), "median_diff": diff.median(), "statistic": float(stat), "p_value": float(p)}
