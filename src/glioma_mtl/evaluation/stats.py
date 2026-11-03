"""Uncertainty and significance: bootstrap CIs and paired tests on per-patient results.

With 25 test patients a single accuracy number has a 95% CI of roughly +/-0.19, and with 74 test
volumes and a per-case Dice SD of ~0.1, a 1-point Dice difference is noise. Report these.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm, wilcoxon
from scipy.stats import t as student_t


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


def icc_agreement(x: Sequence[float], y: Sequence[float]) -> float:
    """ICC(2,1): two-way random effects, absolute agreement, single measurement (Shrout & Fleiss), for one value
    measured two ways on the same subjects, e.g. a radiomics feature from predicted and from expert masks.
    Pairs with a missing value are dropped."""
    data = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    data = data[np.isfinite(data).all(1)]
    n, k = data.shape
    grand = data.mean()
    msr = k * ((data.mean(1) - grand) ** 2).sum() / (n - 1)          # between subjects
    msc = n * ((data.mean(0) - grand) ** 2).sum() / (k - 1)          # between the two measurements
    sse = ((data - data.mean(1, keepdims=True) - data.mean(0, keepdims=True) + grand) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / denom) if denom > 0 else float("nan")


def concordance_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Lin's concordance correlation coefficient: agreement on the identity line (1 = identical values)."""
    data = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    a, b = data[np.isfinite(data).all(1)].T
    denom = a.var() + b.var() + (a.mean() - b.mean()) ** 2
    return float(2 * np.mean((a - a.mean()) * (b - b.mean())) / denom) if denom > 0 else float("nan")


def corrected_resampled_ttest(a: Sequence[float], b: Sequence[float], test_train_ratio: float) -> dict:
    """Two models scored on the same CV folds: Nadeau & Bengio's corrected resampled t-test.

    The folds of repeated k-fold CV share most of their training data, so the plain paired t-test is far too
    liberal. The correction inflates the variance of the mean fold difference from ``var / n`` to
    ``(1 / n + n_test / n_train) * var`` (Nadeau & Bengio 2003; repeated k-fold form of Bouckaert & Frank
    2004), with ``n - 1`` degrees of freedom. Folds where either score is missing are dropped.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2:
        raise ValueError("the corrected t-test needs at least two folds")
    mean, var = float(d.mean()), float(d.var(ddof=1))
    se = np.sqrt((1 / n + test_train_ratio) * var)
    if se == 0:
        t, p = (0.0, 1.0) if mean == 0 else (float(np.sign(mean) * np.inf), 0.0)
    else:
        t = mean / se
        p = float(2 * student_t.sf(abs(t), df=n - 1))
    return {"mean_diff": mean, "t": float(t), "df": n - 1, "p_value": p, "n_folds": n}


def _delong_components(pos: np.ndarray, neg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """DeLong's structural components: per positive, the share of negatives it outranks (ties count half),
    and per negative, the share of positives that outrank it."""
    psi = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
    return psi.mean(1), psi.mean(0)


def delong_test(y: Sequence[int], score_a: Sequence[float], score_b: Sequence[float], alpha: float = 0.05) -> dict:
    """Two-sided DeLong test for the difference of two correlated ROC AUCs (binary ``y``, same patients), with
    the normal-approximation CI of the difference."""
    y = np.asarray(y)
    a, b = np.asarray(score_a, dtype=float), np.asarray(score_b, dtype=float)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("DeLong's test needs binary labels with both classes present")
    v10, v01 = zip(*(_delong_components(s[y == 1], s[y == 0]) for s in (a, b)))
    auc = np.array([v.mean() for v in v10])
    s10, s01 = np.cov(np.vstack(v10)), np.cov(np.vstack(v01))
    contrast = np.array([1.0, -1.0])
    var = contrast @ s10 @ contrast / len(v10[0]) + contrast @ s01 @ contrast / len(v01[0])
    diff = float(auc[0] - auc[1])
    if var <= 0:
        z, p = (0.0, 1.0) if diff == 0 else (float(np.sign(diff) * np.inf), 0.0)
    else:
        z = diff / np.sqrt(var)
        p = float(2 * norm.sf(abs(z)))
    half = norm.ppf(1 - alpha / 2) * np.sqrt(max(var, 0.0))
    return {"auc_a": float(auc[0]), "auc_b": float(auc[1]), "mean_diff": diff, "ci_low": diff - half,
            "ci_high": diff + half, "z": float(z), "p_value": p, "n": len(y)}


def paired_bootstrap_test(y: Sequence[int], score_a: np.ndarray, score_b: np.ndarray, metric: Callable,
                          n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> dict:
    """``metric(a) - metric(b)`` on the same patients, resampling patients for both models together.

    Returns the difference, its percentile CI and a two-sided p-value: twice the share of resamples on the
    far side of zero (at least ``1 / n_boot``). For multi-class AUC, where DeLong's test does not apply directly.
    Resamples where the metric fails (e.g. a class missing) are dropped.
    """
    rng = np.random.default_rng(seed)
    y, a, b = np.asarray(y), np.asarray(score_a), np.asarray(score_b)
    diff = float(metric(y, a) - metric(y, b))
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        try:
            stats.append(metric(y[idx], a[idx]) - metric(y[idx], b[idx]))
        except ValueError:
            continue
    stats = np.asarray(stats)
    low, high = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    tail = min((stats <= 0).mean(), (stats >= 0).mean())
    p = float(min(1.0, max(2 * tail, 1 / len(stats))))
    return {"mean_diff": diff, "ci_low": float(low), "ci_high": float(high), "p_value": p, "n": len(y),
            "n_boot": len(stats)}
