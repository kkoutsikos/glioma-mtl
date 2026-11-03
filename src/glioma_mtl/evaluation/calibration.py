"""Calibration and clinical usefulness of predicted probabilities.

The classifiers are trained with balanced class weights, which pulls probabilities towards the rare classes, so
expect imperfect calibration by design; report it rather than hide it.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def brier_score(y: Sequence[int], prob: np.ndarray) -> float:
    """Multi-class Brier score: mean over patients of the squared distance between the probability vector and the
    one-hot truth (0 = perfect; 2 = confidently wrong). For two classes it is twice the usual binary Brier score."""
    y, prob = np.asarray(y, dtype=int), np.asarray(prob, dtype=float)
    return float(((prob - np.eye(prob.shape[1])[y]) ** 2).sum(1).mean())


def _bins(conf: np.ndarray, n_bins: int) -> np.ndarray:
    return np.clip((conf * n_bins).astype(int), 0, n_bins - 1)


def expected_calibration_error(y: Sequence[int], prob: np.ndarray, n_bins: int = 10) -> float:
    """ECE with equal-width bins: the bin-size-weighted mean |observed - predicted|.

    Two classes: calibration of P(class 1) against the observed class-1 rate. More classes: top-label calibration,
    the confidence of the most probable class against how often that call is right.
    """
    y, prob = np.asarray(y, dtype=int), np.asarray(prob, dtype=float)
    if prob.shape[1] == 2:
        conf, hit = prob[:, 1], (y == 1).astype(float)
    else:
        conf, hit = prob.max(1), (prob.argmax(1) == y).astype(float)
    b = _bins(conf, n_bins)
    return float(sum(abs(hit[b == i].mean() - conf[b == i].mean()) * (b == i).mean() for i in np.unique(b)))


def reliability_curve(y: Sequence[int], p: Sequence[float], n_bins: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Binary reliability diagram: per non-empty bin, mean predicted probability, observed rate and count."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    b = _bins(p, n_bins)
    used = np.unique(b)
    return (np.array([p[b == i].mean() for i in used]), np.array([y[b == i].mean() for i in used]),
            np.array([(b == i).sum() for i in used]))


def net_benefit(y: Sequence[int], p: Sequence[float], thresholds: Sequence[float]) -> np.ndarray:
    """Decision-curve net benefit of treating every patient with ``p >= t``: TP/n - FP/n * t / (1 - t)."""
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    out = []
    for t in thresholds:
        call = p >= t
        out.append(((call & (y == 1)).sum() - (call & (y == 0)).sum() * t / (1 - t)) / len(y))
    return np.asarray(out)


def treat_all_net_benefit(y: Sequence[int], thresholds: Sequence[float]) -> np.ndarray:
    """Net benefit of treating everyone (the reference curve; treating no one is 0)."""
    prevalence = float(np.mean(np.asarray(y) == 1))
    t = np.asarray(thresholds, dtype=float)
    return prevalence - (1 - prevalence) * t / (1 - t)
