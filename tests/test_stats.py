import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from glioma_mtl.evaluation import (brier_score, concordance_correlation, corrected_resampled_ttest, delong_test,
                                   expected_calibration_error, icc_agreement, macro_auc, net_benefit,
                                   paired_bootstrap_test, permutation_p, rank_macro_auc, reliability_curve,
                                   split_gap_null, stratified_partitions, treat_all_net_benefit)


def test_rank_macro_auc_equals_macro_auc_and_counts_ties_as_half():
    rng = np.random.default_rng(0)
    for n_classes in (2, 3, 4):
        y = np.repeat(np.arange(n_classes), 12)
        prob = rng.dirichlet(np.ones(n_classes), len(y))
        assert np.isclose(rank_macro_auc(y, prob), macro_auc(y, prob))
        tied = np.round(prob, 1)
        expected = np.mean([roc_auc_score(y == k, tied[:, k]) for k in range(n_classes)])
        assert np.isclose(rank_macro_auc(y, tied), expected)
    with pytest.raises(ValueError):
        rank_macro_auc(np.zeros(5, int), np.full((5, 2), 0.5))


def test_stratified_partitions_keep_each_class_count_and_vary():
    y = np.array([0] * 10 + [1] * 6 + [2] * 4)
    in_first = np.array([True] * 5 + [False] * 5 + [True] * 2 + [False] * 4 + [True, True, True, False])
    masks = stratified_partitions(y, in_first, n_perm=200, seed=1)
    assert masks.shape == (200, 20)
    for k, count in ((0, 5), (1, 2), (2, 3)):
        assert (masks[:, y == k].sum(1) == count).all()
    assert len({m.tobytes() for m in masks}) > 100
    assert (masks == stratified_partitions(y, in_first, n_perm=200, seed=1)).all()      # same seed, same partitions


def test_split_gap_test_finds_a_real_gap_and_no_gap_when_splits_are_exchangeable():
    rng = np.random.default_rng(3)
    y = np.tile([0, 1], 30)
    in_first = np.r_[np.ones(30, bool), np.zeros(30, bool)]
    partitions = stratified_partitions(y, in_first, n_perm=2000, seed=0)

    def gap_and_p(score):
        prob = np.c_[1 - score, score]
        gap = rank_macro_auc(y[~in_first], prob[~in_first]) - rank_macro_auc(y[in_first], prob[in_first])
        null = split_gap_null(y, prob, partitions)
        assert abs(null.mean()) < 0.02                                                 # chance gaps centre on zero
        return gap, permutation_p(gap, null)

    # The same signal in both splits: no more of a gap than chance gives.
    same = np.clip(0.5 + 0.2 * (y - 0.5) + rng.normal(0, 0.2, 60), 0.01, 0.99)
    assert gap_and_p(same)[1] > 0.05
    # Signal only in the second split: test beats validation by far more than re-assigning patients can explain.
    only_second = np.where(in_first, rng.uniform(0.01, 0.99, 60), np.clip(0.5 + 0.6 * (y - 0.5), 0.01, 0.99))
    gap, p = gap_and_p(only_second)
    assert gap > 0.3 and p < 0.01


def test_split_gap_null_matches_scoring_each_partition_separately():
    rng = np.random.default_rng(5)
    y = np.repeat([0, 1, 2, 3], [20, 6, 18, 6])
    prob = np.round(rng.dirichlet(np.ones(4), len(y)), 2)                # with ties
    partitions = stratified_partitions(y, rng.permutation(np.arange(50) % 2 == 0), n_perm=25, seed=2)
    slow = [rank_macro_auc(y[~m], prob[~m]) - rank_macro_auc(y[m], prob[m]) for m in partitions]
    assert np.allclose(split_gap_null(y, prob, partitions), slow)


def test_permutation_p_counts_the_observed_split_and_uses_distance_from_zero():
    assert permutation_p(0.5, np.zeros(99)) == 1 / 100                   # never zero
    assert permutation_p(0.0, np.zeros(99)) == 1.0
    assert permutation_p(-0.3, [0.3, -0.3, 0.1, 0.0]) == 3 / 5           # both tails count


def test_agreement_coefficients_penalise_a_systematic_offset():
    x, y = np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0])      # same ranking, offset by 1
    assert np.isclose(icc_agreement(x, y), 2 / 3)                      # MSR 2, MSC 1.5, MSE 0
    assert np.isclose(concordance_correlation(x, y), 4 / 7)
    assert np.isclose(icc_agreement(x, x), 1.0) and np.isclose(concordance_correlation(x, x), 1.0)
    assert np.isclose(icc_agreement([1, 2, np.nan, 3], [1, 2, 5, 3]), 1.0)   # incomplete pairs are dropped


def test_calibration_and_decision_curve_values():
    y = np.array([0, 1, 2, 3])
    assert brier_score(y, np.eye(4)[y]) == 0 and np.isclose(brier_score(y, np.full((4, 4), 0.25)), 0.75)
    rng = np.random.default_rng(0)
    p = rng.uniform(size=20000)
    calibrated = (rng.uniform(size=p.size) < p).astype(int)
    assert expected_calibration_error(calibrated, np.c_[1 - p, p]) < 0.02
    assert expected_calibration_error(calibrated, np.c_[1 - p ** 3, p ** 3]) > 0.1
    mean_p, observed, counts = reliability_curve(calibrated, p, n_bins=5)
    assert counts.sum() == p.size and np.allclose(mean_p, observed, atol=0.03)

    yb, pb = np.array([1, 0, 1, 0]), np.array([0.9, 0.2, 0.6, 0.7])
    assert np.allclose(net_benefit(yb, pb, [0.5]), [2 / 4 - 1 / 4])  # calls 0, 2, 3: two true, one false positive
    assert np.allclose(treat_all_net_benefit(yb, [0.5, 0.2]), [0.0, 0.5 - 0.5 * 0.25])


def test_corrected_ttest_inflates_the_variance_of_overlapping_folds():
    rng = np.random.default_rng(0)
    a = 0.80 + rng.normal(0, 0.05, 50)
    b = a - 0.02 + rng.normal(0, 0.03, 50)
    r = corrected_resampled_ttest(a, b, test_train_ratio=0.25)
    d = a - b
    t = d.mean() / np.sqrt((1 / 50 + 0.25) * d.var(ddof=1))
    assert np.isclose(r["t"], t) and r["df"] == 49 and np.isclose(r["mean_diff"], d.mean())
    naive = corrected_resampled_ttest(a, b, test_train_ratio=0.0)    # the plain paired t-test
    assert naive["p_value"] < r["p_value"]
    assert corrected_resampled_ttest(a, a, 0.25)["p_value"] == 1.0
    assert corrected_resampled_ttest(np.full(5, 0.9), np.full(5, 0.8), 0.25)["p_value"] == 0.0   # constant difference
    assert corrected_resampled_ttest([0.8, np.nan, 0.7], [0.7, 0.6, 0.5], 0.25)["n_folds"] == 2


def test_delong_matches_a_hand_computed_example():
    # AUC_a = 0.75 (one positive ranks below one negative), AUC_b = 1. Structural components give
    # var(diff) = 0.125 / 2 + 0.125 / 2, so z = -0.25 / sqrt(0.125).
    y = np.array([0, 0, 1, 1])
    a, b = np.array([0.1, 0.4, 0.35, 0.8]), np.array([0.2, 0.3, 0.6, 0.9])
    r = delong_test(y, a, b)
    assert np.isclose(r["auc_a"], roc_auc_score(y, a)) and np.isclose(r["auc_b"], 1.0)
    assert np.isclose(r["z"], -0.25 / np.sqrt(0.125)) and np.isclose(r["p_value"], 0.4795, atol=1e-4)
    assert r["ci_low"] < r["mean_diff"] < r["ci_high"]
    swapped = delong_test(y, b, a)
    assert np.isclose(swapped["mean_diff"], 0.25) and np.isclose(swapped["p_value"], r["p_value"])
    assert delong_test(y, a, a)["p_value"] == 1.0
    with pytest.raises(ValueError, match="binary"):
        delong_test(np.zeros(4, int), a, b)


def test_delong_detects_a_clearly_better_model():
    rng = np.random.default_rng(1)
    y = np.repeat([0, 1], 60)
    good = y + rng.normal(0, 0.5, len(y))
    poor = y + rng.normal(0, 3.0, len(y))
    assert delong_test(y, good, poor)["p_value"] < 0.01


def test_paired_bootstrap_of_a_multiclass_auc():
    rng = np.random.default_rng(2)
    y = np.repeat([0, 1, 2], 20)
    good = np.eye(3)[y] * 3 + rng.normal(0, 1, (60, 3))
    good = np.exp(good) / np.exp(good).sum(1, keepdims=True)
    noise = rng.dirichlet(np.ones(3), 60)
    r = paired_bootstrap_test(y, good, noise, macro_auc, n_boot=500)
    assert np.isclose(r["mean_diff"], macro_auc(y, good) - macro_auc(y, noise))
    assert r["p_value"] < 0.01 and r["ci_low"] > 0
    same = paired_bootstrap_test(y, good, good, macro_auc, n_boot=200)
    assert same["mean_diff"] == 0 and same["p_value"] == 1.0 and same["ci_low"] == same["ci_high"] == 0
