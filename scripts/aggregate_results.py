"""Collect all paper runs into tables (CSV + Markdown + LaTeX).

    python scripts/aggregate_results.py --plan configs/paper.yaml --root runs/paper --out runs/paper/paper_tables

* Segmentation: mean ± SD of the per-seed test means; 95% bootstrap CI over patients (seed-averaged);
  paired Wilcoxon of every model vs. the best one (Holm-corrected); parameters, training time, peak GPU
  memory, inference time per case and the GPU used. "~" marks a training time partly reconstructed from
  log.txt timestamps (epochs trained before cost logging existed).
* Classification: mean ± SD over seeds, plus the seed ensemble (averaged probabilities) with a 95%
  bootstrap CI over the 25 test patients.
* K-fold late fusion: mean ± SD over the CV folds, and the one-shot hold-out result with its CI.
* Tests of the pre-specified comparisons (``CLS_COMPARISONS``, ``KFOLD_COMPARISONS``): DeLong or a paired
  bootstrap on the test patients, Nadeau–Bengio's corrected resampled t-test over the CV folds.
* Ground-truth-mask sensitivity analysis (the ``oracle`` stage of run_paper.py), against predicted masks.
"""
from __future__ import annotations

import torch  # noqa: F401  (load before pandas; see src/glioma_mtl/__init__.py)
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import balanced_accuracy_score

from glioma_mtl.data.metadata import TASKS
from glioma_mtl.engine import task_arrays, task_calls
from glioma_mtl.engine.common import count_parameters
from glioma_mtl.evaluation import (bootstrap_ci, brier_score, classification_metrics, concordance_correlation,
                                   corrected_resampled_ttest, delong_test, expected_calibration_error, icc_agreement,
                                   macro_auc, paired_bootstrap_test, paired_wilcoxon)
from glioma_mtl.models.segmentation import build_segmentation_model
from glioma_mtl.utils import read_history

SEG_METRICS = ["dice", "hd95", "nsd", "iou", "sensitivity", "precision"]
CLS_METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "macro_auc"]
# Pre-specified comparisons, fixed before any test was run: (A, B, question); the difference is A - B.
# End-to-end classifiers on the test patients (the seed ensembles): each rung of the ladder against the one below,
# each corrected original design against the ladder model with the same inputs.
CLS_COMPARISONS = [
    ("cls_image_clinical", "cls_image_only", "clinical token"),
    ("cls_image_clinical_radiomics", "cls_image_clinical", "radiomics token"),
    ("cls_cascade", "cls_image_clinical", "cascade design"),
    ("cls_multiscale", "cls_image_clinical", "multi-scale design"),
    ("cls_trinity", "cls_image_clinical_radiomics", "Trinity design"),
    ("cls_segmamba", "cls_image_clinical_radiomics", "SegMamba-v3 design"),
]
# Late fusion, fold by fold over the repeated CV, for these models:
KFOLD_COMPARISONS = [
    ("clinical+radiomics", "clinical", "radiomics added to clinical"),
    ("image+clinical", "clinical", "deep features added to clinical"),
    ("image+clinical+radiomics", "clinical+radiomics", "deep features added to clinical + radiomics"),
    ("radiomics", "image", "radiomics vs deep features"),
]
KFOLD_TEST_MODELS = ["xgboost", "logreg"]


def holm(pvalues: list[float]) -> list[float]:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted.tolist()


def fmt(mean: float, sd: float | None = None, lo: float | None = None, hi: float | None = None, digits: int = 3) -> str:
    if mean is None or not np.isfinite(mean):
        return "–"
    s = f"{mean:.{digits}f}"
    if sd is not None and np.isfinite(sd):
        s += f" ± {sd:.{digits}f}"
    if lo is not None and np.isfinite(lo):
        s += f" [{lo:.{digits}f}, {hi:.{digits}f}]"
    return s


def to_markdown(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def save(df: pd.DataFrame, out: Path, name: str, caption: str) -> str:
    df.to_csv(out / f"{name}.csv", index=False)
    try:
        (out / f"{name}.tex").write_text(df.to_latex(index=False, caption=caption, label=f"tab:{name}"),
                                         encoding="utf-8")
    except Exception as e:  # pandas' LaTeX writer needs jinja2
        print(f"LaTeX export skipped for {name}: {e}")
    return f"## {caption}\n\n{to_markdown(df)}\n"


def train_seconds(h: pd.DataFrame) -> tuple[pd.Series, bool]:
    """Wall time per epoch, and whether any of it is estimated.

    Epochs trained before cost logging existed have no ``epoch_seconds``. They take the log.txt interval
    to the previous epoch, or, for the first epoch of a session, the mean of the same kind of epoch (with or
    without validation).
    """
    secs = h["epoch_seconds"] if "epoch_seconds" in h else pd.Series(np.nan, index=h.index)
    estimated = bool(secs.isna().any())
    if "log_seconds" in h:
        secs = secs.fillna(h["log_seconds"])
    validated = h.filter(like="val_").notna().any(axis=1)
    return secs.fillna(secs.groupby(validated).transform("mean")).fillna(secs.mean()), estimated


def seg_params(run_dir: Path) -> float | None:
    """Parameter count (millions) from the run's saved config, for runs trained before model_info.json."""
    try:
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        model = build_segmentation_model(cfg["model"], in_channels=len(cfg["data"].get("modalities", ["T2"])))
        return count_parameters(model)["params"] / 1e6
    except Exception as e:  # e.g. SegMamba needs its CUDA extension
        print(f"[seg] no parameter count for {run_dir.name}: {e}")
        return None


def run_cost(run_dir: Path) -> dict:
    """Parameters, training time, peak GPU memory and GPU of one segmentation run, from model_info.json + history."""
    out: dict = {}
    info = run_dir / "model_info.json"
    if info.exists():
        d = json.loads(info.read_text(encoding="utf-8"))
        out["params_M"], out["gpu"] = d["params"] / 1e6, d.get("gpu")
    elif (run_dir / "config.yaml").exists():
        out["params_M"] = seg_params(run_dir)
    h = read_history(run_dir)
    if "epoch" in h:
        secs, estimated = train_seconds(h)
        if secs.notna().any():
            out["train_h"], out["train_h_estimated"] = secs.sum() / 3600, estimated
        if "peak_mem_gb" in h:
            out["peak_mem_gb"] = h["peak_mem_gb"].max()
        if "gpu" in h:  # several names = the run resumed on another GPU type, so train_h mixes speeds
            out["gpu"] = " + ".join(sorted(h["gpu"].dropna().astype(str).unique()))
    return out


def segmentation_table(plan: dict, root: Path) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    seg = plan["segmentation"]
    per_model: dict[str, list[pd.DataFrame]] = {}
    costs: dict[str, list[dict]] = {}
    for model in seg["models"]:
        runs = [root / f"{model}_s{s}" / "per_case_segmentation_test.csv" for s in seg.get("seeds", [42])]
        found = [pd.read_csv(p) for p in runs if p.exists()]
        costs[model] = [run_cost(p.parent) for p in runs if p.exists()]
        if found:
            per_model[model] = found
        else:
            print(f"[seg] no results for {model}")
    if not per_model:
        return None

    def seed_avg(frames):  # per-patient mean over seeds
        cat = pd.concat(frames).replace([np.inf, -np.inf], np.nan)
        return cat.groupby("patient_id")[SEG_METRICS].mean().reset_index()

    averaged = {m: seed_avg(f) for m, f in per_model.items()}
    # Sensitivity runs (added after the main results) are compared, but never become the reference model.
    sensitivity = {m for m in averaged if seg["models"][m].get("sensitivity")}
    best = max([m for m in averaged if m not in sensitivity] or averaged, key=lambda m: averaged[m]["dice"].mean())
    others = [m for m in averaged if m != best]
    p_dice = holm([paired_wilcoxon(averaged[m], averaged[best], "dice")["p_value"] for m in others]) if others else []
    p_hd = holm([paired_wilcoxon(averaged[m], averaged[best], "hd95")["p_value"] for m in others]) if others else []
    pvals = {m: (pd_, ph) for m, pd_, ph in zip(others, p_dice, p_hd)}

    rows = []
    for model, frames in per_model.items():
        seed_means = pd.DataFrame([f.replace([np.inf, -np.inf], np.nan)[SEG_METRICS].mean() for f in frames])
        row = {"model": model + (" (sensitivity)" if model in sensitivity else ""), "seeds": len(frames),
               "n": len(averaged[model])}
        for metric in SEG_METRICS:
            sd = seed_means[metric].std() if len(frames) > 1 else None
            if metric in ("dice", "hd95"):
                _, lo, hi = bootstrap_ci(averaged[model][metric].values)
                row[metric] = fmt(seed_means[metric].mean(), sd, lo, hi, 2 if metric == "hd95" else 3)
            else:
                row[metric] = fmt(seed_means[metric].mean(), sd, digits=3)
        p = pvals.get(model)
        row["p_dice_vs_best"] = "best" if model == best else f"{p[0]:.3g}"
        row["p_hd95_vs_best"] = "best" if model == best else f"{p[1]:.3g}"
        # Compute cost: mean over seeds; inference = median whole-volume sliding-window seconds per case
        cost = pd.DataFrame(costs[model])
        for col, digits in (("params_M", 1), ("train_h", 1), ("peak_mem_gb", 1)):
            row[col] = fmt(cost[col].astype(float).mean(), digits=digits) if col in cost else "–"
        if "train_h_estimated" in cost and cost["train_h_estimated"].eq(True).any() and row["train_h"] != "–":
            row["train_h"] = "~" + row["train_h"]  # partly reconstructed from log.txt timestamps
        seconds = pd.concat(frames)["seconds"] if all("seconds" in f for f in frames) else pd.Series(dtype=float)
        row["infer_s_per_case"] = fmt(seconds.median(), digits=2) if len(seconds) else "–"
        row["gpu"] = " + ".join(sorted(cost["gpu"].dropna().unique())) if "gpu" in cost else "–"
        rows.append(row)
    return pd.DataFrame(rows), pd.concat({m: a for m, a in averaged.items()}, names=["model"]).reset_index(0)


def experiment_seeds(cls: dict, spec: dict) -> list[int]:
    return spec.get("seeds", cls.get("seeds", [42]))


def ensemble_predictions(root: Path, exp: str, seeds, sub: str = "") -> pd.DataFrame | None:
    """Test predictions of one experiment, averaged over its seeds; ``sub`` is a run subfolder such as ``oracle``."""
    paths = [root / f"{exp}_s{s}" / sub / "predictions_test.csv" for s in seeds]
    preds = [pd.read_csv(p) for p in paths if p.exists()]
    return pd.concat(preds).groupby("patient_id").mean(numeric_only=True).reset_index() if preds else None


def classification_table(plan: dict, root: Path) -> pd.DataFrame | None:
    cls = plan.get("classification") or {}
    rows = []
    for exp, spec in (cls.get("experiments") or {}).items():
        dirs = [root / f"{exp}_s{s}" for s in experiment_seeds(cls, spec)]
        metrics = [pd.read_csv(d / "metrics_test.csv") for d in dirs if (d / "metrics_test.csv").exists()]
        if not metrics:
            print(f"[cls] no results for {exp}")
            continue
        allm = pd.concat(metrics)
        ensemble = ensemble_predictions(root, exp, experiment_seeds(cls, spec))
        for task in allm["task"].unique():
            row = {"experiment": exp + (" (leaky reference)" if spec.get("reference") else ""), "task": task,
                   "seeds": len(metrics)}
            sub = allm[allm["task"] == task]
            for metric in CLS_METRICS:
                vals = sub.loc[sub["metric"] == metric, "value"].astype(float)
                row[metric] = fmt(vals.mean(), vals.std() if len(vals) > 1 else None)
            y, prob = task_arrays(ensemble, task)
            table, _ = classification_metrics(y.astype(int), prob, [str(c) for c in TASKS[task][1]],
                                              pred=task_calls(ensemble, task))
            t = table.set_index("metric")
            for metric in ("balanced_accuracy", "macro_auc"):
                row[f"ensemble_{metric}"] = fmt(t.at[metric, "value"], lo=t.at[metric, "ci_low"], hi=t.at[metric, "ci_high"])
            row["n_test"] = int(t.at["accuracy", "n"])
            rows.append(row)
    return pd.DataFrame(rows) if rows else None


def auc_difference(y: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray, seed: int = 0) -> dict:
    """AUC(A) - AUC(B) on the same patients: DeLong's test for two classes, a paired bootstrap of the macro
    (one-vs-rest) AUC for more."""
    if prob_a.shape[1] == 2:
        out = {**delong_test(y, prob_a[:, 1], prob_b[:, 1]), "test": "DeLong"}
    else:
        out = {**paired_bootstrap_test(y, prob_a, prob_b, macro_auc, seed=seed), "test": "paired bootstrap"}
    return {**out, "auc_a": macro_auc(y, prob_a), "auc_b": macro_auc(y, prob_b)}


def paired_task_arrays(a: pd.DataFrame, b: pd.DataFrame, task: str):
    """Labels and both models' probabilities for the patients that both have, in the same order."""
    in_b = set(b["patient_id"])
    ids = [p for p in a["patient_id"] if p in in_b]
    a, b = (d.set_index("patient_id").loc[ids].reset_index() for d in (a, b))
    y, prob_a = task_arrays(a, task)
    keep = y >= 0
    return y[keep].astype(int), prob_a[keep], task_arrays(b, task)[1][keep], a[keep], b[keep]


def _tasks(*frames: pd.DataFrame) -> list[str]:
    tasks = [c[2:] for c in frames[0].columns if c.startswith("y_")]
    return [t for t in tasks if all(f"y_{t}" in f for f in frames)]


def _with_holm(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    df["p_holm"] = df.groupby(by, sort=False)["p_value"].transform(lambda p: pd.Series(holm(p.tolist()), index=p.index))
    return df


def classification_tests(plan: dict, root: Path) -> pd.DataFrame | None:
    """The pre-specified end-to-end comparisons (``CLS_COMPARISONS``) on the test patients, Holm within each task."""
    cls = plan.get("classification") or {}
    exps = cls.get("experiments") or {}
    ens = {e: ensemble_predictions(root, e, experiment_seeds(cls, s)) for e, s in exps.items()}
    rows = []
    for a, b, question in CLS_COMPARISONS:
        if ens.get(a) is None or ens.get(b) is None:
            continue
        for task in _tasks(ens[a], ens[b]):
            y, pa, pb, _, _ = paired_task_arrays(ens[a], ens[b], task)
            rows.append({"comparison": question, "a": a, "b": b, "task": task, **auc_difference(y, pa, pb)})
    if not rows:
        return None
    df = _with_holm(pd.DataFrame(rows), ["task"])
    return pd.DataFrame({
        "task": df["task"], "comparison": df["comparison"], "A": df["a"], "B": df["b"],
        "auc_A": df["auc_a"].map("{:.3f}".format), "auc_B": df["auc_b"].map("{:.3f}".format),
        "delta_auc": [fmt(d, lo=lo, hi=hi) for d, lo, hi in zip(df["mean_diff"], df["ci_low"], df["ci_high"])],
        "test": df["test"], "p": df["p_value"].map("{:.3g}".format), "p_holm": df["p_holm"].map("{:.3g}".format),
        "n": df["n"]}).sort_values(["task"], kind="stable")


def classification_oracle_table(plan: dict, root: Path) -> pd.DataFrame | None:
    """Each end-to-end classifier with ground-truth masks (``<run>/oracle/``) against predicted masks, same patients."""
    cls = plan.get("classification") or {}
    rows = []
    for exp, spec in (cls.get("experiments") or {}).items():
        seeds = experiment_seeds(cls, spec)
        pred, oracle = ensemble_predictions(root, exp, seeds), ensemble_predictions(root, exp, seeds, "oracle")
        if pred is None or oracle is None:
            continue
        for task in _tasks(oracle, pred):
            y, po, pp, o_df, p_df = paired_task_arrays(oracle, pred, task)
            bal = {k: balanced_accuracy_score(y, task_calls(d, task)) for k, d in (("gt", o_df), ("pred", p_df))}
            rows.append({"experiment": exp + (" (leaky reference)" if spec.get("reference") else ""), "task": task,
                         **auc_difference(y, po, pp), "bal_acc_pred": bal["pred"], "bal_acc_gt": bal["gt"]})
    if not rows:
        return None
    df = _with_holm(pd.DataFrame(rows), ["task"])
    return pd.DataFrame({
        "experiment": df["experiment"], "task": df["task"],
        "auc_predicted_masks": df["auc_b"].map("{:.3f}".format), "auc_gt_masks": df["auc_a"].map("{:.3f}".format),
        "delta_auc": [fmt(d, lo=lo, hi=hi) for d, lo, hi in zip(df["mean_diff"], df["ci_low"], df["ci_high"])],
        "test": df["test"], "p": df["p_value"].map("{:.3g}".format), "p_holm": df["p_holm"].map("{:.3g}".format),
        "bal_acc_predicted_masks": df["bal_acc_pred"].map("{:.3f}".format),
        "bal_acc_gt_masks": df["bal_acc_gt"].map("{:.3f}".format), "n": df["n"]})


def _fold_scores(path: Path, metric: str = "macro_auc") -> pd.DataFrame | None:
    if not path.exists():
        return None
    folds = pd.read_csv(path)
    return folds[(folds["metric"] == metric) & folds["model"].isin(KFOLD_TEST_MODELS)]


def _fold_ttest(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Corrected resampled t-test of two fold-score tables of the same task (paired on the fold index)."""
    m = a.merge(b, on="fold", suffixes=("_a", "_b"))
    ratio = m["n_test_a"].mean() / m["n_train_a"].mean()
    return {"auc_a": m["value_a"].mean(), "auc_b": m["value_b"].mean(),
            **corrected_resampled_ttest(m["value_a"], m["value_b"], ratio)}


def kfold_tests(root: Path) -> pd.DataFrame | None:
    """The pre-specified late-fusion comparisons (``KFOLD_COMPARISONS``), Holm within each task and model."""
    folds = _fold_scores(root / "kfold_cv" / "cv_folds.csv")
    if folds is None:
        return None
    rows = []
    for (task, model), g in folds.groupby(["task", "model"], sort=False):
        by_set = {fs: s for fs, s in g.groupby("features")}
        for a, b, question in KFOLD_COMPARISONS:
            if a in by_set and b in by_set:
                rows.append({"task": task, "model": model, "comparison": question, "A": a, "B": b,
                             **_fold_ttest(by_set[a], by_set[b])})
    if not rows:
        return None
    df = _with_holm(pd.DataFrame(rows), ["task", "model"])
    return pd.DataFrame({
        "task": df["task"], "model": df["model"], "comparison": df["comparison"],
        "auc_A": df["auc_a"].map("{:.3f}".format), "auc_B": df["auc_b"].map("{:.3f}".format),
        "delta_auc": df["mean_diff"].map("{:+.3f}".format), "t": df["t"].map("{:.2f}".format),
        "p": df["p_value"].map("{:.3g}".format), "p_holm": df["p_holm"].map("{:.3g}".format),
        "folds": df["n_folds"]})


def kfold_oracle_table(root: Path) -> pd.DataFrame | None:
    """Late fusion on ground-truth-mask features (``kfold_cv_oracle/``) against predicted-mask features, same folds."""
    pred, oracle = _fold_scores(root / "kfold_cv" / "cv_folds.csv"), _fold_scores(root / "kfold_cv_oracle" / "cv_folds.csv")
    if pred is None or oracle is None:
        return None
    rows = []
    for (task, fs, model), g in oracle.groupby(["task", "features", "model"], sort=False):
        p = pred[(pred["task"] == task) & (pred["features"] == fs) & (pred["model"] == model)]
        if len(p):
            rows.append({"task": task, "features": fs, "model": model, **_fold_ttest(g, p)})
    if not rows:
        return None
    df = _with_holm(pd.DataFrame(rows), ["task", "model"])
    return pd.DataFrame({
        "task": df["task"], "features": df["features"], "model": df["model"],
        "auc_predicted_masks": df["auc_b"].map("{:.3f}".format), "auc_gt_masks": df["auc_a"].map("{:.3f}".format),
        "delta_auc": df["mean_diff"].map("{:+.3f}".format), "p": df["p_value"].map("{:.3g}".format),
        "p_holm": df["p_holm"].map("{:.3g}".format), "folds": df["n_folds"]})


def seed_averaged_cases(plan: dict, root: Path) -> dict[str, pd.DataFrame]:
    """Per segmentation model: per-patient test Dice and ground-truth tumour volume, averaged over seeds."""
    seg, out = plan["segmentation"], {}
    for model in seg["models"]:
        paths = [root / f"{model}_s{s}" / "per_case_segmentation_test.csv" for s in seg.get("seeds", [42])]
        frames = [f for f in (pd.read_csv(p) for p in paths if p.exists()) if "gt_volume_ml" in f]
        if frames:
            cat = pd.concat(frames).replace([np.inf, -np.inf], np.nan)
            out[model] = cat.groupby("patient_id")[["dice", "gt_volume_ml"]].mean().reset_index()
    return out


def segmentation_volume_table(plan: dict, root: Path) -> tuple[pd.DataFrame, np.ndarray] | None:
    """Does Dice depend on tumour size? Spearman correlation, and Dice per volume tertile (tertiles of the test
    patients' ground-truth volumes)."""
    from scipy.stats import spearmanr

    cases = seed_averaged_cases(plan, root)
    if not cases:
        return None
    edges = next(iter(cases.values()))["gt_volume_ml"].quantile([1 / 3, 2 / 3]).to_numpy()
    rows = []
    for model, d in cases.items():
        rho, p = spearmanr(d["gt_volume_ml"], d["dice"], nan_policy="omit")
        tertile = np.digitize(d["gt_volume_ml"], edges)
        row = {"model": model, "n": len(d), "spearman_rho": f"{rho:.2f}", "p": f"{p:.3g}"}
        for i, name in enumerate(("small", "medium", "large")):
            dice = d.loc[tertile == i, "dice"]
            row[f"median_dice_{name}"] = f"{dice.median():.3f}"
            row[f"n_dice_below_0.5_{name}"] = int((dice < 0.5).sum())
        rows.append(row)
    return pd.DataFrame(rows), edges


def _calibration_row(y: np.ndarray, prob: np.ndarray) -> dict:
    return {"n": len(y), "brier": f"{brier_score(y, prob):.3f}", "ece": f"{expected_calibration_error(y, prob):.3f}"}


def calibration_tables(plan: dict, root: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Brier score and ECE: end-to-end models on the test patients (seed ensembles), and late fusion on its
    out-of-fold predictions (all 162 patients, ``kfold_cv/oof_predictions.csv``)."""
    cls = plan.get("classification") or {}
    e2e = []
    for exp, spec in (cls.get("experiments") or {}).items():
        ens = ensemble_predictions(root, exp, experiment_seeds(cls, spec))
        if ens is None:
            continue
        for task in _tasks(ens):
            y, prob = task_arrays(ens, task)
            keep = y >= 0
            e2e.append({"experiment": exp + (" (leaky reference)" if spec.get("reference") else ""), "task": task,
                        **_calibration_row(y[keep].astype(int), prob[keep])})
    late = []
    oof_path = root / "kfold_cv" / "oof_predictions.csv"
    if oof_path.exists():
        oof = pd.read_csv(oof_path)
        for (task, features, model), g in oof.groupby(["task", "features", "model"], sort=False):
            prob = g[[f"p_{k}" for k in range(len(TASKS[task][1]))]].to_numpy()
            late.append({"task": task, "features": features, "model": model,
                         **_calibration_row(g["y"].to_numpy(int), prob)})
    return (pd.DataFrame(e2e) if e2e else None), (pd.DataFrame(late) if late else None)


def learning_curve_table(root: Path) -> pd.DataFrame | None:
    """Mean ± SD of the macro AUC over the test folds, per fraction of the training folds."""
    path = root / "learning_curves" / "learning_curves.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    s = df.groupby(["task", "features", "model", "fraction"], sort=False).agg(
        n_train=("n_train", "mean"), mean=("macro_auc", "mean"), sd=("macro_auc", "std")).reset_index()
    s["v"] = [f"{fmt(m, sd)} (n≈{n:.0f})" for m, sd, n in zip(s["mean"], s["sd"], s["n_train"])]
    s["fraction"] = (s["fraction"] * 100).round().astype(int).astype(str) + "%"
    return s.pivot_table(index=["task", "features", "model"], columns="fraction", values="v", aggfunc="first",
                         sort=False).reset_index()


def radiomics_stability(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, str] | None:
    """Agreement of every radiomics feature between predicted and expert masks (ICC(2,1) and Lin's CCC).

    Only classification validation and test patients count: the backbone was trained on the training patients'
    masks, so their predicted masks are nearly the expert ones and would flatter the agreement.
    """
    from glioma_mtl.radiomics import load_radiomics

    pred_path, gt_path = root / "radiomics" / "radiomics_pred.csv", root / "radiomics_label" / "radiomics_label.csv"
    if not (pred_path.exists() and gt_path.exists()):
        return None
    pred, gt = load_radiomics(pred_path), load_radiomics(gt_path)
    ids = sorted(set(pred.index) & set(gt.index))
    patients = "all cohort patients"
    split_file = root / "splits_seed42.json"
    if split_file.exists():
        held_out = set().union(*(json.loads(split_file.read_text())["classification"][p] for p in ("val", "test")))
        ids, patients = [p for p in ids if p in held_out], "classification validation + test patients"
    rows = []
    for col in [c for c in pred.columns if c in gt.columns]:
        a, b = pred.loc[ids, col].to_numpy(float), gt.loc[ids, col].to_numpy(float)
        if np.nanstd(a) == 0 and np.nanstd(b) == 0:
            continue
        rows.append({"feature": col, "feature_class": col.split("_")[1] if col.count("_") >= 2 else "other",
                     "icc": icc_agreement(a, b), "ccc": concordance_correlation(a, b),
                     "n": int((np.isfinite(a) & np.isfinite(b)).sum())})
    per_feature = pd.DataFrame(rows).sort_values("icc")
    summary = per_feature.groupby("feature_class").agg(
        n_features=("icc", "size"), median_icc=("icc", "median"), q1=("icc", lambda v: v.quantile(0.25)),
        q3=("icc", lambda v: v.quantile(0.75)), good=("icc", lambda v: (v >= 0.75).mean()),
        excellent=("icc", lambda v: (v >= 0.9).mean())).reset_index()
    summary = pd.DataFrame({
        "feature_class": summary["feature_class"], "n_features": summary["n_features"],
        "median_icc": [f"{m:.2f} [{a:.2f}, {b:.2f}]" for m, a, b in zip(summary["median_icc"], summary["q1"], summary["q3"])],
        "share_icc_0.75+": summary["good"].map("{:.0%}".format),
        "share_icc_0.9+": summary["excellent"].map("{:.0%}".format)})
    return per_feature, summary, patients


def kfold_tables(root: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    cv_path, ho_path = root / "kfold_cv" / "cv_results.csv", root / "kfold_cv" / "holdout_results.csv"
    cv = ho = None
    if cv_path.exists():
        df = pd.read_csv(cv_path)
        df["v"] = [fmt(m, s) for m, s in zip(df["mean"], df["sd"])]
        cv = df.pivot_table(index=["task", "features", "model"], columns="metric", values="v", aggfunc="first",
                            sort=False).reset_index()[["task", "features", "model", *CLS_METRICS]]
    if ho_path.exists():
        df = pd.read_csv(ho_path)
        df["v"] = [fmt(v, lo=lo, hi=hi) for v, lo, hi in zip(df["value"], df["ci_low"], df["ci_high"])]
        ho = df.pivot_table(index=["task", "features", "model"], columns="metric", values="v", aggfunc="first",
                            sort=False).reset_index()[["task", "features", "model", *CLS_METRICS]]
    return cv, ho


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with open(args.plan, encoding="utf-8") as f:
        plan = yaml.safe_load(f)
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sections = ["# Paper tables\n"]
    seg = segmentation_table(plan, root)
    if seg is not None:
        seg[1].to_csv(out / "segmentation_per_patient_seed_avg.csv", index=False)
        sections.append(save(seg[0], out, "segmentation", "Segmentation on the test split (mean ± SD over seeds "
                                                           "[95% bootstrap CI]; Wilcoxon vs best, Holm-corrected; "
                                                           "~ = training time partly estimated from log timestamps; "
                                                           "(sensitivity) = run added after the main results, "
                                                           "never the reference)"))
    cls = classification_table(plan, root)
    if cls is not None:
        sections.append(save(cls, out, "classification", "Classification on the held-out test split "
                                                         "(mean ± SD over seeds; ensemble [95% bootstrap CI])"))
    cv, ho = kfold_tables(root)
    if cv is not None:
        sections.append(save(cv, out, "kfold_cv", "Late fusion, repeated stratified k-fold CV (mean ± SD over folds)"))
    if ho is not None:
        sections.append(save(ho, out, "kfold_holdout", "Late fusion fit on train+val, held-out test [95% bootstrap CI]"))
    for table, name, caption in (
            (classification_tests(plan, root), "classification_tests",
             "End-to-end classifiers, pre-specified comparisons on the test split (Δ = A − B [95% CI]; DeLong for "
             "two classes, paired bootstrap of the macro AUC otherwise; Holm within each task)"),
            (kfold_tests(root), "kfold_tests",
             "Late fusion, pre-specified comparisons of macro AUC over the CV folds (Nadeau–Bengio corrected "
             "resampled t-test; Holm within each task and model)"),
            (classification_oracle_table(plan, root), "classification_oracle",
             "End-to-end classifiers with ground-truth masks (crop centre, radiomics) instead of predicted ones, "
             "same trained models and test patients (Δ = GT − predicted [95% CI]; Holm within each task)"),
            (kfold_oracle_table(root), "kfold_oracle",
             "Late fusion with ground-truth-mask features (radiomics, deep features) instead of predicted-mask "
             "features, CV macro AUC on the same folds (Δ = GT − predicted; corrected t-test; Holm within each "
             "task and model)")):
        if table is not None:
            sections.append(save(table, out, name, caption))
    volume = segmentation_volume_table(plan, root)
    if volume is not None:
        sections.append(save(volume[0], out, "segmentation_by_volume",
                             "Segmentation Dice against ground-truth tumour volume on the test split (Spearman; "
                             f"median Dice per volume tertile, edges {volume[1][0]:.0f} and {volume[1][1]:.0f} mL)"))
    e2e_cal, late_cal = calibration_tables(plan, root)
    if e2e_cal is not None:
        sections.append(save(e2e_cal, out, "calibration_end_to_end",
                             "Calibration of the end-to-end classifiers on the test split (multi-class Brier score, "
                             "0 = perfect, uniform probabilities score 0.5 / 0.667 / 0.75 for 2 / 3 / 4 classes; "
                             "ECE, 10 bins: P(class 1) for two classes, top label otherwise)"))
    if late_cal is not None:
        sections.append(save(late_cal, out, "calibration_late_fusion",
                             "Calibration of late fusion on out-of-fold predictions of all cohort patients "
                             "(Brier score and ECE as above)"))
    curves = learning_curve_table(root)
    if curves is not None:
        sections.append(save(curves, out, "learning_curves",
                             "Late fusion, macro AUC (mean ± SD over the test folds) when fit on a stratified "
                             "fraction of each training fold (n = mean training patients)"))
    stability = radiomics_stability(root)
    if stability is not None:
        stability[0].to_csv(out / "radiomics_stability_per_feature.csv", index=False)
        sections.append(save(stability[1], out, "radiomics_stability",
                             f"Radiomics agreement between predicted and expert masks on the {stability[2]} "
                             "(ICC(2,1), median [IQR] per feature class; good ≥ 0.75, excellent ≥ 0.9)"))
    (out / "summary.md").write_text("\n".join(sections), encoding="utf-8")
    print("\n".join(sections))
    print(f"Saved tables to {out}")


if __name__ == "__main__":
    main()
