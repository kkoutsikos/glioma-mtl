"""Collect all paper runs into tables (CSV + Markdown + LaTeX).

    python scripts/aggregate_results.py --plan configs/paper.yaml --root runs/paper --out runs/paper/paper_tables

* Segmentation: mean ± SD of the per-seed test means; 95% bootstrap CI over patients (seed-averaged);
  paired Wilcoxon of every model vs. the best one (Holm-corrected); parameters, training time, peak GPU
  memory, inference time per case and the GPU used. "~" marks a training time partly reconstructed from
  log.txt timestamps (epochs trained before cost logging existed).
* Classification: mean ± SD over seeds, plus the seed ensemble (averaged probabilities) with a 95%
  bootstrap CI over the 25 test patients.
* K-fold late fusion: mean ± SD over the CV folds, and the one-shot hold-out result with its CI.
"""
from __future__ import annotations

import torch  # noqa: F401  (load before pandas; see src/glioma_mtl/__init__.py)
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from glioma_mtl.data.metadata import TASKS
from glioma_mtl.engine import task_arrays
from glioma_mtl.engine.common import count_parameters
from glioma_mtl.evaluation import bootstrap_ci, classification_metrics, paired_wilcoxon
from glioma_mtl.models.segmentation import build_segmentation_model
from glioma_mtl.utils import read_history

SEG_METRICS = ["dice", "hd95", "nsd", "iou", "sensitivity", "precision"]
CLS_METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "macro_auc"]


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
    best = max(averaged, key=lambda m: averaged[m]["dice"].mean())
    others = [m for m in averaged if m != best]
    p_dice = holm([paired_wilcoxon(averaged[m], averaged[best], "dice")["p_value"] for m in others]) if others else []
    p_hd = holm([paired_wilcoxon(averaged[m], averaged[best], "hd95")["p_value"] for m in others]) if others else []
    pvals = {m: (pd_, ph) for m, pd_, ph in zip(others, p_dice, p_hd)}

    rows = []
    for model, frames in per_model.items():
        seed_means = pd.DataFrame([f.replace([np.inf, -np.inf], np.nan)[SEG_METRICS].mean() for f in frames])
        row = {"model": model, "seeds": len(frames), "n": len(averaged[model])}
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


def classification_table(plan: dict, root: Path) -> pd.DataFrame | None:
    cls = plan.get("classification") or {}
    rows = []
    for exp, spec in (cls.get("experiments") or {}).items():
        dirs = [root / f"{exp}_s{s}" for s in cls.get("seeds", [42])]
        metrics = [pd.read_csv(d / "metrics_test.csv") for d in dirs if (d / "metrics_test.csv").exists()]
        preds = [pd.read_csv(d / "predictions_test.csv") for d in dirs if (d / "predictions_test.csv").exists()]
        if not metrics:
            print(f"[cls] no results for {exp}")
            continue
        allm = pd.concat(metrics)
        ensemble = pd.concat(preds).groupby("patient_id").mean(numeric_only=True).reset_index()
        for task in allm["task"].unique():
            row = {"experiment": exp + (" (leaky reference)" if spec.get("reference") else ""), "task": task,
                   "seeds": len(metrics)}
            sub = allm[allm["task"] == task]
            for metric in CLS_METRICS:
                vals = sub.loc[sub["metric"] == metric, "value"].astype(float)
                row[metric] = fmt(vals.mean(), vals.std() if len(vals) > 1 else None)
            y, prob = task_arrays(ensemble, task)
            table, _ = classification_metrics(y.astype(int), prob, [str(c) for c in TASKS[task][1]])
            t = table.set_index("metric")
            for metric in ("balanced_accuracy", "macro_auc"):
                row[f"ensemble_{metric}"] = fmt(t.at[metric, "value"], lo=t.at[metric, "ci_low"], hi=t.at[metric, "ci_high"])
            row["n_test"] = int(t.at["accuracy", "n"])
            rows.append(row)
    return pd.DataFrame(rows) if rows else None


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
                                                           "~ = training time partly estimated from log timestamps)"))
    cls = classification_table(plan, root)
    if cls is not None:
        sections.append(save(cls, out, "classification", "Classification on the held-out test split "
                                                         "(mean ± SD over seeds; ensemble [95% bootstrap CI])"))
    cv, ho = kfold_tables(root)
    if cv is not None:
        sections.append(save(cv, out, "kfold_cv", "Late fusion, repeated stratified k-fold CV (mean ± SD over folds)"))
    if ho is not None:
        sections.append(save(ho, out, "kfold_holdout", "Late fusion fit on train+val, held-out test [95% bootstrap CI]"))
    (out / "summary.md").write_text("\n".join(sections), encoding="utf-8")
    print("\n".join(sections))
    print(f"Saved tables to {out}")


if __name__ == "__main__":
    main()
