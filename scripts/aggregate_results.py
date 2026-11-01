"""Collect all paper runs into tables (CSV + Markdown + LaTeX).

    python scripts/aggregate_results.py --plan configs/paper.yaml --root runs/paper --out runs/paper/paper_tables

* Segmentation: mean ± SD of the per-seed test means; 95% bootstrap CI over patients (seed-averaged);
  paired Wilcoxon of every model vs. the best one (Holm-corrected).
* Classification: mean ± SD over seeds, plus the seed ensemble (averaged probabilities) with a 95%
  bootstrap CI over the 25 test patients.
* K-fold late fusion: mean ± SD over the CV folds, and the one-shot hold-out result with its CI.
"""
from __future__ import annotations

import torch  # noqa: F401  (load before pandas; see src/glioma_mtl/__init__.py)
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from glioma_mtl.data.metadata import TASKS
from glioma_mtl.engine import task_arrays
from glioma_mtl.evaluation import bootstrap_ci, classification_metrics, paired_wilcoxon

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


def segmentation_table(plan: dict, root: Path) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    seg = plan["segmentation"]
    per_model: dict[str, list[pd.DataFrame]] = {}
    for model in seg["models"]:
        runs = [root / f"{model}_s{s}" / "per_case_segmentation_test.csv" for s in seg.get("seeds", [42])]
        found = [pd.read_csv(p) for p in runs if p.exists()]
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
                                                           "[95% bootstrap CI]; Wilcoxon vs best, Holm-corrected)"))
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
