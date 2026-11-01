"""Paper figures from finished runs.

    python scripts/make_figures.py --plan configs/paper.yaml --root runs/paper --out runs/paper/paper_figures
    python scripts/make_figures.py ... --qualitative     # + segmentation overlays (needs the images and a GPU/CPU)

Produces (whatever inputs exist):
  curves/<run>.png                       training curves of every run (SSL, segmentation, classification)
  segmentation_dice.png, _hd95.png        per-patient test distributions per model (seed-averaged)
  segmentation_cases.png                  best / median / worst test case, every model (--qualitative)
  <experiment>_<task>_confusion.png, _roc.png   seed-ensemble on the held-out test split
  kfold_<task>_<metric>.png               repeated k-fold CV, mean ± SD
  kfold_shap_<task>.png                   XGBoost TreeSHAP importance per modality on the hold-out test
  explain/<experiment>_*.png              attention / occlusion summaries copied from the explain stage
"""
from __future__ import annotations

import torch  # noqa: F401  (load before pandas; see src/glioma_mtl/__init__.py)
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import confusion_matrix

from glioma_mtl import plotting
from glioma_mtl.data.metadata import TASKS
from glioma_mtl.engine import task_arrays


def training_curves(root: Path, out: Path) -> None:
    for history in sorted(root.glob("*/history.csv")):
        plotting.plot_history(history, out / "curves" / f"{history.parent.name}.png", history.parent.name)


def segmentation_figures(plan: dict, root: Path, out: Path) -> dict[str, pd.DataFrame]:
    seg = plan["segmentation"]
    averaged = {}
    for model in seg["models"]:
        frames = [pd.read_csv(p) for s in seg.get("seeds", [42])
                  if (p := root / f"{model}_s{s}" / "per_case_segmentation_test.csv").exists()]
        if frames:
            cat = pd.concat(frames).replace([np.inf, -np.inf], np.nan)
            averaged[model] = cat.groupby("patient_id")[["dice", "hd95"]].mean().reset_index()
    if averaged:
        plotting.plot_box({m: d["dice"] for m, d in averaged.items()}, out / "segmentation_dice.png", "Dice",
                          "Segmentation test Dice per patient")
        plotting.plot_box({m: d["hd95"] for m, d in averaged.items()}, out / "segmentation_hd95.png", "HD95 (mm)",
                          "Segmentation test HD95 per patient")
    return averaged


def qualitative_cases(plan: dict, root: Path, out: Path, averaged: dict[str, pd.DataFrame], n_cases: int = 3) -> None:
    """Re-run every segmentation model (first seed) on the best / median / worst case of the best model."""
    from monai.inferers import sliding_window_inference

    from glioma_mtl.config import load_config
    from glioma_mtl.data.files import build_records
    from glioma_mtl.data.transforms import seg_eval_transforms
    from glioma_mtl.experiment import load_context
    from glioma_mtl.models import build_segmentation_model
    from glioma_mtl.utils import get_device, load_weights

    if not averaged:
        return
    best = max(averaged, key=lambda m: averaged[m]["dice"].mean())
    ranked = averaged[best].sort_values("dice", ascending=False).reset_index(drop=True)
    picks = ranked.iloc[[0, len(ranked) // 2, len(ranked) - 1][:n_cases]]["patient_id"].tolist()
    seed = plan["segmentation"].get("seeds", [42])[0]
    device = get_device("auto")
    cases = {pid: {"case": pid, "preds": {}, "dice": {}} for pid in picks}
    for model_name in averaged:
        run = root / f"{model_name}_s{seed}"
        if not (run / "best.pt").exists() or not (run / "config.yaml").exists():
            continue
        cfg = load_config(run / "config.yaml")
        try:
            model = build_segmentation_model(cfg["model"], in_channels=len(cfg["data"].get("modalities", ["T2"])))
        except ImportError as e:  # e.g. SegMamba not built in this session
            print(f"[qualitative] skipping {model_name}: {e}")
            continue
        load_weights(model, run / "best.pt", min_fraction=1.0)
        model.to(device).eval()
        ctx = load_context(cfg)
        transform = seg_eval_transforms(cfg["data"])
        for rec in build_records(picks, ctx.files):
            item = transform(rec)
            with torch.no_grad():
                logits = sliding_window_inference(item["image"][None].to(device), cfg["data"]["roi_size"], 2, model,
                                                  overlap=0.5, mode="gaussian")
            pred = (torch.sigmoid(logits.float()) > 0.5)[0, 0].cpu().numpy()
            case = cases[rec["patient_id"]]
            case.setdefault("image", item["image"][0].numpy())
            case.setdefault("gt", item["label"][0].numpy() > 0.5)
            case["preds"][model_name] = pred
            row = averaged[model_name].set_index("patient_id")
            case["dice"][model_name] = float(row.at[rec["patient_id"], "dice"]) if rec["patient_id"] in row.index else None
        del model
    ready = [c for c in cases.values() if "image" in c]
    if ready:
        plotting.plot_seg_cases(ready, out / "segmentation_cases.png",
                                f"Best / median / worst test case of {best}, all models")


def classification_figures(plan: dict, root: Path, out: Path) -> None:
    cls = plan.get("classification") or {}
    for exp in cls.get("experiments") or {}:
        preds = [pd.read_csv(p) for s in cls.get("seeds", [42])
                 if (p := root / f"{exp}_s{s}" / "predictions_test.csv").exists()]
        if not preds:
            continue
        ens = pd.concat(preds).groupby("patient_id").mean(numeric_only=True).reset_index()
        for task in [c[2:] for c in ens.columns if c.startswith("y_")]:
            y, prob = task_arrays(ens, task)
            y = y.astype(int)
            keep = y >= 0
            classes = [str(c) for c in TASKS[task][1]]
            cm = confusion_matrix(y[keep], prob[keep].argmax(1), labels=list(range(len(classes))))
            title = f"{exp}: {task} (test, {len(preds)}-seed ensemble, n={keep.sum()})"
            plotting.plot_confusion(cm, classes, out / f"{exp}_{task}_confusion.png", title)
            plotting.plot_roc(y[keep], prob[keep], classes, out / f"{exp}_{task}_roc.png", title)
        explain = root / f"{exp}_s{cls.get('seeds', [42])[0]}" / "explain"
        for png in explain.glob("*.png"):
            (out / "explain").mkdir(parents=True, exist_ok=True)
            shutil.copy(png, out / "explain" / f"{exp}_{png.name}")


def kfold_figures(root: Path, out: Path) -> None:
    cv_path = root / "kfold_cv" / "cv_results.csv"
    if cv_path.exists():
        cv = pd.read_csv(cv_path)
        for (task, metric), sub in cv[cv["metric"].isin(["balanced_accuracy", "macro_auc"])].groupby(["task", "metric"]):
            sub = sub.sort_values("mean", ascending=False)
            plotting.plot_bars(sub["features"] + " | " + sub["model"], sub["mean"], sub["sd"],
                               out / f"kfold_{task}_{metric}.png",
                               f"{task}: repeated k-fold CV ({metric}, mean ± SD over folds)", metric)
    shap_path = root / "kfold_cv" / "holdout_shap.csv"
    if shap_path.exists():
        shap = pd.read_csv(shap_path)
        for (task, feats), sub in shap.groupby(["task", "features"]):
            if sub["group"].nunique() < 2:
                continue
            groups = sub.groupby("group")["mean_abs_shap"].sum().sort_values(ascending=False)
            safe = feats.replace("+", "_").replace(" ", "")
            plotting.plot_bars(groups.index, groups.values, None, out / f"kfold_shap_{task}_{safe}_groups.png",
                               f"{task}, XGBoost on {feats}: TreeSHAP importance per modality (hold-out)",
                               "sum of mean |SHAP|")
            top = sub.sort_values("mean_abs_shap", ascending=False).head(15)
            plotting.plot_bars(top["feature"], top["mean_abs_shap"], None, out / f"kfold_shap_{task}_{safe}_top15.png",
                               f"{task}, XGBoost on {feats}: top-15 features (hold-out)", "mean |SHAP|")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--qualitative", action="store_true", help="segmentation overlays (loads models and images)")
    args = parser.parse_args()
    with open(args.plan, encoding="utf-8") as f:
        plan = yaml.safe_load(f)
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    training_curves(root, out)
    averaged = segmentation_figures(plan, root, out)
    if args.qualitative:
        qualitative_cases(plan, root, out, averaged)
    classification_figures(plan, root, out)
    kfold_figures(root, out)
    print(f"Saved {len(list(out.rglob('*.png')))} figures to {out}")


if __name__ == "__main__":
    main()
