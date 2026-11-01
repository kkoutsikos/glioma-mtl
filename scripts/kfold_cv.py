"""Repeated stratified k-fold CV of tabular / late-fusion classifiers (XGBoost, logistic regression, RF).

    python scripts/kfold_cv.py --config configs/kfold_cv.yaml
    python scripts/kfold_cv.py --config configs/kfold_cv.yaml --set deep_features.csv=runs/deep_features/deep_features_pred.csv \
        radiomics.csv=runs/radiomics/radiomics_pred.csv --holdout

Feature blocks: clinical groups (demographics, ...), radiomics, and label-free deep features from the
segmentation backbone (``scripts/extract_deep_features.py``). Every imputer / scaler / ``SelectKBest``
sits inside the pipeline, so it is refit on each training fold.

``--holdout`` also fits each pipeline once on classification train+val and evaluates it on the
classification test split, which no model in the pipeline has seen. Feature sets marked
``(leaky reference)`` use tissue- or surgery-derived variables; they are context, not a prediction model.
"""
from __future__ import annotations

import torch  # noqa: F401  (load before pandas; see src/glioma_mtl/__init__.py)
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from glioma_mtl.config import load_config
from glioma_mtl.data import load_metadata, load_splits
from glioma_mtl.data.metadata import TASKS, clinical_transformer_parts, feature_columns, normalize_patient_id
from glioma_mtl.evaluation import classification_metrics
from glioma_mtl.features import load_deep_features
from glioma_mtl.radiomics import TopK, load_radiomics

# name -> (clinical groups, radiomics?, deep?)
FEATURE_SETS = {
    "demographics": (["demographics"], False, False),
    "radiomics": ([], True, False),
    "demographics+radiomics": (["demographics"], True, False),
    "deep": ([], False, True),
    "deep+demographics": (["demographics"], False, True),
    "deep+demographics+radiomics": (["demographics"], True, True),
    "molecular+surgical (leaky reference)": (["molecular", "surgical"], False, False),
    "thesis clinical vector (leaky reference)": (["demographics", "history", "molecular", "surgical"], False, False),
}
SCORING = {"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "macro_f1": "f1_macro",
           "macro_auc": "roc_auc_ovr"}


def make_models(cfg: dict) -> dict:
    models = {
        "majority": lambda: DummyClassifier(strategy="most_frequent"),
        "logreg": lambda: LogisticRegression(max_iter=5000, class_weight="balanced", C=cfg.get("logreg_C", 0.5)),
        "random_forest": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                        class_weight="balanced", random_state=0, n_jobs=-1),
    }
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("xgboost not installed; skipping it (pip install xgboost)")
    else:
        xgb = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8,
               "colsample_bytree": 0.8, **(cfg.get("xgboost") or {})}  # thesis: 200 trees, depth 4, lr 0.05
        models["xgboost"] = lambda: XGBClassifier(**xgb, tree_method="hist", random_state=0, n_jobs=-1)
    wanted = cfg.get("models")
    return {k: v for k, v in models.items() if wanted is None or k in wanted or k == "majority"}


def _block(cols: list[str], k: int) -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("var", VarianceThreshold()),
                     ("scale", StandardScaler()), ("select", TopK(k))])


def build_pipeline(groups, rad_cols, deep_cols, k_rad, k_deep, model) -> Pipeline:
    parts = clinical_transformer_parts(groups) if groups else []
    if rad_cols:
        parts.append(("rad", _block(rad_cols, k_rad), rad_cols))
    if deep_cols:
        parts.append(("deep", _block(deep_cols, k_deep), deep_cols))
    return Pipeline([("features", ColumnTransformer(parts)), ("clf", model)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--holdout", action="store_true", help="also fit on train+val and score the test split")
    parser.add_argument("--out-dir", default=None, help="default: <output_root>/kfold_cv")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    d, cv_cfg = cfg["data"], cfg.get("cv") or {}

    meta = load_metadata(d["metadata_csv"])
    splits = load_splits(d["splits_file"]) if Path(d["splits_file"]).exists() else None
    if splits:
        cohort = sorted(set().union(*splits["classification"].values()))
    else:
        cohort = sorted(pd.read_csv(d["cohort_csv"])["ID"].map(normalize_patient_id))
    df = meta.loc[cohort].copy()

    blocks: dict[str, list[str]] = {"rad": [], "deep": []}
    for key, loader, name in (("radiomics", load_radiomics, "rad"), ("deep_features", load_deep_features, "deep")):
        path = (cfg.get(key) or {}).get("csv")
        if path:
            extra = loader(path)
            missing = sorted(set(df.index) - set(extra.index))
            if missing:
                raise SystemExit(f"{path} lacks {len(missing)} cohort patients, e.g. {missing[:3]}")
            df = df.join(extra)
            blocks[name] = list(extra.columns)
    k_rad = (cfg.get("radiomics") or {}).get("k", 30)
    k_deep = (cfg.get("deep_features") or {}).get("k", 32)
    print(f"Cohort: {len(df)} patients | radiomics: {len(blocks['rad'])} | deep: {len(blocks['deep'])} features")

    models = make_models(cv_cfg)
    wanted_sets = cv_cfg.get("feature_sets")
    cv = RepeatedStratifiedKFold(n_splits=cv_cfg.get("folds", 5), n_repeats=cv_cfg.get("repeats", 10),
                                 random_state=cfg.get("seed", 42))
    out_dir = Path(args.out_dir or Path(cfg.get("output_root", "runs")) / "kfold_cv")
    out_dir.mkdir(parents=True, exist_ok=True)

    cv_rows, holdout_rows = [], []
    for task in cfg["tasks"]:
        label_col, classes = TASKS[task]
        sub = df[df[label_col] >= 0]
        y = sub[label_col].to_numpy()
        for fs_name, (groups, use_rad, use_deep) in FEATURE_SETS.items():
            if wanted_sets is not None and fs_name not in wanted_sets:
                continue
            if (use_rad and not blocks["rad"]) or (use_deep and not blocks["deep"]):
                continue
            numeric, categorical = feature_columns(groups)
            rad, deep = (blocks["rad"] if use_rad else []), (blocks["deep"] if use_deep else [])
            X = sub[numeric + categorical + rad + deep]
            for model_name, make_model in models.items():
                if model_name == "majority" and fs_name != "demographics":
                    continue
                label = "-" if model_name == "majority" else fs_name
                fit_params = ({"clf__sample_weight": compute_sample_weight("balanced", y)}
                              if model_name == "xgboost" else None)
                res = cross_validate(build_pipeline(groups, rad, deep, k_rad, k_deep, make_model()), X, y, cv=cv,
                                     scoring=SCORING, fit_params=fit_params, error_score=np.nan)
                for metric in SCORING:
                    vals = res[f"test_{metric}"]
                    cv_rows.append({"task": task, "features": label, "model": model_name, "metric": metric,
                                    "mean": np.nanmean(vals), "sd": np.nanstd(vals), "n_folds": int(np.isfinite(vals).sum()),
                                    "n": len(y)})
                print(f"{task:9s} | {label:42s} | {model_name:13s} | " + " ".join(
                    f"{r['metric']}={r['mean']:.3f}±{r['sd']:.3f}" for r in cv_rows[-len(SCORING):]))

                if args.holdout and splits:
                    train_ids = [p for p in splits["classification"]["train"] + splits["classification"]["val"] if p in sub.index]
                    test_ids = [p for p in splits["classification"]["test"] if p in sub.index]
                    y_tr = sub.loc[train_ids, label_col].to_numpy()
                    pipe = build_pipeline(groups, rad, deep, k_rad, k_deep, make_model())
                    kw = {"clf__sample_weight": compute_sample_weight("balanced", y_tr)} if model_name == "xgboost" else {}
                    pipe.fit(X.loc[train_ids], y_tr, **kw)
                    proba = np.zeros((len(test_ids), len(classes)))
                    proba[:, pipe.classes_] = pipe.predict_proba(X.loc[test_ids])
                    table, _ = classification_metrics(sub.loc[test_ids, label_col].to_numpy(), proba,
                                                      [str(c) for c in classes], seed=cfg.get("seed", 42))
                    for _, r in table[~table["metric"].str.startswith("recall")].iterrows():
                        holdout_rows.append({"task": task, "features": label, "model": model_name, "metric": r["metric"],
                                             "value": r["value"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                                             "n": int(r["n"])})

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(out_dir / "cv_results.csv", index=False)
    wide = cv_df.assign(v=cv_df["mean"].map("{:.3f}".format) + " ± " + cv_df["sd"].map("{:.3f}".format)).pivot_table(
        index=["task", "features", "model"], columns="metric", values="v", aggfunc="first", sort=False)
    print(f"\n{wide.to_string()}\n\nSaved {out_dir / 'cv_results.csv'}")
    if holdout_rows:
        pd.DataFrame(holdout_rows).to_csv(out_dir / "holdout_results.csv", index=False)
        print(f"Saved {out_dir / 'holdout_results.csv'}")
    elif args.holdout:
        print("--holdout needs data.splits_file; skipped")


if __name__ == "__main__":
    main()
