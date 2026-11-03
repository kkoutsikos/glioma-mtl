"""Repeated stratified k-fold CV of tabular / late-fusion classifiers (XGBoost, logistic regression, RF).

    python scripts/kfold_cv.py --config configs/kfold_cv.yaml
    python scripts/kfold_cv.py --config configs/kfold_cv.yaml --set deep_features.csv=runs/deep_features/deep_features_pred.csv \
        radiomics.csv=runs/radiomics/radiomics_pred.csv --holdout

Feature blocks: clinical groups (demographics, ...), radiomics, and label-free deep features from the
segmentation backbone (``scripts/extract_deep_features.py``). Every imputer / scaler / feature selector
(radiomics: RFE with a random forest or ANOVA, ``radiomics.select``; deep features: ANOVA) sits inside the
pipeline, so it is refit on each training fold.

Writes ``cv_results.csv`` (mean ± SD over folds), ``cv_folds.csv`` (every fold's score; the folds are the
same for every feature set and model of a task, so scores can be compared fold by fold) and
``oof_predictions.csv`` (one out-of-fold probability per patient from a single 5-fold pass, for calibration).
``--learning-curve`` instead fits each pipeline on stratified fractions of every training fold and scores it on
the full test fold (``learning_curve`` in the config), writing ``learning_curves.csv``: does any feature set
keep improving with more patients?
``--holdout`` also fits each pipeline once on classification train+val and evaluates it on the
classification test split, which no model in the pipeline has seen. Feature sets marked
``(leaky reference)`` use tissue- or surgery-derived variables; they are context, not a prediction model.
"""
from __future__ import annotations

import torch  # noqa: F401  (load before pandas; see src/glioma_mtl/__init__.py)
import argparse
import inspect
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, cross_validate,
                                     train_test_split)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from glioma_mtl.config import load_config
from glioma_mtl.data import load_metadata, load_splits
from glioma_mtl.data.metadata import TASKS, clinical_transformer_parts, feature_columns, normalize_patient_id
from glioma_mtl.evaluation import classification_metrics, macro_auc
from glioma_mtl.features import load_deep_features
from glioma_mtl.radiomics import TopK, load_radiomics, make_selector

# name -> (clinical groups, radiomics?, deep?)
def feature_sets(clinical: list[str]) -> dict[str, tuple[list[str], bool, bool]]:
    """name -> (clinical groups, radiomics?, deep?).

    Mirrors the deep classifiers' ladder: image -> image+clinical -> image+clinical+radiomics, where
    "image" = label-free deep features from the segmentation backbone and "clinical" uses the same groups as
    the deep models (config ``clinical.groups``). Plus the no-imaging baselines and the leaky references.
    """
    sets = {"age+sex": (["demographics"], False, False)}
    if sorted(clinical) != ["demographics"]:
        sets["clinical"] = (list(clinical), False, False)
    sets.update({
        "radiomics": ([], True, False),
        "clinical+radiomics": (list(clinical), True, False),
        "image": ([], False, True),
        "image+clinical": (list(clinical), False, True),
        "image+clinical+radiomics": (list(clinical), True, True),
        "molecular+surgical (leaky reference)": (["molecular", "surgical"], False, False),
        "thesis clinical vector (leaky reference)": (THESIS_CLINICAL, False, False),
        "image+thesis clinical (leaky reference)": (THESIS_CLINICAL, False, True),
        "image+thesis clinical+radiomics (leaky reference)": (THESIS_CLINICAL, True, True),
    })
    return sets


THESIS_CLINICAL = ["demographics", "history", "molecular", "surgical"]
SCORING = {"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "macro_f1": "f1_macro",
           "macro_auc": "roc_auc_ovr"}


def _check_xgboost_works_with_sklearn(xgb_cls) -> None:
    """xgboost < 2.1.4 breaks on scikit-learn >= 1.6 (estimator tags). Inside cross_validate that would only
    turn every XGBoost fold into NaN, so fail here, before hours of CV, with the fix."""
    try:
        from sklearn.utils import get_tags   # scikit-learn >= 1.6
    except ImportError:
        return
    try:
        get_tags(xgb_cls())
    except AttributeError as err:
        import sklearn
        import xgboost
        raise SystemExit(f"xgboost {xgboost.__version__} does not work with scikit-learn {sklearn.__version__}. "
                         "Run: pip install -U 'xgboost>=2.1.4'") from err


def make_models(cfg: dict) -> dict:
    models = {
        "majority": lambda: DummyClassifier(strategy="most_frequent"),
        "logreg": lambda: LogisticRegression(max_iter=5000, class_weight="balanced", C=cfg.get("logreg_C", 0.5)),
        "random_forest": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                        class_weight="balanced", random_state=0, n_jobs=-1),
        # Thesis clinical-only MLP; a fixed small network instead of a grid search (the thesis tuned it on a
        # split that also served as its test set). Early stopping uses an inner 10% of each training fold.
        "mlp": lambda: MLPClassifier(hidden_layer_sizes=tuple(cfg.get("mlp_hidden", (64,))), alpha=cfg.get("mlp_alpha", 1e-2),
                                     max_iter=2000, early_stopping=True, validation_fraction=0.1,
                                     n_iter_no_change=50, random_state=0),
    }
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("xgboost not installed; skipping it (pip install xgboost)")
    else:
        _check_xgboost_works_with_sklearn(XGBClassifier)
        xgb = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8,
               "colsample_bytree": 0.8, **(cfg.get("xgboost") or {})}  # thesis: 200 trees, depth 4, lr 0.05
        models["xgboost"] = lambda: XGBClassifier(**xgb, tree_method="hist", random_state=0, n_jobs=-1)
    wanted = cfg.get("models")
    return {k: v for k, v in models.items() if wanted is None or k in wanted or k == "majority"}


def fit_params_name(fn) -> str:
    """scikit-learn 1.4 renamed ``cross_validate(fit_params=...)`` to ``params`` and 1.6 removed the old name."""
    return "params" if "params" in inspect.signature(fn).parameters else "fit_params"


def _block(selector: TopK) -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("var", VarianceThreshold()),
                     ("scale", StandardScaler()), ("select", selector)])


def build_pipeline(groups, rad_cols, deep_cols, k_rad, k_deep, model, rad_select: str = "anova",
                   memory: Memory | None = None) -> Pipeline:
    """``memory`` caches the fitted feature step, so the classifiers of one fold share one feature selection."""
    parts = clinical_transformer_parts(groups) if groups else []
    if rad_cols:
        parts.append(("rad", _block(make_selector(rad_select, k_rad)), rad_cols))
    if deep_cols:
        parts.append(("deep", _block(TopK(k_deep)), deep_cols))
    return Pipeline([("features", ColumnTransformer(parts)), ("clf", model)], memory=memory)


GROUPS = {"num": "clinical", "cat": "clinical", "rad": "radiomics", "deep": "deep"}


def tree_shap(pipe: Pipeline, X_test: pd.DataFrame, task: str, features: str) -> list[dict]:
    """Mean |SHAP| per selected feature on the hold-out test set, via XGBoost's exact TreeSHAP (pred_contribs)."""
    import xgboost

    Xt = pipe[:-1].transform(X_test)
    names = pipe[:-1].get_feature_names_out()
    contribs = pipe[-1].get_booster().predict(xgboost.DMatrix(Xt), pred_contribs=True)
    contribs = np.abs(contribs[..., :-1])            # drop the bias column; [n, (k,) f]
    mean_abs = contribs.mean(axis=tuple(range(contribs.ndim - 1)))
    return [{"task": task, "features": features, "feature": n, "group": GROUPS.get(n.split("__")[0], "other"),
             "mean_abs_shap": float(v)} for n, v in zip(names, mean_abs)]


def _subsample(train_idx: np.ndarray, y: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """A stratified ``fraction`` of a training fold (all of it at 1.0); unstratified if a class is too rare."""
    if fraction >= 1:
        return train_idx
    try:
        return train_test_split(train_idx, train_size=fraction, stratify=y[train_idx], random_state=seed)[0]
    except ValueError:
        return train_test_split(train_idx, train_size=fraction, random_state=seed)[0]


def learning_curve_rows(task: str, X_by_set: dict, y: np.ndarray, n_classes: int, models: dict, cfg: dict,
                        make_pipe) -> list[dict]:
    """Macro AUC on the full test folds of pipelines fit on stratified fractions of each training fold.

    Every feature set and model sees the same folds and the same subsample of a fold at a given fraction, so the
    curves are paired. A fit that fails (e.g. XGBoost when a small subsample lacks a class) scores NaN.
    """
    lc = cfg.get("learning_curve") or {}
    folds = RepeatedStratifiedKFold(n_splits=(cfg.get("cv") or {}).get("folds", 5), n_repeats=lc.get("repeats", 4),
                                    random_state=cfg.get("seed", 42))
    rows = []
    for fold, (tr, te) in enumerate(folds.split(np.zeros(len(y)), y)):
        for fraction in lc.get("fractions", [0.25, 0.5, 0.75, 1.0]):
            part = _subsample(tr, y, fraction, seed=fold)
            for fs_name, X in X_by_set.items():
                for model_name, make_model in models.items():
                    pipe = make_pipe(fs_name, make_model())
                    kw = {"clf__sample_weight": compute_sample_weight("balanced", y[part])} if model_name == "xgboost" else {}
                    try:
                        pipe.fit(X.iloc[part], y[part], **kw)
                        proba = np.zeros((len(te), n_classes))
                        proba[:, pipe.classes_] = pipe.predict_proba(X.iloc[te])
                        auc = macro_auc(y[te], proba)
                    except ValueError:
                        auc = np.nan
                    rows.append({"task": task, "features": fs_name, "model": model_name, "fraction": fraction,
                                 "n_train": len(part), "fold": fold, "macro_auc": auc})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--holdout", action="store_true", help="also fit on train+val and score the test split")
    parser.add_argument("--learning-curve", action="store_true",
                        help="only the learning curves (config: learning_curve), written to learning_curves.csv")
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
    rad_select = (cfg.get("radiomics") or {}).get("select", "anova")
    memory = Memory(tempfile.mkdtemp(prefix="kfold_features_"), verbose=0)
    print(f"Cohort: {len(df)} patients | radiomics: {len(blocks['rad'])} | deep: {len(blocks['deep'])} features")

    models = make_models(cv_cfg)
    wanted_sets = cv_cfg.get("feature_sets")
    sets = feature_sets((cfg.get("clinical") or {}).get("groups") or ["demographics"])
    cv = RepeatedStratifiedKFold(n_splits=cv_cfg.get("folds", 5), n_repeats=cv_cfg.get("repeats", 10),
                                 random_state=cfg.get("seed", 42))
    out_dir = Path(args.out_dir or Path(cfg.get("output_root", "runs")) / "kfold_cv")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.learning_curve:
        lc = cfg.get("learning_curve") or {}
        lc_models = {k: v for k, v in models.items() if k in lc.get("models", ["xgboost", "logreg"])}
        rows = []
        for task in cfg["tasks"]:
            label_col, classes = TASKS[task]
            sub = df[df[label_col] >= 0]
            X_by_set, spec = {}, {}
            for fs_name in lc.get("feature_sets") or list(sets):
                if fs_name not in sets:
                    continue
                groups, use_rad, use_deep = sets[fs_name]
                if (use_rad and not blocks["rad"]) or (use_deep and not blocks["deep"]):
                    continue
                numeric, categorical = feature_columns(groups)
                rad, deep = (blocks["rad"] if use_rad else []), (blocks["deep"] if use_deep else [])
                X_by_set[fs_name], spec[fs_name] = sub[numeric + categorical + rad + deep], (groups, rad, deep)

            def make_pipe(fs_name, model, spec=spec):
                groups, rad, deep = spec[fs_name]
                return build_pipeline(groups, rad, deep, k_rad, k_deep, model, rad_select, memory)

            rows += learning_curve_rows(task, X_by_set, sub[label_col].to_numpy(), len(classes), lc_models, cfg,
                                        make_pipe)
            print(f"{task}: learning curves for {len(X_by_set)} feature sets x {len(lc_models)} models")
        shutil.rmtree(memory.location, ignore_errors=True)
        pd.DataFrame(rows).to_csv(out_dir / "learning_curves.csv", index=False)
        print(f"Saved {out_dir / 'learning_curves.csv'}")
        return

    oof_cv = StratifiedKFold(n_splits=cv_cfg.get("folds", 5), shuffle=True, random_state=cfg.get("seed", 42))
    cv_rows, fold_rows, holdout_rows, pred_rows, shap_rows, oof_rows = [], [], [], [], [], []
    for task in cfg["tasks"]:
        label_col, classes = TASKS[task]
        sub = df[df[label_col] >= 0]
        y = sub[label_col].to_numpy()
        # Fixed random_state: every feature set and model of this task sees exactly these folds, so their
        # fold scores pair up for the corrected t-test (aggregate_results.py).
        fold_sizes = [(len(tr), len(te)) for tr, te in cv.split(np.zeros(len(y)), y)]
        for fs_name, (groups, use_rad, use_deep) in sets.items():
            if wanted_sets is not None and fs_name not in wanted_sets:
                continue
            if (use_rad and not blocks["rad"]) or (use_deep and not blocks["deep"]):
                continue
            numeric, categorical = feature_columns(groups)
            rad, deep = (blocks["rad"] if use_rad else []), (blocks["deep"] if use_deep else [])
            X = sub[numeric + categorical + rad + deep]
            for model_name, make_model in models.items():
                if model_name == "majority" and fs_name != "age+sex":
                    continue
                label = "-" if model_name == "majority" else fs_name
                fit_params = ({"clf__sample_weight": compute_sample_weight("balanced", y)}
                              if model_name == "xgboost" else None)
                pipeline = build_pipeline(groups, rad, deep, k_rad, k_deep, make_model(), rad_select, memory)
                res = cross_validate(pipeline, X, y, cv=cv, scoring=SCORING, error_score=np.nan,
                                     **{fit_params_name(cross_validate): fit_params})
                for metric in SCORING:
                    vals = res[f"test_{metric}"]
                    cv_rows.append({"task": task, "features": label, "model": model_name, "metric": metric,
                                    "mean": np.nanmean(vals), "sd": np.nanstd(vals), "n_folds": int(np.isfinite(vals).sum()),
                                    "n": len(y)})
                    fold_rows += [{"task": task, "features": label, "model": model_name, "metric": metric, "fold": i,
                                   "value": v, "n_train": n_tr, "n_test": n_te}
                                  for i, (v, (n_tr, n_te)) in enumerate(zip(vals, fold_sizes))]
                print(f"{task:9s} | {label:42s} | {model_name:13s} | " + " ".join(
                    f"{r['metric']}={r['mean']:.3f}±{r['sd']:.3f}" for r in cv_rows[-len(SCORING):]))
                # One out-of-fold probability per patient (a single 5-fold pass), for calibration and decision curves.
                oof = cross_val_predict(build_pipeline(groups, rad, deep, k_rad, k_deep, make_model(), rad_select, memory),
                                        X, y, cv=oof_cv, method="predict_proba",
                                        **{fit_params_name(cross_val_predict): fit_params})
                present = np.unique(y)
                oof_rows += [{"task": task, "features": label, "model": model_name, "patient_id": pid, "y": int(yv),
                              **{f"p_{k}": float(row[list(present).index(k)]) if k in present else 0.0
                                 for k in range(len(classes))}}
                             for pid, yv, row in zip(sub.index, y, oof)]

                if args.holdout and splits:
                    train_ids = [p for p in splits["classification"]["train"] + splits["classification"]["val"] if p in sub.index]
                    test_ids = [p for p in splits["classification"]["test"] if p in sub.index]
                    y_tr = sub.loc[train_ids, label_col].to_numpy()
                    pipe = build_pipeline(groups, rad, deep, k_rad, k_deep, make_model(), rad_select, memory)
                    kw = {"clf__sample_weight": compute_sample_weight("balanced", y_tr)} if model_name == "xgboost" else {}
                    pipe.fit(X.loc[train_ids], y_tr, **kw)
                    proba = np.zeros((len(test_ids), len(classes)))
                    proba[:, pipe.classes_] = pipe.predict_proba(X.loc[test_ids])
                    y_te = sub.loc[test_ids, label_col].to_numpy()
                    pred_rows += [{"task": task, "features": label, "model": model_name, "patient_id": pid, "y": int(yv),
                                   **{f"p_{k}": float(p) for k, p in enumerate(pr)}}
                                  for pid, yv, pr in zip(test_ids, y_te, proba)]
                    if model_name == "xgboost":
                        shap_rows += tree_shap(pipe, X.loc[test_ids], task, label)
                    table, _ = classification_metrics(y_te, proba, [str(c) for c in classes], seed=cfg.get("seed", 42))
                    for _, r in table[~table["metric"].str.startswith("recall")].iterrows():
                        holdout_rows.append({"task": task, "features": label, "model": model_name, "metric": r["metric"],
                                             "value": r["value"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                                             "n": int(r["n"])})

    shutil.rmtree(memory.location, ignore_errors=True)
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(out_dir / "cv_results.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out_dir / "cv_folds.csv", index=False)
    pd.DataFrame(oof_rows).to_csv(out_dir / "oof_predictions.csv", index=False)
    wide = cv_df.assign(v=cv_df["mean"].map("{:.3f}".format) + " ± " + cv_df["sd"].map("{:.3f}".format)).pivot_table(
        index=["task", "features", "model"], columns="metric", values="v", aggfunc="first", sort=False)
    print(f"\n{wide.to_string()}\n\nSaved {out_dir / 'cv_results.csv'}")
    if holdout_rows:
        pd.DataFrame(holdout_rows).to_csv(out_dir / "holdout_results.csv", index=False)
        pd.DataFrame(pred_rows).to_csv(out_dir / "holdout_predictions.csv", index=False)
        if shap_rows:
            pd.DataFrame(shap_rows).to_csv(out_dir / "holdout_shap.csv", index=False)
        print(f"Saved {out_dir / 'holdout_results.csv'}, holdout_predictions.csv, holdout_shap.csv")
    elif args.holdout:
        print("--holdout needs data.splits_file; skipped")


if __name__ == "__main__":
    main()
