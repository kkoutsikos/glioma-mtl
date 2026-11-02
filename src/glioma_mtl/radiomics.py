"""Radiomics extraction (pyradiomics) and leakage-safe feature selection.

The thesis selected features with RFE/LASSO on all 162 patients *before* splitting, so the test
labels influenced which features were kept. Here every scaler/selector is fit on training data only:
:class:`RadiomicsFeaturizer` for the deep model's radiomics token, :func:`tabular_pipeline` inside
cross-validation for the tabular baselines.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data.metadata import normalize_patient_id


class TopK(BaseEstimator, TransformerMixin):
    """Keep the ``k`` features with the highest ANOVA F score (all of them if fewer than ``k`` remain).

    Unlike ``SelectKBest`` it does not fail when an earlier ``VarianceThreshold`` drops columns in a fold.
    """

    def __init__(self, k: int = 30):
        self.k = k

    def fit(self, X, y):
        with np.errstate(divide="ignore", invalid="ignore"):  # constant columns give NaN scores
            scores, _ = f_classif(X, y)
        scores = np.nan_to_num(scores, nan=-np.inf)
        self.support_ = np.sort(np.argsort(scores)[::-1][: min(self.k, X.shape[1])])
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.support_]

    def get_feature_names_out(self, input_features=None):
        names = np.asarray(input_features if input_features is not None
                           else [f"x{i}" for i in range(int(self.support_.max()) + 1)], dtype=object)
        return names[self.support_]

# Settings used in the thesis (T2, intensity-normalised, 1 mm, bin width 5).
DEFAULT_SETTINGS = {
    "normalize": True,
    "normalizeScale": 100,
    "resampledPixelSpacing": [1, 1, 1],
    "binWidth": 5,
    "voxelArrayShift": 300,
}
FEATURE_CLASSES = ("firstorder", "shape", "glcm", "glrlm", "glszm", "gldm")


def extract_radiomics(cases: Iterable[tuple[str, str, str]], settings: Mapping | None = None,
                      feature_classes: Iterable[str] = FEATURE_CLASSES) -> pd.DataFrame:
    """``cases`` = (patient_id, image_path, mask_path). Returns one row per patient, ``original_*`` columns."""
    import logging

    import SimpleITK as sitk
    from radiomics import featureextractor

    logging.getLogger("radiomics").setLevel(logging.WARNING)  # pyradiomics logs every feature class at INFO
    params = dict(DEFAULT_SETTINGS if settings is None else settings)
    params.setdefault("interpolator", sitk.sitkBSpline)
    extractor = featureextractor.RadiomicsFeatureExtractor(**params)
    extractor.disableAllFeatures()
    for name in feature_classes:
        extractor.enableFeatureClassByName(name)

    rows = []
    for pid, image, mask in cases:
        # Whole tumour (> 0), consistent with the binarised masks used by the deep models;
        # passing a multi-label mask with label=1 would select only the necrotic core.
        mask_img = sitk.ReadImage(str(mask))
        whole = sitk.Cast(mask_img > 0, sitk.sitkUInt8)
        whole.CopyInformation(mask_img)
        if not sitk.GetArrayViewFromImage(whole).any():
            # A predicted mask can be empty; pyradiomics would raise. The row stays NaN and is imputed
            # by the training-fold featurizers like any other missing value.
            logging.getLogger("glioma_mtl").warning("%s: empty mask, radiomics left missing", pid)
            rows.append({"patient_id": normalize_patient_id(pid)})
            continue
        result = extractor.execute(sitk.ReadImage(str(image)), whole, label=1)
        rows.append({"patient_id": normalize_patient_id(pid),
                     **{k: float(v) for k, v in result.items() if k.startswith("original_")}})
    return pd.DataFrame(rows).set_index("patient_id")


def load_radiomics(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    id_col = next(c for c in ("patient_id", "Official_ID", "ID") if c in df.columns)
    df["patient_id"] = df[id_col].map(normalize_patient_id)
    features = [c for c in df.columns if c.startswith("original_")]
    return df.set_index("patient_id")[features]


@dataclass
class RadiomicsFeaturizer:
    """Impute -> drop constants -> standardise -> keep the ``k`` best by ANOVA F on *training* labels."""

    k: int | None = 30
    pipeline: Pipeline | None = None
    columns: list[str] = field(default_factory=list)

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> "RadiomicsFeaturizer":
        steps = [("impute", SimpleImputer(strategy="median")), ("var", VarianceThreshold()),
                 ("scale", StandardScaler())]
        if self.k:
            if y is None:
                raise ValueError("feature selection needs training labels")
            steps.append(("select", TopK(self.k)))
        self.columns = list(X.columns)
        keep = y >= 0 if y is not None else slice(None)
        self.pipeline = Pipeline(steps).fit(X.values[keep], None if y is None else y[keep])
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.transform(X[self.columns].values).astype(np.float32)

    @property
    def dim(self) -> int:
        return int(self.pipeline.transform(np.zeros((1, len(self.columns)))).shape[1])


def tabular_pipeline(estimator, k: int | None = None) -> Pipeline:
    """Scaling (and optional selection) inside the pipeline, so CV refits them on each training fold."""
    steps = [("impute", SimpleImputer(strategy="median")), ("var", VarianceThreshold()), ("scale", StandardScaler())]
    if k:
        steps.append(("select", TopK(k)))
    steps.append(("clf", estimator))
    return Pipeline(steps)
