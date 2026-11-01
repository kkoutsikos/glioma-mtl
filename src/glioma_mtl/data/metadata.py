"""UCSF-PDGM clinical metadata: loading, label encoding, leakage-aware feature groups.

Clinical features are grouped by *when they become available*. Only ``demographics`` is known at the
time of the pre-operative MRI. ``molecular`` (IDH, 1p/19q, MGMT) and ``surgical`` (extent of resection)
come from tissue or surgery, and IDH + 1p/19q *define* the WHO-2021 diagnosis. Using them as inputs to
predict grade or diagnosis is label leakage, so :func:`check_leakage` refuses it unless explicitly allowed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RENAME_MAP = {
    "Age at MRI": "Age",
    "WHO CNS Grade": "Grade",
    "Final pathologic diagnosis (WHO 2021)": "Diagnosis",
    "MGMT status": "MGMT_Status",
    "MGMT index": "MGMT_Index",
    "1p/19q": "Codeletion_1p19q",
    "1-dead 0-alive": "Survival_Status",
    "Biopsy prior to imaging": "PreImaging_Biopsy",
    "BraTS21 ID": "BraTS21_ID",
    "BraTS21 Segmentation Cohort": "Segmentation_Cohort",
    "BraTS21 MGMT Cohort": "MGMT_Cohort",
}

GRADE_CLASSES: tuple[int, ...] = (2, 3, 4)
DIAGNOSIS_CLASSES: tuple[str, ...] = (
    "Astrocytoma, IDH-mutant",
    "Astrocytoma, IDH-wildtype",
    "Glioblastoma, IDH-wildtype",
    "Oligodendroglioma, IDH-mutant, 1p/19q-codeleted",
)
IDH_CLASSES: tuple[str, ...] = ("mutant", "wildtype")

# task name -> (label column, class list)
TASKS = {
    "grade": ("grade_label", GRADE_CLASSES),
    "diagnosis": ("diagnosis_label", DIAGNOSIS_CLASSES),
    "idh": ("idh_label", IDH_CLASSES),
}

FEATURE_GROUPS: dict[str, dict[str, list[str]]] = {
    "demographics": {"numeric": ["Age"], "categorical": ["Sex"]},
    "history": {"numeric": [], "categorical": ["PreImaging_Biopsy"]},
    "molecular": {"numeric": ["MGMT_Index"], "categorical": ["IDH_Status", "Codeletion_1p19q", "MGMT_Status"]},
    "surgical": {"numeric": [], "categorical": ["EOR"]},
}

# Groups that are unavailable pre-operatively and/or determine the target.
LEAKY_GROUPS = {"molecular", "surgical"}

# Feature set used in the thesis notebooks (kept for reproduction only).
THESIS_FEATURE_GROUPS = ["demographics", "history", "molecular", "surgical"]

MISSING_TOKENS = {"unknown", "indeterminate", "nan", ""}

# Follow-up scans of patients who also have a baseline scan, e.g. "UCSF-PDGM-0391_FU016d".
FOLLOWUP = re.compile(r"_FU\d+d", re.IGNORECASE)


def normalize_patient_id(raw: str) -> str:
    """'UCSF-PDGM-004', 'UCSF-PDGM-0004_T2.nii.gz', 'sub-4' -> 'UCSF-PDGM-0004'."""
    m = re.search(r"UCSF[-_]PDGM[-_]?(\d+)", str(raw), re.IGNORECASE) or re.search(r"(\d+)\s*$", str(raw))
    if not m:
        raise ValueError(f"Cannot parse a patient number from {raw!r}")
    return f"UCSF-PDGM-{int(m.group(1)):04d}"


def _clean_categorical(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    return s.where(~s.str.lower().isin(MISSING_TOKENS), np.nan)


def load_metadata(path: str | Path, include_followup: bool = False) -> pd.DataFrame:
    """Read the UCSF-PDGM metadata CSV (raw or already-renamed) into a tidy frame indexed by patient.

    The v5 metadata has 501 rows: 495 baseline scans plus 6 follow-up scans (``*_FU###d``) of patients
    already present. Follow-ups are dropped by default so that one patient is one row.
    """
    df = pd.read_csv(path).rename(columns=RENAME_MAP)
    if not include_followup:
        df = df[~df["ID"].astype(str).str.contains(FOLLOWUP)].copy()
    df["patient_id"] = df["ID"].map(normalize_patient_id)
    if df["patient_id"].duplicated().any():
        dups = df.loc[df["patient_id"].duplicated(), "patient_id"].tolist()
        raise ValueError(f"Duplicate patient ids after normalisation: {dups[:5]}")

    for col in ["Sex", "MGMT_Status", "Codeletion_1p19q", "IDH", "EOR", "PreImaging_Biopsy"]:
        if col in df:
            df[col] = _clean_categorical(df[col])
    if "MGMT_Index" in df:
        df["MGMT_Index"] = pd.to_numeric(df["MGMT_Index"], errors="coerce")

    # Collapse the specific mutations (IDH1 p.R132H, ...) into mutant / wildtype.
    df["IDH_Status"] = df["IDH"].map(
        lambda v: v if pd.isna(v) else ("wildtype" if v.lower() == "wildtype" else "mutant"))
    return encode_targets(df.set_index("patient_id", drop=False))


def encode_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add integer label columns with fixed class order; -1 marks a missing label."""
    out = df.copy()
    out["grade_label"] = out["Grade"].map({g: i for i, g in enumerate(GRADE_CLASSES)}).fillna(-1).astype(int)
    out["diagnosis_label"] = out["Diagnosis"].map({d: i for i, d in enumerate(DIAGNOSIS_CLASSES)}).fillna(-1).astype(int)
    out["idh_label"] = out["IDH_Status"].map({c: i for i, c in enumerate(IDH_CLASSES)}).fillna(-1).astype(int)
    return out


def make_balanced_cohort(df: pd.DataFrame, extra_grade4: int = 20, seed: int = 42) -> list[str]:
    """Reproduce the thesis cohort: all grade 2/3 plus (n_grade3 + extra_grade4) random grade-4 patients.

    Same selection rule and random_state as ``analysis.ipynb`` (sklearn ``resample`` on the grade-4 rows
    in file order). Prefer passing the saved cohort CSV so the exact IDs are reused.
    """
    from sklearn.utils import resample

    g4 = df[df["Grade"] == 4]
    n_g3 = int((df["Grade"] == 3).sum())
    g4_kept = resample(g4, replace=False, n_samples=n_g3 + extra_grade4, random_state=seed)
    keep = df[df["Grade"].isin([2, 3])].index.tolist() + g4_kept.index.tolist()
    return sorted(keep)


def check_leakage(tasks: Iterable[str], groups: Iterable[str], allow_leakage: bool = False) -> None:
    leaky = sorted(set(groups) & LEAKY_GROUPS)
    if leaky and not allow_leakage:
        raise ValueError(
            f"Feature groups {leaky} are tissue-/surgery-derived and determine the targets {list(tasks)} "
            "(IDH + 1p/19q define the WHO-2021 diagnosis). Remove them, or set "
            "clinical.allow_leakage: true to run a deliberately leaky reference experiment.")


def feature_columns(groups: Sequence[str]) -> tuple[list[str], list[str]]:
    unknown = set(groups) - FEATURE_GROUPS.keys()
    if unknown:
        raise KeyError(f"Unknown feature groups {sorted(unknown)}; choose from {sorted(FEATURE_GROUPS)}")
    numeric = [c for g in groups for c in FEATURE_GROUPS[g]["numeric"]]
    categorical = [c for g in groups for c in FEATURE_GROUPS[g]["categorical"]]
    return numeric, categorical


def clinical_transformer_parts(groups: Sequence[str]) -> list[tuple]:
    """Unfitted ColumnTransformer parts: median-impute + scale numerics, mode-impute + one-hot categoricals."""
    numeric, categorical = feature_columns(groups)
    parts = []
    if numeric:
        parts.append(("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                                        ("scale", StandardScaler())]), numeric))
    if categorical:
        parts.append(("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
                      categorical))
    return parts


@dataclass
class ClinicalFeaturizer:
    """Impute / scale / one-hot clinical columns. Fit on the training split only."""

    groups: list[str] = field(default_factory=lambda: ["demographics"])
    transformer: ColumnTransformer | None = None

    def fit(self, df: pd.DataFrame) -> "ClinicalFeaturizer":
        numeric, categorical = feature_columns(self.groups)
        self.transformer = ColumnTransformer(clinical_transformer_parts(self.groups)).fit(df[numeric + categorical])
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.transformer is None:
            raise RuntimeError("ClinicalFeaturizer.transform called before fit")
        numeric, categorical = feature_columns(self.groups)
        return self.transformer.transform(df[numeric + categorical]).astype(np.float32)

    @property
    def feature_names(self) -> list[str]:
        return list(self.transformer.get_feature_names_out()) if self.transformer is not None else []

    @property
    def dim(self) -> int:
        return len(self.feature_names)


def class_weights(labels: Sequence[int], num_classes: int) -> np.ndarray:
    """'Balanced' weights n / (k * n_c) from *training* labels; missing labels (-1) are ignored."""
    y = np.asarray([v for v in labels if v >= 0])
    counts = np.bincount(y, minlength=num_classes).astype(float)
    counts[counts == 0] = np.inf
    return (len(y) / (num_classes * counts)).astype(np.float32)
