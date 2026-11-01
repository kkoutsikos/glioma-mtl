"""One patient-level split shared by every stage (SSL, segmentation, classification).

Rules enforced by :func:`validate_splits`:

* splits are disjoint within each stage;
* classification test patients are segmentation test patients, and never appear in SSL or
  segmentation train/val (the thesis notebooks broke this, so the backbone had seen the test cases);
* SSL has no test split; it only uses segmentation train/val patients.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from sklearn.model_selection import train_test_split

PARTS = ("train", "val", "test")


def make_strata(df: pd.DataFrame, columns: Sequence[str], min_count: int = 3) -> pd.Series:
    """Joint strata over ``columns``; rare combinations fall back to coarser strata.

    A combination with fewer than ``min_count`` members is replaced by the first column alone, and if
    that is still rare by the literal ``"rare"`` so that stratified splitting never fails.
    """
    joint = df[list(columns)].astype(str).agg("|".join, axis=1)
    counts = joint.value_counts()
    coarse = df[columns[0]].astype(str)
    strata = joint.where(joint.map(counts) >= min_count, coarse)
    counts = strata.value_counts()
    return strata.where(strata.map(counts) >= min_count, "rare")


def _split(ids: list[str], strata: pd.Series, size: float, seed: int) -> tuple[list[str], list[str]]:
    try:
        return train_test_split(ids, test_size=size, stratify=strata.loc[ids].values, random_state=seed)
    except ValueError as e:  # too few patients to put every stratum on both sides
        warnings.warn(f"Falling back to an unstratified split of {len(ids)} patients: {e}")
        return train_test_split(ids, test_size=size, random_state=seed)


def stratified_three_way(ids: Sequence[str], strata: pd.Series, val_frac: float, test_frac: float,
                         seed: int) -> dict[str, list[str]]:
    trainval, test = _split(list(ids), strata, test_frac, seed)
    train, val = _split(trainval, strata, val_frac / (1 - test_frac), seed)
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def make_splits(meta: pd.DataFrame, cohort_ids: Iterable[str], available_ids: Iterable[str],
                seed: int = 42, val_frac: float = 0.15, test_frac: float = 0.15) -> dict:
    """Build the shared split.

    ``cohort_ids`` is the classification cohort (e.g. the balanced 162 patients). It is split
    stratified on grade x diagnosis. All other ``available_ids`` (patients with images) are split
    stratified on grade and only feed the segmentation / SSL stages.
    """
    available = set(available_ids) & set(meta.index)
    cohort = sorted(set(cohort_ids) & available)
    others = sorted(available - set(cohort))

    cls = stratified_three_way(cohort, make_strata(meta.loc[cohort], ["Grade", "Diagnosis"]),
                               val_frac, test_frac, seed)
    rest = (stratified_three_way(others, make_strata(meta.loc[others], ["Grade"]), val_frac, test_frac, seed)
            if len(others) >= 10 else {"train": others, "val": [], "test": []})
    seg = {p: sorted(cls[p] + rest[p]) for p in PARTS}
    splits = {
        "seed": seed,
        "fractions": {"val": val_frac, "test": test_frac},
        "classification": cls,
        "segmentation": seg,
        "ssl": {"train": seg["train"], "val": seg["val"]},
    }
    validate_splits(splits)
    return splits


def validate_splits(splits: dict) -> None:
    for stage in ("classification", "segmentation", "ssl"):
        parts = splits[stage]
        names = list(parts)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                overlap = set(parts[a]) & set(parts[b])
                if overlap:
                    raise ValueError(f"{stage}: {a}/{b} overlap on {sorted(overlap)[:5]}")
    cls_test = set(splits["classification"]["test"])
    if not cls_test <= set(splits["segmentation"]["test"]):
        raise ValueError("classification test patients must be in the segmentation test split")
    seen = set(splits["segmentation"]["train"]) | set(splits["segmentation"]["val"]) | \
        set(splits["ssl"]["train"]) | set(splits["ssl"]["val"])
    if cls_test & seen:
        raise ValueError(f"classification test patients used for SSL/segmentation: {sorted(cls_test & seen)[:5]}")


def save_splits(splits: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)


def load_splits(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        splits = json.load(f)
    validate_splits(splits)
    return splits


def summarize(splits: dict, meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stage in ("classification", "segmentation"):
        for part in PARTS:
            ids = splits[stage].get(part, [])
            grades = meta.loc[ids, "Grade"].value_counts().to_dict() if ids else {}
            rows.append({"stage": stage, "split": part, "n": len(ids),
                         **{f"G{g}": grades.get(g, 0) for g in (2, 3, 4)}})
    return pd.DataFrame(rows)
