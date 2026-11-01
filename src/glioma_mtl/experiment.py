"""Glue between configs and the data / model building blocks, shared by the scripts."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .data.files import build_records, discover_patient_files
from .data.metadata import TASKS, ClinicalFeaturizer, check_leakage, class_weights, load_metadata
from .data.splits import load_splits
from .radiomics import RadiomicsFeaturizer, load_radiomics


@dataclass
class Context:
    meta: pd.DataFrame
    splits: dict
    files: dict[str, dict]


def load_context(cfg: Mapping) -> Context:
    d = cfg["data"]
    return Context(
        meta=load_metadata(d["metadata_csv"]),
        splits=load_splits(d["splits_file"]),
        files=discover_patient_files(d["image_dir"], d.get("modalities", ["T2"]), d.get("label_dir")),
    )


def stage_records(ctx: Context, stage: str, part: str, require_label: bool = True) -> list[dict]:
    return build_records(ctx.splits[stage][part], ctx.files, require_label=require_label)


@dataclass
class ClassificationData:
    records: dict[str, list[dict]]
    tasks: dict[str, int]                 # task -> number of classes
    tabular_dims: dict[str, int]
    class_weights: dict[str, np.ndarray]
    featurizers: dict

    def save_featurizers(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.featurizers, f)


def classification_data(cfg: Mapping, ctx: Context, featurizers: Mapping | None = None) -> ClassificationData:
    """Build records for every classification split.

    Clinical / radiomics featurizers are **fit on the training split only**, unless already-fit
    ``featurizers`` are passed (evaluation reuses the ones saved at training time).
    """
    tasks = list(cfg["tasks"])
    unknown = set(tasks) - TASKS.keys()
    if unknown:
        raise KeyError(f"Unknown tasks {sorted(unknown)}; choose from {sorted(TASKS)}")
    meta, split = ctx.meta, ctx.splits["classification"]
    train = meta.loc[split["train"]]
    featurizers = dict(featurizers or {})

    clin_cfg = cfg.get("clinical") or {}
    groups = list(clin_cfg.get("groups") or [])
    if groups:
        check_leakage(tasks, groups, clin_cfg.get("allow_leakage", False))
        featurizers.setdefault("clinical", ClinicalFeaturizer(groups).fit(train))

    rad_cfg = cfg.get("radiomics") or {}
    radiomics = None
    if rad_cfg.get("csv"):
        radiomics = load_radiomics(rad_cfg["csv"])
        if "radiomics" not in featurizers:
            label_col = TASKS[rad_cfg.get("select_task", tasks[0])][0]
            featurizers["radiomics"] = RadiomicsFeaturizer(rad_cfg.get("k", 30)).fit(
                radiomics.loc[split["train"]], train[label_col].to_numpy())

    pred_dir = cfg["data"].get("pred_mask_dir") if cfg["data"].get("crop_mask_key") == "pred_mask" else None
    records: dict[str, list[dict]] = {}
    for part, ids in split.items():
        rows = meta.loc[ids]
        extra = {pid: {f"y_{t}": int(rows.at[pid, TASKS[t][0]]) for t in tasks} for pid in ids}
        if "clinical" in featurizers:
            for pid, vec in zip(ids, featurizers["clinical"].transform(rows)):
                extra[pid]["tab_clinical"] = vec
        if radiomics is not None:
            for pid, vec in zip(ids, featurizers["radiomics"].transform(radiomics.loc[ids])):
                extra[pid]["tab_radiomics"] = vec
        records[part] = build_records(ids, ctx.files, extra=extra,
                                      pred_mask_dir=pred_dir if part != "train" else None)

    return ClassificationData(
        records=records,
        tasks={t: len(TASKS[t][1]) for t in tasks},
        tabular_dims={name: f.dim for name, f in featurizers.items()},
        class_weights={t: class_weights(train[TASKS[t][0]], len(TASKS[t][1])) for t in tasks},
        featurizers=featurizers,
    )


def build_classifier(cfg: Mapping, data: ClassificationData):
    from .models import MultiTaskGliomaNet

    m = cfg["model"]
    return MultiTaskGliomaNet(
        tasks=data.tasks, tabular_dims=data.tabular_dims, in_channels=m.get("in_channels", 1),
        feature_size=m["feature_size"], use_checkpoint=m.get("use_checkpoint", True),
        encoder_stages=m["encoder_stages"], use_decoder_token=m["use_decoder_token"], mask_mode=m["mask_mode"],
        fusion=m["fusion"], d_model=m["d_model"], n_heads=m["n_heads"], num_layers=m["num_layers"],
        dropout=m["dropout"], modality_dropout=m.get("modality_dropout", 0.0), aux_heads=m.get("aux_heads", True),
        freeze_backbone=m.get("freeze_backbone", False))


def build_classification_loss(cfg: Mapping, data: ClassificationData):
    import torch

    from .losses import MultiTaskLoss

    lc = cfg["loss"]
    # Only the configured tasks; inherited task_weights for other tasks are ignored.
    weights = {t: float(lc.get("task_weights", {}).get(t, 1.0)) for t in data.tasks}
    seg_weight = 0.0 if cfg["model"].get("freeze_backbone") else lc.get("seg_weight", 1.0)
    return MultiTaskLoss(weights, {t: torch.tensor(w) for t, w in data.class_weights.items()},
                         kind=lc.get("kind", "focal"), gamma=lc.get("gamma", 2.0),
                         label_smoothing=lc.get("label_smoothing", 0.0), seg_weight=seg_weight,
                         aux_weight=lc.get("aux_weight", 0.3), supcon_weight=lc.get("supcon_weight", 0.0))
