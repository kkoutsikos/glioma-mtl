"""Create the single patient-level split used by every stage.

    python scripts/make_splits.py --config configs/base.yaml

The classification cohort comes from ``data.cohort_csv`` (the thesis' 162-patient balanced set); if
that file is missing the cohort is rebuilt with the thesis rule. Patients are restricted to those with
image files in ``data.image_dir``; pass ``--from-metadata`` to split without the images present.
"""
from __future__ import annotations

import torch  # noqa: F401  (load before pandas; see src/glioma_mtl/__init__.py)
import argparse
from pathlib import Path

import pandas as pd

from glioma_mtl.config import load_config
from glioma_mtl.data import discover_patient_files, load_metadata, make_balanced_cohort, make_splits, save_splits
from glioma_mtl.data.metadata import normalize_patient_id
from glioma_mtl.data.splits import summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--from-metadata", action="store_true",
                        help="use every patient in the metadata instead of those with image files")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    d = cfg["data"]

    meta = load_metadata(d["metadata_csv"])
    cohort_csv = d.get("cohort_csv")
    if cohort_csv and Path(cohort_csv).exists():
        cohort = sorted(pd.read_csv(cohort_csv)["ID"].map(normalize_patient_id))
        print(f"Cohort: {len(cohort)} patients from {cohort_csv}")
    else:
        cohort = make_balanced_cohort(meta, seed=cfg.get("seed", 42))
        print(f"Cohort: {len(cohort)} patients rebuilt with the thesis balancing rule")

    if args.from_metadata:
        available = list(meta.index)
    else:
        files = discover_patient_files(d["image_dir"], d.get("modalities", ["T2"]), d.get("label_dir"))
        available = [pid for pid, f in files.items() if f["label"] is not None]
        print(f"{len(available)} patients have images + masks in {d['image_dir']}")
        if not available:
            raise SystemExit("No image/mask pairs found. Fix data.image_dir or use --from-metadata.")

    splits = make_splits(meta, cohort, available, seed=cfg.get("seed", 42),
                         val_frac=cfg["splits"]["val_frac"], test_frac=cfg["splits"]["test_frac"])
    save_splits(splits, d["splits_file"])
    print(summarize(splits, meta).to_string(index=False))
    print(f"Saved {d['splits_file']}")


if __name__ == "__main__":
    main()
