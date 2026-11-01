"""Find NIfTI files per patient and build MONAI data dicts.

Images and masks are always paired **by patient ID**, never by sorted file order (the thesis
segmentation notebook zipped two sorted lists and truncated on a count mismatch).

Works with the official nested layout (``UCSF-PDGM-0004_nifti/UCSF-PDGM-0004_T2.nii.gz``) and with
flat folders (``volumes/UCSF-PDGM-0004_T2.nii.gz``, ``seg/UCSF-PDGM-0004_tumor_segmentation.nii.gz``).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .metadata import FOLLOWUP, normalize_patient_id

_PID = re.compile(r"(UCSF[-_]PDGM[-_]?\d+)", re.IGNORECASE)


def _index(folder: str | Path, suffix: str) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in Path(folder).rglob(f"*{suffix}"):
        m = _PID.search(path.name)
        if not m or FOLLOWUP.search(str(path)):  # follow-up scans: baseline only, see load_metadata
            continue
        pid = normalize_patient_id(m.group(1))
        if pid in found:
            raise ValueError(f"Two files for {pid} with suffix {suffix}: {found[pid]} and {path}")
        found[pid] = path
    return found


def discover_patient_files(image_dir: str | Path, modalities: Sequence[str] = ("T2",),
                           label_dir: str | Path | None = None,
                           label_suffix: str = "_tumor_segmentation.nii.gz") -> dict[str, dict]:
    """Return ``{patient_id: {"image": [paths per modality], "label": path or None}}``.

    Only patients that have *every* requested modality are returned.
    """
    per_mod = [_index(image_dir, f"_{m}.nii.gz") for m in modalities]
    labels = _index(label_dir or image_dir, label_suffix)
    common = set(per_mod[0]).intersection(*per_mod[1:])
    return {pid: {"image": [str(idx[pid]) for idx in per_mod],
                  "label": str(labels[pid]) if pid in labels else None}
            for pid in sorted(common)}


def build_records(patient_ids: Iterable[str], files: Mapping[str, dict], require_label: bool = True,
                  extra: Mapping[str, Mapping] | None = None,
                  pred_mask_dir: str | Path | None = None) -> list[dict]:
    """Data dicts for MONAI. ``extra[pid]`` (labels, tabular vectors) is merged into each record.

    ``pred_mask_dir`` adds a ``pred_mask`` key pointing to ``<dir>/<pid>.nii.gz`` (written by
    ``scripts/evaluate_segmentation.py --save-predictions``) for test-time tumour localisation.
    """
    records = []
    missing = []
    for pid in patient_ids:
        if pid not in files or (require_label and files[pid]["label"] is None):
            missing.append(pid)
            continue
        rec = {"patient_id": pid, "image": files[pid]["image"]}
        if files[pid]["label"] is not None:
            rec["label"] = files[pid]["label"]
        if pred_mask_dir is not None:
            rec["pred_mask"] = str(Path(pred_mask_dir) / f"{pid}.nii.gz")
        if extra is not None:
            for key, value in extra[pid].items():
                rec[key] = np.asarray(value, dtype=np.float32) if isinstance(value, (list, np.ndarray)) else value
        records.append(rec)
    if missing:
        raise FileNotFoundError(f"{len(missing)} patients in the split have no image/label files, "
                                f"e.g. {missing[:5]}. Check data.image_dir / data.label_dir.")
    return records
