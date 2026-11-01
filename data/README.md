# Data

Nothing in this folder except `splits/` is committed (see `.gitignore`). UCSF-PDGM is distributed by
TCIA under its own data usage terms; check them before sharing any derived files.

## 1. Images and masks

Download UCSF-PDGM from TCIA: https://www.cancerimagingarchive.net/collection/ucsf-pdgm/

Either layout works, because files are matched by patient ID (`UCSF-PDGM-0004`, …):

```
data/raw/UCSF-PDGM/UCSF-PDGM-0004_nifti/UCSF-PDGM-0004_T2.nii.gz          # official, nested
data/raw/UCSF-PDGM/UCSF-PDGM-0004_nifti/UCSF-PDGM-0004_tumor_segmentation.nii.gz
```

```
volumes/UCSF-PDGM-0004_T2.nii.gz                                         # flat (as in the thesis)
seg/UCSF-PDGM-0004_tumor_segmentation.nii.gz       -> set data.label_dir=seg
```

- `data.modalities: [T2]` selects `*_T2.nii.gz`. Add e.g. `FLAIR`, `T1c` for multi-channel input and set
  `model.in_channels` to match.
- Masks are binarised (`> 0` = whole tumour), so multi-label masks are fine.
- Follow-up scans (`*_FU###d`) are ignored; only baseline scans are used.

## 2. Metadata

Copy into `data/metadata/`:

| File | Config key | Notes |
|---|---|---|
| `UCSF-PDGM-metadata_v5.csv` | `data.metadata_csv` | the TCIA metadata (501 rows = 495 patients + 6 follow-ups) |
| `balanced_metadata_gradeaware.csv` | `data.cohort_csv` | the thesis' 162-patient classification cohort (all G2/G3 + 63 G4). Only the `ID` column is used. If missing, `make_splits.py` rebuilds it with the same rule and seed. |

## 3. Splits

`splits/splits_seed42.json` was generated from the metadata with

```bash
python scripts/make_splits.py --config configs/base.yaml --from-metadata
```

| Stage | train | val | test |
|---|---|---|---|
| classification (162 cohort) | 112 | 25 | 25 |
| segmentation (all 495) | 345 | 75 | 75 |
| SSL | 345 | 75 | – |

Rerun it **without** `--from-metadata` once the images are in place, so the split only contains patients
that actually have an image and a mask. The training scripts also stop with a list of any patient whose
files are missing.
