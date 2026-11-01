# Notebooks

`colab_paper_runs.ipynb` runs the full corrected experiment plan on Google Colab (see
`docs/paper_protocol.md`).

`archive/` holds the original thesis notebooks (Google Colab), with outputs, execution counts and widget
state stripped to keep the repository small. They are kept for provenance and are **not** maintained.
Hard-coded Google Drive paths and `pip install` cells are left as they were.

| Notebook | Content | Superseded by |
|---|---|---|
| `00_metadata_analysis.ipynb` | metadata EDA, grade balancing, clinical-only MLP + SHAP | `glioma_mtl.data.metadata`, `scripts/kfold_cv.py` |
| `01_ssl_and_segmentation.ipynb` | hybrid SSL pretraining, SwinUNETR / SegResNet / UNet / SegMamba segmentation benchmark | `scripts/pretrain_ssl.py`, `scripts/train_segmentation.py`, `scripts/evaluate_segmentation.py` |
| `02_classification_experiments.ipynb` | cascaded, MTL, transformer, mask-guided, multi-scale, radiomics, XGBoost and Trinity classifiers | `glioma_mtl.models.MultiTaskGliomaNet`, `scripts/train_classifier.py`, `scripts/evaluate_classifier.py` |

Known issues in these notebooks (label leakage through IDH/1p19q inputs, overlapping splits, test
metrics computed on the training loader, all-zero image tokens in the Trinity model, a backbone checkpoint
that silently failed to load) are why the package exists. See the table in the top-level README.

To keep future notebooks output-free:

```bash
pip install nbstripout
```

```bash
nbstripout --install
```
