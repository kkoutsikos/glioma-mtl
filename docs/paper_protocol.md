# Paper experiment protocol

This is the corrected version of the thesis experiments. It keeps the thesis design (T2-only input,
SwinUNETR / SegResNet / UNet / SegMamba segmentation, SSL pretraining, mask-guided multi-task
classification, XGBoost late fusion) and fixes the critical methodological errors. Run everything with

```bash
python scripts/run_paper.py --plan configs/paper.yaml
```

or with `notebooks/colab_paper_runs.ipynb`. **No thesis checkpoint can be reused**: all of them were
trained on splits that contained the classification test patients.

## 1. What is corrected

| Thesis issue | Correction | Where |
|---|---|---|
| IDH / 1p19q / MGMT / EOR fed as inputs when predicting grade and WHO-2021 diagnosis | Main models use imaging only, or imaging + age/sex. The thesis vector appears only as a labelled *leaky reference* | `clinical.groups`, `check_leakage`, `configs/cls_leaky_reference.yaml` |
| "FINAL TEST" metrics computed on the training loader | evaluation scripts take an explicit `--split test` and log its size | `scripts/evaluate_*.py` |
| Different, overlapping splits for SSL / segmentation / classification | one split file; classification test ⊂ segmentation test; never used for SSL or segmentation training | `scripts/make_splits.py`, `validate_splits` |
| Trinity: all image tokens were zero tensors | hook-free feature extraction, channel counts derived and checked | `models/backbone.py`, `models/classification.py` |
| Multi-scale model: backbone checkpoint silently not loaded | `load_weights` raises below 90% matched parameters | `utils.load_weights` |
| Radiomics selection fit on all patients before splitting | selection / scaling inside each training fold | `radiomics.TopK`, `scripts/kfold_cv.py` |
| XGBoost on embeddings from a network trained on the same patients' labels | features from the segmentation backbone (no class labels), repeated k-fold CV + one-shot hold-out | `features.py`, `scripts/kfold_cv.py` |
| Segmentation leaderboard first computed on the validation set | test split only, per-case CSV, bootstrap CI, Wilcoxon + Holm | `scripts/evaluate_segmentation.py`, `scripts/aggregate_results.py` |
| Image/mask pairing by sorted file order; follow-up scans | pairing by patient ID; follow-ups excluded | `data/files.py`, `data/metadata.py` |
| Unequal training budgets across segmentation models | one budget for all (`segmentation.overrides` in the plan) | `configs/paper.yaml` |

These further changes are improvements rather than error fixes. State them in the methods:

- per-volume z-score intensity normalisation instead of a fixed [0, 3000] window;
- tumour-centred classification crops (predicted mask at test time) instead of 50% background / centre
  crops;
- cube masking for SSL instead of voxel masking;
- a learnable rotation pretext task: the rotation view is no longer randomly flipped (two in-plane flips
  equal a 180° rotation, which made the label ambiguous), and the rotation head reads a 2×2×2 grid of the
  bottleneck instead of a single globally pooled vector;
- SSL gradient accumulation 4 instead of 16 (effective batch 8): about 2,150 instead of 540 optimizer
  updates over 50 epochs, at the same compute;
- a stronger UNet configuration;
- checkpoint selection on an EMA-smoothed validation loss.

## 2. Experiment matrix

**Definition of "clinical" used throughout:** everything known before surgery: age, sex, and whether a
biopsy preceded the MRI (`clinical.groups: [demographics, history]`). IDH, 1p/19q, MGMT and extent of
resection come from tissue or surgery and define the WHO-2021 labels. They appear only in the leaky
reference rows.

| Step | Runs | Answers | Table |
|---|---|---|---|
| `splits` | 1 | fixed patient-level split (162 cohort: 112/25/25; 495 segmentation: 345/75/75) | data |
| `ssl` | 1 | pretrains the SwinUNETR encoder on SSL train/val only | – |
| `seg` | the 5 thesis architectures (SwinUNETR scratch / +SSL, SegResNet, 3D U-Net, SegMamba) + the 3D U-Net in its thesis configuration, × seeds | which architecture segments whole tumour best on T2; does SSL help (`seg_swin_unetr` vs `seg_swin_unetr_ssl`); was the thesis U-Net result a configuration problem (`seg_unet` vs `seg_unet_thesis`) | `segmentation` |
| `predict` | backbone on classification train/val/test | predicted masks for crop localisation, deep features, radiomics | – |
| `features` | 1 + 1 | label-free deep features, radiomics from predicted masks | – |
| `cls` | 9 experiments × seeds. **Main ladder** (same model, only inputs change): image only → image + clinical → image + clinical + radiomics. A **leaky reference ladder** with the thesis clinical vector (image + thesis clinical → + radiomics) sits side by side with the last two rungs. The corrected thesis architectures (cascade, multi-scale, Trinity, SegMamba backbone) complete the set. | what each input adds for grade / diagnosis / LGG vs HGG; which fusion design helps; what post-operative molecular data explain (reference) | `classification` |
| `explain` | first seed of each classifier | which modality / image scale drives each prediction: fusion attention, token occlusion, 3D Grad-CAM per test patient | figures |
| `kfold` | repeated 5-fold × 10 + hold-out | the same ladder as late fusion (image → image + clinical → image + clinical + radiomics, with XGBoost / logistic regression / RF / MLP; "image" = label-free deep features), no-imaging baselines (age + sex, clinical, radiomics), leaky references; XGBoost TreeSHAP per modality | `kfold_cv`, `kfold_holdout` |
| `aggregate` | – | writes all tables (CSV, Markdown, LaTeX) | – |
| `figures` | – | training curves, segmentation boxplots + qualitative overlays, confusion matrices, ROC curves, k-fold bars, SHAP | figures |

### What is saved (all under `output_root`, i.e. on Drive)

| Where | What |
|---|---|
| `<run>/best.pt`, `config.yaml`, `history.csv`, `log.txt` | weights, the exact config, per-epoch metrics and log of every SSL, segmentation and classifier run (`last.pt` is removed once a run finishes) |
| `<run>/model_info.json`, `history.csv` columns `epoch_seconds`, `peak_mem_gb`, `gpu` | compute cost of every segmentation and classifier run: parameter count, wall time and peak GPU memory per epoch, GPU type (the segmentation table reports them next to inference time per case). For epochs trained before cost logging existed, the tables and curves fill the gaps from the timestamped epoch lines in `log.txt`; such training times are marked "~" |
| `<seg run>/per_case_segmentation_test.csv`, `summary_*.csv` | per-patient Dice / IoU / HD95 / NSD / sensitivity / precision; bootstrap summary |
| `predictions/<backbone>/` | predicted tumour masks (NIfTI, original grid) of the classification cohort |
| `<cls run>/predictions_test.csv`, `metrics_test.csv`, `val_predictions_best.csv`, `featurizers.pkl` | test probabilities and metrics with CIs; fitted clinical / radiomics preprocessing |
| `<cls run, first seed>/explain/` | `attention.csv`, `occlusion.csv` (+ summaries and plots), `gradcam/<patient>_<task>_<map>.png` |
| `deep_features/`, `radiomics/` | label-free deep features and radiomics of the cohort |
| `kfold_cv/` | `cv_results.csv`, `holdout_results.csv`, `holdout_predictions.csv`, `holdout_shap.csv` |
| `paper_tables/`, `paper_figures/` | all tables (CSV / Markdown / LaTeX) and figures |

### About the k-fold late fusion

The k-fold CV runs over all 162 cohort patients, and every imputer, scaler, feature selector and
classifier is refit inside each training fold. The deep features come from the segmentation SwinUNETR,
which was trained with tumour masks only, so no class label leaks across folds. It *has* seen the images
(and masks) of the classification train/val patients during segmentation training. State this, and use
the **hold-out** result (fit on train+val, scored once on the 25 test patients no model has seen) as the
fully clean confirmation. XGBoost uses the thesis hyperparameters (200 trees, depth 4, learning rate
0.05) with balanced sample weights. Change them in `configs/kfold_cv.yaml` only before looking at results.

## 3. Compute budget

Estimated from your thesis Colab logs (T4, 15 GB). An A100 or L4 is roughly 2–3× faster.

| Stage | Per run | Runs in the default plan | Total |
|---|---|---|---|
| SSL, 50 epochs (about 13 min/epoch) | ~11 h | 1 | ~11 h |
| SwinUNETR (scratch or SSL), 100 epochs, val every 5 | ~11 h | 2 | ~22 h |
| SegResNet | ~4 h | 1 | ~4 h |
| 3D U-Net (wider + thesis configuration) | ~4–6 h | 2 | ~10 h |
| SegMamba (+ one-time CUDA build, ~10–20 min) | ~8 h | 1 | ~8 h |
| Predicted masks + features | ~0.5 h | 1 | ~0.5 h |
| Classifier, 100 epochs (patience 25 often stops earlier) | ~2–4.5 h | 9 experiments × 1 seed (default) / × 3 seeds | ~18–40 h / ~55–120 h |
| Explanations (Grad-CAM per test patient) | ~10–20 min | 8 | ~2 h |
| k-fold CV (CPU, with MLP) | ~0.5–1 h | 1 | ~1 h |

With the default single seed, roughly 85–130 T4-hours in total (about 45–65 on an L4, 25–40 on an
A100). Going to 3 classifier seeds adds about 35–65 T4-hours, for roughly 130–200 in total. With this many runs an L4 or A100
is worth it (roughly 65–100 or 30–60 h). Ways to trim, in order of least harm (the segmentation
benchmark and the classifier variants all stay in):

1. the five thesis-variant classifiers with 2 seeds instead of 3 (`seeds: [42, 43]` on each of those
   experiments in `configs/paper.yaml`), keeping 3 for the main comparison;
2. SSL to 30 epochs;
3. classifier epochs to 60.

**On Colab.** A T4 (15 GB) is enough: the thesis ran the same model sizes on one, and here fp16
mixed precision and gradient checkpointing are on. An L4 is roughly 2× and an A100 3–5× faster with
identical hyperparameters, so results don't depend on the GPU. You can switch GPU type between sessions;
runs resume across GPU types and the cached SegMamba wheels cover all three. The notebook scales data
loaders to the VM's CPUs.
- Free-tier sessions end after at most ~12 h (often sooner) and GPU time is rationed, so the run spans
  many sessions. Each one is: setup cells → stage cell.
- Checkpoints are written to a temporary file and then renamed, so a disconnect during a save leaves the
  previous one intact. Each finished run keeps only `best.pt` (`keep_last_checkpoints: false`). Drive needs about 4 GB in total,
  plus about 1 GB for whichever run is in progress.
- Run the notebook's preflight cell first: it checks the GPU, every path, file counts, Drive space and
  MONAI compatibility.

Extra segmentation seeds (`segmentation.seeds: [42, 43, 44]`) cost the most. For the top two models
they make the comparison much stronger. The plan is resumable, so you can run it across several Colab
sessions.

## 4. Reporting checklist

- Report split sizes and class counts (`make_splits.py` prints them), and say the split is fixed and
  patient-level.
- Report every test number with its 95% CI. With n = 25 classification test patients an accuracy CI is
  about ±0.19. Say so, and lean on the k-fold results for the classification comparison.
- Segmentation: mean ± SD over seeds, bootstrap CI, Wilcoxon vs best with Holm correction. Do not rank
  models whose differences are not significant.
- Classification: always show the no-imaging baselines next to the imaging models (majority, age + sex,
  radiomics). Present the leaky reference as "uses post-operative molecular / surgical data" and keep it
  out of the model comparison.
- Use the test split once per final configuration, and choose nothing (epochs, thresholds, feature sets)
  by looking at it.
- Limitations to state: single centre; T2 only; 162-patient balanced cohort (339 grade-4 patients
  excluded); classification crops trained on ground-truth masks; segmentation backbone saw the
  classification train images; small test set.
- Expect lower classification numbers than the thesis. That is the point of the correction. The
  segmentation numbers should be close to the thesis, because the corrections there are mostly about
  evaluation.
