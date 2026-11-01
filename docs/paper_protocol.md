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
- a stronger UNet configuration;
- checkpoint selection on an EMA-smoothed validation loss.

## 2. Experiment matrix

| Step | Runs | Answers | Table |
|---|---|---|---|
| `splits` | 1 | fixed patient-level split (162 cohort: 112/25/25; 495 segmentation: 345/75/75) | data |
| `ssl` | 1 | pretrains the SwinUNETR encoder on SSL train/val only | – |
| `seg` | the 5 thesis architectures (SwinUNETR scratch / +SSL, SegResNet, 3D U-Net, SegMamba) + the 3D U-Net in its thesis configuration, × seeds | which architecture segments whole tumour best on T2; does SSL help (`seg_swin_unetr` vs `seg_swin_unetr_ssl`); was the thesis U-Net result a configuration problem (`seg_unet` vs `seg_unet_thesis`) | `segmentation` |
| `predict` | backbone on classification train/val/test | predicted masks for crop localisation, deep features, radiomics | – |
| `features` | 1 + 1 | label-free deep features, radiomics from predicted masks | – |
| `cls` | 3 experiments × 3 seeds | what imaging adds for grade / diagnosis; what molecular data explain (reference) | `classification` |
| `kfold` | repeated 5-fold × 10 + hold-out | XGBoost / logistic regression / RF late fusion of demographics, radiomics and deep features; tabular baselines | `kfold_cv`, `kfold_holdout` |
| `aggregate` | – | writes all tables (CSV, Markdown, LaTeX) | – |

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
| Classifier, 100 epochs (patience 25 often stops earlier) | ~2–4.5 h | 9 | ~20–40 h |
| k-fold CV (CPU) | minutes | 1 | – |

Roughly 75–95 T4-hours for one segmentation seed. Ways to trim, in order of least harm (the
segmentation benchmark keeps all thesis models):

1. SSL to 30 epochs;
2. 2 classification seeds;
3. classifier epochs to 60.

**On Colab.** A T4 (15 GB) is enough: the thesis ran the same model sizes on one, and here fp16
mixed precision and gradient checkpointing are on.
- Free-tier sessions end after at most ~12 h (often sooner) and GPU time is rationed, so the run spans
  many sessions. Each one is: setup cells → stage cell.
- Each finished run keeps only `best.pt` (`keep_last_checkpoints: false`). Drive needs about 4 GB in total,
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
