# Analysis plan: the grade-4 patients outside the classification cohort

Written on 4 November 2026, before any classifier has scored these patients. The commit that adds this file is the
plan's date stamp, and its hash goes into every output of the analysis.

## 1. Why this analysis

The classification cohort holds 162 of the 495 patients. The other 333 are all WHO grade 4, because grade 4 was
subsampled to balance the classes. No classifier has been trained, selected or evaluated on them, and they have
survival data. The 25 classification test patients have been looked at repeatedly, so these 333 are the only unspent
labelled data in UCSF-PDGM.

Two questions follow:

1. **Prognosis (primary).** Among IDH-wildtype glioblastomas, do tumours that look more like high grade on T2 have
   shorter survival, beyond known prognostic factors? Put the other way: do glioblastomas that the model would call
   low grade live longer?
2. **Validation (secondary).** On patients no classifier has seen, how often do the models call grade 4 correctly, and
   how well do they separate IDH status?

## 2. What has already been seen

- **Metadata summaries of these patients:** counts, deaths, median survival, MGMT and extent-of-resection frequencies,
  and descriptive survival plots of the whole dataset in the metadata notebook.
- **Images and masks, without class labels:** all 333 are in the segmentation split, so they were used for
  segmentation or self-supervised training, for choosing the segmentation checkpoint, or for the segmentation test.
  The per-patient Dice of the 50 in the segmentation test is known.
- **Where the hypothesis came from:** one classification test patient, a 38-year-old with glioblastoma, called low
  grade by all nine end-to-end models and alive at 1,913 days. That patient is in the 162-patient cohort and is not in
  this analysis.

**Not seen:** any classifier output for these patients, and any link between such output and survival.

## 3. Patients

The metadata has 501 rows: 495 baseline scans and 6 follow-up scans of patients already present. Follow-up scans are
excluded. The 63 grade-4 patients in the cohort were drawn at random, so the 333 are the random remainder of grade 4.

| Set | Definition | Patients | Deaths |
|---|---|---|---|
| Outside the cohort | Baseline scans not in the 162-patient cohort | 333 (311 glioblastoma, IDH-wildtype; 22 astrocytoma, IDH-mutant) | 198 |
| **A: primary** | Glioblastoma, IDH-wildtype, with recorded survival | 310 | 193 |
| A-unseen | A, not in the segmenter's training set (44 validation, 48 test) | 92 | 56 |
| A-seen | A, in the segmenter's training set | 218 | 137 |

The one glioblastoma with no recorded survival is excluded from the survival analyses only.

## 4. Frozen models and scores

Every model below learns from the 162-patient cohort only, with the settings in `configs/kfold_cv.yaml` and the
existing classifier configs. Nothing is tuned, reselected or refit using the 333.

| Score | Definition | Role |
|---|---|---|
| **R** | P(HGG) from XGBoost on radiomics alone, LGG/HGG task, fit on all 162 patients (recursive feature elimination to 30 features inside the pipeline, balanced sample weights) | **Primary** |
| D | The same pipeline on deep features alone (feature set `image`, ANOVA top 32) | Secondary |
| E | P(grade 4) from end-to-end `cls_image_only`, as already trained on the cohort's training split: the ensemble of every seed finished at the freeze (step 2 in section 9), flip-averaged | Secondary |

None of the three uses age, sex or any other clinical input, so a survival association cannot simply restate age.

Features and crops use the shared backbone's predicted masks (`seg_segmamba_lr5e5_s42`, largest connected component),
as in the rest of the study. An empty predicted mask follows the existing rule (crop at the volume centre, radiomics
imputed) and the patient stays in the analysis.

For survival models, each score is logit-transformed after clipping to [0.001, 0.999], then standardised over the
analysis set.

## 5. Questions and estimates

### Primary

**Q1.** On set A, the hazard ratio per 1 SD of score R from a Cox model adjusted for age (per 10 years), sex, MGMT
status (positive, negative, unknown or indeterminate) and extent of resection (gross total, subtotal, biopsy). Report
the estimate, its 95% CI and the two-sided Wald p-value, with alpha 0.05. The expected direction is a hazard ratio
above 1.

### Secondary (Holm correction across these four)

- **Q2a.** Patients called low grade against high grade by score R (P(HGG) < 0.5 or not): Kaplan–Meier curves, log-rank
  test, adjusted hazard ratio. If the smaller group has fewer than 20 patients or 10 deaths, report it descriptively
  and leave it out of the correction.
- **Q2b.** Added prognostic value: likelihood-ratio test and change in Harrell's C-index (2,000 patient bootstrap
  resamples) between the Cox model of clinical covariates alone and that model plus score R.
- **Q2c.** The Q1 model with score D.
- **Q2d.** The Q1 model with score E.

### Validation

- **Q3. Grade-4 recall.** On all 333: the share called high grade, with Wilson 95% CIs, for R, D, E, an age + sex
  XGBoost baseline, late-fusion image + clinical + radiomics, and end-to-end `cls_image_clinical_radiomics`. Report
  patients in and not in the segmenter's training set separately. Everyone here is grade 4, so this measures
  sensitivity only: read it next to each model's specificity from the 162-patient cohort, because always answering
  "high grade" would score 100%.
- **Q4. IDH status.** On all 333 (22 mutant, 311 wildtype): ROC AUC with DeLong CIs and average precision for XGBoost
  fit on the 162 patients with age + sex, radiomics, clinical + radiomics, and image + clinical + radiomics, with
  logistic regression alongside. Compare each imaging set with age + sex by DeLong's test, Holm-corrected. With 22
  mutants, expect wide intervals.
- **Mask quality.** Dice against the expert masks, and its relation to tumour volume, for patients in and not in the
  segmenter's training set.

## 6. Statistical details

- **Survival time and event:** `OS` in days and `1-dead 0-alive` from the metadata, with alive patients censored at
  their recorded time. Report median follow-up by reverse Kaplan–Meier. The start of the survival clock must be
  confirmed from the dataset paper and stated; it does not change the analysis.
- **Ties:** Efron's method.
- **Proportional hazards:** test by Schoenfeld residuals. If violated for the score, also report the difference in
  restricted mean survival time at 2 years between the Q2a groups, and keep the Cox estimate as an average effect.
- **Non-linearity:** check the score with a restricted cubic spline (3 knots) and report the check.
- **Missing data:** unknown or indeterminate MGMT (14 patients in set A) is its own category. No other covariate is
  missing in set A.

## 7. Sensitivity analyses (reported in full, none replaces Q1)

| | Change to Q1 | Question |
|---|---|---|
| S1 | A-unseen only (92 patients, 56 deaths) | Does it hold where the masks are honest? |
| S2 | Adjust for age, sex and MGMT without extent of resection; also unadjusted | Extent of resection may partly follow from the tumour's appearance |
| S3 | Add log predicted tumour volume | Is the score mostly tumour size? |
| S4 | Compute R from the expert masks, same frozen model | What do mask errors cost? |
| S5 | Exclude predicted masks with Dice below 0.5 | Are failed masks driving it? |
| S6 | Exclude patients biopsied before the MRI (48 in set A) | Does prior biopsy change the appearance? |
| S7 | All outside-cohort patients with survival, adding IDH status as a covariate | Does it hold beyond IDH-wildtype? |
| S8 | Logistic regression instead of XGBoost | Does it depend on the classifier? |

## 8. Power

From Schoenfeld's formula: two-sided alpha 0.05, 80% power, ignoring covariates.

| Set | Deaths | Smallest detectable HR per 1 SD | Smallest detectable HR, 20/80 split |
|---|---|---|---|
| A | 193 | 1.22 | 1.66 |
| A-unseen | 56 | 1.45 | 2.55 |

## 9. Order of work

1. Commit this plan. Optionally register it publicly (for example on OSF), which is a stronger date stamp than git.
2. Finish the planned cohort runs, then freeze the models and seeds in section 4.
3. Write the code and test it on synthetic data only (section 11).
4. Compute masks, features and scores for the 333. Save them to a file holding only patient IDs and scores, before any
   label or outcome is joined.
5. Run the analysis once, on the frozen scores. Outputs record this plan's commit hash.
6. Report every result in this plan. Record any deviation, with its reason, in section 12.

## 10. How the results will be read

- **Supported:** the Q1 hazard ratio is above 1, its CI excludes 1, and S1 points the same way (S1 need not be
  significant).
- **Mostly size:** the hazard ratio shrinks substantially in S3. Then say that the association mainly reflects tumour
  size.
- **Not supported:** the CI includes 1. Report the CI as the bound on the effect, and do not call this "no effect".
- **Opposite direction:** report it as found.
- **In every case:** no new thresholds, feature sets, covariates or primary questions after step 5. Anything added
  later is labelled exploratory.

Whatever the result, this is same-centre validation, not external validation.

## 11. Code to write

- A `fresh` patient group: segmentation patients minus classification patients, from the existing split file.
- Scripts extended to that group, with separate outputs:
  - `evaluate_segmentation.py`: masks and `per_case_fresh.csv`. Do not reuse `--stage segmentation`, which would
    overwrite the benchmark's `per_case_segmentation_*` files.
  - `extract_radiomics.py` and `extract_deep_features.py`: predicted and expert masks.
  - `kfold_cv.py`: fit on all 162 patients and save scores for the 333.
  - `evaluate_classifier.py`: end-to-end predictions with the saved featurizers.
- `scripts/fresh_cohort_analysis.py`: everything in sections 5–7, tested on synthetic survival data. New dependency:
  `lifelines`.
- A `fresh` stage in `scripts/run_paper.py`, left out of the default stages, so that it can only run when asked for.

## 12. Limitations and deviations

**Known limitations:**
- Same centre, T2 only.
- Everyone is grade 4, so this says nothing about specificity or about grades 2 and 3.
- 218 of the 310 patients in set A were in the segmenter's training set, so their masks are optimistic (hence S1).
- The metadata has no treatment data beyond extent of resection, so treatment is an unmeasured confounder.
- Extent of resection and MGMT come from surgery and tissue. Here they are covariates for a survival outcome, not
  inputs for predicting a label. Reading a score's added value against them is legitimate; using them to predict
  grade or diagnosis is not.

**Deviations from this plan:** none yet.
