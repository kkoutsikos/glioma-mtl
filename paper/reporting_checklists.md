# Reporting checklists: where the manuscript stands

Draft mapping of the manuscript to two reporting guidelines. Topics are paraphrased and grouped; before
submission, download the official checklists, copy each item number into the right row, and fill in the page or
line numbers.

- **TRIPOD+AI**: Collins GS, Moons KGM, Dhiman P, et al. *BMJ* 2024;385:e078378 (prediction models built with
  regression or machine learning). Official checklist: tripod-statement.org.
- **CLAIM 2024**: Tejani AS, Klontzas ME, Gatti AA, et al. Checklist for Artificial Intelligence in Medical Imaging,
  2024 update. *Radiology: Artificial Intelligence* 2024. It updates Mongan J, Moy L, Kahn CE Jr, *Radiology: AI*
  2020;2(2):e200029.

Status: **done** = in the draft; **partly** = in the draft but incomplete; **missing** = still to add.

## TRIPOD+AI

| Topic | Where in the manuscript | Status | To do |
|---|---|---|---|
| Title says it is a prediction-model study, names the target population and outcomes | Title | partly | Consider adding "development and internal validation" |
| Structured abstract (objective, data, methods, results with uncertainty, conclusion) | Abstract | done | Add the registration/protocol statement if the journal asks for one |
| Clinical context, rationale, intended use and users of the models | Introduction | partly | State the intended use explicitly: a research benchmark, not a clinical tool |
| Objectives: development and/or validation | Introduction (four questions) | done | – |
| Data source, centre, dates of data collection | Methods: Data | partly | Add the acquisition period and setting from the dataset paper |
| Eligibility criteria and how the cohort was formed | Methods: Data (balanced cohort; 339 grade-4 patients excluded) | partly | **Missing:** why 6 patients were excluded from segmentation |
| Outcome definitions (grade, WHO 2021 diagnosis, LGG/HGG, IDH) and how they were determined | Methods: Data | partly | Say they come from the dataset's histopathology and molecular testing |
| Predictors: definition, timing (pre-operative only), and how they were measured | Methods: Splits, Shared backbone | done | – |
| Sample size: justification | – | **missing** | No formal calculation: justify with the available cohort and report the expected CI width (±0.19 accuracy at n = 25) |
| Missing data: how much, and how it was handled | Methods: Late fusion (imputation inside folds) | partly | **Missing:** counts of missing values per clinical variable; empty predicted masks → imputed radiomics |
| Data preparation (preprocessing, feature extraction) | Methods: Preprocessing, Segmentation, Shared backbone | done | – |
| Analytical methods: model types, hyperparameters, how they were chosen | Methods: Segmentation, End-to-end classifiers, Late fusion (fixed before results) | done | – |
| Class imbalance handling | Methods: End-to-end classifiers (focal loss, class weights); Late fusion (balanced weights) | done | – |
| Data partitioning, internal validation (split, repeated CV, hold-out) | Methods: Splits, Late fusion | done | – |
| Performance measures: discrimination, calibration, clinical utility | Methods: Explanations and metrics; Statistical comparisons | partly | Add the calibration and decision-curve results once `kfold` has rerun |
| Fairness: performance across subgroups (sex, age) | – | **missing** | Subgroup AUC by sex and age band from the out-of-fold predictions (cheap) |
| Model updating / recalibration | – | n/a | Say there was none |
| Results: participant flow and characteristics per split | Table 1 | partly | Add age and sex per split; optionally a flow diagram (501 → 495 → 162 → 112/25/25) |
| Results: model performance with CIs | Tables 2–5 | done | Update with three seeds |
| Interpretation, limitations, implications | Discussion | done | – |
| Model availability (code, weights) | Code and data availability | partly | Release the trained weights (e.g. Zenodo) next to the code |
| Protocol / registration | – | partly | Not registered: cite the protocol's commit in the repository history as the dated analysis plan |
| Funding, conflicts of interest | – | **missing** | Add |
| Patient and public involvement | – | **missing** | State that there was none (public, de-identified data) |
| Ethics approval and consent | – | **missing** | Public de-identified dataset; approval and consent at the source institution (see the dataset paper) |

## CLAIM 2024

| Topic | Where in the manuscript | Status | To do |
|---|---|---|---|
| Title/abstract identify an AI method in medical imaging | Title, Abstract | done | – |
| Study design: retrospective, development/validation, intended use | Introduction, Methods | partly | Say "retrospective" explicitly |
| Data source, inclusion/exclusion, de-identification | Methods: Data | partly | De-identification: done by the dataset providers; the 6 segmentation exclusions |
| Image acquisition: scanner, field strength, sequence parameters | Methods: Data (3 T, T2) | partly | Add the T2 sequence parameters from the dataset paper |
| Preprocessing | Methods: Preprocessing | partly | Confirm and state whether volumes were skull-stripped and co-registered by the providers |
| Reference standard: definition, annotators, their expertise, inter-reader agreement | Methods: Data | **missing** | Describe how the expert segmentations and the histopathology/molecular labels were produced (dataset paper) |
| Data partitions: sizes, patient-level separation, stratification | Table 1; Methods: Splits | done | – |
| Model: architecture, inputs, outputs, intermediate layers used | Methods: Segmentation, Shared backbone, End-to-end classifiers | done | – |
| Initialisation and pretraining (SSL, segmentation backbone) | Methods: Segmentation (SSL), Shared backbone, End-to-end classifiers | done | – |
| Training: loss, optimiser, learning rates, epochs, augmentation, checkpoint selection | Methods: Segmentation, End-to-end classifiers | done | – |
| Software and hardware (library versions, GPU) | Table 2 (A100) | partly | Add the PyTorch, MONAI, scikit-learn and XGBoost versions from the Colab environment |
| Evaluation metrics and why they were chosen | Methods: Segmentation; Explanations and metrics | done | – |
| Statistical methods: CIs, significance tests, multiple comparisons | Methods: Segmentation; Statistical comparisons | done | – |
| Robustness / sensitivity analyses | Methods: Segmentation (learning rate, U-Net configuration); Statistical comparisons (ground-truth masks) | done | Add the learning curves and seeds when available |
| Explainability methods and how they were assessed | Methods: Explanations and metrics; Results: Explanations | done | – |
| Failure analysis | Results: Error analysis (missed glioblastomas, small tumours) | done | Add the Dice-vs-volume result |
| External validation | – | **missing** | Listed as a limitation; the Erasmus Glioma Database is a candidate |
| Performance on subgroups | – | **missing** | Same as TRIPOD+AI fairness: by sex and age band |
| Data and code availability | Code and data availability | partly | Weights; tagged release; the split file |

## Gaps worth closing before submission

1. **Subgroup performance** by sex and age band: computable from `kfold_cv/oof_predictions.csv`, no new training.
2. **Missing-data counts** per clinical variable, and how many classification patients had an empty or near-empty
   predicted mask.
3. **Reference standard and acquisition details**, summarised from the dataset paper (Calabrese et al. 2022).
4. **Software versions** from the Colab session (`pip freeze` of the run).
5. **Ethics, funding, conflicts, patient and public involvement** statements.
6. **The 6 patients excluded from segmentation**: the reason.
