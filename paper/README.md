# Paper draft

`main.tex` + `references.bib`. The numbers are the one-seed results in `paper_runs/` (3 Nov 2026); red `[TODO: ...]`
markers flag what still has to change, most importantly the results once seeds 43 and 44 have run.

## Build

The figures are read from `../paper_runs/paper_figures/` (the Colab output folder, not in git):

```bash
latexmk -pdf main.tex
```

On Overleaf, upload `main.tex`, `references.bib` and these files from `paper_runs/paper_figures/`, then change
`\graphicspath` to the folder you put them in:

- `segmentation_dice.png`, `segmentation_cases.png`, `segmentation_dice_vs_volume.png`
- `cls_image_only_diagnosis_confusion.png`, `cls_image_clinical_radiomics_diagnosis_confusion.png`,
  `cls_leaky_reference_radiomics_diagnosis_confusion.png`
- `kfold_shap_diagnosis_image_clinical_radiomics_top15.png`
- `explain/cls_image_clinical_radiomics_gradcam_panel_diagnosis.png` (regenerate first, see the TODO in the caption)

The tables are written by hand from `paper_tables/summary.md`, `kfold_cv/cv_results.csv` and the per-run
`predictions_test.csv` / `val_predictions_best.csv`; re-check them against those files after every rerun.
Every bibliography entry still needs checking against the publisher's page.

`reporting_checklists.md` maps the draft to TRIPOD+AI and CLAIM 2024 and lists the gaps to close before submission.
