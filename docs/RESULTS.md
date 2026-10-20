# Results, plots and interpretation

[Documentation index](../README.md#documentation) · [Workflow](WORKFLOW.md) · [Model definitions](DATA_AND_MODEL.md)

## Saved training artifacts

A run is the directory supplied to `scripts.train_model --output`. It is not overwritten. Use a new run name for every model fit, and use that same run path for plotting.

| Artifact | Contents |
|---|---|
| `model.joblib` | Fitted preprocessing plus SVC; load only trusted artifacts |
| `metrics.json` | SVC settings, feature-source signature, selected catalog features/formulas, validation/test metrics including AUC in current runs |
| `predictions.csv` | Held-out `specobjid`, `label`, `split`, `prediction` and, in current runs, `decision_score` |
| `merged_features.csv` | All matched rows: ID, label, split and transformed model features |
| `plots/` | Generated figures, correlation/ROC data and optional high-resolution display cache |

The cache supplied to training is a separate `.joblib` file containing `X`, row metadata, fitted preprocessing, selected feature names/formulas and source signatures. Do not manually edit it. `metrics.json` structures differ slightly for the alternative all-in-one `ml_training` entry point, which also reports split counts and summed PCA explained variance; `train_model` does not currently export those extra fields. A standard cached run's split/class counts can be read from `merged_features.csv`.

Columns `stamp__pca0`, `stamp__pca1`, … are **standardized image PCA components**. Columns prefixed `catalog__` are imputed/standardized catalog features. Do not pass IDs, labels or split names to a new classifier. Feature correlations describe these transformed values; their CSV is not a raw-magnitude correlation table.

## Generate the full plot set

```powershell
.\.venv\Scripts\python.exe -m scripts.plot_results `
  --run artifacts/runs/svc-sdss-v2-001

explorer artifacts\runs\svc-sdss-v2-001\plots
```

Use your actual run path, for example `artifacts/runs/svc-sample1000-fixed-001` for a migrated sample. `--output` can direct the figures to another folder. Existing figure files with the same names are replaced; original model/training artifacts are not changed. All figures are standalone PNGs, so a desktop Matplotlib window is not needed.

| Figure / file | How to read it |
|---|---|
| `confusion_matrices.png` | Separate validation/test counts; rows are reference labels and columns predictions, ordered elliptical then spiral |
| `scores.png` | Overall accuracy and balanced accuracy (mean recall across classes) |
| `per_class_scores.png` | Precision, recall and F1 for each class and held-out split |
| `pca_projection.png` | First two standardized image PCs, colored by label and separated by split; only produced with at least two PCs |
| `feature_correlations.png` / `.csv` | Pearson correlations among all classifier inputs, using training rows only |
| `catalog_correlations.png` / `.csv` | The catalog-only block of that correlation matrix |
| `roc_curves.png` | Separate validation/test ROC curves, AUC in the legend and diagonal chance baseline |
| `roc_auc.json` | Numerical AUC and positive/negative sample counts by split |
| `roc_curve_points.csv` | FPR, TPR and decision-score thresholds for each curve |

Precision is the fraction of predicted members of a class that have that reference label; recall is the fraction of reference members recovered. F1 balances precision and recall. Examine class counts with scores, especially on a small test set. A zero-division class metric is reported as zero by the configured metric function.

PCA is unsupervised and the SVC uses all merged features, so the 2D projection is not its decision boundary. PCA components should be approximately uncorrelated in training by construction. Correlation is not nonlinear feature importance. Gray heatmap cells mean undefined correlations, typically from constant/imputed columns, not zero correlation.

## ROC and AUC

```powershell
.\.venv\Scripts\python.exe -m scripts.plot_roc `
  --run artifacts/runs/svc-sdss-v2-001
```

ROC means receiver operating characteristic; AUC is the area under that curve. Positive class = **spiral (1)**; negative = elliptical (0). The x axis is false-positive rate and the y axis is true-positive rate (spiral recall). Curves use continuous **SVC decision scores**, not hard class predictions. Larger scores favor class 1 but are not calibrated probabilities. ROC sweeps a threshold; it does not change the stored predictions or train a new model. AUC 0.5 is the chance ranking baseline and 1.0 is perfect ranking on that particular sample. Both classes must be present in each evaluated split.

Current runs save scores directly. For older runs lacking them, the script loads the trusted saved model and aligns predictions to the transformed feature table by ID, verifying labels and splits. It applies only the fitted SVC, avoiding a second PCA/scaling pass. No images or original cache are needed for this fallback. Old `metrics.json` is left intact; computed AUC goes into `plots/roc_auc.json`. The first ROC threshold may be infinity, representing the point that classifies no object as positive.

These plots provide no AUC confidence interval or calibration analysis. Compare candidate settings on validation and reserve test for a final assessment. Repeatedly choosing models based on test AUC makes that test an informal validation set.

## Image prediction gallery

```powershell
.\.venv\Scripts\python.exe -m scripts.plot_predictions `
  --run artifacts/runs/svc-sdss-v2-001 `
  --database datasets/images/sdss_sample1000_v2.sqlite `
  --count 10 --split validation --seed 42 --display-resolution 256

Invoke-Item artifacts/runs/svc-sdss-v2-001/plots/predictions_validation.png
```

The default split is `test`; using validation during exploration avoids unnecessarily reviewing the final test sample. Titles are green for matching predictions and red for mistakes. The displayed reference type comes from the catalog; it is not independently established truth. A fixed seed reproduces the selection when the eligible prediction rows/order remain unchanged. Add `--errors-only` to choose only mistakes; if fewer than the requested number exist, all eligible rows are shown. If none exist, the script reports that explicitly.

The image database must belong to the trained sample; IDs, labels and stored splits are checked. `--display-resolution 256` downloads **only the selected objects** with the same center/provider/FOV and stores them under the run's `plots/display_cache/`. Repeated identical requests reuse those files. Permitted resolutions are 64–1024 pixels; omitting the flag uses original database images without network access. Display download failures warn and fall back to the original stamp. The actual displayed dimensions appear under each panel.

Neither higher pixel resolution nor a centered request guarantees one isolated galaxy. Inspect clipping/neighbors visually. More detail cannot exceed the underlying survey's resolution. High-resolution images do not modify the SQLite training images, PCA, features, predictions or model.

Default gallery names are `predictions_test.png`, `predictions_validation.png` and their `_errors` variants. Different seeds/resolutions with the same selection mode overwrite that gallery; use `--output artifacts/runs/svc-sdss-v2-001/plots/gallery_seed7.png` if preserving multiple views.

## Comparing experiments

Keep the same sample and splits to compare SVC parameters. Record the run name, Git commit, selected features, PCA settings, SVC settings and validation metrics; keep `metrics.json` and input provenance with every run. The training command evaluates both held-out splits automatically, but only validation should guide selection.

A larger reselected sample changes validation/test membership, so score differences also reflect sampling changes. Correcting the old angular-unit bug and changing photometry together likewise measures their combined effect, not the isolated benefit of one change. Small perfect synthetic or real-sample scores do not establish scientific performance.
