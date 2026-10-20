# Galaxy classifier: current algorithm

See [DOWNLOAD_GUIDE.md](DOWNLOAD_GUIDE.md) for source tables, units, the corrected cutout requests, selected feature formulas and PowerShell commands. Earlier hips2fits images with incorrect FOV units must be replaced before extracting features or training.

The model combines 64×64 grayscale image pixels reduced to 64 PCA components with SDSS photometry engineered by `scripts/photometry.py`: extinction-corrected model colors, extinction-corrected r brightness (cModel when supplied, otherwise Petrosian), positive half-light radius, concentration and optional fracDeV. Features absent from the source are not invented. Magnitude errors and extinction are not standalone default predictors. Catalog labels/probabilities, object IDs and split metadata are excluded.

`load_dataset` creates the pixel/engineered-feature DataFrame, checks image-label consistency and rejects known legacy FOV requests. Prepared train/validation/test assignments are preserved. PCA, median imputation and scaling are fitted on training rows only; their outputs feed an RBF SVC with balanced class weights. `scripts.extract_features` saves this preprocessing and the merged matrix once; `scripts.train_model` trains new classifiers from that cache. Use validation for parameter selection and reserve test for the final evaluation.

Run from the repository root:

```bash
python -m scripts.extract_features --catalog YOUR_PREPARED_CSV --database YOUR_CORRECTED_DATABASE --output artifacts/features_v2.joblib --components 64
python -m scripts.train_model --features artifacts/features_v2.joblib --output artifacts/runs/svc-v2-001
```

At least 64 training rows are required for 64 PCA components. Use a smaller component count only for tiny workflow checks. The normal starter sample is 1000 objects per class, not the entire survey. SVC and the full pixel matrix are not designed here for hundreds of thousands of galaxies.

Each run saves `model.joblib`, `metrics.json` (including feature formulas), `predictions.csv` and `merged_features.csv`. The saved model expects the same 4096 pixel columns and engineered catalog columns created by `load_dataset`; it does not accept a raw CSV row or JPEG alone. Load joblib only from trusted sources. Retain metadata for auditing but exclude IDs/labels/splits if using the exported table for another classifier.

Plot existing results using `scripts.plot_results`, `scripts.plot_correlations` and `scripts.plot_predictions`. Higher-resolution selected gallery images are cached separately and never used to train. Reference classes are Galaxy Zoo catalog labels, not independently verified truth.

Tests: `python -m unittest discover -s tests -v`. Synthetic tests establish implementation behavior, not scientific accuracy. A separate live smoke check downloaded 12 real objects and their corrected cutouts and ran a two-component model plus plots; that tiny check is not a performance estimate.
