# Galaxy classification: photometry + stamp PCA + SVC

Run from the project root with Python 3.11+:

```bash
python -m pip install -r requirements-ml.txt
python scripts/ml_training.py \
  --catalog datasets/processed_datasets/galaxies_clean.csv \
  --database datasets/images/cutouts_cnn.sqlite \
  --output artifacts/svc \
  --components 64
```

The inputs are the cleaned CSV produced by `data_processing.ipynb` and the SQLite database produced by `data_preparation.ipynb` using `cutouts.py`. They are not included in the uploaded source files. Run those existing preparation steps first if the data is unavailable. This training script makes no network requests and needs no SciServer credentials.

## Algorithm

1. Read downloaded JPEG/PNG BLOBs from `cutouts`. Convert to grayscale, resize to 64×64 (including the existing 128×128 cutouts), scale to [0,1], and flatten to 4,096 pixels per object.
2. Join to the catalog using exact string `object_id` ↔ `specobjid`. IDs must not have been rounded through floating-point conversion. Duplicate or null catalog IDs fail. Only objects with unambiguous spiral/elliptical flags are eligible; unmatched images are counted and excluded, and conflicting labels fail.
3. Reuse the notebook's train/validation/test assignments. If every assignment is missing, create a reproducible stratified 70/15/15 split. Partial assignments fail rather than silently reshuffling the held-out data. Each partition must contain both classes.
4. Fit PCA with 64 components on training stamps only, then standardize those components. At least 64 training objects are required; use `--components` for smaller datasets.
5. In parallel, select the available photometric columns from the explicit `CATALOG_FEATURES` allowlist. Replace SDSS -9999 sentinels with missing values, median-impute and standardize using training data only. Entirely missing training columns are filled with zero. Morphology flags, vote probabilities, IDs, and split metadata are excluded from predictors.
6. Concatenate both feature branches and train an RBF SVC with `C=1`, `gamma='scale'`, and balanced class weights. The labels are 0=elliptical and 1=spiral. Validation and test data are evaluated separately and never used for fitting. There is no hyperparameter search; if added later, tune within training/validation and keep test data untouched.

## Outputs

- `model.joblib`: complete fitted preprocessing and classifier pipeline. Load only trusted joblib files. Predict with `model.predict(frame)` on a DataFrame containing the same 4,096 pixel columns and selected catalog columns; `load_dataset` provides this format for another labeled CSV/database pair.
- `metrics.json`: split/class counts, selected catalog columns, seed, PCA explained variance, accuracy, balanced accuracy, per-class precision/recall/F1, and confusion matrices (rows=true, columns=predicted; class order 0,1).
- `predictions.csv`: validation/test IDs, labels and predictions.
- `merged_features.csv`: standardized PCA components plus standardized catalog features, with ID, label and split metadata. **Exclude those three metadata columns if using this table in another model.** No full-resolution pixel CSV is written; the 4,096-feature table is held in memory.

This implementation targets the notebook's balanced 2,000-object starter set. Full-survey image matrices require substantial memory, and nonlinear SVC training scales poorly to hundreds of thousands of objects. Use batching/incremental PCA and a scalable classifier before attempting the entire survey. Real-world duplicate observations of the same galaxy should share one split; this implementation assumes the notebook's one-object-per-ID preparation.

## Validation

```bash
python -m unittest discover -s tests -v
```

The tests use synthetic images/catalogs, exercise the full pipeline and serialization, check exact large IDs and training-only PCA, and reject label disagreement. Synthetic scores do not establish galaxy classification performance. No real catalog or cutout database was supplied, so real-data training and evaluation remain to be run with the command above.
