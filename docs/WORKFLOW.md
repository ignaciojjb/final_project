# Complete workflow

[Documentation index](../README.md#documentation) · [Setup](SETUP.md) · [Troubleshooting](TROUBLESHOOTING.md)

Run from `final_project` after installing dependencies. Execute one stage at a time; stop if a command fails. This walkthrough downloads 1,200 candidates per class and prepares **1,000 per class** (2,000 objects). Each stage uses explicit matching paths. Existing samples/runs are protected: choose new names rather than deleting earlier work.

If you already have a sample made with the old hips2fits FOV bug, use the [migration commands](../DOWNLOAD_GUIDE.md#a-repair-an-existing-sample-without-resampling-it-powershell) instead of resampling. That route preserves its object IDs and splits.

## 1. Download the catalog

```powershell
.\.venv\Scripts\python.exe -m scripts.download_catalog `
  --per-class 1200 --page-size 500 `
  --output datasets/raw/sdss_catalog_v2.csv
```

This uses the public SkyServer API with explicit columns and a DR7/DR8 identity join. It downloads legacy DR7 photometry hosted in DR19, plus Galaxy Zoo labels; it does not execute the original SciServer/MyDB script. Wait for both class counts and a final `Saved` message.

Outputs: the CSV and `datasets/raw/sdss_catalog_v2.provenance.json`, containing executed queries, source, selection and units. The first qualifying IDs are collected per class, so this pool is not a random sample of the whole survey. Catalog download is not resumable: records are held until completion and a failure requires rerunning that stage. Reduce `--page-size` if the public API times out.

## 2. Clean, sample and assign splits

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_data `
  --catalog datasets/raw/sdss_catalog_v2.csv `
  --per-class 1000 --seed 42 `
  --output datasets/processed_datasets/sdss_sample1000_v2.csv `
  --database datasets/images/sdss_sample1000_v2.sqlite
```

The downloader already applies the baseline quality cuts; preparation checks them again. Preparation removes exact duplicate rows, rejects missing/conflicting IDs, excludes uncertain labels and chooses the requested objects per class. It creates stratified 70% train, 15% validation, 15% test assignments when no assignments exist. For this sample, expect 700/150/150 objects per class before any image-download losses. Existing complete assignments are retained, not reshuffled.

Outputs: a prepared CSV with labels/splits and a SQLite manifest of pending image requests. The database exists at this point **without downloaded image BLOBs**. It is not yet ready for feature extraction.

## 3. Download the images

```powershell
.\.venv\Scripts\python.exe -m scripts.download_images `
  --database datasets/images/sdss_sample1000_v2.sqlite `
  --workers 4
```

Requests are centered on the photometric coordinates. Training stamps are 64×64 pixels with a field of view based on the Petrosian radius. The code converts arcseconds to degrees for hips2fits. Image decoding and dimensions are checked, but that is not a check of galaxy identity or isolation.

If any downloads fail, rerun this same command. Downloaded rows are skipped; pending and error rows are retried. Do not proceed past unresolved failures without inspecting them. Check status using an existing database:

```powershell
.\.venv\Scripts\python.exe -m scripts.cutouts `
  --database datasets/images/sdss_sample1000_v2.sqlite --status
```

A successful download writes JPEG BLOBs into SQLite, not thousands of loose files. Keep its associated CSV and splits together. Avoid changing the database while extracting features.

## 4. Extract and cache features

```powershell
.\.venv\Scripts\python.exe -m scripts.extract_features `
  --catalog datasets/processed_datasets/sdss_sample1000_v2.csv `
  --database datasets/images/sdss_sample1000_v2.sqlite `
  --output artifacts/features_sdss_sample1000_v2.joblib `
  --components 64 --seed 42
```

This joins catalog and image IDs, converts images to grayscale [0,1] pixels, fits PCA on training rows, derives catalog features, fits training-only imputation/scaling and saves the merged matrix with its fitted preprocessing. At least 64 training objects are needed for 64 components. Look for a final `Saved ... (rows, features)` message.

For the standard downloaded catalog, four colors, one brightness and two structural features are available: normally 64+7 model inputs. Actual names/formulas are recorded in artifacts; compatible external catalogs can supply different subsets.

Reexecuting this command reuses a matching cache. A mismatch in source files, tracked implementation hashes, seed or component count is rejected. The fingerprint includes the SQLite file and WAL if present, so even database storage changes can require another cache. Use a new cache path deliberately; do not keep trying training when extraction has failed.

## 5. Train

```powershell
.\.venv\Scripts\python.exe -m scripts.train_model `
  --features artifacts/features_sdss_sample1000_v2.joblib `
  --C 1 --gamma scale `
  --output artifacts/runs/svc-sdss-v2-001
```

This fits the SVC on cached training rows and evaluates validation/test separately. It does not redownload, decode images or refit PCA. A new run directory contains the model, metrics, held-out predictions and merged feature CSV. See [artifact definitions](RESULTS.md#saved-training-artifacts).

## 6. Plot metrics, ROC and correlations

```powershell
.\.venv\Scripts\python.exe -m scripts.plot_results `
  --run artifacts/runs/svc-sdss-v2-001

explorer artifacts\runs\svc-sdss-v2-001\plots
```

This generates confusion matrices, overall/per-class scores, the first-two-component projection, training-feature correlations and held-out ROC curves/AUC. It does not train a model. Interpretation and output names are in [Results](RESULTS.md).

## 7. Compare images and predictions

```powershell
.\.venv\Scripts\python.exe -m scripts.plot_predictions `
  --run artifacts/runs/svc-sdss-v2-001 `
  --database datasets/images/sdss_sample1000_v2.sqlite `
  --count 10 --seed 42 --display-resolution 256

Invoke-Item artifacts/runs/svc-sdss-v2-001/plots/predictions_test.png
```

Only the selected objects get higher-resolution display downloads. They are cached separately inside the run's `plots/display_cache/`; training stays 64×64. Use `--split validation` during model exploration, `--errors-only` to inspect mistakes or `--seed 7` for another selection. Failed display downloads fall back to the original stamp with a warning. Each panel shows its actual display resolution. Inspect neighboring objects and clipping manually; a single centered galaxy is not guaranteed.

## 8. Retrain without repeating preparation

```powershell
.\.venv\Scripts\python.exe -m scripts.train_model `
  --features artifacts/features_sdss_sample1000_v2.joblib `
  --C 10 --gamma scale `
  --output artifacts/runs/svc-sdss-v2-002
```

Compare validation results to choose parameters. The code exports test results for each run, but test must not guide repeated choices. These scripts perform no automated grid search, cross-validation or best-model selection.

| Change | Stages to repeat |
|---|---|
| Plot layout, gallery seed/count or display resolution | Plotting only |
| SVC `C` or `gamma` | Training into a new run, then plots |
| PCA components, extraction seed or photometric definitions | Extract into a new cache, train, plot |
| Corrected/replaced training images | Finish image stage, extract a new cache, train, plot |
| Different objects or split assignments | Prepare a new sample/database, obtain images, extract, train, plot |
| Old hips2fits FOV requests | Migrate manifest, redownload, extract, train, plot |

## Larger or smaller samples

For a first download check, use 120 catalog candidates and prepare 100 objects per class. Use distinct `sample100` paths consistently through all stages. For the standard sample, 1,000 per class is moderate. Both sizes support 64 components if training downloads complete.

Increasing sample size means requesting enough candidates from the catalog first, then increasing `prepare_data --per-class`. Repreparing creates new assignments; it is not a controlled learning-curve comparison. Keeping validation/test IDs fixed while expanding training would require an additional selection workflow not implemented here. Separate image databases do not share a download cache, even when some IDs overlap. Avoid selecting the full survey for nonlinear SVC without assessing memory and runtime.

## Alternative inputs: your own CSV or ZIP

Use your CSV as `prepare_data --catalog`; its schema must match the [input contract](DATA_AND_MODEL.md#catalog-input-contract). `--already-clean` skips quality cuts only, not ID/label/split checks. Do not use it to hide unknown quality issues.

For a known HTTPS file and its independently supplied SHA256 checksum:

```powershell
.\.venv\Scripts\python.exe -m scripts.download_data `
  --url "https://HOST/catalog.csv" --sha256 "EXPECTED_SHA256" `
  --output datasets/raw/external_catalog.csv
```

Those two placeholders are for your external source, not the public query downloader in step 1. The file downloader checks existing files/checksums and does not resume partial transfers.

To import existing images after creating a manifest, replace step 3 with:

```powershell
.\.venv\Scripts\python.exe -m scripts.import_stamps `
  --archive datasets/raw/ZooStamps.zip `
  --mapping datasets/raw/image_mapping.csv `
  --database datasets/images/sdss_sample1000_v2.sqlite
```

The mapping CSV needs `specobjid,image_path`, one unique row per image, with the exact member path inside the ZIP. Do not infer IDs from filenames without verifying them. The importer skips objects outside the manifest and already downloaded objects, verifies image decoding, and limits each image to 20 MiB. Imported images may have different angular coverage or provenance; the mapping alone does not validate this. Record the source yourself. Never import an unreviewed archive as evidence that all training fields have correct coverage.
