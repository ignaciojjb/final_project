# Script and artifact reference

[Documentation index](../README.md#documentation) · [Workflow](WORKFLOW.md)

Use modules from the repository root: `python -m scripts.MODULE --help`, with your venv's Python. The full workflow uses explicit filenames because historical entry points do not all share the same defaults. This reference describes the code as implemented; there is no single command that silently runs every stage.

## Download and preparation modules

| Module | Main functions | Role |
|---|---|---|
| [download_catalog](../scripts/download_catalog.py) | `build_query`, `query_rows`, `download_catalog` | Public read-only, paginated SDSS query; CSV plus provenance |
| [download_data](../scripts/download_data.py) | `sha256`, `download` | General HTTPS file download with an expected checksum |
| [prepare_data](../scripts/prepare_data.py) | `validate_catalog_ids`, `prepare` | ID checks, quality cuts, balanced selection, splits and image manifest |
| [data_processing](../scripts/data_processing.py) | `clean_sample`, `apply_cut`, legacy `load_data` | Shared baseline quality cuts and cut logs; no CLI |
| [cutouts](../scripts/cutouts.py) | `cutout_url`, `prepare_manifest`, `download_pending`, `status`, `iter_images`, `validate_stored_requests` | Shared SQLite schema, request construction, downloads and legacy-unit checks |
| [download_images](../scripts/download_images.py) | CLI wrapper | Resume pending/error downloads in an existing manifest |
| [import_stamps](../scripts/import_stamps.py) | `import_stamps` | Import ZIP image BLOBs using an explicit ID/member mapping |
| [rebuild_images](../scripts/rebuild_images.py) | `rebuild` | New corrected manifest from a prepared CSV, retaining IDs/splits |

| Command | Options and defaults |
|---|---|
| `scripts.download_catalog` | `--output datasets/raw/sdss_catalog.csv`; `--per-class 1200`; `--page-size 500` (1–1000) |
| `scripts.download_data` | Required `--url`, `--output`, `--sha256` |
| `scripts.prepare_data` | Required `--catalog`; `--output datasets/processed_datasets/galaxies_clean.csv`; `--database datasets/images/cutouts.sqlite`; `--per-class 1000` (0 uses all eligible rows); `--seed 42`; flags `--already-clean`, `--resolve-dr8-matches` |
| `scripts.download_images` | `--database datasets/images/cutouts.sqlite`; `--workers 8`; internally retries pending/error rows with two retries per request |
| `scripts.import_stamps` | Required `--archive`, `--mapping`; `--database datasets/images/cutouts.sqlite` |
| `scripts.rebuild_images` | Required `--catalog`, `--database`; `--provider hips2fits` (or `sdss`) |
| `scripts.cutouts` | Optional positional CSV; `--database datasets/images/cutouts.sqlite`; `--provider hips2fits`; `--workers 16`; optional `--limit`; flags `--retry-errors`, `--status` |

`cutouts` is also an older direct CLI. Its default manifest dimensions are **128×128**, while recommended `prepare_data` and `rebuild_images` request **64×64**. The loader resizes input images to 64×64 either way. Prefer the staged commands for reproducible training. The generic `cutouts --status` helper can initialize a missing database, so check that the file exists before using it as proof of a prepared dataset.

The legacy `data_processing.load_data` drops repeated `specobjid` rows without conflict analysis. The recommended pipeline does not use that helper: it reads IDs as text and validates them in `prepare_data`. Do not use that legacy helper to silently resolve conflicting external records.

## Feature extraction and training modules

| Module | Main functions | Role |
|---|---|---|
| [photometry](../scripts/photometry.py) | `build_photometry`, `FEATURE_DEFINITIONS` | Pure column transformations; no fitted statistics and no CLI |
| [ml_training](../scripts/ml_training.py) | `stamp_to_features`, `load_dataset`, `assign_splits`, `make_pipeline`, `train` | Shared loaders/model construction plus an alternative all-in-one training entry point |
| [extract_features](../scripts/extract_features.py) | `extract` | Training-only preprocessing fit; reusable matrix and transform cache |
| [train_model](../scripts/train_model.py) | `train_cached` | Fit SVC from the cache, evaluate held-out rows, save a new run |

| Command | Options and defaults |
|---|---|
| `scripts.extract_features` | `--catalog datasets/processed_datasets/galaxies_clean.csv`; `--database datasets/images/cutouts.sqlite`; `--output artifacts/features.joblib`; `--components 64`; `--seed 42` |
| `scripts.train_model` | `--features artifacts/features.joblib`; `--output artifacts/runs/svc-001`; `--C 1`; `--gamma scale` (also `auto` or numeric) |
| `scripts.ml_training` | `--catalog datasets/processed_datasets/galaxies_clean.csv`; **`--database datasets/images/cutouts_cnn.sqlite`**; `--output artifacts/svc`; `--components 64`; `--seed 42` |

The recommended path is `extract_features` → `train_model`. The alternative `ml_training` command refits preprocessing each time, uses an older database default and can overwrite its output files in an existing directory. Use explicit paths and a fresh directory if using that entry point. Cached training protects an existing run directory and rejects pre-correction feature-cache versions.

## Plotting modules

| Module | Main function | Options |
|---|---|---|
| [plot_results](../scripts/plot_results.py) | `plot_results` | Required `--run`; optional `--output`; includes scores, confusion matrices, PCA view, correlations, ROC |
| [plot_correlations](../scripts/plot_correlations.py) | `plot_correlations` | Required `--run`; optional `--output`; training-only model/catalog correlations |
| [plot_roc](../scripts/plot_roc.py) | `load_scores`, `plot_roc` | Required `--run`; optional `--output`; held-out continuous-score ROC/AUC |
| [plot_predictions](../scripts/plot_predictions.py) | `display_image`, `plot_predictions` | Required `--run`, `--database`; `--count 10`; `--split test` (or `validation`); `--seed 42`; optional `--errors-only`, `--output`, `--display-resolution` (64–1024) |

`--output` is a directory for metric/correlation/ROC plots, but a **PNG filename** for `plot_predictions`. High-resolution gallery cache location remains the run's `plots/display_cache/` even with a custom PNG output path. Default metric output location is the run's `plots/`. Plotting never retrains; older ROC runs may load their saved trusted model to reconstruct scores.

## Data ownership and repeatability

| Artifact | Writer | Reuse / overwrite behavior |
|---|---|---|
| Raw query CSV and provenance | `download_catalog` | Existing CSV rejected; failed query does not produce a completed CSV; no query resume |
| External HTTPS file | `download_data` | Existing matching SHA256 reused; mismatch rejected; incomplete temporary file removed |
| Prepared CSV and manifest | `prepare_data` | Existing CSV or database rejected; failures late in preparation can leave partial outputs that need inspection |
| New migrated manifest | `rebuild_images` | Existing output database rejected |
| Image BLOBs/status | `download_images` / `import_stamps` | Successful existing images skipped; downloader retries pending/errors |
| Fitted feature cache | `extract_features` | Exact signature match reused; mismatch rejected; final cache written via a temporary file |
| Cached training run | `train_model` | Existing run rejected; a later runtime failure can still leave an incomplete run directory |
| Figures, plot CSV/JSON | Plotting modules | Same output filenames replaced; choose `--output` to preserve variations |
| High-resolution display JPEGs | `plot_predictions` | Separate URL-hashed cache; cached dimensions checked; failed fetch falls back to the original image |

`datasets/`, `artifacts/`, `.venv/` and Python bytecode are ignored by Git. Keep local backups/provenance as appropriate; pushing source does not upload trained results or datasets.

## SQLite schema

`cutouts.object_id` is a TEXT primary key. Each row includes coordinates, label, optional split, request provider/dimensions/FOV/URL, image BLOB, status (`pending`, `downloaded`, `error`), attempts/error and download timestamp. `size_arcsec` in current manifests stores the selected `petroR90_r` radius used to derive the field of view. This name does not imply an independent total galaxy diameter. Successful ZIP imports update the BLOB/status but do not populate a complete external-source download history.

SQLite may have `-wal` and `-shm` sidecars while in WAL mode. Do not copy/delete them independently while a writer is active. Extraction fingerprints the database and its WAL when present; complete downloads before extracting. Label/split metadata in the CSV and database must remain consistent.

## Tests

| File | Main coverage |
|---|---|
| [test_ml_training.py](../tests/test_ml_training.py) | Grayscale shape, full training, training-only PCA, serialization, exact IDs and label consistency |
| [test_stages.py](../tests/test_stages.py) | Duplicate policy, SQLite closure, cache reuse/invalidation, retraining, ZIP import and checksum download |
| [test_sdss.py](../tests/test_sdss.py) | FOV conversion, legacy request rejection, photometric formulas, SQL identity join and sample-preserving rebuild |
| [test_gallery.py](../tests/test_gallery.py) | Only selected objects requested, display cache reuse, read-only dataset and fallback behavior |
| [test_correlations.py](../tests/test_correlations.py) | Training-only correlations and constant-column handling |
| [test_roc.py](../tests/test_roc.py) | Continuous-score AUC, undefined cases and older-run score reconstruction without double preprocessing |

Tests are synthetic/offline; they do not guarantee live service availability or astronomical correctness. Run from the root with `python -m unittest discover -s tests -v`. There is no CI workflow configured in this repository.

## Keeping documentation current

When changing a stage, update its option/default table, input/output contract, downstream artifact expectations and workflow examples together. Keep formulas synchronized with `photometry.FEATURE_DEFINITIONS`; update the migration guide if stored artifacts become incompatible. Verify documented CLI options using each module's `--help` and check local Markdown links. Distinguish synthetic tests, live service checks and scientific performance results; never describe one as another.
