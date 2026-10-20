# Troubleshooting

[Documentation index](../README.md#documentation) · [Complete workflow](WORKFLOW.md)

Fix the first failing stage before running later ones. When asking for help, include the full command, working directory, final error and stage output. Do not include credentials. On Windows, use PowerShell from the repository root.

## Find your sample and run paths

```powershell
Get-Location
Get-ChildItem datasets/processed_datasets
Get-ChildItem datasets/images
Get-ChildItem artifacts
Get-ChildItem artifacts/runs
```

A sample's prepared CSV, image database, feature cache and model run need matching object IDs and splits, even though filenames can differ. The documentation's `sdss_sample1000_v2` and an older `sample1000_fixed` are separate naming examples, not interchangeable datasets.

## Common errors

| Symptom | Cause and next action |
|---|---|
| `Permission denied (publickey)` cloning | SSH key authentication failed. Clone with `git clone https://github.com/ignaciojjb/final_project.git`, or configure GitHub SSH keys. Existing checkout: `git remote set-url origin https://github.com/ignaciojjb/final_project.git` |
| Python executable/module not found | Check repository directory and `.venv`; repeat the setup guide's `uv venv --python 3.12` only if needed, then `uv pip install -r requirements-ml.txt` |
| Prepared catalog or database already exists | Preparation protects prior work. Reuse that sample by continuing downloads, or choose new CSV/database paths. Do not blindly delete a possibly valid database |
| Missing or conflicting `specobjid` | Inspect the raw records. Exact duplicate rows are removed; null IDs need source correction. Use the current explicit-join downloader, or the narrowly defined `--resolve-dr8-matches` only for compatible older cross-match records |
| `Cannot take a larger sample than population` | Fewer eligible objects exist in a class than `--per-class`. Download more candidates or reduce the requested count; use new sample paths |
| Missing `modeDR7`, `nChild`, photometry or coordinate columns | External CSV schema does not match the contract. Check headers/units. `--already-clean` is only for data you have actually quality-cleaned; it does not bypass other requirements |
| No downloaded images found | Manifest exists but has no downloaded image BLOBs. Run `download_images` on the **same database** and inspect status/errors before extraction |
| Feature cache not found | Feature extraction has not succeeded, or training points to a different filename. Create/check the intended cache before training |
| Feature cache is stale | Inputs, SQLite/WAL state, implementation, seed or PCA count changed. Finish all writes, extract into a new cache and train into a new run |
| Legacy feature cache / invalid legacy hips2fits FOV | Follow the migration guide; old wide-field images cannot be repaired by resizing or by redrawing plots |
| PCA component count invalid | At least that many training objects are required. Finish missing downloads or reduce `--components` in a new cache |
| A split must contain both classes | Too few eligible/matched/downloaded objects or invalid split assignments. Inspect class counts; do not force ROC/AUC or silently reshuffle test |
| Run directory exists | Use a new `--output` run name. An earlier failed run may have left partial files; inspect them before moving/removing anything |
| No saved predictions match `errors_only=True` | There are no mistakes in the chosen split, or no eligible saved rows. Drop `--errors-only` or check the split |
| Database labels/splits disagree with predictions | Wrong database/run pair or modified metadata. Restore matching artifacts; do not relabel the gallery to hide the mismatch |
| Windows `WinError 32` deleting SQLite in tests | Update to the current code, which closes connections explicitly. Close external SQLite viewers/writers and rerun tests in a fresh Python process if still locked |

## Check image-download status and errors

First confirm the database is the one you prepared:

```powershell
Test-Path datasets/images/sdss_sample1000_v2.sqlite
```

Only if it exists, inspect status:

```powershell
.\.venv\Scripts\python.exe -m scripts.cutouts `
  --database datasets/images/sdss_sample1000_v2.sqlite --status
```

Show three stored errors:

```powershell
.\.venv\Scripts\python.exe -c "import sqlite3; c=sqlite3.connect('datasets/images/sdss_sample1000_v2.sqlite'); print(*c.execute('SELECT object_id,error FROM cutouts WHERE status=? LIMIT 3',('error',)).fetchall(),sep='\n'); c.close()"
```

Retry the same manifest:

```powershell
.\.venv\Scripts\python.exe -m scripts.download_images `
  --database datasets/images/sdss_sample1000_v2.sqlite --workers 4
```

Timeouts/rate limits can improve with fewer workers. `403`/proxy tunnel errors indicate access restrictions or service refusal: check allowed destinations and the actual response rather than requesting an unrelated token. The catalog API and image service are separate, so success in one does not prove the other is reachable. For catalog timeouts, reduce `download_catalog --page-size` and use a new output if a completed file already exists. Keep TLS and checksum validation enabled.

## Check whether feature extraction succeeded

```powershell
Test-Path artifacts/features_sdss_sample1000_v2.joblib
```

`False` means training cannot use that path. Run the workflow's extraction command and inspect its output; repeating `train_model` cannot create a missing cache. Tests deliberately use temporary paths and do not populate this artifact.

## Blurry or crowded images

First check whether images came from the old hips2fits unit bug. Use [migration](../DOWNLOAD_GUIDE.md#a-repair-an-existing-sample-without-resampling-it-powershell) for those images and rebuild features/models. For corrected data, `--display-resolution 256` can improve only the selected gallery renderings, without changing training. Intrinsic seeing/resolution, a faint source, neighboring objects or clipping can still limit usefulness. Review the fields; no automatic single-galaxy guarantee exists.

## Correlations or ROC look surprising

Near-zero PCA component correlations are expected on training rows. Gray cells mean undefined correlation for a constant feature. Correlation is not SVC feature importance. AUC uses scores, not hard predictions, and is undefined without both classes. A perfect score on a tiny split or synthetic tests is not evidence of broad accuracy; check class counts, data provenance and possible repeated objects across splits. Do not tune parameters to improve repeatedly inspected test scores.

## Recover without losing prior work

Preserve raw input and existing runs. Use a new path when intentionally changing the sample, images, feature definition or model. Catalog/preparation/training failures are not a guarantee that no files were created; inspect partial outputs before retrying. A successful-looking directory or open SQLite file alone is not evidence that downloads or training completed. Check final command status and expected artifacts.
