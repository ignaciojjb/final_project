# SDSS download and feature audit

## Confirmed correction: hips2fits field-of-view units

The earlier code sent `fov_arcsec` directly to the hips2fits `fov` parameter. CDS requires **decimal degrees**, so an intended 16-arcsecond stamp covered 16 degrees. The corrected request divides by 3600:

```text
16 arcsec / 3600 = 0.0044444444 degrees
```

Reference: https://alasky.cds.unistra.fr/hips-image-services/hips2fits (API parameter `fov`).

Increasing image dimensions alone did not fix the old angular coverage. **Old hips2fits images, their PCA caches and derived model results must be rebuilt.** Old images are preserved; the updated downloader/loader detects legacy requests and refuses to silently reuse them. Previously downloaded SDSS-provider images use `scale` in arcseconds/pixel and are not affected by this specific unit bug. Imported ZIP stamps require their own provenance check.

New training requests remain 64×64 pixels. Only the selected gallery objects can request 256×256 display cutouts. FOV is approximately 3×`petroR90_r` (a radius in arcseconds), bounded to 16–120 arcseconds; with no valid radius it uses 16 arcseconds. A generic `size` column is not interpreted as arcseconds. Large galaxies can exceed these bounds: inspect a gallery for clipping. No FOV, `nChild` cut or resolution setting guarantees exactly one galaxy; visual review is still necessary.

## Photometry and identity: what is actually downloaded

- Hosting/API release: **DR19**. Tables: `ZooSpec` and `PhotoObjDR7`.
- Photometry: **legacy DR7**, not newly measured DR19 photometry. The spatial cross-match includes DR8 IDs. The live table does include legacy magnitudes/radii; it is not only an ID mapping.
- Join: `p.dr7objid=z.dr7objid AND p.dr8objid=z.objid`. Matching the existing DR8 object as well avoids taking every neighbor from the DR7–DR8 cross-match. No `SELECT *`; no duplicate `specobjid`/`SpecObjID` or ambiguous RA/Dec aliases.
- Coordinates: `p.ra`, `p.dec`, J2000/ICRS degrees from the photometry record. All object IDs are returned as text, not floats.
- Cross-match `distance` is in **arcminutes**, confirmed in DR19 `DBColumns`; do not interpret it as pixels or degrees. `petroR50_r`/`petroR90_r` are arcseconds. Generic `size` is defined as `sqrt(mRrCc_r/2)` and has no unit declared in that table.
- Labels: Galaxy Zoo `spiral`, `elliptical`; uncertain objects are excluded. Labels/vote probabilities are not model inputs.
- Images: CDS HiPS survey `CDS/P/SDSS9/color`; a color visualization of SDSS imaging, not calibrated single-band flux measurements or guaranteed release-identical DR7 frames. Grayscale JPEG pixels are a morphological input, not a replacement for catalog photometry. The `sdss` alternative uses SkyServer's ImgCutout service.

The code uses public, read-only SkyServer queries; no account, token, `MyDB` creation, table deletion or assumed external CSV URL is needed. The original uploaded SciServer script was not integrated because its destructive table handling and `SELECT *` join needed review.

## Selected galaxy features

Reference supplied by the user: https://www.sdss3.org/dr9/algorithms/magnitudes.php

- **Colors:** adjacent `u-g`, `g-r`, `r-i`, `i-z` from `modelMag`, after subtracting the respective Galactic `extinction` in each band. SDSS model colors use a consistent r-selected profile across bands. Do not combine independently selected per-band profile fits or mix PSF/Petrosian magnitudes to form default galaxy colors.
- **Brightness:** `cModelMag_r - extinction_r` when provided (cModel total-light estimate); otherwise `petroMag_r - extinction_r`. The legacy `PhotoObjDR7` schema tested here has Petrosian magnitudes, not cModel, so the actual download uses **Petrosian fallback**, explicitly named `petro_r0`. Petrosian flux misses a profile-dependent amount of total light, especially for extended de Vaucouleurs profiles; it is not identical to total flux.
- **Structure:** `petroR50_r` and `petroR90_r/petroR50_r` (concentration). Optional `fracDeV_r` only when supplied by a compatible catalog; it is not fabricated or mixed in from a different release.
- **Quality rather than default predictors:** magnitude errors, mode, vote counts, deblend fields. Extinction columns are used for correction, not standalone predictors. Exclude IDs, coordinates, morphology flags and vote probabilities from the classifier.
- Magnitudes retain SDSS **asinh** values; there is no naive logarithmic conversion of low/negative flux. Positive-radii and missing-value handling precede training-only median imputation/scaling. Missing extinction is not silently assumed zero. These are observed, extinction-corrected features, not K-corrected/rest-frame colors or absolute luminosities. Apparent size/brightness can carry selection/redshift effects.

`metrics.json` records actual selected features and formulas under `feature_definitions`. The code implementing these definitions is `scripts/photometry.py`. PCA remains training-only.

Access note: the linked SDSS3 page returned proxy 403 from the development environment; its contents were not fetched successfully. Live DR19 schema queries and the CDS FOV documentation were fetched successfully. The feature choices follow the distinctions above; they are not claimed to be an empirical optimal subset.

## A. Repair an existing sample without resampling it (PowerShell)

Use this for the existing sample1000. It preserves the prepared CSV, IDs and splits, but creates a **new** image database, cache and run. Do not reuse the earlier feature cache.

```powershell
git pull origin main
uv pip install -r requirements-ml.txt

.\.venv\Scripts\python.exe -m scripts.rebuild_images `
  --catalog datasets/processed_datasets/sample1000.csv `
  --database datasets/images/sample1000_fixed.sqlite

.\.venv\Scripts\python.exe -m scripts.download_images `
  --database datasets/images/sample1000_fixed.sqlite --workers 4

.\.venv\Scripts\python.exe -m scripts.extract_features `
  --catalog datasets/processed_datasets/sample1000.csv `
  --database datasets/images/sample1000_fixed.sqlite `
  --output artifacts/features_sample1000_fixed.joblib --components 64

.\.venv\Scripts\python.exe -m scripts.train_model `
  --features artifacts/features_sample1000_fixed.joblib `
  --output artifacts/runs/svc-sample1000-fixed-001

.\.venv\Scripts\python.exe -m scripts.plot_results `
  --run artifacts/runs/svc-sample1000-fixed-001

.\.venv\Scripts\python.exe -m scripts.plot_predictions `
  --run artifacts/runs/svc-sample1000-fixed-001 `
  --database datasets/images/sample1000_fixed.sqlite `
  --count 10 --display-resolution 256
```

For sample100, replace `sample1000` with `sample100` throughout. Each command must succeed before continuing. Network failures are recorded in SQLite; rerun the same downloader to retry. Old sample files are not deleted. Existing `_fixed` paths are also protected; choose another suffix if already used.

## B. Download a fresh catalog directly from SDSS (PowerShell)

No placeholder URL/checksum is required for this query-based path. It downloads 1200 quality-selected candidates per class so preparation can select 1000 each:

```powershell
.\.venv\Scripts\python.exe -m scripts.download_catalog `
  --per-class 1200 --page-size 500 `
  --output datasets/raw/sdss_catalog_v2.csv

.\.venv\Scripts\python.exe -m scripts.prepare_data `
  --catalog datasets/raw/sdss_catalog_v2.csv --per-class 1000 `
  --output datasets/processed_datasets/sdss_sample1000_v2.csv `
  --database datasets/images/sdss_sample1000_v2.sqlite

.\.venv\Scripts\python.exe -m scripts.download_images `
  --database datasets/images/sdss_sample1000_v2.sqlite --workers 4

.\.venv\Scripts\python.exe -m scripts.extract_features `
  --catalog datasets/processed_datasets/sdss_sample1000_v2.csv `
  --database datasets/images/sdss_sample1000_v2.sqlite `
  --output artifacts/features_sdss_sample1000_v2.joblib --components 64

.\.venv\Scripts\python.exe -m scripts.train_model `
  --features artifacts/features_sdss_sample1000_v2.joblib `
  --output artifacts/runs/svc-sdss-v2-001
```

The downloader stores executed SQL and source/unit information in the sibling `.provenance.json`. It uses deterministic keyset pagination on exact `specobjid`, bounded pages and explicit quality/label cuts. It takes the first eligible IDs per class, **not a uniform random sample of the whole survey**; subsequent preparation randomly samples only within that downloaded pool. Do not claim survey-representative accuracy. For full scientific analysis, use a justified selection function and check duplicate physical galaxies across IDs/splits. Public API timeouts/rate limits remain possible; use smaller pages or the official CasJobs interface for large exports. The output is written only after the requested sample completes; a failed request is not a finished catalog.

## Validation

Run `python -m unittest discover -s tests -v`. Tests cover angular unit conversion, rejection of legacy FOV requests, image dimensions, photometric formulas, invalid radii, explicit SQL join, exact IDs, sample-preserving rebuild, caches and training. Tests alone do not prove image identity, isolation or scientific classifier accuracy.

Live validation performed during this correction: fetched DR19 schema/metadata, downloaded 6 eligible objects per class through the new explicit-join catalog downloader, downloaded all 12 corrected 64×64 cutouts without errors, and ran extraction, training, saved-result plots and gallery generation. That tiny workflow check used **2** PCA components because it has only 8 training rows. The normal **64**-component path is covered by the synthetic suite and requires an appropriately sized real sample. A separate request successfully decoded a 256×256 display cutout with the corrected angular extent. These checks do not establish classifier accuracy or guarantee a single isolated galaxy in every stamp.
