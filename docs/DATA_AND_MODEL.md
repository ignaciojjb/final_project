# Data, feature definitions and model

[Documentation index](../README.md#documentation) · [Workflow](WORKFLOW.md) · [Source audit](../DOWNLOAD_GUIDE.md)

## Sources and object identity

The public catalog endpoint is `https://skyserver.sdss.org/dr19/SkyServerWS/SearchTools/SqlSearch`. The downloader joins `ZooSpec z` with `PhotoObjDR7 p` on both:

```sql
p.dr7objid = z.dr7objid AND p.dr8objid = z.objid
```

It selects explicit columns and converts IDs to strings in SQL. This avoids ambiguous `specobjid`/`SpecObjID` fields and selecting every neighboring cross-match. **DR19 is the hosting release; these are legacy DR7 photometric measurements**, linked through DR8 identifiers. No photometry from another release is silently joined to add features. `distance` is the DR7/DR8 positional separation in arcminutes, not a predictor.

Objects are retrieved in increasing `specobjid`, separately by class. Pagination uses the last exact ID as a cursor. The provenance JSON records executed queries, source and units, but is not a complete environment manifest: keep the Git commit, Python/package versions and your run artifacts for reproducibility.

## Catalog input contract

CSV column names are case-sensitive. Read long IDs as text, not float values; precision lost in an earlier spreadsheet/export cannot be restored by converting back to strings.

| Fields | Requirement and meaning |
|---|---|
| `specobjid` | Non-null, unique spectrum/object key used to join `cutouts.object_id` |
| `spiral`, `elliptical` | Binary reference flags with exactly one equal to 1 for eligible objects |
| `uncertain` | If present, must be 0 for eligible objects |
| `ra`, `dec` | Photometric J2000/ICRS coordinates in degrees for manifest creation; RA [0,360), Dec [-90,90] |
| `modeDR7`, `nChild`, `petroMagErr_r`, `petroMag_r`, `nvote` | Needed by normal quality cleaning; only skipped with `--already-clean` |
| `modelMag_u/g/r/i/z`, `extinction_u/g/r/i/z` | Matching magnitude/extinction pairs used to derive available colors |
| `cModelMag_r` or `petroMag_r`, plus `extinction_r` | Extinction-corrected brightness; cModel preferred when supplied |
| `petroR50_r`, `petroR90_r` | Optional structural radii in arcseconds; R90 also controls request FOV |
| `fracDeV_r` | Optional r-band profile fraction in [0,1] |
| `split` | Optional complete assignments `train`, `validation`, `test`; otherwise generated |
| `objid`, `dr7objid`, `dr8objid`, `distance` | Provenance/cross-match information; excluded from predictors |

The photometry builder uses available supported columns. At least one usable derived photometric feature is required; it does not assume missing extinction is zero. Not every optional field is returned by `download_catalog`. In particular, the tested `PhotoObjDR7` download lacks cModel and fracDeV and therefore uses Petrosian brightness.

Preparation removes rows identical across every column. Repeated IDs with other differences fail. For older external catalogs, `--resolve-dr8-matches` resolves only groups differing in `dr8objid`/`distance`, with finite nonnegative distances, non-null match IDs and a unique minimum. Other differences and ties still fail. This opt-in nearest-match rule is not a substitute for the explicit identity join in the current downloader.

## Quality cuts and splits

Normal preparation replaces numeric `-9999` sentinels with missing values, then applies:

| Cut | Current condition |
|---|---|
| Primary DR7 detection | `modeDR7 == 1` |
| No deblend children | `nChild == 0` |
| r-band magnitude uncertainty | `0 <= petroMagErr_r < 0.1` |
| r-band apparent magnitude | `10 <= petroMag_r <= 25` |
| Galaxy Zoo votes | `nvote >= 20` |
| Unambiguous class | Binary spiral/elliptical flags summing to 1; `uncertain == 0` if present |

`nChild == 0` excludes deblend parents with children; it does **not** mean every image is unblended or contains exactly one galaxy. These are baseline cuts, not a comprehensive SDSS photometric flag analysis. The public downloader applies the same baseline restrictions in SQL before collecting its pool.

Preparation draws the requested count per class with seed 42 by default. When all splits are absent, it creates stratified 70/15/15 partitions. Existing complete assignments are retained. Mixed missing/present assignments, unknown split names or a split without both classes fail. Download failures can reduce a split after preparation: the loader reports matched downloaded objects, and extraction again checks class availability. The loader can omit unmatched/ineligible catalog-image pairs; inspect its reported counts rather than assuming all manifest rows were used.

## Image requests and transformations

Default image source: CDS hips2fits survey `CDS/P/SDSS9/color`. These RGB JPEG visualizations are not calibrated single-band flux measurements, and are not guaranteed to be the same-release frames as the legacy DR7 photometry.

| Quantity | Convention |
|---|---|
| Training request dimensions | 64×64 pixels in the recommended preparation/rebuild paths |
| Field of view | `clip(3 * petroR90_r, 16, 120)` arcseconds; 16 if no valid R90 |
| hips2fits request `fov` | FOV arcseconds divided by 3600, yielding degrees |
| Alternative SkyServer `scale` | FOV arcseconds / largest pixel dimension, in arcsec/pixel |
| Model image input | PIL grayscale, resized to 64×64 using Lanczos, flattened and divided by 255 |
| Flattened columns | `pixel_0000` through `pixel_4095` |
| Gallery high-resolution input | Separate request for selected objects only; same center and FOV |

The generic `size` column is not interpreted as arcseconds. A larger pixel count does not imply a larger FOV or improve the telescope's native resolution. The 120-arcsecond upper bound can clip very large galaxies; inspect images. The downloader verifies format, decoding and requested dimensions; it does not identify objects or check isolation. Known legacy hips2fits unit errors are rejected when request provenance is available. External minimal SQLite schemas without URL/FOV metadata cannot receive that provenance check.

## Engineered catalog features

Implementation: [photometry.py](../scripts/photometry.py). Let `m0_b = modelMag_b - extinction_b`.

| Feature | Definition | Interpretation |
|---|---|---|
| `model_u_g0` | `m0_u - m0_g` | Extinction-corrected u−g color |
| `model_g_r0` | `m0_g - m0_r` | Extinction-corrected g−r color |
| `model_r_i0` | `m0_r - m0_i` | Extinction-corrected r−i color |
| `model_i_z0` | `m0_i - m0_z` | Extinction-corrected i−z color |
| `cmodel_r0` | `cModelMag_r - extinction_r` | Total-light estimate when cModel is supplied |
| `petro_r0` | `petroMag_r - extinction_r` | Fallback brightness when cModel column is absent |
| `petroR50_r` | Positive r-band half-light radius | Apparent angular size |
| `concentration_r` | `petroR90_r / petroR50_r` | Concentration; requires positive radii and R90 ≥ R50 |
| `fracDeV_r` | Valid profile fraction | Optional de Vaucouleurs contribution |

Colors use model magnitudes because SDSS fixes the r-selected profile across bands, making them suitable for consistent galaxy colors. Petrosian brightness is a useful fallback but misses a profile-dependent fraction of total light; it is not identical to cModel or true total luminosity. If a cModel column exists but a row is missing, the current builder leaves that feature missing for imputation; it does not switch that row to Petrosian magnitudes.

Magnitudes remain in SDSS asinh form; no naive logarithmic flux conversion is applied. Numeric `-9999`, infinities, invalid radii and invalid profile fractions become missing. The catalog branch uses training medians for imputation and standardization. An entirely missing training column is retained and filled with zero by the configured imputer, so its correlations may be undefined.

Errors, extinction values themselves, IDs, coordinates, vote counts/probabilities, class flags and split metadata are not standalone classifier inputs. Actual selected names and formulas are in each current run's `metrics.json`, under `catalog_features` and `feature_definitions`.

Reference: [SDSS magnitude algorithms](https://www.sdss3.org/dr9/algorithms/magnitudes.php). That page was blocked by a proxy during the download audit; the audit did successfully query the live DR19 column metadata and CDS FOV documentation. These definitions are the implemented rationale, not a claim of empirically optimal features.

## Preprocessing, training and inference

1. Join eligible catalog rows to downloaded images by exact ID and check labels.
2. Use the existing splits, or create them when entirely absent.
3. Fit randomized PCA to the 4,096-pixel **training** matrix, then standardize its components.
4. Fit median imputation and standard scaling to the **training** catalog features.
5. Concatenate both transformed branches and fit the RBF SVC on training labels.
6. Predict/evaluate validation and test rows using the fitted transforms, without refitting.

Default PCA components = 64 and seed = 42. This requires at least 64 training objects and at most 4,096 components. Default SVC settings are `C=1`, `gamma='scale'`, `class_weight='balanced'`, with no probability calibration. `--gamma` also accepts `auto` or a numeric value. Increasing C usually penalizes training errors more strongly; increasing numeric gamma usually makes the kernel more local. Select settings using validation, not repeated test inspection.

The feature cache holds transformed inputs, metadata, fitted preprocessing and input/code signatures. A trained `model.joblib` contains `features` (fitted ColumnTransformer) followed by `svc`. It expects the **pixel plus engineered catalog DataFrame** produced by `load_dataset`, not a raw JPEG or unprocessed CSV row. The repository has no general unlabeled-inference CLI. Applying the full pipeline to the already transformed `merged_features.csv` would preprocess it twice; ROC reconstruction for older runs correctly applies only the fitted SVC.

## Terms used throughout the project

| Term | Meaning here |
|---|---|
| Catalog | Table of object identifiers, photometry, coordinates and reference labels |
| Stamp / cutout | Small sky image centered on the catalog coordinates |
| Feature | Numeric model input, such as a color or an image PCA component |
| PCA | Principal component analysis: unsupervised linear combinations that compress the pixel features |
| SVC / RBF | Support-vector classifier using a radial-basis-function kernel for nonlinear classification |
| Cache | Saved intermediate matrix/transforms used to avoid repeating image processing and PCA |
| Run | One saved classifier fit and its evaluations, linked to one feature cache |
| Reference label | Galaxy Zoo catalog classification used as the target; not independent verification |

## Scientific and computational limits

Catalog labels can be wrong, ambiguous objects are excluded, and the public downloader's first-ID selection is not a survey-wide random sample. Apparent size/brightness can encode redshift, seeing and selection effects. Features are extinction-corrected but not K-corrected or converted to absolute luminosity. Cross-release catalog/image provenance must remain explicit.

Unique spectrum IDs do not guarantee distinct physical galaxies: repeat observations with different IDs must be grouped into the same split for a rigorous study. The current code does not implement spatial grouping, source detection, automatic isolation checks, cross-validation, calibration, uncertainty intervals or a controlled learning-curve experiment. Balanced starter-set accuracy does not establish deployment performance at real class frequencies.

The full pixel table lives in memory. Nonlinear SVC scales poorly for very large samples. Around 2,000 objects is the intended starter workflow; larger-survey work may need streaming/incremental image processing and a scalable classifier. Synthetic tests and the recorded 12-object live smoke check validate execution, not scientific accuracy.
