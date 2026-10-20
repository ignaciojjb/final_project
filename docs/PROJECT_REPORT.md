# Galaxy classification from SDSS photometry and images

## Objective

Classify galaxies as **elliptical (0)** or **spiral (1)** by combining catalog measurements and image features in a reproducible supervised-learning pipeline.

**Workflow:** collect catalog → clean and split → obtain images → extract and merge features → train → evaluate.

## 1. Data collection and dataset construction

The public **SDSS SkyServer DR19 API** provides Galaxy Zoo `ZooSpec` classifications and `PhotoObjDR7` measurements. DR19 hosts the service; the photometry is **legacy DR7**. Matching both DR7 and DR8 identifiers avoids ambiguous catalog associations.

Selection requires a primary detection, at least 20 Galaxy Zoo votes, an unambiguous class, r-band Petrosian magnitude between 10 and 25, magnitude error from 0 to below 0.1, and no deblend children. This last cut does not guarantee isolation. Exact duplicate rows are removed; conflicting IDs require resolution.

A **64×64 RGB stamp** is downloaded for each selected object from the SDSS9-color survey through CDS hips2fits, centered on its coordinates. Coverage follows its Petrosian radius within 16–120 arcseconds, converted to degrees for the service. SQLite stores images, IDs, labels, splits and download status, allowing downloads to resume.

Each dataset row combines one downloaded image with its measurements using exact `specobjid`. Equal class counts are selected—for example, 5,000 of each type for a 10,000-object experiment. Actual usable counts depend on successful matching and downloads.

## 2. Feature-selection criteria

Selection follows astronomical measurement consistency and useful brightness, color and structure information; it is not an automated feature-search procedure.

| Feature group | Representation and reason |
|---|---|
| Images | Grayscale pixels scaled to [0,1]; 4,096 values reduced to 64 PCA components |
| Colors | Extinction-corrected model-magnitude u−g, g−r, r−i and i−z; a consistent r-selected profile supports comparable measurements across bands |
| Brightness | Extinction-corrected Petrosian r magnitude for the standard catalog; cModel preferred when available |
| Structure | Half-light radius R50 and concentration R90/R50; optional fracDeV when supplied |

The standard catalog yields **7 photometric/structural features + 64 image components = 71 model inputs**. IDs, coordinates, morphology flags and vote probabilities are excluded from predictors. Extinction corrects measurements; magnitude errors control quality rather than being default predictors.

Missing catalog values are median-imputed. Both feature branches are standardized. PCA, imputation and scaling are fitted **only on training rows**, preventing held-out information from influencing preprocessing.

## 3. Dataset partitions

Balanced objects are sampled from the downloaded pool with seed 42. Stratified splitting preserves class proportions; existing complete assignments are retained.

| Partition | Share | 10,000-object example | Purpose |
|---|---:|---:|---|
| Training | 70% | 7,000 | Fit preprocessing and classifier |
| Validation | 15% | 1,500 | Choose model settings |
| Test | 15% | 1,500 | Assess the final selected model |

Counts assume complete matching/downloads. Although the code exports both held-out evaluations, test results must not guide repeated parameter choices.

## 4. Machine-learning algorithms

**Principal component analysis (PCA)** compresses image pixels into linear combinations capturing dominant variance, without using class labels.

The merged image and catalog features feed a **support-vector classifier (SVC) with an RBF kernel**, allowing a nonlinear decision boundary. Defaults are `C=1`, `gamma='scale'` and balanced class weights. C controls the error penalty; gamma controls kernel locality. Compare alternatives using validation data. There is no automatic hyperparameter search or probability calibration.

Cached features allow new classifier fits without repeating downloads, image decoding or PCA.

## 5. Performance evaluation

| Metric | Interpretation |
|---|---|
| Accuracy | Fraction correctly classified |
| Balanced accuracy | Mean recall across galaxy types |
| Per-class precision | Reliability of predictions of that type |
| Per-class recall | Fraction of that reference type recovered |
| Per-class F1 | Balance of precision and recall |
| Confusion matrix | Counts of correct classifications and class confusions |
| ROC and AUC | Discrimination across thresholds using continuous SVC scores; spiral is positive |

Validation/test metrics are separate. Correlation plots and prediction galleries support diagnosis; they do not measure feature importance or independent truth. Higher-resolution images are downloaded only for selected gallery examples and never enter training.

## Interpretation limits

The query retrieves the first qualifying IDs per class, so the pool is **not a survey-wide random sample**. Catalog labels can be imperfect, images can contain neighbors, and repeated physical galaxies with different IDs are not automatically grouped across splits. Take performance numbers from the selected run's `metrics.json`; synthetic tests and small smoke checks establish execution, not scientific accuracy.

## References

[Annotated code](../scripts/) · [Workflow](WORKFLOW.md) · [Data/model details](DATA_AND_MODEL.md) · [Results](RESULTS.md) · [SDSS magnitude guidance](https://www.sdss3.org/dr9/algorithms/magnitudes.php) · [CDS hips2fits](https://alasky.cds.unistra.fr/hips-image-services/hips2fits) · [Download audit](../DOWNLOAD_GUIDE.md).
