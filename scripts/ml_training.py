"""Train a leakage-safe photometry + grayscale/PCA galaxy SVC."""
from __future__ import annotations

from contextlib import closing

import argparse
from io import BytesIO
import json
from pathlib import Path
import sqlite3

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from scripts.photometry import build_photometry, FEATURE_DEFINITIONS
from scripts.cutouts import validate_stored_requests

# Bookkeeping columns never enter this fixed 4,096-column image feature branch.
PIXELS = [f'pixel_{i:04d}' for i in range(4096)]


def stamp_to_features(blob: bytes) -> np.ndarray:
    """Convert encoded image bytes to 4,096 grayscale float32 features.

    Decode JPEG/PNG bytes, use PIL luminance conversion, resize to 64x64
    with Lanczos, flatten in row-major order, and scale [0,255] to [0,1].
    This visualization-based pixel representation is not calibrated flux.
    """
    with Image.open(BytesIO(blob)) as image:
        gray = image.convert('L').resize((64, 64), Image.Resampling.LANCZOS)
        return np.asarray(gray, dtype=np.float32).reshape(-1) / 255.0


def load_dataset(catalog_path, database_path):
    """Join decoded stamps to engineered photometry by exact text IDs.

    catalog_path is a labeled CSV; database_path is a downloaded-image SQLite
    file. Return (merged_dataframe, catalog_feature_names). The frame includes
    IDs/labels/splits for bookkeeping plus pixel_* and engineered features.
    Only downloaded, eligible matching objects survive the inner join.
    Check request provenance and label agreement; fit no PCA/scaler here.
    The full pixel table is in memory, so use a moderate starter sample.
    """
    # 1. Read exact identities and reference labels before deriving any model inputs.
    catalog = pd.read_csv(catalog_path, dtype={c: 'string' for c in ['specobjid','objid','dr7objid','dr8objid']})
    required = {'specobjid', 'spiral', 'elliptical'}
    if not required.issubset(catalog):
        raise ValueError(f'Missing catalog columns: {sorted(required - set(catalog))}')
    if catalog.specobjid.isna().any() or catalog.specobjid.duplicated().any():
        raise ValueError('Catalog specobjid must be non-null and unique.')
    valid = catalog.spiral.isin([0, 1]) & catalog.elliptical.isin([0, 1])
    valid &= catalog.spiral.add(catalog.elliptical).eq(1)
    if 'uncertain' in catalog:
        valid &= catalog.uncertain.eq(0)
    catalog = catalog.loc[valid].copy()
    # 2. Feature formulas use measurements only; morphology flags are never predictors.
    engineered, features = build_photometry(catalog)
    for column in features:
        catalog[column] = engineered[column]
    database = Path(database_path).resolve()
    if not database.is_file():
        raise FileNotFoundError(database)
    vectors, records = [], []
    # 3. Read-only access and explicit closure avoid mutation and Windows file locks.
    with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as connection:
        validate_stored_requests(connection)
        for object_id, blob, label, split in connection.execute(
            "SELECT object_id, image, label, split FROM cutouts WHERE status='downloaded' ORDER BY object_id"
        ):
            records.append((str(object_id), label, split))
            try:
                vectors.append(stamp_to_features(blob))
            except Exception as exc:
                raise ValueError(f'Cannot decode downloaded image {object_id}') from exc
    if not records:
        raise ValueError('No downloaded images found.')
    images = pd.DataFrame(records, columns=['specobjid', 'label', 'split'])
    images = pd.concat([images, pd.DataFrame(np.stack(vectors), columns=PIXELS)], axis=1)
    # 4. Match by ID, never row order; incompatible labels expose stale/wrong artifacts.
    merged = images.merge(catalog[['specobjid', 'spiral', *features]], on='specobjid', validate='one_to_one')
    if merged.empty:
        raise ValueError('No image IDs match eligible catalog specobjid values. Check exact ID formatting.')
    if not merged.label.eq(merged.spiral).all():
        raise ValueError('SQLite labels disagree with catalog morphology flags.')
    print(f'Using {len(merged)} matched objects of {len(images)} downloaded images; {len(features)} catalog features.')
    return merged.drop(columns='spiral'), features


def assign_splits(data, seed=42):
    """Validate or create splits, returning a copy without altering the input.

    data needs label and split columns. When every split is missing, create
    reproducible stratified 70/15/15 train/validation/test partitions. Keep
    complete existing splits; partial/unknown assignments or a partition
    missing either class fail. This does not group repeat physical galaxies.
    """
    data = data.copy()
    if data['split'].isna().all():
        train, remainder = train_test_split(data.index, test_size=.30, stratify=data.label, random_state=seed)
        # Half of the 30% remainder is validation and half test: 15% of the full sample each.
        validation, test = train_test_split(remainder, test_size=.50, stratify=data.loc[remainder, 'label'], random_state=seed)
        data.loc[train, 'split'] = 'train'
        data.loc[validation, 'split'] = 'validation'
        data.loc[test, 'split'] = 'test'
    if data['split'].isna().any() or not data['split'].isin(['train', 'validation', 'test']).all():
        raise ValueError('Splits must be either entirely absent or all train/validation/test.')
    for split in ['train', 'validation', 'test']:
        if set(data.loc[data['split'].eq(split), 'label']) != {0, 1}:
            raise ValueError(f'{split} must contain both classes; check sample sizes and download failures.')
    return data


def make_pipeline(catalog_features, n_components=64, seed=42):
    """Construct an UNFITTED image-plus-catalog preprocessing/SVC pipeline.

    catalog_features lists engineered column names, excluding bookkeeping.
    n_components and seed configure randomized image PCA. The two branches
    output standardized PCA scores and imputed/standardized catalog values.
    A later fit must use training rows only; returning a pipeline does not
    fit it. The RBF SVC has balanced class weights and no probability fit.
    """
    # Branch A: center/compress image pixels with PCA, then put component scales on equal footing.
    image_branch = Pipeline([
        ('pca', PCA(n_components=n_components, svd_solver='randomized', random_state=seed)),
        ('scale', StandardScaler()),
    ])
    # Branch B: fill missing measurements with training medians, then standardize units.
    catalog_branch = Pipeline([
        ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('scale', StandardScaler()),
    ])
    # ColumnTransformer concatenates both branches; SVC sees their combined numeric matrix.
    return Pipeline([
        ('features', ColumnTransformer([
            ('stamp', image_branch, PIXELS), ('catalog', catalog_branch, catalog_features),
        ], verbose_feature_names_out=True)),
        ('svc', SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced')),
    ])


def train(catalog, database, output, n_components=64, seed=42):
    """Run the alternative all-in-one workflow; return (fitted_model, metrics).

    catalog/database are the same labeled inputs used by load_dataset.
    Unlike train_cached, this refits preprocessing on each invocation.
    output receives model, metrics, held-out predictions and feature table;
    this older entry point can overwrite files in an existing directory.
    Validation/test rows are evaluated separately and never used to fit.
    """
    data, columns = load_dataset(catalog, database)
    data = assign_splits(data, seed)
    training = data.loc[data['split'].eq('train')]
    if not 1 <= n_components <= min(len(training), 4096):
        raise ValueError(f'PCA components must be between 1 and {min(len(training), 4096)}.')
    model = make_pipeline(columns, n_components, seed)
    # One training-only fit learns PCA, imputations, scales and classifier parameters.
    model.fit(training, training.label)
    metrics = {'classes': {'0': 'elliptical', '1': 'spiral'}, 'seed': seed,
               'n_components': n_components, 'catalog_features': columns,
               'feature_definitions': {c: FEATURE_DEFINITIONS[c] for c in columns},
               'split_counts': data.groupby(['split', 'label']).size().unstack(fill_value=0).to_dict(orient='index')}
    predictions = []
    for split in ['validation', 'test']:
        subset = data.loc[data['split'].eq(split)]
        predicted = model.predict(subset)
        # Ranking scores support ROC/AUC; they are not calibrated class probabilities.
        score = model.decision_function(subset)
        metrics[split] = {
            'roc_auc': float(roc_auc_score(subset.label, score)),
            'accuracy': accuracy_score(subset.label, predicted),
            'balanced_accuracy': balanced_accuracy_score(subset.label, predicted),
            'confusion_matrix': confusion_matrix(subset.label, predicted, labels=[0, 1]).tolist(),
            'classification_report': classification_report(subset.label, predicted, labels=[0, 1], target_names=['elliptical', 'spiral'], output_dict=True, zero_division=0),
        }
        predictions.append(pd.DataFrame({'specobjid': subset.specobjid, 'split': split, 'label': subset.label, 'prediction': predicted, 'decision_score': score}))
    pca = model.named_steps['features'].named_transformers_['stamp'].named_steps['pca']
    metrics['pca_explained_variance_ratio_sum'] = float(pca.explained_variance_ratio_.sum())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / 'model.joblib')
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    pd.concat(predictions).to_csv(output / 'predictions.csv', index=False)
    # Features are transformed with the training-fitted preprocessing only.
    values = model.named_steps['features'].transform(data)
    table = pd.DataFrame(values, columns=model.named_steps['features'].get_feature_names_out())
    table.insert(0, 'specobjid', data.specobjid.to_numpy())
    table['label'] = data.label.to_numpy()
    table['split'] = data['split'].to_numpy()
    table.to_csv(output / 'merged_features.csv', index=False)
    print(json.dumps(metrics, indent=2))
    return model, metrics


def main():
    """Parse paths/PCA settings for the alternative all-in-one training CLI.

    For repeated SVC experiments, prefer extract_features then train_model
    so downloaded pixels and fitted preprocessing do not need recomputing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, default=Path('datasets/processed_datasets/galaxies_clean.csv'))
    parser.add_argument('--database', type=Path, default=Path('datasets/images/cutouts_cnn.sqlite'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/svc'))
    parser.add_argument('--components', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train(args.catalog, args.database, args.output, args.components, args.seed)


if __name__ == '__main__':
    main()
