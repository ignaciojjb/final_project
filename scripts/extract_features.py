"""Cache training-fitted PCA and merged features for cheap repeated SVC fits."""
import argparse
from pathlib import Path
import joblib
from scripts.download_data import sha256
from scripts.photometry import FEATURE_DEFINITIONS
from scripts.ml_training import assign_splits, load_dataset, make_pipeline


def extract(catalog, database, output, components=64, seed=42):
    output = Path(output)
    signature = {'photometry_version': 2, 'catalog': sha256(catalog), 'database': sha256(database),
                 'wal': sha256(str(database) + '-wal') if Path(str(database) + '-wal').exists() else None,
                 'components': components, 'seed': seed,
                 'implementation': [sha256(Path(__file__)), sha256(Path(__file__).with_name('ml_training.py')), sha256(Path(__file__).with_name('photometry.py')), sha256(Path(__file__).with_name('cutouts.py'))]}
    if output.exists():
        cached = joblib.load(output)
        if cached['signature'] != signature:
            raise ValueError('Feature cache is stale. Choose a new output path or remove the old cache explicitly.')
        print(f'Reusing {output}; no image decoding or PCA fitting.')
        return cached
    data, columns = load_dataset(catalog, database)
    data = assign_splits(data, seed)
    training = data['split'].eq('train')
    if not 1 <= components <= min(training.sum(), 4096):
        raise ValueError('Invalid PCA component count for the available training set.')
    preprocessing = make_pipeline(columns, components, seed).named_steps['features']
    preprocessing.fit(data.loc[training])
    cache = {'signature': signature, 'preprocessing': preprocessing,
             'X': preprocessing.transform(data), 'metadata': data[['specobjid', 'label', 'split']],
             'catalog_features': columns, 'feature_definitions': {c: FEATURE_DEFINITIONS[c] for c in columns}}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp')
    joblib.dump(cache, temporary)
    temporary.replace(output)
    print(f'Saved {output}: {cache["X"].shape}')
    return cache


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog', default='datasets/processed_datasets/galaxies_clean.csv')
    p.add_argument('--database', default='datasets/images/cutouts.sqlite')
    p.add_argument('--output', default='artifacts/features.joblib')
    p.add_argument('--components', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    extract(a.catalog, a.database, a.output, a.components, a.seed)
