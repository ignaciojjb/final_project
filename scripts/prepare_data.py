"""Clean a catalog, fix splits once and create a resumable image manifest."""
import argparse
from pathlib import Path
import pandas as pd
from scripts.data_processing import clean_sample
from scripts.ml_training import assign_splits
from scripts.cutouts import prepare_manifest


def prepare(catalog, output, database, per_class=1000, seed=42, clean=True):
    output = Path(output)
    if output.exists() or Path(database).exists():
        raise ValueError('Prepared catalog or database already exists. Reuse it, or choose new paths.')
    data = pd.read_csv(catalog, dtype={'specobjid': 'string'})
    if data.specobjid.isna().any() or data.specobjid.duplicated().any():
        raise ValueError('specobjid must be non-null and unique.')
    if clean:
        data, log = clean_sample(data.replace(-9999, float('nan')))
        print(log.to_string(index=False))
    valid = data.spiral.isin([0, 1]) & data.elliptical.isin([0, 1])
    valid &= data.spiral.add(data.elliptical).eq(1)
    if 'uncertain' in data:
        valid &= data.uncertain.eq(0)
    data = data.loc[valid].copy()
    data['label'] = data.spiral.astype(int)
    if per_class:
        data = pd.concat([data.loc[data.label.eq(label)].sample(n=per_class, random_state=seed) for label in (0, 1)], ignore_index=True)
    if 'split' not in data:
        data['split'] = None
    data = assign_splits(data, seed)
    # The supplied downloader expects size to be present.
    if 'size' not in data:
        data['size'] = float('nan')
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False)
    prepare_manifest(data, database, width=64, height=64)
    print(data.groupby(['split', 'label']).size())
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('datasets/processed_datasets/galaxies_clean.csv'))
    parser.add_argument('--database', type=Path, default=Path('datasets/images/cutouts.sqlite'))
    parser.add_argument('--per-class', type=int, default=1000, help='0 uses all eligible rows (expensive for SVC).')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--already-clean', action='store_true')
    args = parser.parse_args()
    prepare(args.catalog, args.output, args.database, args.per_class, args.seed, not args.already_clean)
