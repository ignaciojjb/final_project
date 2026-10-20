"""Clean a catalog, fix splits once and create a resumable image manifest."""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scripts.data_processing import clean_sample
from scripts.ml_training import assign_splits
from scripts.cutouts import prepare_manifest


def validate_catalog_ids(data, resolve_dr8_matches=False):
    """Return a copy with exact duplicate rows removed and IDs validated.

    data must contain text specobjid values. Missing IDs and conflicting
    records raise ValueError rather than selecting an arbitrary row.
    resolve_dr8_matches opts into nearest-distance resolution only when
    every column except dr8objid/distance agrees and the minimum is unique.
    The caller's DataFrame and source CSV are not modified.
    """
    if 'specobjid' not in data:
        raise ValueError('Catalog must contain a specobjid column.')
    missing = data.specobjid.isna() | data.specobjid.str.strip().eq('')
    if missing.any():
        raise ValueError(f'Catalog contains {int(missing.sum())} rows with missing specobjid. Inspect these rows before preparing data.')
    # Equal IDs alone are insufficient: only entirely equal rows are safe duplicates.
    unique = data.drop_duplicates().copy()
    removed = len(data) - len(unique)
    if removed:
        print(f'Removed {removed} exact duplicate catalog rows.')
    repeated = unique.specobjid.duplicated(keep=False)
    if resolve_dr8_matches and repeated.any() and {'dr8objid', 'distance'}.issubset(unique):
        discard = []
        for object_id, group in unique.loc[repeated].groupby('specobjid', sort=False):
            # Restrict nearest-match resolution to otherwise identical measurements.
            different = set(group.columns[group.nunique(dropna=False).gt(1)])
            distances = pd.to_numeric(group.distance, errors='coerce')
            if (different <= {'dr8objid', 'distance'}
                    and np.isfinite(distances).all() and distances.ge(0).all()
                    and group.dr8objid.notna().all()
                    and distances.eq(distances.min()).sum() == 1):
                best = distances.idxmin()
                discard.extend(group.index[group.index != best])
        unique = unique.drop(index=discard)
        print(f'Resolved {len(discard)} extra DR8 matches by unique nearest distance; all other columns agree.')
        repeated = unique.specobjid.duplicated(keep=False)
    if repeated.any():
        ids = unique.loc[repeated, 'specobjid'].drop_duplicates()
        raise ValueError(
            f'Catalog contains {len(ids)} specobjid values with conflicting records '
            f'({int(repeated.sum())} rows). Example IDs: {ids.head(5).tolist()}. '
            'Inspect or resolve the catalog join; no arbitrary record was selected.'
        )
    return unique


def prepare(catalog, output, database, per_class=1000, seed=42, clean=True, resolve_dr8_matches=False):
    """Create a balanced prepared catalog and a pending-image SQLite manifest.

    catalog is a raw compatible CSV; output and database must be new paths.
    clean controls baseline quality cuts, not ID or class validation.
    per_class is the count per label (0 means all eligible rows); seed makes
    selection/splitting reproducible. Existing complete splits are retained.
    Return prepared rows after writing the CSV and manifest. This stage
    downloads no images and fits no preprocessing or classifier.
    """
    output = Path(output)
    if output.exists() or Path(database).exists():
        raise ValueError('Prepared catalog or database already exists. Reuse it, or choose new paths.')
    # 1. Preserve exact IDs before cleaning or cross-matching rows.
    data = pd.read_csv(catalog, dtype={c: 'string' for c in ['specobjid','objid','dr7objid','dr8objid']})
    data = validate_catalog_ids(data, resolve_dr8_matches)
    if clean:
        data, log = clean_sample(data.replace(-9999, float('nan')))
        print(log.to_string(index=False))
    # 2. Labels are reference targets; discard ambiguous flags before balancing.
    valid = data.spiral.isin([0, 1]) & data.elliptical.isin([0, 1])
    valid &= data.spiral.add(data.elliptical).eq(1)
    if 'uncertain' in data:
        valid &= data.uncertain.eq(0)
    data = data.loc[valid].copy()
    data['label'] = data.spiral.astype(int)
    # 3. Balance first, then split, so each partition can retain both classes.
    if per_class:
        data = pd.concat([data.loc[data.label.eq(label)].sample(n=per_class, random_state=seed) for label in (0, 1)], ignore_index=True)
    if 'split' not in data:
        data['split'] = None
    # 4. Freeze partitions in the CSV and SQLite manifest before image downloads.
    data = assign_splits(data, seed)
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
    parser.add_argument('--resolve-dr8-matches', action='store_true', help='Keep the unique nearest DR8 match only if all non-match columns agree.')
    args = parser.parse_args()
    prepare(args.catalog, args.output, args.database, args.per_class, args.seed, not args.already_clean, args.resolve_dr8_matches)
