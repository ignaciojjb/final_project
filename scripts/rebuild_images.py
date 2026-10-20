"""Create a corrected image manifest from an existing prepared sample, preserving splits."""
import argparse
from pathlib import Path
import pandas as pd
from scripts.cutouts import prepare_manifest
from scripts.prepare_data import validate_catalog_ids
from scripts.ml_training import assign_splits


def rebuild(catalog, database, provider='hips2fits'):
    if Path(database).exists():
        raise ValueError('Choose a NEW database path; old data will not be overwritten.')
    data = pd.read_csv(catalog, dtype={'specobjid': 'string'})
    data = validate_catalog_ids(data)
    if 'split' not in data or data['split'].isna().any():
        raise ValueError('An existing prepared catalog with complete splits is required.')
    assign_splits(data.assign(label=data.spiral.astype(int)))
    added = prepare_manifest(data, database, provider=provider, width=64, height=64)
    print(f'Created {added} pending images in {database}; catalog and splits unchanged.')
    return added


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog', required=True)
    p.add_argument('--database', required=True)
    p.add_argument('--provider', choices=['hips2fits','sdss'], default='hips2fits')
    a = p.parse_args()
    rebuild(a.catalog, a.database, a.provider)
