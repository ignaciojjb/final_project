"""Display held-out cutouts with catalog labels and saved SVC predictions."""
import argparse
import hashlib
import warnings
from contextlib import closing
from io import BytesIO
import math
from pathlib import Path
import sqlite3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from scripts.cutouts import cutout_url, _download_one

NAMES = {0: 'Elliptical', 1: 'Spiral'}


def display_image(blob, metadata, resolution, cache):
    """Fetch only a selected object's display cutout; never write to the dataset."""
    ra, dec, fov, provider = metadata
    url = cutout_url(ra, dec, fov, resolution, resolution, provider)
    target = Path(cache) / (hashlib.sha256(url.encode()).hexdigest() + '.jpg')

    def decode(data):
        with Image.open(BytesIO(data)) as image:
            if image.size != (resolution, resolution):
                raise ValueError(f'Expected {resolution}x{resolution}, received {image.size}')
            return image.convert('RGB').copy()

    try:
        if target.exists():
            try:
                return decode(target.read_bytes())
            except (ValueError, OSError):
                target.unlink()
        result = _download_one('gallery', url, timeout=20, retries=1)
        if result.image is None:
            raise ValueError(result.error)
        image = decode(result.image)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.part')
        temporary.write_bytes(result.image)
        temporary.replace(target)
        return image
    except (OSError, ValueError) as exc:
        warnings.warn(f'High-resolution display unavailable: {exc}. Using original stamp.', stacklevel=2)
        with Image.open(BytesIO(blob)) as image:
            return image.convert('RGB').copy()


def plot_predictions(run, database, count=10, split='test', seed=42, errors_only=False, output=None,
                     display_resolution=None):
    if count < 1:
        raise ValueError('count must be positive.')
    if display_resolution is not None and not 64 <= display_resolution <= 1024:
        raise ValueError('Display resolution must be between 64 and 1024 pixels.')
    run, database = Path(run), Path(database).resolve()
    predictions = pd.read_csv(run / 'predictions.csv', dtype={'specobjid': 'string'})
    rows = predictions.loc[predictions['split'].eq(split)].copy()
    if errors_only:
        rows = rows.loc[rows.label.ne(rows.prediction)]
    if rows.empty:
        raise ValueError(f'No saved predictions match split={split}, errors_only={errors_only}.')
    if rows.specobjid.isna().any() or rows.specobjid.duplicated().any():
        raise ValueError('Saved prediction IDs must be non-null and unique within the selected split.')
    rows = rows.sample(n=min(count, len(rows)), random_state=seed)
    if not database.is_file():
        raise FileNotFoundError(database)
    images = []
    with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as connection:
        for row in rows.itertuples(index=False):
            record = connection.execute(
                "SELECT image, label, split FROM cutouts WHERE object_id=? AND status='downloaded'",
                (row.specobjid,),
            ).fetchone()
            if record is None:
                raise ValueError(f'No downloaded image for {row.specobjid}. Use the same database as training.')
            blob, label, stored_split = record
            if label != row.label or (stored_split is not None and stored_split != split):
                raise ValueError(f'Database label/split disagrees with predictions for {row.specobjid}.')
            if display_resolution is not None:
                metadata = connection.execute(
                    'SELECT ra, dec, fov_arcsec, provider FROM cutouts WHERE object_id=?',
                    (row.specobjid,),
                ).fetchone()
                image = display_image(blob, metadata, display_resolution, run / 'plots' / 'display_cache')
                images.append((row, image))
            else:
                with Image.open(BytesIO(blob)) as image:
                    images.append((row, image.convert('RGB').copy()))
    columns = min(5, len(images))
    fig, axes = plt.subplots(math.ceil(len(images) / columns), columns,
                             figsize=(3.2 * columns, 3.5 * math.ceil(len(images) / columns)),
                             squeeze=False, constrained_layout=True)
    for axis, (row, image) in zip(axes.flat, images):
        correct = row.label == row.prediction
        axis.imshow(image)
        axis.set_title(f'Catalog: {NAMES[row.label]}\nPredicted: {NAMES[row.prediction]}',
                       color='#18753c' if correct else '#b52323', fontsize=10)
        axis.set_xlabel(f'{row.specobjid} | display {image.width}x{image.height}', fontsize=7)
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in list(axes.flat)[len(images):]:
        axis.axis('off')
    fig.suptitle(f'{split.title()} sample (n={len(images)}): green = correct, red = incorrect\n'
                 'Reference labels are catalog morphology classifications', fontsize=12)
    output = Path(output) if output else run / 'plots' / f'predictions_{split}{"_errors" if errors_only else ""}.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f'Saved {output.resolve()}')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--database', required=True)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--split', choices=['test', 'validation'], default='test')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--errors-only', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--display-resolution', type=int, default=None, help='Download only selected display cutouts at this size (e.g. 256); reuse a separate cache.')
    args = parser.parse_args()
    plot_predictions(args.run, args.database, args.count, args.split, args.seed, args.errors_only, args.output, args.display_resolution)
