"""Import ZIP images into an existing manifest through an explicit ID/path mapping."""
import argparse
from io import BytesIO
from pathlib import Path
import sqlite3
import zipfile
import pandas as pd
from PIL import Image


def import_stamps(archive, mapping, database):
    mapping = pd.read_csv(mapping, dtype=str)
    if not {'specobjid', 'image_path'}.issubset(mapping):
        raise ValueError('Mapping CSV needs specobjid,image_path columns (exact ZIP member names).')
    if mapping.isna().any().any() or mapping.specobjid.duplicated().any() or mapping.image_path.duplicated().any():
        raise ValueError('Mapping must contain unique non-null IDs and image paths.')
    with zipfile.ZipFile(archive) as source, sqlite3.connect(Path(database).resolve().as_uri() + '?mode=rw', uri=True) as con:
        imported = 0
        for row in mapping.itertuples(index=False):
            record = con.execute('SELECT status FROM cutouts WHERE object_id=?', (row.specobjid,)).fetchone()
            if record is None or record[0] == 'downloaded':
                continue
            info = source.getinfo(row.image_path)
            if info.file_size > 20 * 1024 * 1024:
                raise ValueError(f'Image too large: {row.image_path}')
            blob = source.read(info)
            with Image.open(BytesIO(blob)) as image:
                image.verify()
            con.execute("UPDATE cutouts SET image=?,status='downloaded',error=NULL WHERE object_id=?", (blob, row.specobjid))
            imported += 1
    print(f'Imported {imported} images; reruns skip downloaded objects.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--mapping', required=True)
    parser.add_argument('--database', default='datasets/images/cutouts.sqlite')
    args = parser.parse_args()
    import_stamps(args.archive, args.mapping, args.database)
