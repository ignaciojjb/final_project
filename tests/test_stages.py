from contextlib import closing
import io
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
import test_ml_training
from scripts.extract_features import extract
from scripts.train_model import train_cached
from scripts.prepare_data import prepare, validate_catalog_ids
from scripts.import_stamps import import_stamps
from scripts.download_data import download, sha256


class StagesTests(unittest.TestCase):
    def test_catalog_duplicate_policy(self):
        data = pd.DataFrame({'specobjid': pd.Series(['123', '123'], dtype='string'), 'spiral': [1, 1]})
        self.assertEqual(len(validate_catalog_ids(data)), 1)
        data.loc[1, 'spiral'] = 0
        with self.assertRaisesRegex(ValueError, 'conflicting records'):
            validate_catalog_ids(data)
        data.loc[1, 'specobjid'] = pd.NA
        with self.assertRaisesRegex(ValueError, 'missing specobjid'):
            validate_catalog_ids(data)

    def test_reader_closes_connection_on_success_and_error(self):
        from scripts.ml_training import load_dataset
        with tempfile.TemporaryDirectory() as folder:
            catalog, database = test_ml_training.TrainingTests().fixture(Path(folder))
            original_connect = sqlite3.connect
            connections = []

            def tracked_connect(*args, **kwargs):
                connection = original_connect(*args, **kwargs)
                connections.append(connection)
                return connection

            with patch('scripts.ml_training.sqlite3.connect', side_effect=tracked_connect):
                load_dataset(catalog, database)
                with patch('scripts.ml_training.stamp_to_features', side_effect=ValueError('invalid image')):
                    with self.assertRaisesRegex(ValueError, 'Cannot decode'):
                        load_dataset(catalog, database)
            for connection in connections:
                with self.assertRaises(sqlite3.ProgrammingError):
                    connection.execute('SELECT 1')

    def test_cache_retraining_and_invalidation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            catalog, db = test_ml_training.TrainingTests().fixture(root)
            cached = extract(catalog, db, root/'features.joblib', 64)
            with patch('scripts.extract_features.load_dataset', side_effect=AssertionError('Decoded again')):
                extract(catalog, db, root/'features.joblib', 64)
            model, report = train_cached(root/'features.joblib', root/'run1')
            train_cached(root/'features.joblib', root/'run2', c=2)
            self.assertEqual(report['test']['accuracy'], 1)
            with catalog.open('a') as f:
                f.write('\n')
            with self.assertRaisesRegex(ValueError, 'stale'):
                extract(catalog, db, root/'features.joblib', 64)

    def test_prepare_zip_and_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            catalog, _ = test_ml_training.TrainingTests().fixture(root)
            data = pd.read_csv(catalog, dtype={'specobjid': str})
            data['ra'], data['dec'] = 120., 20.
            data.to_csv(catalog, index=False)
            db = root/'manifest.sqlite'
            prepare(catalog, root/'prepared.csv', db, per_class=0, clean=False)
            buffer = io.BytesIO()
            Image.new('RGB', (64, 64), 'red').save(buffer, format='PNG')
            with zipfile.ZipFile(root/'stamps.zip', 'w') as z:
                z.writestr('stamp.png', buffer.getvalue())
            pd.DataFrame({'specobjid': [data.specobjid.iloc[0]], 'image_path': ['stamp.png']}).to_csv(root/'mapping.csv', index=False)
            import_stamps(root/'stamps.zip', root/'mapping.csv', db)
            import_stamps(root/'stamps.zip', root/'mapping.csv', db)
            with closing(sqlite3.connect(db)) as con, con:
                self.assertEqual(con.execute("SELECT count(*) FROM cutouts WHERE status='downloaded'").fetchone()[0], 1)

    def test_download_checksum_and_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            sample = root/'sample'
            sample.write_bytes(b'catalog data')
            digest = sha256(sample)
            response = io.BytesIO(b'catalog data')
            response.geturl = lambda: 'https://example.com/data'
            with patch('scripts.download_data.urlopen', return_value=response):
                download('https://example.com/data', root/'download', digest)
            with patch('scripts.download_data.urlopen', side_effect=AssertionError('Downloaded again')):
                download('https://example.com/data', root/'download', digest)
            with self.assertRaisesRegex(ValueError, 'different checksum'):
                download('https://example.com/data', root/'download', '0'*64)
