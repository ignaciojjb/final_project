from contextlib import closing
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from scripts.ml_training import load_dataset, stamp_to_features, train


class TrainingTests(unittest.TestCase):
    def test_grayscale_flatten(self):
        buffer = io.BytesIO()
        Image.new('RGB', (128, 128), 'white').save(buffer, format='PNG')
        pixels = stamp_to_features(buffer.getvalue())
        self.assertEqual(pixels.shape, (4096,))
        np.testing.assert_allclose(pixels, 1)

    def fixture(self, root):
        rng = np.random.default_rng(42)
        rows = []
        db = root / 'images.sqlite'
        with closing(sqlite3.connect(db)) as con, con:
            con.execute('CREATE TABLE cutouts(object_id TEXT PRIMARY KEY, image BLOB, label INTEGER, split TEXT, status TEXT)')
            for i in range(112):
                label = i % 2
                split = 'train' if i < 80 else 'validation' if i < 96 else 'test'
                object_id = str(9000000000000000000 + i)
                image = np.clip(rng.normal(40 + label * 150, 10, (64, 64, 3)), 0, 255).astype('uint8')
                buffer = io.BytesIO()
                Image.fromarray(image).save(buffer, format='PNG')
                con.execute('INSERT INTO cutouts VALUES(?,?,?,?,?)', (object_id, buffer.getvalue(), label, split, 'downloaded'))
                rows.append(dict(specobjid=object_id, spiral=label, elliptical=1-label, petroMag_r=15 + label, p_cs=label))
        catalog = root / 'catalog.csv'
        pd.DataFrame(rows).to_csv(catalog, index=False)
        return catalog, db

    def test_end_to_end_and_training_only_pca(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            catalog, db = self.fixture(root)
            model, metrics = train(catalog, db, root / 'output', n_components=64)
            data, features = load_dataset(catalog, db)
            self.assertEqual(features, ['petroMag_r'])
            pca = model.named_steps['features'].named_transformers_['stamp'].named_steps['pca']
            self.assertEqual(pca.n_samples_, 80)
            self.assertEqual(metrics['test']['accuracy'], 1)
            restored = joblib.load(root / 'output/model.joblib')
            np.testing.assert_array_equal(restored.predict(data), model.predict(data))
            table = pd.read_csv(root / 'output/merged_features.csv', dtype={'specobjid': str})
            self.assertEqual(table.shape, (112, 68))
            self.assertEqual(table.specobjid.iloc[0], '9000000000000000000')
            self.assertEqual(json.loads((root / 'output/metrics.json').read_text())['n_components'], 64)

    def test_label_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            catalog, db = self.fixture(Path(folder))
            with closing(sqlite3.connect(db)) as con, con:
                con.execute('UPDATE cutouts SET label=1-label')
            with self.assertRaisesRegex(ValueError, 'disagree'):
                load_dataset(catalog, db)


if __name__ == '__main__':
    unittest.main()
