from contextlib import closing
from io import BytesIO
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import warnings

import pandas as pd
from PIL import Image
from scripts.cutouts import DownloadResult
from scripts.download_data import sha256
from scripts.plot_predictions import plot_predictions


class GalleryTests(unittest.TestCase):
    def test_selected_only_cache_and_read_only_dataset(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            database = root / 'images.sqlite'
            buffer = BytesIO()
            Image.new('RGB', (64, 64)).save(buffer, format='JPEG')
            with closing(sqlite3.connect(database)) as con, con:
                con.execute('CREATE TABLE cutouts(object_id TEXT, image BLOB, label INTEGER, split TEXT, status TEXT, ra REAL, dec REAL, fov_arcsec REAL, provider TEXT)')
                for i in range(12):
                    con.execute('INSERT INTO cutouts VALUES(?,?,?,?,?,?,?,?,?)', (str(i),buffer.getvalue(),i%2,'test','downloaded',120.+i,20.,16.,'hips2fits'))
            pd.DataFrame({'specobjid':[str(i) for i in range(12)], 'label':[i%2 for i in range(12)], 'prediction':[i%2 for i in range(12)], 'split':'test'}).to_csv(root/'predictions.csv',index=False)
            original = sha256(database)
            buffer = BytesIO()
            Image.new('RGB', (256, 256)).save(buffer, format='JPEG')
            result = DownloadResult('gallery', buffer.getvalue(), 'image/jpeg', None)
            with patch('scripts.plot_predictions._download_one', return_value=result) as download:
                output = plot_predictions(root, database, count=10, display_resolution=256)
                self.assertEqual(download.call_count, 10)
                self.assertTrue(all('width=256' in call.args[1] for call in download.call_args_list))
                plot_predictions(root, database, count=10, display_resolution=256)
                self.assertEqual(download.call_count, 10)
            self.assertEqual(sha256(database), original)
            self.assertEqual(len(list((root/'plots/display_cache').glob('*.jpg'))), 10)
            with Image.open(output) as image:
                image.verify()
            with patch('scripts.plot_predictions._download_one', return_value=DownloadResult('gallery',None,None,'network unavailable')):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    plot_predictions(root, database, count=1, display_resolution=512)
                    self.assertTrue(any('Using original stamp' in str(w.message) for w in caught))
            self.assertEqual(sha256(database), original)
