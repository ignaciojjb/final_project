from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest
from urllib.parse import parse_qs, urlsplit
from unittest.mock import patch
import numpy as np
import pandas as pd

from scripts.cutouts import cutout_url, prepare_manifest, validate_stored_requests
from scripts.photometry import build_photometry
from scripts.download_catalog import build_query, download_catalog
from scripts.rebuild_images import rebuild


class SDSSTests(unittest.TestCase):
    def test_fov_units_and_gallery_size(self):
        for pixels in [64, 256]:
            query = parse_qs(urlsplit(cutout_url(120.,20.,36.,pixels,pixels)).query)
            self.assertAlmostEqual(float(query['fov'][0]), .01)
            self.assertEqual(int(query['width'][0]), pixels)
        query = parse_qs(urlsplit(cutout_url(120.,20.,36.,64,64,'sdss')).query)
        self.assertAlmostEqual(float(query['scale'][0]),36/64)
        with self.assertRaisesRegex(ValueError, 'RA'):
            cutout_url(400.,20.,16.)

    def test_legacy_request_rejected_and_radius_has_known_units(self):
        with tempfile.TemporaryDirectory() as folder:
            db = Path(folder)/'images.sqlite'
            frame = pd.DataFrame({'specobjid':['123'], 'ra':[120.], 'dec':[20.],
                                  'spiral':[1], 'elliptical':[0], 'petroR90_r':[10.], 'size':[999.]})
            prepare_manifest(frame, db, width=64,height=64)
            with closing(sqlite3.connect(db)) as con, con:
                validate_stored_requests(con)
                fov = con.execute('SELECT fov_arcsec FROM cutouts').fetchone()[0]
                self.assertEqual(fov,30.)
                con.execute("UPDATE cutouts SET url='https://example.com/?fov=30&width=64&height=64'")
                with self.assertRaisesRegex(ValueError,'legacy hips2fits'):
                    validate_stored_requests(con)

    def test_dereddened_model_colors_and_petro_fallback(self):
        frame = pd.DataFrame({'modelMag_g':[18.], 'extinction_g':[.3],
                              'modelMag_r':[17.], 'extinction_r':[.2],
                              'petroMag_r':[16.5], 'petroMagErr_r':[.05],
                              'petroR50_r':[2.], 'petroR90_r':[6.],
                              'psfMag_r':[30.], 'p_el':[1.], 'spiral':[0]})
        result, features = build_photometry(frame)
        self.assertAlmostEqual(result.model_g_r0.iloc[0], .9)
        self.assertAlmostEqual(result.petro_r0.iloc[0],16.3)
        self.assertEqual(result.concentration_r.iloc[0],3.)
        self.assertNotIn('petroMagErr_r',features)
        self.assertNotIn('extinction_r',features)
        frame['cModelMag_r']=16.4
        result, features=build_photometry(frame)
        self.assertIn('cmodel_r0',features)
        self.assertNotIn('petro_r0',features)
        self.assertAlmostEqual(result.cmodel_r0.iloc[0],16.2)
        frame['petroR50_r']=0.
        result,_=build_photometry(frame)
        self.assertTrue(np.isnan(result.concentration_r.iloc[0]))

    def test_public_catalog_query_and_exact_ids(self):
        query = build_query(1,0,10)
        self.assertIn('p.dr8objid=z.objid',query)
        self.assertIn('CONVERT(varchar(20),z.specobjid)',query)
        self.assertNotIn('SELECT *',query)
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'catalog.csv'
            with patch('scripts.download_catalog.query_rows',side_effect=[
                [{'specobjid':'9000000000000000000','spiral':0,'elliptical':1}],
                [{'specobjid':'9000000000000000001','spiral':1,'elliptical':0}],
            ]):
                download_catalog(output,per_class=1)
            data=pd.read_csv(output,dtype={'specobjid':str})
            self.assertEqual(data.specobjid.iloc[0],'9000000000000000000')
            self.assertTrue(output.with_suffix('.provenance.json').is_file())

    def test_rebuild_preserves_sample_and_splits(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            frame=pd.DataFrame({'specobjid':[str(i) for i in range(6)],
                                'spiral':[0,1]*3,'elliptical':[1,0]*3,
                                'split':['train','train','validation','validation','test','test'],
                                'ra':120.,'dec':20.})
            frame.to_csv(root/'prepared.csv',index=False)
            self.assertEqual(rebuild(root/'prepared.csv',root/'new.sqlite'),6)
            with closing(sqlite3.connect(root/'new.sqlite')) as con:
                actual=con.execute('SELECT object_id,split FROM cutouts ORDER BY object_id').fetchall()
                self.assertEqual(actual,list(zip(frame.specobjid,frame['split'])))
                validate_stored_requests(con)
