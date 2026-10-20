from pathlib import Path
import tempfile
import unittest
import warnings
import numpy as np
import pandas as pd
from PIL import Image
from scripts.plot_correlations import plot_correlations


class CorrelationTests(unittest.TestCase):
    def test_training_features_only_and_constant_handling(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pd.DataFrame({'specobjid': ['1','2','3','4','5'], 'label': [0,1,0,1,0],
                          'split': ['train','train','train','test','validation'],
                          'stamp__pca0': [1.,2.,3.,-100.,-200.],
                          'catalog__modelMag_r': [2.,4.,6.,100.,200.],
                          'catalog__constant': [7.,7.,7.,8.,9.]}).to_csv(root/'merged_features.csv',index=False)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                correlation = plot_correlations(root)
            self.assertEqual(correlation.shape, (3,3))
            self.assertAlmostEqual(correlation.loc['stamp__pca0','catalog__modelMag_r'],1.)
            self.assertTrue(np.isnan(correlation.loc['catalog__constant','catalog__constant']))
            self.assertTrue(any('constant training features' in str(w.message) for w in caught))
            self.assertEqual(len(list((root/'plots').glob('*.csv'))),2)
            for path in (root/'plots').glob('*.png'):
                with Image.open(path) as image:
                    image.verify()
