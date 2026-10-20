from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scripts.plot_roc import load_scores, plot_roc


class ROCTests(unittest.TestCase):
    def test_uses_continuous_scores_not_hard_predictions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = pd.DataFrame({'specobjid':[str(i) for i in range(8)],
                'split':['validation']*4+['test']*4,'label':[0,0,1,1]*2,
                'prediction':[0]*8,'decision_score':[.1,.4,.35,.8]*2})
            rows.to_csv(root/'predictions.csv',index=False)
            with patch('scripts.plot_roc.joblib.load',side_effect=AssertionError('Must not load a model')):
                result=plot_roc(root)
            self.assertAlmostEqual(result['validation']['roc_auc'],.75)
            self.assertAlmostEqual(result['test']['roc_auc'],.75)
            self.assertTrue((root/'plots/roc_auc.json').is_file())
            self.assertTrue((root/'plots/roc_curve_points.csv').is_file())
            with Image.open(root/'plots/roc_curves.png') as image:
                image.verify()
            rows.loc[rows['split'].eq('test'),'label']=0
            rows.to_csv(root/'predictions.csv',index=False)
            with self.assertRaisesRegex(ValueError,'both classes'):
                plot_roc(root)

    def test_legacy_run_reconstructs_scores_without_double_preprocessing(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            raw=pd.DataFrame({'stamp__pca0':[10.,20.,30.,40.], 'catalog__petro_r0':[1.,2.,4.,5.]})
            labels=np.array([0,0,1,1])
            preprocessing=StandardScaler().fit(raw)
            transformed=preprocessing.transform(raw)
            classifier=SVC().fit(transformed,labels)
            model=Pipeline([('features',preprocessing),('svc',classifier)])
            joblib.dump(model,root/'model.joblib')
            metadata=pd.DataFrame({'specobjid':['9000000000000000000','2','3','4'],
                                   'label':labels,'split':['test']*4})
            table=pd.concat([metadata,pd.DataFrame(transformed,columns=raw.columns)],axis=1)
            # Shuffle table to prove ID-based alignment rather than row alignment.
            table.iloc[::-1].to_csv(root/'merged_features.csv',index=False)
            metadata.to_csv(root/'predictions.csv',index=False)
            loaded=load_scores(root)
            np.testing.assert_allclose(loaded.decision_score,model.decision_function(raw))
            self.assertEqual(roc_auc_score(loaded.label,loaded.decision_score),1.)
            table.loc[0,'label']=1
            table.to_csv(root/'merged_features.csv',index=False)
            with self.assertRaisesRegex(ValueError,'disagree'):
                load_scores(root)
