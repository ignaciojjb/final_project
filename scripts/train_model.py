"""Train a new SVC from cached features without downloading, decoding or refitting PCA."""
import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, roc_auc_score


def train_cached(features, output, c=1.0, gamma='scale'):
    """Fit an SVC from saved features; return (complete_model, report).

    features is a trusted version-2 feature-cache file, output is a new run
    directory, and c/gamma configure the RBF kernel. Cached X is already
    preprocessed: fit only the SVC on training rows, then evaluate held-out
    rows. Save preprocessing+SVC together for pixel/engineered-column input,
    plus metrics, predictions/scores and the transformed feature CSV.
    No image download, decoding or PCA fitting is performed here.
    """
    output = Path(output)
    if output.exists():
        raise ValueError('Run directory exists. Choose a different --output to preserve results.')
    if not Path(features).is_file():
        raise FileNotFoundError(
            f'Feature cache not found: {features}. First run '
            'python -m scripts.extract_features --components 64 '
            '(requires a prepared catalog and downloaded/imported images).'
        )
    # 1. Cached X is already transformed; feeding it through preprocessing again would be wrong.
    cache = joblib.load(features)  # Only use artifacts created by this project/trusted sources.
    if cache.get('signature', {}).get('photometry_version') != 2:
        raise ValueError('Legacy feature cache: extract new photometric features and retrain. For old hips2fits images, rebuild the image database first. See DOWNLOAD_GUIDE.md.')
    metadata, x = cache['metadata'], cache['X']
    # 2. Only the classifier changes between C/gamma experiments on this fixed cache.
    classifier = SVC(C=c, gamma=gamma, kernel='rbf', class_weight='balanced')
    mask = metadata['split'].eq('train').to_numpy()
    # Training rows alone learn the decision boundary; held-out labels are evaluated below.
    classifier.fit(x[mask], metadata.loc[mask, 'label'])
    output = Path(output)
    if output.exists():
        raise ValueError('Run directory exists. Choose a different --output to preserve results.')
    output.mkdir(parents=True)
    # Package the existing transforms with the new SVC for future pixel/engineered-column inputs.
    model = Pipeline([('features', cache['preprocessing']), ('svc', classifier)])
    joblib.dump(model, output / 'model.joblib')
    report = {'C': c, 'gamma': gamma, 'feature_signature': cache['signature'], 'catalog_features': cache['catalog_features'], 'feature_definitions': cache.get('feature_definitions', {})}
    predictions = []
    # 3. Report splits independently; test must not drive repeated hyperparameter choices.
    for split in ('validation', 'test'):
        mask = metadata['split'].eq(split).to_numpy()
        y = metadata.loc[mask, 'label']
        pred = classifier.predict(x[mask])
        # Continuous margins rank spiral (1) above elliptical (0) for ROC/AUC.
        score = classifier.decision_function(x[mask])
        report[split] = {'roc_auc': float(roc_auc_score(y, score)), 'accuracy': accuracy_score(y, pred), 'balanced_accuracy': balanced_accuracy_score(y, pred),
                         'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1]).tolist(),
                         'classification_report': classification_report(y, pred, labels=[0, 1], output_dict=True, zero_division=0)}
        rows = metadata.loc[mask].copy()
        rows['prediction'] = pred
        rows['decision_score'] = score
        predictions.append(rows)
    (output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    pd.concat(predictions).to_csv(output / 'predictions.csv', index=False)
    # Keep metadata beside exported features for auditing; metadata is not a predictor.
    table = pd.DataFrame(x, columns=cache['preprocessing'].get_feature_names_out())
    pd.concat([metadata.reset_index(drop=True), table], axis=1).to_csv(output / 'merged_features.csv', index=False)
    print(json.dumps(report, indent=2))
    return model, report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features', default='artifacts/features.joblib')
    p.add_argument('--output', default='artifacts/runs/svc-001')
    p.add_argument('--C', type=float, default=1.0)
    p.add_argument('--gamma', default='scale')
    a = p.parse_args()
    gamma = a.gamma if a.gamma in ('scale', 'auto') else float(a.gamma)
    train_cached(a.features, a.output, a.C, gamma)
