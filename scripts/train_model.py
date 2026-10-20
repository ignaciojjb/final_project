"""Train a new SVC from cached features without downloading, decoding or refitting PCA."""
import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix


def train_cached(features, output, c=1.0, gamma='scale'):
    output = Path(output)
    if output.exists():
        raise ValueError('Run directory exists. Choose a different --output to preserve results.')
    if not Path(features).is_file():
        raise FileNotFoundError(
            f'Feature cache not found: {features}. First run '
            'python -m scripts.extract_features --components 64 '
            '(requires a prepared catalog and downloaded/imported images).'
        )
    cache = joblib.load(features)  # Only use artifacts created by this project/trusted sources.
    metadata, x = cache['metadata'], cache['X']
    classifier = SVC(C=c, gamma=gamma, kernel='rbf', class_weight='balanced')
    mask = metadata['split'].eq('train').to_numpy()
    classifier.fit(x[mask], metadata.loc[mask, 'label'])
    output = Path(output)
    if output.exists():
        raise ValueError('Run directory exists. Choose a different --output to preserve results.')
    output.mkdir(parents=True)
    model = Pipeline([('features', cache['preprocessing']), ('svc', classifier)])
    joblib.dump(model, output / 'model.joblib')
    report = {'C': c, 'gamma': gamma, 'feature_signature': cache['signature'], 'catalog_features': cache['catalog_features']}
    predictions = []
    for split in ('validation', 'test'):
        mask = metadata['split'].eq(split).to_numpy()
        y = metadata.loc[mask, 'label']
        pred = classifier.predict(x[mask])
        report[split] = {'accuracy': accuracy_score(y, pred), 'balanced_accuracy': balanced_accuracy_score(y, pred),
                         'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1]).tolist(),
                         'classification_report': classification_report(y, pred, labels=[0, 1], output_dict=True, zero_division=0)}
        rows = metadata.loc[mask].copy()
        rows['prediction'] = pred
        predictions.append(rows)
    (output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    pd.concat(predictions).to_csv(output / 'predictions.csv', index=False)
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
