"""Plot held-out ROC curves using SVC decision scores (positive class: spiral=1)."""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def load_scores(run):
    run = Path(run)
    predictions = pd.read_csv(run / 'predictions.csv', dtype={'specobjid': 'string'})
    required = {'specobjid', 'label', 'split'}
    if not required.issubset(predictions):
        raise ValueError(f'Missing prediction columns: {sorted(required-set(predictions))}')
    if predictions.specobjid.isna().any() or predictions.specobjid.duplicated().any():
        raise ValueError('Held-out prediction IDs must be non-null and unique.')
    if not predictions.label.isin([0, 1]).all():
        raise ValueError('ROC requires binary labels 0=elliptical, 1=spiral.')
    if 'decision_score' not in predictions:
        # Older runs already store transformed features. Apply only the fitted SVC,
        # not the complete pipeline, which would preprocess them for a second time.
        model = joblib.load(run / 'model.joblib')  # Load trusted local model artifacts only.
        classifier = model.named_steps['svc']
        if list(classifier.classes_) != [0, 1]:
            raise ValueError('Saved classifier must have class order [0, 1].')
        names = list(model.named_steps['features'].get_feature_names_out())
        table = pd.read_csv(run / 'merged_features.csv', dtype={'specobjid': 'string'})
        if table.specobjid.isna().any() or table.specobjid.duplicated().any():
            raise ValueError('Feature table IDs must be non-null and unique.')
        if not set(names).issubset(table):
            raise ValueError('Saved feature table does not match the fitted preprocessing.')
        aligned = predictions.merge(table[['specobjid', 'label', 'split', *names]],
                                    on='specobjid', how='left', validate='one_to_one',
                                    suffixes=('', '_features'), indicator=True)
        if (not aligned['_merge'].eq('both').all()
                or not aligned.label.eq(aligned.label_features).all()
                or not aligned['split'].eq(aligned.split_features).all()):
            raise ValueError('Prediction IDs, labels or splits disagree with saved features.')
        values = aligned[names].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError('Saved model features must be finite.')
        predictions['decision_score'] = classifier.decision_function(values)
    if not np.isfinite(predictions.decision_score.to_numpy(dtype=float)).all():
        raise ValueError('ROC decision scores must be finite.')
    return predictions


def plot_roc(run, output=None):
    run = Path(run)
    rows = load_scores(run)
    curves, summary = [], {'positive_class': 1, 'positive_class_name': 'spiral',
                            'score_type': 'SVC decision_function (not a probability)'}
    for split in ['validation', 'test']:
        subset = rows.loc[rows['split'].eq(split)]
        if set(subset.label) != {0, 1}:
            raise ValueError(f'ROC/AUC is undefined: {split} must contain both classes.')
        scores = subset.decision_score.to_numpy(dtype=float)
        fpr, tpr, thresholds = roc_curve(subset.label, scores, pos_label=1)
        auc = float(roc_auc_score(subset.label, scores))
        summary[split] = {'roc_auc': auc, 'n': len(subset),
                          'n_positive': int(subset.label.eq(1).sum()),
                          'n_negative': int(subset.label.eq(0).sum())}
        curves.append(pd.DataFrame({'split': split, 'false_positive_rate': fpr,
                                    'true_positive_rate': tpr, 'decision_threshold': thresholds}))
    output = Path(output) if output else run / 'plots'
    output.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    for curve in curves:
        split = curve['split'].iloc[0]
        axis.plot(curve.false_positive_rate, curve.true_positive_rate,
                  label=f'{split.title()} (AUC = {summary[split]["roc_auc"]:.3f})', linewidth=2)
    axis.plot([0, 1], [0, 1], '--', color='gray', label='Chance (AUC = 0.5)')
    axis.set(xlabel='False positive rate', ylabel='True positive rate (spiral recall)',
             title='Held-out ROC curves — positive class: spiral', xlim=(0, 1), ylim=(0, 1.02))
    axis.legend(loc='lower right')
    axis.grid(alpha=.2)
    fig.savefig(output / 'roc_curves.png', dpi=180)
    plt.close(fig)
    pd.concat(curves, ignore_index=True).to_csv(output / 'roc_curve_points.csv', index=False)
    (output / 'roc_auc.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'ROC/AUC saved to {output.resolve()}')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    plot_roc(args.run, args.output)
