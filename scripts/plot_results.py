"""Plot saved classification results without retraining or loading a model."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from scripts.plot_correlations import plot_correlations


def plot_results(run, output=None):
    run = Path(run)
    metrics = json.loads((run / 'metrics.json').read_text())
    output = Path(output) if output else run / 'plots'
    output.mkdir(parents=True, exist_ok=True)
    names = ['Elliptical', 'Spiral']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, split in zip(axes, ['validation', 'test']):
        matrix = np.asarray(metrics[split]['confusion_matrix'])
        ConfusionMatrixDisplay(matrix, display_labels=names).plot(ax=axis, cmap='Blues', colorbar=False, values_format='d')
        axis.set_title(f'{split.title()} (n={matrix.sum()})')
    fig.savefig(output / 'confusion_matrices.png', dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    x = np.arange(2)
    for offset, split, color in [(-.18, 'validation', '#2878b5'), (.18, 'test', '#e07b39')]:
        values = [metrics[split]['accuracy'], metrics[split]['balanced_accuracy']]
        bars = axis.bar(x + offset, values, width=.36, label=split.title(), color=color)
        axis.bar_label(bars, fmt='%.3f', padding=3)
    axis.set(xticks=x, xticklabels=['Accuracy', 'Balanced accuracy'], ylim=(0, 1.12), ylabel='Score', title='Held-out classification scores')
    axis.legend()
    fig.savefig(output / 'scores.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, split in zip(axes, ['validation', 'test']):
        report = metrics[split]['classification_report']
        for offset, label, name in [(-.18, 0, 'Elliptical'), (.18, 1, 'Spiral')]:
            entry = report.get(str(label), report.get(name.lower()))
            if entry is None:
                raise ValueError(f'Missing class {label} in {split} classification report')
            axis.bar(np.arange(3) + offset, [entry[k] for k in ['precision', 'recall', 'f1-score']], width=.36, label=f'{name} (n={int(entry["support"])})')
        axis.set(xticks=np.arange(3), xticklabels=['Precision', 'Recall', 'F1'], ylim=(0, 1.15), title=split.title(), ylabel='Score')
        axis.legend(fontsize=8)
    fig.savefig(output / 'per_class_scores.png', dpi=180)
    plt.close(fig)

    table = pd.read_csv(run / 'merged_features.csv', dtype={'specobjid': str})
    components = [c for c in table if c.startswith('stamp__pca')]
    if len(components) >= 2:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
        for axis, split in zip(axes, ['train', 'validation', 'test']):
            for label, name in enumerate(names):
                rows = table.loc[table['split'].eq(split) & table.label.eq(label)]
                axis.scatter(rows[components[0]], rows[components[1]], s=18, alpha=.6, label=name)
            axis.set(title=split.title(), xlabel='Standardized image PC1', ylabel='Standardized image PC2')
            axis.legend(fontsize=8)
        fig.suptitle('First two image components only; SVC uses all merged features', fontsize=11)
        fig.savefig(output / 'pca_projection.png', dpi=180)
        plt.close(fig)
    plot_correlations(run, output)
    print(f'Plots saved to {output.resolve()}')
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, help='Directory containing metrics.json and merged_features.csv')
    p.add_argument('--output', default=None)
    a = p.parse_args()
    plot_results(a.run, a.output)
