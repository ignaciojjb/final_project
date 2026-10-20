"""Pearson correlations of model input features, using training rows only."""
import argparse
from pathlib import Path
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_heatmap(correlation, output, title, n_train):
    """Save a labeled Pearson-correlation heatmap and close its figure.

    correlation is a square feature-indexed DataFrame, output is a PNG path,
    and n_train is the training sample count displayed in the title.
    NaN correlations stay gray, rather than suggesting a zero relationship.
    Annotate numbers only for small matrices to keep larger plots readable.
    """
    n = len(correlation)
    side = min(24, max(7, .25 * n + 3))
    fig, axis = plt.subplots(figsize=(side, side), constrained_layout=True)
    cmap = plt.get_cmap('RdBu_r').copy()
    # Undefined correlations (for example constant columns) must not look like zero correlation.
    cmap.set_bad('#d9d9d9')
    image = axis.imshow(np.ma.masked_invalid(correlation.to_numpy()), cmap=cmap, vmin=-1, vmax=1)
    labels = [c.replace('stamp__pca', 'PC').replace('catalog__', '') for c in correlation.columns]
    axis.set_xticks(range(n), labels, rotation=90, fontsize=7 if n > 25 else 9)
    axis.set_yticks(range(n), labels, fontsize=7 if n > 25 else 9)
    axis.set_title(f'{title}\nPearson r; training rows only (n={n_train}); gray = undefined')
    if n <= 15:
        for row in range(n):
            for col in range(n):
                value = correlation.iloc[row, col]
                if pd.notna(value):
                    axis.text(col, row, f'{value:.2f}', ha='center', va='center', fontsize=8,
                              color='white' if abs(value) > .6 else 'black')
    fig.colorbar(image, ax=axis, shrink=.7, label='Pearson correlation', ticks=[-1, -.5, 0, .5, 1])
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_correlations(run, output=None):
    """Compute training-only input correlations; return the full matrix.

    Read merged_features.csv from run; output defaults to run/plots. Select
    only stamp__pca*/catalog__ columns, excluding labels, IDs and split.
    Save full and catalog-only PNG/CSV matrices. Constant columns remain
    undefined with a warning; this is not a feature-importance calculation.
    """
    run = Path(run)
    table = pd.read_csv(run / 'merged_features.csv', dtype={'specobjid': str})
    # Prefixes select actual classifier inputs, excluding target labels/IDs/split metadata.
    features = [c for c in table if c.startswith(('stamp__pca', 'catalog__'))]
    # Held-out rows are excluded from this exploratory feature relationship matrix.
    training = table.loc[table['split'].eq('train'), features]
    if not features or len(training) < 2:
        raise ValueError('Need model feature columns and at least two training rows.')
    if not np.isfinite(training.to_numpy(dtype=float)).all():
        raise ValueError('Model features must be finite after preprocessing.')
    constant = training.columns[training.nunique().lt(2)].tolist()
    if constant:
        warnings.warn(f'Undefined correlations for constant training features: {constant}', stacklevel=2)
    output = Path(output) if output else run / 'plots'
    output.mkdir(parents=True, exist_ok=True)
    # Pearson captures linear dependence; this is not nonlinear SVC feature importance.
    full = training.corr(method='pearson')
    full.to_csv(output / 'feature_correlations.csv')
    save_heatmap(full, output / 'feature_correlations.png', 'All model input features', len(training))
    catalog = [c for c in features if c.startswith('catalog__')]
    if catalog:
        subset = full.loc[catalog, catalog]
        subset.to_csv(output / 'catalog_correlations.csv')
        save_heatmap(subset, output / 'catalog_correlations.png', 'Selected catalog features (preprocessed)', len(training))
    print(f'Correlation matrices saved to {output.resolve()}')
    return full


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--output', default=None)
    a = p.parse_args()
    plot_correlations(a.run, a.output)
