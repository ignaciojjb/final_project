"""SDSS galaxy features: matched-profile model colors and total-light proxy.

Reference: https://www.sdss3.org/dr9/algorithms/magnitudes.php
Magnitudes are SDSS asinh magnitudes; no logarithmic flux conversion is applied.
"""
import numpy as np
import pandas as pd

FEATURE_DEFINITIONS = {
    'model_u_g0': '(modelMag_u - extinction_u) - (modelMag_g - extinction_g)',
    'model_g_r0': '(modelMag_g - extinction_g) - (modelMag_r - extinction_r)',
    'model_r_i0': '(modelMag_r - extinction_r) - (modelMag_i - extinction_i)',
    'model_i_z0': '(modelMag_i - extinction_i) - (modelMag_z - extinction_z)',
    'cmodel_r0': 'cModelMag_r - extinction_r (galaxy total-light estimate)',
    'petro_r0': 'petroMag_r - extinction_r (fallback; profile-dependent flux loss)',
    'petroR50_r': 'r-band Petrosian half-light radius in arcseconds (apparent size)',
    'concentration_r': 'petroR90_r / petroR50_r (positive radii only)',
    'fracDeV_r': 'r-band de Vaucouleurs component fraction, when available (0..1)',
}


def build_photometry(catalog):
    result = pd.DataFrame(index=catalog.index)

    def numeric(column):
        return pd.to_numeric(catalog[column], errors='raise').replace([-9999, np.inf, -np.inf], np.nan)

    corrected = {}
    for band in 'ugriz':
        if {f'modelMag_{band}', f'extinction_{band}'}.issubset(catalog):
            corrected[band] = numeric(f'modelMag_{band}') - numeric(f'extinction_{band}')
    for first, second in zip('ugri', 'griz'):
        if first in corrected and second in corrected:
            result[f'model_{first}_{second}0'] = corrected[first] - corrected[second]
    # Model colors share the r-selected profile; do not mix Petrosian/PSF bands for colors.
    if 'extinction_r' in catalog:
        if 'cModelMag_r' in catalog:
            result['cmodel_r0'] = numeric('cModelMag_r') - numeric('extinction_r')
        elif 'petroMag_r' in catalog:
            result['petro_r0'] = numeric('petroMag_r') - numeric('extinction_r')
    if 'petroR50_r' in catalog:
        radius = numeric('petroR50_r').where(lambda values: values > 0)
        result['petroR50_r'] = radius
        if 'petroR90_r' in catalog:
            outer = numeric('petroR90_r').where(lambda values: values > 0)
            result['concentration_r'] = (outer / radius).where(outer >= radius)
    if 'fracDeV_r' in catalog:
        fraction = numeric('fracDeV_r')
        result['fracDeV_r'] = fraction.where(fraction.between(0, 1))
    if result.empty or result.notna().sum().sum() == 0:
        raise ValueError('No usable SDSS photometric features. Supply modelMag/extinction pairs, '
                         'petroMag_r with extinction_r, or documented structural columns. '
                         'Extinction is not silently assumed to be zero.')
    return result, list(result.columns)
