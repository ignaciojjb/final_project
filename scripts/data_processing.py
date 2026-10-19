"""
Module to load and clean the galaxy dataset.
"""

import numpy as np
import pandas as pd

CLEANING_PARAMS = {
    "max_mag_err": 0.1,
    "mag_range": (10, 25),
    "min_nvote": 20,
}

def load_data(path):
    """Load the CSV, drop columns starting with 'Column', replace -9999 with NaN, and drop duplicates."""
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Column")])
    df = df.replace(-9999, np.nan)
    return df.drop_duplicates("specobjid")

def apply_cut(df, mask, name, log):
    """Apply a boolean mask, record how many rows it removes."""
    kept = df[mask]
    log.append((name, len(df) - len(kept), len(kept)))
    return kept

def clean_sample(df, params=CLEANING_PARAMS):
    """Apply a series of cuts to the DataFrame and return the cleaned DataFrame along with a log of cuts."""
    log = []
    df = apply_cut(df, df["modeDR7"] == 1, "primary only", log)
    df = apply_cut(df, df["nChild"] == 0, "not blended", log)
    df = apply_cut(df, df["petroMagErr_r"] < params["max_mag_err"], "mag error", log)
    df = apply_cut(df, df["petroMag_r"].between(*params["mag_range"]), "mag range", log)
    df = apply_cut(df, df["nvote"] >= params["min_nvote"], "enough votes", log)
    return df, pd.DataFrame(log, columns=["cut", "removed", "remaining"])
