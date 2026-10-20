"""
Module to load and clean the galaxy dataset.
"""

import numpy as np
import pandas as pd

# Baseline starter-set thresholds; these are not a complete SDSS flag-quality model.
CLEANING_PARAMS = {
    "max_mag_err": 0.1,
    "mag_range": (10, 25),
    "min_nvote": 20,
}

def load_data(path):
    """Load a CSV using the original notebook's convenience cleaning.

    Returns a DataFrame after dropping export-only Column* fields, replacing
    -9999 and retaining the first occurrence of each specobjid. This legacy
    helper does NOT distinguish conflicting repeated IDs or enforce text
    ID parsing; the staged pipeline uses prepare_data validation instead.
    """
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Column")])
    df = df.replace(-9999, np.nan)
    return df.drop_duplicates("specobjid")

def apply_cut(df, mask, name, log):
    """Filter rows with an index-aligned mask and append one audit-log entry.

    log is mutated with (cut name, removed count, remaining count); return
    the surviving DataFrame. Recording cuts sequentially makes their order
    and sample losses visible instead of reporting only a final row count.
    """
    kept = df[mask]
    log.append((name, len(df) - len(kept), len(kept)))
    return kept

def clean_sample(df, params=CLEANING_PARAMS):
    """Apply baseline SDSS quality cuts and return (cleaned_rows, cut_log).

    params supplies magnitude-error, magnitude-range and vote thresholds.
    The input must contain the fields used by each cut; missing columns
    fail rather than silently skipping quality controls. nChild == 0
    removes parents with deblend children but does not guarantee isolation.
    Label filtering and train/validation/test splitting happen later.
    """
    log = []
    df = apply_cut(df, df["modeDR7"] == 1, "primary only", log)
    # No children does not exclude every neighboring/deblended source in the image.
    df = apply_cut(df, df["nChild"] == 0, "no deblend children (not a guarantee of isolation)", log)
    df = apply_cut(df, (df["petroMagErr_r"] >= 0) & (df["petroMagErr_r"] < params["max_mag_err"]), "mag error", log)
    df = apply_cut(df, df["petroMag_r"].between(*params["mag_range"]), "mag range", log)
    df = apply_cut(df, df["nvote"] >= params["min_nvote"], "enough votes", log)
    return df, pd.DataFrame(log, columns=["cut", "removed", "remaining"])
