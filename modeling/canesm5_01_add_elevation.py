# Purpose: canesm5 01 add elevation.
# Source: cmip6_figure_CanESM5_XGB_plot_latest_checked.ipynb; see docs/source_coverage.csv for cell provenance.
# Run order and required inputs: docs/workflow.md.
# Run from repository root: python -m modeling.canesm5_01_add_elevation

# -*- coding: utf-8 -*-
"""
Build feature tables with SRTM elevation (NO training / NO prediction)

- Historical features: remove old elev, add SRTM elev
- Future features: add SRTM elev
- Interpolation: RegularGridInterpolator
"""

from __future__ import annotations

from publication_config import resource_path

import os
import numpy as np
import pandas as pd
import rioxarray as rxr
from scipy.interpolate import RegularGridInterpolator

# ======================================================
# 0. PATHS
# ======================================================
BASE_TRAIN = resource_path('models', 'CanESM5/feature/ml_train_historical')
BASE_FUT   = resource_path('models', 'CanESM5/feature/ml_pred_ssp585')

OUT_DIR = resource_path('models', 'CanESM5/ml_feature_withSRTM')
os.makedirs(OUT_DIR, exist_ok=True)

HIST_FEAT = os.path.join(BASE_TRAIN, "features_historical_1995_2014.parquet")

FUT_FILES = [
    "features_ssp585_2050_2059.parquet",
    "features_ssp585_2060_2069.parquet",
    "features_ssp585_2070_2079.parquet",
    "features_ssp585_2080_2089.parquet",
    "features_ssp585_2090_2099.parquet",
]

ELEV_TIF = resource_path('observations', 'data/elevation/elevation_1KMmn_SRTM.tif')

LAT = "Latitude"
LON = "Longitude"

# ======================================================
# 1. Load SRTM elevation (CORRECT WAY)
# ======================================================
print("[LOAD] SRTM elevation")

elev = rxr.open_rasterio(ELEV_TIF, masked=True).squeeze()

# Coordinates
elev_lat = elev.y.values          # descending (north -> south)
elev_lon = elev.x.values

# Interpolator (IMPORTANT: reverse latitude axis)
interp = RegularGridInterpolator(
    (elev_lat[::-1], elev_lon),
    elev.values[::-1, :],
    bounds_error=False,
    fill_value=np.nan
)

def interp_elev(lat, lon):
    pts = np.column_stack([lat, lon])
    return interp(pts)

# ======================================================
# 2. Process historical features
# ======================================================
print("[HIST] add SRTM elevation")

df_hist = pd.read_parquet(HIST_FEAT)

# --- remove old elevation if exists ---
for c in ["elev", "elevation"]:
    if c in df_hist.columns:
        df_hist = df_hist.drop(columns=c)

# interpolate
df_hist["elev"] = interp_elev(
    df_hist[LAT].values,
    df_hist[LON].values
).astype("float32")

out_hist = os.path.join(
    OUT_DIR, "features_historical_1995_2014_withSRTM.parquet"
)
df_hist.to_parquet(out_hist)
print("[SAVED]", out_hist)

# ======================================================
# 3. Process future features
# ======================================================
for fname in FUT_FILES:
    print("[FUT] processing", fname)

    df_f = pd.read_parquet(os.path.join(BASE_FUT, fname))

    # remove old elev if exists
    for c in ["elev", "elevation"]:
        if c in df_f.columns:
            df_f = df_f.drop(columns=c)

    df_f["elev"] = interp_elev(
        df_f[LAT].values,
        df_f[LON].values
    ).astype("float32")

    out_f = os.path.join(
        OUT_DIR, fname.replace(".parquet", "_withSRTM.parquet")
    )
    df_f.to_parquet(out_f)
    print("[SAVED]", out_f)

print("\n[DONE] All feature tables rebuilt with SRTM elevation")
