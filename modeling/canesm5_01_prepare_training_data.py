# -*- coding: utf-8 -*-
"""
Prepare CanESM5 feature and training tables used by the final XGBoost workflow.

This script consolidates the original elevation, label-merge, cleaning and
strict-land steps without changing the scientific workflow:

1. Add SRTM elevation to historical and SSP585 feature tables using the
   original RegularGridInterpolator method.
2. Merge only the required observational target columns with historical
   features.
3. Apply |latitude| <= 63 degrees and Land_Sea == 0.
4. Apply the Natural Earth 110m point-in-land strict-land filter.
5. Save the final strict-land training table expected by the training scripts.

Run from the repository root:
    python -m modeling.canesm5_01_prepare_training_data
"""

from __future__ import annotations

import os

import cartopy.io.shapereader as shpreader
import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as rxr
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import Point

from publication_config import resource_path

# -----------------------------------------------------------------------------
# Paths and columns
# -----------------------------------------------------------------------------
BASE_HIST = resource_path("models", "CanESM5/feature/ml_train_historical")
BASE_FUTURE = resource_path("models", "CanESM5/feature/ml_pred_ssp585")
OUT_DIR = resource_path("models", "CanESM5/ml_feature_withSRTM")
os.makedirs(OUT_DIR, exist_ok=True)

LABEL_XLSX = resource_path("observations", "flashhail_matched_new2.xlsx")
ELEV_TIF = resource_path("observations", "data/elevation/elevation_1KMmn_SRTM.tif")

HIST_FEATURE_FILE = os.path.join(
    BASE_HIST, "features_historical_1995_2014.parquet"
)
FUTURE_FILES = [
    "features_ssp585_2050_2059.parquet",
    "features_ssp585_2060_2069.parquet",
    "features_ssp585_2070_2079.parquet",
    "features_ssp585_2080_2089.parquet",
    "features_ssp585_2090_2099.parquet",
]

LAT = "Latitude"
LON = "Longitude"
LAND_SEA = "Land_Sea"
Y_LIGHTNING = "LISOTD_Flash"
Y_HAIL = "ni_HailPF"

LAT_MAX = 63.0
LAND_VALUE = 0

HIST_WITH_SRTM = os.path.join(
    OUT_DIR, "features_historical_1995_2014_withSRTM.parquet"
)
TRAIN_STRICT = os.path.join(
    OUT_DIR, "train_table_flash_hail_withSRTM_strictLand.parquet"
)


# -----------------------------------------------------------------------------
# Elevation
# -----------------------------------------------------------------------------
def build_elevation_interpolator():
    print("[LOAD] SRTM elevation")
    elev = rxr.open_rasterio(ELEV_TIF, masked=True).squeeze()
    elev_lat = elev.y.values
    elev_lon = elev.x.values

    # Preserve the original CanESM5 workflow: reverse the descending latitude
    # axis before RegularGridInterpolator.
    return RegularGridInterpolator(
        (elev_lat[::-1], elev_lon),
        elev.values[::-1, :],
        bounds_error=False,
        fill_value=np.nan,
    )


def add_srtm_elevation(df: pd.DataFrame, interpolator) -> pd.DataFrame:
    for col in ("elev", "elevation"):
        if col in df.columns:
            df = df.drop(columns=col)

    points = np.column_stack(
        [
            pd.to_numeric(df[LAT], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(df[LON], errors="coerce").to_numpy(dtype=float),
        ]
    )
    df = df.copy()
    df["elev"] = interpolator(points).astype("float32")
    return df


def prepare_feature_tables(interpolator):
    print("[HIST] add SRTM elevation")
    df_hist = pd.read_parquet(HIST_FEATURE_FILE)
    df_hist = add_srtm_elevation(df_hist, interpolator)
    df_hist.to_parquet(HIST_WITH_SRTM, index=False)
    print("[SAVED]", HIST_WITH_SRTM)

    for filename in FUTURE_FILES:
        src = os.path.join(BASE_FUTURE, filename)
        if not os.path.exists(src):
            raise FileNotFoundError(src)

        print("[FUT] add SRTM elevation:", filename)
        df_future = pd.read_parquet(src)
        df_future = add_srtm_elevation(df_future, interpolator)

        dst = os.path.join(
            OUT_DIR, filename.replace(".parquet", "_withSRTM.parquet")
        )
        df_future.to_parquet(dst, index=False)
        print("[SAVED]", dst)

    return df_hist


# -----------------------------------------------------------------------------
# Labels and strict-land selection
# -----------------------------------------------------------------------------
def load_required_labels() -> pd.DataFrame:
    usecols = [LAT, LON, LAND_SEA, Y_LIGHTNING, Y_HAIL]
    df = pd.read_excel(LABEL_XLSX, usecols=usecols)

    if df.duplicated([LAT, LON]).any():
        n_dup = int(df.duplicated([LAT, LON], keep=False).sum())
        raise ValueError(f"Label table contains duplicate lat/lon rows: {n_dup}")
    return df


def merge_labels_and_features(df_labels: pd.DataFrame, df_features: pd.DataFrame) -> pd.DataFrame:
    if df_features.duplicated([LAT, LON]).any():
        n_dup = int(df_features.duplicated([LAT, LON], keep=False).sum())
        raise ValueError(f"Feature table contains duplicate lat/lon rows: {n_dup}")

    print("[MERGE] labels + historical features")
    merged = df_labels.merge(
        df_features,
        on=[LAT, LON],
        how="left",
        validate="one_to_one",
    )

    feature_cols = [
        c for c in df_features.columns if c not in {LAT, LON}
    ]
    all_missing = merged[feature_cols].isna().all(axis=1)
    print(f"[CHECK] merged rows: {len(merged)}")
    print(f"[CHECK] rows with all predictors missing: {int(all_missing.sum())}")
    if all_missing.any():
        print("[WARN] Some label rows did not match usable historical predictors.")
    return merged


def apply_strict_land(df: pd.DataFrame) -> pd.DataFrame:
    # Preserve the original pre-filter.
    df0 = df[
        (pd.to_numeric(df[LAT], errors="coerce").abs() <= LAT_MAX)
        & (df[LAND_SEA] == LAND_VALUE)
    ].copy()
    df0 = df0.dropna(subset=[LAT, LON]).reset_index(drop=True)
    print(f"[LAND 1] latitude + Land_Sea: {len(df)} -> {len(df0)}")

    shp = shpreader.natural_earth(
        resolution="110m", category="physical", name="land"
    )
    land = gpd.read_file(shp).to_crs("EPSG:4326")
    land_union = land.geometry.union_all()

    points = [Point(xy) for xy in zip(df0[LON].values, df0[LAT].values)]
    in_land = np.array([land_union.contains(point) for point in points], dtype=bool)
    strict = df0.loc[in_land].reset_index(drop=True)

    print(f"[LAND 2] Natural Earth strict land: {len(df0)} -> {len(strict)}")
    print(f"[REMOVED] polygon-ocean points: {len(df0) - len(strict)}")
    return strict


def main():
    interpolator = build_elevation_interpolator()
    df_hist = prepare_feature_tables(interpolator)

    labels = load_required_labels()
    merged = merge_labels_and_features(labels, df_hist)
    strict = apply_strict_land(merged)

    # The table now contains only coordinates, the two targets, Land_Sea,
    # environmental predictors and SRTM elevation. No auxiliary observational
    # proxy columns are imported, so a separate clean-features step is unnecessary.
    strict.to_parquet(TRAIN_STRICT, index=False)
    print("[SAVED]", TRAIN_STRICT)
    print("\n[DONE] CanESM5 training/prediction feature tables prepared.")


if __name__ == "__main__":
    main()
