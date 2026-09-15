# -*- coding: utf-8 -*-
"""
Build CanESM5 historical and SSP585 environmental feature tables.

This script performs environmental feature engineering only. It does not add
SRTM elevation, merge observational targets, filter strict-land samples, train
models, or generate predictions.

Candidate environmental predictors:
- 14 environmental parameters
- 15 full-period statistics per parameter
- seasonal mean, q90 and max for DJF/MAM/JJA/SON

The resulting environmental pool contains 378 columns. SRTM elevation is added
in ``canesm5_01_prepare_training_data.py`` to form the 379-feature candidate
pool used by the CanESM5 modeling workflow.

Run from the repository root:
    python -m modeling.canesm5_00_build_features
"""

from __future__ import annotations

import glob
import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
from tqdm import tqdm

from publication_config import resource_path

xr.set_options(file_cache_maxsize=1)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
LABEL_XLSX = resource_path("observations", "flashhail_matched_new2.xlsx")
ENV_DIR = resource_path("models", "CanESM5/envdiag")

OUT_TRAIN_DIR = resource_path("models", "CanESM5/feature/ml_train_historical")
OUT_FUTURE_DIR = resource_path("models", "CanESM5/feature/ml_pred_ssp585")
os.makedirs(OUT_TRAIN_DIR, exist_ok=True)
os.makedirs(OUT_FUTURE_DIR, exist_ok=True)

ROI_N, ROI_W, ROI_S, ROI_E = 63, -180, -63, 180
LAT_COL = "Latitude"
LON_COL = "Longitude"

VARS = [
    "mucape", "mucin", "pw", "s06", "h_10c", "h_30c", "dh_10_30",
    "div500", "td_sfc", "theta_e", "cape", "cin", "flh", "k_index",
]

HIST_YEARS = (1995, 2014)
FUTURE_START = 2050
FUTURE_END_EXCLUSIVE = 2100
DECADE_STEP = 10

QUANTILE_LEVELS = {
    "q05": 0.05,
    "q10": 0.10,
    "q25": 0.25,
    "q50": 0.50,
    "q75": 0.75,
    "q90": 0.90,
    "q95": 0.95,
}
STATISTIC_SUFFIXES = (
    "mean", "std", "min", "max",
    "q05", "q10", "q25", "q50", "q75", "q90", "q95",
    "range", "iqr", "q95_q50", "q90_q10",
)
SEASONS = ("DJF", "MAM", "JJA", "SON")
SEASONAL_SUFFIXES = ("mean", "q90", "max")
FEATURE_PREFIX = "env"

warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="invalid value encountered in divide"
)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="All-NaN slice encountered"
)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def lon_to_180(lon):
    lon = np.asarray(lon, dtype=float)
    return ((lon + 180) % 360) - 180


def build_2deg_axis_from_full(vals: np.ndarray) -> np.ndarray:
    """Construct a 2-degree axis from the full reference-grid coordinates."""
    vals = np.round(np.asarray(vals, dtype=float), 3)
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))

    start = np.round(np.floor(vmin / 2.0) * 2.0, 3)
    end = np.round(np.ceil(vmax / 2.0) * 2.0, 3)
    axis = np.round(np.arange(start, end + 0.001, 2.0), 3)
    return axis[(axis >= vmin - 1e-6) & (axis <= vmax + 1e-6)]


def load_reference_grid() -> tuple[np.ndarray, np.ndarray]:
    """Use the observational table only to define the common 2-degree grid."""
    df = pd.read_excel(LABEL_XLSX, usecols=[LAT_COL, LON_COL])
    df[LON_COL] = lon_to_180(df[LON_COL].to_numpy())
    df = df[
        (df[LAT_COL] >= ROI_S)
        & (df[LAT_COL] <= ROI_N)
        & (df[LON_COL] >= ROI_W)
        & (df[LON_COL] <= ROI_E)
    ].copy()

    lat2 = build_2deg_axis_from_full(df[LAT_COL].to_numpy())
    lon2 = build_2deg_axis_from_full(df[LON_COL].to_numpy())
    print(f"[GRID] lat2: {lat2.min()}..{lat2.max()} n={len(lat2)}")
    print(f"[GRID] lon2: {lon2.min()}..{lon2.max()} n={len(lon2)}")
    return lat2, lon2


def decade_ranges(start=2050, end_exclusive=2100, step=10):
    out = []
    year = start
    while year + step <= end_exclusive:
        out.append((year, year + step - 1))
        year += step
    return out


def season_mask(time: xr.DataArray, season: str) -> xr.DataArray:
    month = time.dt.month
    if season == "DJF":
        return (month == 12) | (month == 1) | (month == 2)
    if season == "MAM":
        return (month >= 3) & (month <= 5)
    if season == "JJA":
        return (month >= 6) & (month <= 8)
    if season == "SON":
        return (month >= 9) & (month <= 11)
    raise ValueError(f"Unknown season: {season}")


# -----------------------------------------------------------------------------
# Environmental data
# -----------------------------------------------------------------------------
def open_envdiag(kind: str, year0: int, year1: int) -> xr.Dataset:
    if kind == "historical":
        pattern = os.path.join(ENV_DIR, "CanESM5_historical_envdiag_*.nc")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No files found: {pattern}")
    elif kind == "ssp585":
        files = [
            os.path.join(ENV_DIR, f"CanESM5_ssp585_envdiag_{year}.nc")
            for year in range(year0, year1 + 1)
            if os.path.exists(os.path.join(ENV_DIR, f"CanESM5_ssp585_envdiag_{year}.nc"))
        ]
        if not files:
            raise FileNotFoundError(f"No SSP585 files found for {year0}-{year1}")
    else:
        raise ValueError(kind)

    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        parallel=False,
        engine="netcdf4",
        coords="minimal",
        data_vars="minimal",
        compat="override",
        chunks={},
    ).chunk({"time": 240})

    missing = [v for v in VARS if v not in ds.data_vars]
    if missing:
        raise ValueError(f"{kind} envdiag missing variables: {missing}")
    ds = ds[VARS]

    if float(ds["lon"].max()) > 180:
        ds = ds.assign_coords(lon=lon_to_180(ds["lon"])).sortby("lon")

    ds = ds.sel(time=slice(f"{year0}-01-01", f"{year1}-12-31"))

    latmin, latmax = float(min(ROI_S, ROI_N)), float(max(ROI_S, ROI_N))
    if ds["lat"][0] < ds["lat"][-1]:
        ds = ds.sel(lat=slice(latmin, latmax))
    else:
        ds = ds.sel(lat=slice(latmax, latmin))

    # Preserve the original workflow: use absolute CIN/MUCIN magnitudes.
    return ds.assign(cin=abs(ds["cin"]), mucin=abs(ds["mucin"]))


def regrid_env_to_2deg(ds: xr.Dataset, lat2, lon2) -> xr.Dataset:
    return ds.interp(
        lat=xr.DataArray(lat2, dims="lat"),
        lon=xr.DataArray(lon2, dims="lon"),
        method="linear",
    )


# -----------------------------------------------------------------------------
# Feature engineering
# -----------------------------------------------------------------------------
def _time_quantile(da: xr.DataArray, q: float) -> xr.DataArray:
    result = da.quantile(q, dim="time", skipna=True)
    if "quantile" in result.coords:
        result = result.reset_coords("quantile", drop=True)
    return result


def add_full_period_statistics(feats: xr.Dataset, da: xr.DataArray, prefix: str):
    mean = da.mean("time", skipna=True)
    std = da.std("time", skipna=True)
    minimum = da.min("time", skipna=True)
    maximum = da.max("time", skipna=True)
    quantiles = {
        name: _time_quantile(da, q) for name, q in QUANTILE_LEVELS.items()
    }

    feats[f"{prefix}_mean"] = mean
    feats[f"{prefix}_std"] = std
    feats[f"{prefix}_min"] = minimum
    feats[f"{prefix}_max"] = maximum
    for name, values in quantiles.items():
        feats[f"{prefix}_{name}"] = values

    feats[f"{prefix}_range"] = maximum - minimum
    feats[f"{prefix}_iqr"] = quantiles["q75"] - quantiles["q25"]
    feats[f"{prefix}_q95_q50"] = quantiles["q95"] - quantiles["q50"]
    feats[f"{prefix}_q90_q10"] = quantiles["q90"] - quantiles["q10"]


def add_seasonal_statistics(feats: xr.Dataset, da: xr.DataArray, prefix: str):
    for season in SEASONS:
        seasonal = da.where(season_mask(da["time"], season), drop=True)
        feats[f"{prefix}_{season}_mean"] = seasonal.mean("time", skipna=True)
        feats[f"{prefix}_{season}_q90"] = _time_quantile(seasonal, 0.90)
        feats[f"{prefix}_{season}_max"] = seasonal.max("time", skipna=True)


def build_features(ds2: xr.Dataset) -> xr.Dataset:
    feats = xr.Dataset()
    for var in tqdm(list(ds2.data_vars), desc="Build features", unit="var"):
        da = ds2[var]
        prefix = f"{FEATURE_PREFIX}_{var}"
        add_full_period_statistics(feats, da, prefix)
        add_seasonal_statistics(feats, da, prefix)

    expected = len(VARS) * (
        len(STATISTIC_SUFFIXES) + len(SEASONS) * len(SEASONAL_SUFFIXES)
    )
    if len(feats.data_vars) != expected:
        raise RuntimeError(
            f"Feature schema mismatch: found {len(feats.data_vars)}, expected {expected}."
        )
    return feats


def compute_features_to_parquet(
    kind: str,
    year0: int,
    year1: int,
    lat2,
    lon2,
    out_dir: str,
) -> pd.DataFrame:
    tag = f"{kind}_{year0}_{year1}"
    feat_path = os.path.join(out_dir, f"features_{tag}.parquet")
    if os.path.exists(feat_path):
        print(f"[RESUME] Found features: {os.path.basename(feat_path)}")
        return pd.read_parquet(feat_path)

    ds = open_envdiag(kind, year0, year1)
    ds2 = regrid_env_to_2deg(ds, lat2, lon2)
    feat = build_features(ds2)

    print(f"[DASK] computing features: {tag} ...")
    import dask

    with ProgressBar():
        with dask.config.set(scheduler="single-threaded"):
            feat = feat.compute()

    df = feat.to_dataframe().reset_index().rename(
        columns={"lat": LAT_COL, "lon": LON_COL}
    )
    df[LAT_COL] = df[LAT_COL].astype(float).round(3)
    df[LON_COL] = df[LON_COL].astype(float).round(3)

    feature_columns = [c for c in df.columns if c.startswith(f"{FEATURE_PREFIX}_")]
    expected = len(VARS) * (
        len(STATISTIC_SUFFIXES) + len(SEASONS) * len(SEASONAL_SUFFIXES)
    )
    if len(feature_columns) != expected or "quantile" in df.columns:
        raise RuntimeError(
            f"Parquet feature schema mismatch: found {len(feature_columns)} "
            f"environmental features, expected {expected}."
        )

    df.to_parquet(feat_path, index=False)
    print(f"[SAVED] {feat_path}")
    return df


if __name__ == "__main__":
    lat2, lon2 = load_reference_grid()

    y0, y1 = HIST_YEARS
    compute_features_to_parquet(
        "historical", y0, y1, lat2, lon2, OUT_TRAIN_DIR
    )

    decades = decade_ranges(FUTURE_START, FUTURE_END_EXCLUSIVE, DECADE_STEP)
    print("[FUTURE] Decades:", decades)
    for start, end in decades:
        compute_features_to_parquet(
            "ssp585", start, end, lat2, lon2, OUT_FUTURE_DIR
        )

    print("\n[DONE] CanESM5 environmental feature tables generated.")
    print("[OUT] historical:", OUT_TRAIN_DIR)
    print("[OUT] SSP585    :", OUT_FUTURE_DIR)
