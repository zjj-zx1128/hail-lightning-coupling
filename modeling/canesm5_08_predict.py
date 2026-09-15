# Purpose: canesm5 08 predict.
# Source: cmip6_figure_CanESM5_XGB_plot_latest_checked.ipynb; see docs/source_coverage.csv for cell provenance.
# Run order and required inputs: docs/workflow.md.
# Run from repository root: python -m modeling.canesm5_08_predict


# -*- coding: utf-8 -*-
'\nStep B. Predict CanESM5 SSP585 decades using standardized best XGBoost models.\n\nInput expected:\n    [configured path; see config.json]\n    ...\n    [configured path; see config.json]\n\nOutput:\n    [configured path; see config.json]\nc\\pred_ssp585_2050_2059_2deg.nc\n    ...\n    masks_2deg.nc\n\nIf SSP585 feature files are not yet available, this cell will save/update masks_2deg.nc and skip future prediction.1\n'

from __future__ import annotations

from publication_config import resource_path

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb


# ============================================================
# 0. Paths and config
# ============================================================

MODEL_NAME = "CanESM5"
BASE_DIR = Path(resource_path('models', 'CanESM5'))

SSP585_FEATURE_DIR = BASE_DIR / "ml_feature_withSRTM"
TRAIN_DIR = BASE_DIR / "ml_feature_withSRTM"
# CanESM5 uses the strict-land training table as the source of historical grid points.
LABEL_FILE = TRAIN_DIR / "train_table_flash_hail_withSRTM_strictLand.parquet"
TRAIN_TABLE = LABEL_FILE

RUN_DIR = BASE_DIR / "ml_xgboost_result" / 'FINAL_XGB'
MODEL_DIR = RUN_DIR / "models"
FEATURE_DIR = RUN_DIR / "feature_lists"
METRIC_DIR = RUN_DIR / "metrics"

FLASH_MODEL = MODEL_DIR / "FLASH_model.json"
HAIL_MODEL = MODEL_DIR / "HAIL_model.json"
FLASH_FEATURE_TXT = FEATURE_DIR / "FLASH_features.txt"
HAIL_FEATURE_TXT = FEATURE_DIR / "HAIL_features.txt"
FLASH_SUMMARY_JSON = METRIC_DIR / "FLASH_summary.json"
HAIL_SUMMARY_JSON = METRIC_DIR / "HAIL_summary.json"

OUT_DIR = RUN_DIR / "PRED_SSP585"
OUT_NC_DIR = OUT_DIR / "nc"
OUT_NC_DIR.mkdir(parents=True, exist_ok=True)

OUT_SUMMARY_CSV = OUT_DIR / "prediction_summary_ssp585.csv"
OUT_META_JSON = OUT_DIR / "prediction_run_meta.json"

LAT_COL = "Latitude"
LON_COL = "Longitude"
ELEV_COL = "elev"
ELEV_MAX = 2000.0

DECADES = [
    (2050, 2059),
    (2060, 2069),
    (2070, 2079),
    (2080, 2089),
    (2090, 2099),
]

CLIP_NEGATIVE = True
USE_BEST_ITERATION = True


# ============================================================
# 1. Utility functions
# ============================================================

def wrap_lon180(lon):
    lon = np.asarray(lon, dtype=float)
    return ((lon + 180.0) % 360.0) - 180.0


def read_feature_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        features = [line.strip() for line in f if line.strip()]
    if not features:
        raise ValueError(f"No feature found in {path}")
    return features


def read_best_iteration(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    try:
        return int(obj["metrics"]["best_iteration"])
    except Exception:
        return None


def load_booster(path: Path) -> xgb.Booster:
    if not path.exists():
        raise FileNotFoundError(path)
    bst = xgb.Booster()
    bst.load_model(str(path))
    return bst


def make_axes_from_df(df: pd.DataFrame):
    lat = np.array(sorted(df[LAT_COL].dropna().astype(float).round(3).unique()), dtype=np.float32)
    lon = np.array(sorted(wrap_lon180(df[LON_COL].dropna().astype(float).to_numpy()).round(3)), dtype=np.float32)
    lon = np.array(sorted(np.unique(lon)), dtype=np.float32)
    return lat, lon


def dataframe_to_grid(df: pd.DataFrame, values: np.ndarray, lat_axis: np.ndarray, lon_axis: np.ndarray) -> np.ndarray:
    grid = np.full((len(lat_axis), len(lon_axis)), np.nan, dtype=np.float32)
    lat_to_i = {float(v): i for i, v in enumerate(lat_axis)}
    lon_to_j = {float(v): j for j, v in enumerate(lon_axis)}
    lat_values = df[LAT_COL].astype(float).round(3).to_numpy()
    lon_values = np.round(wrap_lon180(df[LON_COL].astype(float).to_numpy()), 3)

    for la, lo, val in zip(lat_values, lon_values, values):
        if float(la) in lat_to_i and float(lo) in lon_to_j:
            grid[lat_to_i[float(la)], lon_to_j[float(lo)]] = val

    return grid


def find_ssp585_feature_file(y0: int, y1: int) -> Path:
    """
    Return the SSP585 feature parquet for a given decade.

    CanESM5 may store future feature files using either of the following names:
        features_ssp585_2050_2059_withSRTM.parquet
        features_ssp585_2050_2059.parquet

    The withSRTM version is preferred because it is consistent with the
    standardized feature set used by the selected XGBoost models.
    """
    candidates = [
        SSP585_FEATURE_DIR / f"features_ssp585_{y0}_{y1}_withSRTM.parquet",
        SSP585_FEATURE_DIR / f"features_ssp585_{y0}_{y1}.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def build_strict_land_mask(label_file: Path, lat_axis: np.ndarray, lon_axis: np.ndarray) -> np.ndarray:
    if not label_file.exists():
        raise FileNotFoundError(label_file)
    df_lab = pd.read_parquet(label_file)
    mask = np.zeros((len(lat_axis), len(lon_axis)), dtype=bool)
    lat_to_i = {float(v): i for i, v in enumerate(lat_axis)}
    lon_to_j = {float(v): j for j, v in enumerate(lon_axis)}
    for la, lo in zip(df_lab[LAT_COL].astype(float).round(3), np.round(wrap_lon180(df_lab[LON_COL].astype(float)), 3)):
        if float(la) in lat_to_i and float(lo) in lon_to_j:
            mask[lat_to_i[float(la)], lon_to_j[float(lo)]] = True
    return mask


def predict_to_grid(
    df: pd.DataFrame,
    features: list[str],
    booster: xgb.Booster,
    best_iteration,
    lat_axis: np.ndarray,
    lon_axis: np.ndarray,
    strict_land_mask: np.ndarray,
) -> tuple[np.ndarray, dict]:
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise KeyError(f"Missing selected features in prediction dataframe: {missing[:20]}")

    # XGBoost can handle NaN values through learned default split directions.
    # Therefore, do NOT require every selected predictor to be finite.
    # This is important for high-terrain grids, where some pressure-level or
    # interpolation-derived environmental predictors can be NaN although the
    # grid cell is still a valid strict-land point.
    X = df[features].copy().replace([np.inf, -np.inf], np.nan)

    rows_with_any_missing = X.isna().any(axis=1).to_numpy()
    rows_with_all_missing = X.isna().all(axis=1).to_numpy()

    # Avoid predicting only when all selected predictors are missing.
    # In normal feature tables, elev should be finite, so this usually keeps
    # almost all strict-land rows while still protecting against invalid rows.
    valid = ~rows_with_all_missing

    pred = np.full(len(df), np.nan, dtype=np.float32)
    if valid.sum() > 0:
        dmat = xgb.DMatrix(
            X.loc[valid, features],
            feature_names=features,
            missing=np.nan,
        )
        if USE_BEST_ITERATION and best_iteration is not None and int(best_iteration) >= 0:
            try:
                pred_valid = booster.predict(dmat, iteration_range=(0, int(best_iteration) + 1))
            except TypeError:
                pred_valid = booster.predict(dmat, ntree_limit=int(best_iteration) + 1)
        else:
            pred_valid = booster.predict(dmat)

        if CLIP_NEGATIVE:
            pred_valid = np.clip(pred_valid, 0, None)
        pred[valid] = pred_valid.astype(np.float32)

    grid = dataframe_to_grid(df, pred, lat_axis, lon_axis)
    grid = np.where(strict_land_mask, grid, np.nan).astype(np.float32)

    info = {
        "rows": int(len(df)),
        "valid_feature_rows": int(valid.sum()),
        "rows_with_any_missing_feature": int(rows_with_any_missing.sum()),
        "rows_with_all_missing_features": int(rows_with_all_missing.sum()),
        "finite_grid_cells": int(np.isfinite(grid).sum()),
        "mean": float(np.nanmean(grid)) if np.isfinite(grid).any() else np.nan,
        "max": float(np.nanmax(grid)) if np.isfinite(grid).any() else np.nan,
    }
    return grid, info


def save_masks(source_df: pd.DataFrame, lat_axis: np.ndarray, lon_axis: np.ndarray):
    strict_land_mask = build_strict_land_mask(LABEL_FILE, lat_axis, lon_axis)

    if ELEV_COL in source_df.columns:
        elev = pd.to_numeric(source_df[ELEV_COL], errors="coerce").to_numpy(dtype=np.float32)
        elev_grid = dataframe_to_grid(source_df, elev, lat_axis, lon_axis)
    else:
        elev_grid = np.full((len(lat_axis), len(lon_axis)), np.nan, dtype=np.float32)

    high_alt = np.isfinite(elev_grid) & (elev_grid > ELEV_MAX)

    mask_ds = xr.Dataset(
        data_vars={
            "strict_land": (("lat", "lon"), strict_land_mask.astype(np.int8)),
            "elevation_m": (("lat", "lon"), elev_grid.astype(np.float32)),
            "high_alt": (("lat", "lon"), high_alt.astype(np.int8)),
        },
        coords={
            "lat": lat_axis.astype(np.float32),
            "lon": lon_axis.astype(np.float32),
        },
        attrs={
            "model": MODEL_NAME,
            "grid": "2deg",
            "strict_land_source": str(LABEL_FILE),
            "high_alt_threshold_m": float(ELEV_MAX),
            "note": (
                "strict_land was reconstructed from labels_land_strict.parquet. "
                "Predicted values outside strict_land are set to NaN. "
                "High-altitude cells are not removed here; they are only saved as a mask."
            ),
        },
    )
    mask_path = OUT_NC_DIR / "masks_2deg.nc"
    mask_ds.to_netcdf(mask_path)
    print(f"[SAVED MASK] {mask_path}")
    return strict_land_mask


# ============================================================
# 2. Main prediction
# ============================================================

print("=" * 100)
print(f"[PREDICT SSP585] {MODEL_NAME}")
print("=" * 100)

features_flash = read_feature_list(FLASH_FEATURE_TXT)
features_hail = read_feature_list(HAIL_FEATURE_TXT)
bst_flash = load_booster(FLASH_MODEL)
bst_hail = load_booster(HAIL_MODEL)
best_iter_flash = read_best_iteration(FLASH_SUMMARY_JSON)
best_iter_hail = read_best_iteration(HAIL_SUMMARY_JSON)

print("[FEATURES]")
print("FLASH:", len(features_flash), "best_iteration:", best_iter_flash)
print("HAIL :", len(features_hail), "best_iteration:", best_iter_hail)

future_files = [find_ssp585_feature_file(y0, y1) for y0, y1 in DECADES]
existing_future_files = [p for p in future_files if p.exists()]

# Use future grid if available; otherwise use standard historical train table for masks only.
if existing_future_files:
    df_grid = pd.read_parquet(existing_future_files[0])
    print(f"[GRID SOURCE] first SSP585 file: {existing_future_files[0]}")
else:
    if not TRAIN_TABLE.exists():
        raise FileNotFoundError(
            f"No SSP585 feature files found and standard train table missing: {TRAIN_TABLE}. "
            "Run the standardization cell first."
        )
    df_grid = pd.read_parquet(TRAIN_TABLE)
    print("[WARNING] No SSP585 feature parquet found. Future prediction will be skipped.")
    print(f"[GRID SOURCE] historical train table for mask only: {TRAIN_TABLE}")

df_grid[LON_COL] = np.round(wrap_lon180(df_grid[LON_COL].astype(float)), 3)
df_grid[LAT_COL] = df_grid[LAT_COL].astype(float).round(3)
lat_axis, lon_axis = make_axes_from_df(df_grid)
strict_land_mask = save_masks(df_grid, lat_axis, lon_axis)

summary_rows = []

for y0, y1 in DECADES:
    feature_file = find_ssp585_feature_file(y0, y1)
    if not feature_file.exists():
        print(f"[SKIP] Missing SSP585 feature file: {feature_file}")
        continue

    print("\n" + "-" * 100)
    print(f"[PREDICT] SSP585 {y0}-{y1}")
    print("-" * 100)

    df = pd.read_parquet(feature_file)
    df[LON_COL] = np.round(wrap_lon180(df[LON_COL].astype(float)), 3)
    df[LAT_COL] = df[LAT_COL].astype(float).round(3)

    lat_this, lon_this = make_axes_from_df(df)
    if not np.array_equal(lat_this, lat_axis):
        raise ValueError(f"Latitude axis mismatch in {feature_file.name}")
    if not np.array_equal(lon_this, lon_axis):
        raise ValueError(f"Longitude axis mismatch in {feature_file.name}")

    grid_flash, info_flash = predict_to_grid(
        df=df,
        features=features_flash,
        booster=bst_flash,
        best_iteration=best_iter_flash,
        lat_axis=lat_axis,
        lon_axis=lon_axis,
        strict_land_mask=strict_land_mask,
    )
    grid_hail, info_hail = predict_to_grid(
        df=df,
        features=features_hail,
        booster=bst_hail,
        best_iteration=best_iter_hail,
        lat_axis=lat_axis,
        lon_axis=lon_axis,
        strict_land_mask=strict_land_mask,
    )

    ds = xr.Dataset(
        data_vars={
            "pred_flash": (("lat", "lon"), grid_flash.astype(np.float32)),
            "pred_hail": (("lat", "lon"), grid_hail.astype(np.float32)),
        },
        coords={
            "lat": lat_axis.astype(np.float32),
            "lon": lon_axis.astype(np.float32),
        },
        attrs={
            "model": MODEL_NAME,
            "scenario": "ssp585",
            "decade": f"{y0}-{y1}",
            "grid": "2deg",
            "strict_land_mask_applied": "true",
            "negative_predictions_clipped_to_zero": str(CLIP_NEGATIVE).lower(),
            "high_altitude_removed": "false",
            "high_altitude_mask_file": "masks_2deg.nc",
            "flash_model": str(FLASH_MODEL),
            "hail_model": str(HAIL_MODEL),
            "flash_features": str(FLASH_FEATURE_TXT),
            "hail_features": str(HAIL_FEATURE_TXT),
            "flash_best_iteration": int(best_iter_flash) if best_iter_flash is not None else -1,
            "hail_best_iteration": int(best_iter_hail) if best_iter_hail is not None else -1,
        },
    )

    out_nc = OUT_NC_DIR / f"pred_ssp585_{y0}_{y1}_2deg.nc"
    ds.to_netcdf(out_nc)
    print(f"[SAVED] {out_nc}")

    summary_rows.append({
        "model": MODEL_NAME,
        "scenario": "ssp585",
        "start_year": y0,
        "end_year": y1,
        "feature_file": str(feature_file),
        "out_nc": str(out_nc),
        "flash_finite_grid_cells": info_flash["finite_grid_cells"],
        "hail_finite_grid_cells": info_hail["finite_grid_cells"],
        "flash_mean": info_flash["mean"],
        "hail_mean": info_hail["mean"],
        "flash_max": info_flash["max"],
        "hail_max": info_hail["max"],
    })

pd.DataFrame(summary_rows).to_csv(OUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")
with open(OUT_META_JSON, "w", encoding="utf-8") as f:
    json.dump({
        "model": MODEL_NAME,
        "base_dir": str(BASE_DIR),
        "feature_dir": str(SSP585_FEATURE_DIR),
        "n_decades_predicted": len(summary_rows),
        "decades_requested": DECADES,
        "missing_files": [str(p) for p in future_files if not p.exists()],
    }, f, ensure_ascii=False, indent=2)

print("\n[DONE]")
print(f"[SUMMARY] {OUT_SUMMARY_CSV}")
print(f"[META] {OUT_META_JSON}")
if len(summary_rows) == 0:
    print("[NOTE] No future NetCDF was created because SSP585 feature parquet files were not found.")
