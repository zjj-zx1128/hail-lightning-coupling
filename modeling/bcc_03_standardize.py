# -*- coding: utf-8 -*-
"""Standardize selected BCC-CSM2-MR XGBoost outputs.

Run ``bcc_02_train.py`` first. This script selects completed lightning and
hail runs, copies the selected model artifacts into the common ``FINAL_XGB``
layout, rebuilds standard prediction tables, and writes the grid masks used by
downstream prediction and analysis scripts.

Run from the repository root:
    python -m modeling.bcc_03_standardize
"""

from __future__ import annotations

from publication_config import resource_path

import json
import re
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
from sklearn.model_selection import train_test_split


# ============================================================
# 0. CONFIG
# ============================================================

MODEL_NAME = "BCC-CSM2-MR"

# Keep your existing folder spelling.
BASE_DIR = Path(resource_path('models', 'BCC-CSM2-MR'))

TUNING_ROOT = BASE_DIR / "TUNING_XGB"
TRAIN_DIR = BASE_DIR / "feature" / "ml_train_historical"
FINAL_DIR = BASE_DIR / "ml_xgboost_result" / 'FINAL_XGB'

MODEL_DIR = FINAL_DIR / "models"
FEATURE_DIR = FINAL_DIR / "feature_lists"
PRED_DIR = FINAL_DIR / "predictions"
IMP_DIR = FINAL_DIR / "importance"
METRIC_DIR = FINAL_DIR / "metrics"
MASK_NC_DIR = FINAL_DIR / "PRED_SSP585" / "nc"

for p in [FINAL_DIR, MODEL_DIR, FEATURE_DIR, PRED_DIR, IMP_DIR, METRIC_DIR, MASK_NC_DIR]:
    p.mkdir(parents=True, exist_ok=True)

LABEL_STRICT = TRAIN_DIR / "labels_land_strict.parquet"
STANDARD_TRAIN_TABLE = TRAIN_DIR / "train_table_flash_hail_withSRTM_strictLand.parquet"

LAT_COL = "Latitude"
LON_COL = "Longitude"
TARGET_FLASH = "LISOTD_Flash"
TARGET_HAIL = "ni_HailPF"
SEED = 42
TEST_SIZE = 0.15
VALID_SIZE = 0.15
ELEV_COL = "elev"
ELEV_MAX = 2000.0

CLIP_NEGATIVE_PRED = True
SELECT_BEST_BY = "valid_r2"  # fixed: best model selected by validation performance, not test performance


# ============================================================
# 1. Utility functions
# ============================================================

def find_historical_feature_file(train_dir: Path) -> Path:
    """
    Auto-detect historical feature parquet.
    Priority:
      1) features_historical_1995_2014.parquet if present
      2) longest year span among features_historical_*.parquet
      3) latest modified candidate
    """
    preferred = train_dir / "features_historical_1995_2014.parquet"
    if preferred.exists():
        return preferred

    candidates = sorted(train_dir.glob("features_historical_*.parquet"))
    if not candidates:
        raise FileNotFoundError(
            f"No historical feature parquet found in {train_dir}. "
            "Expected files like features_historical_2012_2014.parquet."
        )

    def score(path: Path):
        m = re.search(r"features_historical_(\d{4})_(\d{4})\.parquet$", path.name)
        if m:
            y0, y1 = int(m.group(1)), int(m.group(2))
            span = y1 - y0 + 1
        else:
            span = -1
        return (span, path.stat().st_mtime)

    return max(candidates, key=score)


def finite_metric(value, default=-np.inf) -> float:
    try:
        v = float(value)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def select_best_target_run(target_name: str) -> tuple[Path, dict]:
    """
    Select best run for FLASH or HAIL from TUNING_XGB/{target}_*/metrics.json.
    """
    if not TUNING_ROOT.exists():
        raise FileNotFoundError(f"TUNING_ROOT not found: {TUNING_ROOT}")

    candidates = []
    for d in sorted(TUNING_ROOT.glob(f"{target_name}_*")):
        metrics_path = d / "metrics.json"
        model_path = d / "xgb_model.json"
        feat_path = d / "feature_columns.csv"
        if not (metrics_path.exists() and model_path.exists() and feat_path.exists()):
            continue

        obj = load_json(metrics_path)
        valid = obj.get("valid", {})
        test = obj.get("test", {})
        candidates.append({
            "run_dir": d,
            "metrics": obj,
            "valid_r2": finite_metric(valid.get("r2")),
            "valid_rmse": finite_metric(valid.get("rmse"), default=np.inf),
            "test_r2": finite_metric(test.get("r2")),
            "mtime": metrics_path.stat().st_mtime,
        })

    if not candidates:
        raise FileNotFoundError(
            f"No completed {target_name} run found under {TUNING_ROOT}. "
            f"Expected {target_name}_*/metrics.json + xgb_model.json + feature_columns.csv."
        )

    # Highest valid_r2, then lowest valid_rmse, then newest metrics.json.
    best = sorted(
        candidates,
        key=lambda x: (x["valid_r2"], -x["valid_rmse"], x["mtime"]),
        reverse=True,
    )[0]

    print(f"[BEST {target_name}] {best['run_dir']}")
    print(f"  valid R²   = {best['valid_r2']:.6f}")
    print(f"  valid RMSE = {best['valid_rmse']:.6g}")
    print(f"  test  R²   = {best['test_r2']:.6f}")

    return best["run_dir"], best["metrics"]


def read_feature_columns(path: Path) -> list[str]:
    df = pd.read_csv(path)
    if "feature" in df.columns:
        col = "feature"
    else:
        col = df.columns[0]
    features = [str(v) for v in df[col].dropna().tolist()]
    if not features:
        raise ValueError(f"No features found in {path}")
    return features


def write_feature_txt(features: list[str], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(features))


def write_summary_json(
    target_name: str,
    source_run_dir: Path,
    source_metrics: dict,
    features: list[str],
    out_path: Path,
):
    """
    Convert BCC metrics.json to the summary structure expected by MIROC-style plotting cells.
    """
    summary = {
        "model_name": MODEL_NAME,
        "target_name": target_name,
        "target_col": source_metrics.get("target"),
        "seed": SEED,
        "test_size": TEST_SIZE,
        "val_size": VALID_SIZE,
        "metrics": {
            "best_iteration": int(source_metrics.get("best_iteration", -1)),
            "n_features": int(len(features)),
            "train": source_metrics.get("train", {}),
            "valid": source_metrics.get("valid", {}),
            "test": source_metrics.get("test", {}),
        },
        "feature_policy": {
            "selected_topk_label": source_metrics.get("selected_topk_label"),
            "selected_topk_n": source_metrics.get("selected_topk_n"),
            "n_features": int(len(features)),
            "source_feature_columns_csv": str(source_run_dir / "feature_columns.csv"),
        },
        "final_params": source_metrics.get("final_params", {}),
        "source_metrics_json": str(source_run_dir / "metrics.json"),
        "source_run_dir": str(source_run_dir),
        "paths": {
            "model_path": str(MODEL_DIR / f"{target_name}_model.json"),
            "feature_list_path": str(FEATURE_DIR / f"{target_name}_features.txt"),
            "gain_path": str(IMP_DIR / f"{target_name}_gain.csv"),
        },
        "notes": (
            "Standardized from BCC-CSM2-MR TUNING_XGB output. "
            "Best run was selected by validation R², with validation RMSE as tie breaker. "
            "No test metric was used for model selection."
        ),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def build_standard_train_table(feature_hist: Path, label_strict: Path) -> pd.DataFrame:
    """
    Build merged train table containing coords, observed targets, env_* features and elev.
    This is required by historical diagnostic plotting cells.
    """
    if not feature_hist.exists():
        raise FileNotFoundError(feature_hist)
    if not label_strict.exists():
        raise FileNotFoundError(label_strict)

    df_feat = pd.read_parquet(feature_hist).copy()
    df_lab = pd.read_parquet(label_strict).copy()

    for c in [LAT_COL, LON_COL]:
        if c not in df_feat.columns or c not in df_lab.columns:
            raise KeyError(f"Missing coordinate column {c} in feature or label file.")
        df_feat[c] = df_feat[c].astype(float).round(3)
        df_lab[c] = df_lab[c].astype(float).round(3)

    keep_lab_cols = [c for c in df_lab.columns if c not in df_feat.columns or c in [LAT_COL, LON_COL]]
    for c in [TARGET_FLASH, TARGET_HAIL]:
        if c not in keep_lab_cols and c in df_lab.columns:
            keep_lab_cols.append(c)

    df = df_lab[keep_lab_cols].merge(df_feat, on=[LAT_COL, LON_COL], how="inner")
    df = df.replace([np.inf, -np.inf], np.nan)

    STANDARD_TRAIN_TABLE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(STANDARD_TRAIN_TABLE, index=False)
    print(f"[STANDARD TRAIN TABLE] {STANDARD_TRAIN_TABLE}")
    print(f"  rows={len(df)}, cols={len(df.columns)}")
    return df


def make_split_indices(n: int):
    idx = np.arange(n)
    idx_train_valid, idx_test = train_test_split(idx, test_size=TEST_SIZE, random_state=SEED)
    valid_ratio_inside = VALID_SIZE / (1.0 - TEST_SIZE)
    idx_train, idx_valid = train_test_split(
        idx_train_valid,
        test_size=valid_ratio_inside,
        random_state=SEED,
    )
    return idx_train, idx_valid, idx_test


def predict_with_best_iteration(model: xgb.Booster, dmat: xgb.DMatrix, best_iteration: Optional[int]) -> np.ndarray:
    if best_iteration is not None and best_iteration >= 0:
        try:
            return model.predict(dmat, iteration_range=(0, int(best_iteration) + 1))
        except TypeError:
            # older xgboost fallback
            return model.predict(dmat, ntree_limit=int(best_iteration) + 1)
    return model.predict(dmat)


def rebuild_standard_prediction_csvs(
    target_name: str,
    target_col: str,
    features: list[str],
    source_metrics: dict,
    train_table: pd.DataFrame,
):
    """
    Reconstruct the exact split used in BCC training and save standard CSVs with coords.
    """
    need = [LAT_COL, LON_COL, target_col] + features
    missing = [c for c in need if c not in train_table.columns]
    if missing:
        raise KeyError(f"{target_name}: missing columns in train table: {missing[:20]}")

    df = train_table[need].copy().replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna(subset=[target_col] + features).reset_index(drop=True)
    print(f"[{target_name}] rows before/after finite filter: {before} -> {len(df)}")

    idx_train, idx_valid, idx_test = make_split_indices(len(df))
    model = xgb.Booster()
    model.load_model(str(MODEL_DIR / f"{target_name}_model.json"))

    best_iter = source_metrics.get("best_iteration", None)
    best_iter = int(best_iter) if best_iter is not None else None

    for split_name, idx in [
        ("train", idx_train),
        ("valid", idx_valid),
        ("test", idx_test),
    ]:
        X = df.loc[idx, features]
        y = df.loc[idx, target_col].to_numpy(dtype=float)
        dmat = xgb.DMatrix(X, feature_names=features)
        pred_raw = predict_with_best_iteration(model, dmat, best_iter)
        pred = np.clip(pred_raw, 0, None) if CLIP_NEGATIVE_PRED else pred_raw

        out = pd.DataFrame({
            LAT_COL: df.loc[idx, LAT_COL].to_numpy(dtype=float),
            LON_COL: df.loc[idx, LON_COL].to_numpy(dtype=float),
            "y_true": y,
            "y_pred": pred.astype(float),
            "y_pred_raw": pred_raw.astype(float),
        })

        out_path = PRED_DIR / f"{target_name}_pred_{split_name}.csv"
        out.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[PRED CSV] {out_path}")


def dataframe_to_grid(df: pd.DataFrame, values: np.ndarray, lat_axis: np.ndarray, lon_axis: np.ndarray) -> np.ndarray:
    grid = np.full((len(lat_axis), len(lon_axis)), np.nan, dtype=np.float32)
    lat_to_i = {float(v): i for i, v in enumerate(lat_axis)}
    lon_to_j = {float(v): j for j, v in enumerate(lon_axis)}

    for la, lo, val in zip(df[LAT_COL].astype(float), df[LON_COL].astype(float), values):
        la = round(float(la), 3)
        lo = round(float(lo), 3)
        if la in lat_to_i and lo in lon_to_j:
            grid[lat_to_i[la], lon_to_j[lo]] = val

    return grid


def build_mask_nc(train_table: pd.DataFrame):
    lat_axis = np.array(sorted(train_table[LAT_COL].dropna().astype(float).round(3).unique()), dtype=np.float32)
    lon_axis = np.array(sorted(train_table[LON_COL].dropna().astype(float).round(3).unique()), dtype=np.float32)

    label_df = pd.read_parquet(LABEL_STRICT).copy()
    label_df[LAT_COL] = label_df[LAT_COL].astype(float).round(3)
    label_df[LON_COL] = label_df[LON_COL].astype(float).round(3)

    strict_land = np.zeros((len(lat_axis), len(lon_axis)), dtype=bool)
    lat_to_i = {float(v): i for i, v in enumerate(lat_axis)}
    lon_to_j = {float(v): j for j, v in enumerate(lon_axis)}
    for la, lo in zip(label_df[LAT_COL], label_df[LON_COL]):
        if float(la) in lat_to_i and float(lo) in lon_to_j:
            strict_land[lat_to_i[float(la)], lon_to_j[float(lo)]] = True

    if ELEV_COL in train_table.columns:
        elev_grid = dataframe_to_grid(
            train_table[[LAT_COL, LON_COL]].copy(),
            pd.to_numeric(train_table[ELEV_COL], errors="coerce").to_numpy(dtype=np.float32),
            lat_axis,
            lon_axis,
        )
    else:
        elev_grid = np.full_like(strict_land, np.nan, dtype=np.float32)

    high_alt = np.isfinite(elev_grid) & (elev_grid > ELEV_MAX)

    ds = xr.Dataset(
        data_vars={
            "strict_land": (("lat", "lon"), strict_land.astype(np.int8)),
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
            "strict_land_source": str(LABEL_STRICT),
            "high_alt_threshold_m": float(ELEV_MAX),
            "note": "Generated by BCC standardization cell from training grid.",
        },
    )
    out_nc = MASK_NC_DIR / "masks_2deg.nc"
    ds.to_netcdf(out_nc)
    print(f"[MASK NC] {out_nc}")


def standardize_one_target(target_name: str, target_col: str, train_table: pd.DataFrame):
    run_dir, metrics = select_best_target_run(target_name)

    # Copy model.
    shutil.copy2(run_dir / "xgb_model.json", MODEL_DIR / f"{target_name}_model.json")

    # Feature list.
    features = read_feature_columns(run_dir / "feature_columns.csv")
    write_feature_txt(features, FEATURE_DIR / f"{target_name}_features.txt")

    # Importance.
    src_imp = run_dir / "feature_importance_gain.csv"
    if src_imp.exists():
        imp = pd.read_csv(src_imp)
        # Standard MIROC plotting cell expects columns: feature, gain.
        if "gain" not in imp.columns and len(imp.columns) >= 2:
            imp = imp.rename(columns={imp.columns[1]: "gain"})
        if "feature" not in imp.columns:
            imp = imp.rename(columns={imp.columns[0]: "feature"})
        imp[["feature", "gain"]].to_csv(IMP_DIR / f"{target_name}_gain.csv", index=False, encoding="utf-8-sig")
    else:
        print(f"[WARNING] Missing importance file: {src_imp}")

    # Summary.
    write_summary_json(
        target_name=target_name,
        source_run_dir=run_dir,
        source_metrics=metrics,
        features=features,
        out_path=METRIC_DIR / f"{target_name}_summary.json",
    )

    # Prediction CSVs.
    rebuild_standard_prediction_csvs(target_name, target_col, features, metrics, train_table)

    return {
        "target_name": target_name,
        "target_col": target_col,
        "source_run_dir": str(run_dir),
        "n_features": len(features),
        "best_iteration": int(metrics.get("best_iteration", -1)),
        "train_r2": metrics.get("train", {}).get("r2"),
        "valid_r2": metrics.get("valid", {}).get("r2"),
        "test_r2": metrics.get("test", {}).get("r2"),
        "valid_rmse": metrics.get("valid", {}).get("rmse"),
        "test_rmse": metrics.get("test", {}).get("rmse"),
    }


# ============================================================
# 2. MAIN
# ============================================================

if __name__ == "__main__":
    import re  # used by find_historical_feature_file

    print("=" * 100)
    print(f"[STANDARDIZE] {MODEL_NAME} best XGB runs for MIROC-style prediction/plotting")
    print("=" * 100)

    feature_hist = find_historical_feature_file(TRAIN_DIR)
    print(f"[HIST FEATURE] {feature_hist}")

    train_table = build_standard_train_table(feature_hist, LABEL_STRICT)
    build_mask_nc(train_table)

    rows = []
    rows.append(standardize_one_target("FLASH", TARGET_FLASH, train_table))
    rows.append(standardize_one_target("HAIL", TARGET_HAIL, train_table))

    metrics_csv = METRIC_DIR / "metrics_xgb_final_seed42.csv"
    pd.DataFrame(rows).to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    print(f"[COMBINED METRICS] {metrics_csv}")

    print("\n[FINAL STANDARD DIR]")
    print(FINAL_DIR)
    print("\n[DONE] Now you can run the prediction and plotting cells below.")
