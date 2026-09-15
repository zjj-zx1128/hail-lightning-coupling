# Run from repository root: python -m modeling.miroc6_03_predict

# ============================================================
# Step 3. Predict MIROC6 SSP585 decades using saved XGBoost models
#
# 功能：
#   1. 读取 Step 2 保存的 FLASH/HAIL 模型与 feature list
#   2. 读取 SSP585 五个年代的 feature parquet
#   3. 预测 pred_flash / pred_hail
#   4. 转为 2° lat-lon 网格
#   5. 使用 labels_land_strict 的经纬度生成 strict-land mask
#   6. 非 strict-land 区域设为 NaN
#   7. 负值裁剪为 0
#   8. 输出 NetCDF 文件
#
# 注意：
#   - 不剔除 elev > 2000 m
#   - elev > 2000 m 只保存为 high_alt mask，供后续制图/区域统计使用
#   - 预测时使用 Step 2 summary.json 中记录的 best_iteration
# ============================================================

from __future__ import annotations

from publication_config import resource_path

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb


# ============================================================
# 0. Path configuration
# ============================================================

MODEL_NAME = "MIROC6"

BASE_DIR = Path(resource_path('models', 'MIROC6'))

SSP585_FEATURE_DIR = BASE_DIR / "feature" / "ml_pred_ssp585"
TRAIN_DIR = BASE_DIR / "feature" / "ml_train_historical"

LABEL_FILE = TRAIN_DIR / "labels_land_strict.parquet"

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

# True：使用 Step 2 中保存的 best_iteration 进行预测，更符合模型评估阶段
# False：完全按照 CanESM5 旧预测代码，直接 bst.predict(dmx)
USE_BEST_ITERATION = True


# ============================================================
# 1. Utility functions
# ============================================================

def wrap_lon180(lon):
    lon = np.asarray(lon, dtype=float)
    return ((lon + 180.0) % 360.0) - 180.0


def read_feature_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Feature list not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        features = [line.strip() for line in f if line.strip()]

    if len(features) == 0:
        raise ValueError(f"Empty feature list: {path}")

    return features


def load_booster(path: Path) -> xgb.Booster:
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    bst = xgb.Booster()
    bst.load_model(str(path))
    return bst


def read_best_iteration(summary_json: Path):
    if not summary_json.exists():
        print(f"[WARNING] Summary JSON not found: {summary_json}")
        return None

    with open(summary_json, "r", encoding="utf-8") as f:
        obj = json.load(f)

    try:
        return int(obj["metrics"]["best_iteration"])
    except Exception:
        print(f"[WARNING] Cannot read best_iteration from: {summary_json}")
        return None


def ensure_columns(df: pd.DataFrame, cols: list[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"[{name}] missing required columns: "
            f"{missing[:30]}{' ...' if len(missing) > 30 else ''}"
        )


def make_axes_from_df(df: pd.DataFrame):
    lat_axis = np.sort(df[LAT_COL].astype(float).unique()).astype(np.float32)
    lon_axis = np.sort(wrap_lon180(df[LON_COL].astype(float).unique())).astype(np.float32)
    return lat_axis, lon_axis


def dataframe_to_grid(
    df: pd.DataFrame,
    values: np.ndarray,
    lat_axis: np.ndarray,
    lon_axis: np.ndarray,
) -> np.ndarray:
    """
    将一维预测结果转为 2D lat-lon 网格。
    """
    grid = np.full((len(lat_axis), len(lon_axis)), np.nan, dtype=np.float32)

    lat_to_i = {float(v): i for i, v in enumerate(lat_axis)}
    lon_to_j = {float(v): j for j, v in enumerate(lon_axis)}

    lat_values = df[LAT_COL].astype(float).to_numpy()
    lon_values = wrap_lon180(df[LON_COL].astype(float).to_numpy())

    ii = np.array([lat_to_i[float(v)] for v in lat_values], dtype=int)
    jj = np.array([lon_to_j[float(v)] for v in lon_values], dtype=int)

    grid[ii, jj] = values.astype(np.float32)

    return grid


def build_strict_land_mask_from_labels(
    label_file: Path,
    lat_axis: np.ndarray,
    lon_axis: np.ndarray,
) -> np.ndarray:
    """
    使用 labels_land_strict.parquet 中的经纬度生成 strict-land mask。
    这样可以保证与训练标签的 strict-land 范围一致。
    """
    if not label_file.exists():
        raise FileNotFoundError(f"Label file not found: {label_file}")

    df_lab = pd.read_parquet(label_file)

    ensure_columns(df_lab, [LAT_COL, LON_COL], "labels_land_strict")

    mask = np.zeros((len(lat_axis), len(lon_axis)), dtype=bool)

    lat_to_i = {float(v): i for i, v in enumerate(lat_axis)}
    lon_to_j = {float(v): j for j, v in enumerate(lon_axis)}

    lat_lab = df_lab[LAT_COL].astype(float).to_numpy()
    lon_lab = wrap_lon180(df_lab[LON_COL].astype(float).to_numpy())

    inside_count = 0
    outside_count = 0

    for la, lo in zip(lat_lab, lon_lab):
        la = float(la)
        lo = float(lo)

        if la in lat_to_i and lo in lon_to_j:
            mask[lat_to_i[la], lon_to_j[lo]] = True
            inside_count += 1
        else:
            outside_count += 1

    print("[STRICT LAND MASK]")
    print("Label strict-land points inside prediction grid :", inside_count)
    print("Label strict-land points outside prediction grid:", outside_count)
    print("Strict-land grid count:", int(mask.sum()))

    return mask


def predict_one_target_to_grid(
    df: pd.DataFrame,
    features: list[str],
    booster: xgb.Booster,
    best_iteration,
    lat_axis: np.ndarray,
    lon_axis: np.ndarray,
    target_name: str,
) -> np.ndarray:
    ensure_columns(df, features, f"{target_name}_features")

    # Keep NaN predictors as NaN instead of dropping grid cells.
    # XGBoost handles missing values natively, which avoids artificial blanks
    # over high-terrain regions where some environmental predictors are undefined.
    X = df[features].copy().replace([np.inf, -np.inf], np.nan)

    dmx = xgb.DMatrix(
        X,
        feature_names=features,
        missing=np.nan,
    )

    if USE_BEST_ITERATION and best_iteration is not None:
        pred = booster.predict(
            dmx,
            iteration_range=(0, int(best_iteration) + 1),
        )
    else:
        pred = booster.predict(dmx)

    pred = pred.astype(np.float32)

    grid = dataframe_to_grid(
        df=df,
        values=pred,
        lat_axis=lat_axis,
        lon_axis=lon_axis,
    )

    return grid


def summarize_grid(grid: np.ndarray, name: str) -> dict:
    finite = np.isfinite(grid)
    vals = grid[finite]

    if vals.size == 0:
        return {
            f"{name}_finite_n": 0,
            f"{name}_min": np.nan,
            f"{name}_max": np.nan,
            f"{name}_mean": np.nan,
            f"{name}_negative_n": 0,
        }

    return {
        f"{name}_finite_n": int(vals.size),
        f"{name}_min": float(np.nanmin(vals)),
        f"{name}_max": float(np.nanmax(vals)),
        f"{name}_mean": float(np.nanmean(vals)),
        f"{name}_negative_n": int(np.sum(vals < 0)),
    }


# ============================================================
# 2. Load models and feature lists
# ============================================================

print("=" * 100)
print("[STEP 3] Predict MIROC6 SSP585 decades")
print("=" * 100)

print("\n[LOAD MODELS]")
print("FLASH:", FLASH_MODEL)
print("HAIL :", HAIL_MODEL)

bst_flash = load_booster(FLASH_MODEL)
bst_hail = load_booster(HAIL_MODEL)

features_flash = read_feature_list(FLASH_FEATURE_TXT)
features_hail = read_feature_list(HAIL_FEATURE_TXT)

best_iter_flash = read_best_iteration(FLASH_SUMMARY_JSON)
best_iter_hail = read_best_iteration(HAIL_SUMMARY_JSON)

print("\n[FEATURE LIST]")
print("FLASH n_features:", len(features_flash))
print("HAIL  n_features:", len(features_hail))

print("\n[BEST ITERATION]")
print("FLASH:", best_iter_flash)
print("HAIL :", best_iter_hail)
print("USE_BEST_ITERATION:", USE_BEST_ITERATION)


# ============================================================
# 3. Build grid and masks from first SSP585 decade
# ============================================================

first_y0, first_y1 = DECADES[0]
first_file = SSP585_FEATURE_DIR / f"features_ssp585_{first_y0}_{first_y1}.parquet"

if not first_file.exists():
    raise FileNotFoundError(f"First SSP585 feature file not found: {first_file}")

df_first = pd.read_parquet(first_file)
ensure_columns(df_first, [LAT_COL, LON_COL, ELEV_COL], first_file.name)

df_first[LON_COL] = wrap_lon180(df_first[LON_COL].to_numpy())

lat_axis, lon_axis = make_axes_from_df(df_first)

print("\n[GRID]")
print("nlat:", len(lat_axis))
print("nlon:", len(lon_axis))
print("lat range:", float(lat_axis.min()), "to", float(lat_axis.max()))
print("lon range:", float(lon_axis.min()), "to", float(lon_axis.max()))
print("expected grid cells:", len(lat_axis) * len(lon_axis))
print("first feature rows:", len(df_first))

strict_land_mask = build_strict_land_mask_from_labels(
    label_file=LABEL_FILE,
    lat_axis=lat_axis,
    lon_axis=lon_axis,
)

elev_grid = dataframe_to_grid(
    df=df_first,
    values=pd.to_numeric(df_first[ELEV_COL], errors="coerce").to_numpy(dtype=np.float32),
    lat_axis=lat_axis,
    lon_axis=lon_axis,
)

high_alt_mask = np.isfinite(elev_grid) & (elev_grid > ELEV_MAX)

print("\n[ELEVATION MASK]")
print("Finite elevation grid count:", int(np.isfinite(elev_grid).sum()))
print(f"High-altitude grid count > {ELEV_MAX} m:", int(high_alt_mask.sum()))


# ============================================================
# 4. Save masks_2deg.nc
# ============================================================

mask_ds = xr.Dataset(
    data_vars={
        "strict_land": (("lat", "lon"), strict_land_mask.astype(np.int8)),
        "elevation_m": (("lat", "lon"), elev_grid.astype(np.float32)),
        "high_alt": (("lat", "lon"), high_alt_mask.astype(np.int8)),
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
print("\n[SAVED MASK]")
print(mask_path)


# ============================================================
# 5. Predict each decade and save NetCDF
# ============================================================

summary_rows = []

for y0, y1 in DECADES:
    print("\n" + "-" * 100)
    print(f"[PREDICT] SSP585 {y0}-{y1}")
    print("-" * 100)

    feature_file = SSP585_FEATURE_DIR / f"features_ssp585_{y0}_{y1}.parquet"

    if not feature_file.exists():
        raise FileNotFoundError(f"Missing SSP585 feature file: {feature_file}")

    df = pd.read_parquet(feature_file)
    ensure_columns(df, [LAT_COL, LON_COL], feature_file.name)

    df[LON_COL] = wrap_lon180(df[LON_COL].to_numpy())

    lat_this, lon_this = make_axes_from_df(df)

    if not np.array_equal(lat_this, lat_axis):
        raise ValueError(f"Latitude axis mismatch in {feature_file.name}")

    if not np.array_equal(lon_this, lon_axis):
        raise ValueError(f"Longitude axis mismatch in {feature_file.name}")

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    grid_flash = predict_one_target_to_grid(
        df=df,
        features=features_flash,
        booster=bst_flash,
        best_iteration=best_iter_flash,
        lat_axis=lat_axis,
        lon_axis=lon_axis,
        target_name="FLASH",
    )

    grid_hail = predict_one_target_to_grid(
        df=df,
        features=features_hail,
        booster=bst_hail,
        best_iteration=best_iter_hail,
        lat_axis=lat_axis,
        lon_axis=lon_axis,
        target_name="HAIL",
    )

    # Summary before mask/clip
    flash_before = summarize_grid(grid_flash, "flash_before_mask_clip")
    hail_before = summarize_grid(grid_hail, "hail_before_mask_clip")

    # Apply strict-land mask
    grid_flash = grid_flash.astype(np.float32)
    grid_hail = grid_hail.astype(np.float32)

    grid_flash[~strict_land_mask] = np.nan
    grid_hail[~strict_land_mask] = np.nan

    # Clip negative predictions to zero
    flash_neg_after_mask = int(np.sum(np.isfinite(grid_flash) & (grid_flash < 0)))
    hail_neg_after_mask = int(np.sum(np.isfinite(grid_hail) & (grid_hail < 0)))

    grid_flash = np.where(
        np.isfinite(grid_flash) & (grid_flash < 0),
        0.0,
        grid_flash,
    ).astype(np.float32)

    grid_hail = np.where(
        np.isfinite(grid_hail) & (grid_hail < 0),
        0.0,
        grid_hail,
    ).astype(np.float32)

    flash_after = summarize_grid(grid_flash, "pred_flash")
    hail_after = summarize_grid(grid_hail, "pred_hail")

    ds = xr.Dataset(
        data_vars={
            "pred_flash": (("lat", "lon"), grid_flash),
            "pred_hail": (("lat", "lon"), grid_hail),
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
            "negative_predictions_clipped_to_zero": "true",
            "high_altitude_removed": "false",
            "high_altitude_mask_file": "masks_2deg.nc",
            "flash_model": str(FLASH_MODEL),
            "hail_model": str(HAIL_MODEL),
            "flash_features": str(FLASH_FEATURE_TXT),
            "hail_features": str(HAIL_FEATURE_TXT),
            "use_best_iteration": str(USE_BEST_ITERATION),
            "flash_best_iteration": (
                int(best_iter_flash) if best_iter_flash is not None else -1
            ),
            "hail_best_iteration": (
                int(best_iter_hail) if best_iter_hail is not None else -1
            ),
        },
    )

    out_nc = OUT_NC_DIR / f"pred_ssp585_{y0}_{y1}_2deg.nc"
    ds.to_netcdf(out_nc)

    row = {
        "model": MODEL_NAME,
        "scenario": "ssp585",
        "start_year": y0,
        "end_year": y1,
        "feature_file": str(feature_file),
        "out_nc": str(out_nc),
        "strict_land_count": int(strict_land_mask.sum()),
        "high_alt_count": int(high_alt_mask.sum()),
        "flash_negative_after_mask_before_clip": flash_neg_after_mask,
        "hail_negative_after_mask_before_clip": hail_neg_after_mask,
    }

    row.update(flash_before)
    row.update(hail_before)
    row.update(flash_after)
    row.update(hail_after)

    summary_rows.append(row)

    print("[SAVED]", out_nc)
    print("pred_flash finite n/min/max/mean:",
          flash_after["pred_flash_finite_n"],
          flash_after["pred_flash_min"],
          flash_after["pred_flash_max"],
          flash_after["pred_flash_mean"])
    print("pred_hail finite n/min/max/mean:",
          hail_after["pred_hail_finite_n"],
          hail_after["pred_hail_min"],
          hail_after["pred_hail_max"],
          hail_after["pred_hail_mean"])
    print("negative flash after mask before clip:", flash_neg_after_mask)
    print("negative hail  after mask before clip:", hail_neg_after_mask)


# ============================================================
# 6. Save summary and metadata
# ============================================================

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")

meta = {
    "created_at": time.strftime("%Y%m%d_%H%M%S"),
    "model": MODEL_NAME,
    "scenario": "ssp585",
    "run_dir": str(RUN_DIR),
    "out_dir": str(OUT_DIR),
    "out_nc_dir": str(OUT_NC_DIR),
    "mask_file": str(mask_path),
    "summary_csv": str(OUT_SUMMARY_CSV),
    "decades": DECADES,
    "use_best_iteration": USE_BEST_ITERATION,
    "flash_best_iteration": best_iter_flash,
    "hail_best_iteration": best_iter_hail,
    "flash_model": str(FLASH_MODEL),
    "hail_model": str(HAIL_MODEL),
    "flash_feature_list": str(FLASH_FEATURE_TXT),
    "hail_feature_list": str(HAIL_FEATURE_TXT),
    "strict_land_source": str(LABEL_FILE),
    "notes": [
        "Predictions are saved as NetCDF files for each SSP585 decade.",
        "Values outside strict land are set to NaN.",
        "Negative predictions are clipped to zero.",
        "High-altitude cells are not removed during prediction; high_alt is saved in masks_2deg.nc.",
    ],
}

with open(OUT_META_JSON, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 100)
print("[DONE] Step 3 finished successfully.")
print("=" * 100)
print("NC output directory:", OUT_NC_DIR)
print("Mask file          :", mask_path)
print("Summary CSV        :", OUT_SUMMARY_CSV)
print("Meta JSON          :", OUT_META_JSON)
print("\nPrediction summary preview:")
print(summary_df)
