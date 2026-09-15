# Run order: after the model's envdiag script and before label merging/training.
# Run from repository root: python -m modeling.miroc6_00_build_features

# -*- coding: utf-8 -*-
from __future__ import annotations

from publication_config import resource_path

import os
import glob
import warnings
import numpy as np
import pandas as pd
import xarray as xr

from tqdm import tqdm
from dask.diagnostics import ProgressBar

# strict_land
import cartopy.io.shapereader as shpreader
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.prepared import prep

# raster -> 2deg average
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin

xr.set_options(file_cache_maxsize=1)

# ---------------------------
# 0) 配置区：你只改这里
# ---------------------------
LABEL_XLSX = resource_path('observations', 'flashhail_matched_new2.xlsx')
ENV_DIR    = resource_path('models', 'MIROC6/envdiag')
SRTM_TIF   = resource_path('observations', 'data/elevation/elevation_1KMmn_SRTM.tif')

# 输出目录：与 CanESM5 风格一致，只改为 MIROC6
OUT_TRAIN_DIR  = resource_path('models', 'MIROC6/feature/ml_train_historical')
OUT_FUTURE_DIR = resource_path('models', 'MIROC6/feature/ml_pred_ssp585')
os.makedirs(OUT_TRAIN_DIR, exist_ok=True)
os.makedirs(OUT_FUTURE_DIR, exist_ok=True)

# ROI: [N, W, S, E]
ROI_N, ROI_W, ROI_S, ROI_E = 63, -180, -63, 180

# 标签列
LAT_COL = "Latitude"
LON_COL = "Longitude"
Y_FLASH = "LISOTD_Flash"
Y_HAIL  = "ni_HailPF"
LS_COL  = "Land_Sea"

# envdiag 变量（与 CanESM5 保持一致）
VARS = [
    "mucape", "mucin", "pw", "s06", "h_10c", "h_30c", "dh_10_30",
    "div500", "td_sfc", "theta_e", "cape", "cin", "flh", "k_index"
]

# 历史期（排除边界残年）
HIST_YEARS = (1995, 2014)

# 未来 decade：2050-2059 ... 2090-2099（2100 不参与）
FUTURE_START = 2050
FUTURE_END_EXCLUSIVE = 2100
DECADE_STEP = 10

# Feature configuration used to build the MIROC6 candidate pool.
# The final MIROC6 training workflow uses the non-seasonal subset only.
QUANTILES_A = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
SEASONS = ("DJF", "MAM", "JJA", "SON")
ANNUAL_SUFFIXES = (
    "mean", "std", "min", "max",
    "q01", "q05", "q10", "q25", "q50", "q75", "q90", "q95", "q99",
)
SEASONAL_SUFFIXES = ("mean", "q90", "max")

# 统一命名前缀
FEATURE_PREFIX = "env"

# 屏蔽刷屏 warning
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice encountered")


# ---------------------------
# 1) 工具函数
# ---------------------------
def lon_to_180(lon):
    lon = np.asarray(lon, dtype=float)
    return ((lon + 180) % 360) - 180


def build_2deg_axis_from_full(vals: np.ndarray) -> np.ndarray:
    """
    从值集合构造 2°轴（允许缺行缺列；不可用 diff 平均判断）。
    """
    vals = np.asarray(vals, dtype=float)
    vals = np.round(vals, 3)
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))

    start = np.round(np.floor(vmin / 2.0) * 2.0, 3)
    end   = np.round(np.ceil(vmax / 2.0) * 2.0, 3)
    axis  = np.round(np.arange(start, end + 0.001, 2.0), 3)
    axis = axis[(axis >= vmin - 1e-6) & (axis <= vmax + 1e-6)]
    return axis


def xr_to_df(ds: xr.Dataset) -> pd.DataFrame:
    return ds.to_dataframe().reset_index()


def decade_ranges(start=2050, end_exclusive=2100, step=10):
    out = []
    y = start
    while y + step <= end_exclusive:
        out.append((y, y + step - 1))
        y += step
    return out


def season_mask(time: xr.DataArray, season: str) -> xr.DataArray:
    """Return a boolean mask for a meteorological season."""
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


def file_exists_or_raise(path: str, what: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{what} 不存在：{path}")


# ---------------------------
# 2) Step-1：读取标签（ROI）+ Land_Sea==0
# ---------------------------
def load_labels_land_roi():
    file_exists_or_raise(LABEL_XLSX, "标签文件")

    df_all = pd.read_excel(LABEL_XLSX, usecols=[LAT_COL, LON_COL, Y_FLASH, Y_HAIL, LS_COL])
    df_all[LON_COL] = lon_to_180(df_all[LON_COL].values)

    df_all = df_all[
        (df_all[LAT_COL] >= ROI_S) & (df_all[LAT_COL] <= ROI_N) &
        (df_all[LON_COL] >= ROI_W) & (df_all[LON_COL] <= ROI_E)
    ].copy()

    # 从裁剪后的全表构造 2°轴（Land-only 子集不能用来构造轴）
    lat2 = build_2deg_axis_from_full(df_all[LAT_COL].values)
    lon2 = build_2deg_axis_from_full(df_all[LON_COL].values)

    # 陆地粗筛：Land_Sea==0
    df_land = df_all[df_all[LS_COL] == 0].copy()

    print(f"[GRID] lat2: {lat2.min()}..{lat2.max()} n={len(lat2)} step≈{np.diff(lat2).mean():.3f}")
    print(f"[GRID] lon2: {lon2.min()}..{lon2.max()} n={len(lon2)} step≈{np.diff(lon2).mean():.3f}")
    print(f"[LABEL] rows in ROI: {len(df_all)} | land rows after Land_Sea==0: {len(df_land)}")

    df_land[LAT_COL] = df_land[LAT_COL].astype(float).round(3)
    df_land[LON_COL] = df_land[LON_COL].astype(float).round(3)
    return df_land, lat2, lon2


# ---------------------------
# 3) Step-2：strict_land（Natural Earth 110m）
# ---------------------------
def compute_strict_land_mask(lat2, lon2) -> np.ndarray:
    mask_path = os.path.join(OUT_TRAIN_DIR, "strict_land_mask_latlon.npy")
    if os.path.exists(mask_path):
        print("[RESUME] Found strict_land mask, skip recompute.")
        return np.load(mask_path)

    shp = shpreader.natural_earth(resolution="110m", category="physical", name="land")
    geoms = list(shpreader.Reader(shp).geometries())
    land_union = unary_union(geoms)
    land_prep = prep(land_union)

    mask = np.zeros((len(lat2), len(lon2)), dtype=bool)
    total = len(lat2) * len(lon2)

    with tqdm(total=total, desc="strict_land (point-in-land)", unit="pt") as pbar:
        for i, lat in enumerate(lat2):
            for j, lon in enumerate(lon2):
                mask[i, j] = land_prep.contains(Point(float(lon), float(lat)))
                pbar.update(1)

    np.save(mask_path, mask)
    return mask


def apply_strict_land(df_land, lat2, lon2, landmask):
    lat_to_i = {float(v): i for i, v in enumerate(lat2)}
    lon_to_j = {float(v): j for j, v in enumerate(lon2)}

    idx_i = df_land[LAT_COL].map(lambda x: lat_to_i[float(x)]).to_numpy()
    idx_j = df_land[LON_COL].map(lambda x: lon_to_j[float(x)]).to_numpy()

    ok = landmask[idx_i, idx_j]
    out = df_land.loc[ok].copy()
    print(f"[STRICT_LAND] kept rows: {len(out)} / {len(df_land)}")
    return out


# ---------------------------
# 4) Step-3：SRTM -> 2° elev_mean（一个 elev）
# ---------------------------
def srtm_to_2deg_mean(lat2, lon2, out_dir) -> pd.DataFrame:
    elev_grid_path = os.path.join(out_dir, "elev_2deg.parquet")
    elev_npy_path  = os.path.join(out_dir, "elev_2deg.npy")

    if os.path.exists(elev_grid_path) and os.path.exists(elev_npy_path):
        print("[RESUME] Found elev_2deg (parquet+npy), skip SRTM reprojection.")
        return pd.read_parquet(elev_grid_path)

    file_exists_or_raise(SRTM_TIF, "SRTM 高程文件")

    west = float(lon2.min() - 1.0)
    north = float(lat2.max() + 1.0)
    xres = 2.0
    yres = 2.0

    dst_h = len(lat2)
    dst_w = len(lon2)
    dst = np.full((dst_h, dst_w), np.nan, dtype=np.float32)

    dst_transform = from_origin(west, north, xres, yres)
    dst_crs = "EPSG:4326"

    print("[SRTM] reproject+average to 2deg ...")
    with rasterio.open(SRTM_TIF) as src:
        src_data = src.read(1, masked=True)
        reproject(
            source=src_data,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.average,
            dst_nodata=np.nan
        )

    dst = np.flipud(dst)  # lat ascending

    Lon, Lat = np.meshgrid(lon2, lat2)
    df = pd.DataFrame({
        LAT_COL: Lat.ravel().astype(float).round(3),
        LON_COL: Lon.ravel().astype(float).round(3),
        "elev": dst.ravel().astype(np.float32)
    })

    np.save(elev_npy_path, dst)
    df.to_parquet(elev_grid_path, index=False)
    print(f"[ELEV] Saved: {elev_grid_path}")
    return df


# ---------------------------
# 5) Step-4：读取 envdiag（historical/ssp585）+ ROI + CIN/MUCIN abs
# ---------------------------
def open_envdiag(kind: str, year0: int, year1: int) -> xr.Dataset:
    if kind == "historical":
        files = []
        for y in range(year0, year1 + 1):
            f = os.path.join(ENV_DIR, f"MIROC6_historical_envdiag_{y}.nc")
            if os.path.exists(f):
                files.append(f)
        if not files:
            raise FileNotFoundError(f"未找到 MIROC6 historical 文件：{year0}-{year1}")

    elif kind == "ssp585":
        files = []
        for y in range(year0, year1 + 1):
            f = os.path.join(ENV_DIR, f"MIROC6_ssp585_envdiag_{y}.nc")
            if os.path.exists(f):
                files.append(f)
        if not files:
            raise FileNotFoundError(f"未找到 MIROC6 ssp585 文件：{year0}-{year1}")
    else:
        raise ValueError(kind)

    print(f"[ENV] open {kind} files: {len(files)}")

    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        parallel=False,
        engine="netcdf4",
        coords="minimal",
        data_vars="minimal",
        compat="override",
        chunks={}
    ).chunk({"time": 240})

    missing = [v for v in VARS if v not in ds.data_vars]
    if missing:
        raise ValueError(f"{kind} envdiag 缺变量：{missing}")
    ds = ds[VARS]

    # lon -> [-180,180)
    if float(ds["lon"].max()) > 180:
        ds = ds.assign_coords(lon=lon_to_180(ds["lon"])).sortby("lon")

    # time slice
    ds = ds.sel(time=slice(f"{year0}-01-01", f"{year1}-12-31"))

    # lat ROI slice
    latmin, latmax = float(min(ROI_S, ROI_N)), float(max(ROI_S, ROI_N))
    if ds["lat"][0] < ds["lat"][-1]:
        ds = ds.sel(lat=slice(latmin, latmax))
    else:
        ds = ds.sel(lat=slice(latmax, latmin))

    # CIN/MUCIN abs
    ds = ds.assign(cin=abs(ds["cin"]), mucin=abs(ds["mucin"]))

    return ds


def regrid_env_to_2deg(ds, lat2, lon2):
    return ds.interp(
        lat=xr.DataArray(lat2, dims="lat"),
        lon=xr.DataArray(lon2, dims="lon"),
        method="linear"
    )


# ---------------------------
# 6) Step-5: feature engineering
# ---------------------------
def _time_quantile(da: xr.DataArray, q: float) -> xr.DataArray:
    """Calculate one time quantile without retaining a scalar quantile coordinate."""
    result = da.quantile(q, dim="time", skipna=True)
    if "quantile" in result.coords:
        result = result.reset_coords("quantile", drop=True)
    return result


def add_annual_statistics(feats: xr.Dataset, da: xr.DataArray, prefix: str):
    """Add the 13 full-period statistics used in the original MIROC6 feature table."""
    feats[f"{prefix}_mean"] = da.mean("time", skipna=True)
    feats[f"{prefix}_std"] = da.std("time", skipna=True)
    feats[f"{prefix}_min"] = da.min("time", skipna=True)
    feats[f"{prefix}_max"] = da.max("time", skipna=True)

    for q in QUANTILES_A:
        name = f"q{int(round(q * 100)):02d}"
        feats[f"{prefix}_{name}"] = _time_quantile(da, q)


def add_seasonal_statistics(feats: xr.Dataset, da: xr.DataArray, prefix: str):
    """Add seasonal mean, q90 and maximum for DJF/MAM/JJA/SON."""
    for season in SEASONS:
        seasonal = da.where(season_mask(da["time"], season), drop=True)
        feats[f"{prefix}_{season}_mean"] = seasonal.mean("time", skipna=True)
        feats[f"{prefix}_{season}_q90"] = _time_quantile(seasonal, 0.90)
        feats[f"{prefix}_{season}_max"] = seasonal.max("time", skipna=True)


def build_features(ds2: xr.Dataset) -> xr.Dataset:
    feats = xr.Dataset()
    for var in tqdm(list(ds2.data_vars), desc="Build features (vars)", unit="var"):
        da = ds2[var]
        prefix = f"{FEATURE_PREFIX}_{var}"
        add_annual_statistics(feats, da, prefix)
        add_seasonal_statistics(feats, da, prefix)

    expected = len(VARS) * (
        len(ANNUAL_SUFFIXES) + len(SEASONS) * len(SEASONAL_SUFFIXES)
    )
    if len(feats.data_vars) != expected:
        raise RuntimeError(
            f"Feature schema mismatch: found {len(feats.data_vars)}, expected {expected}."
        )
    return feats


def compute_features_to_parquet(kind: str, year0: int, year1: int,
                                lat2, lon2, out_dir: str,
                                df_elev_grid: pd.DataFrame) -> pd.DataFrame:
    """
    Save features parquet that includes elev.
    """
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

    df_feat = xr_to_df(feat).rename(columns={"lat": LAT_COL, "lon": LON_COL})
    df_feat[LAT_COL] = df_feat[LAT_COL].astype(float).round(3)
    df_feat[LON_COL] = df_feat[LON_COL].astype(float).round(3)

    # merge elev into features
    df_out = df_feat.merge(df_elev_grid, on=[LAT_COL, LON_COL], how="left")

    feature_columns = [
        c for c in df_out.columns if c.startswith(f"{FEATURE_PREFIX}_")
    ]
    expected = len(VARS) * (
        len(ANNUAL_SUFFIXES) + len(SEASONS) * len(SEASONAL_SUFFIXES)
    )
    if len(feature_columns) != expected or "quantile" in df_out.columns:
        raise RuntimeError(
            f"Parquet feature schema mismatch: found {len(feature_columns)} "
            f"environmental features, expected {expected}."
        )

    df_out.to_parquet(feat_path, index=False)
    print(f"[FEAT] Saved: {feat_path}")
    return df_out


# ---------------------------
# 7) main：仅生成特征与空间骨架
# ---------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("MIROC6 feature generation (same framework as CanESM5)")
    print("=" * 80)

    # A) labels -> 2° grid
    df_land, lat2, lon2 = load_labels_land_roi()

    # B) strict_land mask + strict land labels
    landmask = compute_strict_land_mask(lat2, lon2)
    df_land_strict = apply_strict_land(df_land, lat2, lon2, landmask)

    strict_path = os.path.join(OUT_TRAIN_DIR, "labels_land_strict.parquet")
    if not os.path.exists(strict_path):
        df_land_strict.to_parquet(strict_path, index=False)
        print(f"[SKELETON] Saved: {strict_path}")
    else:
        print(f"[RESUME] Found {os.path.basename(strict_path)}")

    # C) elev 2° cache
    df_elev_grid = srtm_to_2deg_mean(lat2, lon2, OUT_TRAIN_DIR)

    # D) historical features (includes elev)
    y0, y1 = HIST_YEARS
    _ = compute_features_to_parquet("historical", y0, y1, lat2, lon2, OUT_TRAIN_DIR, df_elev_grid)

    # E) future decade features (includes elev)
    decades = decade_ranges(FUTURE_START, FUTURE_END_EXCLUSIVE, DECADE_STEP)
    print("\n[FUTURE] Decades:", decades)
    for (a, b) in decades:
        _ = compute_features_to_parquet("ssp585", a, b, lat2, lon2, OUT_FUTURE_DIR, df_elev_grid)

    print("\n[DONE] MIROC6 feature generation only.")
    print("[OUT] historical skeleton/features:", OUT_TRAIN_DIR)
    print("[OUT] future decade features     :", OUT_FUTURE_DIR)
