# Purpose: figure 04 trends.
# Source: cmip6_figure_MMM_3models_joint2d_trend_with_markers.ipynb; see docs/source_coverage.csv for cell provenance.
# Run order and required inputs: docs/workflow.md.
# Run from repository root: python -m figures.figure_04_trends

# ============================================================
# Step 6. Plot MMM SSP585 trend maps + regional decadal trajectories
#        with three-model intermodel range
#
# Outputs
#   (a) MMM Lightning Sen's-slope map
#   (b) MMM Hail Sen's-slope map
#   (c) Regional lightning trajectories: MMM + min–max model range
#   (d) Regional hail trajectories: MMM + min–max model range
#
# Core methodology
#   1. Read future predictions separately for BCC-CSM2-MR, CanESM5, and MIROC6.
#   2. Compute the equal-weight multi-model mean (MMM) for each future decade.
#   3. Derive Sen's slope and Mann–Kendall/Kendall p values from the five MMM decades.
#   4. Compute regional means separately for each model and decade.
#   5. Plot regional MMM trajectories, with vertical whiskers representing the
#      full intermodel range (minimum to maximum across the three models).
#
# Important interpretation
#   The whiskers represent structural spread among the three climate models,
#   not statistical confidence intervals.
# ============================================================

from __future__ import annotations

from publication_config import resource_path

import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import scipy.stats as stats

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.ticker import FixedLocator

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


# ============================================================
# 0. User configuration
# ============================================================

# Model display name used in logs and output metadata.
MODEL_NAME = "MMM (BCC-CSM2-MR + CanESM5 + MIROC6)"

# Update only if your local folders differ.
MODEL_BASE_DIRS = {
    "BCC-CSM2-MR": Path(resource_path('models', 'BCC-CSM2-MR')),
    "CanESM5": Path(resource_path('models', 'CanESM5')),
    "MIROC6": Path(resource_path('models', 'MIROC6')),
}

# All three models are cropped to the shared 2° domain |lat| <= 63°.
LAT_MAX_COMMON = 63.0

# This model only supplies the common strict-land / elevation masks.
REFERENCE_MASK_MODEL = "CanESM5"

OUT_ROOT = Path(resource_path('figures', ''))
OUT_DIR = OUT_ROOT / 'figure_04_trends'
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FIG = OUT_DIR / 'figure_04_trends_mmm.jpg'
OUT_REGIONAL_CSV = OUT_DIR / "regional_decadal_summary_MMM_intermodel_range.csv"
OUT_REGIONAL_MODEL_CSV = OUT_DIR / "regional_decadal_mean_by_model.csv"
OUT_TREND_NPZ = OUT_DIR / "trend_sen_slope_flash_hail_MMM.npz"

DECADES = [
    (2050, 2059),
    (2060, 2069),
    (2070, 2079),
    (2080, 2089),
    (2090, 2099),
]

DECADE_LABELS = ["2050s", "2060s", "2070s", "2080s", "2090s"]
LINE_LABELS = DECADE_LABELS.copy()
LINE_PERIOD_YEARS = DECADES.copy()
LINE_PERIOD_SCENARIOS = ["ssp585"] * len(DECADES)

VAR_FLASH = "pred_flash"
VAR_HAIL = "pred_hail"

EXTENT = [-180, 180, -63, 63]

# High-elevation handling.
ELEV_MAX = 2000.0
FILTER_HIGH_ALT_FOR_REGION_MEAN = False
FILTER_HIGH_ALT_FOR_TREND_MAP = False

# Regional / global mean handling. Recommended: True.
AREA_WEIGHTED_REGION_MEAN = True

# Trend-map significance.
DRAW_SIGNIFICANCE = True
P_SIG = 0.05

# Regional intermodel-spread styling.
SHOW_INTERMODEL_RANGE = True
INTERMODEL_RANGE_ALPHA = 0.8
INTERMODEL_RANGE_LINEWIDTH = 1
INTERMODEL_RANGE_CAPSIZE = 2.8
INTERMODEL_RANGE_CAPTHICK = 1

# Optional fixed colour-bar limits. Set None for automatic symmetric limits.
VMAX_FLASH_TREND = None
VMAX_HAIL_TREND = None

FIG_DPI = 600


# ============================================================
# 1. Layout configuration
# ============================================================

LAYOUT = {
    # Overall canvas.
    "figsize": (20.0, 9.6),

    # Keep False because axes are manually positioned with fig.add_axes().
    "save_bbox_tight": False,

    # Left maps: [left, bottom, width, height].
    "ax_map_flash": [0.02, 0.635, 0.50, 0.305],
    "ax_map_hail": [0.02, 0.140, 0.50, 0.305],

    # Right line panels. Vertical bounds are dynamically aligned to maps.
    "line_flash_left": 0.57,
    "line_hail_left": 0.80,
    "line_w": 0.18,

    # Horizontal colour bars.
    "cbar_gap_flash": 0.060,
    "cbar_gap_hail": 0.060,
    "cbar_h": 0.022,

    # Shared region legend.
    "legend_anchor": (0.760, 0.015),
    "legend_ncol": 5,

    # Valid values: "bottom" or "top".
    "cbar_label_position": "bottom",

    "print_layout": True,
}


# ============================================================
# 2. Plot style
# ============================================================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 18
plt.rcParams["axes.titlesize"] = 24
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18
plt.rcParams["legend.fontsize"] = 18
plt.rcParams["figure.dpi"] = FIG_DPI
plt.rcParams["axes.unicode_minus"] = False

TITLE_FONTSIZE = 24
LABEL_FONTSIZE = 20
TICK_FONTSIZE = 18
LEGEND_FONTSIZE = 18
CBAR_LABEL_FONTSIZE = 18
CBAR_TICK_FONTSIZE = 17
BORDER_WIDTH = 1.1


# ============================================================
# 3. Region definitions
# ============================================================

REGIONS = {
    "FL1": {
        "box": (-98, -78, 24, 35),
        "color": "#F2A900",
        "marker": "o",
        "ls": "-",
    },
    "FL2": {
        "box": (-76, -58, -10, 5),
        "color": "#F0E442",
        "marker": "*",
        "ls": "-",
    },
    "FL3": {
        "box": (12, 35, -8, 5),
        "color": "#D94801",
        "marker": "^",
        "ls": "-",
    },
    "FL4": {
        "box": (95, 150, -12, 8),
        "color": "#CC79A7",
        "marker": "D",
        "ls": "-",
    },
    "HL1": {
        "box": (-104, -90, 35, 47),
        "color": "#1B9E77",
        "marker": "o",
        "ls": "--",
    },
    "HL2": {
        "box": (-68, -50, -42, -23),
        "color": "#0072B2",
        "marker": "*",
        "ls": "--",
    },
    "HL3": {
        "box": (-10, 20, 35, 55),
        "color": "#56B4E9",
        "marker": "^",
        "ls": "--",
    },
    "HL4": {
        "box": (112, 138, 35, 55),
        "color": "#3B4CC0",
        "marker": "D",
        "ls": "--",
    },
    "Global": {
        "box": None,
        "color": "#9E9E9E",
        "marker": "P",
        "ls": "-",
    },
}

MAP_BOXES = [
    (-104, -90, 35, 47, "blue"),
    (-68, -50, -42, -23, "blue"),
    (-10, 20, 35, 55, "blue"),
    (112, 138, 35, 55, "blue"),
    (-98, -78, 24, 35, "red"),
    (-76, -58, -10, 5, "red"),
    (12, 35, -8, 5, "red"),
    (95, 150, -12, 8, "red"),
]


# ============================================================
# 4. Helper functions: input and grid handling
# ============================================================

def wrap_lon180(lon: np.ndarray) -> np.ndarray:
    """Convert longitude to [-180, 180)."""
    lon = np.asarray(lon, dtype=float)
    return ((lon + 180.0) % 360.0) - 180.0


def get_model_run_dir(base_dir: Path) -> Path:
    return base_dir / "ml_xgboost_result" / 'FINAL_XGB'


def get_model_nc_dir(base_dir: Path) -> Path:
    return get_model_run_dir(base_dir) / "PRED_SSP585" / "nc"


def read_one_decade_nc(path_nc: Path):
    """Read one future-decade prediction file and crop to shared latitude rows."""
    if not path_nc.exists():
        raise FileNotFoundError(path_nc)

    with xr.open_dataset(path_nc) as ds:
        if VAR_FLASH not in ds:
            raise KeyError(f"{path_nc.name} is missing variable '{VAR_FLASH}'.")
        if VAR_HAIL not in ds:
            raise KeyError(f"{path_nc.name} is missing variable '{VAR_HAIL}'.")
        if "lat" not in ds or "lon" not in ds:
            raise KeyError(f"{path_nc.name} must contain 'lat' and 'lon' coordinates.")

        flash = ds[VAR_FLASH].values.astype(np.float64)
        hail = ds[VAR_HAIL].values.astype(np.float64)
        lat = ds["lat"].values.astype(np.float64)
        lon = wrap_lon180(ds["lon"].values.astype(np.float64))

    if flash.ndim != 2 or hail.ndim != 2:
        raise ValueError(
            f"{path_nc.name}: expected 2D [lat, lon] fields; "
            f"got flash={flash.shape}, hail={hail.shape}."
        )

    lat_keep = np.abs(lat) <= LAT_MAX_COMMON
    if lat_keep.sum() == 0:
        raise ValueError(
            f"No latitude remains after |lat| <= {LAT_MAX_COMMON}: {path_nc}"
        )

    flash = flash[lat_keep, :]
    hail = hail[lat_keep, :]
    lat = lat[lat_keep]

    # Sort longitudes so all model grids are in the same order.
    order = np.argsort(lon)
    lon = lon[order]
    flash = flash[:, order]
    hail = hail[:, order]

    # Predictions should not be negative, but protect against occasional artifacts.
    flash = np.where(np.isfinite(flash) & (flash < 0), 0.0, flash)
    hail = np.where(np.isfinite(hail) & (hail < 0), 0.0, hail)

    return flash, hail, lat, lon


def load_one_model_future_stack(base_dir: Path, model_name: str):
    """Load five SSP585 decadal fields for one climate model."""
    flash_list = []
    hail_list = []
    lat_ref = None
    lon_ref = None

    nc_dir = get_model_nc_dir(base_dir)

    for y0, y1 in DECADES:
        path_nc = nc_dir / f"pred_ssp585_{y0}_{y1}_2deg.nc"
        flash, hail, lat, lon = read_one_decade_nc(path_nc)

        if lat_ref is None:
            lat_ref = lat
            lon_ref = lon
        else:
            if not np.allclose(lat, lat_ref, equal_nan=True):
                raise ValueError(f"[{model_name}] latitude mismatch: {path_nc.name}")
            if not np.allclose(lon, lon_ref, equal_nan=True):
                raise ValueError(f"[{model_name}] longitude mismatch: {path_nc.name}")

        flash_list.append(flash)
        hail_list.append(hail)
        print(f"  [LOAD] {model_name:<12s} | {path_nc.name} | shape={flash.shape}")

    return (
        np.stack(flash_list, axis=0),
        np.stack(hail_list, axis=0),
        lat_ref,
        lon_ref,
    )


def load_prediction_stacks():
    """
    Load model-specific and MMM stacks.

    Returns
    -------
    flash_models : ndarray, shape [model, decade, lat, lon]
    hail_models  : ndarray, shape [model, decade, lat, lon]
    flash_mmm    : ndarray, shape [decade, lat, lon]
    hail_mmm     : ndarray, shape [decade, lat, lon]
    lat_ref, lon_ref : ndarray
    model_names : list[str]
    """
    flash_models = []
    hail_models = []
    model_names = []

    lat_ref = None
    lon_ref = None

    for model_name, base_dir in MODEL_BASE_DIRS.items():
        flash_stack, hail_stack, lat, lon = load_one_model_future_stack(
            base_dir=base_dir,
            model_name=model_name,
        )

        if lat_ref is None:
            lat_ref = lat
            lon_ref = lon
        else:
            if not np.allclose(lat, lat_ref, equal_nan=True):
                raise ValueError(f"[{model_name}] future latitude axis differs from reference.")
            if not np.allclose(lon, lon_ref, equal_nan=True):
                raise ValueError(f"[{model_name}] future longitude axis differs from reference.")

        flash_models.append(flash_stack)
        hail_models.append(hail_stack)
        model_names.append(model_name)

    flash_models = np.stack(flash_models, axis=0)
    hail_models = np.stack(hail_models, axis=0)

    # Equal-weight arithmetic multi-model mean at each grid cell and decade.
    flash_mmm = np.nanmean(flash_models, axis=0)
    hail_mmm = np.nanmean(hail_models, axis=0)

    return (
        flash_models,
        hail_models,
        flash_mmm,
        hail_mmm,
        lat_ref,
        lon_ref,
        model_names,
    )


def load_masks(lat: np.ndarray, lon: np.ndarray):
    """Read common strict-land, high-altitude, and elevation masks."""
    mask_path = get_model_nc_dir(MODEL_BASE_DIRS[REFERENCE_MASK_MODEL]) / "masks_2deg.nc"

    if not mask_path.exists():
        raise FileNotFoundError(mask_path)

    with xr.open_dataset(mask_path) as ds:
        required = ["strict_land", "high_alt", "elevation_m", "lat", "lon"]
        missing = [v for v in required if v not in ds]
        if missing:
            raise KeyError(f"{mask_path.name} is missing: {missing}")

        strict_land = ds["strict_land"].values.astype(bool)
        high_alt = ds["high_alt"].values.astype(bool)
        elev = ds["elevation_m"].values.astype(np.float64)
        mlat = ds["lat"].values.astype(np.float64)
        mlon = wrap_lon180(ds["lon"].values.astype(np.float64))

    mlat_keep = np.abs(mlat) <= LAT_MAX_COMMON
    strict_land = strict_land[mlat_keep, :]
    high_alt = high_alt[mlat_keep, :]
    elev = elev[mlat_keep, :]
    mlat = mlat[mlat_keep]

    order = np.argsort(mlon)
    mlon = mlon[order]
    strict_land = strict_land[:, order]
    high_alt = high_alt[:, order]
    elev = elev[:, order]

    if mlat.shape != lat.shape or not np.allclose(mlat, lat, equal_nan=True):
        raise ValueError("Mask latitude axis does not match MMM prediction latitude axis.")
    if mlon.shape != lon.shape or not np.allclose(mlon, lon, equal_nan=True):
        raise ValueError("Mask longitude axis does not match MMM prediction longitude axis.")

    return strict_land, high_alt, elev


# ============================================================
# 5. Helper functions: statistics
# ============================================================

def sen_slope_and_pvalue(stack_3d: np.ndarray, valid_mask_2d: np.ndarray | None = None):
    """Compute Theil–Sen slopes and two-sided Kendall p values grid cell by grid cell."""
    nt, nlat, nlon = stack_3d.shape
    x = np.arange(nt, dtype=float)

    slope = np.full((nlat, nlon), np.nan, dtype=np.float32)
    pval = np.full((nlat, nlon), np.nan, dtype=np.float32)

    for i in range(nlat):
        for j in range(nlon):
            if valid_mask_2d is not None and not valid_mask_2d[i, j]:
                continue

            y = stack_3d[:, i, j].astype(float)
            valid = np.isfinite(y)

            if valid.sum() < 3:
                continue

            xx = x[valid]
            yy = y[valid]

            try:
                slope[i, j] = stats.theilslopes(yy, xx)[0]
            except Exception:
                slope[i, j] = np.nan

            try:
                _, p = stats.kendalltau(xx, yy)
                pval[i, j] = p
            except Exception:
                pval[i, j] = np.nan

    return slope, pval


def symmetric_vmax(arr: np.ndarray, pct: float = 98, fallback: float = 1.0) -> float:
    """Robust symmetric colour-bar extent based on a high percentile of |values|."""
    vals = np.asarray(arr, dtype=float)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return float(fallback)

    vmax = np.nanpercentile(np.abs(vals), pct)

    if not np.isfinite(vmax) or vmax <= 0:
        vmax = np.nanmax(np.abs(vals))

    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(fallback)

    return float(vmax)


def region_mask(lat: np.ndarray, lon: np.ndarray, region_box):
    """Boolean 2D mask for a named region or the entire analysis domain."""
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")

    if region_box is None:
        return np.ones_like(lat2d, dtype=bool)

    lon0, lon1, lat0, lat1 = region_box
    return (
        (lat2d >= lat0)
        & (lat2d <= lat1)
        & (lon2d >= lon0)
        & (lon2d <= lon1)
    )


def area_weighted_mean(arr: np.ndarray, valid_mask: np.ndarray, lat: np.ndarray):
    """Calculate a simple or cos(latitude)-weighted spatial mean."""
    if valid_mask.sum() == 0:
        return np.nan, 0

    arr = np.asarray(arr, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    if AREA_WEIGHTED_REGION_MEAN:
        lat_weight = np.cos(np.deg2rad(np.asarray(lat, dtype=float)))[:, None]
        weights = np.broadcast_to(lat_weight, arr.shape)

        w = np.where(valid_mask, weights, np.nan)
        a = np.where(valid_mask, arr, np.nan)
        denom = np.nansum(w)

        if not np.isfinite(denom) or denom <= 0:
            return np.nan, 0

        mean_val = np.nansum(a * w) / denom
    else:
        mean_val = np.nanmean(np.where(valid_mask, arr, np.nan))

    return float(mean_val), int(valid_mask.sum())


def compute_region_means_by_model(
    stack_4d: np.ndarray,
    model_names: list[str],
    lat: np.ndarray,
    lon: np.ndarray,
    strict_land: np.ndarray,
    high_alt: np.ndarray,
    var_name: str,
    period_labels: list[str],
    period_years: list[tuple[int, int]],
    period_scenarios: list[str],
):
    """Calculate regional decadal means separately for every model."""
    rows = []

    for rname, cfg in REGIONS.items():
        rmask = region_mask(lat, lon, cfg["box"])
        base_mask = rmask & strict_land

        if FILTER_HIGH_ALT_FOR_REGION_MEAN:
            base_mask = base_mask & (~high_alt)

        for imodel, model_name in enumerate(model_names):
            for it, (label, (y0, y1), scenario) in enumerate(
                zip(period_labels, period_years, period_scenarios)
            ):
                arr = stack_4d[imodel, it, :, :]
                valid = base_mask & np.isfinite(arr)
                mean_val, n_val = area_weighted_mean(arr, valid, lat)

                rows.append(
                    {
                        "variable": var_name,
                        "region": rname,
                        "model": model_name,
                        "scenario": scenario,
                        "decade_label": label,
                        "start_year": y0,
                        "end_year": y1,
                        "plot_order": it,
                        "mean": mean_val,
                        "n_grid": n_val,
                        "filter_high_alt": FILTER_HIGH_ALT_FOR_REGION_MEAN,
                        "area_weighted": AREA_WEIGHTED_REGION_MEAN,
                    }
                )

    return rows


def summarize_region_means(model_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize model-specific regional means into MMM and intermodel range."""
    group_cols = [
        "variable",
        "region",
        "scenario",
        "decade_label",
        "start_year",
        "end_year",
        "plot_order",
    ]

    summary = (
        model_df.groupby(group_cols, as_index=False)
        .agg(
            mmm_mean=("mean", "mean"),
            model_min=("mean", "min"),
            model_max=("mean", "max"),
            model_sd=("mean", "std"),
            n_models=("mean", lambda x: int(np.isfinite(np.asarray(x, dtype=float)).sum())),
            n_grid=("n_grid", "max"),
        )
    )

    # With only one valid model, pandas returns NaN for SD. Use 0 for reporting.
    summary["model_sd"] = summary["model_sd"].fillna(0.0)
    summary["range_low"] = summary["mmm_mean"] - summary["model_min"]
    summary["range_high"] = summary["model_max"] - summary["mmm_mean"]
    summary["filter_high_alt"] = FILTER_HIGH_ALT_FOR_REGION_MEAN
    summary["area_weighted"] = AREA_WEIGHTED_REGION_MEAN

    return summary


# ============================================================
# 6. Helper functions: map drawing
# ============================================================

def make_diverging_cmap():
    return mcolors.LinearSegmentedColormap.from_list(
        "blue_gray_red",
        [
            (0.20, 0.30, 0.75),
            (0.95, 0.95, 0.95),
            (0.75, 0.20, 0.20),
        ],
        N=256,
    )


def add_map_features(ax):
    proj = ccrs.PlateCarree()

    ax.set_extent(EXTENT, crs=proj)
    ax.coastlines(linewidth=0.55)
    ax.add_feature(cfeature.BORDERS, linewidth=0.25)

    gl = ax.gridlines(
        draw_labels=False,
        linewidth=0.45,
        color="gray",
        alpha=0.45,
        linestyle="--",
    )
    gl.xlocator = FixedLocator([-180, -120, -60, 0, 60, 120, 180])
    gl.ylocator = FixedLocator([-60, -30, 0, 30, 60])

    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=proj)
    ax.set_yticks([-60, -30, 0, 30, 60], crs=proj)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_FONTSIZE,
        direction="out",
        length=4.5,
        width=0.9,
        top=False,
        right=False,
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(BORDER_WIDTH)


def add_region_boxes(ax):
    proj = ccrs.PlateCarree()

    for lon0, lon1, lat0, lat1, color in MAP_BOXES:
        rect = mpatches.Rectangle(
            (lon0, lat0),
            lon1 - lon0,
            lat1 - lat0,
            linewidth=1.5,
            edgecolor=color,
            facecolor="none",
            transform=proj,
            zorder=8,
        )
        ax.add_patch(rect)


def add_significance_x(ax, lon: np.ndarray, lat: np.ndarray, sig_mask: np.ndarray, stride: int = 1):
    if not DRAW_SIGNIFICANCE:
        return

    lon2d, lat2d = np.meshgrid(lon, lat)
    mask = sig_mask & np.isfinite(lon2d) & np.isfinite(lat2d)

    if stride > 1:
        sampled = np.zeros_like(mask, dtype=bool)
        sampled[::stride, ::stride] = True
        mask = mask & sampled

    ax.scatter(
        lon2d[mask],
        lat2d[mask],
        marker="x",
        s=7,
        linewidths=0.45,
        color="black",
        alpha=0.70,
        transform=ccrs.PlateCarree(),
        zorder=9,
    )


def plot_trend_map(ax, slope, pval, lat, lon, title, cmap, vmax):
    proj = ccrs.PlateCarree()

    slope_cyclic, lon_cyclic = add_cyclic_point(slope, coord=lon)
    mesh = ax.pcolormesh(
        lon_cyclic,
        lat,
        slope_cyclic,
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        shading="auto",
        transform=proj,
        zorder=1,
    )

    add_map_features(ax)
    add_region_boxes(ax)

    sig = np.isfinite(pval) & (pval < P_SIG) & np.isfinite(slope)
    add_significance_x(ax, lon, lat, sig, stride=1)

    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold", pad=7)
    return mesh


# ============================================================
# 7. Helper functions: line drawing and layout
# ============================================================

def style_line_axis(ax, title: str, ylabel: str | None = None):
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold", pad=8)

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)

    x = np.arange(len(LINE_LABELS))
    ax.set_xticks(x)
    ax.set_xticklabels(LINE_LABELS, fontsize=TICK_FONTSIZE)

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        labelsize=TICK_FONTSIZE,
        length=5,
        width=1.0,
        top=False,
        right=False,
    )

    ax.grid(True, linestyle="--", linewidth=0.7, color="0.70", alpha=0.70)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(BORDER_WIDTH)


def plot_regional_lines(ax, regional_df: pd.DataFrame, variable: str, title: str, ylabel: str | None = None):
    """Plot regional MMM trajectories and min–max intermodel whiskers."""
    x = np.arange(len(LINE_LABELS))

    handles = []
    labels = []

    for rname, cfg in REGIONS.items():
        d = regional_df[
            (regional_df["variable"] == variable)
            & (regional_df["region"] == rname)
        ].copy()

        d = d.sort_values("plot_order") if "plot_order" in d.columns else d.sort_values("start_year")

        y = d["mmm_mean"].to_numpy(dtype=float)
        y_min = d["model_min"].to_numpy(dtype=float)
        y_max = d["model_max"].to_numpy(dtype=float)

        if len(y) != len(x):
            raise ValueError(
                f"{variable}-{rname} has {len(y)} points, "
                f"but LINE_LABELS has {len(x)} labels: {LINE_LABELS}"
            )

        # Draw the full range across the three model-specific regional means.
        if SHOW_INTERMODEL_RANGE:
            valid = np.isfinite(y) & np.isfinite(y_min) & np.isfinite(y_max)
            if valid.any():
                lower = y[valid] - y_min[valid]
                upper = y_max[valid] - y[valid]

                # Guard against tiny numerical negatives.
                lower = np.maximum(lower, 0.0)
                upper = np.maximum(upper, 0.0)

                ax.errorbar(
                    x[valid],
                    y[valid],
                    yerr=np.vstack([lower, upper]),
                    fmt="none",
                    ecolor=cfg["color"],
                    elinewidth=INTERMODEL_RANGE_LINEWIDTH,
                    alpha=INTERMODEL_RANGE_ALPHA,
                    capsize=INTERMODEL_RANGE_CAPSIZE,
                    capthick=INTERMODEL_RANGE_CAPTHICK,
                    zorder=1,
                )

        # MMM primary trajectory.
        handle, = ax.plot(
            x,
            y,
            linestyle=cfg["ls"],
            marker=cfg["marker"],
            markersize=8 if cfg["marker"] != "*" else 13,
            linewidth=2.0,
            color=cfg["color"],
            label=rname,
            zorder=2,
        )

        handles.append(handle)
        labels.append(rname)

    style_line_axis(ax, title=title, ylabel=ylabel)
    return handles, labels


def style_horizontal_colorbar(cb, label: str):
    cb.set_label(label, fontsize=CBAR_LABEL_FONTSIZE, labelpad=4)

    if LAYOUT["cbar_label_position"] == "top":
        cb.ax.xaxis.set_label_position("top")
        cb.ax.xaxis.tick_bottom()
    else:
        cb.ax.xaxis.set_label_position("bottom")
        cb.ax.xaxis.tick_bottom()

    cb.ax.tick_params(
        labelsize=CBAR_TICK_FONTSIZE,
        length=4,
        width=0.9,
        pad=2,
    )


def align_axes_colorbars_and_lines(
    fig,
    ax_map_flash,
    ax_map_hail,
    cax_flash,
    cax_hail,
    ax_line_flash,
    ax_line_hail,
):
    """Align colour bars to map widths and line panels to map top/bottom positions."""
    fig.canvas.draw()

    pos_a = ax_map_flash.get_position()
    pos_b = ax_map_hail.get_position()

    cax_flash.set_position(
        [
            pos_a.x0,
            pos_a.y0 - LAYOUT["cbar_gap_flash"],
            pos_a.width,
            LAYOUT["cbar_h"],
        ]
    )
    cax_hail.set_position(
        [
            pos_b.x0,
            pos_b.y0 - LAYOUT["cbar_gap_hail"],
            pos_b.width,
            LAYOUT["cbar_h"],
        ]
    )

    line_bottom = pos_b.y0
    line_top = pos_a.y1
    line_height = line_top - line_bottom

    ax_line_flash.set_position(
        [LAYOUT["line_flash_left"], line_bottom, LAYOUT["line_w"], line_height]
    )
    ax_line_hail.set_position(
        [LAYOUT["line_hail_left"], line_bottom, LAYOUT["line_w"], line_height]
    )

    print("\n[DYNAMIC ALIGNMENT]")
    print("Map a actual position:", pos_a)
    print("Map b actual position:", pos_b)
    print("Colorbar flash:", cax_flash.get_position())
    print("Colorbar hail :", cax_hail.get_position())
    print("Line flash    :", ax_line_flash.get_position())
    print("Line hail     :", ax_line_hail.get_position())


def safe_savefig(fig, out_path: Path, dpi: int = 600):
    """Save atomically to avoid partially written image files."""
    out_path = Path(os.path.normpath(str(out_path)))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = Path(str(out_path) + ".tmp.png")

    if tmp_path.exists():
        tmp_path.unlink()
    if out_path.exists():
        out_path.unlink()

    if LAYOUT["save_bbox_tight"]:
        fig.savefig(tmp_path, dpi=dpi, bbox_inches="tight")
    else:
        fig.savefig(tmp_path, dpi=dpi)

    os.replace(tmp_path, out_path)


def print_layout():
    print("\n[LAYOUT PARAMETERS]")
    for key, value in LAYOUT.items():
        print(f"{key}: {value}")


# ============================================================
# 8. Main
# ============================================================

def main():
    print("=" * 100)
    print("[STEP 6 MMM] Trend maps + regional MMM trajectories with intermodel range")
    print("=" * 100)
    print("Model set:", MODEL_NAME)

    # Validate all model root folders before processing.
    for model_name, base_dir in MODEL_BASE_DIRS.items():
        if not base_dir.exists():
            raise FileNotFoundError(f"[{model_name}] base directory not found: {base_dir}")

    if LAYOUT["print_layout"]:
        print_layout()

    # --------------------------------------------------------
    # Read predictions and masks
    # --------------------------------------------------------
    (
        flash_model_stack,
        hail_model_stack,
        flash_stack,
        hail_stack,
        lat,
        lon,
        model_names,
    ) = load_prediction_stacks()

    strict_land, high_alt, elev = load_masks(lat, lon)

    line_period_labels = LINE_LABELS
    line_period_years = LINE_PERIOD_YEARS
    line_period_scenarios = LINE_PERIOD_SCENARIOS

    print("\n[STACK]")
    print("flash_model_stack [model, decade, lat, lon]:", flash_model_stack.shape)
    print("hail_model_stack  [model, decade, lat, lon]:", hail_model_stack.shape)
    print("flash MMM stack   [decade, lat, lon]:", flash_stack.shape)
    print("hail MMM stack    [decade, lat, lon]:", hail_stack.shape)
    print("model order:", model_names)
    print("lat range:", float(np.nanmin(lat)), "to", float(np.nanmax(lat)))
    print("lon range:", float(np.nanmin(lon)), "to", float(np.nanmax(lon)))
    print("strict_land count:", int(strict_land.sum()))
    print(f"high_alt count > {ELEV_MAX} m:", int(high_alt.sum()))
    print("AREA_WEIGHTED_REGION_MEAN:", AREA_WEIGHTED_REGION_MEAN)
    print("FILTER_HIGH_ALT_FOR_REGION_MEAN:", FILTER_HIGH_ALT_FOR_REGION_MEAN)
    print("FILTER_HIGH_ALT_FOR_TREND_MAP:", FILTER_HIGH_ALT_FOR_TREND_MAP)

    # --------------------------------------------------------
    # Trend maps: calculated from five MMM decadal fields
    # --------------------------------------------------------
    valid_trend_mask = strict_land.copy()
    if FILTER_HIGH_ALT_FOR_TREND_MAP:
        valid_trend_mask &= ~high_alt

    print("\n[TREND]")
    print("valid trend grid count:", int(valid_trend_mask.sum()))

    slope_flash, p_flash = sen_slope_and_pvalue(
        flash_stack,
        valid_mask_2d=valid_trend_mask,
    )
    slope_hail, p_hail = sen_slope_and_pvalue(
        hail_stack,
        valid_mask_2d=valid_trend_mask,
    )

    vmax_flash = (
        symmetric_vmax(slope_flash, pct=98, fallback=0.006)
        if VMAX_FLASH_TREND is None
        else float(VMAX_FLASH_TREND)
    )
    vmax_hail = (
        symmetric_vmax(slope_hail, pct=98, fallback=0.002)
        if VMAX_HAIL_TREND is None
        else float(VMAX_HAIL_TREND)
    )

    print("Flash slope min/max:", float(np.nanmin(slope_flash)), float(np.nanmax(slope_flash)))
    print("Hail  slope min/max:", float(np.nanmin(slope_hail)), float(np.nanmax(slope_hail)))
    print("Flash vmax for colourbar:", vmax_flash)
    print("Hail  vmax for colourbar:", vmax_hail)
    print("Flash significant grid count:", int(np.sum(np.isfinite(p_flash) & (p_flash < P_SIG))))
    print("Hail  significant grid count:", int(np.sum(np.isfinite(p_hail) & (p_hail < P_SIG))))

    np.savez_compressed(
        OUT_TREND_NPZ,
        slope_flash=slope_flash.astype(np.float32),
        slope_hail=slope_hail.astype(np.float32),
        p_flash=p_flash.astype(np.float32),
        p_hail=p_hail.astype(np.float32),
        lat=lat.astype(np.float32),
        lon=lon.astype(np.float32),
        strict_land=strict_land.astype(np.int8),
        high_alt=high_alt.astype(np.int8),
        elev=elev.astype(np.float32),
        line_labels=np.asarray(LINE_LABELS, dtype=object),
        line_scenarios=np.asarray(LINE_PERIOD_SCENARIOS, dtype=object),
        model_names=np.asarray(model_names, dtype=object),
        mmm_method=np.asarray("equal_weight_arithmetic_mean", dtype=object),
        intermodel_spread=np.asarray("regional_min_max_across_three_models", dtype=object),
        area_weighted_region_mean=np.asarray(AREA_WEIGHTED_REGION_MEAN, dtype=bool),
    )
    print("[SAVED]", OUT_TREND_NPZ)

    # --------------------------------------------------------
    # Regional statistics: compute each model first, then summarize
    # --------------------------------------------------------
    rows = []

    rows.extend(
        compute_region_means_by_model(
            stack_4d=flash_model_stack,
            model_names=model_names,
            lat=lat,
            lon=lon,
            strict_land=strict_land,
            high_alt=high_alt,
            var_name="flash",
            period_labels=line_period_labels,
            period_years=line_period_years,
            period_scenarios=line_period_scenarios,
        )
    )

    rows.extend(
        compute_region_means_by_model(
            stack_4d=hail_model_stack,
            model_names=model_names,
            lat=lat,
            lon=lon,
            strict_land=strict_land,
            high_alt=high_alt,
            var_name="hail",
            period_labels=line_period_labels,
            period_years=line_period_years,
            period_scenarios=line_period_scenarios,
        )
    )

    regional_model_df = pd.DataFrame(rows)
    regional_df = summarize_region_means(regional_model_df)

    regional_model_df.to_csv(OUT_REGIONAL_MODEL_CSV, index=False, encoding="utf-8-sig")
    regional_df.to_csv(OUT_REGIONAL_CSV, index=False, encoding="utf-8-sig")

    print("[SAVED]", OUT_REGIONAL_MODEL_CSV)
    print("[SAVED]", OUT_REGIONAL_CSV)
    print("\n[REGIONAL MODEL PREVIEW]")
    print(regional_model_df.head(20).to_string(index=False))
    print("\n[REGIONAL SUMMARY PREVIEW]")
    print(regional_df.head(20).to_string(index=False))

    # --------------------------------------------------------
    # Plot combined figure
    # --------------------------------------------------------
    cmap_trend = make_diverging_cmap()
    proj = ccrs.PlateCarree()

    fig = plt.figure(figsize=LAYOUT["figsize"], dpi=FIG_DPI)

    # Map axes.
    ax_map_flash = fig.add_axes(LAYOUT["ax_map_flash"], projection=proj)
    ax_map_hail = fig.add_axes(LAYOUT["ax_map_hail"], projection=proj)

    # Placeholder colour-bar axes; aligned after Cartopy has computed real map extents.
    cax_flash = fig.add_axes([0.05, 0.05, 0.10, 0.02])
    cax_hail = fig.add_axes([0.05, 0.02, 0.10, 0.02])

    # Placeholder line axes; aligned after Cartopy has computed real map extents.
    ax_line_flash = fig.add_axes([0.635, 0.210, 0.155, 0.700])
    ax_line_hail = fig.add_axes([0.815, 0.210, 0.155, 0.700])

    # Trend maps.
    m_flash = plot_trend_map(
        ax=ax_map_flash,
        slope=slope_flash,
        pval=p_flash,
        lat=lat,
        lon=lon,
        title="(a) Lightning Frequency",
        cmap=cmap_trend,
        vmax=vmax_flash,
    )
    m_hail = plot_trend_map(
        ax=ax_map_hail,
        slope=slope_hail,
        pval=p_hail,
        lat=lat,
        lon=lon,
        title="(b) Hail Frequency",
        cmap=cmap_trend,
        vmax=vmax_hail,
    )

    align_axes_colorbars_and_lines(
        fig=fig,
        ax_map_flash=ax_map_flash,
        ax_map_hail=ax_map_hail,
        cax_flash=cax_flash,
        cax_hail=cax_hail,
        ax_line_flash=ax_line_flash,
        ax_line_hail=ax_line_hail,
    )

    # Colour bars.
    cb_flash = fig.colorbar(m_flash, cax=cax_flash, orientation="horizontal")
    style_horizontal_colorbar(
        cb_flash,
        "Sen's Slope (Lightning Frequency per Decade)",
    )

    cb_hail = fig.colorbar(m_hail, cax=cax_hail, orientation="horizontal")
    style_horizontal_colorbar(
        cb_hail,
        "Sen's Slope (Hail Frequency per Decade)",
    )

    # Regional trajectories: MMM plus full min–max range across the three models.
    handles, labels = plot_regional_lines(
        ax=ax_line_flash,
        regional_df=regional_df,
        variable="flash",
        title="(c) Lightning Frequency",
        ylabel="Predicted Frequency",
    )
    plot_regional_lines(
        ax=ax_line_hail,
        regional_df=regional_df,
        variable="hail",
        title="(d) Hail Frequency",
        ylabel=None,
    )

    ax_line_flash.tick_params(axis="x", labelrotation=0)
    ax_line_hail.tick_params(axis="x", labelrotation=0)

    # Shared region legend. The caption should explain that whiskers are min–max intermodel range.
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=LAYOUT["legend_anchor"],
        ncol=LAYOUT["legend_ncol"],
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        handlelength=2.0,
        columnspacing=1.25,
        handletextpad=0.55,
        borderaxespad=0.0,
    )

    safe_savefig(fig, OUT_FIG, dpi=FIG_DPI)
    plt.close(fig)

    print("\n[SAVED FIGURE]")
    print(OUT_FIG)
    print("\n[DONE] Step 6 finished successfully.")


if __name__ == "__main__":
    main()
