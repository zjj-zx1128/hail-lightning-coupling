# Purpose: figure 03 joint distribution.
# Source: cmip6_figure_MMM_3models_joint2d_trend_with_markers.ipynb; see docs/source_coverage.csv for cell provenance.
# Run order and required inputs: docs/workflow.md.
# Run from repository root: python -m figures.figure_03_joint_distribution

# ============================================================
# Step 5. Plot BCC-CSM2-MR future Flash-Hail joint distribution figure
#
# 对应目标图：
#   Figure 3:
#   (a) Joint Distribution of Prediction Bias
#   (b) Predicted Two-dimensional Frequency
#   (c) Two-dimensional Frequency Difference
#
# 输入：
#   1. Step 3 输出的 pred_ssp585_*_*_2deg.nc
#   2. Step 1 输出的 train_table_flash_hail_withSRTM_strictLand.parquet
#
# 输出：
#   figure_03_joint_distribution.png
# ============================================================

from __future__ import annotations

from publication_config import resource_path

import os
import re
import glob
from pathlib import Path

from matplotlib.ticker import (
    LogLocator,
    NullLocator,
    MaxNLocator,
    FormatStrFormatter,
)

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.stats as stats

from matplotlib.ticker import LogLocator, NullLocator, MaxNLocator


# ============================================================
# 0. PATHS & CONFIG
# ============================================================

# ============================================================
# >>> MARKER: MMM_STEP5_USER_CONFIGURATION_START
# Purpose: three-model mean (MMM) future joint-distribution figure.
# Edit only these paths if your local directory layout differs.
# ============================================================

MODEL_NAME = "MMM (BCC-CSM2-MR + CanESM5 + MIROC6)"

MODEL_BASE_DIRS = {
    "BCC-CSM2-MR": Path(resource_path('models', 'BCC-CSM2-MR')),
    "CanESM5": Path(resource_path('models', 'CanESM5')),
    "MIROC6": Path(resource_path('models', 'MIROC6')),
}

# This model only supplies the common observed/reference grid and masks.
# It does NOT receive a higher weight in the MMM calculation.
REFERENCE_MODEL = "CanESM5"

OUT_ROOT = Path(resource_path('figures', ''))
OUT_DIR = OUT_ROOT / 'figure_03_joint_distribution'
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / 'figure_03_joint_distribution_mmm.jpg'
OUT_CSV_B = OUT_DIR / "fig_b_predicted_2d_frequency_multidecadal_mean_vectors_MMM.csv"
OUT_DEBUG_NPZ = OUT_DIR / "debug_predmean_obs_delta_grids_MMM.npz"

OBS_PARQ = (
    MODEL_BASE_DIRS[REFERENCE_MODEL]
    / "ml_feature_withSRTM"
    / "train_table_flash_hail_withSRTM_strictLand.parquet"
)

# nc variable names
VAR_FLASH = "pred_flash"
VAR_HAIL = "pred_hail"

# nc coordinate names
NC_LAT = "lat"
NC_LON = "lon"

# observed/reference table columns
LAT = "Latitude"
LON = "Longitude"
LANDSEA = "Land_Sea"
OBS_FLASH = "LISOTD_Flash"
OBS_HAIL = "ni_HailPF"

LAT_MAX = 63.0

DECADES = [
    (2050, 2059),
    (2060, 2069),
    (2070, 2079),
    (2080, 2089),
    (2090, 2099),
]

# Figure bins
NBIN_X = 70
NBIN_Y = 70
LEVELS_A = 18
ROBUST_PCT_DELTA = 99

FLASH_BINS = np.logspace(-4, 0, 50)
HAIL_BINS = np.logspace(-4, -1, 30)

XLIM = (1e-4, 1)
YLIM = (1e-4, 1e-1)

FREQ_LEVELS_B = 18
DIFF_LEVELS_C = 21
FIT_FIRST_N_HAIL_BINS = 25
ANNOTATION_FONTSIZE = 17

# Your NCL colourmap path. The code falls back to viridis if unavailable.
C_MAP_MAT = resource_path('colormaps', 'so4_21.mat')

# <<< MARKER: MMM_STEP5_USER_CONFIGURATION_END
# ============================================================
# 1. STYLE
# ============================================================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 18
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 600

TITLE_FONTSIZE = 25
LABEL_FONTSIZE = 20
TICK_FONTSIZE = 20
TEXT_FONTSIZE = 18
LEGEND_FONTSIZE = 16
CBAR_LABEL_FONTSIZE = 18
BORDER_WIDTH = 1.2

FIT_LINE_COLOR = "#1E90FF"
MEAN_LINE_COLOR = "black"


# ============================================================
# 2. COLORMAP HELPERS
# ============================================================

def array2cmap(X):
    N = X.shape[0]

    r = np.linspace(0.0, 1.0, N + 1)
    r = np.sort(np.concatenate((r, r)))[1:-1]

    rd = np.concatenate([[X[i, 0], X[i, 0]] for i in range(N)])
    gr = np.concatenate([[X[i, 1], X[i, 1]] for i in range(N)])
    bl = np.concatenate([[X[i, 2], X[i, 2]] for i in range(N)])

    rd = tuple([(r[i], rd[i], rd[i]) for i in range(2 * N)])
    gr = tuple([(r[i], gr[i], gr[i]) for i in range(2 * N)])
    bl = tuple([(r[i], bl[i], bl[i]) for i in range(2 * N)])

    cdict = {
        "red": rd,
        "green": gr,
        "blue": bl,
    }

    return matplotlib.colors.LinearSegmentedColormap("my_colormap", cdict, N)


def load_ncl_mat_cmap(mat_path: str, key: str = "cmap", reverse: bool = True):
    if not os.path.exists(mat_path):
        print(f"[WARN] C_MAP_MAT not found, use viridis instead: {mat_path}")
        return plt.get_cmap("viridis")

    try:
        import scipy.io as scio
        a = scio.loadmat(mat_path)
        X = np.array(a[key])

        if reverse:
            X = X[::-1]

        return array2cmap(X)

    except Exception as e:
        print(f"[WARN] Failed to load NCL cmap, use viridis instead: {e}")
        return plt.get_cmap("viridis")


def diverging_with_gray_center(gray=0.95, N=256):
    cols = [
        (0.23, 0.30, 0.75),
        (gray, gray, gray),
        (0.75, 0.20, 0.20),
    ]

    return mcolors.LinearSegmentedColormap.from_list(
        "blue_gray_red",
        cols,
        N=N,
    )


# ============================================================
# 3. BASIC HELPERS
# ============================================================

def parse_decade_from_name(path: str) -> tuple[int, int]:
    b = os.path.basename(path)
    m = re.search(r"_(\d{4})_(\d{4})_", b)

    if not m:
        raise ValueError(f"Cannot parse decade from filename: {b}")

    return int(m.group(1)), int(m.group(2))


def robust_symmetric_range(v, pct=99):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]

    if v.size == 0:
        return -1.0, 1.0

    vmax = np.percentile(np.abs(v), pct)
    vmax = float(vmax) if np.isfinite(vmax) and vmax > 0 else float(np.max(np.abs(v)))

    if vmax <= 0:
        vmax = 1.0

    return -vmax, vmax


def _prep_pos(a):
    a = np.asarray(a, float).copy()
    a[a <= 0] = np.nan
    return a


def hist2d_counts(x, y, xbins, ybins):
    H, xedges, yedges = np.histogram2d(x, y, bins=[xbins, ybins])
    return H, xedges, yedges


def wrap_lon180(lon):
    lon = np.asarray(lon, dtype=float)
    return ((lon + 180.0) % 360.0) - 180.0


# ============================================================
# 4. AXIS STYLE HELPERS
# ============================================================

def style_linear_axes(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(BORDER_WIDTH)
        spine.set_color("black")

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        bottom=True,
        left=True,
        top=False,
        right=False,
        length=6,
        width=1.2,
        labelsize=TICK_FONTSIZE,
    )

    ax.grid(
        True,
        which="major",
        linestyle="--",
        linewidth=0.7,
        alpha=0.55,
        color="0.65",
        zorder=20,
    )


def style_log_axes(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(BORDER_WIDTH)
        spine.set_zorder(30)

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        labelsize=TICK_FONTSIZE,
        color="black",
        width=1.2,
        length=6,
        top=False,
        right=False,
        bottom=True,
        left=True,
    )

    ax.tick_params(
        axis="both",
        which="minor",
        bottom=False,
        left=False,
        top=False,
        right=False,
    )

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=10))
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=10))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())

    ax.set_axisbelow(False)

    ax.grid(
        True,
        which="major",
        linestyle="--",
        linewidth=0.7,
        alpha=0.65,
        color="0.65",
        zorder=20,
    )


def style_colorbar(cbar):
    cbar.ax.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_FONTSIZE,
        color="black",
        width=1.0,
        length=5,
    )


# ============================================================
# 5. DATA LOADING
# ============================================================

def load_nc_decade(path_nc: str):
    with xr.open_dataset(path_nc) as ds:
        if VAR_FLASH not in ds:
            raise KeyError(f"{VAR_FLASH} not found in {path_nc}")

        if VAR_HAIL not in ds:
            raise KeyError(f"{VAR_HAIL} not found in {path_nc}")

        if NC_LAT not in ds:
            raise KeyError(f"{NC_LAT} not found in {path_nc}")

        if NC_LON not in ds:
            raise KeyError(f"{NC_LON} not found in {path_nc}")

        flash = ds[VAR_FLASH].values.astype(np.float64)
        hail = ds[VAR_HAIL].values.astype(np.float64)
        lat = ds[NC_LAT].values.astype(np.float64)
        lon = ds[NC_LON].values.astype(np.float64)

    lat_ok = np.abs(lat) <= float(LAT_MAX)

    if lat_ok.sum() != len(lat):
        flash = flash[lat_ok, :]
        hail = hail[lat_ok, :]
        lat = lat[lat_ok]

    lon = wrap_lon180(lon)

    # 确保 lon 从 -180 到 178 排序
    order = np.argsort(lon)
    lon = lon[order]
    flash = flash[:, order]
    hail = hail[:, order]

    return flash, hail, lat, lon



# ============================================================
# >>> MARKER: MMM_STEP5_MULTI_MODEL_MEAN_CORE_START
# Scientific definition:
# (1) At each grid cell and decade, take the arithmetic mean of the
#     three model predictions.
# (2) Then take the arithmetic mean across the five future decades.
# This gives each model and each decade equal weight.
# ============================================================

def get_model_nc_path(base_dir: Path, y0: int, y1: int) -> Path:
    return (
        base_dir
        / "ml_xgboost_result"
        / 'FINAL_XGB'
        / "PRED_SSP585"
        / "nc"
        / f"pred_ssp585_{y0}_{y1}_2deg.nc"
    )


def load_nc_decade_from_model(base_dir: Path, model_name: str, y0: int, y1: int):
    path_nc = get_model_nc_path(base_dir, y0, y1)

    if not path_nc.exists():
        raise FileNotFoundError(f"[{model_name}] missing SSP585 prediction file: {path_nc}")

    flash, hail, lat, lon = load_nc_decade(str(path_nc))
    print(f"  [LOAD] {model_name:<12s} | {y0}-{y1} | shape={flash.shape}")
    return flash, hail, lat, lon


def compute_mmm_multidecadal_mean_prediction():
    """Return the multi-model / multi-decadal mean Flash and Hail fields."""
    decade_flash_mmm = []
    decade_hail_mmm = []
    nmodel_flash = []
    nmodel_hail = []

    lat_ref = None
    lon_ref = None

    for y0, y1 in DECADES:
        print(f"\n[MMM DECADE] {y0}-{y1}")
        flash_models = []
        hail_models = []

        for model_name, base_dir in MODEL_BASE_DIRS.items():
            flash, hail, lat, lon = load_nc_decade_from_model(base_dir, model_name, y0, y1)

            if lat_ref is None:
                lat_ref = lat
                lon_ref = lon
            else:
                if len(lat) != len(lat_ref) or len(lon) != len(lon_ref):
                    raise ValueError(f"[{model_name}] grid shape mismatch in {y0}-{y1}")
                if not np.allclose(lat, lat_ref, equal_nan=True):
                    raise ValueError(f"[{model_name}] latitude mismatch in {y0}-{y1}")
                if not np.allclose(lon, lon_ref, equal_nan=True):
                    raise ValueError(f"[{model_name}] longitude mismatch in {y0}-{y1}")

            flash_models.append(flash)
            hail_models.append(hail)

        flash_model_stack = np.stack(flash_models, axis=0)
        hail_model_stack = np.stack(hail_models, axis=0)

        decade_flash_mmm.append(np.nanmean(flash_model_stack, axis=0))
        decade_hail_mmm.append(np.nanmean(hail_model_stack, axis=0))
        nmodel_flash.append(np.sum(np.isfinite(flash_model_stack), axis=0))
        nmodel_hail.append(np.sum(np.isfinite(hail_model_stack), axis=0))

    stack_flash_mmm = np.stack(decade_flash_mmm, axis=0)
    stack_hail_mmm = np.stack(decade_hail_mmm, axis=0)

    pred_mean_flash = np.nanmean(stack_flash_mmm, axis=0)
    pred_mean_hail = np.nanmean(stack_hail_mmm, axis=0)

    diagnostics = {
        "decadal_flash_mmm": stack_flash_mmm,
        "decadal_hail_mmm": stack_hail_mmm,
        "nmodel_flash_by_decade": np.stack(nmodel_flash, axis=0),
        "nmodel_hail_by_decade": np.stack(nmodel_hail, axis=0),
    }

    return pred_mean_flash, pred_mean_hail, lat_ref, lon_ref, DECADES, diagnostics

# <<< MARKER: MMM_STEP5_MULTI_MODEL_MEAN_CORE_END

def build_grid_from_parquet(lat, lon, parquet_path: str | Path, value_col: str) -> np.ndarray:
    parquet_path = str(parquet_path)

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(parquet_path)

    df = pd.read_parquet(parquet_path)

    need = {LAT, LON, value_col}
    miss = need - set(df.columns)

    if miss:
        raise KeyError(f"Parquet missing columns: {sorted(list(miss))}")

    d = df.copy()

    if LANDSEA in d.columns:
        d = d[d[LANDSEA] == 0]

    d = d[np.abs(d[LAT].astype(float)) <= float(LAT_MAX)]

    d[LAT] = d[LAT].astype(float)
    d[LON] = wrap_lon180(d[LON].astype(float).to_numpy())

    d = d.groupby([LAT, LON], as_index=False)[[value_col]].mean()

    lat_axis = np.round(np.asarray(lat, dtype=float), 6)
    lon_axis = np.round(np.asarray(lon, dtype=float), 6)

    lat_to_i = {v: i for i, v in enumerate(lat_axis)}
    lon_to_j = {v: j for j, v in enumerate(lon_axis)}

    dlat = np.round(d[LAT].to_numpy(float), 6)
    dlon = np.round(d[LON].to_numpy(float), 6)

    ok = np.isin(dlat, lat_axis) & np.isin(dlon, lon_axis)

    d = d.loc[ok].copy()

    grid = np.full((len(lat_axis), len(lon_axis)), np.nan, dtype=np.float32)

    if d.empty:
        return grid

    ii = d[LAT].round(6).map(lat_to_i).to_numpy()
    jj = d[LON].round(6).map(lon_to_j).to_numpy()

    grid[ii, jj] = d[value_col].to_numpy(dtype=np.float32)

    return grid


# ============================================================
# 6. PANEL A
# ============================================================

def draw_panel_a(ax, cax, pred_mean_f, pred_mean_h, obs_f, obs_h, cmap_freq):
    dflash2d = pred_mean_f - obs_f
    dhail2d = pred_mean_h - obs_h

    valid = np.isfinite(dflash2d) & np.isfinite(dhail2d)

    x = dflash2d[valid].ravel()
    y = dhail2d[valid].ravel()

    print("[PANEL A] valid samples =", x.size)

    xmin, xmax = robust_symmetric_range(x, pct=ROBUST_PCT_DELTA)
    ymin, ymax = robust_symmetric_range(y, pct=ROBUST_PCT_DELTA)

    xbins = np.linspace(xmin, xmax, NBIN_X + 1)
    ybins = np.linspace(ymin, ymax, NBIN_Y + 1)

    H, xedges, yedges = hist2d_counts(x, y, xbins, ybins)

    Xg, Yg = np.meshgrid(xedges[:-1], yedges[:-1])

    cf = ax.contourf(
        Xg,
        Yg,
        H.T,
        levels=LEVELS_A,
        cmap=cmap_freq,
        zorder=1,
    )

    cbar = plt.colorbar(cf, cax=cax)
    cbar.set_label("Grid Count", fontsize=CBAR_LABEL_FONTSIZE)
    style_colorbar(cbar)

    ax.axvline(0, color="k", linewidth=1.0, alpha=0.6, zorder=10)
    ax.axhline(0, color="k", linewidth=1.0, alpha=0.6, zorder=10)

    N = x.size

    if N > 0:
        q1 = np.sum((x > 0) & (y > 0))
        q2 = np.sum((x <= 0) & (y > 0))
        q3 = np.sum((x <= 0) & (y <= 0))
        q4 = np.sum((x > 0) & (y <= 0))

        p1, p2, p3, p4 = 100 * q1 / N, 100 * q2 / N, 100 * q3 / N, 100 * q4 / N

        txt_kw = dict(
            transform=ax.transAxes,
            fontsize=22,
            ha="center",
            va="center",
            zorder=25,
        )

        ax.text(0.85, 0.85, f"Q1\n{p1:.2f}%", **txt_kw)
        ax.text(0.15, 0.85, f"Q2\n{p2:.2f}%", **txt_kw)
        ax.text(0.15, 0.15, f"Q3\n{p3:.2f}%", **txt_kw)
        ax.text(0.85, 0.15, f"Q4\n{p4:.2f}%", **txt_kw)

    ax.set_xlabel("ΔLightning Frequency (Decadal − Observed)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("ΔHail Frequency (Decadal − Observed)", fontsize=LABEL_FONTSIZE)

    ax.set_xlim(-0.02, 0.02)
    ax.set_ylim(-0.01, 0.01)
    ax.set_yticks([-0.010, -0.005, 0.000, 0.005, 0.010])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    ax.set_title(
        "(a) Joint Distribution of Projected Change",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
        pad=19,
    )

    style_linear_axes(ax)


# ============================================================
# 7. PANEL B
# ============================================================

def draw_panel_b(
    ax,
    cax,
    flash_pred,
    hail_pred,
    cmap_freq,
    flash_bins=FLASH_BINS,
    hail_bins=HAIL_BINS,
    levels=FREQ_LEVELS_B,
    fit_first_n_hail_bins=FIT_FIRST_N_HAIL_BINS,
    xlim=XLIM,
    ylim=YLIM,
):
    flash = _prep_pos(flash_pred)
    hail = _prep_pos(hail_pred)

    valid = np.isfinite(flash) & np.isfinite(hail)

    flash = flash[valid]
    hail = hail[valid]

    print("[PANEL B] valid predicted samples =", flash.size)

    # ============================================================
    # All valid predicted grid cells: log10-log10 R²
    # This is used only for the R² text annotation.
    # ============================================================
    log_flash_all = np.log10(flash)
    log_hail_all = np.log10(hail)

    if flash.size >= 3:
        _, _, r_value_all, p_value_all, _ = stats.linregress(
            log_hail_all,
            log_flash_all,
        )
        r2_all_loglog = r_value_all ** 2
    else:
        r2_all_loglog = np.nan
        p_value_all = np.nan

    print(
        "[PANEL B] all-point log-log R² = "
        f"{r2_all_loglog:.4f}, "
        f"N = {flash.size}, "
        f"p = {p_value_all:.3e}"
    )

    if len(flash) < 10:
        raise ValueError("Too few valid predicted samples for panel B.")

    hist, xedges, yedges = np.histogram2d(
        flash,
        hail,
        bins=[flash_bins, hail_bins],
    )

    xcent = np.sqrt(xedges[:-1] * xedges[1:])
    ycent = np.sqrt(yedges[:-1] * yedges[1:])

    Xc, Yc = np.meshgrid(xcent, ycent)

    hail_centers = (hail_bins[:-1] + hail_bins[1:]) / 2

    flash_log_means = []
    flash_log_stds = []

    for i in range(len(hail_bins) - 1):
        m = (hail >= hail_bins[i]) & (hail < hail_bins[i + 1])
        f_in = flash[m]

        if len(f_in) > 0:
            lv = np.log10(f_in)
            flash_log_means.append(np.mean(lv))

            if len(lv) > 1:
                flash_log_stds.append(np.std(lv, ddof=1))
            else:
                flash_log_stds.append(np.nan)
        else:
            flash_log_means.append(np.nan)
            flash_log_stds.append(np.nan)

    flash_log_means = np.array(flash_log_means)
    flash_log_stds = np.array(flash_log_stds)

    flash_mean = 10 ** flash_log_means
    flash_upper = 10 ** (flash_log_means + flash_log_stds)
    flash_lower = 10 ** (flash_log_means - flash_log_stds)

    valid_idx = np.isfinite(flash_log_means)[:fit_first_n_hail_bins]

    if int(valid_idx.sum()) >= 3:
        hail_fit = np.log10(hail_centers[:fit_first_n_hail_bins])[valid_idx]
        flash_fit = flash_log_means[:fit_first_n_hail_bins][valid_idx]

        slope, intercept, r_value, p_value, std_err = stats.linregress(
            hail_fit,
            flash_fit,
        )

    else:
        slope = np.nan
        intercept = np.nan

    contour = ax.contourf(
        Xc,
        Yc,
        hist.T,
        levels=levels,
        cmap=cmap_freq,
        zorder=1,
    )

    cbar = plt.colorbar(contour, cax=cax)
    cbar.set_label("Grid Count", fontsize=CBAR_LABEL_FONTSIZE)
    style_colorbar(cbar)

    ax.plot(
        flash_mean,
        hail_centers,
        color=MEAN_LINE_COLOR,
        linestyle="-",
        linewidth=1.8,
        label="Mean Lightning Frequency",
        zorder=10,
    )

    ax.plot(
        flash_upper,
        hail_centers,
        color=MEAN_LINE_COLOR,
        linestyle="--",
        linewidth=1.5,
        label="Mean ± 1 SD",
        zorder=10,
    )

    ax.plot(
        flash_lower,
        hail_centers,
        color=MEAN_LINE_COLOR,
        linestyle="--",
        linewidth=1.5,
        zorder=10,
    )

    if np.isfinite(slope) and np.isfinite(intercept):
        y_min, y_max = ylim

        hail_ext = np.logspace(
            np.log10(y_min),
            np.log10(y_max),
            200,
        )

        flash_fit_ext = 10 ** (
            slope * np.log10(hail_ext) + intercept
        )

        ax.plot(
            flash_fit_ext,
            hail_ext,
            linestyle="-.",
            linewidth=2.2,
            color=FIT_LINE_COLOR,
            label="Fitted Mean",
            zorder=12,
        )

        eq_str = (
            r"$\log_{10}(\mathrm{Lightning}) = "
            f"{slope:.2f}"
            r"\cdot \log_{10}(\mathrm{Hail}) "
            f"{intercept:+.2f}$"
        )

        r_str = (
            rf"$\mathrm{{log}}_{{10}}$: "
            rf"$R^2 = {r2_all_loglog:.2f}$"
        )

        ax.text(
            0.02,
            0.98,
            eq_str,
            transform=ax.transAxes,
            fontsize=ANNOTATION_FONTSIZE,
            va="top",
            zorder=25,
        )

        ax.text(
            0.02,
            0.91,
            r_str,
            transform=ax.transAxes,
            fontsize=ANNOTATION_FONTSIZE,
            va="top",
            zorder=25,
        )
    else:
        ax.text(
            0.02,
            0.98,
            "Fitted Mean: insufficient valid bins",
            transform=ax.transAxes,
            fontsize=TEXT_FONTSIZE,
            va="top",
            zorder=25,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xlabel("Lightning Frequency", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Hail Frequency", fontsize=LABEL_FONTSIZE)

    ax.set_title(
        "(b) Predicted Two-dimensional Frequency",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
        pad=19,
    )

    ax.legend(
        loc="lower right",
        fontsize=LEGEND_FONTSIZE,
        frameon=False,
    )

    style_log_axes(ax)


# ============================================================
# 8. PANEL C
# ============================================================

def draw_panel_c(ax, cax, pred_mean_f, pred_mean_h, obs_f, obs_h, cmap_diff):
    def _prep_pair(a2d, b2d):
        a = np.asarray(a2d, float).ravel()
        b = np.asarray(b2d, float).ravel()

        a[a <= 0] = np.nan
        b[b <= 0] = np.nan

        m = np.isfinite(a) & np.isfinite(b)

        return a[m], b[m]

    fp, hp = _prep_pair(pred_mean_f, pred_mean_h)
    fo, ho = _prep_pair(obs_f, obs_h)

    print("[PANEL C] valid pred samples =", fp.size)
    print("[PANEL C] valid obs samples  =", fo.size)

    if fp.size < 10 or fo.size < 10:
        raise ValueError("Too few valid samples for panel C.")

    Hp, xedges, yedges = hist2d_counts(fp, hp, FLASH_BINS, HAIL_BINS)
    Ho, _, _ = hist2d_counts(fo, ho, FLASH_BINS, HAIL_BINS)

    D = np.full_like(Hp, np.nan, dtype=float)
    mask = ~((Hp == 0) & (Ho == 0))
    D[mask] = Hp[mask] - Ho[mask]

    xcent = np.sqrt(xedges[:-1] * xedges[1:])
    ycent = np.sqrt(yedges[:-1] * yedges[1:])

    Xc, Yc = np.meshgrid(xcent, ycent)

    vmax = float(np.nanmax(np.abs(D)))

    if (not np.isfinite(vmax)) or vmax == 0:
        vmax = 1e-6

    lev = np.linspace(-vmax, vmax, DIFF_LEVELS_C)

    contour = ax.contourf(
        Xc,
        Yc,
        D.T,
        levels=lev,
        cmap=cmap_diff,
        extend="both",
        zorder=1,
    )

    cbar = plt.colorbar(contour, cax=cax)
    cbar.set_label(
        "ΔGrid Count (Decadal − Observed)",
        fontsize=CBAR_LABEL_FONTSIZE,
    )

    cbar.locator = MaxNLocator(integer=True)
    cbar.update_ticks()
    style_colorbar(cbar)

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)

    ax.set_xlabel("Lightning Frequency", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Hail Frequency", fontsize=LABEL_FONTSIZE)

    ax.set_title(
        "(c) Two-dimensional Frequency Difference",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
        pad=19,
    )

    style_log_axes(ax)


# ============================================================
# 9. MAIN
# ============================================================

def main():
    # >>> MARKER: MMM_STEP5_INPUT_VALIDATION_START
    for model_name, base_dir in MODEL_BASE_DIRS.items():
        if not base_dir.exists():
            raise FileNotFoundError(f"[{model_name}] base directory not found: {base_dir}")

    if not OBS_PARQ.exists():
        raise FileNotFoundError(f"OBS_PARQ not found: {OBS_PARQ}")

    print("=" * 100)
    print("[STEP 5 MMM] Plot future flash-hail joint distribution")
    print("=" * 100)

    print("[STEP1] Compute MMM multi-decadal mean prediction from SSP585 nc files...")
    (
        pred_mean_flash,
        pred_mean_hail,
        lat,
        lon,
        picked_decades,
        mmm_diagnostics,
    ) = compute_mmm_multidecadal_mean_prediction()
    # <<< MARKER: MMM_STEP5_INPUT_VALIDATION_END

    print("[INFO] pred_mean_flash shape:", pred_mean_flash.shape)
    print("[INFO] pred_mean_hail  shape:", pred_mean_hail.shape)
    print("[INFO] valid flash count:", int(np.isfinite(pred_mean_flash).sum()))
    print("[INFO] valid hail  count:", int(np.isfinite(pred_mean_hail).sum()))

    print("[STEP2] Build observed grids aligned to prediction grid...")
    obs_flash = build_grid_from_parquet(lat, lon, OBS_PARQ, OBS_FLASH)
    obs_hail = build_grid_from_parquet(lat, lon, OBS_PARQ, OBS_HAIL)

    print("[INFO] observed flash valid count:", int(np.isfinite(obs_flash).sum()))
    print("[INFO] observed hail  valid count:", int(np.isfinite(obs_hail).sum()))

    np.savez_compressed(
        OUT_DEBUG_NPZ,
        pred_mean_flash=pred_mean_flash.astype(np.float32),
        pred_mean_hail=pred_mean_hail.astype(np.float32),
        obs_flash=obs_flash.astype(np.float32),
        obs_hail=obs_hail.astype(np.float32),
        lat=lat.astype(np.float32),
        lon=lon.astype(np.float32),
        decadal_flash_mmm=mmm_diagnostics["decadal_flash_mmm"].astype(np.float32),
        decadal_hail_mmm=mmm_diagnostics["decadal_hail_mmm"].astype(np.float32),
        nmodel_flash_by_decade=mmm_diagnostics["nmodel_flash_by_decade"].astype(np.int8),
        nmodel_hail_by_decade=mmm_diagnostics["nmodel_hail_by_decade"].astype(np.int8),
        model_names=np.asarray(list(MODEL_BASE_DIRS.keys()), dtype=object),
    )

    print("[SAVED]", OUT_DEBUG_NPZ)

    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")

    save_df = pd.DataFrame({
        "Latitude": lat2d.ravel(),
        "Longitude": lon2d.ravel(),
        "flash_pred_multidecadal_mean": pred_mean_flash.ravel(),
        "hail_pred_multidecadal_mean": pred_mean_hail.ravel(),
    })

    save_df.to_csv(OUT_CSV_B, index=False, encoding="utf-8-sig")
    print("[SAVED]", OUT_CSV_B)

    cmap_freq = load_ncl_mat_cmap(C_MAP_MAT, key="cmap", reverse=True)
    cmap_diff = diverging_with_gray_center(gray=0.95)

    # =====================================================
    # Fixed-layout 1 × 3 figure
    # =====================================================

    fig = plt.figure(figsize=(23.5, 6.2), dpi=600)

    outer_gs = fig.add_gridspec(
        nrows=1,
        ncols=3,
        left=0.07,
        right=0.95,
        bottom=0.14,
        top=0.90,
        wspace=0.35,
    )

    gs_a = outer_gs[0, 0].subgridspec(
        nrows=1,
        ncols=2,
        width_ratios=[1.0, 0.045],
        wspace=0.03,
    )

    gs_b = outer_gs[0, 1].subgridspec(
        nrows=1,
        ncols=2,
        width_ratios=[1.0, 0.045],
        wspace=0.03,
    )

    gs_c = outer_gs[0, 2].subgridspec(
        nrows=1,
        ncols=2,
        width_ratios=[1.0, 0.045],
        wspace=0.03,
    )

    ax_a = fig.add_subplot(gs_a[0, 0])
    cax_a = fig.add_subplot(gs_a[0, 1])

    ax_b = fig.add_subplot(gs_b[0, 0])
    cax_b = fig.add_subplot(gs_b[0, 1])

    ax_c = fig.add_subplot(gs_c[0, 0])
    cax_c = fig.add_subplot(gs_c[0, 1])

    draw_panel_a(
        ax=ax_a,
        cax=cax_a,
        pred_mean_f=pred_mean_flash,
        pred_mean_h=pred_mean_hail,
        obs_f=obs_flash,
        obs_h=obs_hail,
        cmap_freq=cmap_freq,
    )

    draw_panel_b(
        ax=ax_b,
        cax=cax_b,
        flash_pred=pred_mean_flash,
        hail_pred=pred_mean_hail,
        cmap_freq=cmap_freq,
        flash_bins=FLASH_BINS,
        hail_bins=HAIL_BINS,
        levels=FREQ_LEVELS_B,
        fit_first_n_hail_bins=FIT_FIRST_N_HAIL_BINS,
        xlim=XLIM,
        ylim=YLIM,
    )

    draw_panel_c(
        ax=ax_c,
        cax=cax_c,
        pred_mean_f=pred_mean_flash,
        pred_mean_h=pred_mean_hail,
        obs_f=obs_flash,
        obs_h=obs_hail,
        cmap_diff=cmap_diff,
    )

    fig.savefig(OUT_PNG, dpi=600)
    plt.close(fig)

    print("\n[SAVED]", OUT_PNG)
    print("[DONE] Decades used:", picked_decades)
    print("[DONE] OUT_DIR:", OUT_DIR)


if __name__ == "__main__":
    main()