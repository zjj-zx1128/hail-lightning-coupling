# Purpose: figure s09 s10 historical joint.
# Source: cmip6_figureS_joint_hist.ipynb; see docs/source_coverage.csv for cell provenance.
# Run order and required inputs: docs/workflow.md.
# Run from repository root: python -m figures.figure_s09_s10_historical_joint

# ============================================================
# Step 9. Historical observed-versus-reconstruction joint-frequency figures
#
# Purpose
# -------
# Diagnose how well the historical XGBoost reconstructions reproduce the
# observed Lightning–Hail joint distribution for CanESM5, MIROC6,
# BCC-CSM2-MR, and their equal-weight three-model mean (MMM).
#
# IMPORTANT SCIENTIFIC DEFINITIONS
# --------------------------------
# 1. Each model is reconstructed from its own historical strict-land training
#    table using its saved FLASH and HAIL XGBoost models.
# 2. Negative predictions are clipped to zero before any diagnostic.
# 3. MMM is the arithmetic mean of the three model predictions at grid cells
#    where all three model predictions are finite. Each model receives equal
#    weight.
# 4. The R² annotated in panels (a) and (b) follows Main Figure 3 exactly:
#    R² is calculated from ALL valid positive grid cells in log10 space,
#    i.e., corr[log10(Hail), log10(Lightning)]². It is a coupling-structure
#    metric, not an observed-versus-predicted performance R².
# 5. The black mean and ±1 SD curves and the blue fitted-mean curve retain
#    the Main Figure 3 convention: the fitted-mean line uses the first
#    FIT_FIRST_N_HAIL_BINS Hail bins.
#
# Figure panels for each model / MMM
# ----------------------------------
# (a) Observed Two-dimensional Frequency
# (b) Historical Reconstruction Two-dimensional Frequency
# (c) Two-dimensional Frequency Difference (Reconstruction − Observed)
#
# The figure geometry, GridSpec construction, spacing, typography, colourbar
# placement, and axis style intentionally follow the Main Figure 3 code.
# ============================================================

from __future__ import annotations

from publication_config import resource_path

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, MaxNLocator, NullLocator
from scipy import stats


# ============================================================
# 0. USER CONFIGURATION
# ============================================================

MODEL_ORDER = ["CanESM5", "MIROC6", "BCC-CSM2-MR"]
REFERENCE_MODEL = "CanESM5"  # Supplies observed values for the MMM merge.

# Keep these source paths consistent with the three existing single-model
# Step 9 scripts supplied by the user.
MODEL_CONFIG: dict[str, dict[str, Path]] = {
    "CanESM5": {
        "base_dir": Path(resource_path('models', 'CanESM5')),
        "train_table": Path(
            resource_path('models', 'CanESM5/ml_feature_withSRTM/train_table_flash_hail_withSRTM_strictLand.parquet')
        ),
    },
    "MIROC6": {
        "base_dir": Path(resource_path('models', 'MIROC6')),
        "train_table": Path(
            resource_path('models', 'MIROC6/feature/ml_train_historical/train_table_flash_hail_withSRTM_strictLand.parquet')
        ),
    },
    "BCC-CSM2-MR": {
        # The local disk directory intentionally retains the user's existing
        # BCC-CSM2-MR spelling; the displayed scientific name remains
        # BCC-CSM2-MR everywhere in the figure.
        "base_dir": Path(resource_path('models', 'BCC-CSM2-MR')),
        "train_table": Path(
            resource_path('models', 'BCC-CSM2-MR/feature/ml_train_historical/train_table_flash_hail_withSRTM_strictLand.parquet')
        ),
    },
}

OUT_DIR = Path(resource_path('figures', 'figure_s09_s10_historical_joint'))
FIGURE_PREFIX = 'figure_s09_s10_historical_joint'

LAT_COL = "Latitude"
LON_COL = "Longitude"
LAND_COL = "Land_Sea"
OBS_FLASH = "LISOTD_Flash"
OBS_HAIL = "ni_HailPF"
LAT_MAX = 63.0

USE_BEST_ITERATION = True

# Bins and plotting ranges are exactly aligned with Main Figure 3.
FLASH_BINS = np.logspace(-4, 0, 50)
HAIL_BINS = np.logspace(-4, -1, 30)
XLIM = (1e-4, 1.0)
YLIM = (1e-4, 1e-1)

FREQ_LEVELS = 18
DIFF_LEVELS = 21
FIT_FIRST_N_HAIL_BINS = 25
ANNOTATION_FONTSIZE = 17

# Same NCL colormap used by Main Figure 3. The script falls back to viridis.
C_MAP_MAT = resource_path('colormaps', 'so4_21.mat')


# ============================================================
# 1. STYLE — MATCH MAIN FIGURE 3
# ============================================================

FIG_DPI = 600

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 18
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = FIG_DPI

TITLE_FONTSIZE = 25
LABEL_FONTSIZE = 20
TICK_FONTSIZE = 20
TEXT_FONTSIZE = 18
LEGEND_FONTSIZE = 16
CBAR_LABEL_FONTSIZE = 18
BORDER_WIDTH = 1.2

FIT_LINE_COLOR = "#1E90FF"
MEAN_LINE_COLOR = "black"

# This is intentionally the same outer GridSpec layout used by Main Figure 3.
# Adjust these values only when the Main Figure 3 layout itself is changed.
MAIN_FIGURE3_LAYOUT = {
    "figsize": (23.5, 6.2),
    "left": 0.07,
    "right": 0.95,
    "bottom": 0.14,
    "top": 0.90,
    "wspace": 0.35,
    "panel_cbar_width_ratio": 0.045,
    "panel_cbar_wspace": 0.03,
}


# ============================================================
# 2. COLORMAP HELPERS — MATCH MAIN FIGURE 3
# ============================================================

def array2cmap(X: np.ndarray):
    """Convert an N×3 RGB array to a continuous Matplotlib colormap."""
    N = X.shape[0]

    r = np.linspace(0.0, 1.0, N + 1)
    r = np.sort(np.concatenate((r, r)))[1:-1]

    rd = np.concatenate([[X[i, 0], X[i, 0]] for i in range(N)])
    gr = np.concatenate([[X[i, 1], X[i, 1]] for i in range(N)])
    bl = np.concatenate([[X[i, 2], X[i, 2]] for i in range(N)])

    rd = tuple((r[i], rd[i], rd[i]) for i in range(2 * N))
    gr = tuple((r[i], gr[i], gr[i]) for i in range(2 * N))
    bl = tuple((r[i], bl[i], bl[i]) for i in range(2 * N))

    cdict = {"red": rd, "green": gr, "blue": bl}
    return matplotlib.colors.LinearSegmentedColormap("my_colormap", cdict, N)


def load_ncl_mat_cmap(mat_path: str, key: str = "cmap", reverse: bool = True):
    """Load the NCL colourmap used by Main Figure 3, with a safe fallback."""
    if not os.path.exists(mat_path):
        print(f"[WARN] C_MAP_MAT not found; using viridis: {mat_path}")
        return plt.get_cmap("viridis")

    try:
        import scipy.io as scio

        obj = scio.loadmat(mat_path)
        rgb = np.asarray(obj[key])
        if reverse:
            rgb = rgb[::-1]
        return array2cmap(rgb)
    except Exception as exc:
        print(f"[WARN] Failed to load NCL colourmap; using viridis: {exc}")
        return plt.get_cmap("viridis")


def diverging_with_gray_center(gray: float = 0.95, N: int = 256):
    """Blue–gray–red diverging colormap used in Main Figure 3 panel (c)."""
    return mcolors.LinearSegmentedColormap.from_list(
        "blue_gray_red",
        [(0.23, 0.30, 0.75), (gray, gray, gray), (0.75, 0.20, 0.20)],
        N=N,
    )


# ============================================================
# 3. COMMON HELPERS
# ============================================================

def ensure_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(
            f"{label} is missing required columns: "
            f"{missing[:30]}{' ...' if len(missing) > 30 else ''}"
        )


def read_feature_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        features = [line.strip() for line in f if line.strip()]

    if not features:
        raise ValueError(f"Empty feature list: {path}")

    return features


def read_best_iteration(path: Path) -> int | None:
    """Read the saved best tree iteration, if it exists."""
    if not path.exists():
        print(f"[WARN] Summary JSON not found; use all trees: {path}")
        return None

    with path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    try:
        return int(summary["metrics"]["best_iteration"])
    except Exception:
        print(f"[WARN] Cannot read best_iteration from: {path}")
        return None


def load_booster(path: Path) -> xgb.Booster:
    if not path.exists():
        raise FileNotFoundError(path)

    booster = xgb.Booster()
    booster.load_model(str(path))
    return booster


def predict_with_booster(
    df: pd.DataFrame,
    features: list[str],
    booster: xgb.Booster,
    best_iteration: int | None,
    target_name: str,
) -> np.ndarray:
    ensure_columns(df, features, f"{target_name} feature matrix")

    X = df[features].copy().replace([np.inf, -np.inf], np.nan)
    dmatrix = xgb.DMatrix(X, feature_names=features, missing=np.nan)

    if USE_BEST_ITERATION and best_iteration is not None:
        pred = booster.predict(dmatrix, iteration_range=(0, best_iteration + 1))
    else:
        pred = booster.predict(dmatrix)

    return np.asarray(pred, dtype=np.float64)


def prep_positive_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return finite positive Lightning/Hail pairs for log-space diagnostics."""
    x = np.asarray(x, dtype=float).copy()
    y = np.asarray(y, dtype=float).copy()
    x[x <= 0] = np.nan
    y[y <= 0] = np.nan
    valid = np.isfinite(x) & np.isfinite(y)
    return x[valid], y[valid]


def hist2d_counts(
    x: np.ndarray,
    y: np.ndarray,
    xbins: np.ndarray,
    ybins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.histogram2d(x, y, bins=[xbins, ybins])


# ============================================================
# 4. AXIS STYLE — MATCH MAIN FIGURE 3
# ============================================================

def style_log_axes(ax: plt.Axes) -> None:
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


def style_colorbar(cbar) -> None:
    cbar.ax.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_FONTSIZE,
        color="black",
        width=1.0,
        length=5,
    )


# ============================================================
# 5. HISTORICAL RECONSTRUCTION AND MMM
# ============================================================

def get_model_run_paths(model_name: str) -> dict[str, Path]:
    """Return standard saved-model paths for one CMIP6 model."""
    base_dir = MODEL_CONFIG[model_name]["base_dir"]
    run_dir = base_dir / "ml_xgboost_result" / 'FINAL_XGB'

    return {
        "run_dir": run_dir,
        "model_dir": run_dir / "models",
        "feature_dir": run_dir / "feature_lists",
        "metric_dir": run_dir / "metrics",
    }


def load_historical_vector(model_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reconstruct historical Lightning and Hail for one model's strict-land grid."""
    cfg = MODEL_CONFIG[model_name]
    paths = get_model_run_paths(model_name)

    train_table = cfg["train_table"]
    if not train_table.exists():
        raise FileNotFoundError(f"[{model_name}] TRAIN_TABLE not found: {train_table}")

    print("\n" + "=" * 100)
    print(f"[READ / RECONSTRUCT] {model_name}")
    print("=" * 100)
    print("[READ]", train_table)

    df = pd.read_parquet(train_table)
    ensure_columns(
        df,
        [LAT_COL, LON_COL, OBS_FLASH, OBS_HAIL],
        f"[{model_name}] historical train table",
    )

    if LAND_COL in df.columns:
        df = df[df[LAND_COL] == 0].copy()

    df = df[np.abs(pd.to_numeric(df[LAT_COL], errors="coerce")) <= LAT_MAX].copy()
    df = df.reset_index(drop=True)

    print(f"[DATA] {model_name} strict-land rows: {len(df)}")

    model_dir = paths["model_dir"]
    feature_dir = paths["feature_dir"]
    metric_dir = paths["metric_dir"]

    flash_model_path = model_dir / "FLASH_model.json"
    hail_model_path = model_dir / "HAIL_model.json"
    flash_feature_path = feature_dir / "FLASH_features.txt"
    hail_feature_path = feature_dir / "HAIL_features.txt"
    flash_summary_path = metric_dir / "FLASH_summary.json"
    hail_summary_path = metric_dir / "HAIL_summary.json"

    print("[LOAD MODELS]")
    booster_flash = load_booster(flash_model_path)
    booster_hail = load_booster(hail_model_path)
    features_flash = read_feature_list(flash_feature_path)
    features_hail = read_feature_list(hail_feature_path)
    best_iter_flash = read_best_iteration(flash_summary_path)
    best_iter_hail = read_best_iteration(hail_summary_path)

    print(
        f"[MODEL] FLASH features={len(features_flash)}, "
        f"best_iteration={best_iter_flash}"
    )
    print(
        f"[MODEL] HAIL  features={len(features_hail)}, "
        f"best_iteration={best_iter_hail}"
    )

    pred_flash = predict_with_booster(
        df=df,
        features=features_flash,
        booster=booster_flash,
        best_iteration=best_iter_flash,
        target_name="FLASH",
    )
    pred_hail = predict_with_booster(
        df=df,
        features=features_hail,
        booster=booster_hail,
        best_iteration=best_iter_hail,
        target_name="HAIL",
    )

    n_negative_flash = int(np.sum(np.isfinite(pred_flash) & (pred_flash < 0)))
    n_negative_hail = int(np.sum(np.isfinite(pred_hail) & (pred_hail < 0)))
    pred_flash = np.where(np.isfinite(pred_flash) & (pred_flash < 0), 0.0, pred_flash)
    pred_hail = np.where(np.isfinite(pred_hail) & (pred_hail < 0), 0.0, pred_hail)

    print(f"[CLIP] Negative Lightning predictions clipped: {n_negative_flash}")
    print(f"[CLIP] Negative Hail predictions clipped     : {n_negative_hail}")

    vector = pd.DataFrame(
        {
            LAT_COL: pd.to_numeric(df[LAT_COL], errors="coerce"),
            LON_COL: pd.to_numeric(df[LON_COL], errors="coerce"),
            "obs_lightning": pd.to_numeric(df[OBS_FLASH], errors="coerce"),
            "obs_hail": pd.to_numeric(df[OBS_HAIL], errors="coerce"),
            "pred_lightning": pred_flash,
            "pred_hail": pred_hail,
        }
    )

    vector = vector.dropna(subset=[LAT_COL, LON_COL]).copy()
    vector = (
        vector.groupby([LAT_COL, LON_COL], as_index=False)
        .mean(numeric_only=True)
        .sort_values([LAT_COL, LON_COL])
        .reset_index(drop=True)
    )

    metadata = {
        "model": model_name,
        "n_input_rows": int(len(df)),
        "n_grid_cells": int(len(vector)),
        "negative_lightning_clipped": n_negative_flash,
        "negative_hail_clipped": n_negative_hail,
        "best_iteration_lightning": best_iter_flash,
        "best_iteration_hail": best_iter_hail,
    }

    return vector, metadata


def build_mmm_vector(model_vectors: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Align the three historical reconstructions and compute their equal-weight MMM."""
    if REFERENCE_MODEL not in model_vectors:
        raise KeyError(f"REFERENCE_MODEL not available: {REFERENCE_MODEL}")

    merged: pd.DataFrame | None = None

    for model_name in MODEL_ORDER:
        df = model_vectors[model_name].copy()
        required = [
            LAT_COL,
            LON_COL,
            "obs_lightning",
            "obs_hail",
            "pred_lightning",
            "pred_hail",
        ]
        ensure_columns(df, required, f"[{model_name}] reconstruction vector")

        renamed = df[required].rename(
            columns={
                "obs_lightning": f"obs_lightning__{model_name}",
                "obs_hail": f"obs_hail__{model_name}",
                "pred_lightning": f"pred_lightning__{model_name}",
                "pred_hail": f"pred_hail__{model_name}",
            }
        )

        if merged is None:
            merged = renamed
        else:
            merged = merged.merge(renamed, on=[LAT_COL, LON_COL], how="inner")

    if merged is None or merged.empty:
        raise ValueError("No common grid cells remain after aligning the three model vectors.")

    ref_flash_col = f"obs_lightning__{REFERENCE_MODEL}"
    ref_hail_col = f"obs_hail__{REFERENCE_MODEL}"

    # The three historical tables should carry the same observed target values.
    # This check warns instead of stopping, allowing minor floating-point storage
    # differences across separately written parquet files.
    max_obs_flash_diff = 0.0
    max_obs_hail_diff = 0.0
    for model_name in MODEL_ORDER:
        flash_col = f"obs_lightning__{model_name}"
        hail_col = f"obs_hail__{model_name}"

        f_diff = np.abs(
            merged[flash_col].to_numpy(float) - merged[ref_flash_col].to_numpy(float)
        )
        h_diff = np.abs(
            merged[hail_col].to_numpy(float) - merged[ref_hail_col].to_numpy(float)
        )

        if np.isfinite(f_diff).any():
            max_obs_flash_diff = max(max_obs_flash_diff, float(np.nanmax(f_diff)))
        if np.isfinite(h_diff).any():
            max_obs_hail_diff = max(max_obs_hail_diff, float(np.nanmax(h_diff)))

    if max_obs_flash_diff > 1e-12 or max_obs_hail_diff > 1e-12:
        print(
            "[WARN] Observed targets are not numerically identical across model "
            f"tables after alignment: max Lightning difference={max_obs_flash_diff:.3e}, "
            f"max Hail difference={max_obs_hail_diff:.3e}. "
            f"The {REFERENCE_MODEL} observed values are used for MMM diagnostics."
        )

    pred_flash_cols = [f"pred_lightning__{m}" for m in MODEL_ORDER]
    pred_hail_cols = [f"pred_hail__{m}" for m in MODEL_ORDER]

    pred_flash_stack = merged[pred_flash_cols].to_numpy(dtype=float)
    pred_hail_stack = merged[pred_hail_cols].to_numpy(dtype=float)

    # MMM is defined only where all three individual predictions are finite,
    # ensuring that every grid cell has equal three-model weighting.
    valid_flash = np.all(np.isfinite(pred_flash_stack), axis=1)
    valid_hail = np.all(np.isfinite(pred_hail_stack), axis=1)

    pred_flash_mmm = np.full(len(merged), np.nan, dtype=float)
    pred_hail_mmm = np.full(len(merged), np.nan, dtype=float)
    pred_flash_mmm[valid_flash] = np.mean(pred_flash_stack[valid_flash], axis=1)
    pred_hail_mmm[valid_hail] = np.mean(pred_hail_stack[valid_hail], axis=1)

    mmm = pd.DataFrame(
        {
            LAT_COL: merged[LAT_COL].to_numpy(float),
            LON_COL: merged[LON_COL].to_numpy(float),
            "obs_lightning": merged[ref_flash_col].to_numpy(float),
            "obs_hail": merged[ref_hail_col].to_numpy(float),
            "pred_lightning": pred_flash_mmm,
            "pred_hail": pred_hail_mmm,
        }
    )

    metadata = {
        "model": "Muti-model Mean",
        "n_common_grid_cells": int(len(mmm)),
        "n_mmm_finite_lightning": int(np.sum(np.isfinite(pred_flash_mmm))),
        "n_mmm_finite_hail": int(np.sum(np.isfinite(pred_hail_mmm))),
        "max_observed_lightning_difference_between_tables": max_obs_flash_diff,
        "max_observed_hail_difference_between_tables": max_obs_hail_diff,
    }

    return mmm, metadata


# ============================================================
# 6. JOINT-DISTRIBUTION STATISTICS — MAIN FIGURE 3 DEFINITION
# ============================================================

def compute_joint_statistics(flash: np.ndarray, hail: np.ndarray) -> dict[str, Any]:
    """Compute all-point log10 R² plus the Figure 3 binned mean/SD fit."""
    flash, hail = prep_positive_pair(flash, hail)

    if flash.size < 3:
        raise ValueError("Too few positive Lightning–Hail pairs for joint-distribution statistics.")

    log_lightning = np.log10(flash)
    log_hail = np.log10(hail)

    _, _, r_value_all, p_value_all, _ = stats.linregress(log_hail, log_lightning)
    r2_all_loglog = float(r_value_all ** 2)

    # Keep the arithmetic bin-center convention used in Main Figure 3.
    hail_centers = (HAIL_BINS[:-1] + HAIL_BINS[1:]) / 2.0
    lightning_log_means: list[float] = []
    lightning_log_stds: list[float] = []

    for i in range(len(HAIL_BINS) - 1):
        in_bin = (hail >= HAIL_BINS[i]) & (hail < HAIL_BINS[i + 1])
        lightning_in_bin = flash[in_bin]

        if lightning_in_bin.size > 0:
            log_values = np.log10(lightning_in_bin)
            lightning_log_means.append(float(np.mean(log_values)))
            if log_values.size > 1:
                lightning_log_stds.append(float(np.std(log_values, ddof=1)))
            else:
                lightning_log_stds.append(np.nan)
        else:
            lightning_log_means.append(np.nan)
            lightning_log_stds.append(np.nan)

    lightning_log_means_arr = np.asarray(lightning_log_means, dtype=float)
    lightning_log_stds_arr = np.asarray(lightning_log_stds, dtype=float)

    fit_valid = np.isfinite(lightning_log_means_arr)[:FIT_FIRST_N_HAIL_BINS]
    if int(fit_valid.sum()) >= 3:
        hail_fit = np.log10(hail_centers[:FIT_FIRST_N_HAIL_BINS])[fit_valid]
        lightning_fit = lightning_log_means_arr[:FIT_FIRST_N_HAIL_BINS][fit_valid]
        slope, intercept, _, _, _ = stats.linregress(hail_fit, lightning_fit)
    else:
        slope = np.nan
        intercept = np.nan

    return {
        "n_positive_pairs": int(flash.size),
        "r2_log10_all_points": r2_all_loglog,
        "p_value_log10_all_points": float(p_value_all),
        "hail_centers": hail_centers,
        "lightning_mean": 10.0 ** lightning_log_means_arr,
        "lightning_upper": 10.0 ** (lightning_log_means_arr + lightning_log_stds_arr),
        "lightning_lower": 10.0 ** (lightning_log_means_arr - lightning_log_stds_arr),
        "fit_slope": float(slope) if np.isfinite(slope) else np.nan,
        "fit_intercept": float(intercept) if np.isfinite(intercept) else np.nan,
    }


# ============================================================
# 7. PANEL DRAWING — HISTORICAL VERSION WITH MAIN FIGURE 3 STYLE
# ============================================================

def draw_frequency_panel(
    ax: plt.Axes,
    cax: plt.Axes,
    flash: np.ndarray,
    hail: np.ndarray,
    title: str,
    cmap_freq,
    show_legend: bool,
) -> dict[str, Any]:
    """Draw observed or reconstructed 2-D joint frequency with Figure 3 styling."""
    flash_valid, hail_valid = prep_positive_pair(flash, hail)
    if flash_valid.size < 10:
        raise ValueError(f"Too few valid samples for panel: {title}")

    joint_stats = compute_joint_statistics(flash_valid, hail_valid)

    print(
        f"[{title}] valid samples={joint_stats['n_positive_pairs']}; "
        f"all-point log10 R²={joint_stats['r2_log10_all_points']:.4f}"
    )

    hist, xedges, yedges = hist2d_counts(
        flash_valid,
        hail_valid,
        FLASH_BINS,
        HAIL_BINS,
    )

    xcent = np.sqrt(xedges[:-1] * xedges[1:])
    ycent = np.sqrt(yedges[:-1] * yedges[1:])
    Xc, Yc = np.meshgrid(xcent, ycent)

    contour = ax.contourf(
        Xc,
        Yc,
        hist.T,
        levels=FREQ_LEVELS,
        cmap=cmap_freq,
        zorder=1,
    )

    cbar = plt.colorbar(contour, cax=cax)
    cbar.set_label("Grid Count", fontsize=CBAR_LABEL_FONTSIZE)
    style_colorbar(cbar)

    ax.plot(
        joint_stats["lightning_mean"],
        joint_stats["hail_centers"],
        color=MEAN_LINE_COLOR,
        linestyle="-",
        linewidth=1.8,
        label="Mean Lightning Frequency",
        zorder=10,
    )
    ax.plot(
        joint_stats["lightning_upper"],
        joint_stats["hail_centers"],
        color=MEAN_LINE_COLOR,
        linestyle="--",
        linewidth=1.5,
        label="Mean ± 1 SD",
        zorder=10,
    )
    ax.plot(
        joint_stats["lightning_lower"],
        joint_stats["hail_centers"],
        color=MEAN_LINE_COLOR,
        linestyle="--",
        linewidth=1.5,
        zorder=10,
    )

    slope = joint_stats["fit_slope"]
    intercept = joint_stats["fit_intercept"]

    if np.isfinite(slope) and np.isfinite(intercept):
        hail_ext = np.logspace(np.log10(YLIM[0]), np.log10(YLIM[1]), 200)
        lightning_fit = 10.0 ** (slope * np.log10(hail_ext) + intercept)

        ax.plot(
            lightning_fit,
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
        r_str = rf"$\mathrm{{log}}_{{10}}$: $R^2 = {joint_stats['r2_log10_all_points']:.2f}$"

        # Identical annotation locations to Main Figure 3 panel (b).
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
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)

    ax.set_xlabel("Lightning Frequency", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Hail Frequency", fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold", pad=19)

    if show_legend:
        ax.legend(loc="lower right", fontsize=LEGEND_FONTSIZE, frameon=False)

    style_log_axes(ax)

    return {
        "n_positive_pairs": joint_stats["n_positive_pairs"],
        "r2_log10_all_points": joint_stats["r2_log10_all_points"],
        "p_value_log10_all_points": joint_stats["p_value_log10_all_points"],
        "fit_slope_first_25_hail_bins": joint_stats["fit_slope"],
        "fit_intercept_first_25_hail_bins": joint_stats["fit_intercept"],
        "histogram_max_grid_count": float(np.nanmax(hist)),
    }


def draw_difference_panel(
    ax: plt.Axes,
    cax: plt.Axes,
    pred_lightning: np.ndarray,
    pred_hail: np.ndarray,
    obs_lightning: np.ndarray,
    obs_hail: np.ndarray,
    title: str,
    cmap_diff,
) -> dict[str, Any]:
    """Draw reconstructed-minus-observed 2-D count difference, as in Figure 3(c)."""
    pred_flash_valid, pred_hail_valid = prep_positive_pair(pred_lightning, pred_hail)
    obs_flash_valid, obs_hail_valid = prep_positive_pair(obs_lightning, obs_hail)

    if pred_flash_valid.size < 10 or obs_flash_valid.size < 10:
        raise ValueError(f"Too few valid samples for difference panel: {title}")

    H_pred, xedges, yedges = hist2d_counts(
        pred_flash_valid,
        pred_hail_valid,
        FLASH_BINS,
        HAIL_BINS,
    )
    H_obs, _, _ = hist2d_counts(
        obs_flash_valid,
        obs_hail_valid,
        FLASH_BINS,
        HAIL_BINS,
    )

    difference = np.full_like(H_pred, np.nan, dtype=float)
    has_any_count = ~((H_pred == 0) & (H_obs == 0))
    difference[has_any_count] = H_pred[has_any_count] - H_obs[has_any_count]

    xcent = np.sqrt(xedges[:-1] * xedges[1:])
    ycent = np.sqrt(yedges[:-1] * yedges[1:])
    Xc, Yc = np.meshgrid(xcent, ycent)

    vmax = float(np.nanmax(np.abs(difference)))
    if (not np.isfinite(vmax)) or vmax == 0:
        vmax = 1e-6

    levels = np.linspace(-vmax, vmax, DIFF_LEVELS)
    contour = ax.contourf(
        Xc,
        Yc,
        difference.T,
        levels=levels,
        cmap=cmap_diff,
        extend="both",
        zorder=1,
    )

    cbar = plt.colorbar(contour, cax=cax)
    cbar.set_label(
        "ΔGrid Count (Reconstruction − Observed)",
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
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold", pad=19)

    style_log_axes(ax)

    return {
        "n_positive_pairs_reconstruction": int(pred_flash_valid.size),
        "n_positive_pairs_observed": int(obs_flash_valid.size),
        "max_abs_grid_count_difference": vmax,
    }


# ============================================================
# 8. FIGURE CONSTRUCTION — EXACT MAIN FIGURE 3 GRID SPEC
# ============================================================

def make_historical_comparison_figure(
    label: str,
    vector: pd.DataFrame,
    cmap_freq,
    cmap_diff,
) -> list[dict[str, Any]]:
    """Make one 1×3 historical comparison figure with Main Figure 3 layout."""
    ensure_columns(
        vector,
        ["obs_lightning", "obs_hail", "pred_lightning", "pred_hail"],
        f"[{label}] historical vector",
    )

    obs_lightning = vector["obs_lightning"].to_numpy(dtype=float)
    obs_hail = vector["obs_hail"].to_numpy(dtype=float)
    pred_lightning = vector["pred_lightning"].to_numpy(dtype=float)
    pred_hail = vector["pred_hail"].to_numpy(dtype=float)

    # =====================================================
    # Fixed-layout 1 × 3 figure — copied from Main Figure 3
    # =====================================================
    fig = plt.figure(figsize=MAIN_FIGURE3_LAYOUT["figsize"], dpi=FIG_DPI)

    outer_gs = fig.add_gridspec(
        nrows=1,
        ncols=3,
        left=MAIN_FIGURE3_LAYOUT["left"],
        right=MAIN_FIGURE3_LAYOUT["right"],
        bottom=MAIN_FIGURE3_LAYOUT["bottom"],
        top=MAIN_FIGURE3_LAYOUT["top"],
        wspace=MAIN_FIGURE3_LAYOUT["wspace"],
    )

    gs_a = outer_gs[0, 0].subgridspec(
        nrows=1,
        ncols=2,
        width_ratios=[1.0, MAIN_FIGURE3_LAYOUT["panel_cbar_width_ratio"]],
        wspace=MAIN_FIGURE3_LAYOUT["panel_cbar_wspace"],
    )
    gs_b = outer_gs[0, 1].subgridspec(
        nrows=1,
        ncols=2,
        width_ratios=[1.0, MAIN_FIGURE3_LAYOUT["panel_cbar_width_ratio"]],
        wspace=MAIN_FIGURE3_LAYOUT["panel_cbar_wspace"],
    )
    gs_c = outer_gs[0, 2].subgridspec(
        nrows=1,
        ncols=2,
        width_ratios=[1.0, MAIN_FIGURE3_LAYOUT["panel_cbar_width_ratio"]],
        wspace=MAIN_FIGURE3_LAYOUT["panel_cbar_wspace"],
    )

    ax_a = fig.add_subplot(gs_a[0, 0])
    cax_a = fig.add_subplot(gs_a[0, 1])
    ax_b = fig.add_subplot(gs_b[0, 0])
    cax_b = fig.add_subplot(gs_b[0, 1])
    ax_c = fig.add_subplot(gs_c[0, 0])
    cax_c = fig.add_subplot(gs_c[0, 1])

    stats_obs = draw_frequency_panel(
        ax=ax_a,
        cax=cax_a,
        flash=obs_lightning,
        hail=obs_hail,
        title="(a) Observed Two-dimensional Frequency",
        cmap_freq=cmap_freq,
        show_legend=False,
    )

    stats_pred = draw_frequency_panel(
        ax=ax_b,
        cax=cax_b,
        flash=pred_lightning,
        hail=pred_hail,
        title=f"{label}: (b) Historical Reconstruction",
        cmap_freq=cmap_freq,
        show_legend=True,
    )

    stats_diff = draw_difference_panel(
        ax=ax_c,
        cax=cax_c,
        pred_lightning=pred_lightning,
        pred_hail=pred_hail,
        obs_lightning=obs_lightning,
        obs_hail=obs_hail,
        title="(c) Two-dimensional Frequency Difference",
        cmap_diff=cmap_diff,
    )

    out_png = OUT_DIR / f"{FIGURE_PREFIX}_{label.replace(' ', '_')}.jpg"
    fig.savefig(out_png, dpi=FIG_DPI)
    plt.close(fig)

    print("[SAVED]", out_png)

    return [
        {"figure": label, "panel": "a_observed", **stats_obs},
        {"figure": label, "panel": "b_historical_reconstruction", **stats_pred},
        {"figure": label, "panel": "c_reconstruction_minus_observed", **stats_diff},
    ]


# ============================================================
# 9. MAIN
# ============================================================

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("[STEP 9] Historical Lightning–Hail joint-frequency comparison")
    print("[LAYOUT] Main Figure 3 GridSpec geometry and styling")
    print("=" * 100)

    # --------------------------------------------------------
    # 1. Reconstruct each model's historical grid.
    # --------------------------------------------------------
    model_vectors: dict[str, pd.DataFrame] = {}
    model_metadata: list[dict[str, Any]] = []

    for model_name in MODEL_ORDER:
        vector, metadata = load_historical_vector(model_name)
        model_vectors[model_name] = vector
        model_metadata.append(metadata)

        vector_out = OUT_DIR / f"{FIGURE_PREFIX}_vectors_{model_name}.csv"
        vector.to_csv(vector_out, index=False, encoding="utf-8-sig")
        print("[SAVED]", vector_out)

    # --------------------------------------------------------
    # 2. Build and save the equal-weight three-model mean.
    # --------------------------------------------------------
    mmm_vector, mmm_metadata = build_mmm_vector(model_vectors)
    model_vectors["Muti-model Mean"] = mmm_vector
    model_metadata.append(mmm_metadata)

    mmm_vector_out = OUT_DIR / f"{FIGURE_PREFIX}_vectors_Muti-model_Mean.csv"
    mmm_vector.to_csv(mmm_vector_out, index=False, encoding="utf-8-sig")
    print("[SAVED]", mmm_vector_out)

    metadata_out = OUT_DIR / f"{FIGURE_PREFIX}_grid_coverage_summary.csv"
    pd.DataFrame(model_metadata).to_csv(metadata_out, index=False, encoding="utf-8-sig")
    print("[SAVED]", metadata_out)

    # --------------------------------------------------------
    # 3. Draw four figures with the Main Figure 3 layout.
    # --------------------------------------------------------
    cmap_freq = load_ncl_mat_cmap(C_MAP_MAT, key="cmap", reverse=True)
    cmap_diff = diverging_with_gray_center(gray=0.95)

    all_stats: list[dict[str, Any]] = []
    for label in [*MODEL_ORDER, "Muti-model Mean"]:
        print("\n" + "=" * 100)
        print(f"[PLOT] {label}")
        print("=" * 100)
        all_stats.extend(
            make_historical_comparison_figure(
                label=label,
                vector=model_vectors[label],
                cmap_freq=cmap_freq,
                cmap_diff=cmap_diff,
            )
        )

    stats_out = OUT_DIR / f"{FIGURE_PREFIX}_joint_distribution_summary.csv"
    pd.DataFrame(all_stats).to_csv(stats_out, index=False, encoding="utf-8-sig")
    print("\n[SAVED]", stats_out)

    print("\n[DONE] Historical joint-frequency figures completed.")
    print("[OUT_DIR]", OUT_DIR)


if __name__ == "__main__":
    main()
