# Purpose: figure s03 test scatter.
# Source: cmip6_figure_test_散点.ipynb; see docs/source_coverage.csv for cell provenance.
# Run order and required inputs: docs/workflow.md.
# Run from repository root: python -m figures.figure_s03_test_scatter

# ============================================================
# Step 8. Plot three-model test scatter panels with a GridSpec/
# subplots layout that can be adjusted using wspace and hspace.
#
# Figure layout
#   (a) Lightning Frequency | CanESM5
#   (b) Lightning Frequency | MIROC6
#   (c) Lightning Frequency | BCC-CSM2-MR
#   (d) Hail Frequency      | CanESM5
#   (e) Hail Frequency      | MIROC6
#   (f) Hail Frequency      | BCC-CSM2-MR
#
# Scientific definitions (unchanged)
#   1. R², RMSE, MAE, and Bias are calculated in original-value
#      space using all finite test samples.
#   2. Scatter points and OLS fit use only y_true > 0 and y_pred > 0,
#      transformed into log10 space.
#   3. Blue line: 1:1 line in log10 space.
#   4. Orange line: log10 ordinary least-squares (OLS) fit.
#
# Layout control
#   Adjust only SUBPLOT_LAYOUT["wspace"] and ["hspace"] to change
#   the horizontal and vertical gaps between panels. Use left/right/
#   bottom/top to modify the outer margins.
# ============================================================

from __future__ import annotations

from publication_config import resource_path

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 0. User configuration
# ============================================================

# Keep the model order consistent with all manuscript multipanel figures.
MODEL_RUN_DIRS = {
    "CanESM5": Path(resource_path('models', 'CanESM5/ml_xgboost_result/FINAL_XGB')),
    "MIROC6": Path(resource_path('models', 'MIROC6/ml_xgboost_result/FINAL_XGB')),
    "BCC-CSM2-MR": Path(resource_path('models', 'BCC-CSM2-MR/ml_xgboost_result/FINAL_XGB')),
}

OUT_DIR = Path(resource_path('figures', ''))
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FIG = OUT_DIR / 'figure_s03_test_scatter.jpg'
OUT_METRICS_CSV = OUT_DIR / 'figure_s03_test_scatter_summary.csv'

TARGETS = {
    "lightning": {
        "input_file": "FLASH_pred_test.csv",
        "display_name": "Lightning Frequency",
        "csv_target_name": "FLASH",
        "panel_labels": {
            "CanESM5": "(a)",
            "MIROC6": "(b)",
            "BCC-CSM2-MR": "(c)",
        },
    },
    "hail": {
        "input_file": "HAIL_pred_test.csv",
        "display_name": "Hail Frequency",
        "csv_target_name": "HAIL",
        "panel_labels": {
            "CanESM5": "(d)",
            "MIROC6": "(e)",
            "BCC-CSM2-MR": "(f)",
        },
    },
}

# ============================================================
# 1. Figure style and layout
# ============================================================

FIG_DPI = 600

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 16,
    "axes.unicode_minus": False,
    "figure.dpi": FIG_DPI,
    "savefig.dpi": FIG_DPI,
})

# These values are intentionally slightly smaller than the main Figure 5
# because the panel titles contain long model names.
PANEL_TITLE_SIZE = 22
PANEL_TITLE_PAD = 6
SHARED_LABEL_SIZE = 20
TICK_LABEL_SIZE = 20
TEXT_FONTSIZE = 20
LEGEND_SIZE = 20
BORDER_WIDTH = 1.2

SCATTER_COLOR = "#9ECAE1"
ONE_TO_ONE_COLOR = "#0072B2"
FIT_COLOR = "#D55E00"

# ----------------------------------------------------------------
# Main layout controls
# ----------------------------------------------------------------
# wspace: horizontal gap between adjacent columns.
# hspace: vertical gap between the two rows.
# left/right/top/bottom: outer margins around the entire 2×3 panel grid.
# ----------------------------------------------------------------
SUBPLOT_LAYOUT = {
    "figsize": (22.0, 15.5),
    "left": 0.085,
    "right": 0.985,
    "bottom": 0.130,
    "top": 0.930,
    "wspace": 0.05,
    "hspace": 0.22,
}

# Square plot boxes preserve the visual meaning of the 1:1 line while still
# allowing wspace/hspace to control the panel arrangement.
USE_SQUARE_PANELS = True

# Scatter style. These do not change the underlying samples or metrics.
SCATTER_SIZE = 12
SCATTER_ALPHA = 0.40

# Common log10-axis range padding for each target row.
AXIS_PAD_FRACTION = 0.08

# Shared-label and legend offsets are measured relative to the final subplot
# grid after wspace/hspace are applied; normally they do not need adjustment.
SHARED_XLABEL_GAP = 0.035
SHARED_YLABEL_GAP = 0.025
LEGEND_GAP = 0.09

SAVE_BBOX_TIGHT = False


# ============================================================
# 2. Data and statistics helpers
# ============================================================

def read_test_prediction(path: Path, target_name: str, model_name: str) -> pd.DataFrame:
    """Read one model-target test-prediction table."""
    if not path.exists():
        raise FileNotFoundError(f"Missing test-prediction file: {path}")

    df = pd.read_csv(path)
    required = {"y_true", "y_pred"}
    missing = required - set(df.columns)

    if missing:
        raise KeyError(f"{path.name} is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["model"] = model_name
    df["target"] = target_name
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")

    return df


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute original-space metrics using all finite test samples."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[valid]
    yp = y_pred[valid]

    if yt.size < 2:
        return {
            "r2": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "bias": np.nan,
            "n": int(yt.size),
        }

    return {
        "r2": float(r2_score(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae": float(mean_absolute_error(yt, yp)),
        "bias": float(np.mean(yp - yt)),
        "n": int(yt.size),
    }


def log10_fit(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Fit log10(predicted) against log10(observed) for positive pairs."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    positive = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
        & (y_true > 0)
        & (y_pred > 0)
    )

    if int(positive.sum()) < 3:
        return {
            "slope": np.nan,
            "intercept": np.nan,
            "r2_log": np.nan,
            "p_value": np.nan,
            "std_err": np.nan,
            "n_pos": int(positive.sum()),
            "x_log": np.array([], dtype=float),
            "y_log": np.array([], dtype=float),
        }

    x_log = np.log10(y_true[positive])
    y_log = np.log10(y_pred[positive])

    slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2_log": float(r_value ** 2),
        "p_value": float(p_value),
        "std_err": float(std_err),
        "n_pos": int(positive.sum()),
        "x_log": x_log,
        "y_log": y_log,
    }


def common_log_limits(dataframes: list[pd.DataFrame], pad_frac: float) -> tuple[float, float]:
    """Return one shared square x/y range for all panels in one row."""
    values = []

    for df in dataframes:
        y_true = df["y_true"].to_numpy(dtype=float)
        y_pred = df["y_pred"].to_numpy(dtype=float)
        valid = (
            np.isfinite(y_true)
            & np.isfinite(y_pred)
            & (y_true > 0)
            & (y_pred > 0)
        )

        if valid.any():
            values.append(np.log10(y_true[valid]))
            values.append(np.log10(y_pred[valid]))

    if not values:
        raise ValueError("No positive finite observed-predicted pairs are available.")

    values = np.concatenate(values)
    values = values[np.isfinite(values)]

    if values.size == 0:
        raise ValueError("No finite log10 values are available for shared axis limits.")

    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError("Unable to determine finite shared axis limits.")

    if np.isclose(vmin, vmax):
        return vmin - 0.5, vmax + 0.5

    pad = (vmax - vmin) * float(pad_frac)
    return vmin - pad, vmax + pad


# ============================================================
# 3. Plot helpers
# ============================================================

def style_axis(ax, show_y_ticklabels: bool) -> None:
    """Apply the common non-map style to one scatter panel."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(BORDER_WIDTH)
        spine.set_color("black")

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=5,
        width=1.1,
        labelsize=TICK_LABEL_SIZE,
        top=False,
        right=False,
    )
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    if not show_y_ticklabels:
        ax.tick_params(axis="y", labelleft=False)

    ax.grid(
        True,
        which="major",
        axis="both",
        linestyle="--",
        linewidth=0.75,
        alpha=0.60,
        color="0.70",
        zorder=0,
    )
    ax.set_axisbelow(True)


def format_fit_equation(slope: float, intercept: float) -> str:
    """Retain the original in-panel log10-fit text format."""
    if not np.isfinite(slope) or not np.isfinite(intercept):
        return "Fit (log10): unavailable"

    return f"Fit (log10): y = {slope:.3f}x + {intercept:.3g}"


def plot_one_scatter(
    ax,
    df: pd.DataFrame,
    title: str,
    shared_limits: tuple[float, float],
    show_y_ticklabels: bool,
) -> dict:
    """Draw one model-target test scatter panel."""
    y_true = df["y_true"].to_numpy(dtype=float)
    y_pred = df["y_pred"].to_numpy(dtype=float)

    metrics = compute_metrics(y_true, y_pred)
    fit = log10_fit(y_true, y_pred)

    x_log = fit["x_log"]
    y_log = fit["y_log"]

    if x_log.size < 3:
        raise ValueError(f"Too few positive finite samples for log10 scatter: {title}")

    axis_min, axis_max = shared_limits

    ax.scatter(
        x_log,
        y_log,
        s=SCATTER_SIZE,
        alpha=SCATTER_ALPHA,
        color=SCATTER_COLOR,
        edgecolors="none",
        rasterized=True,
        zorder=2,
    )

    # 1:1 line in log10 space.
    ax.plot(
        [axis_min, axis_max],
        [axis_min, axis_max],
        color=ONE_TO_ONE_COLOR,
        linewidth=2.15,
        zorder=3,
    )

    # Log10 ordinary least-squares fit.
    if np.isfinite(fit["slope"]) and np.isfinite(fit["intercept"]):
        xx = np.linspace(axis_min, axis_max, 300)
        yy = fit["slope"] * xx + fit["intercept"]
        ax.plot(
            xx,
            yy,
            color=FIT_COLOR,
            linewidth=2.15,
            zorder=4,
        )

    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)

    # box_aspect controls the physical box shape but leaves subplot spacing
    # under the control of fig.subplots_adjust(wspace=..., hspace=...).
    if USE_SQUARE_PANELS:
        ax.set_box_aspect(1)

    ax.set_title(
        title,
        loc="left",
        fontsize=PANEL_TITLE_SIZE,
        fontweight="bold",
        pad=PANEL_TITLE_PAD,
    )

    # Full metrics content retained exactly as requested.
    text = (
        f"R²={metrics['r2']:.3f}  RMSE={metrics['rmse']:.5f}\n"
        f"MAE={metrics['mae']:.5f}  \n"
        f"{format_fit_equation(fit['slope'], fit['intercept'])}"
    )

    ax.text(
        0.025,
        0.975,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=TEXT_FONTSIZE,
        color="black",
        linespacing=1.35,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.78,
            "pad": 3.0,
        },
        zorder=10,
    )

    style_axis(ax, show_y_ticklabels=show_y_ticklabels)

    return {
        "r2_raw": metrics["r2"],
        "rmse_raw": metrics["rmse"],
        "mae_raw": metrics["mae"],
        "bias_raw": metrics["bias"],
        "n_all_finite": metrics["n"],
        "n_positive_for_log_plot": fit["n_pos"],
        "log_fit_slope": fit["slope"],
        "log_fit_intercept": fit["intercept"],
        "log_fit_r2": fit["r2_log"],
        "log_fit_p_value": fit["p_value"],
        "log_fit_std_err": fit["std_err"],
        "axis_min": axis_min,
        "axis_max": axis_max,
    }


def get_subplot_bounds(axes: np.ndarray) -> dict[str, float]:
    """Return figure-normalized bounds after box_aspect has been applied."""
    positions = [ax.get_position() for ax in axes.ravel()]
    return {
        "left": min(pos.x0 for pos in positions),
        "right": max(pos.x1 for pos in positions),
        "bottom": min(pos.y0 for pos in positions),
        "top": max(pos.y1 for pos in positions),
        "top_row_bottom": min(axes[0, col].get_position().y0 for col in range(axes.shape[1])),
        "bottom_row_bottom": min(axes[1, col].get_position().y0 for col in range(axes.shape[1])),
        "top_row_center_y": np.mean([
            axes[0, col].get_position().y0 + axes[0, col].get_position().height / 2
            for col in range(axes.shape[1])
        ]),
        "bottom_row_center_y": np.mean([
            axes[1, col].get_position().y0 + axes[1, col].get_position().height / 2
            for col in range(axes.shape[1])
        ]),
    }


def safe_savefig(fig, out_path: Path, dpi: int = FIG_DPI) -> None:
    """Safely write the figure without leaving a partial file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = Path(str(out_path) + ".tmp.jpg")

    if temp_path.exists():
        temp_path.unlink()
    if out_path.exists():
        out_path.unlink()

    save_kwargs = {"dpi": dpi}
    if SAVE_BBOX_TIGHT:
        save_kwargs["bbox_inches"] = "tight"

    fig.savefig(temp_path, **save_kwargs)
    temp_path.replace(out_path)


# ============================================================
# 4. Main workflow
# ============================================================

def main() -> None:
    print("=" * 100)
    print("[STEP 8] Plot Three-Model Test Scatter Panels Using Subplots Layout")
    print("=" * 100)

    # ----------------------------------------------------------------
    # Read six prediction tables first.
    # ----------------------------------------------------------------
    data = {target_key: {} for target_key in TARGETS}

    for target_key, target_cfg in TARGETS.items():
        for model_name, run_dir in MODEL_RUN_DIRS.items():
            input_path = run_dir / "predictions" / target_cfg["input_file"]
            print(
                f"[READ] {model_name:<12s} | "
                f"{target_cfg['csv_target_name']:<5s} | {input_path}"
            )
            data[target_key][model_name] = read_test_prediction(
                path=input_path,
                target_name=target_cfg["csv_target_name"],
                model_name=model_name,
            )

    # ----------------------------------------------------------------
    # One common square log10 range per target row.
    # ----------------------------------------------------------------
    shared_limits = {}
    for target_key, target_cfg in TARGETS.items():
        row_dfs = [data[target_key][model_name] for model_name in MODEL_RUN_DIRS]
        shared_limits[target_key] = common_log_limits(
            row_dfs,
            pad_frac=AXIS_PAD_FRACTION,
        )
        print(
            f"[AXIS] {target_cfg['display_name']}: "
            f"{shared_limits[target_key][0]:.3f} to "
            f"{shared_limits[target_key][1]:.3f}"
        )

    # ----------------------------------------------------------------
    # Create a standard 2 × 3 subplot grid.
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=SUBPLOT_LAYOUT["figsize"],
        dpi=FIG_DPI,
        squeeze=False,
    )

    fig.subplots_adjust(
        left=SUBPLOT_LAYOUT["left"],
        right=SUBPLOT_LAYOUT["right"],
        bottom=SUBPLOT_LAYOUT["bottom"],
        top=SUBPLOT_LAYOUT["top"],
        wspace=SUBPLOT_LAYOUT["wspace"],
        hspace=SUBPLOT_LAYOUT["hspace"],
    )

    metrics_rows = []
    model_names = list(MODEL_RUN_DIRS)

    for row_index, target_key in enumerate(("lightning", "hail")):
        target_cfg = TARGETS[target_key]

        for col_index, model_name in enumerate(model_names):
            ax = axes[row_index, col_index]
            panel_label = target_cfg["panel_labels"][model_name]
            panel_title = f"{panel_label} {target_cfg['display_name']} | {model_name}"

            result = plot_one_scatter(
                ax=ax,
                df=data[target_key][model_name],
                title=panel_title,
                shared_limits=shared_limits[target_key],
                show_y_ticklabels=(col_index == 0),
            )

            metrics_rows.append({
                "model": model_name,
                "target": target_cfg["csv_target_name"],
                "target_display_name": target_cfg["display_name"],
                **result,
            })

    # Finalize square panel boxes before dynamically positioning shared labels.
    fig.canvas.draw()
    bounds = get_subplot_bounds(axes)
    grid_x_center = (bounds["left"] + bounds["right"]) / 2

    # ----------------------------------------------------------------
    # Shared row labels. They follow the actual final axes positions, so only
    # hspace/wspace and the gap controls above need adjustment.
    # ----------------------------------------------------------------
    fig.text(
        grid_x_center,
        bounds["top_row_bottom"] - SHARED_XLABEL_GAP,
        r"$\log_{10}$(Observed Lightning Frequency)",
        ha="center",
        va="center",
        fontsize=SHARED_LABEL_SIZE,
    )
    fig.text(
        bounds["left"] - SHARED_YLABEL_GAP,
        bounds["top_row_center_y"],
        r"$\log_{10}$(Predicted Lightning Frequency)",
        ha="center",
        va="center",
        rotation=90,
        fontsize=SHARED_LABEL_SIZE,
    )
    fig.text(
        grid_x_center,
        bounds["bottom_row_bottom"] - SHARED_XLABEL_GAP,
        r"$\log_{10}$(Observed Hail Frequency)",
        ha="center",
        va="center",
        fontsize=SHARED_LABEL_SIZE,
    )
    fig.text(
        bounds["left"] - SHARED_YLABEL_GAP,
        bounds["bottom_row_center_y"],
        r"$\log_{10}$(Predicted Hail Frequency)",
        ha="center",
        va="center",
        rotation=90,
        fontsize=SHARED_LABEL_SIZE,
    )

    # Shared legend.
    line_handles = [
        Line2D([0], [0], color=ONE_TO_ONE_COLOR, linewidth=2.2, label="1:1 Line"),
        Line2D([0], [0], color=FIT_COLOR, linewidth=2.2, label="Log10 Fit"),
    ]
    legend_y = max(0.01, bounds["bottom"] - LEGEND_GAP)
    fig.legend(
        handles=line_handles,
        loc="lower center",
        bbox_to_anchor=(grid_x_center, legend_y),
        ncol=2,
        frameon=False,
        fontsize=LEGEND_SIZE,
        handlelength=2.4,
        columnspacing=2.4,
    )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(OUT_METRICS_CSV, index=False, encoding="utf-8-sig")

    safe_savefig(fig, OUT_FIG, dpi=FIG_DPI)
    plt.close(fig)

    print("\n[METRICS]")
    print(metrics_df.to_string(index=False))
    print("\n[SAVED FIGURE]")
    print(OUT_FIG)
    print("\n[SAVED SUMMARY]")
    print(OUT_METRICS_CSV)
    print("\n[DONE] Step 8 finished successfully.")


if __name__ == "__main__":
    main()
