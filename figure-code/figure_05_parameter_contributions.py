# Purpose: figure 05 parameter contributions.
# Source: cmip6_figure5_contribution.ipynb; see docs/source_coverage.csv for cell provenance.
# Run order and required inputs: docs/workflow.md.
# Run from repository root: python -m figures.figure_05_parameter_contributions


'\nFigure 5: Six-panel two-way signed parameter contributions.\n\nLayout:\n(a) Flash | CanESM5      (b) Flash | MIROC6      (c) Flash | BCC-CSM2-MR\n(d) Hail  | CanESM5      (e) Hail  | MIROC6      (f) Hail  | BCC-CSM2-MR\n\nBar length:\n    Absolute value of the five-decade mean signed contribution.\n\nBar color:\n    Orange = positive signed contribution.\n    Blue   = negative signed contribution.\n\nInput:\n    [configured path; see config.json]\n\nOutput:\n    [configured path; see config.json]\n'

from __future__ import annotations

from publication_config import resource_path

from pathlib import Path
import math

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


# ============================================================
# >>> MARKER: LOCAL_PLOT_STYLE_START
# ============================================================

FIG_DPI = 600
BORDER_WIDTH = 1.2

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 18,
    "axes.unicode_minus": False,
    "figure.dpi": FIG_DPI,
    "savefig.dpi": FIG_DPI,
})


def style_plain_axis(ax, grid: bool = True) -> None:
    """Apply a clean SCI-style format to one non-map axis."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(BORDER_WIDTH)

    ax.tick_params(
        direction="out",
        length=5.0,
        width=1.0,
        top=False,
        right=False,
    )

    if grid:
        ax.grid(
            axis="x",
            linestyle="--",
            linewidth=0.65,
            color="0.72",
            alpha=0.55,
            zorder=0,
        )
        ax.set_axisbelow(True)


def save_figure(fig, path: Path, tight: bool = True) -> None:
    """Save figure and ensure its parent directory exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    save_kwargs = {"dpi": FIG_DPI}
    if tight:
        save_kwargs["bbox_inches"] = "tight"

    fig.savefig(path, **save_kwargs)
    plt.close(fig)


# <<< MARKER: LOCAL_PLOT_STYLE_END


# ============================================================
# >>> MARKER: USER_CONFIGURATION_START
# ============================================================

DEFAULT_ROOT = Path(resource_path('attribution', ''))

ATTRIBUTION_DIR = "attribution_parameter14_single"

MODELS = [
    "CanESM5",
    "MIROC6",
    "BCC-CSM2-MR",
]

# Only plot two-way attribution results.
DIRECTION = "two_way"

DECADES = [
    "2050s",
    "2060s",
    "2070s",
    "2080s",
    "2090s",
]

POSITIVE_COLOR = "#D55E00"
NEGATIVE_COLOR = "#0072B2"

PANEL_LABELS = {
    ("flash", "CanESM5"): "(a)",
    ("flash", "MIROC6"): "(b)",
    ("flash", "BCC-CSM2-MR"): "(c)",
    ("hail", "CanESM5"): "(d)",
    ("hail", "MIROC6"): "(e)",
    ("hail", "BCC-CSM2-MR"): "(f)",
}

MODEL_DISPLAY_NAMES = {
    "CanESM5": "CanESM5",
    "MIROC6": "MIROC6",
    "BCC-CSM2-MR": "BCC-CSM2-MR",
}

TARGET_DISPLAY_NAMES = {
    "flash": "Lightning Frequency",
    "hail": "Hail Frequency",
}

PARAMETER_LABELS = {
    "mucape": "MUCAPE",
    "dh_10_30": r"$\mathrm{\Delta H}_{-10,-30}$",
    "mucin": "MUCIN",
    "k_index": "K index",
    "theta_e": r"$\mathrm{\theta}_{\mathrm{e}}$",
    "flh": "FLH",
    "s06": "S06",
    "td_sfc": r"$\mathrm{T}_{\mathrm{d,sfc}}$",
    "h_30c": r"$\mathrm{H}_{-30}$",
    "h_10c": r"$\mathrm{H}_{-10}$",
    "pw": "PW",
    "cin": "CIN",
    "div500": r"$\mathrm{Div}_{500}$",
    "cape": "CAPE",
}

# ------------------------------------------------------------
# Output paths
# ------------------------------------------------------------

FINAL_FIG_PATH = Path(
    resource_path('figures', 'figure_05_parameter_contributions.jpg')
)

SUMMARY_CSV_PATH = Path(
    resource_path('figures', 'figure_05_parameter_contributions_summary.csv')
)

# ------------------------------------------------------------
# Figure style and compact layout
# ------------------------------------------------------------

FIGSIZE = (24.0, 15.5)

PANEL_TITLE_SIZE = 24
GLOBAL_XLABEL_SIZE = 22
TICK_LABEL_SIZE = 17
PARAMETER_LABEL_SIZE = 18
VALUE_LABEL_SIZE = 16
LEGEND_SIZE = 21

SHOW_VALUE_LABELS = True

# Fixed shared x-axis limits within each row.
USE_FIXED_X_LIMITS = True

# Display scaling for bar lengths and value labels.
DISPLAY_SCALE = 1.0e4

# Fixed shared x-axis limits after multiplication by 10^4.
USE_FIXED_X_LIMITS = True

FIXED_XMAX = {
    "flash": 50.0,   # original 0.0050 × 10^4
    "hail": 27.0,    # original 0.0027 × 10^4
}

X_MAJOR_TICK = {
    "flash": 10.0,
    "hail": 5.0,
}

X_TICK_FORMAT = {
    "flash": "%.0f",
    "hail": "%.0f",
}

# Used only when USE_FIXED_X_LIMITS = False.
X_LIMIT_PADDING_RATIO = 0.15

# ============================================================
# <<< MARKER: USER_CONFIGURATION_END
# ============================================================


# ============================================================
# >>> MARKER: DATA_SUMMARY_START
# ============================================================

def calculate_signed_means(root: Path, model: str) -> pd.DataFrame:
    """
    Read one model's two-way attribution data and calculate the
    five-decade mean signed contribution for 2050s-2090s.
    """
    path = (
        root
        / "models"
        / model
        / ATTRIBUTION_DIR
        / "parameter_area_summary.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Attribution file not found:\n{path}"
        )

    data = pd.read_csv(path)

    required_columns = {
        "model",
        "target",
        "direction",
        "parameter",
        "decade",
        "signed_area_mean",
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise KeyError(
            f"{path.name} is missing columns:\n"
            f"{sorted(missing_columns)}"
        )

    data = data[
        (data["decade"].isin(DECADES))
        & (data["direction"] == DIRECTION)
    ].copy()

    if data.empty:
        raise ValueError(
            f"No valid two-way attribution records for {model}."
        )

    summary = data.groupby(
        ["model", "target", "direction", "parameter"],
        as_index=False,
    ).agg(
        five_decade_mean_signed_contribution=(
            "signed_area_mean",
            "mean",
        ),
        minimum_decadal_signed_contribution=(
            "signed_area_mean",
            "min",
        ),
        maximum_decadal_signed_contribution=(
            "signed_area_mean",
            "max",
        ),
        n_decades=("decade", "nunique"),
    )

    summary["bar_length"] = (
        summary["five_decade_mean_signed_contribution"].abs()
    )

    summary["sign"] = summary[
        "five_decade_mean_signed_contribution"
    ].apply(
        lambda value: "positive" if value >= 0 else "negative"
    )

    summary["parameter_label"] = (
        summary["parameter"]
        .map(PARAMETER_LABELS)
        .fillna(summary["parameter"])
    )

    return summary


# <<< MARKER: DATA_SUMMARY_END


# ============================================================
# >>> MARKER: X_AXIS_CONFIGURATION_START
# ============================================================

def round_up_nice(value: float, step: float) -> float:
    """Round upward to a clean multiple of step."""
    if value <= 0:
        return step

    return math.ceil(value / step) * step


def get_shared_xmax(summary: pd.DataFrame, target: str) -> float:
    """
    Return one shared x-axis maximum for all three panels
    within the Lightning row or Hail row.

    Values are displayed after multiplication by DISPLAY_SCALE.
    """
    if USE_FIXED_X_LIMITS:
        return FIXED_XMAX[target]

    subset = summary[
        (summary["target"] == target)
        & (summary["direction"] == DIRECTION)
    ].copy()

    if subset.empty:
        raise ValueError(f"No data available for target={target}")

    maximum = float(subset["bar_length"].max()) * DISPLAY_SCALE

    if not pd.notna(maximum) or maximum <= 0:
        maximum = 1.0

    padded = maximum * (1.0 + X_LIMIT_PADDING_RATIO)

    return round_up_nice(
        padded,
        X_MAJOR_TICK[target],
    )


# <<< MARKER: X_AXIS_CONFIGURATION_END


# ============================================================
# >>> MARKER: PANEL_PLOTTING_START
# ============================================================

def format_signed_value(value: float) -> str:
    """Format scaled signed values for bar-end annotations."""
    return f"{value * DISPLAY_SCALE:+.2f}"


def plot_signed_contribution_panel(
    ax,
    summary: pd.DataFrame,
    model: str,
    target: str,
    shared_xmax: float,
    show_ylabels: bool,
) -> None:
    """Draw one signed-contribution horizontal bar panel."""
    data = summary[
        (summary["model"] == model)
        & (summary["target"] == target)
        & (summary["direction"] == DIRECTION)
    ].copy()

    if data.empty:
        raise ValueError(
            f"No data for model={model}, target={target}"
        )

    data = data.sort_values("bar_length", ascending=True)

    colors = data["sign"].map({
        "positive": POSITIVE_COLOR,
        "negative": NEGATIVE_COLOR,
    }).tolist()

    display_bar_length = data["bar_length"] * DISPLAY_SCALE

    bars = ax.barh(
        y=data["parameter_label"],
        width=display_bar_length,
        color=colors,
        edgecolor="black",
        linewidth=0,
        height=0.74,
        zorder=3,
    )

    ax.set_xlim(0, shared_xmax)

    ax.xaxis.set_major_locator(
        MultipleLocator(X_MAJOR_TICK[target])
    )

    ax.xaxis.set_major_formatter(
        FormatStrFormatter(X_TICK_FORMAT[target])
    )

    if SHOW_VALUE_LABELS:
        # Slightly closer to bars than before, but still readable.
        text_offset = shared_xmax * 0.012

        for bar, signed_value in zip(
            bars,
            data["five_decade_mean_signed_contribution"],
        ):
            ax.text(
                x=bar.get_width() + text_offset,
                y=bar.get_y() + bar.get_height() / 2,
                s=format_signed_value(signed_value),
                ha="left",
                va="center",
                fontsize=VALUE_LABEL_SIZE,
                color="0.18",
                clip_on=False,
                zorder=5,
            )

    panel_label = PANEL_LABELS[(target, model)]
    target_name = TARGET_DISPLAY_NAMES[target]
    model_name = MODEL_DISPLAY_NAMES[model]

    ax.set_title(
        f"{panel_label} {target_name} | {model_name}",
        loc="left",
        fontsize=PANEL_TITLE_SIZE,
        fontweight="bold",
        pad=10,
    )

    ax.set_ylabel("")

    ax.tick_params(
        axis="x",
        labelsize=TICK_LABEL_SIZE,
        width=BORDER_WIDTH,
        length=5,
        pad=4,
    )

    ax.tick_params(
        axis="y",
        labelsize=PARAMETER_LABEL_SIZE,
        width=BORDER_WIDTH,
        length=4,
        pad=4,
    )

    if not show_ylabels:
        ax.tick_params(
            axis="y",
            labelleft=False,
            left=False,
        )

    style_plain_axis(ax, grid=True)

    for spine in ax.spines.values():
        spine.set_linewidth(BORDER_WIDTH)


# <<< MARKER: PANEL_PLOTTING_END


# ============================================================
# >>> MARKER: COMPOSITE_FIGURE_START
# ============================================================

def plot_composite_two_way_figure(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Create a compact 2 × 3 composite figure.

    Top row: Flash
    Bottom row: Hail
    Columns: CanESM5, MIROC6, BCC-CSM2-MR
    """
    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=FIGSIZE,
        dpi=FIG_DPI,
        sharey="row",
    )

    flash_xmax = get_shared_xmax(summary, "flash")
    hail_xmax = get_shared_xmax(summary, "hail")

    for col, model in enumerate(MODELS):
        plot_signed_contribution_panel(
            ax=axes[0, col],
            summary=summary,
            model=model,
            target="flash",
            shared_xmax=flash_xmax,
            show_ylabels=(col == 0),
        )

        plot_signed_contribution_panel(
            ax=axes[1, col],
            summary=summary,
            model=model,
            target="hail",
            shared_xmax=hail_xmax,
            show_ylabels=(col == 0),
        )

    # Remove repeated x-axis labels in all six panels.
    for row in range(2):
        for col in range(3):
            axes[row, col].set_xlabel("")

    # One global x-axis title beneath the panels.
    # Shared x-axis label for the Lightning row.
    fig.text(
        0.53,
        0.55,
        r"Absolute Five-decade Mean Contribution "
        r"(Lightning Frequency $\times 10^{4}$)",
        ha="center",
        va="center",
        fontsize=GLOBAL_XLABEL_SIZE,
    )

    # Shared x-axis label for the Hail row.
    fig.text(
        0.53,
        0.085,
        r"Absolute Five-decade Mean Contribution "
        r"(Hail Frequency $\times 10^{4}$)",
        ha="center",
        va="center",
        fontsize=GLOBAL_XLABEL_SIZE,
    )

    # One shared legend.
    legend_handles = [
        Patch(
            facecolor=POSITIVE_COLOR,
            edgecolor="none",
            linewidth=0.75,
            label="Positive Contribution",
        ),
        Patch(
            facecolor=NEGATIVE_COLOR,
            edgecolor="none",
            linewidth=0.75,
            label="Negative Contribution",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.02),
        ncol=2,
        frameon=False,
        fontsize=LEGEND_SIZE,
        handlelength=1.9,
        handleheight=0.95,
        columnspacing=2.8,
        handletextpad=0.8,
    )

    # Compact spacing while retaining room for labels and titles.
    fig.subplots_adjust(
        left=0.095,
        right=0.988,
        top=0.950,
        bottom=0.130,
        wspace=0.02,
        hspace=0.3,
    )

    save_figure(
        fig=fig,
        path=output_path,
        tight=True,
    )

    print(f"[SAVED] Composite figure:\n{output_path}")


# <<< MARKER: COMPOSITE_FIGURE_END


# ============================================================
# >>> MARKER: MAIN_START
# ============================================================

def main(
    root: Path = DEFAULT_ROOT,
) -> None:
    """
    Run the complete Figure 5 plotting workflow.

    Compatible with Jupyter Notebook, Spyder, VS Code, and Python scripts.
    """
    root = Path(root)

    print("=" * 100)
    print("[STEP] Plot Figure 5: two-way signed parameter contributions")
    print("=" * 100)

    summary_list = []

    for model in MODELS:
        print(f"[LOAD] {model}")
        model_summary = calculate_signed_means(
            root=root,
            model=model,
        )
        summary_list.append(model_summary)

    combined = pd.concat(
        summary_list,
        ignore_index=True,
    )

    SUMMARY_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    combined.to_csv(
        SUMMARY_CSV_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"[SAVED] Summary table:\n{SUMMARY_CSV_PATH}")

    plot_composite_two_way_figure(
        summary=combined,
        output_path=FINAL_FIG_PATH,
    )

    print("=" * 100)
    print("[DONE] Finished successfully.")
    print("=" * 100)


# <<< MARKER: MAIN_END


# ============================================================
# >>> MARKER: RUN_START
# ============================================================

# Jupyter / Spyder / VS Code interactive usage:
main()

# For direct .py execution, use:
# if __name__ == "__main__":
#     main()

# <<< MARKER: RUN_END
# ============================================================

