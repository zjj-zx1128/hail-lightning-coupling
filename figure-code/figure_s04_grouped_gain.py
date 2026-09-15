# Purpose: figure s04 grouped gain.
# Source: cmip6_figure_gain_3model.ipynb; see docs/source_coverage.csv for cell provenance.
# Run order and required inputs: docs/workflow.md.
# Run from repository root: python -m figures.figure_s04_grouped_gain

# ============================================================
# Step 7. Plot grouped XGBoost gain importance for three models
#
# Figure layout (independent y-axis sorting in every panel):
#   (a) Lightning Frequency | CanESM5
#   (b) Lightning Frequency | MIROC6
#   (c) Lightning Frequency | BCC-CSM2-MR
#   (d) Hail Frequency      | CanESM5
#   (e) Hail Frequency      | MIROC6
#   (f) Hail Frequency      | BCC-CSM2-MR
#
# Scientific definition:
#   1. Sum XGBoost gain across all statistical features belonging to each
#      environmental parameter group.
#   2. Divide each grouped gain by the total grouped gain across the 15 groups.
#   3. Display the resulting relative gain as a percentage.
#
# Important plotting rule:
#   Each model-target panel is sorted independently from high to low gain.
#   Therefore, the six panels do NOT share a y-axis.
#
# Input for each model:
#   <RUN_DIR>\importance\FLASH_gain.csv
#   <RUN_DIR>\importance\HAIL_gain.csv
#
# Outputs:
#   [configured path; see config.json]
#   [configured path; see config.json]
# ============================================================

from __future__ import annotations

from publication_config import resource_path

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


# ============================================================
# 0. User configuration
# ============================================================

# Keep BCC input path consistent with the existing local directory name.
MODEL_RUN_DIRS = {
    "CanESM5": Path(
        resource_path('models', 'CanESM5/ml_xgboost_result/FINAL_XGB')
    ),
    "MIROC6": Path(
        resource_path('models', 'MIROC6/ml_xgboost_result/FINAL_XGB')
    ),
    "BCC-CSM2-MR": Path(
        resource_path('models', 'BCC-CSM2-MR/ml_xgboost_result/FINAL_XGB')
    ),
}

MODEL_ORDER = ["CanESM5", "MIROC6", "BCC-CSM2-MR"]
TARGET_ORDER = ["flash", "hail"]

TARGET_DISPLAY_NAMES = {
    "flash": "Lightning Frequency",
    "hail": "Hail Frequency",
}

TARGET_FILE_STEMS = {
    "flash": "FLASH_gain.csv",
    "hail": "HAIL_gain.csv",
}

OUT_DIR = Path(resource_path('figures', ''))
OUT_FIG = OUT_DIR / 'figure_s04_grouped_gain.jpg'
OUT_SUMMARY_CSV = OUT_DIR / 'figure_s04_grouped_gain_summary.csv'

FIG_DPI = 600


# ============================================================
# 1. Figure style and layout
# ============================================================

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 18,
    "axes.unicode_minus": False,
    "figure.dpi": FIG_DPI,
    "savefig.dpi": FIG_DPI,
})

PANEL_TITLE_SIZE = 24
PARAMETER_LABEL_SIZE = 17
TICK_LABEL_SIZE = 16
VALUE_LABEL_SIZE = 14
GLOBAL_XLABEL_SIZE = 21
BORDER_WIDTH = 1.2

BAR_COLOR = "#2C7FB8"

LAYOUT = {
    # Match the visual scale of the six-panel Figure 5.
    "figsize": (24.0, 15.5),

    # Enough horizontal separation for independent parameter labels
    # in all three columns.
    "left": 0.075,
    "right": 0.985,
    "bottom": 0.100,
    "top": 0.925,
    "wspace": 0.22,
    "hspace": 0.15,

    "bar_height": 0.72,
    "label_dx_frac": 0.012,
    "save_bbox_tight": False,
}


# ============================================================
# 2. Parameter groups and Figure 5-consistent display labels
# ============================================================

GROUP_ORDER = [
    "mucape",
    "mucin",
    "pw",
    "s06",
    "h_10c",
    "h_30c",
    "h_dif",
    "div500",
    "td_sfc",
    "theta_e",
    "cape",
    "cin",
    "flh",
    "k_index",
    "elev",
]

# First 14 labels are aligned with the labels used in the main Figure 5.
# Elevation is retained because it is included in the XGBoost gain inputs.
GROUP_LABELS = {
    "mucape": "MUCAPE",
    "mucin": "MUCIN",
    "pw": "PW",
    "s06": "S06",
    "h_10c": r"$\mathrm{H}_{-10}$",
    "h_30c": r"$\mathrm{H}_{-30}$",
    "h_dif": r"$\mathrm{\Delta H}_{-10,-30}$",
    "div500": r"$\mathrm{Div}_{500}$",
    "td_sfc": r"$\mathrm{T}_{\mathrm{d,sfc}}$",
    "theta_e": r"$\mathrm{\theta}_{\mathrm{e}}$",
    "cape": "CAPE",
    "cin": "CIN",
    "flh": "FLH",
    "k_index": "K index",
    "elev": "Elevation",
}


# ============================================================
# 3. Data helpers
# ============================================================

def infer_group(feature_name: str) -> str:
    """Assign one raw XGBoost feature to one of the 15 parameter groups."""
    feature = str(feature_name).lower()

    if (
        "elev" in feature
        or "elevation" in feature
        or "srtm" in feature
        or "dem" in feature
    ):
        return "elev"

    if feature.startswith("env_"):
        feature = feature[4:]

    # Check longer/more specific strings before shorter strings.
    if (
        feature.startswith("dh_10_30")
        or feature.startswith("h_dif")
        or feature.startswith("h_diff")
    ):
        return "h_dif"
    if feature.startswith("theta_e"):
        return "theta_e"
    if feature.startswith("td_sfc"):
        return "td_sfc"
    if feature.startswith("k_index"):
        return "k_index"
    if feature.startswith("h_10c"):
        return "h_10c"
    if feature.startswith("h_30c"):
        return "h_30c"
    if feature.startswith("div500"):
        return "div500"
    if feature.startswith("mucape"):
        return "mucape"
    if feature.startswith("mucin"):
        return "mucin"
    if feature.startswith("cape"):
        return "cape"
    if feature.startswith("cin"):
        return "cin"
    if feature.startswith("pw"):
        return "pw"
    if feature.startswith("s06"):
        return "s06"
    if feature.startswith("flh"):
        return "flh"

    return "other"


def read_gain_csv(path: Path, model: str, target: str) -> pd.DataFrame:
    """Read one raw feature-level gain CSV and infer parameter groups."""
    if not path.exists():
        raise FileNotFoundError(f"Gain file not found:\n{path}")

    data = pd.read_csv(path)
    required = {"feature", "gain"}
    missing = required - set(data.columns)

    if missing:
        raise KeyError(
            f"{path.name} is missing required columns: {sorted(missing)}"
        )

    data = data.copy()
    data["model"] = model
    data["target"] = target
    data["feature"] = data["feature"].astype(str)
    data["gain"] = pd.to_numeric(data["gain"], errors="coerce").fillna(0.0)
    data["group"] = data["feature"].map(infer_group)

    return data


def aggregate_group_gain(raw_gain: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw feature gain into the 15 environmental parameter groups."""
    model = raw_gain["model"].iloc[0]
    target = raw_gain["target"].iloc[0]

    grouped = (
        raw_gain.groupby("group", as_index=False)["gain"]
        .sum()
    )

    other_gain = grouped.loc[grouped["group"] == "other", "gain"].sum()
    if np.isfinite(other_gain) and other_gain > 0:
        unknown = raw_gain.loc[
            raw_gain["group"] == "other", ["feature", "gain"]
        ].sort_values("gain", ascending=False)
        raise ValueError(
            f"[{model} | {target}] Unclassified gain is non-zero "
            f"({other_gain:.8g}). Please classify these features before plotting:\n"
            f"{unknown.to_string(index=False)}"
        )

    expected = pd.DataFrame({"group": GROUP_ORDER})
    grouped = expected.merge(grouped, on="group", how="left")
    grouped["gain"] = grouped["gain"].fillna(0.0)

    total_gain = float(grouped["gain"].sum())
    if not np.isfinite(total_gain) or total_gain <= 0:
        raise ValueError(f"[{model} | {target}] Total grouped gain is not positive.")

    grouped["gain_fraction"] = grouped["gain"] / total_gain
    grouped["gain_percent"] = grouped["gain_fraction"] * 100.0
    grouped["model"] = model
    grouped["target"] = target
    grouped["parameter_label"] = grouped["group"].map(GROUP_LABELS)

    # Independent descending order in every model-target panel.
    grouped = grouped.sort_values(
        "gain_fraction", ascending=False, kind="mergesort"
    ).reset_index(drop=True)
    grouped["rank"] = np.arange(1, len(grouped) + 1, dtype=int)

    return grouped[
        [
            "model",
            "target",
            "rank",
            "group",
            "parameter_label",
            "gain",
            "gain_fraction",
            "gain_percent",
        ]
    ]


def load_all_grouped_gain() -> pd.DataFrame:
    """Read and aggregate both targets for each of the three models."""
    summaries = []

    for model in MODEL_ORDER:
        run_dir = MODEL_RUN_DIRS[model]
        importance_dir = run_dir / "importance"

        for target in TARGET_ORDER:
            source = importance_dir / TARGET_FILE_STEMS[target]
            print(f"[READ] {model:<12s} | {target:<5s} | {source}")
            raw = read_gain_csv(source, model=model, target=target)
            grouped = aggregate_group_gain(raw)
            summaries.append(grouped)

    summary = pd.concat(summaries, axis=0, ignore_index=True)

    summary["target_display"] = summary["target"].map(TARGET_DISPLAY_NAMES)
    model_rank = {name: i for i, name in enumerate(MODEL_ORDER)}
    target_rank = {name: i for i, name in enumerate(TARGET_ORDER)}
    summary["_model_order"] = summary["model"].map(model_rank)
    summary["_target_order"] = summary["target"].map(target_rank)

    summary = summary.sort_values(
        ["_target_order", "_model_order", "rank"],
        kind="mergesort",
    ).drop(columns=["_model_order", "_target_order"])

    return summary.reset_index(drop=True)


# ============================================================
# 4. Axis-range and drawing helpers
# ============================================================

def choose_nice_tick_step(vmax_percent: float, target_major_intervals: int = 5) -> float:
    """Return a readable percentage tick interval."""
    if not np.isfinite(vmax_percent) or vmax_percent <= 0:
        return 1.0

    raw_step = vmax_percent / float(target_major_intervals)
    power = 10.0 ** math.floor(math.log10(raw_step))

    for multiplier in (1.0, 2.0, 2.5, 5.0, 10.0):
        step = multiplier * power
        if step >= raw_step:
            return float(step)

    return float(10.0 * power)


def get_shared_row_xlim(summary: pd.DataFrame, target: str) -> tuple[float, float, float]:
    """Create one common x-axis range for the three model panels in one row."""
    values = summary.loc[
        summary["target"] == target, "gain_percent"
    ].to_numpy(dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return 0.0, 1.0, 0.2

    padded_max = float(np.max(values)) * 1.15
    tick_step = choose_nice_tick_step(padded_max, target_major_intervals=5)
    x_max = math.ceil(padded_max / tick_step) * tick_step

    # Ensure a minimal axis range even in unusual test cases.
    x_max = max(float(x_max), float(tick_step))
    return 0.0, float(x_max), float(tick_step)


def style_axis(ax) -> None:
    """Apply the non-map style used by the main Figure 5."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(BORDER_WIDTH)

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=5.0,
        width=1.0,
        top=False,
        right=False,
        pad=4,
    )

    # Keep only x-direction dashed gridlines.
    ax.grid(
        axis="x",
        linestyle="--",
        linewidth=0.65,
        color="0.72",
        alpha=0.55,
        zorder=0,
    )
    ax.set_axisbelow(True)


def plot_one_panel(
    ax,
    panel_data: pd.DataFrame,
    panel_label: str,
    target: str,
    model: str,
    xlim: tuple[float, float],
    xtick_step: float,
) -> None:
    """Draw one independently sorted grouped-gain panel."""
    data = panel_data.copy().sort_values("rank", kind="mergesort")

    y = np.arange(len(data))
    values = data["gain_percent"].to_numpy(dtype=float)
    labels = data["parameter_label"].tolist()

    ax.barh(
        y,
        values,
        height=LAYOUT["bar_height"],
        color=BAR_COLOR,
        edgecolor="none",   # Requested: no outer bar border.
        linewidth=0.0,
        zorder=3,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=PARAMETER_LABEL_SIZE)
    ax.invert_yaxis()

    ax.set_xlim(xlim)
    ax.xaxis.set_major_locator(MultipleLocator(xtick_step))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax.tick_params(axis="x", labelsize=TICK_LABEL_SIZE)
    ax.tick_params(axis="y", labelsize=PARAMETER_LABEL_SIZE)

    ax.set_title(
        f"{panel_label} {TARGET_DISPLAY_NAMES[target]} | {model}",
        loc="left",
        fontsize=PANEL_TITLE_SIZE,
        fontweight="bold",
        pad=10,
    )

    dx = (xlim[1] - xlim[0]) * LAYOUT["label_dx_frac"]
    for yi, value in zip(y, values):
        text_x = value + dx
        ha = "left"

        # Move the label inside a near-limit bar instead of clipping it.
        if text_x > xlim[1] * 0.985:
            text_x = max(value - dx, xlim[0] + dx)
            ha = "right"

        ax.text(
            text_x,
            yi,
            f"{value:.1f}%",
            va="center",
            ha=ha,
            fontsize=VALUE_LABEL_SIZE,
            color="0.18",
            zorder=5,
        )

    style_axis(ax)


def save_figure(fig, out_path: Path) -> None:
    """Save the finished figure without changing manual canvas geometry."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"dpi": FIG_DPI}
    if LAYOUT["save_bbox_tight"]:
        save_kwargs["bbox_inches"] = "tight"

    fig.savefig(out_path, **save_kwargs)
    plt.close(fig)


# ============================================================
# 5. Main
# ============================================================

def main() -> None:
    print("=" * 100)
    print("[STEP 7] Plot three-model grouped XGBoost gain importance")
    print("=" * 100)

    summary = load_all_grouped_gain()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    print("\n[SAVED SUMMARY CSV]")
    print(OUT_SUMMARY_CSV)

    for target in TARGET_ORDER:
        print(f"\n[{TARGET_DISPLAY_NAMES[target]}]")
        for model in MODEL_ORDER:
            subset = summary[
                (summary["target"] == target)
                & (summary["model"] == model)
            ]
            print(f"\n{model}")
            print(subset[["rank", "group", "gain_percent"]].to_string(index=False))

    row_scales = {
        target: get_shared_row_xlim(summary, target)
        for target in TARGET_ORDER
    }

    print("\n[SHARED ROW X-AXIS RANGES]")
    for target, (x0, x1, step) in row_scales.items():
        print(
            f"{TARGET_DISPLAY_NAMES[target]}: "
            f"xlim=({x0:.1f}, {x1:.1f}), tick step={step:.1f}"
        )

    fig = plt.figure(figsize=LAYOUT["figsize"], dpi=FIG_DPI)
    grid = fig.add_gridspec(
        nrows=2,
        ncols=3,
        left=LAYOUT["left"],
        right=LAYOUT["right"],
        bottom=LAYOUT["bottom"],
        top=LAYOUT["top"],
        wspace=LAYOUT["wspace"],
        hspace=LAYOUT["hspace"],
    )

    panel_labels = {
        ("flash", "CanESM5"): "(a)",
        ("flash", "MIROC6"): "(b)",
        ("flash", "BCC-CSM2-MR"): "(c)",
        ("hail", "CanESM5"): "(d)",
        ("hail", "MIROC6"): "(e)",
        ("hail", "BCC-CSM2-MR"): "(f)",
    }

    for row, target in enumerate(TARGET_ORDER):
        x0, x1, tick_step = row_scales[target]

        for col, model in enumerate(MODEL_ORDER):
            ax = fig.add_subplot(grid[row, col])
            panel_data = summary[
                (summary["target"] == target)
                & (summary["model"] == model)
            ]

            plot_one_panel(
                ax=ax,
                panel_data=panel_data,
                panel_label=panel_labels[(target, model)],
                target=target,
                model=model,
                xlim=(x0, x1),
                xtick_step=tick_step,
            )

    fig.text(
        0.5,
        0.05,
        "Relative Gain (%)",
        ha="center",
        va="center",
        fontsize=GLOBAL_XLABEL_SIZE,
    )

    save_figure(fig, OUT_FIG)

    print("\n[SAVED FIGURE]")
    print(OUT_FIG)
    print("\n[DONE] Step 7 finished successfully.")


if __name__ == "__main__":
    main()
