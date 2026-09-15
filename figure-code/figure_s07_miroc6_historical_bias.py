# Purpose: figure s07 miroc6 historical bias.
# Source: cmip6_figure_MIROC6_XGB_plot_latest_checked.ipynb; see docs/source_coverage.csv for cell provenance.
# Run order and required inputs: docs/workflow.md.
# Run from repository root: python -m figures.figure_s07_miroc6_historical_bias

# ============================================================
# Step 10. Plot MIROC6 historical predicted - observed maps
#
# 对应目标图：
#   Figure S9:
#   (a) Flash | Predicted (Historical) - Observed
#   (b) Hail  | Predicted (Historical) - Observed
#
# 输入：
#   1. train_table_flash_hail_withSRTM_strictLand.parquet
#   2. FLASH_model.json / HAIL_model.json
#   3. FLASH_features.txt / HAIL_features.txt
#   4. FLASH_summary.json / HAIL_summary.json
#
# 输出：
#   figure_s07_miroc6_historical_bias_maps.png
#   figure_s07_miroc6_historical_bias_vectors.csv
#   figure_s07_miroc6_historical_bias_grids.npz
# ============================================================

from __future__ import annotations

from publication_config import resource_path

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.ticker import FixedLocator


# ============================================================
# 0. Path configuration
# ============================================================

MODEL_NAME = "MIROC6"

BASE_DIR = Path(resource_path('models', 'MIROC6'))
RUN_DIR = BASE_DIR / "ml_xgboost_result" / 'FINAL_XGB'

TRAIN_TABLE = (
    BASE_DIR
    / "feature"
    / "ml_train_historical"
    / "train_table_flash_hail_withSRTM_strictLand.parquet"
)

MODEL_DIR = RUN_DIR / "models"
FEATURE_DIR = RUN_DIR / "feature_lists"
METRIC_DIR = RUN_DIR / "metrics"

FLASH_MODEL = MODEL_DIR / "FLASH_model.json"
HAIL_MODEL = MODEL_DIR / "HAIL_model.json"

FLASH_FEATURE_TXT = FEATURE_DIR / "FLASH_features.txt"
HAIL_FEATURE_TXT = FEATURE_DIR / "HAIL_features.txt"

FLASH_SUMMARY_JSON = METRIC_DIR / "FLASH_summary.json"
HAIL_SUMMARY_JSON = METRIC_DIR / "HAIL_summary.json"

# 如果 Step 3 已生成 masks_2deg.nc，可用于统一 2° 经纬度轴和高海拔掩膜
MASK_FILE = RUN_DIR / "PRED_SSP585" / "nc" / "masks_2deg.nc"

OUT_DIR = RUN_DIR / "figs_model_diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / 'figure_s07_miroc6_historical_bias_maps.png'
OUT_VECTOR_CSV = OUT_DIR / 'figure_s07_miroc6_historical_bias_vectors.csv'
OUT_GRID_NPZ = OUT_DIR / 'figure_s07_miroc6_historical_bias_grids.npz'

LAT_COL = "Latitude"
LON_COL = "Longitude"
LAND_COL = "Land_Sea"
ELEV_COL = "elev"

OBS_FLASH = "LISOTD_Flash"
OBS_HAIL = "ni_HailPF"

USE_BEST_ITERATION = True

EXTENT = [-180, 180, -63, 63]

# 是否在 Figure S9 中给 high-altitude 区域加斜线。
# 你的示例 Figure S9 更像是不加斜线；如果需要，改为 True。
DRAW_HIGH_ALT_HATCH = False
ELEV_MAX = 2000.0


# ============================================================
# 1. Layout and style
# ============================================================

FIG_DPI = 600

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 18
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = FIG_DPI
plt.rcParams["hatch.linewidth"] = 0.35

TITLE_FONTSIZE = 24
LABEL_FONTSIZE = 20
TICK_FONTSIZE = 18
CBAR_LABEL_FONTSIZE = 18
CBAR_TICK_FONTSIZE = 17
BORDER_WIDTH = 1.1

LAYOUT = {
    # 整体画布
    "figsize": (12, 11.0),

    # 两幅地图位置：[left, bottom, width, height]
    "ax_flash": [0.08, 0.6, 0.850, 0.350],
    "ax_hail":  [0.08, 0.10, 0.850, 0.350],

    # 色标设置：色标长度会自动与地图真实图框对齐
    "cbar_gap_flash": 0.060,
    "cbar_gap_hail":  0.060,
    "cbar_h": 0.026,

    # 色标标签位置：bottom 更接近你的示例图
    "cbar_label_position": "bottom",

    # 保存时是否 tight。手动布局建议 False。
    "save_bbox_tight": False,

    # 是否打印动态布局参数
    "print_layout": True,
}

# 色标范围。None 表示自动使用 98 分位数对称范围。
# 如果想强行匹配示例，可设置：
# VMAX_FLASH_DIFF = 0.016
# VMAX_HAIL_DIFF = 0.012
VMAX_FLASH_DIFF = None
VMAX_HAIL_DIFF = None


# ============================================================
# 2. Region boxes
# ============================================================

MAP_BOXES = [
    (-104, -90, 35, 47, "blue"),     # HL1
    (-68, -50, -42, -23, "blue"),    # HL2
    (-10, 20, 35, 55, "blue"),       # HL3
    (112, 138, 35, 55, "blue"),      # HL4

    (-98, -78, 24, 35, "red"),       # FL1
    (-76, -58, -10, 5, "red"),       # FL2
    (12, 35, -8, 5, "red"),          # FL3
    (95, 150, -12, 8, "red"),        # FL4
]


# ============================================================
# 3. Helper functions
# ============================================================

def wrap_lon180(lon):
    lon = np.asarray(lon, dtype=float)
    return ((lon + 180.0) % 360.0) - 180.0


def read_feature_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        features = [line.strip() for line in f if line.strip()]

    if len(features) == 0:
        raise ValueError(f"Empty feature list: {path}")

    return features


def load_booster(path: Path) -> xgb.Booster:
    if not path.exists():
        raise FileNotFoundError(path)

    booster = xgb.Booster()
    booster.load_model(str(path))
    return booster


def read_best_iteration(path: Path):
    if not path.exists():
        print(f"[WARNING] summary json not found: {path}")
        return None

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    try:
        return int(obj["metrics"]["best_iteration"])
    except Exception:
        print(f"[WARNING] cannot read best_iteration from: {path}")
        return None


def ensure_columns(df: pd.DataFrame, cols: list[str], label: str):
    missing = [c for c in cols if c not in df.columns]

    if missing:
        raise KeyError(
            f"{label} missing columns: "
            f"{missing[:30]}{' ...' if len(missing) > 30 else ''}"
        )


def predict_with_booster(
    df: pd.DataFrame,
    features: list[str],
    booster: xgb.Booster,
    best_iteration,
    target_name: str,
) -> np.ndarray:
    ensure_columns(df, features, f"{target_name}_features")

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

    pred = np.asarray(pred, dtype=np.float64)

    return pred


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


def symmetric_vmax(arr, pct=98, fallback=1.0):
    v = np.asarray(arr, dtype=float)
    v = v[np.isfinite(v)]

    if v.size == 0:
        return fallback

    vmax = np.nanpercentile(np.abs(v), pct)

    if not np.isfinite(vmax) or vmax <= 0:
        vmax = np.nanmax(np.abs(v))

    if not np.isfinite(vmax) or vmax <= 0:
        vmax = fallback

    return float(vmax)


def load_grid_axes_from_mask_or_table(df: pd.DataFrame):
    """
    优先使用 masks_2deg.nc 的完整 2° 网格；
    如果不存在，则根据 historical train table 的经纬度构建网格。
    """
    if MASK_FILE.exists():
        print("[GRID] Use mask grid:", MASK_FILE)

        with xr.open_dataset(MASK_FILE) as ds:
            lat = ds["lat"].values.astype(float)
            lon = wrap_lon180(ds["lon"].values.astype(float))

            strict_land = ds["strict_land"].values.astype(bool) if "strict_land" in ds else None
            high_alt = ds["high_alt"].values.astype(bool) if "high_alt" in ds else None
            elev = ds["elevation_m"].values.astype(float) if "elevation_m" in ds else None

        order = np.argsort(lon)
        lon = lon[order]

        if strict_land is not None:
            strict_land = strict_land[:, order]

        if high_alt is not None:
            high_alt = high_alt[:, order]

        if elev is not None:
            elev = elev[:, order]

        return lat, lon, strict_land, high_alt, elev

    print("[GRID] Use train-table grid.")

    lat = np.sort(df[LAT_COL].astype(float).unique())
    lon = np.sort(wrap_lon180(df[LON_COL].astype(float).unique()))

    strict_land = None
    high_alt = None
    elev = None

    return lat, lon, strict_land, high_alt, elev


def points_to_grid(df: pd.DataFrame, values: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    grid = np.full((len(lat), len(lon)), np.nan, dtype=np.float32)

    lat_key = np.round(np.asarray(lat, dtype=float), 6)
    lon_key = np.round(np.asarray(lon, dtype=float), 6)

    lat_to_i = {v: i for i, v in enumerate(lat_key)}
    lon_to_j = {v: j for j, v in enumerate(lon_key)}

    dlat = np.round(df[LAT_COL].astype(float).to_numpy(), 6)
    dlon = np.round(wrap_lon180(df[LON_COL].astype(float).to_numpy()), 6)

    n_inside = 0
    n_outside = 0

    for la, lo, val in zip(dlat, dlon, values):
        if la in lat_to_i and lo in lon_to_j:
            grid[lat_to_i[la], lon_to_j[lo]] = val
            n_inside += 1
        else:
            n_outside += 1

    print(f"[GRID] points inside grid={n_inside}, outside={n_outside}")

    return grid


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


def add_high_alt_hatch(ax, lon, lat, high_alt):
    if high_alt is None:
        return

    if not DRAW_HIGH_ALT_HATCH:
        return

    ax.contourf(
        lon,
        lat,
        high_alt.astype(int),
        levels=[0.5, 1.5],
        colors="none",
        hatches=["////////"],
        transform=ccrs.PlateCarree(),
        zorder=9,
    )


def plot_one_map(
    ax,
    diff_grid,
    lat,
    lon,
    title,
    cmap,
    vmax,
    high_alt=None,
):
    proj = ccrs.PlateCarree()

    zc, lonc = add_cyclic_point(diff_grid, coord=lon)

    m = ax.pcolormesh(
        lonc,
        lat,
        zc,
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        shading="auto",
        transform=proj,
        zorder=1,
    )

    add_map_features(ax)
    add_region_boxes(ax)
    add_high_alt_hatch(ax, lon, lat, high_alt)

    ax.set_title(
        title,
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
        pad=8,
    )

    return m


def style_horizontal_colorbar(cb, label):
    cb.set_label(
        label,
        fontsize=CBAR_LABEL_FONTSIZE,
        labelpad=4,
    )

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

    cb.outline.set_linewidth(1.0)


def align_colorbars_to_maps(fig, ax_flash, ax_hail, cax_flash, cax_hail):
    fig.canvas.draw()

    pos_f = ax_flash.get_position()
    pos_h = ax_hail.get_position()

    cax_flash.set_position([
        pos_f.x0,
        pos_f.y0 - LAYOUT["cbar_gap_flash"],
        pos_f.width,
        LAYOUT["cbar_h"],
    ])

    cax_hail.set_position([
        pos_h.x0,
        pos_h.y0 - LAYOUT["cbar_gap_hail"],
        pos_h.width,
        LAYOUT["cbar_h"],
    ])

    if LAYOUT["print_layout"]:
        print("\n[DYNAMIC LAYOUT]")
        print("Flash map position:", pos_f)
        print("Hail  map position:", pos_h)
        print("Flash cbar position:", cax_flash.get_position())
        print("Hail  cbar position:", cax_hail.get_position())


def safe_savefig(fig, out_png: Path, dpi=600):
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    tmp_png = Path(str(out_png) + ".tmp.png")

    if tmp_png.exists():
        tmp_png.unlink()

    if out_png.exists():
        out_png.unlink()

    if LAYOUT["save_bbox_tight"]:
        fig.savefig(tmp_png, dpi=dpi, bbox_inches="tight")
    else:
        fig.savefig(tmp_png, dpi=dpi)

    tmp_png.replace(out_png)


# ============================================================
# 4. Main
# ============================================================

def main():
    print("=" * 100)
    print("[STEP 10] Plot historical predicted - observed maps")
    print("=" * 100)

    if not TRAIN_TABLE.exists():
        raise FileNotFoundError(TRAIN_TABLE)

    print("[READ]", TRAIN_TABLE)
    df = pd.read_parquet(TRAIN_TABLE)

    ensure_columns(
        df,
        [LAT_COL, LON_COL, OBS_FLASH, OBS_HAIL],
        "historical train table",
    )

    if LAND_COL in df.columns:
        df = df[df[LAND_COL] == 0].copy()

    df = df[np.abs(df[LAT_COL].astype(float)) <= 63].copy()
    df = df.reset_index(drop=True)

    print("[DATA] strict-land rows:", len(df))

    obs_flash = pd.to_numeric(df[OBS_FLASH], errors="coerce").to_numpy(dtype=float)
    obs_hail = pd.to_numeric(df[OBS_HAIL], errors="coerce").to_numpy(dtype=float)

    print("[LOAD MODELS]")
    bst_flash = load_booster(FLASH_MODEL)
    bst_hail = load_booster(HAIL_MODEL)

    features_flash = read_feature_list(FLASH_FEATURE_TXT)
    features_hail = read_feature_list(HAIL_FEATURE_TXT)

    best_iter_flash = read_best_iteration(FLASH_SUMMARY_JSON)
    best_iter_hail = read_best_iteration(HAIL_SUMMARY_JSON)

    print("FLASH features:", len(features_flash), "best_iteration:", best_iter_flash)
    print("HAIL  features:", len(features_hail), "best_iteration:", best_iter_hail)

    pred_flash = predict_with_booster(
        df=df,
        features=features_flash,
        booster=bst_flash,
        best_iteration=best_iter_flash,
        target_name="FLASH",
    )

    pred_hail = predict_with_booster(
        df=df,
        features=features_hail,
        booster=bst_hail,
        best_iteration=best_iter_hail,
        target_name="HAIL",
    )

    # 负值裁剪为 0
    n_neg_flash = int(np.sum(np.isfinite(pred_flash) & (pred_flash < 0)))
    n_neg_hail = int(np.sum(np.isfinite(pred_hail) & (pred_hail < 0)))

    pred_flash = np.where(np.isfinite(pred_flash) & (pred_flash < 0), 0.0, pred_flash)
    pred_hail = np.where(np.isfinite(pred_hail) & (pred_hail < 0), 0.0, pred_hail)

    print("Negative FLASH predictions before clip:", n_neg_flash)
    print("Negative HAIL predictions before clip :", n_neg_hail)

    diff_flash = pred_flash - obs_flash
    diff_hail = pred_hail - obs_hail

    out_vec = pd.DataFrame({
        LAT_COL: df[LAT_COL].to_numpy(),
        LON_COL: df[LON_COL].to_numpy(),
        "obs_flash": obs_flash,
        "obs_hail": obs_hail,
        "pred_flash": pred_flash,
        "pred_hail": pred_hail,
        "diff_flash": diff_flash,
        "diff_hail": diff_hail,
    })

    out_vec.to_csv(
        OUT_VECTOR_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print("[SAVED]", OUT_VECTOR_CSV)

    # --------------------------------------------------------
    # Grid conversion
    # --------------------------------------------------------

    lat, lon, strict_land, high_alt, elev = load_grid_axes_from_mask_or_table(df)

    grid_diff_flash = points_to_grid(df, diff_flash, lat, lon)
    grid_diff_hail = points_to_grid(df, diff_hail, lat, lon)
    grid_pred_flash = points_to_grid(df, pred_flash, lat, lon)
    grid_pred_hail = points_to_grid(df, pred_hail, lat, lon)
    grid_obs_flash = points_to_grid(df, obs_flash, lat, lon)
    grid_obs_hail = points_to_grid(df, obs_hail, lat, lon)

    np.savez_compressed(
        OUT_GRID_NPZ,
        lat=lat.astype(np.float32),
        lon=lon.astype(np.float32),
        diff_flash=grid_diff_flash.astype(np.float32),
        diff_hail=grid_diff_hail.astype(np.float32),
        pred_flash=grid_pred_flash.astype(np.float32),
        pred_hail=grid_pred_hail.astype(np.float32),
        obs_flash=grid_obs_flash.astype(np.float32),
        obs_hail=grid_obs_hail.astype(np.float32),
        high_alt=high_alt.astype(np.int8) if high_alt is not None else np.array([]),
        elev=elev.astype(np.float32) if elev is not None else np.array([]),
    )

    print("[SAVED]", OUT_GRID_NPZ)

    # --------------------------------------------------------
    # Color range
    # --------------------------------------------------------

    if VMAX_FLASH_DIFF is None:
        vmax_flash = symmetric_vmax(grid_diff_flash, pct=98, fallback=0.015)
    else:
        vmax_flash = float(VMAX_FLASH_DIFF)

    if VMAX_HAIL_DIFF is None:
        vmax_hail = symmetric_vmax(grid_diff_hail, pct=98, fallback=0.010)
    else:
        vmax_hail = float(VMAX_HAIL_DIFF)

    print("[COLOR RANGE]")
    print("vmax_flash:", vmax_flash)
    print("vmax_hail :", vmax_hail)

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    cmap = make_diverging_cmap()

    fig = plt.figure(figsize=LAYOUT["figsize"], dpi=FIG_DPI)
    proj = ccrs.PlateCarree()

    ax_flash = fig.add_axes(LAYOUT["ax_flash"], projection=proj)
    ax_hail = fig.add_axes(LAYOUT["ax_hail"], projection=proj)

    # 占位，后续自动和地图真实图框对齐
    cax_flash = fig.add_axes([0.05, 0.05, 0.10, 0.02])
    cax_hail = fig.add_axes([0.05, 0.02, 0.10, 0.02])

    m_flash = plot_one_map(
        ax=ax_flash,
        diff_grid=grid_diff_flash,
        lat=lat,
        lon=lon,
        title="(a) Flash | Predicted (Historical) – Observed",
        cmap=cmap,
        vmax=vmax_flash,
        high_alt=high_alt,
    )

    m_hail = plot_one_map(
        ax=ax_hail,
        diff_grid=grid_diff_hail,
        lat=lat,
        lon=lon,
        title="(b) Hail | Predicted (Historical) – Observed",
        cmap=cmap,
        vmax=vmax_hail,
        high_alt=high_alt,
    )

    align_colorbars_to_maps(
        fig=fig,
        ax_flash=ax_flash,
        ax_hail=ax_hail,
        cax_flash=cax_flash,
        cax_hail=cax_hail,
    )

    cb_flash = fig.colorbar(
        m_flash,
        cax=cax_flash,
        orientation="horizontal",
    )

    style_horizontal_colorbar(
        cb_flash,
        "Predicted - Observed",
    )

    cb_hail = fig.colorbar(
        m_hail,
        cax=cax_hail,
        orientation="horizontal",
    )

    style_horizontal_colorbar(
        cb_hail,
        "Predicted - Observed",
    )

    safe_savefig(fig, OUT_PNG, dpi=FIG_DPI)
    plt.close(fig)

    print("\n[SAVED FIGURE]")
    print(OUT_PNG)

    print("\n[DONE] Step 10 finished successfully.")


if __name__ == "__main__":
    main()