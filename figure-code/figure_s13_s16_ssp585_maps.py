# Purpose: figure s13 s16 ssp585 maps.
# Source: cmip6_figure_SSP585.ipynb; see docs/source_coverage.csv for cell provenance.
# Run order and required inputs: docs/workflow.md.
# Run from repository root: python -m figures.figure_s13_s16_ssp585_maps

# ============================================================
# Step 4. Plot SSP585 decadal Lightning/Hail maps
# for three individual models and the multi-model mean (MMM)
#
# Outputs
#   1. CanESM5  : 5×2 decadal Lightning/Hail maps + 2 delta maps
#   2. MIROC6   : 5×2 decadal Lightning/Hail maps + 2 delta maps
#   3. BCC-CSM2-MR : 5×2 decadal Lightning/Hail maps + 2 delta maps
#   4. MMM      : 5×2 decadal Lightning/Hail maps + 2 delta maps
#
# Key updates relative to the original scripts
#   1. Visible "Flash" text is changed to "Lightning Frequency".
#   2. All visible labels are standardized to Title Case.
#   3. The three individual models and MMM use shared color scales
#      for Lightning / Hail maps and for their delta maps.
#   4. A new MMM figure set is added with the same plotting style.
#
# Notes
#   - Internal variable names such as pred_flash / pred_hail are unchanged.
#   - Figure style, layout, region boxes, hatching, and map appearance
#     are intentionally kept as close as possible to the original scripts.
# ============================================================

from __future__ import annotations

from publication_config import resource_path

import os
from pathlib import Path

import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from matplotlib.ticker import FixedLocator
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


# ============================================================
# 0. User configuration
# ============================================================

MODEL_BASE_DIRS = {
    "CanESM5": Path(resource_path('models', 'CanESM5')),
    "MIROC6": Path(resource_path('models', 'MIROC6')),
    "BCC-CSM2-MR": Path(resource_path('models', 'BCC-CSM2-MR')),
}

# Used to provide the common MMM mask / high-altitude hatching.
REFERENCE_MASK_MODEL = "CanESM5"

DECADES = [
    (2050, 2059),
    (2060, 2069),
    (2070, 2079),
    (2080, 2089),
    (2090, 2099),
]

DECADE_LABELS = ["2050s", "2060s", "2070s", "2080s", "2090s"]

EXTENT = [-180, 180, -63, 63]
LAT_MAX_COMMON = 63.0
FIG_DPI = 600

# Output directory for the new MMM figures
MMM_OUT_DIR = Path(resource_path('figures', 'figure_s13_mmm'))
MMM_OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. Global style
# ============================================================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 18
plt.rcParams["axes.titlesize"] = 22
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["legend.fontsize"] = 20
plt.rcParams["figure.titlesize"] = 22
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["hatch.linewidth"] = 0.3

TITLE_FONTSIZE = 25
TICK_FONTSIZE = 19
CBAR_LABEL_FONTSIZE = 20
CBAR_TICK_FONTSIZE = 20


# ============================================================
# 2. Region boxes
# ============================================================

BOXES = [
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
# 3. Colormaps
# ============================================================

colors = [
    (0.92, 0.92, 0.92),
    (0.75, 0.85, 0.90),
    (0.55, 0.75, 0.90),
    (0.30, 0.65, 0.85),
    (0.25, 0.75, 0.70),
    (0.55, 0.82, 0.65),
    (0.80, 0.90, 0.60),
    (0.95, 0.90, 0.30),
    (1.00, 0.75, 0.00),
    (1.00, 0.50, 0.00),
    (0.90, 0.10, 0.10),
    (0.70, 0.00, 0.20),
    (0.80, 0.20, 0.50),
    (0.95, 0.45, 0.70),
    (1.00, 0.70, 0.85),
]

nodes = [
    0.00, 0.08, 0.15, 0.25, 0.35, 0.45, 0.55,
    0.65, 0.72, 0.78, 0.85, 0.90, 0.94, 0.97, 1.00,
]

storm_hours_cmap = mcolors.LinearSegmentedColormap.from_list(
    "storm_hours",
    list(zip(nodes, colors)),
    N=256,
)


def diverging_with_gray_center(gray=0.95, n=256):
    cols = [
        (0.23, 0.30, 0.75),
        (gray, gray, gray),
        (0.75, 0.20, 0.20),
    ]
    return mcolors.LinearSegmentedColormap.from_list(
        "blue_gray_red",
        cols,
        N=n,
    )


delta_cmap = diverging_with_gray_center(gray=0.95)


# ============================================================
# 4. Basic helpers
# ============================================================

def wrap_lon180(lon: np.ndarray) -> np.ndarray:
    lon = np.asarray(lon, dtype=float)
    return ((lon + 180.0) % 360.0) - 180.0


def prep_cyclic_2d(z2d: np.ndarray, lon: np.ndarray):
    zc, lonc = add_cyclic_point(z2d, coord=lon)
    return zc, lonc


def get_run_dir(base_dir: Path) -> Path:
    return base_dir / "ml_xgboost_result" / 'FINAL_XGB' / "PRED_SSP585"


def get_nc_dir(base_dir: Path) -> Path:
    return get_run_dir(base_dir) / "nc"


def get_out_fig_dir(base_dir: Path) -> Path:
    out_dir = get_run_dir(base_dir) / "figs_ssp585"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def crop_and_sort_latlon_2d(field2d: np.ndarray, lat: np.ndarray, lon: np.ndarray):
    lat = np.asarray(lat, dtype=float)
    lon = wrap_lon180(np.asarray(lon, dtype=float))
    field2d = np.asarray(field2d, dtype=float)

    lat_keep = np.abs(lat) <= LAT_MAX_COMMON
    if lat_keep.sum() == 0:
        raise ValueError("No latitude rows remain after applying |lat| <= 63°.")

    field2d = field2d[lat_keep, :]
    lat = lat[lat_keep]

    order = np.argsort(lon)
    lon = lon[order]
    field2d = field2d[:, order]

    return field2d, lat, lon


def crop_and_sort_mask_2d(mask2d: np.ndarray, lat: np.ndarray, lon: np.ndarray):
    lat = np.asarray(lat, dtype=float)
    lon = wrap_lon180(np.asarray(lon, dtype=float))
    mask2d = np.asarray(mask2d)

    lat_keep = np.abs(lat) <= LAT_MAX_COMMON
    if lat_keep.sum() == 0:
        raise ValueError("No latitude rows remain after applying |lat| <= 63°.")

    mask2d = mask2d[lat_keep, :]
    lat = lat[lat_keep]

    order = np.argsort(lon)
    lon = lon[order]
    mask2d = mask2d[:, order]

    return mask2d, lat, lon


def safe_savefig(fig, out_png, dpi=600):
    out_png = os.path.normpath(str(out_png))
    out_dir = os.path.dirname(out_png)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    tmp_png = out_png + ".tmp.png"

    if os.path.exists(tmp_png):
        os.remove(tmp_png)

    if os.path.exists(out_png):
        os.remove(out_png)

    fig.savefig(tmp_png, bbox_inches="tight", dpi=dpi)
    os.replace(tmp_png, out_png)


# ============================================================
# 5. Map-style helpers
# ============================================================

def add_map_features(ax):
    ax.coastlines(linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)

    gl = ax.gridlines(
        draw_labels=False,
        linewidth=0.5,
        color="gray",
        alpha=0.4,
        linestyle="--",
    )

    gl.xlocator = FixedLocator([-180, -120, -60, 0, 60, 120, 180])
    gl.ylocator = FixedLocator([-60, -30, 0, 30, 60])

    ax.set_xticks(
        [-180, -120, -60, 0, 60, 120, 180],
        crs=ccrs.PlateCarree(),
    )
    ax.set_yticks(
        [-60, -30, 0, 30, 60],
        crs=ccrs.PlateCarree(),
    )

    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_FONTSIZE,
        direction="out",
        length=4,
        width=0.8,
        top=False,
        right=False,
    )


def add_region_boxes(ax, region_boxes, linewidth=1.6, linestyle="-"):
    for lon0, lon1, lat0, lat1, color in region_boxes:
        rect = mpatches.Rectangle(
            (lon0, lat0),
            lon1 - lon0,
            lat1 - lat0,
            linewidth=linewidth,
            edgecolor=color,
            facecolor="none",
            linestyle=linestyle,
            transform=ccrs.PlateCarree(),
            zorder=6,
        )
        ax.add_patch(rect)


def add_high_alt_hatch(
    ax,
    lon,
    lat,
    high_alt_bool2d,
    hatch="//////////",
    zorder_hatch=8,
):
    ax.contourf(
        lon,
        lat,
        high_alt_bool2d.astype(int),
        levels=[0.5, 1.5],
        colors="none",
        hatches=[hatch],
        transform=ccrs.PlateCarree(),
        zorder=zorder_hatch,
    )


def apply_black_frame(ax, linewidth=1.0):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(linewidth)


# ============================================================
# 6. Statistics / color-range helpers
# ============================================================

def common_vmax_from_list(stack_list, robust=(2, 98)) -> float:
    arr = np.concatenate(
        [np.ravel(np.asarray(x, dtype=float)) for x in stack_list]
    )
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return 1.0

    _, hi = np.percentile(arr, robust)
    hi = float(hi) if np.isfinite(hi) else float(np.nanmax(arr))

    if hi <= 0:
        hi = float(np.nanmax(arr))

    return hi if hi > 0 else 1.0


def symmetric_vmax_from_list(array_list, robust=98) -> float:
    arr = np.concatenate(
        [np.ravel(np.asarray(x, dtype=float)) for x in array_list]
    )
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return 1.0

    vmax = float(np.percentile(np.abs(arr), robust))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(np.abs(arr)))

    return vmax if vmax > 0 else 1.0


# ============================================================
# 7. Data loading
# ============================================================

def load_mask_for_model(base_dir: Path):
    mask_path = get_nc_dir(base_dir) / "masks_2deg.nc"
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing masks file: {mask_path}")

    with xr.open_dataset(mask_path) as ds:
        if "lat" not in ds or "lon" not in ds:
            raise KeyError(f"{mask_path.name} must contain lat / lon.")
        if "high_alt" not in ds:
            raise KeyError(f"{mask_path.name} is missing high_alt.")

        lat = ds["lat"].values.astype(np.float32)
        lon = ds["lon"].values.astype(np.float32)
        high_alt = ds["high_alt"].values.astype(np.int8) == 1

    high_alt, lat, lon = crop_and_sort_mask_2d(high_alt, lat, lon)
    return high_alt.astype(bool), lat.astype(np.float32), lon.astype(np.float32)


def load_prediction_stack_for_model(base_dir: Path, model_name: str):
    nc_dir = get_nc_dir(base_dir)

    labels = []
    flash_stack = []
    hail_stack = []

    lat_ref = None
    lon_ref = None

    for y0, y1 in DECADES:
        nc_file = nc_dir / f"pred_ssp585_{y0}_{y1}_2deg.nc"

        if not nc_file.exists():
            raise FileNotFoundError(
                f"[{model_name}] Missing prediction file: {nc_file}"
            )

        with xr.open_dataset(nc_file) as ds:
            if "pred_flash" not in ds:
                raise KeyError(f"{nc_file.name} is missing pred_flash.")
            if "pred_hail" not in ds:
                raise KeyError(f"{nc_file.name} is missing pred_hail.")
            if "lat" not in ds or "lon" not in ds:
                raise KeyError(f"{nc_file.name} must contain lat / lon.")

            flash = ds["pred_flash"].values.astype(np.float32)
            hail = ds["pred_hail"].values.astype(np.float32)
            lat = ds["lat"].values.astype(np.float32)
            lon = ds["lon"].values.astype(np.float32)

        flash, lat_c, lon_c = crop_and_sort_latlon_2d(flash, lat, lon)
        hail, lat_h, lon_h = crop_and_sort_latlon_2d(hail, lat, lon)

        if not np.allclose(lat_c, lat_h):
            raise ValueError(f"[{model_name}] latitude mismatch within {nc_file.name}")
        if not np.allclose(lon_c, lon_h):
            raise ValueError(f"[{model_name}] longitude mismatch within {nc_file.name}")

        flash = np.where(np.isfinite(flash) & (flash < 0), 0.0, flash)
        hail = np.where(np.isfinite(hail) & (hail < 0), 0.0, hail)

        if lat_ref is None:
            lat_ref = lat_c
            lon_ref = lon_c
        else:
            if not np.allclose(lat_c, lat_ref, equal_nan=True):
                raise ValueError(f"[{model_name}] latitude mismatch among decades.")
            if not np.allclose(lon_c, lon_ref, equal_nan=True):
                raise ValueError(f"[{model_name}] longitude mismatch among decades.")

        labels.append(f"{y0}s")
        flash_stack.append(flash.astype(np.float32))
        hail_stack.append(hail.astype(np.float32))

    flash_stack = np.stack(flash_stack, axis=0)
    hail_stack = np.stack(hail_stack, axis=0)

    return flash_stack, hail_stack, lat_ref.astype(np.float32), lon_ref.astype(np.float32), labels


def load_all_models():
    model_results = {}
    common_model_flash = []
    common_model_hail = []

    lat_common = None
    lon_common = None

    for model_name, base_dir in MODEL_BASE_DIRS.items():
        flash_stack, hail_stack, lat, lon, labels = load_prediction_stack_for_model(
            base_dir=base_dir,
            model_name=model_name,
        )

        high_alt, lat_mask, lon_mask = load_mask_for_model(base_dir)

        if not np.allclose(lat, lat_mask, equal_nan=True):
            raise ValueError(f"[{model_name}] mask latitude does not match prediction latitude.")
        if not np.allclose(lon, lon_mask, equal_nan=True):
            raise ValueError(f"[{model_name}] mask longitude does not match prediction longitude.")

        model_results[model_name] = {
            "flash_stack": flash_stack,
            "hail_stack": hail_stack,
            "lat": lat,
            "lon": lon,
            "labels": labels,
            "high_alt": high_alt,
            "out_dir": get_out_fig_dir(base_dir),
        }

        if lat_common is None:
            lat_common = lat
            lon_common = lon
        else:
            if not np.allclose(lat, lat_common, equal_nan=True):
                raise ValueError(
                    f"[{model_name}] latitude axis differs from other models "
                    f"after cropping to |lat| <= 63°."
                )
            if not np.allclose(lon, lon_common, equal_nan=True):
                raise ValueError(
                    f"[{model_name}] longitude axis differs from other models "
                    f"after cropping to |lat| <= 63°."
                )

        common_model_flash.append(flash_stack)
        common_model_hail.append(hail_stack)

    # Build MMM from aligned model stacks
    flash_models_4d = np.stack(common_model_flash, axis=0)   # [model, decade, lat, lon]
    hail_models_4d = np.stack(common_model_hail, axis=0)

    flash_mmm = np.nanmean(flash_models_4d, axis=0)
    hail_mmm = np.nanmean(hail_models_4d, axis=0)

    high_alt_mmm, lat_mask, lon_mask = load_mask_for_model(MODEL_BASE_DIRS[REFERENCE_MASK_MODEL])

    if not np.allclose(lat_common, lat_mask, equal_nan=True):
        raise ValueError("Reference MMM mask latitude does not match aligned common latitude.")
    if not np.allclose(lon_common, lon_mask, equal_nan=True):
        raise ValueError("Reference MMM mask longitude does not match aligned common longitude.")

    model_results["MMM"] = {
        "flash_stack": flash_mmm.astype(np.float32),
        "hail_stack": hail_mmm.astype(np.float32),
        "lat": lat_common.astype(np.float32),
        "lon": lon_common.astype(np.float32),
        "labels": DECADE_LABELS.copy(),
        "high_alt": high_alt_mmm.astype(bool),
        "out_dir": MMM_OUT_DIR,
    }

    return model_results


# ============================================================
# 8. Plotting functions
# ============================================================

def plot_5x2_maps(
    flash_stack,
    hail_stack,
    lat,
    lon,
    labels,
    out_png,
    main_title,
    high_alt,
    vmin_lightning=0.0,
    vmax_lightning=None,
    vmin_hail=0.0,
    vmax_hail=None,
):
    left0 = 0.065
    right1 = 0.930
    bottom = 0.070
    top = 0.950

    col_gap = 0.035
    vspace = 0.060

    fig = plt.figure(figsize=(15, 13), dpi=FIG_DPI)
    proj = ccrs.PlateCarree()

    col_w = (right1 - left0 - col_gap) / 2.0
    x_left = left0
    x_right = left0 + col_w + col_gap

    h = (top - bottom) / 5.0

    axes_left = []
    axes_right = []

    m_left = None
    m_right = None

    for i in range(5):
        y0 = top - (i + 1) * h + vspace / 2
        hh = h - vspace

        # ------------------------------
        # Lightning
        # ------------------------------
        ax_l = fig.add_axes([x_left, y0, col_w, hh], projection=proj)
        ax_l.set_extent(EXTENT, crs=proj)

        ax_l.set_title(
            f"({chr(ord('a') + i)}) Lightning | {labels[i]}",
            fontsize=TITLE_FONTSIZE,
            fontweight="bold",
            pad=8,
        )

        zc_l, lonc_l = prep_cyclic_2d(flash_stack[i, :, :], lon)

        m_left = ax_l.pcolormesh(
            lonc_l,
            lat,
            zc_l,
            transform=proj,
            shading="auto",
            cmap=storm_hours_cmap,
            vmin=vmin_lightning,
            vmax=vmax_lightning,
        )

        add_map_features(ax_l)
        add_region_boxes(ax_l, BOXES)
        add_high_alt_hatch(ax_l, lon, lat, high_alt)
        apply_black_frame(ax_l, linewidth=1.0)

        # ------------------------------
        # Hail
        # ------------------------------
        ax_r = fig.add_axes([x_right, y0, col_w, hh], projection=proj)
        ax_r.set_extent(EXTENT, crs=proj)

        ax_r.set_title(
            f"({chr(ord('f') + i)}) Hail | {labels[i]}",
            fontsize=TITLE_FONTSIZE,
            fontweight="bold",
            pad=8,
        )

        zc_r, lonc_r = prep_cyclic_2d(hail_stack[i, :, :], lon)

        m_right = ax_r.pcolormesh(
            lonc_r,
            lat,
            zc_r,
            transform=proj,
            shading="auto",
            cmap=storm_hours_cmap,
            vmin=vmin_hail,
            vmax=vmax_hail,
        )

        add_map_features(ax_r)
        add_region_boxes(ax_r, BOXES)
        add_high_alt_hatch(ax_r, lon, lat, high_alt)
        apply_black_frame(ax_r, linewidth=1.0)

        axes_left.append(ax_l)
        axes_right.append(ax_r)

    fig.suptitle(
        main_title,
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
        y=0.985,
    )

    # ------------------------------
    # Colorbars
    # ------------------------------
    pos_l_top = axes_left[0].get_position()
    pos_l_bot = axes_left[-1].get_position()
    pos_r_top = axes_right[0].get_position()
    pos_r_bot = axes_right[-1].get_position()

    cbar_w = 0.015

    cax_l = fig.add_axes([
        pos_l_top.x1 + 0.02,
        pos_l_bot.y0,
        cbar_w,
        pos_l_top.y1 - pos_l_bot.y0,
    ])

    cb_l = fig.colorbar(m_left, cax=cax_l, orientation="vertical")
    cb_l.set_label("Predicted Lightning Frequency", fontsize=CBAR_LABEL_FONTSIZE)
    cb_l.ax.tick_params(labelsize=CBAR_TICK_FONTSIZE)

    cax_r = fig.add_axes([
        pos_r_top.x1 + 0.02,
        pos_r_bot.y0,
        cbar_w,
        pos_r_top.y1 - pos_r_bot.y0,
    ])

    cb_r = fig.colorbar(m_right, cax=cax_r, orientation="vertical")
    cb_r.set_label("Predicted Hail Frequency", fontsize=CBAR_LABEL_FONTSIZE)
    cb_r.ax.tick_params(labelsize=CBAR_TICK_FONTSIZE)

    safe_savefig(fig, out_png, dpi=FIG_DPI)
    plt.close(fig)

    print("[SAVED]", out_png)


def plot_delta_map(
    delta2d,
    lat,
    lon,
    title,
    out_png,
    high_alt,
    cbar_label,
    cmap=delta_cmap,
    vmax_fixed=None,
):
    fig = plt.figure(figsize=(12, 8), dpi=FIG_DPI)
    proj = ccrs.PlateCarree()

    left, right = 0.060, 0.860
    bottom, top = 0.200, 0.880

    ax = fig.add_axes(
        [left, bottom, right - left, top - bottom],
        projection=proj,
    )

    ax.set_extent(EXTENT, crs=proj)

    ax.set_title(
        title,
        fontsize=TITLE_FONTSIZE,
        pad=8,
        fontweight="bold",
    )

    zc, lonc = prep_cyclic_2d(delta2d, lon)

    if vmax_fixed is None:
        vals = zc[np.isfinite(zc)]
        vmax_fixed = np.percentile(np.abs(vals), 98) if vals.size > 0 else 1.0

    vmin = -float(vmax_fixed)
    vmax = float(vmax_fixed)

    m = ax.pcolormesh(
        lonc,
        lat,
        zc,
        transform=proj,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    add_map_features(ax)
    add_region_boxes(ax, BOXES)
    add_high_alt_hatch(ax, lon, lat, high_alt)
    apply_black_frame(ax, linewidth=1.0)

    pos = ax.get_position()
    cbar_h = 0.035
    cbar_gap = 0.060

    cax = fig.add_axes([
        pos.x0,
        pos.y0 - cbar_gap - cbar_h,
        pos.width,
        cbar_h,
    ])

    cb = fig.colorbar(m, cax=cax, orientation="horizontal")
    cb.set_label(cbar_label, fontsize=CBAR_LABEL_FONTSIZE)
    cb.ax.tick_params(labelsize=CBAR_TICK_FONTSIZE)

    safe_savefig(fig, out_png, dpi=FIG_DPI)
    plt.close(fig)

    print("[SAVED]", out_png)


# ============================================================
# 9. Main
# ============================================================

def main():
    print("=" * 100)
    print("[STEP 4] Plot SSP585 Decadal Lightning/Hail Maps for Three Models and MMM")
    print("=" * 100)

    model_results = load_all_models()

    # --------------------------------------------------------
    # Shared color ranges across the three individual models + MMM
    # --------------------------------------------------------
    flash_all_for_scale = []
    hail_all_for_scale = []
    delta_flash_all_for_scale = []
    delta_hail_all_for_scale = []

    for key, result in model_results.items():
        flash_stack = result["flash_stack"]
        hail_stack = result["hail_stack"]

        flash_all_for_scale.append(flash_stack)
        hail_all_for_scale.append(hail_stack)

        delta_flash_all_for_scale.append(flash_stack[-1] - flash_stack[0])
        delta_hail_all_for_scale.append(hail_stack[-1] - hail_stack[0])

    vmax_lightning = common_vmax_from_list(flash_all_for_scale, robust=(2, 98))
    vmax_hail = common_vmax_from_list(hail_all_for_scale, robust=(2, 98))

    vmax_delta_lightning = symmetric_vmax_from_list(delta_flash_all_for_scale, robust=98)
    vmax_delta_hail = symmetric_vmax_from_list(delta_hail_all_for_scale, robust=98)

    print("\n[SHARED COLOR RANGE]")
    print("Lightning vmax:", vmax_lightning)
    print("Hail vmax     :", vmax_hail)
    print("Delta Lightning symmetric vmax:", vmax_delta_lightning)
    print("Delta Hail symmetric vmax     :", vmax_delta_hail)

    # --------------------------------------------------------
    # Plot each model and MMM
    # --------------------------------------------------------
    for model_name, result in model_results.items():
        flash_stack = result["flash_stack"]
        hail_stack = result["hail_stack"]
        lat = result["lat"]
        lon = result["lon"]
        labels = result["labels"]
        high_alt = result["high_alt"]
        out_dir = result["out_dir"]

        print("\n" + "-" * 100)
        print(f"[PLOT] {model_name}")
        print("-" * 100)
        print("Output directory:", out_dir)
        print("flash_stack shape:", flash_stack.shape)
        print("hail_stack shape :", hail_stack.shape)
        print("high_alt count:", int(np.sum(high_alt)))
        print("Lightning finite min/max/mean:",
              float(np.nanmin(flash_stack)),
              float(np.nanmax(flash_stack)),
              float(np.nanmean(flash_stack)))
        print("Hail finite min/max/mean:",
              float(np.nanmin(hail_stack)),
              float(np.nanmax(hail_stack)),
              float(np.nanmean(hail_stack)))

        if model_name == "MMM":
            main_title = (
                "Multi-Model Mean"
                
            )
            title_delta_lightning = (
                "Multi-Model Mean: Δ Lightning Frequency "
                "(2090s - 2050s) Under SSP585"
            )
            title_delta_hail = (
                "Multi-Model Mean: Δ Hail Frequency "
                "(2090s - 2050s) Under SSP585"
            )
        else:
            main_title = (
                f"{model_name}"
            )
            title_delta_lightning = (
                f"{model_name}: Δ Lightning Frequency "
                f"(2090s - 2050s) Under SSP585"
            )
            title_delta_hail = (
                f"{model_name}: Δ Hail Frequency "
                f"(2090s - 2050s) Under SSP585"
            )

        # Keep filenames close to the original scripts for minimal disruption.
        out_5x2 = out_dir / "ssp585_flash_hail_2050s_2090s_5x2.png"
        out_delta_lightning = out_dir / "delta_flash_2090s_minus_2050s.png"
        out_delta_hail = out_dir / "delta_hail_2090s_minus_2050s.png"

        plot_5x2_maps(
            flash_stack=flash_stack,
            hail_stack=hail_stack,
            lat=lat,
            lon=lon,
            labels=labels,
            out_png=out_5x2,
            main_title=main_title,
            high_alt=high_alt,
            vmin_lightning=0.0,
            vmax_lightning=vmax_lightning,
            vmin_hail=0.0,
            vmax_hail=vmax_hail,
        )

        delta_lightning = flash_stack[-1] - flash_stack[0]
        delta_hail = hail_stack[-1] - hail_stack[0]

        plot_delta_map(
            delta2d=delta_lightning,
            lat=lat,
            lon=lon,
            title=title_delta_lightning,
            out_png=out_delta_lightning,
            high_alt=high_alt,
            cbar_label="Difference in Predicted Lightning Frequency",
            cmap=delta_cmap,
            vmax_fixed=vmax_delta_lightning,
        )

        plot_delta_map(
            delta2d=delta_hail,
            lat=lat,
            lon=lon,
            title=title_delta_hail,
            out_png=out_delta_hail,
            high_alt=high_alt,
            cbar_label="Difference in Predicted Hail Frequency",
            cmap=delta_cmap,
            vmax_fixed=vmax_delta_hail,
        )

    print("\n[DONE] All figures have been saved successfully.")


if __name__ == "__main__":
    main()