

from __future__ import annotations

from publication_config import resource_path

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

REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL_SPECS = {
    "CanESM5": {
        "output_tag": "canesm5",
        "base_dir": Path(resource_path("models", "CanESM5")),
        "train_table_rel": (
            Path("ml_feature_withSRTM")
            / "train_table_flash_hail_withSRTM_strictLand.parquet"
        ),
    },
    "MIROC6": {
        "output_tag": "miroc6",
        "base_dir": Path(resource_path("models", "MIROC6")),
        "train_table_rel": (
            Path("feature")
            / "ml_train_historical"
            / "train_table_flash_hail_withSRTM_strictLand.parquet"
        ),
    },
    "BCC-CSM2-MR": {
        "output_tag": "bcc_csm2_mr",
        "base_dir": Path(resource_path("models", "BCC-CSM2-MR")),
        "train_table_rel": (
            Path("feature")
            / "ml_train_historical"
            / "train_table_flash_hail_withSRTM_strictLand.parquet"
        ),
    },
}

for _spec in MODEL_SPECS.values():
    _spec["run_dir"] = _spec["base_dir"] / "ml_xgboost_result" / "FINAL_XGB"
    _spec["train_table"] = _spec["base_dir"] / _spec["train_table_rel"]

OUT_DIR = REPO_ROOT / "outputs" / "figure_s06_historical_bias"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REGIONAL_MEAN_CSV = OUT_DIR / "figure_s06_historical_bias_regional_means.csv"

LAT_COL = "Latitude"
LON_COL = "Longitude"
LAND_COL = "Land_Sea"
ELEV_COL = "elev"

OBS_FLASH = "LISOTD_Flash"
OBS_HAIL = "ni_HailPF"

USE_BEST_ITERATION = True

EXTENT = [-180, 180, -63, 63]
DRAW_HIGH_ALT_HATCH = False
ELEV_MAX = 2000.0
FILTER_HIGH_ALT_FOR_REGION_MEAN = False


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
    "figsize": (12, 11.0),
    "ax_flash": [0.08, 0.6, 0.850, 0.350],
    "ax_hail":  [0.08, 0.10, 0.850, 0.350],
    "cbar_gap_flash": 0.060,
    "cbar_gap_hail":  0.060,
    "cbar_h": 0.026,
    "cbar_label_position": "bottom",
    "save_bbox_tight": False,
    "print_layout": True,
}

# Keep the original per-figure automatic symmetric range logic.
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

REGION_SPECS = [
    ("HL1", -104, -90, 35, 47),
    ("HL2", -68, -50, -42, -23),
    ("HL3", -10, 20, 35, 55),
    ("HL4", 112, 138, 35, 55),
    ("FL1", -98, -78, 24, 35),
    ("FL2", -76, -58, -10, 5),
    ("FL3", 12, 35, -8, 5),
    ("FL4", 95, 150, -12, 8),
]

# ============================================================
# 3. Helper functions (mostly preserved from the original code)
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
        raise KeyError(f"{label} missing columns: {missing[:30]}{' ...' if len(missing) > 30 else ''}")


def predict_with_booster(df: pd.DataFrame, features: list[str], booster: xgb.Booster, best_iteration, target_name: str) -> np.ndarray:
    ensure_columns(df, features, f"{target_name}_features")
    X = df[features].copy().replace([np.inf, -np.inf], np.nan)
    dmx = xgb.DMatrix(X, feature_names=features, missing=np.nan)
    if USE_BEST_ITERATION and best_iteration is not None:
        pred = booster.predict(dmx, iteration_range=(0, int(best_iteration) + 1))
    else:
        pred = booster.predict(dmx)
    return np.asarray(pred, dtype=np.float64)


def make_diverging_cmap():
    return mcolors.LinearSegmentedColormap.from_list(
        "blue_gray_red",
        [(0.20, 0.30, 0.75), (0.95, 0.95, 0.95), (0.75, 0.20, 0.20)],
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


def load_grid_axes_from_mask_or_table(df: pd.DataFrame, mask_file: Path):
    """Load the plotting grid and enforce the common analysis domain |lat| <= 63°.

    The historical training table is already restricted to |lat| <= 63°. Some
    masks_2deg.nc files retain the two extra latitude rows at ±64°, whereas
    others do not. The explicit latitude filter here makes all model grids
    consistent before the MMM calculation, without changing the figure style.
    """
    if mask_file.exists():
        print("[GRID] Use mask grid:", mask_file)
        with xr.open_dataset(mask_file) as ds:
            lat = ds["lat"].values.astype(float)
            lon = wrap_lon180(ds["lon"].values.astype(float))
            strict_land = ds["strict_land"].values.astype(bool) if "strict_land" in ds else None
            high_alt = ds["high_alt"].values.astype(bool) if "high_alt" in ds else None
            elev = ds["elevation_m"].values.astype(float) if "elevation_m" in ds else None

        # Match the |lat| <= 63° analysis domain used for the historical table.
        lat_keep = np.abs(lat) <= 63.0
        if not np.all(lat_keep):
            print(f"[GRID] Restrict mask latitude axis: {lat.size} -> {int(lat_keep.sum())} rows (|lat| <= 63°)")
            lat = lat[lat_keep]
            if strict_land is not None:
                strict_land = strict_land[lat_keep, :]
            if high_alt is not None:
                high_alt = high_alt[lat_keep, :]
            if elev is not None:
                elev = elev[lat_keep, :]

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
    return lat, lon, None, None, None


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
    gl = ax.gridlines(draw_labels=False, linewidth=0.45, color="gray", alpha=0.45, linestyle="--")
    gl.xlocator = FixedLocator([-180, -120, -60, 0, 60, 120, 180])
    gl.ylocator = FixedLocator([-60, -30, 0, 30, 60])
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=proj)
    ax.set_yticks([-60, -30, 0, 30, 60], crs=proj)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE, direction="out", length=4.5, width=0.9, top=False, right=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(BORDER_WIDTH)


def add_region_boxes(ax):
    proj = ccrs.PlateCarree()
    for lon0, lon1, lat0, lat1, color in MAP_BOXES:
        rect = mpatches.Rectangle((lon0, lat0), lon1 - lon0, lat1 - lat0,
                                  linewidth=1.5, edgecolor=color, facecolor="none",
                                  transform=proj, zorder=8)
        ax.add_patch(rect)


def add_high_alt_hatch(ax, lon, lat, high_alt):
    if high_alt is None or not DRAW_HIGH_ALT_HATCH:
        return
    ax.contourf(lon, lat, high_alt.astype(int), levels=[0.5, 1.5], colors="none",
                hatches=["////////"], transform=ccrs.PlateCarree(), zorder=9)


def plot_one_map(ax, diff_grid, lat, lon, title, cmap, vmax, high_alt=None):
    proj = ccrs.PlateCarree()
    zc, lonc = add_cyclic_point(diff_grid, coord=lon)
    m = ax.pcolormesh(lonc, lat, zc, cmap=cmap, vmin=-vmax, vmax=vmax,
                      shading="auto", transform=proj, zorder=1)
    add_map_features(ax)
    add_region_boxes(ax)
    add_high_alt_hatch(ax, lon, lat, high_alt)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold", pad=8)
    return m


def style_horizontal_colorbar(cb, label):
    cb.set_label(label, fontsize=CBAR_LABEL_FONTSIZE, labelpad=4)
    if LAYOUT["cbar_label_position"] == "top":
        cb.ax.xaxis.set_label_position("top")
        cb.ax.xaxis.tick_bottom()
    else:
        cb.ax.xaxis.set_label_position("bottom")
        cb.ax.xaxis.tick_bottom()
    cb.ax.tick_params(labelsize=CBAR_TICK_FONTSIZE, length=4, width=0.9, pad=2)
    cb.outline.set_linewidth(1.0)


def align_colorbars_to_maps(fig, ax_flash, ax_hail, cax_flash, cax_hail):
    fig.canvas.draw()
    pos_f = ax_flash.get_position()
    pos_h = ax_hail.get_position()
    cax_flash.set_position([pos_f.x0, pos_f.y0 - LAYOUT["cbar_gap_flash"], pos_f.width, LAYOUT["cbar_h"]])
    cax_hail.set_position([pos_h.x0, pos_h.y0 - LAYOUT["cbar_gap_hail"], pos_h.width, LAYOUT["cbar_h"]])
    if LAYOUT["print_layout"]:
        print("\n[DYNAMIC LAYOUT]")
        print("Lightning map position:", pos_f)
        print("Hail  map position:", pos_h)
        print("Lightning cbar position:", cax_flash.get_position())
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


def region_mask(lat, lon, lon0, lon1, lat0, lat1):
    lon2d, lat2d = np.meshgrid(lon, lat)
    return (lon2d >= lon0) & (lon2d <= lon1) & (lat2d >= lat0) & (lat2d <= lat1)


def compute_regional_mean_rows(result_name, lat, lon, diff_flash_grid, diff_hail_grid, high_alt=None):
    rows = []
    for region, lon0, lon1, lat0, lat1 in REGION_SPECS:
        mask = region_mask(lat, lon, lon0, lon1, lat0, lat1)
        if FILTER_HIGH_ALT_FOR_REGION_MEAN and high_alt is not None:
            mask = mask & (~high_alt)
        for variable, grid in [("lightning", diff_flash_grid), ("hail", diff_hail_grid)]:
            vals = np.asarray(grid, dtype=float)[mask]
            vals = vals[np.isfinite(vals)]
            mean_val = float(np.nanmean(vals)) if vals.size > 0 else np.nan
            rows.append({
                "result": result_name,
                "region": region,
                "variable": variable,
                "mean_diff": mean_val,
                "n_grid": int(vals.size),
            })
    return rows


def load_one_model_result(model_name, spec):
    print("\n" + "=" * 100)
    print(f"[LOAD MODEL] {model_name}")
    print("=" * 100)

    run_dir = spec["run_dir"]
    train_table = spec["train_table"]
    model_dir = run_dir / "models"
    feature_dir = run_dir / "feature_lists"
    metric_dir = run_dir / "metrics"
    mask_file = run_dir / "PRED_SSP585" / "nc" / "masks_2deg.nc"

    flash_model = model_dir / "FLASH_model.json"
    hail_model = model_dir / "HAIL_model.json"
    flash_feature_txt = feature_dir / "FLASH_features.txt"
    hail_feature_txt = feature_dir / "HAIL_features.txt"
    flash_summary_json = metric_dir / "FLASH_summary.json"
    hail_summary_json = metric_dir / "HAIL_summary.json"

    if not train_table.exists():
        raise FileNotFoundError(train_table)

    print("[READ]", train_table)
    df = pd.read_parquet(train_table)
    ensure_columns(df, [LAT_COL, LON_COL, OBS_FLASH, OBS_HAIL], f"{model_name} historical train table")
    if LAND_COL in df.columns:
        df = df[df[LAND_COL] == 0].copy()
    df = df[np.abs(df[LAT_COL].astype(float)) <= 63].copy().reset_index(drop=True)
    print("[DATA] strict-land rows:", len(df))

    obs_flash = pd.to_numeric(df[OBS_FLASH], errors="coerce").to_numpy(dtype=float)
    obs_hail = pd.to_numeric(df[OBS_HAIL], errors="coerce").to_numpy(dtype=float)

    bst_flash = load_booster(flash_model)
    bst_hail = load_booster(hail_model)
    features_flash = read_feature_list(flash_feature_txt)
    features_hail = read_feature_list(hail_feature_txt)
    best_iter_flash = read_best_iteration(flash_summary_json)
    best_iter_hail = read_best_iteration(hail_summary_json)

    pred_flash = predict_with_booster(df, features_flash, bst_flash, best_iter_flash, "FLASH")
    pred_hail = predict_with_booster(df, features_hail, bst_hail, best_iter_hail, "HAIL")

    n_neg_flash = int(np.sum(np.isfinite(pred_flash) & (pred_flash < 0)))
    n_neg_hail = int(np.sum(np.isfinite(pred_hail) & (pred_hail < 0)))
    pred_flash = np.where(np.isfinite(pred_flash) & (pred_flash < 0), 0.0, pred_flash)
    pred_hail = np.where(np.isfinite(pred_hail) & (pred_hail < 0), 0.0, pred_hail)
    print("Negative LIGHTNING predictions before clip:", n_neg_flash)
    print("Negative HAIL predictions before clip :", n_neg_hail)

    diff_flash = pred_flash - obs_flash
    diff_hail = pred_hail - obs_hail

    lat, lon, strict_land, high_alt, elev = load_grid_axes_from_mask_or_table(df, mask_file)
    grid_diff_flash = points_to_grid(df, diff_flash, lat, lon)
    grid_diff_hail = points_to_grid(df, diff_hail, lat, lon)
    grid_pred_flash = points_to_grid(df, pred_flash, lat, lon)
    grid_pred_hail = points_to_grid(df, pred_hail, lat, lon)
    grid_obs_flash = points_to_grid(df, obs_flash, lat, lon)
    grid_obs_hail = points_to_grid(df, obs_hail, lat, lon)

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

    return {
        "name": model_name,
        "output_tag": spec["output_tag"],
        "lat": lat,
        "lon": lon,
        "strict_land": strict_land,
        "high_alt": high_alt,
        "elev": elev,
        "grid_diff_flash": grid_diff_flash,
        "grid_diff_hail": grid_diff_hail,
        "grid_pred_flash": grid_pred_flash,
        "grid_pred_hail": grid_pred_hail,
        "grid_obs_flash": grid_obs_flash,
        "grid_obs_hail": grid_obs_hail,
        "vector_df": out_vec,
    }


def _common_axis(values_list, axis_name: str) -> np.ndarray:
    """Return exact common grid-coordinate values, rounded to avoid float noise."""
    common = np.round(np.asarray(values_list[0], dtype=float), 6)
    for values in values_list[1:]:
        common = np.intersect1d(common, np.round(np.asarray(values, dtype=float), 6))
    if common.size == 0:
        raise ValueError(f"No common {axis_name} coordinates are available for the MMM calculation.")
    return np.sort(common.astype(float))


def _subset_result_to_common_grid(result: dict, common_lat: np.ndarray, common_lon: np.ndarray) -> dict:
    """Subset an existing result to shared exact 2° coordinates; no interpolation."""
    lat_key = np.round(np.asarray(result["lat"], dtype=float), 6)
    lon_key = np.round(np.asarray(result["lon"], dtype=float), 6)
    lat_map = {v: i for i, v in enumerate(lat_key)}
    lon_map = {v: j for j, v in enumerate(lon_key)}

    try:
        ii = np.array([lat_map[np.round(v, 6)] for v in common_lat], dtype=int)
        jj = np.array([lon_map[np.round(v, 6)] for v in common_lon], dtype=int)
    except KeyError as exc:
        raise ValueError(f"[{result['name']}] missing a coordinate required for the common MMM grid: {exc}") from exc

    def subset_grid(key):
        arr = result.get(key)
        if arr is None:
            return None
        return np.asarray(arr)[np.ix_(ii, jj)]

    return {
        **result,
        "lat": common_lat,
        "lon": common_lon,
        "strict_land": subset_grid("strict_land"),
        "high_alt": subset_grid("high_alt"),
        "elev": subset_grid("elev"),
        "grid_diff_flash": subset_grid("grid_diff_flash"),
        "grid_diff_hail": subset_grid("grid_diff_hail"),
        "grid_pred_flash": subset_grid("grid_pred_flash"),
        "grid_pred_hail": subset_grid("grid_pred_hail"),
        "grid_obs_flash": subset_grid("grid_obs_flash"),
        "grid_obs_hail": subset_grid("grid_obs_hail"),
    }


def make_mmm_result(model_results):
    print("\n" + "=" * 100)
    print("[BUILD MMM]")
    print("=" * 100)

    # All models are reduced to their exact shared 2° cells. This handles
    # mask files with optional ±64° rows without interpolating any values.
    common_lat = _common_axis([r["lat"] for r in model_results], "latitude")
    common_lon = _common_axis([r["lon"] for r in model_results], "longitude")
    print(f"[MMM GRID] common shape = ({common_lat.size}, {common_lon.size})")

    aligned = [_subset_result_to_common_grid(r, common_lat, common_lon) for r in model_results]
    ref = aligned[0]

    diff_flash_stack = np.stack([r["grid_diff_flash"] for r in aligned], axis=0)
    diff_hail_stack = np.stack([r["grid_diff_hail"] for r in aligned], axis=0)
    pred_flash_stack = np.stack([r["grid_pred_flash"] for r in aligned], axis=0)
    pred_hail_stack = np.stack([r["grid_pred_hail"] for r in aligned], axis=0)
    obs_flash_stack = np.stack([r["grid_obs_flash"] for r in aligned], axis=0)
    obs_hail_stack = np.stack([r["grid_obs_hail"] for r in aligned], axis=0)

    # MMM is calculated only where all three model reconstructions are finite.
    valid_flash = np.all(np.isfinite(diff_flash_stack), axis=0)
    valid_hail = np.all(np.isfinite(diff_hail_stack), axis=0)

    grid_diff_flash = np.where(valid_flash, np.mean(diff_flash_stack, axis=0), np.nan).astype(np.float32)
    grid_diff_hail = np.where(valid_hail, np.mean(diff_hail_stack, axis=0), np.nan).astype(np.float32)
    grid_pred_flash = np.where(valid_flash, np.mean(pred_flash_stack, axis=0), np.nan).astype(np.float32)
    grid_pred_hail = np.where(valid_hail, np.mean(pred_hail_stack, axis=0), np.nan).astype(np.float32)
    grid_obs_flash = np.where(valid_flash, np.mean(obs_flash_stack, axis=0), np.nan).astype(np.float32)
    grid_obs_hail = np.where(valid_hail, np.mean(obs_hail_stack, axis=0), np.nan).astype(np.float32)

    return {
        "name": "Multi-model Mean",
        "output_tag": "mmm",
        "lat": common_lat,
        "lon": common_lon,
        "strict_land": ref["strict_land"],
        "high_alt": ref["high_alt"],
        "elev": ref["elev"],
        "grid_diff_flash": grid_diff_flash,
        "grid_diff_hail": grid_diff_hail,
        "grid_pred_flash": grid_pred_flash,
        "grid_pred_hail": grid_pred_hail,
        "grid_obs_flash": grid_obs_flash,
        "grid_obs_hail": grid_obs_hail,
        "vector_df": None,
    }


def save_result_artifacts(result):
    tag = result["output_tag"]
    stem = f"figure_s06_{tag}_historical_bias"

    np.savez_compressed(
        OUT_DIR / f"{stem}_grids.npz",
        lat=result["lat"].astype(np.float32),
        lon=result["lon"].astype(np.float32),
        diff_flash=result["grid_diff_flash"].astype(np.float32),
        diff_hail=result["grid_diff_hail"].astype(np.float32),
        pred_flash=result["grid_pred_flash"].astype(np.float32),
        pred_hail=result["grid_pred_hail"].astype(np.float32),
        obs_flash=result["grid_obs_flash"].astype(np.float32),
        obs_hail=result["grid_obs_hail"].astype(np.float32),
        high_alt=result["high_alt"].astype(np.int8) if result["high_alt"] is not None else np.array([]),
        elev=result["elev"].astype(np.float32) if result["elev"] is not None else np.array([]),
    )
    print("[SAVED]", OUT_DIR / f"{stem}_grids.npz")

    if result["vector_df"] is not None:
        result["vector_df"].to_csv(OUT_DIR / f"{stem}_vectors.csv", index=False, encoding="utf-8-sig")
        print("[SAVED]", OUT_DIR / f"{stem}_vectors.csv")


def plot_result(result):
    name = result["name"]
    lat = result["lat"]
    lon = result["lon"]
    high_alt = result["high_alt"]
    grid_diff_flash = result["grid_diff_flash"]
    grid_diff_hail = result["grid_diff_hail"]

    if VMAX_FLASH_DIFF is None:
        vmax_flash = symmetric_vmax(grid_diff_flash, pct=98, fallback=0.015)
    else:
        vmax_flash = float(VMAX_FLASH_DIFF)
    if VMAX_HAIL_DIFF is None:
        vmax_hail = symmetric_vmax(grid_diff_hail, pct=98, fallback=0.010)
    else:
        vmax_hail = float(VMAX_HAIL_DIFF)

    print(f"\n[COLOR RANGE] {name}")
    print("vmax_lightning:", vmax_flash)
    print("vmax_hail :", vmax_hail)

    cmap = make_diverging_cmap()
    fig = plt.figure(figsize=LAYOUT["figsize"], dpi=FIG_DPI)
    proj = ccrs.PlateCarree()
    fig.suptitle(
        name,
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
        y=1,
    )
    ax_flash = fig.add_axes(LAYOUT["ax_flash"], projection=proj)
    ax_hail = fig.add_axes(LAYOUT["ax_hail"], projection=proj)
    cax_flash = fig.add_axes([0.05, 0.05, 0.10, 0.02])
    cax_hail = fig.add_axes([0.05, 0.02, 0.10, 0.02])

    m_flash = plot_one_map(
        ax=ax_flash,
        diff_grid=grid_diff_flash,
        lat=lat,
        lon=lon,
        title="(a) Lightning: Historical Reconstruction – Observed",
        cmap=cmap,
        vmax=vmax_flash,
        high_alt=high_alt,
    )
    m_hail = plot_one_map(
        ax=ax_hail,
        diff_grid=grid_diff_hail,
        lat=lat,
        lon=lon,
        title="(b) Hail: Historical Reconstruction – Observed",
        cmap=cmap,
        vmax=vmax_hail,
        high_alt=high_alt,
    )

    align_colorbars_to_maps(fig, ax_flash, ax_hail, cax_flash, cax_hail)

    cb_flash = fig.colorbar(m_flash, cax=cax_flash, orientation="horizontal")
    style_horizontal_colorbar(cb_flash, "Reconstruction - Observed")
    cb_hail = fig.colorbar(m_hail, cax=cax_hail, orientation="horizontal")
    style_horizontal_colorbar(cb_hail, "Reconstruction - Observed")

    tag = result["output_tag"]
    out_png = OUT_DIR / f"figure_s06_{tag}_historical_bias_maps.png"
    safe_savefig(fig, out_png, dpi=FIG_DPI)
    plt.close(fig)
    print("\n[SAVED FIGURE]")
    print(out_png)


def main():
    print("=" * 100)
    print("[STEP 10] Plot historical predicted - observed maps for all models and MMM")
    print("=" * 100)

    model_results = [load_one_model_result(name, spec) for name, spec in MODEL_SPECS.items()]
    mmm_result = make_mmm_result(model_results)
    all_results = model_results + [mmm_result]

    regional_rows = []
    for result in all_results:
        save_result_artifacts(result)
        plot_result(result)
        regional_rows.extend(
            compute_regional_mean_rows(
                result_name=result["name"],
                lat=result["lat"],
                lon=result["lon"],
                diff_flash_grid=result["grid_diff_flash"],
                diff_hail_grid=result["grid_diff_hail"],
                high_alt=result["high_alt"],
            )
        )

    regional_df = pd.DataFrame(regional_rows)
    out_csv = REGIONAL_MEAN_CSV
    regional_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("\n" + "=" * 100)
    print("[REGIONAL MEAN DIFFERENCES]")
    print("=" * 100)
    for result_name in regional_df["result"].unique():
        print(f"\n--- {result_name} ---")
        tmp = regional_df[regional_df["result"] == result_name]
        for _, row in tmp.iterrows():
            var_name = "Lightning" if row["variable"] == "lightning" else "Hail"
            print(f"{row['region']:>3s} | {var_name:<5s} | mean diff = {row['mean_diff']:.6f} | n = {int(row['n_grid'])}")
    print("\n[SAVED]", out_csv)
    print("\n[DONE] Step 10 finished successfully.")


if __name__ == "__main__":
    main()
