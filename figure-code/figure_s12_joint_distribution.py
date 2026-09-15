# Purpose: figure s12 joint distribution.
# Source: cmip6_figure_MMM_3models_joint2d_trend_with_markers.ipynb; see docs/source_coverage.csv for cell provenance.
# Run order and required inputs: docs/workflow.md.
# Run from repository root: python -m figures.figure_s12_joint_distribution

from __future__ import annotations

from publication_config import resource_path
import os
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import LogLocator, NullLocator, MaxNLocator

# ==================== 用户配置 ====================
MODEL_BASE_DIRS = {
    "CanESM5": Path(resource_path('models', 'CanESM5')),
    "MIROC6": Path(resource_path('models', 'MIROC6')),
    "BCC-CSM2-MR": Path(resource_path('models', 'BCC-CSM2-MR')),
}
# 三个模式目录统一使用正式模式名称。
REFERENCE_MODEL = "CanESM5"
OBS_PARQ = (MODEL_BASE_DIRS[REFERENCE_MODEL] /
            "ml_feature_withSRTM" / "train_table_flash_hail_withSRTM_strictLand.parquet")
OUT_DIR = Path(resource_path('figures', 'figure_s12_joint_distribution'))
OUTPUT_STEM = 'figure_s12_joint_distribution'
C_MAP_MAT = resource_path('colormaps', 'so4_21.mat')

DECADES = [(2050, 2059), (2060, 2069), (2070, 2079), (2080, 2089), (2090, 2099)]
VAR_FLASH, VAR_HAIL = "pred_flash", "pred_hail"
NC_LAT, NC_LON = "lat", "lon"
LAT, LON, LANDSEA = "Latitude", "Longitude", "Land_Sea"
OBS_FLASH, OBS_HAIL = "LISOTD_Flash", "ni_HailPF"
LAT_MAX = 63.0
FLASH_BINS = np.logspace(-4, 0, 50)
HAIL_BINS = np.logspace(-4, -1, 30)
XLIM, YLIM = (1e-4, 1), (1e-4, 1e-1)
FREQ_LEVELS_B, DIFF_LEVELS_C = 18, 21
FIT_FIRST_N_HAIL_BINS = 25

# ==================== 完整恢复正文图3的尺寸与风格 ====================
# 原图3为23.5×6.2英寸。本图把两行原尺寸面板上下堆叠，
# 额外增加0.6英寸顶部空间放模式名，每个列间隙增加1英寸，
# 避免完整标题/坐标标签重叠；这些新增留白不缩放原面板。
REFERENCE_FIGSIZE_INCH = (23.5, 6.2)
MODEL_HEADER_HEIGHT_INCH = 0.6
COLUMN_EXTRA_GAP_INCH = 1.0
FIGSIZE_INCH = (REFERENCE_FIGSIZE_INCH[0] + 2 * COLUMN_EXTRA_GAP_INCH,
                2 * REFERENCE_FIGSIZE_INCH[1] + MODEL_HEADER_HEIGHT_INCH)
FONT_FAMILY = "Arial"
TITLE_FONTSIZE = 25
MODEL_FONTSIZE = 25
LABEL_FONTSIZE = 20
TICK_FONTSIZE = 20
TEXT_FONTSIZE = 18
LEGEND_FONTSIZE = 16
CBAR_LABEL_FONTSIZE = 18
ANNOTATION_FONTSIZE = 17
BORDER_WIDTH = 1.2
FIT_LINE_COLOR, MEAN_LINE_COLOR = "#1E90FF", "black"
DPI = 600
SHOW_FIGURE = True

# ==================== 原图3共用函数 ====================
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

def load_nc_decade(path_nc: str):
    import xarray as xr

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


# ==================== 各模式分别计算五年代平均 ====================
def load_individual_model_means():
    results = {}
    lat_ref, lon_ref = None, None
    for model_name, base_dir in MODEL_BASE_DIRS.items():
        flash_decades, hail_decades = [], []
        for y0, y1 in DECADES:
            flash, hail, lat, lon = load_nc_decade_from_model(base_dir, model_name, y0, y1)
            if lat_ref is None:
                lat_ref, lon_ref = lat, lon
            else:
                if len(lat) != len(lat_ref) or len(lon) != len(lon_ref):
                    raise ValueError(f"[{model_name}] grid shape mismatch in {y0}-{y1}")
                if not np.allclose(lat, lat_ref, equal_nan=True):
                    raise ValueError(f"[{model_name}] latitude mismatch in {y0}-{y1}")
                if not np.allclose(lon, lon_ref, equal_nan=True):
                    raise ValueError(f"[{model_name}] longitude mismatch in {y0}-{y1}")
            flash_decades.append(flash)
            hail_decades.append(hail)
        # 与原图3的时间平均一致；本图分别保留每个模式，无跨模式平均。
        results[model_name] = (
            np.nanmean(np.stack(flash_decades, axis=0), axis=0),
            np.nanmean(np.stack(hail_decades, axis=0), axis=0),
        )
    return results, lat_ref, lon_ref


def make_figure(model_fields, obs_flash, obs_hail, cmap_freq, cmap_diff):
    """两行各自复用原图3的物理尺寸和GridSpec参数。"""
    plt.rcParams.update({
        "font.family": FONT_FAMILY, "font.size": 18,
        "axes.unicode_minus": False, "figure.dpi": DPI,
    })
    fig = plt.figure(figsize=FIGSIZE_INCH, dpi=DPI)
    row_height = REFERENCE_FIGSIZE_INCH[1]
    total_height = FIGSIZE_INCH[1]
    panels = np.empty((2, 3), dtype=object)
    # 原图3每个面板的高度=(0.90−0.14)×6.2=4.712英寸。
    # 根据原图3的列组物理宽度换算新增留白，保持面板/色标原尺寸。
    reference_width = REFERENCE_FIGSIZE_INCH[0]
    group_width = reference_width * (0.95 - 0.07) / (3 + 2 * 0.35)
    total_width = FIGSIZE_INCH[0]
    for row in range(2):
        row_offset = row_height if row == 0 else 0.0
        outer = fig.add_gridspec(
            nrows=1, ncols=3,
            left=0.07 * reference_width / total_width,
            right=(0.95 * reference_width + 2 * COLUMN_EXTRA_GAP_INCH) / total_width,
            bottom=(row_offset + 0.14 * row_height) / total_height,
            top=(row_offset + 0.90 * row_height) / total_height,
            wspace=0.35 + COLUMN_EXTRA_GAP_INCH / group_width,
        )
        for col, model_name in enumerate(MODEL_BASE_DIRS):
            future_flash, future_hail = model_fields[model_name]
            inner = outer[0, col].subgridspec(
                nrows=1, ncols=2, width_ratios=[1.0, 0.045], wspace=0.03,
            )
            ax = fig.add_subplot(inner[0, 0])
            cax = fig.add_subplot(inner[0, 1])
            panels[row, col] = ax
            print(f"\n[FIGURE] {model_name}, row {row + 1}")
            if row == 0:
                draw_panel_b(ax, cax, future_flash, future_hail, cmap_freq,
                             flash_bins=FLASH_BINS, hail_bins=HAIL_BINS,
                             levels=FREQ_LEVELS_B,
                             fit_first_n_hail_bins=FIT_FIRST_N_HAIL_BINS,
                             xlim=XLIM, ylim=YLIM)
            else:
                # 传副本，防止原图3c的数组预处理影响后续模式的共用观测。
                draw_panel_c(ax, cax, future_flash.copy(), future_hail.copy(),
                             obs_flash.copy(), obs_hail.copy(), cmap_diff)
            # 仅替换组合图的面板编号；原题目文字、字号、粗细与pad均保留。
            letter = chr(ord("a") + row * 3 + col)
            ax.title.set_text(f"({letter})" + ax.get_title()[3:])
            if row == 0:
                pos = ax.get_position()
                fig.text(pos.x0 + pos.width / 2,
                         (2 * row_height + MODEL_HEADER_HEIGHT_INCH / 2) / total_height,
                         model_name, ha="center", va="center",
                         fontsize=MODEL_FONTSIZE, fontweight="bold")
    return fig, panels


def main():
    if not OBS_PARQ.exists():
        raise FileNotFoundError(f"Observed reference table missing: {OBS_PARQ}")
    model_fields, lat, lon = load_individual_model_means()
    obs_flash = build_grid_from_parquet(lat, lon, OBS_PARQ, OBS_FLASH)
    obs_hail = build_grid_from_parquet(lat, lon, OBS_PARQ, OBS_HAIL)
    cmap_freq = load_ncl_mat_cmap(C_MAP_MAT, key="cmap", reverse=True)
    cmap_diff = diverging_with_gray_center(gray=0.95)
    fig, _ = make_figure(model_fields, obs_flash, obs_hail, cmap_freq, cmap_diff)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        path = OUT_DIR / f"{OUTPUT_STEM}.{extension}"
        fig.savefig(path, dpi=DPI, facecolor="white")
        print("[SAVED]", path)
    if SHOW_FIGURE:
        plt.show()
    plt.close(fig)
    print("[DONE] Six independent colorbars; original Figure 3b/c calculations retained.")


if __name__ == "__main__":
    main()
