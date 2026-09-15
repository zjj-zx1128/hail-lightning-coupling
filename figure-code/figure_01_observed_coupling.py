# Purpose: figure 01 observed coupling.
# Source: cmip6_figure1_paper2.ipynb; see docs/source_coverage.csv for cell provenance.
# Run order and required inputs: docs/workflow.md.
# Run from repository root: python -m figures.figure_01_observed_coupling


from publication_config import resource_path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.io as scio
import scipy.stats as stats
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import rioxarray as rxr

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.patches import Rectangle
from rasterio.enums import Resampling


# =========================
# 全局绘图风格
# =========================
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False

# =========================
# 统一字体参数
# =========================
PANEL_TITLE_SIZE = 29
LABEL_SIZE = 23
TICK_SIZE = 21
LEGEND_SIZE = 19

CBAR_LABEL_SIZE = 23
CBAR_TICK_SIZE = 20


# =========================
# 文件路径
# =========================
DATA_PATH = resource_path('observations', 'flashhail_matched_new2.xlsx')
ELEVATION_PATH = resource_path('observations', 'data/elevation/elevation_1KMmn_SRTM.tif')

FREQUENCY_CMAP_PATH = (
    resource_path('colormaps', 'so4_21.mat')
)
DIFFERENCE_CMAP_PATH = (
    resource_path('colormaps', 'BlWhRe.mat')
)

OUTPUT_PATH = resource_path('figures', 'figure_01_observed_coupling.jpg')

LAT_MIN, LAT_MAX = -63, 63
ELEVATION_MAX = 2000


# =========================
# 布局控制参数
# =========================
# b/c 面板之间的纵向间距；原来是 0.10，这里略微减小
BC_HSPACE = 0.055

# b/c 色标相对于对应地图高度的比例
# 小于 1 表示缩短，并在对应地图中心垂直居中
BC_CBAR_HEIGHT_RATIO = 0.99


# =========================
# 共用函数
# =========================
def array2cmap(colors, name):
    n_colors = colors.shape[0]

    positions = np.linspace(0.0, 1.0, n_colors + 1)
    positions = np.sort(np.concatenate((positions, positions)))[1:-1]

    red = np.concatenate(
        [[colors[i, 0], colors[i, 0]] for i in range(n_colors)]
    )
    green = np.concatenate(
        [[colors[i, 1], colors[i, 1]] for i in range(n_colors)]
    )
    blue = np.concatenate(
        [[colors[i, 2], colors[i, 2]] for i in range(n_colors)]
    )

    color_dict = {
        "red": tuple(
            (positions[i], red[i], red[i]) for i in range(2 * n_colors)
        ),
        "green": tuple(
            (positions[i], green[i], green[i]) for i in range(2 * n_colors)
        ),
        "blue": tuple(
            (positions[i], blue[i], blue[i]) for i in range(2 * n_colors)
        ),
    }

    return matplotlib.colors.LinearSegmentedColormap(
        name,
        color_dict,
        n_colors,
    )


def load_colormaps():
    frequency_colors = np.asarray(
        scio.loadmat(FREQUENCY_CMAP_PATH)["cmap"]
    )[::-1]

    difference_colors = np.asarray(
        scio.loadmat(DIFFERENCE_CMAP_PATH)["cmap"]
    )

    frequency_cmap = array2cmap(
        frequency_colors,
        "frequency_cmap",
    )
    difference_cmap = array2cmap(
        difference_colors,
        "difference_cmap",
    )

    return frequency_cmap, difference_cmap


def add_region_boxes(ax):
    boxes = [
        (-104, -90, 35, 47, "blue"),
        (-68, -50, -42, -23, "blue"),
        (-10, 20, 35, 55, "blue"),
        (112, 138, 35, 55, "blue"),
        (-98, -78, 24, 35, "red"),
        (-76, -58, -10, 5, "red"),
        (12, 35, -8, 5, "red"),
        (95, 150, -12, 8, "red"),
    ]

    for lon_min, lon_max, lat_min, lat_max, color in boxes:
        ax.add_patch(
            Rectangle(
                (lon_min, lat_min),
                lon_max - lon_min,
                lat_max - lat_min,
                linewidth=1.5,
                edgecolor=color,
                facecolor="none",
                transform=ccrs.PlateCarree(),
                zorder=10,
            )
        )


def format_map_axis(ax, title):
    ax.coastlines(
        resolution="110m",
        color="black",
        linewidth=1,
        zorder=5,
    )

    ax.add_feature(
        cfeature.BORDERS,
        linewidth=0.6,
        linestyle=":",
        zorder=5,
    )

    add_region_boxes(ax)

    gridliner = ax.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )

    gridliner.right_labels = False
    gridliner.top_labels = False
    gridliner.xlabel_style = {"size": TICK_SIZE}
    gridliner.ylabel_style = {"size": TICK_SIZE}
    gridliner.xformatter = LONGITUDE_FORMATTER
    gridliner.yformatter = LATITUDE_FORMATTER

    ax.set_extent(
        [-180, 180, LAT_MIN, LAT_MAX],
        crs=ccrs.PlateCarree(),
    )

    ax.set_title(
        title,
        fontsize=PANEL_TITLE_SIZE,
        pad=18,
        fontweight="bold",
    )


# =========================
# 读取数据
# =========================
frequency_cmap, difference_cmap = load_colormaps()
df = pd.read_excel(DATA_PATH)

lat_all = df["Latitude"].to_numpy()
lon_all = df["Longitude"].to_numpy()

flash_all = df["LISOTD_Flash"].replace(0, np.nan).to_numpy()
hail_all = df["ni_HailPF"].replace(0, np.nan).to_numpy()
land_sea_all = df["Land_Sea"].to_numpy()


# ============================================================
# 图 a：不进行海拔筛选
# ============================================================
base_mask_a = (
    (lat_all >= LAT_MIN)
    & (lat_all <= LAT_MAX)
    & (land_sea_all == 0)
    & np.isfinite(flash_all)
    & np.isfinite(hail_all)
)

lat_a = lat_all[base_mask_a]
lon_a = lon_all[base_mask_a]
flash_a = flash_all[base_mask_a]
hail_a = hail_all[base_mask_a]

print(f"Valid land grid cells for panel (a): {flash_a.size}")


# ============================================================
# 图 b / c：与图 a 使用完全相同的统计样本
# 不进行海拔筛选；海拔仅用于地图上的斜线掩膜显示。
# ============================================================
lat = lat_a.copy()
lon = lon_a.copy()
flash = flash_a.copy()
hail = hail_a.copy()

print(f"Valid land grid cells for panels (a,b,c): {flash.size}")

# 海拔数据仅用于绘制海拔 > 2000 m 的斜线区域，
# 不参与图 b/c 的样本筛选、分箱统计或 standardized difference 计算。
elevation = rxr.open_rasterio(
    ELEVATION_PATH,
    masked=True,
).squeeze()

if elevation.rio.crs is None:
    elevation = elevation.rio.write_crs("EPSG:4326")

elev_2deg = elevation.rio.reproject(
    "EPSG:4326",
    resolution=2.0,
    resampling=Resampling.average,
)


# =========================
# 图 a、b、c 共用的 hail-bin 条件统计（log10 lightning 空间）
# =========================
hail_bins = np.logspace(-4, -1, 30)

# 对每个 hail bin，计算与图 a 完全一致的 log10(lightning) mean 和 sample SD。
flash_log_means = np.full(len(hail_bins) - 1, np.nan, dtype=np.float64)
flash_log_stds = np.full(len(hail_bins) - 1, np.nan, dtype=np.float64)
diff = np.full(flash.shape, np.nan, dtype=np.float64)

for i in range(len(hail_bins) - 1):
    bin_mask = (
        (hail >= hail_bins[i])
        & (hail < hail_bins[i + 1])
    )

    if not np.any(bin_mask):
        continue

    log_flash_in_bin = np.log10(flash[bin_mask])
    mean_log = np.mean(log_flash_in_bin)
    flash_log_means[i] = mean_log

    if log_flash_in_bin.size >= 2:
        std_log = np.std(log_flash_in_bin, ddof=1)
        flash_log_stds[i] = std_log

        if std_log > 0 and np.isfinite(std_log):
            diff[bin_mask] = (
                log_flash_in_bin - mean_log
            ) / std_log

# 图 a 的黑线与图 b/c 的 standardized difference 使用同一组统计量。
flash_means = 10 ** flash_log_means
flash_upper = 10 ** (flash_log_means + flash_log_stds)
flash_lower = 10 ** (flash_log_means - flash_log_stds)


# ============================================================
# 组合版式
# ============================================================
fig = plt.figure(figsize=(24, 12))

outer_grid = fig.add_gridspec(
    1,
    2,
    width_ratios=[1.05, 1.45],
    left=0.045,
    right=0.985,
    bottom=0.08,
    top=0.94,
    wspace=0.15,
)

left_grid = outer_grid[0, 0].subgridspec(
    1,
    2,
    width_ratios=[1.0, 0.045],
    wspace=0.05,
)

right_grid = outer_grid[0, 1].subgridspec(
    2,
    2,
    width_ratios=[1.0, 0.025],
    height_ratios=[1.0, 1.0],
    wspace=0.08,
    hspace=BC_HSPACE,
)

ax_a = fig.add_subplot(left_grid[0, 0])
cax_a = fig.add_subplot(left_grid[0, 1])

ax_b = fig.add_subplot(
    right_grid[0, 0],
    projection=ccrs.PlateCarree(),
)

ax_c = fig.add_subplot(
    right_grid[1, 0],
    projection=ccrs.PlateCarree(),
)

# 图 b、c 共用右侧同一个颜色条
cax_bc = fig.add_subplot(right_grid[:, 1])


# =========================
# 
# -Hail 二维频数分布
# =========================
flash_bins = np.logspace(-5, 0, 50)
hail_bin_centers = (hail_bins[:-1] + hail_bins[1:]) / 2

# ============================================================
# Additional R² diagnostics: raw grid-cell data only
# Does not alter the figure or the fitted line shown in panel (a)
# ============================================================
all_valid_mask = (
    np.isfinite(flash_a)
    & np.isfinite(hail_a)
    & (flash_a > 0)
    & (hail_a > 0)
)

# Same range as the first 25 hail bins used in the existing code:
# bins 0–24, i.e., [hail_bins[0], hail_bins[25])
first25_bin_mask = (
    all_valid_mask
    & (hail_a >= hail_bins[0])
    & (hail_a < hail_bins[25])
)


def calculate_r2(x, y):
    """Return R² from a simple linear regression with an intercept."""
    result = stats.linregress(x, y)
    return result.rvalue ** 2, result.pvalue, x.size


# 1. First 25 hail bins: log10(Hail) versus log10(Lightning)
r2_first25_loglog, p_first25_loglog, n_first25_loglog = calculate_r2(
    np.log10(hail_a[first25_bin_mask]),
    np.log10(flash_a[first25_bin_mask]),
)

# 2. All valid grid cells: raw Hail versus raw Lightning
r2_all_raw, p_all_raw, n_all_raw = calculate_r2(
    hail_a[all_valid_mask],
    flash_a[all_valid_mask],
)

# 3. First 25 hail bins: raw Hail versus raw Lightning
r2_first25_raw, p_first25_raw, n_first25_raw = calculate_r2(
    hail_a[first25_bin_mask],
    flash_a[first25_bin_mask],
)

# 4. Same as item 2 in your request: all valid grid cells, raw scale
r2_all_raw_repeat = r2_all_raw
p_all_raw_repeat = p_all_raw
n_all_raw_repeat = n_all_raw


print("\n" + "=" * 72)
print("R² DIAGNOSTICS: LIGHTNING–HAIL RAW GRID-CELL RELATIONSHIP")
print("=" * 72)

print(
    "1. First 25 hail bins, log-log raw grid cells: "
    f"N = {n_first25_loglog}, "
    f"R² = {r2_first25_loglog:.4f}, "
    f"p = {p_first25_loglog:.3e}"
)

print(
    "2. All valid grid cells, raw scale: "
    f"N = {n_all_raw}, "
    f"R² = {r2_all_raw:.4f}, "
    f"p = {p_all_raw:.3e}"
)

print(
    "3. First 25 hail bins, raw scale: "
    f"N = {n_first25_raw}, "
    f"R² = {r2_first25_raw:.4f}, "
    f"p = {p_first25_raw:.3e}"
)

print(
    "4. All valid grid cells, raw scale (same as item 2): "
    f"N = {n_all_raw_repeat}, "
    f"R² = {r2_all_raw_repeat:.4f}, "
    f"p = {p_all_raw_repeat:.3e}"
)

print("=" * 72 + "\n")

# ============================================================
# 原始格点的 log-log 线性回归
# 仅使用图 a 实际显示范围内的有效格点
# ============================================================
raw_fit_mask = (
    np.isfinite(flash_a)
    & np.isfinite(hail_a)
    & (flash_a >= flash_bins[0])
    & (flash_a <= flash_bins[-1])
    & (hail_a >= hail_bins[0])
    & (hail_a <= hail_bins[-1])
)

log_hail_raw = np.log10(hail_a[raw_fit_mask])
log_flash_raw = np.log10(flash_a[raw_fit_mask])

slope, intercept, r_value, p_value, std_err = stats.linregress(
    log_hail_raw,
    log_flash_raw,
)

r2_loglog = r_value ** 2

print(
    f"Raw-point log-log regression: "
    f"N = {log_hail_raw.size}, "
    f"slope = {slope:.4f}, "
    f"R² = {r2_loglog:.4f}, "
    f"p = {p_value:.3e}"
)

hist, xedges, yedges = np.histogram2d(
    flash_a,
    hail_a,
    bins=[flash_bins, hail_bins],
)

x_grid, y_grid = np.meshgrid(
    xedges[:-1],
    yedges[:-1],
)

frequency_contour = ax_a.contourf(
    x_grid,
    y_grid,
    hist.T,
    levels=18,
    cmap=frequency_cmap,
)

ax_a.plot(
    flash_means,
    hail_bin_centers,
    color="black",
    linestyle="-",
    linewidth=1.8,
    label="Mean Lightning Frequency",
)

ax_a.plot(
    flash_upper,
    hail_bin_centers,
    color="black",
    linestyle="--",
    linewidth=1.2,
    label=r"Mean $\pm$ 1 SD",
)

ax_a.plot(
    flash_lower,
    hail_bin_centers,
    color="black",
    linestyle="--",
    linewidth=1.2,
    # label="Mean - 1 Std",
)

hail_extended = np.logspace(
    np.log10(1e-4),
    np.log10(1e-1),
    100,
)

flash_fit_extended = 10 ** (
    slope * np.log10(hail_extended) + intercept
)

ax_a.plot(
    flash_fit_extended,
    hail_extended,
    color="#1E90FF",
    linestyle="-.",
    linewidth=2,
    label="Fitted Mean",
)

ref_hail = 10 ** (-2.2)

ax_a.axhline(
    y=ref_hail,
    color="red",
    linestyle=":",
    linewidth=1.5,
)

ax_a.text(
    1.3e-4,
    ref_hail * 1.1,
    r"$10^{-2.2}$",
    fontsize=LABEL_SIZE,
    ha="left",
    va="center",
)

equation_text = (
    rf"$\log_{{10}}(\mathrm{{Lightning}}) = {slope:.2f}\,"
    rf"\log_{{10}}(\mathrm{{Hail}}) {intercept:+.2f}$"
)

r_squared_text = rf"$\mathrm{{log}}_{{10}}: R^2 = {r2_loglog:.2f}$"

ax_a.text(
    0.02,
    0.98,
    equation_text,
    transform=ax_a.transAxes,
    fontsize=LABEL_SIZE,
    color="#1E90FF",
    va="top",
)

ax_a.text(
    0.02,
    0.93,
    r_squared_text,
    transform=ax_a.transAxes,
    fontsize=LABEL_SIZE,
    color="#1E90FF",
    va="top",
)

ax_a.set_xscale("log")
ax_a.set_yscale("log")
ax_a.set_xlim([1e-4, 1])
ax_a.set_ylim([1e-4, 1e-1])

# ========= 仅修改这里：图 a 横纵坐标 =========
ax_a.set_xlabel(
    "Lightning Frequency: LIS/OTD Flash Rate Density (flashes km$^{-2}$ day$^{-1}$)",
    fontsize=LABEL_SIZE,
)

ax_a.set_ylabel(
    "Hail Frequency: GPM Hail PF Occurrence",
    fontsize=LABEL_SIZE,
)

ax_a.tick_params(
    axis="both",
    which="major",
    labelsize=TICK_SIZE,
)

ax_a.set_title(
    "(a) Lightning and Hail Two-Dimensional Frequency",
    fontsize=PANEL_TITLE_SIZE,
    pad=18,
    fontweight="bold",
)

ax_a.grid(
    linewidth=1,
    color="gray",
    alpha=0.5,
    linestyle="--",
)

ax_a.legend(
    loc="lower right",
    frameon=False,
    fontsize=LEGEND_SIZE,
)

cbar_a = fig.colorbar(
    frequency_contour,
    cax=cax_a,
    orientation="vertical",
)

cbar_a.set_label(
    "Grid count",
    fontsize=CBAR_LABEL_SIZE,
)

cbar_a.ax.tick_params(
    labelsize=CBAR_TICK_SIZE,
)


# =========================
# (b) 连续标准化差异地图
# =========================
high_alt_global = elev_2deg > ELEVATION_MAX

ax_b.contourf(
    elev_2deg["x"],
    elev_2deg["y"],
    high_alt_global.to_numpy(),
    levels=[0.5, 1.5],
    colors="none",
    hatches=["//////////"],
    transform=ccrs.PlateCarree(),
    zorder=7,
)

diff_scatter = ax_b.scatter(
    lon,
    lat,
    c=diff,
    cmap=difference_cmap,
    vmin=-5,
    vmax=5,
    s=5,
    transform=ccrs.PlateCarree(),
    zorder=6,
)

format_map_axis(
    ax_b,
    "(b)Lightning and Hail Frequency Distribution Difference",
)

cbar_bc = fig.colorbar(
    diff_scatter,
    cax=cax_bc,
    extend="both",
    orientation="vertical",
)

cbar_bc.set_ticks([-4, -2, 0, 2, 4])

cbar_bc.set_label(
    r"$D_{i,j}$",
    fontsize=CBAR_LABEL_SIZE,
)

cbar_bc.ax.tick_params(
    labelsize=CBAR_TICK_SIZE,
)


# =========================
# (c) 带条件的差异地图
# =========================
hist_conditions, _, _ = np.histogram2d(
    flash,
    hail,
    bins=[flash_bins, hail_bins],
)

frequency_threshold = 25

print(
    "Max frequency value: "
    f"{np.nanmax(hist_conditions):.0f}, "
    f"threshold: {frequency_threshold:.0f}"
)

mask_below = diff < -1
mask_above = diff > 1

flash_indices = np.digitize(flash, flash_bins) - 1
hail_indices = np.digitize(hail, hail_bins) - 1

valid_indices = (
    (flash_indices >= 0)
    & (flash_indices < hist_conditions.shape[0])
    & (hail_indices >= 0)
    & (hail_indices < hist_conditions.shape[1])
)

frequency_per_point = np.full(
    flash.shape,
    np.nan,
)

frequency_per_point[valid_indices] = hist_conditions[
    flash_indices[valid_indices],
    hail_indices[valid_indices],
]

frequency_mask_high_hail = (
    (frequency_per_point > frequency_threshold)
    & (hail > ref_hail)
)

frequency_mask_low_hail = (
    (frequency_per_point > frequency_threshold)
    & (hail <= ref_hail)
)

condition_scatter = ax_c.scatter(
    lon[mask_below],
    lat[mask_below],
    c=diff[mask_below],
    cmap=difference_cmap,
    vmin=-5,
    vmax=5,
    s=4,
    marker="X",
    label=r"Lightning < Mean $-$ 1 SD",
    transform=ccrs.PlateCarree(),
    zorder=6,
)

ax_c.scatter(
    lon[mask_above],
    lat[mask_above],
    c=diff[mask_above],
    cmap=difference_cmap,
    vmin=-5,
    vmax=5,
    s=4,
    marker="X",
    label=r"Lightning > Mean $+$ 1 SD",
    transform=ccrs.PlateCarree(),
    zorder=6,
)

ax_c.scatter(
    lon[frequency_mask_high_hail],
    lat[frequency_mask_high_hail],
    color="lightcoral",
    s=6,
    marker="d",
    label=r"Grid count > 25 & Hail > $10^{-2.2}$",
    transform=ccrs.PlateCarree(),
    zorder=7,
)

ax_c.scatter(
    lon[frequency_mask_low_hail],
    lat[frequency_mask_low_hail],
    color="cornflowerblue",
    s=6,
    marker="d",
    label=r"Grid count > 25 & Hail $\leq$ $10^{-2.2}$",
    transform=ccrs.PlateCarree(),
    zorder=7,
)

ax_c.contourf(
    elev_2deg["x"],
    elev_2deg["y"],
    high_alt_global.to_numpy(),
    levels=[0.5, 1.5],
    colors="none",
    hatches=["//////////"],
    transform=ccrs.PlateCarree(),
    zorder=8,
)

format_map_axis(
    ax_c,
    "(c) Conditioned Lightning and Hail Distribution Difference",
)

ax_c.legend(
    loc="lower left",
    frameon=False,
    fontsize=LEGEND_SIZE,
    markerscale=1.8,
    ncol=2,
)

# =========================
# 图(a)与图(b)(c)对齐；缩短 b/c 共用色标
# =========================
fig.canvas.draw()

pos_a = ax_a.get_position()
pos_b = ax_b.get_position()
pos_c = ax_c.get_position()

pos_cbar_a = cax_a.get_position()
pos_cbar_bc = cax_bc.get_position()


# ---- 图 a 保持：顶端对齐 b，底端对齐 c ----
new_a_y0 = pos_c.y0
new_a_y1 = pos_b.y1
new_a_height = new_a_y1 - new_a_y0

ax_a.set_position([
    pos_a.x0,
    new_a_y0,
    pos_a.width,
    new_a_height,
])

cax_a.set_position([
    pos_cbar_a.x0,
    new_a_y0,
    pos_cbar_a.width,
    new_a_height,
])


# ---- 缩短 b/c 共用色标，并在两幅地图整体高度内垂直居中 ----
bc_total_y0 = pos_c.y0
bc_total_y1 = pos_b.y1
bc_total_height = bc_total_y1 - bc_total_y0

new_cbar_bc_height = bc_total_height * BC_CBAR_HEIGHT_RATIO
new_cbar_bc_y0 = (
    bc_total_y0
    + (bc_total_height - new_cbar_bc_height) / 2
)

cax_bc.set_position([
    pos_cbar_bc.x0,
    new_cbar_bc_y0,
    pos_cbar_bc.width,
    new_cbar_bc_height,
])


# =========================
# 保存与展示
# =========================
fig.savefig(
    OUTPUT_PATH,
    dpi=600,
    bbox_inches="tight",
)

plt.show()

print("[SAVED]", OUTPUT_PATH)