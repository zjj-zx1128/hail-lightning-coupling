# Purpose: figure s11 change quadrants.
# Source: cmip6_figure_MMM_3models_joint2d_trend_with_markers.ipynb; see docs/source_coverage.csv for cell provenance.
# Run order and required inputs: docs/workflow.md.
# Run from repository root: python -m figures.figure_s11_change_quadrants

from __future__ import annotations

from publication_config import resource_path

import os
from pathlib import Path
import numpy as np
import pandas as pd

# ==================== 1. 用户配置 ====================
MODEL_BASE_DIRS = {
    "BCC-CSM2-MR": Path(resource_path('models', 'BCC-CSM2-MR')),
    "CanESM5": Path(resource_path('models', 'CanESM5')),
    "MIROC6": Path(resource_path('models', 'MIROC6')),
}
REFERENCE_MODEL = "CanESM5"
OBS_PARQ = (MODEL_BASE_DIRS[REFERENCE_MODEL] /
            "ml_feature_withSRTM" / "train_table_flash_hail_withSRTM_strictLand.parquet")
ELEVATION_PATH = Path(resource_path('observations', 'data/elevation/elevation_1KMmn_SRTM.tif'))
OUT_DIR = Path(resource_path('figures', 'figure_s11_change_quadrants'))
OUTPUT_STEM = 'figure_s11_change_quadrants'

DECADES = [(2050, 2059), (2060, 2069), (2070, 2079), (2080, 2089), (2090, 2099)]
VAR_FLASH, VAR_HAIL = "pred_flash", "pred_hail"
NC_LAT, NC_LON = "lat", "lon"
LAT, LON, LANDSEA = "Latitude", "Longitude", "Land_Sea"
OBS_FLASH, OBS_HAIL = "LISOTD_Flash", "ni_HailPF"
LAT_MAX = 63.0

# 类别色不用于表达数值大小；这是本图的设计选择，并非期刊指定配色。
CLASS_COLORS = {
    1: "#FDA876",  # Q1 柔和赭橙：闪电增加，冰雹增加
    2: "#1B9CE6",  # Q2 蓝色：闪电不增加，冰雹增加
    3: "#6B6969",  # Q3 中灰：闪电不增加，冰雹不增加
    4: "#FF5FB7",  # Q4 柔紫：闪电增加，冰雹不增加
}
FIGSIZE_MM = (180, 86)
FONT_FAMILY = "Arial"
FONT_SIZE = 7.0
TITLE_SIZE = 8.5
LEGEND_SIZE = 6.8
REGION_LABEL_SIZE = 6.8
DPI = 600
SHOW_FIGURE = True
ADD_REGIONS = True
ADD_ELEVATION_HATCH = True
ELEVATION_THRESHOLD = 2000.0
ELEVATION_HATCH = "//////"
HATCH_LINEWIDTH = 0.18
REGION_COLOR = "#333333"
MAP_AXES_RECT = [0.064, 0.255, 0.915, 0.67]
CLASS_LEGEND_ANCHOR = (0.51, 0.063)
HATCH_LEGEND_ANCHOR = (0.51, 0.004)
MAP_TITLE = "Spatial distribution of hail–lightning changes under SSP585"
# 图例使用 ΔL、ΔH：分别为未来多年代平均闪电、冰雹频率减去观测值。
# >0 / <=0 与原图3a分类一致，零变化明确归入<=0。
CLASS_LEGEND_CONDITIONS = {
    1: r"$\Delta L>0,\ \Delta H>0$",
    2: r"$\Delta L\leq0,\ \Delta H>0$",
    3: r"$\Delta L\leq0,\ \Delta H\leq0$",
    4: r"$\Delta L>0,\ \Delta H\leq0$",
}
# 可选本地 Cartopy 数据缓存根目录（已有 Natural Earth 文件时设置）。
# Cartopy 缺少海岸线数据时会尝试自动下载。
CARTOPY_DATA_DIR = None

# 区域范围与图1一致；标签放在框外以减少遮挡。
# (名称, 最小经度, 最大经度, 最小纬度, 最大纬度, 标签经度, 标签纬度)
REGIONS = [
    ("HL1", -104, -90, 35, 47, -97, 51),
    ("HL2", -68, -50, -42, -23, -59, -47),
    ("HL3", -10, 20, 35, 55, 5, 59),
    ("HL4", 112, 138, 35, 55, 125, 59),
    ("FL1", -98, -78, 24, 35, -69, 30),
    ("FL2", -76, -58, -10, 5, -86, -3),
    ("FL3", 12, 35, -8, 5, 23.5, 9),
    ("FL4", 95, 150, -12, 8, 122.5, -16),
]

# ==================== 2. 沿用图3的数据读取与平均方法 ====================
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

def build_grid_from_parquet(lat, lon, parquet_path: str | Path, value_col: str) -> np.ndarray:
    parquet_path = str(parquet_path)

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(parquet_path)

    df = pd.read_parquet(parquet_path)

    need = {LAT, LON, LANDSEA, value_col}
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

# ==================== 3. 四象限分类与统计 ====================
def classify_changes(pred_f, pred_h, obs_f, obs_h):
    """保持图3a的减法精度、有效样本和严格 >0 / <=0 条件。"""
    fields = [np.asarray(a) for a in (pred_f, pred_h, obs_f, obs_h)]
    if len({a.shape for a in fields}) != 1 or fields[0].ndim != 2:
        raise ValueError("Four input fields must share one 2-D grid.")
    df = fields[0] - fields[2]
    dh = fields[1] - fields[3]
    valid = np.isfinite(df) & np.isfinite(dh)
    masks = [
        valid & (df > 0) & (dh > 0),
        valid & (df <= 0) & (dh > 0),
        valid & (df <= 0) & (dh <= 0),
        valid & (df > 0) & (dh <= 0),
    ]
    membership = np.sum(np.stack(masks), axis=0)
    if not np.array_equal(membership, valid.astype(int)):
        raise AssertionError("Quadrants must be mutually exclusive and exhaustive.")
    n = int(valid.sum())
    if n == 0:
        raise ValueError("No grid cells with two finite changes.")
    classes = np.zeros(valid.shape, dtype=np.uint8)  # 0 为无效格点
    rows = []
    zero_f = int(np.count_nonzero(valid & (df == 0)))
    zero_h = int(np.count_nonzero(valid & (dh == 0)))
    f_nonpositive = "not increasing" if zero_f else "decreasing"
    h_nonpositive = "not increasing" if zero_h else "decreasing"
    labels = [
        "Lightning increasing; hail increasing",
        f"Lightning {f_nonpositive}; hail increasing",
        f"Lightning {f_nonpositive}; hail {h_nonpositive}",
        f"Lightning increasing; hail {h_nonpositive}",
    ]
    conditions = ["> 0, > 0", "<= 0, > 0", "<= 0, <= 0", "> 0, <= 0"]
    for k, mask in enumerate(masks, start=1):
        classes[mask] = k
        count = int(mask.sum())
        rows.append({
            "quadrant": f"Q{k}", "class_id": k,
            "delta_lightning_delta_hail": conditions[k-1],
            "description": labels[k-1], "hex_color": CLASS_COLORS[k],
            "grid_count": count, "total_valid_grid_count": n,
            "percent": 100.0 * count / n,
            "zero_delta_lightning_count": int(np.count_nonzero(mask & (df == 0))),
            "zero_delta_hail_count": int(np.count_nonzero(mask & (dh == 0))),
        })
    summary = pd.DataFrame(rows)
    if int(summary.grid_count.sum()) != n:
        raise AssertionError("Class counts do not sum to the denominator.")
    print(f"\n[CLASSIFICATION] valid N = {n}; categories are disjoint and exhaustive")
    print(f"[ZERO CHANGE] lightning = {zero_f}, hail = {zero_h}")
    print(summary[["quadrant", "grid_count", "percent"]].to_string(index=False))
    return classes, valid, df, dh, summary


def centers_to_edges(axis):
    """完整规则2°中心坐标转网格边界；不额外插值或改变分类。"""
    axis = np.asarray(axis, dtype=float)
    if axis.ndim != 1 or axis.size < 2 or not np.all(np.isfinite(axis)):
        raise ValueError("Invalid coordinate axis.")
    if not np.allclose(np.diff(axis), 2.0, rtol=0, atol=1e-5):
        raise ValueError("Expected a complete, ascending 2-degree grid.")
    return np.r_[axis[0] - 1.0, (axis[:-1] + axis[1:]) / 2.0, axis[-1] + 1.0]


def plot_map(classes, lat, lon, summary, output_dir):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.patheffects as pe
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.ticker import FixedLocator
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

    if CARTOPY_DATA_DIR is not None:
        cartopy.config["data_dir"] = str(CARTOPY_DATA_DIR)
    plt.rcParams.update({
        "font.family": FONT_FAMILY, "font.size": FONT_SIZE,
        "axes.unicode_minus": False, "pdf.fonttype": 42,
        "ps.fonttype": 42, "hatch.linewidth": HATCH_LINEWIDTH,
        "hatch.color": "#454545", "savefig.facecolor": "white",
    })
    # 只为绘图排序，对类别、坐标进行同样的重排。
    iy, ix = np.argsort(lat), np.argsort(lon)
    lat = np.asarray(lat)[iy]
    lon = np.asarray(lon)[ix]
    plotted = classes[np.ix_(iy, ix)]
    lat_edges, lon_edges = centers_to_edges(lat), centers_to_edges(lon)
    cmap = mcolors.ListedColormap([CLASS_COLORS[k] for k in range(1, 5)])
    cmap.set_bad((0, 0, 0, 0))
    norm = mcolors.BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], 4)
    crs = ccrs.PlateCarree()
    fig = plt.figure(figsize=tuple(mm / 25.4 for mm in FIGSIZE_MM))
    ax = fig.add_axes(MAP_AXES_RECT, projection=crs)
    ax.set_extent([-180, 180, -LAT_MAX, LAT_MAX], crs=crs)
    ax.set_facecolor("white")
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#F2F2F2",
                   edgecolor="none", zorder=0)
    # 2°方格逐格点填色；避免散点大小随图尺寸改变而制造空隙/重叠。
    ax.pcolormesh(lon_edges, lat_edges, np.ma.masked_equal(plotted, 0),
                  cmap=cmap, norm=norm, shading="flat", edgecolors="none",
                  antialiased=False, transform=crs, zorder=2)
    ax.coastlines(resolution="110m", color="#303030", linewidth=0.35, zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.22,
                   edgecolor="#707070", linestyle=":", zorder=4)

    if ADD_ELEVATION_HATCH:
        # 与图1一致：SRTM先平均到2°，再判断该2°海拔是否>2000 m。
        # 斜线为独立地形显示层，不参与有效样本筛选。
        import rioxarray as rxr
        from rasterio.enums import Resampling
        if not ELEVATION_PATH.exists():
            raise FileNotFoundError(f"Elevation file missing: {ELEVATION_PATH}")
        elevation = rxr.open_rasterio(ELEVATION_PATH, masked=True).squeeze()
        try:
            if elevation.rio.crs is None:
                elevation = elevation.rio.write_crs("EPSG:4326")
            elev_2deg = elevation.rio.reproject(
                "EPSG:4326", resolution=2.0, resampling=Resampling.average)
            high = elev_2deg > ELEVATION_THRESHOLD
            hatch = ax.contourf(elev_2deg["x"], elev_2deg["y"], high.to_numpy(),
                               levels=[0.5, 1.5], colors="none", hatches=[ELEVATION_HATCH],
                               transform=crs, zorder=6)
            # 兼容新旧 Matplotlib 的等值线接口。
            if hasattr(hatch, "set_edgecolor"):
                hatch.set_edgecolor("#454545")
                hatch.set_linewidth(0.0)
            else:
                for collection in hatch.collections:
                    collection.set_edgecolor("#454545")
                    collection.set_linewidth(0.0)
        finally:
            elevation.close()

    if ADD_REGIONS:
        for name, x0, x1, y0, y1, tx, ty in REGIONS:
            color = REGION_COLOR
            rect = Rectangle((x0, y0), x1-x0, y1-y0, facecolor="none",
                             edgecolor=color, linewidth=0.7, transform=crs, zorder=8)
            rect.set_path_effects([pe.Stroke(linewidth=1.15, foreground="white"), pe.Normal()])
            ax.add_patch(rect)
            ax.text(tx, ty, name, color=color, fontsize=REGION_LABEL_SIZE,
                    fontweight="bold", ha="center", va="center", transform=crs,
                    zorder=9, path_effects=[pe.withStroke(linewidth=1.3, foreground="white")])

    gl = ax.gridlines(draw_labels=True, linewidth=0.25, color="#737373",
                     alpha=0.35, linestyle="--", zorder=3)
    gl.top_labels = gl.right_labels = False
    gl.xlocator = FixedLocator([-120, -60, 0, 60, 120])
    gl.ylocator = FixedLocator([-60, -30, 0, 30, 60])
    gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    gl.xlabel_style = {"size": FONT_SIZE}
    gl.ylabel_style = {"size": FONT_SIZE}
    gl.xpadding = gl.ypadding = 3
    ax.spines["geo"].set_linewidth(0.5)
    ax.set_title(MAP_TITLE, fontsize=TITLE_SIZE, pad=7)

    rows_by_class = {int(row.class_id): row for row in summary.itertuples()}
    # Matplotlib多列图例按列填充，输入Q1,Q3,Q2,Q4，显示为：
    # 第一行 Q1 Q2；第二行 Q3 Q4。
    handles = [Patch(facecolor=CLASS_COLORS[k], edgecolor="none",
                     label=f"Q{k}: {CLASS_LEGEND_CONDITIONS[k]} "
                           f"({rows_by_class[k].percent:.2f}%)")
               for k in (1, 3, 2, 4)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=CLASS_LEGEND_ANCHOR,
               ncol=2, frameon=False, fontsize=LEGEND_SIZE, handlelength=1.2,
               handleheight=0.9, columnspacing=1.4, labelspacing=0.65)
    if ADD_ELEVATION_HATCH:
        fig.legend(handles=[Patch(facecolor="white", edgecolor="#454545",
                                  linewidth=HATCH_LINEWIDTH, hatch=ELEVATION_HATCH,
                                  label="Elevation > 2,000 m")],
                   loc="lower center", bbox_to_anchor=HATCH_LEGEND_ANCHOR, frameon=False,
                   fontsize=LEGEND_SIZE, handlelength=1.5)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        path = output_dir / f"{OUTPUT_STEM}.{extension}"
        # 保持180 mm成图宽度，不使用tight裁剪改变尺寸。
        fig.savefig(path, dpi=DPI, facecolor="white")
        print("[SAVED]", path)
    if SHOW_FIGURE:
        plt.show()
    plt.close(fig)


def main():
    # 重新读取原始NC数据，不使用原图3的float32调试NPZ，
    # 以免近零变化因额外舍入而改变象限。
    for name, base in MODEL_BASE_DIRS.items():
        for y0, y1 in DECADES:
            path = get_model_nc_path(base, y0, y1)
            if not path.exists():
                raise FileNotFoundError(f"[{name}] missing prediction: {path}")
    if not OBS_PARQ.exists():
        raise FileNotFoundError(f"Observed table missing: {OBS_PARQ}")
    if ADD_ELEVATION_HATCH and not ELEVATION_PATH.exists():
        raise FileNotFoundError(f"Elevation file missing: {ELEVATION_PATH}")

    pred_f, pred_h, lat, lon, _, diagnostics = compute_mmm_multidecadal_mean_prediction()
    obs_f = build_grid_from_parquet(lat, lon, OBS_PARQ, OBS_FLASH)
    obs_h = build_grid_from_parquet(lat, lon, OBS_PARQ, OBS_HAIL)
    classes, valid, delta_f, delta_h, summary = classify_changes(pred_f, pred_h, obs_f, obs_h)
    for target in ("flash", "hail"):
        counts = diagnostics[f"nmodel_{target}_by_decade"][:, valid]
        n_incomplete = int(np.count_nonzero(np.any(counts < len(MODEL_BASE_DIRS), axis=0)))
        print(f"[COMPLETENESS] {target}: {n_incomplete} classified cells have fewer "
              "than 3 available models in at least one decade; original nanmean rule retained.")

    # 数值参照来自当前正文，不强行固定为这些比例。
    expected = np.array([76.81, 5.74, 5.94, 11.51])  # Q1, Q2, Q3, Q4
    if np.all(np.abs(summary.percent.to_numpy() - expected) < 0.0051):
        print("[CHECK] Rounded percentages match the current manuscript's Figure 3a.")
    else:
        print("[CHECK] Percentages differ from the current manuscript. Compare the input "
              "files/reference table with those used for Figure 3a; use the computed values.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / f"{OUTPUT_STEM}_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    grid_table = pd.DataFrame({
        "Latitude": lat2d[valid], "Longitude": lon2d[valid],
        "observed_lightning_frequency": obs_f[valid],
        "observed_hail_frequency": obs_h[valid],
        "future_mean_lightning_frequency": pred_f[valid],
        "future_mean_hail_frequency": pred_h[valid],
        "delta_lightning_frequency": delta_f[valid],
        "delta_hail_frequency": delta_h[valid], "class_id": classes[valid],
    })
    grid_table["quadrant"] = grid_table["class_id"].map({k: f"Q{k}" for k in range(1, 5)})
    grid_path = OUT_DIR / f"{OUTPUT_STEM}_grid_classes.csv"
    grid_table.to_csv(grid_path, index=False, encoding="utf-8-sig")
    print("[SAVED]", summary_path)
    print("[SAVED]", grid_path)
    plot_map(classes, lat, lon, summary, OUT_DIR)


if __name__ == "__main__":
    main()