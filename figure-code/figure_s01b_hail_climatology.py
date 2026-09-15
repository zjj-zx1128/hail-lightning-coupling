# Purpose: figure s01b hail climatology.
# Source: cmip6_figure_difference_flash_hail.ipynb; see docs/source_coverage.csv for cell provenance.
# Run order and required inputs: docs/workflow.md.
# Run from repository root: python -m figures.figure_s01b_hail_climatology


from publication_config import resource_path
# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import xni.readhdf as readhdf
import xni.lib_ni as lib_ni

from shapely.ops import unary_union
from shapely import vectorized

plt.rcParams["font.family"] = "Arial"

# =========================
# 0) 读取 pickle 数据
# =========================
with open(resource_path('observations', 'hail_ni/plot_data.pickle'), "rb") as f:
    data = pickle.load(f)

LON = data["LON"]              # edges: (181, 66)
LAT = data["LAT"]              # edges: (181, 66)
fraction = data["fraction"]    # data : (180, 65)

print("fraction", fraction.shape)
print("LAT", LAT.shape, "LON", LON.shape)
print("LAT sample", LAT[:5, 0], LAT[-5:, 0])

# =========================
# 1) 自定义颜色条（ez_color）
# =========================
colors = readhdf.data(resource_path('colormaps', 'ez_color.hdf'))
a = np.array([
    colors["R"][0:55] / 255.,
    colors["G"][0:55] / 255.,
    colors["B"][0:55] / 255.
]).T
a = np.delete(a, np.arange(1, 15, 3), 0)
a = np.delete(a, np.arange(1, 10, 2), 0)
a = np.delete(a, np.arange(10, 25, 3), 0)
cmap = lib_ni.array2cmap(a)

# =========================
# 2) 裁剪纬度到 ±63°
#    关键：你的纬度沿 axis=1 变化（LAT[:,0] 恒定）
# =========================
LAT_MIN, LAT_MAX = -63, 63

# edges 1D：取第一行即可得到 lat edges（长度 66）
lat_edges_1d = LAT[0, :]  # (nlat_edges,)
# centers 1D：长度 nlat，对应 fraction 的第二维
lat_centers_1d = (lat_edges_1d[:-1] + lat_edges_1d[1:]) / 2.0  # (nlat_centers,)

col_mask = (lat_centers_1d >= LAT_MIN) & (lat_centers_1d <= LAT_MAX)
col_idx = np.where(col_mask)[0]
if col_idx.size == 0:
    raise ValueError("未找到落在 [-63,63] 的纬向列索引，请检查 LAT 是否为度、以及网格方向。")

j0, j1 = col_idx[0], col_idx[-1]

# 裁剪数据：fraction 的列 (lat dimension)
fraction = fraction[:, j0:j1+1]           # (nlon, nlat_sel)

# 裁剪边界网格：+2 才闭合（edges 比 centers 多 1）
LAT = LAT[:, j0:j1+2]                     # (nlon_edges, nlat_sel+1)
LON = LON[:, j0:j1+2]                     # (nlon_edges, nlat_sel+1)

print("[INFO] After lat clip:", fraction.shape, LAT.shape, LON.shape)

# =========================
# 3) 仅绘制陆地：land mask (Natural Earth)
# =========================
# 3.1 网格中心点（用于陆地判定）
lon_edges_1d = LON[:, 0]  # (nlon_edges,)
lon_centers_1d = (lon_edges_1d[:-1] + lon_edges_1d[1:]) / 2.0  # (nlon,)

lat_edges_1d_sel = LAT[0, :]  # (nlat_edges_sel,)
lat_centers_1d_sel = (lat_edges_1d_sel[:-1] + lat_edges_1d_sel[1:]) / 2.0  # (nlat_sel,)

LONC, LATC = np.meshgrid(lon_centers_1d, lat_centers_1d_sel, indexing="ij")  # (nlon, nlat_sel)

# 3.2 读取 land 多边形并合并
land_shp = shpreader.natural_earth(resolution="110m", category="physical", name="land")
geoms = list(shpreader.Reader(land_shp).geometries())
land_union = unary_union(geoms)

# 3.3 判定中心点是否在陆地
land_mask = vectorized.contains(land_union, LONC, LATC)  # True=land

# 3.4 海洋置 NaN（只绘制陆地）
fraction_land = fraction.astype(float).copy()
fraction_land[~land_mask] = np.nan

print("[INFO] land fraction =", np.isfinite(fraction_land).mean())
print(f"最大值（冰雹观测, land-only）: {np.nanmax(fraction_land)}")

# =========================
# 4) 绘图
# =========================
fig = plt.figure(figsize=(14, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

im = ax.pcolormesh(
    LONC, LATC,
    fraction_land,   # <- land-only
    cmap=cmap,
    vmin=0,
    vmax=0.09,
    shading="auto",
    transform=ccrs.PlateCarree()
)

# 地图要素
ax.coastlines(resolution="110m", linewidth=1)
ax.add_feature(cfeature.BORDERS, linewidth=0.6, linestyle=":")
ax.set_extent([-180, 180, -63, 63], crs=ccrs.PlateCarree())

# 网格线与标签字号
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
gl.right_labels = False
gl.top_labels = False
gl.xlabel_style = {"size": 18}
gl.ylabel_style = {"size": 18}

# =========================
# 5) 颜色条：与主图框上下严格对齐
# =========================
fig.canvas.draw()
pos = ax.get_position()
pad = 0.012
cbar_w = 0.018
cax = fig.add_axes([pos.x1 + pad, pos.y0, cbar_w, pos.height])

cbar = fig.colorbar(im, cax=cax, orientation="vertical")
cbar.set_label("GPM Hail PF Occurrence", fontsize=16)
cbar.ax.tick_params(labelsize=18)

# 标题
ax.set_title("(b) Global Distribution of Hail Frequency", loc="center", fontsize=25, pad=10,fontweight="bold")

# 保存
out_png = resource_path('figures', 'figure_s01b_hail_climatology.jpg')
plt.savefig(out_png, dpi=600, bbox_inches="tight")
plt.show()

print("[SAVED]", out_png)
