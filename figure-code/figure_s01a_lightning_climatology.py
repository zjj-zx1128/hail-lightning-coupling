# Purpose: figure s01a lightning climatology.
# Source: cmip6_figure_difference_flash_hail.ipynb; see docs/source_coverage.csv for cell provenance.
# Run order and required inputs: docs/workflow.md.
# Run from repository root: python -m figures.figure_s01a_lightning_climatology


from publication_config import resource_path
# -*- coding: utf-8 -*-
# [configured path; see config.json]
from pyhdf.SD import SD, SDC
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

# shapely (for land mask)
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union
from shapely import vectorized

plt.rcParams['font.family'] = 'Arial'

# =========================
# 0) Read HDF4
# =========================
file_path = resource_path('observations', 'data/lisotd/LISOTD_HRFC_V2.3.2015.hdf')
hdf = SD(file_path, SDC.READ)

flash = hdf.select('HRFC_COM_FR')[:] / 365.0  # per day
lat = hdf.select('Latitude')[:]
lon = hdf.select('Longitude')[:]

# =========================
# 1) Resample to 2°
# =========================
def block_mean(arr, factor=4):
    """块平均降采样 0.5° → 2°"""
    m, n = arr.shape
    arr = arr[:m - m % factor, :n - n % factor]
    return arr.reshape(m // factor, factor, n // factor, factor).mean(axis=(1, 3))

flash_2deg = block_mean(flash, factor=4)

# 2° 中心点纬经度（用原始 lat/lon 做块平均得到中心序列）
lat_2deg = block_mean(np.tile(lat[:, None], (1, flash.shape[1])), factor=4)[:, 0]
lon_2deg = block_mean(np.tile(lon[None, :], (flash.shape[0], 1)), factor=4)[0, :]

# =========================
# 2) Lat band: ±63°
# =========================
LAT_MIN, LAT_MAX = -63, 63
lat_keep = (lat_2deg >= LAT_MIN) & (lat_2deg <= LAT_MAX)

flash_2deg = flash_2deg[lat_keep, :]
lat_2deg = lat_2deg[lat_keep]

# 边界网格（用于 pcolormesh）
lat_edges = np.linspace(lat_2deg.min() - 1.0, lat_2deg.max() + 1.0, flash_2deg.shape[0] + 1)  # 近似 2° edge
lon_edges = np.linspace(-180, 180, flash_2deg.shape[1] + 1)
lon_grid, lat_grid = np.meshgrid(lon_edges, lat_edges)

# =========================
# 3) Land mask on 2° centers (Natural Earth)
# =========================
# 3.1 读 Natural Earth land polygons（cartopy 自带）
land_shp = shpreader.natural_earth(resolution='110m', category='physical', name='land')
geoms = list(shpreader.Reader(land_shp).geometries())
land_union = unary_union(geoms)  # 合并为一个(多)面

# 3.2 构建中心点网格，判断是否落在陆地多边形内
LONC, LATC = np.meshgrid(lon_2deg, lat_2deg)  # centers
land_mask = vectorized.contains(land_union, LONC, LATC)  # True=land, False=ocean

# 3.3 海洋置 NaN（只绘陆地）
flash_land = flash_2deg.copy()
flash_land[~land_mask] = np.nan

print("[INFO] land fraction =", np.isfinite(flash_land).mean())
print(f"最大值（每天平均闪电频率, land-only）: {np.nanmax(flash_land)} flashes/day/km²")

# =========================
# 4) Custom colormap (ez_color)
# =========================
import xni.readhdf as readhdf
import xni.lib_ni as lib_ni
colors = readhdf.data(resource_path('colormaps', 'ez_color.hdf'))
a = np.array([colors["R"][0:55] / 255., colors["G"][0:55] / 255., colors["B"][0:55] / 255.]).T
a = np.delete(a, np.arange(1, 15, 3), 0)
a = np.delete(a, np.arange(1, 10, 2), 0)
a = np.delete(a, np.arange(10, 25, 3), 0)
cmap = lib_ni.array2cmap(a)

# =========================
# 5) Plot
# =========================
fig = plt.figure(figsize=(14, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

im = ax.pcolormesh(
    lon_grid[:-1, :-1], lat_grid[:-1, :-1],
    flash_land,                 # <- land-only
    cmap=cmap,
    vmin=0, vmax=0.26,
    shading='auto',
    transform=ccrs.PlateCarree()
)

ax.coastlines(resolution='110m', linewidth=1)
ax.add_feature(cfeature.BORDERS, linewidth=0.6, linestyle=':')
ax.set_extent([-180, 180, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())

gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.right_labels = False
gl.top_labels = False
gl.xlabel_style = {'size': 18}  # 经度刻度字体
gl.ylabel_style = {'size': 18}  # 纬度刻度字体

# ---- colorbar aligned to axes box ----
fig.canvas.draw()
pos = ax.get_position()
pad = 0.012
cbar_w = 0.018
cax = fig.add_axes([pos.x1 + pad, pos.y0, cbar_w, pos.height])
cbar = fig.colorbar(im, cax=cax, orientation="vertical")
cbar.set_label("LIS/OTD Flash Rate Density (flash km$^{-2}$ day$^{-1}$)",fontsize=14)
cbar.ax.tick_params(labelsize=18)

ax.set_title("(a) Global Distribution of Lightning Frequency", fontsize=25, pad=10,fontweight="bold")

plt.savefig(resource_path('figures', 'figure_s01a_lightning_climatology.jpg'),
            dpi=600, bbox_inches='tight')
plt.show()
