# Purpose: figure 02i s06 pdf.
# Source: hdf_S06.ipynb; see docs/source_coverage.csv for cell provenance.
# Run order and required inputs: docs/workflow.md.
# Run from repository root: python -m figures.figure_02i_s06_pdf

# -*- coding: utf-8 -*-
"""
绘制 8 个区域的 S06 概率密度分布图
数据源：S06_corrected_with_latlon.npz

变量：
- S06
- LAT
- LON
- ELEV

筛选：
- ELEV <= 2000
- S06 > 0
- finite

输出：
figure_02i_s06_pdf.png
"""

from publication_config import resource_path

import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator

# =========================================================
# 1. 路径设置
# =========================================================
input_file = resource_path('pf_processed', 's06/S06_with_latlon.npz')
output_folder = resource_path('figures', 'figure_02i')
os.makedirs(output_folder, exist_ok=True)

output_path = os.path.join(output_folder, 'figure_02i_s06_pdf.png')


# =========================================================
# 2. 区域设置
# 格式: [lat_min, lat_max, lon_min, lon_max]
# =========================================================
regions = {
    "HL1": [35, 47, -104, -90],
    "HL2": [-42, -23, -68, -50],
    "HL3": [35, 55, -10, 20],
    "HL4": [35, 55, 112, 138],
    "FL1": [24, 35, -98, -78],
    "FL2": [-10, 5, -76, -58],
    "FL3": [-8, 5, 12, 35],
    "FL4": [-12, 8, 95, 150]
}

hl_regions = ["HL1", "HL2", "HL3", "HL4"]
fl_regions = ["FL1", "FL2", "FL3", "FL4"]


# =========================================================
# 3. 颜色设置
# =========================================================
region_colors = {
    "FL1": "#F1A809",
    "FL2": "#F0E442",
    "FL3": "#D53C00",
    "FL4": "#CC79A7",
    "HL1": "#1B9E77",
    "HL2": "#0072B2",
    "HL3": "#56B4E9",
    "HL4": "#3B4CC0"
}


# =========================================================
# 4. marker 设置
# HLi 与 FLi 使用相同 marker
# =========================================================
region_markers = {
    "HL1": "o", "FL1": "o",   # 圆形
    "HL2": "*", "FL2": "*",   # 星形
    "HL3": "^", "FL3": "^",   # 三角形
    "HL4": "D", "FL4": "D",   # 菱形
}

# 每个区域单独设置 marker 大小
# 注意：plt.plot() 中 markersize 与 scatter 的 s 不同
region_marker_sizes = {
    "HL1": 7,
    "HL2": 12,
    "HL3": 8,
    "HL4": 6,

    "FL1": 7,
    "FL2": 12,
    "FL3": 8,
    "FL4": 6,
}


# =========================================================
# 5. 读取数据
# =========================================================
data = np.load(input_file)

S06 = data["S06"].astype(np.float32)
LAT = data["LAT"].astype(np.float32)
LON = data["LON"].astype(np.float32)
ELEV = data["ELEV"].astype(np.float32)

print("数据读取完成：")
print("S06 shape :", S06.shape)
print("LAT shape :", LAT.shape)
print("LON shape :", LON.shape)
print("ELEV shape:", ELEV.shape)


# =========================================================
# 6. 设置 bin
# =========================================================
fixed_bins = np.linspace(0, 60, 27)
bin_centers = (fixed_bins[:-1] + fixed_bins[1:]) / 2


# =========================================================
# 7. 检查各区域 S06 PDF 是否重合
# =========================================================
region_pdf = {}
region_sample_count = {}

for region, bounds in regions.items():
    lat_min, lat_max, lon_min, lon_max = bounds

    mask = (
        (LAT >= lat_min) & (LAT <= lat_max) &
        (LON >= lon_min) & (LON <= lon_max) &
        (ELEV <= 2000) &
        np.isfinite(S06) &
        (S06 > 0)
    )

    region_values = S06[mask]
    region_sample_count[region] = region_values.size

    if region_values.size >= 10:
        counts, _ = np.histogram(region_values, bins=fixed_bins, density=True)
        region_pdf[region] = counts
    else:
        region_pdf[region] = np.full(len(fixed_bins) - 1, np.nan)

print("\n========== 区域样本数 ==========")
for region in regions:
    print(f"{region}: {region_sample_count[region]}")

print("\n========== FL3 与 FL4 PDF 差异检查 ==========")
fl3 = region_pdf["FL3"]
fl4 = region_pdf["FL4"]

diff = fl3 - fl4

print("FL3 counts:", fl3)
print("FL4 counts:", fl4)
print("max abs difference:", np.nanmax(np.abs(diff)))
print("mean abs difference:", np.nanmean(np.abs(diff)))
print("sum abs difference :", np.nansum(np.abs(diff)))

if np.allclose(fl3, fl4, rtol=1e-6, atol=1e-8):
    print("结论：FL3 和 FL4 的 PDF 数值几乎完全一致。")
else:
    print("结论：FL3 和 FL4 的 PDF 不完全一致，只是视觉上非常接近或被覆盖。")


# =========================================================
# 8. 绘图
# =========================================================
plt.figure(figsize=(6, 6))

ymax_all = []


# ---------------------------------------------------------
# 8.1 先绘制 HL
# ---------------------------------------------------------
for region in hl_regions:
    lat_min, lat_max, lon_min, lon_max = regions[region]

    mask = (
        (LAT >= lat_min) & (LAT <= lat_max) &
        (LON >= lon_min) & (LON <= lon_max) &
        (ELEV <= 2000) &
        np.isfinite(S06) &
        (S06 > 0)
    )

    region_values = S06[mask]

    print(f"{region}: 样本数 = {region_values.size}")

    if region_values.size < 10:
        print(f"{region} 有效样本过少，跳过绘图。")
        continue

    counts, bin_edges = np.histogram(region_values, bins=fixed_bins, density=True)
    ymax_all.append(np.nanmax(counts))

    marker = region_markers.get(region, "o")
    marker_size = region_marker_sizes.get(region, 6)

    plt.plot(
        bin_centers, counts,
        marker=marker,
        markersize=marker_size,
        markeredgecolor='w',
        markeredgewidth=1,
        linestyle='-',
        linewidth=3,
        color=region_colors[region],
        label=region,
        zorder=2
    )


# ---------------------------------------------------------
# 8.2 再绘制 FL
# ---------------------------------------------------------
for region in fl_regions:
    lat_min, lat_max, lon_min, lon_max = regions[region]

    mask = (
        (LAT >= lat_min) & (LAT <= lat_max) &
        (LON >= lon_min) & (LON <= lon_max) &
        (ELEV <= 2000) &
        np.isfinite(S06) &
        (S06 > 0)
    )

    region_values = S06[mask]

    print(f"{region}: 样本数 = {region_values.size}")

    if region_values.size < 10:
        print(f"{region} 有效样本过少，跳过绘图。")
        continue

    counts, bin_edges = np.histogram(region_values, bins=fixed_bins, density=True)
    ymax_all.append(np.nanmax(counts))

    linestyle = '--' if region == "FL3" else '-'
    zorder = 5 if region == "FL3" else 3

    marker = region_markers.get(region, "o")
    marker_size = region_marker_sizes.get(region, 6)

    plt.plot(
        bin_centers, counts,
        marker=marker,
        markersize=marker_size,
        markeredgecolor='w',
        markeredgewidth=1,
        linestyle=linestyle,
        linewidth=3,
        color=region_colors[region],
        label=region,
        zorder=zorder
    )


# =========================================================
# 9. 图像样式
# =========================================================
plt.xlabel("S06 (m/s)", fontsize=20)
plt.ylabel("Probability Density", fontsize=20)
plt.title("(i) S06", fontsize=25, fontweight='bold', pad=10)

# plt.legend(loc="upper right", ncol=2, fontsize=12, frameon=False)

plt.xlim(0, 60)
plt.ylim(0, 0.13)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.grid(True, linestyle='--', alpha=0.5)

# x 轴刻度尽量显示整数
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

# 主图黑框
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(1)
    spine.set_color("black")


# =========================================================
# 10. 保存图像
# =========================================================
plt.savefig(output_path, dpi=600, bbox_inches='tight')
plt.show()

print(f"图像已保存至: {output_path}")