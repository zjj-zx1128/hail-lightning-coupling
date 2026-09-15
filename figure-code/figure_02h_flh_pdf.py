# Purpose: figure 02h flh pdf.
# Source: hdfcanshu_vs.ipynb; see docs/source_coverage.csv for cell provenance.
# Run order and required inputs: docs/workflow.md.
# Run from repository root: python -m figures.figure_02h_flh_pdf


from publication_config import resource_path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib import cm

# 输入输出路径
input_folder = resource_path('pf_processed', 'environment/filtered_land_regions/')
output_folder = resource_path('figures', 'figure_02h/')
os.makedirs(output_folder, exist_ok=True)

# 获取所有 CSV 文件并分类
csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
hl_files = [f for f in csv_files if f.startswith("HL")]
fl_files = [f for f in csv_files if f.startswith("FL")]

# 设置颜色
# 颜色方案（与第一个代码完全一致）
region_colors = {
    **{f'FL{i}': ['#F1A809', '#F0E442', '#D53C00', '#CC79A7'][i-1] for i in range(1, 5)},
    **{f'HL{i}': ['#1B9E77', '#0072B2', '#56B4E9', '#3B4CC0'][i-1] for i in range(1, 5)}
}

# marker 方案：HLi 与 FLi 使用相同 marker
region_markers = {
    "HL1": "o", "FL1": "o",   # 圆形
    "HL2": "*", "FL2": "*",   # 星形
    "HL3": "^", "FL3": "^",   # 三角形
    "HL4": "D", "FL4": "D",   # 菱形
}

# 每个区域单独设置 marker 大小
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

# 开始绘图
plt.figure(figsize=(6, 6))

# 1. 绘制 HL 文件
for i, f in enumerate(hl_files):
    df = pd.read_csv(os.path.join(input_folder, f))
    df = df[df['ELEV'] <= 2000]
    if "DEG0L" not in df.columns:
        continue

    # 获取区域编号（HL1 -> 1）
    region_num = int(f[2])  # 假设文件名格式为"HL1.csv"
    region = f"HL{region_num}"
    
    deg0l_values = df["DEG0L"]
    deg0l_values = deg0l_values[(deg0l_values > 0) & (~deg0l_values.isna())]
    
    fixed_bins = np.linspace(0, 7000, 26)

    counts, bin_edges = np.histogram(deg0l_values, bins=fixed_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

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

# 2. 绘制 FL 文件
for i, f in enumerate(fl_files):
    df = pd.read_csv(os.path.join(input_folder, f))
    df = df[df['ELEV'] <= 2000]
    if "DEG0L" not in df.columns:
        continue

    # 获取区域编号（FL1 -> 1）
    region_num = int(f[2])  # 假设文件名格式为"FL1.csv"
    region = f"FL{region_num}"
    
    deg0l_values = df["DEG0L"]
    deg0l_values = deg0l_values[(deg0l_values > 0) & (~deg0l_values.isna())]
    
    fixed_bins = np.linspace(0, 7000, 26)

    counts, bin_edges = np.histogram(deg0l_values, bins=fixed_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

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

# 图形设置
plt.xlabel("Freezing Level Height (m)", fontsize=20)
plt.ylabel("Probability Density", fontsize=20)
plt.title("(h) FLH", fontsize=25, fontweight='bold', pad=10)

plt.xlim(0, 7000)
plt.ylim(0, 0.0015)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True, linestyle='--', alpha=0.5)
# plt.legend(loc="upper left", fontsize=12, ncol=2, frameon=False)

# 保存图像
output_path = os.path.join(output_folder, 'figure_02h_flh_pdf.png')
plt.savefig(output_path, dpi=600, bbox_inches='tight')
plt.show()
print(f"图像已保存至: {output_path}")