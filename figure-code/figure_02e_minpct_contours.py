# Purpose: figure 02e minpct contours.
# Source: hdfcanshu_vs.ipynb; see docs/source_coverage.csv for cell provenance.
# Run order and required inputs: docs/workflow.md.
# Run from repository root: python -m figures.figure_02e_minpct_contours


from publication_config import resource_path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.lines as mlines
from matplotlib.ticker import MaxNLocator

# 输入输出路径
input_folder = resource_path('pf_processed', 'environment/filtered_land_regions/')
output_folder = resource_path('figures', 'figure_02e/')
os.makedirs(output_folder, exist_ok=True)

# 获取并分类文件（先HL后FL）
hl_files = sorted([f for f in os.listdir(input_folder) if f.startswith("HL") and f.endswith('.csv')])
fl_files = sorted([f for f in os.listdir(input_folder) if f.startswith("FL") and f.endswith('.csv')])
csv_files = hl_files + fl_files  # 关键修改：确保HL文件在前

# 设置 bin 范围
x_bins = np.linspace(200, 320, 100)
y_bins = np.linspace(200, 320, 100)

# 颜色方案（完全不变）
region_colors = {
    **{f'FL{i}': ['#F1A809', '#F0E442', '#D53C00', '#CC79A7'][i-1] for i in range(1,5)},
    **{f'HL{i}': ['#1B9E77', '#0072B2', '#56B4E9', '#3B4CC0'][i-1] for i in range(1,5)}
}

# 创建图像（不变）
plt.figure(figsize=(6, 6))
file_handles = {}

# 遍历文件（现在HL会先被处理）
for file_name in tqdm(csv_files, desc="Processing files"):
    file_path = os.path.join(input_folder, file_name)
    df = pd.read_csv(file_path)
    df = df[df['ELEV'] <= 2000]
    
    # 获取区域编号和颜色（完全不变）
    region_num = int(file_name[2])
    region = f"{file_name[:2]}{region_num}"
    color = region_colors.get(region, 'gray')
    
    # 数据筛选（完全不变）
    x = df['MIN85PCT'].dropna()
    y = df['MIN37PCT'].dropna()
    
    # 2D直方图统计（完全不变）
    H, xedges, yedges = np.histogram2d(x, y, bins=[x_bins, y_bins])
    H = H.T
    
    # 计算50th分位数阈值（完全不变）
    H_flat = H.flatten()
    H_sorted = np.sort(H_flat[H_flat > 0])
    cum_freq = np.cumsum(H_sorted) / np.sum(H_sorted)
    h_50 = H_sorted[np.searchsorted(cum_freq, 0.50)]
    
    # 绘制50th分位线（完全不变）
    contour = plt.contour(xedges[:-1], yedges[:-1], H, 
                         levels=[h_50], 
                         colors=[color], 
                         linewidths=3,
                         linestyles='-')
    
    # 标记最大值点（完全不变）
    max_idx = np.unravel_index(np.argmax(H), H.shape)
    max_x = 0.5 * (xedges[max_idx[1]] + xedges[max_idx[1] + 1])
    max_y = 0.5 * (yedges[max_idx[0]] + yedges[max_idx[0] + 1])
    
    marker = 'o' if file_name.startswith('HL') else 's'  # HL圆形，FL方形
    plt.scatter(max_x, max_y, color=color, 
                marker=marker, s=100)
    
    # 添加参考线（完全不变）
    plt.axhline(y=max_y, color=color, linestyle='--', linewidth=1.5, alpha=0.8)
    plt.axvline(x=max_x, color=color, linestyle='--', linewidth=1.5, alpha=0.8)
    
    # 保存图例元素（完全不变）
    file_handles[region] = {
        'color': color,
        'contour': contour.collections[0],
        'marker': marker
    }

# 创建自定义图例（完全不变）
legend_elements = []
for region, props in file_handles.items():
    line = mlines.Line2D([], [], 
                        color=props['color'],
                        linestyle='-',
                        linewidth=2,
                        label=f'{region}')
    marker = mlines.Line2D([], [], 
                          color=props['color'],
                          marker=props['marker'],
                          markersize=8,
                          markeredgecolor='white',
                          linestyle='None',
                          label=f'{region} Max')
    legend_elements.extend([marker])

# 图形设置（完全不变）
plt.title("(e) MIN37/85PCT 50th Percentile ", fontsize=25, fontweight='bold',pad=10)
plt.xlabel("MIN85PCT (K)", fontsize=20)
plt.ylabel("MIN37PCT (K)", fontsize=20)
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlim(250, 300)
plt.ylim(250, 300)
plt.xticks(fontsize=20)  # X 轴刻度字体大小
plt.yticks(fontsize=20)  # Y 轴刻度字体大小
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
from matplotlib.ticker import MultipleLocator
plt.gca().xaxis.set_major_locator(MultipleLocator(10)) # X 轴每隔 5
plt.gca().yaxis.set_major_locator(MultipleLocator(10)) # Y 轴每隔 5

# 添加图例（完全不变）
plt.legend(handles=legend_elements, 
          loc='lower right', 
          fontsize=16, 
          ncol=2,frameon=False)


# 保存图像（完全不变）
output_path = os.path.join(output_folder, 'figure_02e_minpct_contours.png')
plt.savefig(output_path, dpi=600, bbox_inches='tight')
print(f"图像已保存至: {output_path}")