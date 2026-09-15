# Purpose: figure 02abcd maxdbz profiles.
# Source: maxdbz.ipynb; see docs/source_coverage.csv for cell provenance.
# Run order and required inputs: docs/workflow.md.
# Run from repository root: python -m figures.figure_02abcd_maxdbz_profiles


from publication_config import resource_path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# 设置Arial字体
plt.rcParams['font.family'] = 'Arial'

# 输入输出路径
input_folder = resource_path('pf_processed', 'maxdbz/filtered_regions_interpolated/')
output_folder = resource_path('figures', 'figure_02abcd/')
os.makedirs(output_folder, exist_ok=True)

# 文件列表
hl_files = [f"HL{i}.csv" for i in range(1, 5)]
fl_files = [f"FL{i}.csv" for i in range(1, 5)]

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
# 注意：plt.plot() 中 markersize 表示标记显示尺寸
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

# 分位数配置
quantiles = {
    "50th": 50,
    "90th": 90,
    "95th": 95,
    "99th": 99
}

panel_labels = {
    "50th": "(a)",
    "90th": "(b)",
    "95th": "(c)",
    "99th": "(d)"
}


# 处理单个文件，返回分位线与对应温度
def extract_quantile_profile(file_path, q_value):
    df = pd.read_csv(file_path)
    df = df[df['ELEV'] <= 2000]

    dbz_columns = [f"maxdbz{i+1}" for i in range(40)]
    temp_columns = [f"t_ele{i+1}" for i in range(40)]

    dbz_data = df[dbz_columns].to_numpy()
    temp_data = df[temp_columns].to_numpy() - 273.15  # 转换为°C

    q_list = []
    temp_mean_list = []

    for i in range(40):
        col_dbz = dbz_data[:, i]
        col_temp = temp_data[:, i]
        mask = col_dbz >= 0

        valid_dbz = col_dbz[mask]
        valid_temp = col_temp[mask]

        if len(valid_dbz) > 0:
            q_val = np.percentile(valid_dbz, q_value)
            mean_temp = np.mean(valid_temp)

            q_list.append(q_val)
            temp_mean_list.append(mean_temp)

    return q_list, temp_mean_list


# 主循环：每个分位数绘一张图
for q_label, q_value in quantiles.items():
    # 创建图像
    plt.figure(figsize=(6, 8))

    # 1. 绘制HL区域
    for i, f in enumerate(hl_files):
        region = f"HL{i+1}"
        path = os.path.join(input_folder, f)

        q_vals, temps = extract_quantile_profile(path, q_value)

        marker = region_markers.get(region, "o")
        marker_size = region_marker_sizes.get(region, 8)

        plt.plot(
            q_vals,
            temps,
            color=region_colors[region],
            marker=marker,
            markersize=marker_size,
            markeredgecolor='w',
            markeredgewidth=1,
            linestyle='-',
            linewidth=3,
            label=region,
            zorder=2
        )

    # 2. 绘制FL区域
    for i, f in enumerate(fl_files):
        region = f"FL{i+1}"
        path = os.path.join(input_folder, f)

        q_vals, temps = extract_quantile_profile(path, q_value)

        marker = region_markers.get(region, "o")
        marker_size = region_marker_sizes.get(region, 8)

        plt.plot(
            q_vals,
            temps,
            color=region_colors[region],
            marker=marker,
            markersize=marker_size,
            markeredgecolor='w',
            markeredgewidth=1,
            linestyle='-',
            linewidth=3,
            label=region,
            zorder=3
        )

    # 图形设置
    panel = panel_labels.get(q_label, "")
    plt.title(
        f"{panel} MAXDBZ {q_label} Quantile Profile",
        fontsize=25,
        fontweight='bold',
        pad=15
    )

    plt.xlabel("Maximum Reflectivity (dBZ)", fontsize=20)
    plt.ylabel("Temperature (°C)", fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.xlim(10, 60)
    plt.ylim(-90, 30)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.gca().invert_yaxis()  # 温度高度坐标反转

    # 坐标轴刻度设置为整数
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    # 只有图(a)显示图例，其余不显示
    if q_label == "50th":
        plt.legend(
            loc="upper right",
            ncol=2,
            fontsize=20,
            frameon=False,
            markerscale=1.5
        )

    # 保存图像
    output_path = os.path.join(output_folder, f'figure_02abcd_maxdbz_{q_label}.png')
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"图表已保存至: {output_path}")