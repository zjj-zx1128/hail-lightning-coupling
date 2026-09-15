# Purpose: interpolate maxdbz temperature.
# Source: maxdbz.ipynb; see docs/source_coverage.csv for cell provenance.
# Run order and required inputs: docs/workflow.md.
# Run from repository root: python -m preprocessing.interpolate_maxdbz_temperature


from publication_config import resource_path
import os
import pandas as pd
import numpy as np

input_dir = resource_path('pf_processed', 'maxdbz/filtered_regions')
output_dir = resource_path('pf_processed', 'maxdbz/filtered_regions_interpolated')
os.makedirs(output_dir, exist_ok=True)

# 原始高度和温度列
hgt_cols = [f"hgt{i}" for i in range(1, 29)]
temp_cols = [f"t{i}" for i in range(1, 29)]

# 目标插值高度（米）
target_heights = np.arange(500, 20500, 500)  # 40层，每500米一层
t_interp_cols = [f"t_ele{i}" for i in range(1, len(target_heights) + 1)]

def process_file(file_path, out_path):
    df = pd.read_csv(file_path)

    interp_temps = []
    for idx, row in df.iterrows():
        z_orig = row[hgt_cols].values.astype(float)/9.8 #换算为m
        t_orig = row[temp_cols].values.astype(float)

        if np.any(pd.isna(z_orig)) or np.any(pd.isna(t_orig)):
            interp_temps.append([np.nan] * len(target_heights))
        else:
            # 这里插值时确保原始高度是递增的，否则np.interp会报错
            # 对z_orig排序及对应t_orig排序
            sort_idx = np.argsort(z_orig)
            z_sorted = z_orig[sort_idx]
            t_sorted = t_orig[sort_idx]
            
            # 插值，超出范围的插值结果用边界值填充
            interp_t = np.interp(target_heights, z_sorted, t_sorted, left=t_sorted[0], right=t_sorted[-1])
            interp_temps.append(interp_t)

    interp_df = pd.DataFrame(interp_temps, columns=t_interp_cols)
    df_new = pd.concat([df, interp_df], axis=1)
    df_new.to_csv(out_path, index=False)
    print(f"✅ {os.path.basename(file_path)} 处理完成，保存至 {out_path}")

# 遍历目录批量处理所有CSV文件
for filename in os.listdir(input_dir):
    if filename in ("HL1.csv", "HL2.csv", "HL3.csv", "HL4.csv", "FL1.csv", "FL2.csv", "FL3.csv", "FL4.csv"):
        in_fp = os.path.join(input_dir, filename)
        out_fp = os.path.join(output_dir, filename)
        process_file(in_fp, out_fp)

print("所有文件处理完成！")