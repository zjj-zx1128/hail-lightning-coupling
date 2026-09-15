# Purpose: extract pf environment.
# Source: hdfcanshu_data.ipynb; see docs/source_coverage.csv for cell provenance.
# Run order and required inputs: docs/workflow.md.
# Run from repository root: python -m preprocessing.extract_pf_environment


from publication_config import resource_path
import pandas as pd
import numpy as np
from tqdm import tqdm
import pyhdf.SD as H

# 读取 HDF 文件中指定变量的数据
def read_sds(var, path):
    out = {}
    for v in tqdm(var, desc="Reading HDF files"): 
        file = path + v.upper() + '.HDF'
        tqdm.write(f"  → Reading {v.upper()}.HDF")
        a = H.SD(file)
        dic = a.datasets()
        keys = dic.keys()
        for n in keys:
            sd = a.select(n)
            d = sd.get()
            out[n] = d
    return out

# === 主程序入口 ===
data_dir = resource_path('observations', 'data/pfdata/')
output_file = resource_path('pf_processed', 'environment/variablesdata.csv')
var = ["LAT", "LON", "YEAR", "MONTH", "DAY", "HOUR", 
       "LANDOCEAN", "ELEV", "FLS10A_WWLLN_ND", "FLS10B_WWLLN_ND",
       "MAXHT20", "MAXHT30", "MAXHT40",
       "T_MAXHT20", "T_MAXHT30", "T_MAXHT40","T_MAXHT",
       "echo_top_05-44","T2M",
       "MIN37PCT", "MIN85PCT", "CAPE", "CIN", "DEG0L"]

# 读取数据
data = read_sds(var, data_dir)

# 基准长度
base_len = len(data['LAT'])

# 创建 DataFrame 并处理补齐
df = pd.DataFrame()
for key in data:
    arr = np.ravel(data[key])
    if len(arr) < base_len:
        print(f"{key} 补齐 NaN：{len(arr)} → {base_len}")
        arr = np.pad(arr, (0, base_len - len(arr)), constant_values=np.nan)
    df[key] = arr

# 保存 CSV
print(f"保存 CSV 文件到 {output_file}...")
df.to_csv(output_file, index=False)
print("CSV 文件保存成功！")