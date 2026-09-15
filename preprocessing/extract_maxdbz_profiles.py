# Purpose: extract maxdbz profiles.
# Source: maxdbz.ipynb; see docs/source_coverage.csv for cell provenance.
# Run order and required inputs: docs/workflow.md.
# Run from repository root: python -m preprocessing.extract_maxdbz_profiles


from publication_config import resource_path
import pandas as pd
import numpy as np
from scipy.io import readsav
import pyhdf.SD as H
from tqdm import tqdm  # 导入 tqdm 库

# 读取 HDF 文件中指定变量的数据
def read_sds(var, path):
    out = {}
    for v in tqdm(var, desc="Reading HDF files"): 
        file = path + v.upper() + '.HDF'
        tqdm.write(f"  → Reading {v.upper()}.HDF")  #进度条
        a = H.SD(file)
        dic = a.datasets()
        keys = dic.keys()
        for n in keys:
            sd = a.select(n)
            d = sd.get()
            out[n] = d
    return out

# 读取 .sav 文件数据
def read_sav_file(file_path, var_name):
    tqdm.write(f"Reading {file_path}") #进度条
    sav_data = readsav(file_path)
    tqdm.write(f"Finished reading {var_name} from .sav file") #进度条
    return sav_data

# 将数据写入 CSV 文件
def output_csv(file_name, data, sav_variables):
    """
    保存多个变量（如 MAXDBZ 和 T）以及 HDF 数据到 CSV
    :param file_name: 输出文件路径
    :param data: HDF 变量数据 dict
    :param sav_variables: 一个 dict，例如 { 'MAXDBZ': array, 'T': array }
    """
    tqdm.write("Preparing HDF data for DataFrame...")  # 进度条

    # 基本信息 DataFrame
    data_df = pd.DataFrame({
        "LAT": data["LAT"],
        "LON": data["LON"],
        "YEAR": data["YEAR"],
        "MONTH": data["MONTH"],
        "DAY": data["DAY"],
        "HOUR": data["HOUR"],
        "LANDOCEAN": data["LANDOCEAN"],
        "ELEV": data["ELEV"]
    })

    tqdm.write("Converting .sav data to DataFrame columns...")  # 进度条

    # 初始化存放所有 .sav 数据的 DataFrame
    combined_sav_df = pd.DataFrame()

    # 遍历所有 sav 变量
    for var_name, var_array in sav_variables.items():
        for i in tqdm(range(var_array.shape[1]), desc=f"Adding {var_name} layers"):
            combined_sav_df[f"{var_name.lower()}{i+1}"] = var_array[:, i]

    # 合并最终结果
    final_df = pd.concat([data_df, combined_sav_df], axis=1)

    # 保存
    tqdm.write(f"Saving CSV file to {file_name}...")  # 进度条
    final_df.to_csv(file_name, index=False)
    tqdm.write("CSV file saved successfully!")  # 进度条

# 读取数据并调用输出函数
data_dir = resource_path('observations', 'data/pfdata/')
var = ["LAT", "LON", "YEAR", "MONTH", "DAY", "HOUR", "LANDOCEAN", "ELEV"]

# 读取 HDF 数据
data = read_sds(var, data_dir)

# 读取 .sav 文件
maxdbz_data = read_sav_file(data_dir + "MAXDBZ.sav", "MAXDBZ")['out']['MAXDBZ'][0]
t_data = read_sav_file(data_dir + "T.sav", "T")['out']['T'][0]
hgt_data = read_sav_file(data_dir + "HGT.sav", "HGT")['out']['HGT'][0]

# 汇总 .sav 数据为一个字典传入
sav_variables = {
    "MAXDBZ": maxdbz_data,
    "T": t_data,
    "HGT": hgt_data
}

# 输出 CSV
output_csv(resource_path('pf_processed', 'maxdbz/output_data.csv'), data, sav_variables)