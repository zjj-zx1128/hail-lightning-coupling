# Purpose: select maxdbz lightning regions.
# Source: maxdbz.ipynb; see docs/source_coverage.csv for cell provenance.
# Run order and required inputs: docs/workflow.md.
# Run from repository root: python -m preprocessing.select_maxdbz_lightning_regions


from publication_config import resource_path
import pandas as pd
from tqdm import tqdm
import os

# 定义地理区域
regions = {
    "FL1": [24, 35, -98, -78],
    "FL2": [-10, 5, -76, -58],
    "FL3": [-8, 5, 12, 35],
    "FL4": [-12, 8, 95, 150]
}

# 输入和输出路径
input_csv = resource_path('pf_processed', 'maxdbz/output_data.csv')
output_folder = resource_path('pf_processed', 'maxdbz/filtered_regions/')
os.makedirs(output_folder, exist_ok=True)

# 读取文件的列名并为每个区域准备一个空 CSV 文件（只写入列名）
header = pd.read_csv(input_csv, nrows=0)
for region in regions:
    header.to_csv(f"{output_folder}{region}.csv", index=False)

# 读取文件的总行数
total_lines = sum(1 for _ in open(input_csv)) - 1  # 减去表头
total_chunks = total_lines // 100_000 + 1  # 总块数

# 分块读取并逐块筛选写入
chunksize = 100_000  # 每次读取10万行，可根据内存调整

with tqdm(pd.read_csv(input_csv, chunksize=chunksize), total=total_chunks, desc="Filtering by chunks") as pbar:
    for chunk in pbar:
        for region_name, (lat_min, lat_max, lon_min, lon_max) in regions.items():
            # 筛选区域数据
            region_chunk = chunk[
                (chunk["LAT"] >= lat_min) & (chunk["LAT"] <= lat_max) &
                (chunk["LON"] >= lon_min) & (chunk["LON"] <= lon_max) &
                (chunk["LANDOCEAN"] == 1)
            ]
            # 若筛选结果非空，则追加到相应的 CSV 文件中
            if not region_chunk.empty:
                region_chunk.to_csv(f"{output_folder}{region_name}.csv", mode='a', index=False, header=False)