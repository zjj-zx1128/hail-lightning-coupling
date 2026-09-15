# Purpose: select maxdbz hail regions.
# Source: maxdbz.ipynb; see docs/source_coverage.csv for cell provenance.
# Run order and required inputs: docs/workflow.md.
# Run from repository root: python -m preprocessing.select_maxdbz_hail_regions


from publication_config import resource_path
import pandas as pd
from tqdm import tqdm
import os

# 定义地理区域
regions = {
    "HL1": [35, 47, -104, -90],
    "HL2": [-42, -23, -68, -50],
    "HL3": [35, 55, -10, 20],
    "HL4": [35, 55, 112, 138]
}

# 输入和输出路径
input_csv = resource_path('pf_processed', 'maxdbz/output_data.csv')
output_folder = resource_path('pf_processed', 'maxdbz/filtered_regions/')
os.makedirs(output_folder, exist_ok=True)

# 为每个区域准备一个空 CSV（写入列名）
header = pd.read_csv(input_csv, nrows=0)
for region in regions:
    header.to_csv(f"{output_folder}{region}.csv", index=False)

# 分块读取并逐块筛选写入
chunksize = 100_000  # 每次读取10万行，可根据内存调整
total_chunks = sum(1 for _ in open(input_csv)) // chunksize

for chunk in tqdm(pd.read_csv(input_csv, chunksize=chunksize), desc="Filtering by chunks"):
    for region_name, (lat_min, lat_max, lon_min, lon_max) in regions.items():
        region_chunk = chunk[
            (chunk["LAT"] >= lat_min) & (chunk["LAT"] <= lat_max) &
            (chunk["LON"] >= lon_min) & (chunk["LON"] <= lon_max) &
            (chunk["LANDOCEAN"] == 1)
        ]
        if not region_chunk.empty:
            region_chunk.to_csv(f"{output_folder}{region_name}.csv", mode='a', index=False, header=False)