# Purpose: canesm5 04 select land.
# Source: cmip6_figure_CanESM5_XGB_plot_latest_checked.ipynb; see docs/source_coverage.csv for cell provenance.
# Run order and required inputs: docs/workflow.md.
# Run from repository root: python -m modeling.canesm5_04_select_land

# -*- coding: utf-8 -*-
"""
Strict land filter:
Land_Sea==0  AND  point-in-land-polygon (Natural Earth)

Requires: geopandas, shapely, cartopy (only for dataset path)
"""

from __future__ import annotations

from publication_config import resource_path
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import cartopy.io.shapereader as shpreader

PARQ_IN  = resource_path('models', 'CanESM5/ml_feature_withSRTM/train_table_flash_hail_withSRTM_clean.parquet')
PARQ_OUT = resource_path('models', 'CanESM5/ml_feature_withSRTM/train_table_flash_hail_withSRTM_strictLand.parquet')

LAT, LON, LS = "Latitude", "Longitude", "Land_Sea"
LAT_MAX = 63.0
LAND_VALUE = 0  # 你当前假设 0=land

def main():
    df = pd.read_parquet(PARQ_IN)

    # 先做你已有的硬条件：|lat|<=63 + Land_Sea==0
    df0 = df[(df[LAT].abs() <= LAT_MAX) & (df[LS] == LAND_VALUE)].copy()
    df0 = df0.dropna(subset=[LAT, LON]).reset_index(drop=True)
    print(f"[STEP1] lat & Land_Sea filter: {len(df)} -> {len(df0)}")

    # 读取 Natural Earth 陆地多边形（110m 足够；想更严格可用 50m）
    shp = shpreader.natural_earth(resolution="110m", category="physical", name="land")
    land = gpd.read_file(shp)
    land = land.to_crs("EPSG:4326")

    # 合并成一个大几何体，加速 contains 判断
    land_union = land.geometry.union_all()

    # 对点做 contains 判断（点必须在陆地多边形内部）
    # 注意：contains 对“刚好在边界上”的点会返回 False；
    # 如果你担心边界点误杀，可改用 intersects
    pts = [Point(xy) for xy in zip(df0[LON].values, df0[LAT].values)]
    in_land = np.array([land_union.contains(p) for p in pts], dtype=bool)

    df_strict = df0.loc[in_land].reset_index(drop=True)

    print(f"[STEP2] strict polygon land: {len(df0)} -> {len(df_strict)}")
    print(f"[REMOVED] ocean-misclassified points removed: {len(df0) - len(df_strict)}")

    df_strict.to_parquet(PARQ_OUT, index=False)
    print("[SAVED]", PARQ_OUT)

if __name__ == "__main__":
    main()
