# Purpose: canesm5 03 clean features.
# Source: cmip6_figure_CanESM5_XGB_plot_latest_checked.ipynb; see docs/source_coverage.csv for cell provenance.
# Run order and required inputs: docs/workflow.md.
# Run from repository root: python -m modeling.canesm5_03_clean_features

# -*- coding: utf-8 -*-
"""
Load train_table_flash_hail_withSRTM.parquet
1) Print ALL columns (before)
2) Drop unwanted features
3) Print columns AFTER drop
4) (Optional) Save cleaned parquet
"""

from publication_config import resource_path

import pandas as pd
import os

# ======================================================
# 0) PATH
# ======================================================
PARQ_PATH = resource_path('models', 'CanESM5/ml_feature_withSRTM/train_table_flash_hail_withSRTM.parquet')

if not os.path.exists(PARQ_PATH):
    raise FileNotFoundError(f"File not found: {PARQ_PATH}")

# ======================================================
# 1) LOAD
# ======================================================
print("[LOAD]", PARQ_PATH)
df = pd.read_parquet(PARQ_PATH)

# ======================================================
# 2) PRINT ALL COLUMNS (BEFORE)
# ======================================================
print("\n================= ALL COLUMNS (BEFORE) =================")
for i, c in enumerate(df.columns, 1):
    print(f"{i:03d}: {c}")
print(f"\n[TOTAL COLUMNS] {len(df.columns)}")

# ======================================================
# 3) FEATURES YOU WANT TO REMOVE
# ======================================================
DROP_COLS = [
    "ni_pct_f",
    "le_radar_f",
    "mroz_radar_f",
    "ni_radar_f",
    "Le_Hail",
    "bang_pct_f",
    "ferraro_pct_n","quantile","id"
]

# check existence
exist = [c for c in DROP_COLS if c in df.columns]
missing = [c for c in DROP_COLS if c not in df.columns]

print("\n[CHECK] Columns to drop:")
print("  exist  :", exist)
print("  missing:", missing)

# ======================================================
# 4) DROP
# ======================================================
df_clean = df.drop(columns=exist)

# ======================================================
# 5) PRINT ALL COLUMNS (AFTER)
# ======================================================
print("\n================= ALL COLUMNS (AFTER DROP) =================")
for i, c in enumerate(df_clean.columns, 1):
    print(f"{i:03d}: {c}")
print(f"\n[TOTAL COLUMNS AFTER DROP] {len(df_clean.columns)}")

# ======================================================
# 6) OPTIONAL: SAVE CLEAN VERSION
# ======================================================
OUT_PATH = resource_path('models', 'CanESM5/ml_feature_withSRTM/train_table_flash_hail_withSRTM_clean.parquet')
df_clean.to_parquet(OUT_PATH)
print("\n[SAVED CLEAN PARQUET]")
print(" ", OUT_PATH)
