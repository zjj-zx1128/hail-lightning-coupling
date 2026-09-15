# Purpose: canesm5 02 merge labels.
# Source: cmip6_figure_CanESM5_XGB_plot_latest_checked.ipynb; see docs/source_coverage.csv for cell provenance.
# Run order and required inputs: docs/workflow.md.
# Run from repository root: python -m modeling.canesm5_02_merge_labels

# -*- coding: utf-8 -*-
"""
Build training table by merging:
- Labels (flashhail_matched_new2.xlsx)
- Historical features with SRTM elevation

Output:
- train_table_flash_hail_withSRTM.parquet
"""

from __future__ import annotations

from publication_config import resource_path
import os
import pandas as pd

# =========================
# PATHS
# =========================
LABEL_XLSX = resource_path('observations', 'flashhail_matched_new2.xlsx')

FEAT_PARQ = (
    resource_path('models', 'CanESM5/ml_feature_withSRTM/features_historical_1995_2014_withSRTM.parquet')
)

OUT_DIR = resource_path('models', 'CanESM5/ml_feature_withSRTM')
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PARQ = os.path.join(
    OUT_DIR, "train_table_flash_hail_withSRTM.parquet"
)

# =========================
# COLUMN NAMES
# =========================
LAT_COL = "Latitude"
LON_COL = "Longitude"
Y_FLASH = "LISOTD_Flash"
Y_HAIL  = "ni_HailPF"
LS_COL  = "Land_Sea"

# =========================
# MAIN
# =========================
def main():
    print("[LOAD] labels (xlsx)")
    df_lab = pd.read_excel(LABEL_XLSX)

    print("[LOAD] historical features with SRTM")
    df_feat = pd.read_parquet(FEAT_PARQ)

    # ---- basic checks ----
    for c in [LAT_COL, LON_COL, Y_FLASH, Y_HAIL]:
        if c not in df_lab.columns:
            raise KeyError(f"[LABEL] missing column: {c}")

    for c in [LAT_COL, LON_COL]:
        if c not in df_feat.columns:
            raise KeyError(f"[FEAT] missing column: {c}")

    print("[INFO] labels rows :", len(df_lab))
    print("[INFO] features rows:", len(df_feat))

    # =========================
    # MERGE (label LEFT JOIN feature)
    # =========================
    print("[MERGE] labels ⨝ features (on lat/lon)")
    df = (
        df_lab
        .merge(
            df_feat,
            on=[LAT_COL, LON_COL],
            how="left",
            validate="one_to_one"
        )
    )

    # =========================
    # SANITY CHECK
    # =========================
    n_all = len(df)
    n_feat_nan = df.isna().all(axis=1).sum()

    print(f"[CHECK] merged rows: {n_all}")
    print(f"[CHECK] rows with all-feature-NaN: {n_feat_nan}")

    if n_feat_nan > 0:
        print("[WARN] some points have no matched features (will be kept, XGB handles NaN)")

    # =========================
    # SAVE
    # =========================
    df.to_parquet(OUT_PARQ, index=False)
    print("[SAVED]", OUT_PARQ)


if __name__ == "__main__":
    main()
