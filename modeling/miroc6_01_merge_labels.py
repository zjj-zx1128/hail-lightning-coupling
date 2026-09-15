# Run from repository root: python -m modeling.miroc6_01_merge_labels


from publication_config import resource_path
# ============================================================
# Step 1. Build MIROC6 standard historical strict-land training table
#
# 目的：
#   将 MIROC6 的 historical feature 表与 labels_land_strict 表合并，
#   生成后续训练 XGBoost 所需的标准训练表。
#
# 注意：
#   1. 本代码不训练模型；
#   2. 本代码不剔除 elev > 2000 m；
#   3. 本代码不剔除环境变量缺失值；
#   4. 本代码不对目标变量做 finite 筛选；
#   5. 目标变量 finite 筛选应放在后续模型训练阶段按目标变量分别进行。
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# 0. Path configuration
# ============================================================

MODEL_NAME = "MIROC6"

BASE_DIR = Path(resource_path('models', 'MIROC6'))
TRAIN_DIR = BASE_DIR / "feature" / "ml_train_historical"

FEATURE_FILE = TRAIN_DIR / "features_historical_1995_2014.parquet"
LABEL_FILE = TRAIN_DIR / "labels_land_strict.parquet"

OUT_FILE = TRAIN_DIR / "train_table_flash_hail_withSRTM_strictLand.parquet"
OUT_CHECK_CSV = TRAIN_DIR / "train_table_flash_hail_withSRTM_strictLand_check.csv"

LAT_COL = "Latitude"
LON_COL = "Longitude"

TARGET_FLASH = "LISOTD_Flash"
TARGET_HAIL = "ni_HailPF"
TARGET_COLS = [TARGET_FLASH, TARGET_HAIL]

LAND_COL = "Land_Sea"
ELEV_COL = "elev"

# 是否做纬度保险筛选。CanESM5 流程中有限制 |lat| <= 63。
# MIROC6 labels_land_strict 本身已经是 -62 到 62，但保留这个安全检查。
APPLY_LAT_FILTER = True
LAT_LIMIT = 63.0


# ============================================================
# 1. Read files
# ============================================================

print("=" * 100)
print(f"[STEP 1] Build standard strict-land training table for {MODEL_NAME}")
print("=" * 100)

print(f"\n[READ] Feature file:")
print(FEATURE_FILE)
df_feat = pd.read_parquet(FEATURE_FILE)

print(f"\n[READ] Label file:")
print(LABEL_FILE)
df_lab = pd.read_parquet(LABEL_FILE)

print("\n[FEATURE TABLE]")
print("Rows   :", len(df_feat))
print("Columns:", len(df_feat.columns))
print("First 20 columns:")
print(list(df_feat.columns[:20]))

print("\n[LABEL TABLE]")
print("Rows   :", len(df_lab))
print("Columns:", len(df_lab.columns))
print("All columns:")
print(list(df_lab.columns))


# ============================================================
# 2. Basic checks
# ============================================================

required_feat_cols = [LAT_COL, LON_COL]
required_label_cols = [LAT_COL, LON_COL, TARGET_FLASH, TARGET_HAIL]

if ELEV_COL not in df_feat.columns:
    print(f"\n[WARNING] Elevation column '{ELEV_COL}' was not found in feature table.")
else:
    print(f"\n[OK] Elevation column found: {ELEV_COL}")

missing_feat = [c for c in required_feat_cols if c not in df_feat.columns]
missing_lab = [c for c in required_label_cols if c not in df_lab.columns]

if missing_feat:
    raise ValueError(f"[ERROR] Missing columns in feature table: {missing_feat}")

if missing_lab:
    raise ValueError(f"[ERROR] Missing columns in label table: {missing_lab}")

feat_dup = df_feat.duplicated(subset=[LAT_COL, LON_COL]).sum()
lab_dup = df_lab.duplicated(subset=[LAT_COL, LON_COL]).sum()

print("\n[DUPLICATE CHECK]")
print("Feature duplicated coordinates:", feat_dup)
print("Label duplicated coordinates  :", lab_dup)

if feat_dup > 0:
    raise ValueError(f"[ERROR] Feature table has duplicated coordinates: {feat_dup}")

if lab_dup > 0:
    raise ValueError(f"[ERROR] Label table has duplicated coordinates: {lab_dup}")


# ============================================================
# 3. Merge feature and strict-land label table
#
# 关键：
#   使用 inner merge，只保留 labels_land_strict 中存在的严格陆地格点。
#   不额外剔除 elev > 2000 m。
#   不额外 dropna。
# ============================================================

label_keep_cols = [LAT_COL, LON_COL, TARGET_FLASH, TARGET_HAIL]

if LAND_COL in df_lab.columns:
    label_keep_cols.append(LAND_COL)

df = df_feat.merge(
    df_lab[label_keep_cols],
    on=[LAT_COL, LON_COL],
    how="inner"
)

print("\n[MERGE]")
print("Rows before merge, feature table:", len(df_feat))
print("Rows before merge, label table  :", len(df_lab))
print("Rows after inner merge          :", len(df))
print("Columns after merge             :", len(df.columns))

print("\n[TARGET MISSING CHECK AFTER MERGE]")
print(f"{TARGET_FLASH} missing:", df[TARGET_FLASH].isna().sum())
print(f"{TARGET_HAIL} missing :", df[TARGET_HAIL].isna().sum())


# ============================================================
# 4. Optional latitude safety filter
#
# 不做高程剔除。
# 不做环境变量缺失值剔除。
# ============================================================

n_before_lat = len(df)

if APPLY_LAT_FILTER:
    df = df.loc[df[LAT_COL].abs() <= LAT_LIMIT].copy()

n_after_lat = len(df)

print("\n[LATITUDE SAFETY FILTER]")
print(f"Apply |{LAT_COL}| <= {LAT_LIMIT}: {APPLY_LAT_FILTER}")
print("Rows before latitude filter:", n_before_lat)
print("Rows after latitude filter :", n_after_lat)
print("Removed rows               :", n_before_lat - n_after_lat)


# ============================================================
# 5. Sort and reset index
# ============================================================

df = df.sort_values([LAT_COL, LON_COL]).reset_index(drop=True)


# ============================================================
# 6. Diagnostics
# ============================================================

print("\n[FINAL TABLE]")
print("Rows   :", len(df))
print("Columns:", len(df.columns))
print("Latitude range :", df[LAT_COL].min(), "to", df[LAT_COL].max())
print("Longitude range:", df[LON_COL].min(), "to", df[LON_COL].max())

if ELEV_COL in df.columns:
    print("Elevation range:", df[ELEV_COL].min(), "to", df[ELEV_COL].max())
    print("Number of grids with elev > 2000 m:", int((df[ELEV_COL] > 2000).sum()))
else:
    print("Elevation column not found in final table.")

print("\n[TARGET SUMMARY]")
print(df[TARGET_COLS].describe())

print("\n[TARGET FINITE CHECK]")
for col in TARGET_COLS:
    arr = pd.to_numeric(df[col], errors="coerce").to_numpy()
    print(f"{col}: finite = {np.isfinite(arr).sum()}, non-finite = {(~np.isfinite(arr)).sum()}")

if LAND_COL in df.columns:
    print("\n[LAND_SEA VALUE COUNTS]")
    print(df[LAND_COL].value_counts(dropna=False))

print("\n[MISSING VALUES SUMMARY]")
print("Total missing values in final table:", int(df.isna().sum().sum()))
print("Columns with missing values:")
missing_by_col = df.isna().sum()
missing_by_col = missing_by_col[missing_by_col > 0].sort_values(ascending=False)
if len(missing_by_col) > 0:
    print(missing_by_col.head(30))
else:
    print("No missing values found.")


# ============================================================
# 7. Save outputs
# ============================================================

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

df.to_parquet(OUT_FILE, index=False)

summary_items = {
    "model": MODEL_NAME,
    "feature_file": str(FEATURE_FILE),
    "label_file": str(LABEL_FILE),
    "output_file": str(OUT_FILE),
    "feature_rows": len(df_feat),
    "label_rows": len(df_lab),
    "merged_rows_before_lat_filter": n_before_lat,
    "final_rows": len(df),
    "final_columns": len(df.columns),
    "lat_min": df[LAT_COL].min(),
    "lat_max": df[LAT_COL].max(),
    "lon_min": df[LON_COL].min(),
    "lon_max": df[LON_COL].max(),
    "flash_missing": int(df[TARGET_FLASH].isna().sum()),
    "hail_missing": int(df[TARGET_HAIL].isna().sum()),
    "flash_finite": int(np.isfinite(pd.to_numeric(df[TARGET_FLASH], errors="coerce")).sum()),
    "hail_finite": int(np.isfinite(pd.to_numeric(df[TARGET_HAIL], errors="coerce")).sum()),
}

if ELEV_COL in df.columns:
    summary_items.update({
        "elev_min": df[ELEV_COL].min(),
        "elev_max": df[ELEV_COL].max(),
        "n_elev_gt_2000": int((df[ELEV_COL] > 2000).sum()),
    })

summary_df = pd.DataFrame({
    "item": list(summary_items.keys()),
    "value": list(summary_items.values())
})

summary_df.to_csv(OUT_CHECK_CSV, index=False, encoding="utf-8-sig")

print("\n[SAVED]")
print("Training table:", OUT_FILE)
print("Check summary :", OUT_CHECK_CSV)

print("\n[DONE] Step 1 finished successfully.")
