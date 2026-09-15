# Purpose: figure s01c hail method correlations.
# Source: cmip6_figure_difference_flash_hail.ipynb; see docs/source_coverage.csv for cell provenance.
# Run order and required inputs: docs/workflow.md.
# Run from repository root: python -m figures.figure_s01c_hail_method_correlations

# -*- coding: utf-8 -*-
"""
Correlation analysis among hail identification methods (Spearman only)
- ROI: |lat| <= 63
- Land only: Land_Sea == 0
- Exclude le_radar_f and ni_radar_f from statistics
- Rename methods for publication-ready labels
- Heatmap: Spearman r (RdBu_r, [-1,1]) with in-cell values
- Right side: row-wise Mean R^2 (exclude diagonal), annotated
- Colorbar aligned with main axes (top/bottom)
"""

from publication_config import resource_path

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Arial"

# =========================
# 0) Input
# =========================
XLSX = resource_path('observations', 'flashhail_matched_new2.xlsx')
SHEET = 0
OUTDIR = resource_path('figures', '')
os.makedirs(OUTDIR, exist_ok=True)

# =========================
# 1) Read
# =========================
df = pd.read_excel(XLSX, sheet_name=SHEET)

def find_col(patterns, columns):
    for p in patterns:
        for c in columns:
            if re.search(p, str(c), flags=re.I):
                return c
    return None

lat_col = find_col([r"^lat$", r"^latitude$", r"lat"], df.columns)
lon_col = find_col([r"^lon$", r"^longitude$", r"lon"], df.columns)
if lat_col is None or lon_col is None:
    raise ValueError(f"未找到经纬度列。当前列名：{list(df.columns)}")

# =========================
# 2) ROI filtering: |lat| <= 63, land only
# =========================
LAT_MIN, LAT_MAX = -63, 63
df = df[(df[lat_col] >= LAT_MIN) & (df[lat_col] <= LAT_MAX)]

if "Land_Sea" in df.columns:
    df = df[df["Land_Sea"] == 0]
else:
    raise ValueError("表中未找到 Land_Sea 列，无法进行陆地筛选。")

print("[INFO] Samples after region filtering:", len(df))

# =========================
# 3) Detect hail/method columns
#    rule: contains 'hail' OR ends with _f/_n
# =========================
exclude = {lat_col, lon_col}
for c in df.columns:
    lc = str(c).lower()
    if lc in ("id", "land_sea") or "flash" in lc:
        exclude.add(c)

hail_cols = []
for c in df.columns:
    if c in exclude:
        continue
    lc = str(c).lower()
    if re.search(r"hail", lc) or re.search(r"(_f|_n)$", lc):
        hail_cols.append(c)

if len(hail_cols) < 2:
    raise ValueError(f"识别到的冰雹相关列过少：{hail_cols}\n请检查列名或手动指定 hail_cols。")

print("[INFO] Hail/method cols (raw):", hail_cols)

# =========================
# 4) Exclude specific methods from statistics
# =========================
EXCLUDE_METHODS = {"le_radar_f", "ni_radar_f"}
hail_cols = [c for c in hail_cols if c not in EXCLUDE_METHODS]
print("[INFO] Hail/method cols (after exclude):", hail_cols)

# =========================
# 5) Build numeric matrix
# =========================
X = df[hail_cols].apply(pd.to_numeric, errors="coerce")

# drop all-NaN columns (safety)
X = X.dropna(axis=1, how="all")

# =========================
# 6) Rename methods (publication-ready)
# =========================
METHOD_RENAME = {
    "Le_Hail":         "Le et al. (Radar)",
    "ni_pct_f":        "Ni et al. (MW)",
    "ni_HailPF":       "Ni et al. (Radar)",
    "mroz_radar_f":    "Mroz et al. (Radar)",
    "bang_pct_f":      "Bang et al. (MW)",
    "ferraro_pct_n":   "Ferraro et al. (MW)",
}
X = X.rename(columns=lambda c: METHOD_RENAME.get(c, c))
cols_final = list(X.columns)

print("[INFO] Columns used (renamed):", cols_final)

# =========================
# 7) Spearman correlation matrix
# =========================
MIN_PERIODS = 50
corr_spearman = X.corr(method="spearman", min_periods=MIN_PERIODS)

# =========================
# 8) Plot function: heatmap + right Mean R^2 + aligned colorbar
# =========================
def plot_corr_heatmap_with_mean_r2(
    C: pd.DataFrame,
    title: str,
    out_png: str,
    cmap_name: str = "RdBu_r",
    show_cell_values: bool = True,
    cell_fontsize: int = 18,
    right_fontsize: int = 18,
    mean_r2_decimals: int = 4,
):
    cols = list(C.columns)
    M = C.values.astype(float)
    n = len(cols)

    # ---- mean R^2 per row (exclude diagonal) ----
    R2 = M ** 2
    R2_no_diag = R2.copy()
    np.fill_diagonal(R2_no_diag, np.nan)
    mean_r2 = np.nanmean(R2_no_diag, axis=1)

    # ---- figure: slim & tall (you can tweak) ----
    fig = plt.figure(figsize=(10,12))
    ax = plt.gca()

    # main heatmap (r)
    im = ax.imshow(
        M,
        vmin=-1, vmax=1,
        cmap=cmap_name,
        interpolation="nearest",
        aspect="auto"   # allow non-square axes
    )

    # ticks/labels/title
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=18)
    ax.set_yticklabels(cols, fontsize=18)
    ax.set_title(title, fontsize=27, pad=10,fontweight="bold")

    # in-cell values (r)
    if show_cell_values:
        for i in range(n):
            for j in range(n):
                v = M[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=cell_fontsize)

    # ---- reserve room on right INSIDE axes coordinates for text ----
    # keep the matrix region [0..n-1], and add extra logical space to the right
    extra = 1.3
    ax.set_xlim(-0.5, n - 0.5 + extra)
    x_text = n - 0.5 + 0.15  # start a bit to the right of last column

    fmt = f"{{:.{mean_r2_decimals}f}}"
    for i in range(n):
        ax.text(
            x_text, i,
            f"Mean R²\n\n={fmt.format(mean_r2[i])}",
            ha="left", va="center",
            fontsize=right_fontsize
        )

    # ---- Layout: DO NOT use tight_layout (it will move ax after we place cax)
    # set margins manually so xlabels fit + space for colorbar
    fig.subplots_adjust(left=0.18, right=0.86, bottom=0.22, top=0.92)

    # ---- Colorbar aligned with main axes (top/bottom) ----
    fig.canvas.draw()
    pos = ax.get_position()

    pad = 0.012
    cbar_w = 0.015
    cax = fig.add_axes([pos.x1 + pad, pos.y0, cbar_w, pos.height])

    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
    cbar.set_label("Spearman Correlation (r)", fontsize=18)
    cbar.ax.tick_params(labelsize=18)

    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.show()

# =========================
# 9) Run (Spearman only)
# =========================
out_png = os.path.join(OUTDIR, 'figure_s01c_hail_method_correlations.jpg')
plot_corr_heatmap_with_mean_r2(
    corr_spearman,
    "(c) Correlation of Hail Identification Methods",
    out_png,
    cmap_name="RdBu_r",
    show_cell_values=True,
    cell_fontsize=18,      # <-- 单元格里 r 的字体
    right_fontsize=18,     # <-- 右侧 Mean R² 的字体
    mean_r2_decimals=4     # <-- Mean R² 保留小数位
)

print("[SAVED]", out_png)
