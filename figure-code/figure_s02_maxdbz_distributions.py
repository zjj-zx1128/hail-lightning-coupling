# Purpose: figure s02 maxdbz distributions.
# Source: maxdbz.ipynb; see docs/source_coverage.csv for cell provenance.
# Run order and required inputs: docs/workflow.md.
# Run from repository root: python -m figures.figure_s02_maxdbz_distributions

# -*- coding: utf-8 -*-
"""
Batch plot: MaxDBZ–Temperature 2D frequency + percentile curves
Only plot FL/HL series (1–4), with panel labels:
  FL1–FL4 -> (a)–(d)
  HL1–HL4 -> (e)–(h)

Also:
- robustly skip non-profile CSVs
- robustly handle ELEV column name/format
- legend: longer handles + no frame
"""

from publication_config import resource_path

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- 读取自定义 colormap ----------
import xni.readhdf as readhdf
import xni.lib_ni as lib_ni

plt.rcParams["font.family"] = "Arial"

colors = readhdf.data(resource_path('colormaps', 'ez_color.hdf'))
a = np.array([colors["R"][0:55] / 255., colors["G"][0:55] / 255., colors["B"][0:55] / 255.]).T
a = np.delete(a, np.arange(1, 15, 3), 0)
a = np.delete(a, np.arange(1, 10, 2), 0)
a = np.delete(a, np.arange(10, 25, 3), 0)
cmap = lib_ni.array2cmap(a)

# === 路径 ===
input_folder = resource_path('pf_processed', 'maxdbz/filtered_regions_interpolated/')
output_folder = resource_path('figures', 'figure_s02/')
os.makedirs(output_folder, exist_ok=True)

# === bins ===
x_bins = np.linspace(10, 70, 40)   # dBZ
y_bins = np.linspace(-90, 50, 40)  # °C

# === 只选 FL/HL(1-4) ===
# 支持：FL1.csv / HL_2.csv / xxx-FL3-xxx.csv 等
pattern = re.compile(r"(?:^|[^A-Za-z0-9])(?P<grp>FL|HL)\s*[-_]?(\s*)?(?P<idx>[1-4])(?:[^A-Za-z0-9]|$)", re.I)

picked = []
for f in os.listdir(input_folder):
    if not f.lower().endswith(".csv"):
        continue
    stem = os.path.splitext(f)[0]
    m = pattern.search(stem)
    if m:
        grp = m.group("grp").upper()
        idx = int(m.group("idx"))
        picked.append((grp, idx, f))

grp_order = {"FL": 0, "HL": 1}
picked.sort(key=lambda x: (grp_order[x[0]], x[1]))
csv_files = [f for _, _, f in picked]

print("[INFO] Files to plot:", csv_files)

# === panel mapping ===
panel_map = {
    ("FL", 1): "(a)", ("FL", 2): "(b)", ("FL", 3): "(c)", ("FL", 4): "(d)",
    ("HL", 1): "(e)", ("HL", 2): "(f)", ("HL", 3): "(g)", ("HL", 4): "(h)",
}

# === profile columns ===
dbz_columns = [f"maxdbz{i+1}" for i in range(40)]
temp_columns = [f"t_ele{i+1}" for i in range(40)]

# === loop ===
for file_name in csv_files:
    file_path = os.path.join(input_folder, file_name)

    # --- read ---
    df = pd.read_csv(file_path, encoding="gbk")

    # --- clean columns (strip/BOM) ---
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    # --- ensure this is a "profile" CSV ---
    if not (set(dbz_columns).issubset(df.columns) and set(temp_columns).issubset(df.columns)):
        print(f"[SKIP] {file_name}: not a complete profile CSV.")
        continue

    # --- elevation filter if exists ---
    colmap = {c.lower(): c for c in df.columns}
    elev_col = colmap.get("elev", None)
    if elev_col is not None:
        df[elev_col] = pd.to_numeric(df[elev_col], errors="coerce")
        df = df[df[elev_col] <= 2000]
    else:
        print(f"[WARN] {file_name}: no ELEV column -> skip elevation filter.")

    # --- grab arrays ---
    dbz_data = df[dbz_columns].to_numpy(dtype=float)
    temp_data = df[temp_columns].to_numpy(dtype=float) - 273.15  # K -> °C

    # === flatten all valid (dbz>=0) for histogram ===
    dbz_vals, temp_vals = [], []
    for col in range(40):
        column_dbz = dbz_data[:, col]
        column_temp = temp_data[:, col]
        valid_mask = np.isfinite(column_dbz) & np.isfinite(column_temp) & (column_dbz >= 0)
        if np.any(valid_mask):
            dbz_vals.extend(column_dbz[valid_mask])
            temp_vals.extend(column_temp[valid_mask])

    dbz_vals = np.asarray(dbz_vals, dtype=float)
    temp_vals = np.asarray(temp_vals, dtype=float)

    if dbz_vals.size == 0:
        print(f"[SKIP] {file_name}: no valid dbz>=0 samples.")
        continue

    # === 2D histogram ===
    H, xedges, yedges = np.histogram2d(dbz_vals, temp_vals, bins=[x_bins, y_bins])
    H = H.T  # (temp, dbz)

    # === percentile lines along vertical levels (40 columns) ===
    q25_list, q50_list, q75_list = [], [], []
    q90_list, q95_list, q99_list = [], [], []
    temp_line = []

    for col in range(40):
        column_dbz = dbz_data[:, col]
        column_temp = temp_data[:, col]
        valid_mask = np.isfinite(column_dbz) & np.isfinite(column_temp) & (column_dbz >= 0)

        valid_dbz = column_dbz[valid_mask]
        valid_temp = column_temp[valid_mask]

        if valid_dbz.size > 0:
            q25_list.append(np.percentile(valid_dbz, 25))
            q50_list.append(np.percentile(valid_dbz, 50))
            q75_list.append(np.percentile(valid_dbz, 75))
            q90_list.append(np.percentile(valid_dbz, 90))
            q95_list.append(np.percentile(valid_dbz, 95))
            q99_list.append(np.percentile(valid_dbz, 99))
            temp_line.append(np.nanmean(valid_temp))
        else:
            # keep alignment
            q25_list.append(np.nan)
            q50_list.append(np.nan)
            q75_list.append(np.nan)
            q90_list.append(np.nan)
            q95_list.append(np.nan)
            q99_list.append(np.nan)
            temp_line.append(np.nan)

    q25_list = np.asarray(q25_list, dtype=float)
    q50_list = np.asarray(q50_list, dtype=float)
    q75_list = np.asarray(q75_list, dtype=float)
    q90_list = np.asarray(q90_list, dtype=float)
    q95_list = np.asarray(q95_list, dtype=float)
    q99_list = np.asarray(q99_list, dtype=float)
    temp_line = np.asarray(temp_line, dtype=float)

    # === infer FL/HL index for panel label ===
    stem = os.path.splitext(file_name)[0]
    m = pattern.search(stem)
    grp = m.group("grp").upper()
    idx = int(m.group("idx"))
    panel = panel_map.get((grp, idx), "")

    # title core (customize if needed)
    title_core = f"{grp}{idx}"

    # === plot ===
    fig, ax = plt.subplots(figsize=(6, 8))

    # main filled contours
    cf = ax.contourf(xedges[:-1], yedges[:-1], H, levels=50, cmap=cmap)

    # ---- aligned colorbar with axis (right side, top/bottom aligned) ----
    fig.canvas.draw()
    pos = ax.get_position()
    pad = 0.012
    cbar_w = 0.022
    cax = fig.add_axes([pos.x1 + pad, pos.y0, cbar_w, pos.height])
    cbar = fig.colorbar(cf, cax=cax, orientation="vertical")
    cbar.set_label("Frequency", fontsize=20)
    cbar.ax.tick_params(labelsize=20)

    # percentile curves
    ax.plot(q25_list, temp_line, linestyle="--", color="black", linewidth=2, label="25th")
    ax.plot(q50_list, temp_line, linestyle="-",  color="black", linewidth=3, label="50th")
    ax.plot(q75_list, temp_line, linestyle="--", color="black", linewidth=2, label="75th")
    ax.plot(q90_list, temp_line, linestyle="--", color="red",   linewidth=2, label="90th")
    ax.plot(q95_list, temp_line, linestyle="-",  color="red",   linewidth=3, label="95th")
    ax.plot(q99_list, temp_line, linestyle="--", color="red",   linewidth=2, label="99th")

    # labels/ticks/title
    ax.set_xlabel("Maximum Reflectivity (dBZ)", fontsize=20)
    ax.set_ylabel("Temperature (°C)", fontsize=20)
    ax.set_title(f"{panel} {title_core}", fontsize=25, fontweight="bold", pad=10)
    ax.tick_params(axis="both", which="major", labelsize=20)

    ax.grid(True, linestyle="--", alpha=0.5)

    ax.set_xlim(10, 70)
    ax.set_ylim(-80, 30)
    
    ax.invert_yaxis()

    # legend: bigger icon+font, longer handles, no frame
    leg = ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=18,
        handlelength=3.0,
        handletextpad=0.6,
        labelspacing=0.5,
        borderaxespad=0.6
    )
    # optional: make legend lines thicker for visibility
    for line in leg.get_lines():
        line.set_linewidth(2.4)

    # save
    out_name = f"{stem}_maxdbz_temp.jpg"
    output_path = os.path.join(output_folder, out_name)

    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {output_path}")
