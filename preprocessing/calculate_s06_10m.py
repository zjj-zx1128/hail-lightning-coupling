# Purpose: calculate s06 10m.
# Source: hdf_S06.ipynb; see docs/source_coverage.csv for cell provenance.
# Run order and required inputs: docs/workflow.md.
# Run from repository root: python -m preprocessing.calculate_s06_10m

# -*- coding: utf-8 -*-
"""
快速计算 0–6 km AGL bulk wind shear, S06

S06 = sqrt((U_6kmAGL - U10)^2 + (V_6kmAGL - V10)^2)

数据结构：
U.sav   : U   shape = (22284695, 28)
V.sav   : V   shape = (22284695, 28)
HGT.sav : HGT shape = (22284695, 28)

U10.HDF : U10  shape = (22284695,)
V10.HDF : V10  shape = (22284695,)
ELEV.HDF: ELEV shape = (22284695,)
LAT.HDF : LAT  shape = (22284695,)
LON.HDF : LON  shape = (22284695,)

输出：
S06.npy
S06_with_latlon.npz
S06.HDF
S06_check.txt
"""

from publication_config import resource_path

import os
import time
import numpy as np
from scipy.io import readsav

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from pyhdf.SD import SD, SDC
except ImportError:
    SD = None
    SDC = None


# =========================================================
# 1. 路径设置
# =========================================================
DATA_DIR = resource_path('observations', 'data/pfdata')
OUT_DIR = resource_path('pf_processed', 's06')

U_PATH    = os.path.join(DATA_DIR, "U.sav")
V_PATH    = os.path.join(DATA_DIR, "V.sav")
HGT_PATH  = os.path.join(DATA_DIR, "HGT.sav")

U10_PATH  = os.path.join(DATA_DIR, "U10.HDF")
V10_PATH  = os.path.join(DATA_DIR, "V10.HDF")
ELEV_PATH = os.path.join(DATA_DIR, "ELEV.HDF")
LAT_PATH  = os.path.join(DATA_DIR, "LAT.HDF")
LON_PATH  = os.path.join(DATA_DIR, "LON.HDF")

OUT_NPY   = os.path.join(OUT_DIR, "S06.npy")
OUT_NPZ   = os.path.join(OUT_DIR, "S06_with_latlon.npz")
OUT_HDF   = os.path.join(OUT_DIR, "S06.HDF")
OUT_TXT   = os.path.join(OUT_DIR, "S06_check.txt")


# =========================================================
# 2. 读取 SAV 中的二维垂直廓线变量
# =========================================================
def read_sav_profile(path, field_name):
    """
    读取 .sav 文件中 out[0][field_name] 的二维数组。
    """
    data = readsav(path, python_dict=True, verbose=False)
    arr = data["out"][0][field_name]
    arr = np.asarray(arr, dtype=np.float32)
    return arr


# =========================================================
# 3. 读取 HDF4 一维变量
# =========================================================
def read_hdf4_1d(path, var_name):
    """
    读取 HDF4 文件中的一维变量。
    """
    if SD is None:
        raise ImportError(
            "当前环境未安装 pyhdf，无法读取 HDF4 文件。请先安装：pip install pyhdf"
        )

    hdf = SD(path, SDC.READ)
    data = hdf.select(var_name)[:]
    hdf.end()

    return np.asarray(data, dtype=np.float32).squeeze()


# =========================================================
# 4. 基本统计函数
# =========================================================
def stat_line(name, arr):
    arr = np.asarray(arr)
    finite = np.isfinite(arr)

    if finite.sum() == 0:
        return f"{name}: shape={arr.shape}, finite=0, nan={np.isnan(arr).sum()}, min=nan, max=nan, mean=nan"

    return (
        f"{name}: shape={arr.shape}, "
        f"finite={finite.sum()}, "
        f"nan={np.isnan(arr).sum()}, "
        f"min={np.nanmin(arr):.6g}, "
        f"max={np.nanmax(arr):.6g}, "
        f"mean={np.nanmean(arr):.6g}"
    )


# =========================================================
# 5. 快速向量化垂直插值函数
# =========================================================
def interp_to_target_fast(HGT, VAR, target_height, var_name="VAR"):
    """
    快速向量化插值。

    对每一个样本点，找到 target_height 所在的相邻高度层：
    HGT[:, k] 与 HGT[:, k+1]

    然后进行线性插值：
    VAR_target = VAR_k + w * (VAR_k+1 - VAR_k)

    该方法只循环垂直层，而不是循环 22284695 个样本点。
    """

    HGT = np.asarray(HGT, dtype=np.float32)
    VAR = np.asarray(VAR, dtype=np.float32)
    target_height = np.asarray(target_height, dtype=np.float32)

    n_sample, n_level = HGT.shape
    out = np.full(n_sample, np.nan, dtype=np.float32)

    level_iter = range(n_level - 1)

    if tqdm is not None:
        level_iter = tqdm(level_iter, desc=f"Interpolating {var_name}", ncols=100)

    total_matched = 0

    for k in level_iter:
        z0 = HGT[:, k]
        z1 = HGT[:, k + 1]

        x0 = VAR[:, k]
        x1 = VAR[:, k + 1]

        valid = (
            np.isfinite(z0) &
            np.isfinite(z1) &
            np.isfinite(x0) &
            np.isfinite(x1) &
            np.isfinite(target_height) &
            (z1 != z0)
        )

        # 兼容高度从低到高或从高到低排列
        between = (
            ((target_height >= z0) & (target_height <= z1)) |
            ((target_height >= z1) & (target_height <= z0))
        )

        # 每个样本只插值一次，已经插值成功的点不再覆盖
        mask = valid & between & np.isnan(out)

        n_match = int(mask.sum())
        total_matched += n_match

        if n_match == 0:
            continue

        weight = (target_height[mask] - z0[mask]) / (z1[mask] - z0[mask])
        out[mask] = x0[mask] + weight * (x1[mask] - x0[mask])

        if tqdm is None:
            print(
                f"{var_name}: layer {k:02d}-{k+1:02d}, "
                f"matched={n_match}, total={total_matched}"
            )

    print(f"{var_name}: interpolation finished, matched total = {total_matched}")

    return out


# =========================================================
# 6. 保存 HDF4
# =========================================================
def write_hdf4_output(out_hdf, S06, LAT, LON, ELEV):
    """
    保存 S06、LAT、LON、ELEV 到 HDF4。
    """
    if SD is None:
        raise ImportError("当前环境未安装 pyhdf，无法写出 HDF4 文件。")

    n_sample = S06.shape[0]

    hdf_out = SD(out_hdf, SDC.WRITE | SDC.CREATE)

    # S06
    sds = hdf_out.create("S06", SDC.FLOAT32, (n_sample,))
    sds[:] = S06.astype(np.float32)
    sds.attributes()["long_name"] = "0-6 km AGL bulk wind shear"
    sds.attributes()["units"] = "m s-1"
    sds.attributes()["description"] = "sqrt((U_6kmAGL-U10)^2 + (V_6kmAGL-V10)^2)"
    sds.endaccess()

    # LAT
    sds = hdf_out.create("LAT", SDC.FLOAT32, (n_sample,))
    sds[:] = LAT.astype(np.float32)
    sds.attributes()["long_name"] = "latitude"
    sds.attributes()["units"] = "degrees_north"
    sds.endaccess()

    # LON
    sds = hdf_out.create("LON", SDC.FLOAT32, (n_sample,))
    sds[:] = LON.astype(np.float32)
    sds.attributes()["long_name"] = "longitude"
    sds.attributes()["units"] = "degrees_east"
    sds.endaccess()

    # ELEV
    sds = hdf_out.create("ELEV", SDC.FLOAT32, (n_sample,))
    sds[:] = ELEV.astype(np.float32)
    sds.attributes()["long_name"] = "surface elevation"
    sds.attributes()["units"] = "m"
    sds.endaccess()

    hdf_out.end()


# =========================================================
# 7. 主程序
# =========================================================
if __name__ == "__main__":

    t0 = time.time()

    lines = []
    lines.append("=" * 100)
    lines.append("S06 calculation check: fast vectorized version")
    lines.append("=" * 100)

    # -----------------------------------------------------
    # 读取数据
    # -----------------------------------------------------
    print("Reading U, V, HGT ...")
    U = read_sav_profile(U_PATH, "U")
    V = read_sav_profile(V_PATH, "V")
    HGT = read_sav_profile(HGT_PATH, "HGT")
    HGT = HGT / 9.80665

    print("Reading U10, V10, ELEV, LAT, LON ...")
    U10 = read_hdf4_1d(U10_PATH, "U10")
    V10 = read_hdf4_1d(V10_PATH, "V10")
    ELEV = read_hdf4_1d(ELEV_PATH, "ELEV")
    LAT = read_hdf4_1d(LAT_PATH, "LAT")
    LON = read_hdf4_1d(LON_PATH, "LON")

    # -----------------------------------------------------
    # 检查形状
    # -----------------------------------------------------
    print("Checking shapes ...")

    n_sample, n_level = U.shape

    if V.shape != U.shape:
        raise ValueError(f"V shape {V.shape} does not match U shape {U.shape}")

    if HGT.shape != U.shape:
        raise ValueError(f"HGT shape {HGT.shape} does not match U shape {U.shape}")

    for name, arr in {
        "U10": U10,
        "V10": V10,
        "ELEV": ELEV,
        "LAT": LAT,
        "LON": LON,
    }.items():
        if arr.shape[0] != n_sample:
            raise ValueError(f"{name} length {arr.shape[0]} does not match U sample {n_sample}")

    lines.append(f"n_sample: {n_sample}")
    lines.append(f"n_level : {n_level}")
    lines.append("")
    lines.append(stat_line("U", U))
    lines.append(stat_line("V", V))
    lines.append(stat_line("HGT", HGT))
    lines.append(stat_line("U10", U10))
    lines.append(stat_line("V10", V10))
    lines.append(stat_line("ELEV", ELEV))
    lines.append(stat_line("LAT", LAT))
    lines.append(stat_line("LON", LON))
    lines.append("")

    # -----------------------------------------------------
    # 目标高度 = 地表高度 + 6000 m
    # -----------------------------------------------------
    target_hgt = ELEV + 6000.0

    lines.append(stat_line("target_hgt = ELEV + 6000", target_hgt))
    lines.append("")

    # -----------------------------------------------------
    # 快速插值
    # -----------------------------------------------------
    print("Interpolating U to 6 km AGL using fast vectorized method ...")
    U6 = interp_to_target_fast(HGT, U, target_hgt, var_name="U6")

    print("Interpolating V to 6 km AGL using fast vectorized method ...")
    V6 = interp_to_target_fast(HGT, V, target_hgt, var_name="V6")

    lines.append(stat_line("U6", U6))
    lines.append(stat_line("V6", V6))
    lines.append("")

    # -----------------------------------------------------
    # 计算 S06
    # -----------------------------------------------------
    print("Calculating S06 ...")

    valid = (
        np.isfinite(U6) &
        np.isfinite(V6) &
        np.isfinite(U10) &
        np.isfinite(V10)
    )

    S06 = np.full(n_sample, np.nan, dtype=np.float32)
    S06[valid] = np.sqrt(
        (U6[valid] - U10[valid]) ** 2 +
        (V6[valid] - V10[valid]) ** 2
    )

    lines.append(stat_line("S06", S06))
    lines.append("")
    lines.append(f"valid S06 count: {valid.sum()}")
    lines.append(f"valid S06 ratio: {valid.sum() / n_sample:.6f}")
    lines.append("")

    # -----------------------------------------------------
    # 保存 NPY
    # -----------------------------------------------------
    print("Saving S06.npy ...")
    np.save(OUT_NPY, S06.astype(np.float32))

    # -----------------------------------------------------
    # 保存 NPZ，包含经纬度
    # -----------------------------------------------------
    print("Saving S06_with_latlon.npz ...")
    np.savez_compressed(
        OUT_NPZ,
        S06=S06.astype(np.float32),
        LAT=LAT.astype(np.float32),
        LON=LON.astype(np.float32),
        ELEV=ELEV.astype(np.float32)
    )

    # -----------------------------------------------------
    # 保存 HDF4，包含经纬度
    # -----------------------------------------------------
    print("Saving S06.HDF ...")

    if SD is None:
        lines.append("WARNING: pyhdf not installed, S06.HDF was not saved.")
    else:
        write_hdf4_output(OUT_HDF, S06, LAT, LON, ELEV)

    # -----------------------------------------------------
    # 输出检查信息
    # -----------------------------------------------------
    elapsed = time.time() - t0

    lines.append(f"elapsed_seconds: {elapsed:.2f}")
    lines.append(f"elapsed_minutes: {elapsed / 60:.2f}")
    lines.append("")
    lines.append(f"OUT_NPY: {OUT_NPY}")
    lines.append(f"OUT_NPZ: {OUT_NPZ}")
    lines.append(f"OUT_HDF: {OUT_HDF}")
    lines.append(f"OUT_TXT: {OUT_TXT}")

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("=" * 100)
    print("S06 计算完成！")
    print(f"耗时: {elapsed / 60:.2f} 分钟")
    print("输出 NPY :", OUT_NPY)
    print("输出 NPZ :", OUT_NPZ)
    print("输出 HDF :", OUT_HDF)
    print("检查 TXT:", OUT_TXT)
    print("=" * 100)