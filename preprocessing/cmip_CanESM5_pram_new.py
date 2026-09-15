# Purpose: cmip CanESM5 pram new.
# Source: cmip_CanESM5_pram_new.py; see docs/source_coverage.csv for cell provenance.
# Run order and required inputs: docs/workflow.md.
# Run from repository root: python -m preprocessing.cmip_CanESM5_pram_new

# -*- coding: utf-8 -*-
"""
CanESM5 hybrid-sigma (6hrLev) 环境参数诊断（原始数据）
- 输入：ta, hus, ua, va, ps + orog（historical/ssp585）
- 压强：p = ap + b*ps  (Pa)
- 高度：静力方程积分得到 z_agl，和 orog 合成 z_asl
- 输出：mucape/mucin/cape/cin/pw/s06/flh/h_10c/h_30c/dh_10_30/div500/td_sfc/theta_e/k_index

增强：
- 一年一输出
- 真·断点续传：按 time 分块落盘（part 文件），可中断继续
"""

from __future__ import annotations

from publication_config import resource_path
import os
from pathlib import Path
from typing import Tuple, Dict, List
import re
import numpy as np
import xarray as xr
from tqdm import tqdm

# ---------------- 线程设置 ----------------
os.environ.setdefault("NUMBA_NUM_THREADS", "16")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numba
from numba import njit, prange, set_num_threads

# ---------------- 常数 ----------------
Rd = 287.05
Rv = 461.5
Cp = 1004.0
G  = 9.80665
P0 = 100000.0
EPS = Rd / Rv

# ---------------- 用户配置 ----------------
ROOT = Path(resource_path('cmip6_raw', 'CanESM5'))
OROG_DIR = ROOT / "orog"
OUT_DIR  = Path(resource_path('models', 'CanESM5/envdiag'))

TIME_SLICE = slice(None)     # 正式跑 slice(None)
MU_DEPTH_PA = 30000.0
MASK_MARGIN_M = 10.0
TD_MIN_C = -80.0
MOIST_NSUB = 10

# 断点续传：每块 time 数
TIME_BLOCK = 500
# 合并后是否删除 part 文件
CLEAN_PARTS_AFTER_MERGE = True


# =========================
# 年份提取（更稳）
# =========================
def extract_year_from_name(p: Path) -> int:
    """
    从 CMIP6 6hr 文件名中的时间段提取年份。
    支持形如：..._199501010000-199512311800.nc
    """
    m = re.search(r"_(\d{4})\d{8,}-\d{4}\d{8,}", p.name)
    if not m:
        raise ValueError(f"无法从文件名提取年份: {p.name}")
    return int(m.group(1))


# =========================
# 基础热力学（numpy）
# =========================
def es_bolton_pa(Tk):
    Tc = np.clip(np.asarray(Tk) - 273.15, -100.0, 60.0)
    return 611.2 * np.exp(17.67 * Tc / (Tc + 243.5))

def mixing_ratio_from_e(p_pa, e_pa):
    return EPS * e_pa / np.maximum(p_pa - e_pa, 1.0)

def dewpoint_from_specific_humidity(q, p):
    r = q / np.maximum(1.0 - q, 1e-6)
    e = r * p / (EPS + r)
    e_hpa = np.maximum(e * 0.01, 1e-3)
    ln_ratio = np.log(e_hpa / 6.112)
    Td_c = 243.5 * ln_ratio / (17.67 - ln_ratio)
    return Td_c + 273.15

def potential_temperature(T, p):
    return T * (P0 / np.maximum(p, 1.0)) ** (Rd / Cp)

def theta_e_bolton(T, Td, p):
    Tlc = 1.0 / (1.0 / (Td - 273.15 - 56.0) + np.log(np.maximum(T,100.0)/np.maximum(Td,100.0))/800.0) + 56.0 + 273.15
    e = es_bolton_pa(Td)
    r = mixing_ratio_from_e(p, e)
    theta = potential_temperature(T, p)
    return theta * np.exp((3376.0 / np.maximum(Tlc, 100.0) - 2.54) * (r * (1.0 + 0.81 * r)))


# =========================
# Numba：qsat / moist lapse
# =========================
@njit(cache=True, fastmath=True)
def es_bolton_pa_nj(Tk):
    Tc = min(max(Tk - 273.15, -100.0), 60.0)
    return 611.2 * np.exp(17.67 * Tc / (Tc + 243.5))

@njit(cache=True, fastmath=True)
def qsat_nj(T, p):
    e = es_bolton_pa_nj(T)
    r = EPS * e / max(p - e, 1.0)
    return r / (1.0 + r)

@njit(cache=True, fastmath=True)
def qsat_vec_nj(T_arr, p_arr, out_arr):
    n = T_arr.size
    for i in range(n):
        out_arr[i] = qsat_nj(T_arr[i], p_arr[i])

@njit(cache=True, fastmath=True)
def latent_heat_vap_nj(T):
    Tc = min(max(T - 273.15, -100.0), 60.0)
    return 2.501e6 - 2.37e3 * Tc

@njit(cache=True, fastmath=True)
def gamma_moist_varLv_nj(T, p, q_s):
    Lv = latent_heat_vap_nj(T)
    num = G * (1.0 + (Lv * q_s) / (Rd * T))
    den = Cp + (Lv*Lv * q_s * EPS) / (Rd * T * T)
    if den < 1e-6:
        den = 1e-6
    return num / den

@njit(cache=True, fastmath=True)
def moist_lapse_to_p_scalar_nj(T_start, p_start, p_end, nsub):
    T = T_start
    dp = (p_end - p_start) / float(nsub)
    p_cur = p_start
    for _ in range(nsub):
        p0 = p_cur
        p1 = p0 + dp
        pmid = 0.5 * (p0 + p1)
        qs = qsat_nj(T, p0)
        gamma = gamma_moist_varLv_nj(T, p0, qs)
        dz = -Rd * T / (G * pmid) * (p1 - p0)
        T = T - gamma * dz
        p_cur = p1
    return T

@njit(parallel=True, cache=True, fastmath=True)
def moist_lapse_to_p_vec_nj(T_start_arr, p_start_arr, p_end_arr, nsub, out_T):
    n = T_start_arr.size
    for i in prange(n):
        out_T[i] = moist_lapse_to_p_scalar_nj(T_start_arr[i], p_start_arr[i], p_end_arr[i], nsub)


# =========================
# Numba：orog 插值
# =========================
@njit(parallel=True, cache=True, fastmath=True)
def interp_to_orog_linear_nj(var, H_asl, valid, orog_asl_vec):
    nlev, N = var.shape
    out = np.empty(N, dtype=np.float64)
    for j in prange(N):
        out[j] = np.nan
        i0 = -1; i1 = -1
        for i in range(nlev):
            if valid[i, j]:
                if i0 == -1:
                    i0 = i
                elif i1 == -1:
                    i1 = i
                    break
        if i0 == -1:
            continue
        if i1 == -1:
            out[j] = var[i0, j]
            continue
        dz = H_asl[i1, j] - H_asl[i0, j]
        if abs(dz) < 1e-6:
            out[j] = var[i0, j]
            continue
        w = (orog_asl_vec[j] - H_asl[i0, j]) / dz
        out[j] = var[i0, j] + w * (var[i1, j] - var[i0, j])
    return out


# =========================
# CAPE/CIN：逐列 ln(p)
# =========================
@njit(cache=True, fastmath=True)
def _trapz_sum(x0, y0, x1, y1):
    return 0.5 * (y0 + y1) * (x1 - x0)

@njit(cache=True, fastmath=True)
def cape_cin_first_numba_col(lnp_col, dv_col, start_idx, valid_mask):
    n = lnp_col.size
    b = -1
    for i in range(start_idx, n):
        if valid_mask[i]:
            b = i
            break
    if b == -1:
        return 0.0, 0.0, False, np.nan

    t = b
    for i in range(b, n):
        if valid_mask[i]:
            t = i
    if t - b < 1:
        return 0.0, 0.0, False, np.nan

    has_lfc = False
    lfc_x = 0.0

    yb = dv_col[b]
    if np.isfinite(yb) and (yb > 0.0):
        has_lfc = True
        lfc_x = lnp_col[b]
    else:
        for i in range(b, t):
            y1 = dv_col[i]; y2 = dv_col[i+1]
            if np.isfinite(y1) and np.isfinite(y2) and (y1 <= 0.0) and (y2 > 0.0):
                x1 = lnp_col[i]; x2 = lnp_col[i+1]
                if y2 != y1:
                    w = y1 / (y1 - y2)
                    lfc_x = x1 + (x2 - x1) * w
                else:
                    lfc_x = x1
                has_lfc = True
                break

    if not has_lfc:
        return 0.0, 0.0, False, np.nan

    # CIN: negative area up to LFC
    cin_area = 0.0
    x_prev = lnp_col[b]
    y_prev = dv_col[b] if np.isfinite(dv_col[b]) else 0.0

    for i in range(b, t):
        x_cur = lnp_col[i+1]
        y_cur = dv_col[i+1] if np.isfinite(dv_col[i+1]) else 0.0
        if x_cur >= lfc_x:
            cin_area += _trapz_sum(x_prev, y_prev, x_cur, y_cur)
            x_prev = x_cur; y_prev = y_cur
        else:
            cin_area += _trapz_sum(x_prev, y_prev, lfc_x, 0.0)
            break

    cin = -Rd * cin_area
    if cin > 0.0:
        cin = 0.0

    # CAPE: positive area LFC to first EL
    cape_area = 0.0
    x0 = lfc_x; y0 = 0.0

    for i in range(b, t):
        x1 = lnp_col[i]; x2 = lnp_col[i+1]
        if (x1 >= lfc_x) and (x2 < lfc_x):
            y2 = dv_col[i+1] if np.isfinite(dv_col[i+1]) else 0.0
            cape_area += _trapz_sum(x0, y0, x2, y2)
            x0 = x2; y0 = y2

            for j in range(i+1, t):
                x1j = lnp_col[j]; x2j = lnp_col[j+1]
                y1j = dv_col[j]   if np.isfinite(dv_col[j])   else 0.0
                y2j = dv_col[j+1] if np.isfinite(dv_col[j+1]) else 0.0

                if (y1j > 0.0) and (y2j <= 0.0):
                    el_x = x1j if (y2j == y1j) else (x1j + (x2j - x1j) * (y1j / (y1j - y2j)))
                    cape_area += _trapz_sum(x0, y0, el_x, 0.0)
                    x0 = el_x; y0 = 0.0
                    break
                else:
                    cape_area += _trapz_sum(x0, y0, x2j, y2j)
                    x0 = x2j; y0 = y2j
            break

    cape = -Rd * cape_area
    if cape < 0.0:
        cape = 0.0

    return cape, cin, True, lfc_x

@njit(parallel=True, cache=True, fastmath=True)
def compute_cape_cin_all_numba_col(lnp_all, dv_all, k_par, valid,
                                   cape_col, cin_col, has_lfc_col, p_lfc_col):
    nlev, N = dv_all.shape
    for j in prange(N):
        cape, cin, has_lfc, lfc_lnp = cape_cin_first_numba_col(
            lnp_all[:, j], dv_all[:, j], int(k_par[j]), valid[:, j]
        )
        cape_col[j]    = cape
        cin_col[j]     = cin
        has_lfc_col[j] = has_lfc
        p_lfc_col[j]   = np.exp(lfc_lnp) if has_lfc else np.nan


# =========================
# 网格几何 / 散度
# =========================
def grid_dx_dy_m(latitudes, longitudes) -> Tuple[np.ndarray, np.ndarray]:
    R = 6_371_000.0
    lat = np.asarray(latitudes, dtype=np.float64)
    lon = np.asarray(longitudes, dtype=np.float64)
    lat_rad = np.deg2rad(lat); lon_rad = np.deg2rad(lon)
    dlon = float(lon_rad[1] - lon_rad[0])
    dlat = float(lat_rad[1] - lat_rad[0])
    c = np.cos(lat_rad); c = np.where(c < 1e-6, 1e-6, c)
    dx_row = R * abs(dlon) * c
    dy_row = R * abs(dlat) * np.ones_like(lat_rad)
    ny, nx = lat.size, lon.size
    dx_full = np.repeat(dx_row[:, None], nx, axis=1).astype(np.float32)
    dy_full = np.repeat(dy_row[:, None], nx, axis=1).astype(np.float32)
    return dx_full, dy_full

def divergence_2d(u, v, dx_full, dy_full) -> np.ndarray:
    u, v = np.asarray(u, float), np.asarray(v, float)
    dx, dy = np.asarray(dx_full, float), np.asarray(dy_full, float)
    u_f, u_b = np.roll(u, -1, axis=1), np.roll(u,  1, axis=1)
    dx_f, dx_b = np.roll(dx, -1, axis=1), np.roll(dx,  1, axis=1)
    du_dx = 2 * (u_f - u_b) / (dx_f + dx_b)

    dv_dy = np.empty_like(v, float)
    dv_dy[1:-1, :] = (v[2:, :] - v[:-2, :]) / (dy[2:, :] + dy[:-2, :])
    dv_dy[0, :]  = (v[1, :] - v[0, :]) / dy[0, :]
    dv_dy[-1,:]  = (v[-1,:] - v[-2,:]) / dy[-1,:]
    return (du_dx + dv_dy).astype(np.float32)


# =========================
# 等温面：贴地规则
# =========================
def find_isotherm_height_masked_with_surface(T, H_asl, valid, target_T, T_sfc, orog_asl_vec):
    nlev, N = T.shape
    T_eff = np.where(valid, T, 1e9)
    cross = (T_eff[:-1, :] > target_T) & (T_eff[1:, :] <= target_T)
    has = cross.any(axis=0)
    k = np.where(has, cross.argmax(axis=0), np.maximum(0, (nlev - 2)))
    col = np.arange(N)

    T_dn, T_up = T_eff[k, col], T_eff[k+1, col]
    H_dn, H_up = H_asl[k, col], H_asl[k+1, col]
    dT = np.where(T_up != T_dn, T_up - T_dn, -1e-6)
    w = np.clip((target_T - T_dn) / dT, 0.0, 1.0)
    h_interp = H_dn + w * (H_up - H_dn)

    out = np.full(N, np.nan, dtype=np.float64)

    use_surface = np.isfinite(T_sfc) & (T_sfc <= target_T)
    out[use_surface] = orog_asl_vec[use_surface]

    any_valid = valid.any(axis=0)
    k0 = np.where(any_valid, valid.argmax(axis=0), 0)
    t0 = T[k0, col]
    use_interp = (~use_surface) & any_valid & (t0 > target_T) & has
    out[use_interp] = h_interp[use_interp]
    return out


# =========================
# hybrid 压强与高度
# =========================
def compute_pressure_pa(ap_lev: np.ndarray, b_lev: np.ndarray, ps_2d: np.ndarray) -> np.ndarray:
    return ap_lev[:, None, None] + b_lev[:, None, None] * ps_2d[None, :, :]

def compute_height_asl_from_hydrostatic(p_pa: np.ndarray, T_k: np.ndarray, q: np.ndarray, ps_2d: np.ndarray, orog_m: np.ndarray):
    lev, ny, nx = p_pa.shape
    Tv = T_k * (1.0 + 0.61 * q)

    p_med = np.nanmedian(p_pa.reshape(lev, -1), axis=1)
    if p_med[0] < p_med[-1]:
        p_pa = p_pa[::-1, :, :]
        Tv   = Tv[::-1, :, :]
        p_med = p_med[::-1]

    lnps = np.log(np.maximum(ps_2d, 1.0))
    lnp  = np.log(np.maximum(p_pa, 1.0))

    H_agl = np.zeros_like(p_pa, dtype=np.float64)
    dz0 = (Rd / G) * Tv[0] * (lnps - lnp[0])
    H_agl[0] = np.maximum(dz0, 0.0)

    for k in range(1, lev):
        ln1 = lnp[k-1]
        ln2 = lnp[k]
        Tv_mid = 0.5 * (Tv[k-1] + Tv[k])
        dz = (Rd / G) * Tv_mid * (ln1 - ln2)
        H_agl[k] = H_agl[k-1] + np.maximum(dz, 0.0)

    return (orog_m[None, :, :] + H_agl).astype(np.float64)


# =========================
# 插值到指定压强
# =========================
@njit(parallel=True, cache=True, fastmath=True)
def interp_to_pressure_ln_nj(p_lev, var_lev, target_p, out_var):
    nlev, N = p_lev.shape
    for j in prange(N):
        out_var[j] = np.nan
        for k in range(nlev - 1):
            p1 = p_lev[k, j]
            p2 = p_lev[k+1, j]
            if np.isnan(p1) or np.isnan(p2):
                continue
            if (p1 >= target_p) and (p2 <= target_p):
                lp1 = np.log(max(p1, 1.0))
                lp2 = np.log(max(p2, 1.0))
                lpt = np.log(max(target_p, 1.0))
                w = 0.0
                if abs(lp2 - lp1) > 1e-12:
                    w = (lpt - lp1) / (lp2 - lp1)
                out_var[j] = var_lev[k, j] + w * (var_lev[k+1, j] - var_lev[k, j])
                break


# =========================
# parcel 抬升
# =========================
def lift_parcel_from_start(T0, q0, p0, Pcol, validcol, k_start, moist_nsub):
    lev, N = Pcol.shape
    T_par = np.full((lev, N), np.nan, dtype=np.float64)
    q_par = np.full((lev, N), np.nan, dtype=np.float64)

    col = np.arange(N)
    ok0 = np.isfinite(T0) & np.isfinite(q0) & np.isfinite(p0) & (k_start >= 0) & (k_start < lev)
    ok0 = ok0 & validcol[k_start, col]

    T_par[k_start[ok0], col[ok0]] = T0[ok0]
    q_par[k_start[ok0], col[ok0]] = q0[ok0]

    Td0 = dewpoint_from_specific_humidity(q0, p0)
    Td0 = np.where(Td0 < (TD_MIN_C + 273.15), np.nan, Td0)

    Td0c = Td0 - 273.15
    Tlcl = 1.0/(1.0/(Td0c-56.0) + np.log(np.maximum(T0,100.0)/np.maximum(Td0,100.0))/800.0) + 56.0 + 273.15
    theta0 = potential_temperature(T0, p0)
    p_lcl = P0 / np.maximum((theta0 / Tlcl) ** (Cp / Rd), 1e-6)

    for k in range(lev - 1):
        active = np.isfinite(T_par[k, :])
        if not np.any(active):
            continue

        pcur  = Pcol[k, :]
        pnext = Pcol[k+1, :]
        plcl  = p_lcl

        only_dry   = active & (pcur > plcl) & (pnext >  plcl)
        only_moist = active & (pcur <= plcl) & (pnext <= plcl)
        is_cross   = active & (pcur > plcl) & (pnext <= plcl)

        if np.any(only_dry):
            jj = np.where(only_dry)[0]
            T_par[k+1, jj] = T_par[k, jj] * (pnext[jj] / pcur[jj]) ** (Rd/Cp)
            q_par[k+1, jj] = q_par[k, jj]

        if np.any(only_moist):
            jj = np.where(only_moist)[0]
            Tm = T_par[k, jj].copy()
            pm = pcur[jj].copy()
            pn = pnext[jj].copy()
            outT = np.empty_like(Tm)
            moist_lapse_to_p_vec_nj(Tm, pm, pn, moist_nsub, outT)
            T_par[k+1, jj] = outT
            q_out = np.empty_like(outT)
            qsat_vec_nj(outT, pn, q_out)
            q_par[k+1, jj] = np.minimum(q_par[k, jj], q_out)

        if np.any(is_cross):
            jj = np.where(is_cross)[0]
            pl = plcl[jj]
            Tl = T_par[k, jj] * (pl / pcur[jj]) ** (Rd/Cp)
            pm = pl.copy()
            pn = pnext[jj].copy()
            outT2 = np.empty_like(Tl)
            moist_lapse_to_p_vec_nj(Tl, pm, pn, moist_nsub, outT2)
            T_par[k+1, jj] = outT2
            q_out2 = np.empty_like(outT2)
            qsat_vec_nj(outT2, pn, q_out2)
            q_par[k+1, jj] = q_out2

    return T_par, q_par


# =========================
# 核心：单时次诊断（保持你的逻辑不变）
# =========================
def diag_one_time(ap_lev, b_lev, ps2d, orog2d, T_lev, q_lev, u_lev, v_lev, lat, lon, dx, dy):

    # 1) p 与 H（确保 surface->top：由 p_med 判定反转）
    p3 = compute_pressure_pa(ap_lev, b_lev, ps2d).astype(np.float64)
    lev, ny, nx = p3.shape
    p_med = np.nanmedian(p3.reshape(lev, -1), axis=1)
    if p_med[0] < p_med[-1]:
        p3    = p3[::-1, :, :]
        T_lev = T_lev[::-1, :, :]
        q_lev = q_lev[::-1, :, :]
        u_lev = u_lev[::-1, :, :]
        v_lev = v_lev[::-1, :, :]

    H_asl = compute_height_asl_from_hydrostatic(p3, T_lev, q_lev, ps2d, orog2d)

    N = ny * nx
    r2 = lambda a: a.reshape(lev, N)

    T  = r2(T_lev.astype(np.float64))
    q  = r2(q_lev.astype(np.float64))
    U  = r2(u_lev.astype(np.float64))
    V  = r2(v_lev.astype(np.float64))
    P  = r2(p3.astype(np.float64))
    H  = r2(H_asl.astype(np.float64))

    orog_vec = orog2d.reshape(-1).astype(np.float64)

    # 2) valid：在地形以上
    valid = H >= (orog_vec[None, :] + MASK_MARGIN_M)
    any_valid = valid.any(axis=0)

    # 3) surface 变量（orog 处插值）
    T_sfc = interp_to_orog_linear_nj(T, H, valid, orog_vec)
    U_sfc = interp_to_orog_linear_nj(U, H, valid, orog_vec)
    V_sfc = interp_to_orog_linear_nj(V, H, valid, orog_vec)
    q_sfc = interp_to_orog_linear_nj(q, H, valid, orog_vec)
    p_sfc = ps2d.reshape(-1).astype(np.float64)

    Td_sfc = dewpoint_from_specific_humidity(q_sfc, p_sfc)
    Td_sfc = np.where(Td_sfc < (TD_MIN_C + 273.15), np.nan, Td_sfc)

    thetae_sfc = np.full(N, np.nan, dtype=np.float64)
    ok_sfc = np.isfinite(Td_sfc) & np.isfinite(T_sfc)
    thetae_sfc[ok_sfc] = theta_e_bolton(T_sfc[ok_sfc], Td_sfc[ok_sfc], p_sfc[ok_sfc])

    # 4) PW：q dp / g
    dp = -(np.diff(P, axis=0))
    q_mid = 0.5 * (q[:-1,:] + q[1:,:])
    valid_mid = valid[:-1,:] & valid[1:,:]
    PW_kgm2 = (np.where(valid_mid, q_mid * dp, 0.0)).sum(axis=0) / G
    out_pw = PW_kgm2.astype(np.float32)

    # 5) S06：6km AGL
    H6_asl = orog_vec + 6000.0
    mask6 = valid & (H >= H6_asl[None, :])
    has6 = mask6.any(axis=0)
    idx_up = np.where(has6, mask6.argmax(axis=0), 1)
    idx_dn = np.clip(idx_up - 1, 0, lev-2)
    col = np.arange(N)
    h_dn, h_up = H[idx_dn, col], H[idx_up, col]
    u_dn, u_up = U[idx_dn, col], U[idx_up, col]
    v_dn, v_up = V[idx_dn, col], V[idx_up, col]
    w6 = np.clip((H6_asl - h_dn) / np.maximum(h_up - h_dn, 1e-3), 0.0, 1.0)
    u6 = u_dn + w6 * (u_up - u_dn)
    v6 = v_dn + w6 * (v_up - v_dn)
    out_s06 = np.sqrt((u6 - U_sfc)**2 + (v6 - V_sfc)**2)
    out_s06 = np.where(has6 & any_valid, out_s06, np.nan).astype(np.float32)

    # 6) 等温面高度（ASL -> AGL）
    flh_asl = find_isotherm_height_masked_with_surface(T, H, valid, 273.15, T_sfc, orog_vec)
    h10_asl = find_isotherm_height_masked_with_surface(T, H, valid, 263.15, T_sfc, orog_vec)
    h30_asl = find_isotherm_height_masked_with_surface(T, H, valid, 243.15, T_sfc, orog_vec)

    out_flh = (flh_asl - orog_vec).astype(np.float32)
    out_h10 = (h10_asl - orog_vec).astype(np.float32)
    out_h30 = (h30_asl - orog_vec).astype(np.float32)
    out_dh1030 = np.where(np.isfinite(out_h10) & np.isfinite(out_h30),
                          np.maximum(out_h30 - out_h10, 0.0),
                          np.nan).astype(np.float32)

    # 7) div500
    u500 = np.full(N, np.nan, dtype=np.float64)
    v500 = np.full(N, np.nan, dtype=np.float64)
    interp_to_pressure_ln_nj(P, U, 50000.0, u500)
    interp_to_pressure_ln_nj(P, V, 50000.0, v500)
    out_div500 = divergence_2d(u500.reshape(ny,nx), v500.reshape(ny,nx), dx, dy).astype(np.float32)

    # 8) K index
    T850 = np.full(N, np.nan, dtype=np.float64)
    T700 = np.full(N, np.nan, dtype=np.float64)
    T500 = np.full(N, np.nan, dtype=np.float64)
    q850 = np.full(N, np.nan, dtype=np.float64)
    q700 = np.full(N, np.nan, dtype=np.float64)

    interp_to_pressure_ln_nj(P, T, 85000.0, T850)
    interp_to_pressure_ln_nj(P, T, 70000.0, T700)
    interp_to_pressure_ln_nj(P, T, 50000.0, T500)
    interp_to_pressure_ln_nj(P, q, 85000.0, q850)
    interp_to_pressure_ln_nj(P, q, 70000.0, q700)

    Td850 = dewpoint_from_specific_humidity(q850, np.full(N, 85000.0))
    Td700 = dewpoint_from_specific_humidity(q700, np.full(N, 70000.0))

    Kidx = ((T850-273.15) - (T500-273.15)) + (Td850-273.15) - ((T700-273.15) - (Td700-273.15))
    Kidx = np.where(p_sfc >= 85000.0, Kidx, np.nan).astype(np.float32)

    # 9) CAPE/CIN：SB + MU（逐列 ln(p)）
    Pcol = P
    Tcol = T
    qcol = q
    validcol = valid
    lnp_all = np.log(np.maximum(Pcol, 1.0))

    # SB 起点：最低有效层
    k0_sb = np.where(validcol.any(axis=0), validcol.argmax(axis=0), 0).astype(np.int64)
    T0_sb = T_sfc.astype(np.float64)
    q0_sb = q_sfc.astype(np.float64)
    p0_sb = p_sfc.astype(np.float64)

    Tpar_sb, qpar_sb = lift_parcel_from_start(T0_sb, q0_sb, p0_sb, Pcol, validcol, k0_sb, MOIST_NSUB)
    Tv_par_sb = Tpar_sb * (1.0 + 0.61 * qpar_sb)
    Tv_env    = Tcol * (1.0 + 0.61 * qcol)
    dv_sb = Tv_par_sb - Tv_env

    cape_sb = np.full(N, np.nan, dtype=np.float64)
    cin_sb  = np.full(N, np.nan, dtype=np.float64)
    has_lfc_sb = np.zeros(N, dtype=np.bool_)
    p_lfc_sb   = np.full(N, np.nan, dtype=np.float64)
    compute_cape_cin_all_numba_col(lnp_all, dv_sb, k0_sb, validcol, cape_sb, cin_sb, has_lfc_sb, p_lfc_sb)

    # MU 起点：ps 到 ps-300hPa 内 theta-e 最大
    Td_env_col = dewpoint_from_specific_humidity(qcol, Pcol)
    Td_env_col = np.where(Td_env_col < (TD_MIN_C + 273.15), np.nan, Td_env_col)
    thetae_all = theta_e_bolton(Tcol, Td_env_col, Pcol)

    p_top = np.maximum(p_sfc - MU_DEPTH_PA, np.nanmin(Pcol, axis=0))
    cand = (Pcol >= p_top[None, :]) & (Pcol <= p_sfc[None, :]) & validcol & np.isfinite(thetae_all)
    thetae_cand = np.where(cand, thetae_all, -np.inf)
    has_cand = np.isfinite(thetae_cand).any(axis=0)
    k_mu = np.argmax(thetae_cand, axis=0).astype(np.int64)

    col = np.arange(N)
    T0_mu = np.where(has_cand, Tcol[k_mu, col], np.nan)
    q0_mu = np.where(has_cand, qcol[k_mu, col], np.nan)
    p0_mu = np.where(has_cand, Pcol[k_mu, col], np.nan)

    Tpar_mu, qpar_mu = lift_parcel_from_start(T0_mu, q0_mu, p0_mu, Pcol, validcol, k_mu, MOIST_NSUB)
    Tv_par_mu = Tpar_mu * (1.0 + 0.61 * qpar_mu)
    dv_mu = Tv_par_mu - Tv_env

    mucape = np.full(N, np.nan, dtype=np.float64)
    mucin  = np.full(N, np.nan, dtype=np.float64)
    has_lfc_mu = np.zeros(N, dtype=np.bool_)
    p_lfc_mu   = np.full(N, np.nan, dtype=np.float64)
    compute_cape_cin_all_numba_col(lnp_all, dv_mu, k_mu, validcol, mucape, mucin, has_lfc_mu, p_lfc_mu)

    ny, nx = ps2d.shape
    c2 = lambda a: a.reshape(ny, nx).astype(np.float32)

    out = {
        "cape":     c2(cape_sb),
        "cin":      c2(-cin_sb),     # magnitude >=0
        "mucape":   c2(mucape),
        "mucin":    c2(-mucin),      # magnitude >=0
        "pw":       out_pw.reshape(ny, nx),
        "s06":      out_s06.reshape(ny, nx),
        "flh":      out_flh.reshape(ny, nx),
        "h_10c":    out_h10.reshape(ny, nx),
        "h_30c":    out_h30.reshape(ny, nx),
        "dh_10_30": out_dh1030.reshape(ny, nx),
        "div500":   out_div500,
        "td_sfc":   (Td_sfc - 273.15).reshape(ny, nx).astype(np.float32),
        "theta_e":  thetae_sfc.reshape(ny, nx).astype(np.float32),
        "k_index":  Kidx.reshape(ny, nx).astype(np.float32),
    }
    return out


# =========================
# orog 读取（更稳：直接用目标 lat/lon DataArray）
# =========================
def load_orog_m(orog_nc: Path, target_lat: xr.DataArray, target_lon: xr.DataArray) -> xr.DataArray:
    with xr.open_dataset(orog_nc) as ds:
        if "orog" not in ds.data_vars:
            raise KeyError("orog 文件中未找到变量 orog")
        da = ds["orog"]
        for tdim in ["time", "valid_time"]:
            if tdim in da.dims:
                da = da.isel({tdim: 0})
        da = da.sel(lat=target_lat, lon=target_lon, method="nearest")
        da.name = "orog"
        da.attrs["units"] = "m"
        return da.astype(np.float64)


# =========================
# 文件发现：按年匹配
# =========================
def find_yearly_file_map(exp: str) -> Dict[int, Dict[str, Path]]:
    patt = {
        "ta":  f"ta_6hrLev_CanESM5_{exp}_r1i1p2f1_gn_*.nc",
        "hus": f"hus_6hrLev_CanESM5_{exp}_r1i1p2f1_gn_*.nc",
        "ua":  f"ua_6hrLev_CanESM5_{exp}_r1i1p2f1_gn_*.nc",
        "va":  f"va_6hrLev_CanESM5_{exp}_r1i1p2f1_gn_*.nc",
    }

    year_map: Dict[int, Dict[str, Path]] = {}
    for var, pattern in patt.items():
        hits = sorted(ROOT.rglob(pattern))
        if not hits:
            raise FileNotFoundError(f"找不到 {var} 文件：{pattern}（ROOT={ROOT}）")

        for f in hits:
            y = extract_year_from_name(f)
            year_map.setdefault(y, {})
            if var in year_map[y]:
                # 明确提示覆盖
                old = year_map[y][var]
                year_map[y][var] = f
                print(f"[WARN] duplicate {var} for year {y}: override {old.name} -> {f.name}")
            else:
                year_map[y][var] = f

    complete = {y: d for y, d in year_map.items() if all(k in d for k in ["ta","hus","ua","va"])}
    missing_years = sorted(set(year_map.keys()) - set(complete.keys()))
    if missing_years:
        print(f"[WARN] 有年份变量不齐全，将跳过这些年份: {missing_years[:30]}{'...' if len(missing_years)>30 else ''}")

    if not complete:
        raise RuntimeError(f"未找到任何四变量齐全的年份（exp={exp}）。请检查 ROOT。")

    return dict(sorted(complete.items()))


def infer_orog_file(exp: str) -> Path:
    cand = sorted(OROG_DIR.glob(f"orog_fx_CanESM5_{exp}_r1i1p2f1_gn.nc"))
    if not cand:
        raise FileNotFoundError(f"找不到地形文件：orog_fx_CanESM5_{exp}_r1i1p2f1_gn.nc 于 {OROG_DIR}")
    return cand[0]


# =========================
# 写一个 time-block part 文件
# =========================
def write_part_nc(out_path: Path, tvals, lat, lon, out_block: Dict[str, np.ndarray]):
    ds_out = xr.Dataset(
        data_vars=dict(
            mucape=(("time","lat","lon"), out_block["mucape"], {"units":"J kg-1","long_name":"Most-unstable CAPE"}),
            mucin =(("time","lat","lon"), out_block["mucin"],  {"units":"J kg-1","long_name":"Most-unstable CIN (magnitude, >=0)"}),
            cape  =(("time","lat","lon"), out_block["cape"],   {"units":"J kg-1","long_name":"Surface-based CAPE"}),
            cin   =(("time","lat","lon"), out_block["cin"],    {"units":"J kg-1","long_name":"Surface-based CIN (magnitude, >=0)"}),
            pw    =(("time","lat","lon"), out_block["pw"],     {"units":"mm","long_name":"Precipitable water"}),
            s06   =(("time","lat","lon"), out_block["s06"],    {"units":"m s-1","long_name":"0-6 km bulk wind shear (AGL)"}),
            flh   =(("time","lat","lon"), out_block["flh"],    {"units":"m","long_name":"Freezing level height (0C) AGL"}),
            h_10c =(("time","lat","lon"), out_block["h_10c"],  {"units":"m","long_name":"-10C isotherm height AGL"}),
            h_30c =(("time","lat","lon"), out_block["h_30c"],  {"units":"m","long_name":"-30C isotherm height AGL"}),
            dh_10_30=(("time","lat","lon"), out_block["dh_10_30"], {"units":"m","long_name":"Thickness h30-h10 (>=0)"}),
            div500=(("time","lat","lon"), out_block["div500"], {"units":"s-1","long_name":"Horizontal divergence at 500 hPa"}),
            td_sfc=(("time","lat","lon"), out_block["td_sfc"], {"units":"degC","long_name":"Surface dewpoint"}),
            theta_e=(("time","lat","lon"), out_block["theta_e"],{"units":"K","long_name":"Equivalent potential temperature at surface"}),
            k_index=(("time","lat","lon"), out_block["k_index"],{"units":"degC","long_name":"K index"}),
        ),
        coords=dict(time=tvals, lat=lat, lon=lon)
    )

    ny = len(lat); nx = len(lon)
    enc = {v: {"zlib": True, "complevel": 2,
               "chunksizes": (1, max(1, ny//4), max(1, nx//4))}
           for v in ds_out.data_vars}

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    ds_out.to_netcdf(tmp, encoding=enc)
    tmp.replace(out_path)


# =========================
# 合并 part 文件 -> 年度文件
# =========================
def merge_parts_to_year(out_nc: Path, part_files: List[Path], clean_parts: bool = True):
    part_files = sorted(part_files)
    dss = [xr.open_dataset(p) for p in part_files]
    try:
        ds_all = xr.concat(dss, dim="time")
        tmp = out_nc.with_suffix(out_nc.suffix + ".tmp")
        ds_all.to_netcdf(tmp)
        tmp.replace(out_nc)
    finally:
        for ds in dss:
            ds.close()

    if clean_parts:
        for p in part_files:
            try:
                p.unlink()
            except Exception:
                pass


# =========================
# 主流程：按年 + 按 block（断点续传）
# =========================
def process_experiment(exp: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    orog_nc = infer_orog_file(exp)
    print(f"[INFO] exp={exp} orog: {orog_nc}")

    year_map = find_yearly_file_map(exp)
    years = list(year_map.keys())
    print(f"[INFO] exp={exp} complete years: {years[0]}..{years[-1]} (n={len(years)})")

    # 用第一年的 hus 建立网格 + orog
    sample_hus = year_map[years[0]]["hus"]
    with xr.open_dataset(sample_hus) as ds_tmp:
        lat_da = ds_tmp["lat"]
        lon_da = ds_tmp["lon"]
        lat = lat_da.values
        lon = lon_da.values
        ny, nx = len(lat), len(lon)

    orog2d = load_orog_m(orog_nc, lat_da, lon_da).values.astype(np.float64)
    dx, dy = grid_dx_dy_m(lat, lon)

    out_vars = ["mucape","mucin","cape","cin","pw","s06","flh","h_10c","h_30c","dh_10_30","div500","td_sfc","theta_e","k_index"]

    for y in years:
        files = year_map[y]
        out_nc = OUT_DIR / f"CanESM5_{exp}_envdiag_{y}.nc"
        if out_nc.exists():
            print(f"[SKIP] {out_nc.name} exists")
            continue

        print(f"\n[INFO] ===== YEAR {y} ({exp}) =====")
        for k, v in files.items():
            print(f"  {k}: {v.name}")

        ds_hus = xr.open_dataset(files["hus"], chunks={"time": 1})
        ds_ta  = xr.open_dataset(files["ta"],  chunks={"time": 1})
        ds_ua  = xr.open_dataset(files["ua"],  chunks={"time": 1})
        ds_va  = xr.open_dataset(files["va"],  chunks={"time": 1})

        try:
            ap = ds_hus["ap"].values.astype(np.float64)
            b  = ds_hus["b"].values.astype(np.float64)

            time_name = "time"
            ntime_total = ds_hus.sizes[time_name]
            time_idx_full = np.arange(ntime_total)[TIME_SLICE]
            tvals_full = ds_hus[time_name].isel({time_name: time_idx_full}).values
            nt = time_idx_full.size

            # 需要的 part 数
            n_parts = int(np.ceil(nt / TIME_BLOCK))

            # 逐 part：存在则跳过（断点续传核心）
            part_files = []
            for ip in range(n_parts):
                i0 = ip * TIME_BLOCK
                i1 = min((ip + 1) * TIME_BLOCK, nt)

                part_path = OUT_DIR / f"CanESM5_{exp}_envdiag_{y}.part{ip:03d}.nc"
                part_files.append(part_path)

                if part_path.exists():
                    print(f"[SKIP] part exists: {part_path.name}")
                    continue

                # 当前块的 time 索引与 time 值
                time_idx = time_idx_full[i0:i1]
                tvals = tvals_full[i0:i1]
                nb = len(time_idx)

                # 块内输出数组（小内存）
                out_block = {k: np.full((nb, ny, nx), np.nan, dtype=np.float32) for k in out_vars}

                for it in tqdm(range(nb), desc=f"{exp} {y} part{ip:03d}", ncols=100):
                    ti = int(time_idx[it])
                    sel = {time_name: ti}

                    T_lev = ds_ta["ta"].isel(sel).load().values.astype(np.float64)
                    q_lev = ds_hus["hus"].isel(sel).load().values.astype(np.float64)
                    u_lev = ds_ua["ua"].isel(sel).load().values.astype(np.float64)
                    v_lev = ds_va["va"].isel(sel).load().values.astype(np.float64)
                    ps2d  = ds_hus["ps"].isel(sel).load().values.astype(np.float64)

                    res = diag_one_time(ap, b, ps2d, orog2d, T_lev, q_lev, u_lev, v_lev, lat, lon, dx, dy)
                    for k in out_vars:
                        out_block[k][it] = res[k]

                # 写 part（原子写：先 tmp 再 replace）
                write_part_nc(part_path, tvals=tvals, lat=lat, lon=lon, out_block=out_block)
                print(f"[OK] wrote {part_path.name}")

            # 全部 part 就绪后合并
            # 重新扫描一次，避免中途跳过导致列表不齐
            part_files = sorted(OUT_DIR.glob(f"CanESM5_{exp}_envdiag_{y}.part*.nc"))
            if len(part_files) != n_parts:
                raise RuntimeError(f"YEAR {y}: part files incomplete: {len(part_files)}/{n_parts}. 可能有中断或某 part 写失败。")

            print(f"[INFO] merging {len(part_files)} parts -> {out_nc.name}")
            merge_parts_to_year(out_nc, part_files, clean_parts=CLEAN_PARTS_AFTER_MERGE)
            print(f"[DONE] saved -> {out_nc}")

        finally:
            ds_hus.close(); ds_ta.close(); ds_ua.close(); ds_va.close()


if __name__ == "__main__":
    try:
        set_num_threads(int(os.getenv("NUMBA_NUM_THREADS","16")))
        print(f"[INFO] Numba threads: {numba.get_num_threads()}")
    except Exception:
        pass

    TIME_SLICE = slice(None)   # 正式跑
    process_experiment("historical")
    process_experiment("ssp585")
