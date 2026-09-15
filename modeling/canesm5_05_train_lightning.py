# Purpose: canesm5 05 train lightning.
# Source: cmip6_figure_CanESM5_XGB_plot_latest_checked.ipynb; see docs/source_coverage.csv for cell provenance.
# Run order and required inputs: docs/workflow.md.
# Run from repository root: python -m modeling.canesm5_05_train_lightning

# -*- coding: utf-8 -*-1
from __future__ import annotations

from publication_config import resource_path

import os
import json
import time
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# =========================
# 0) PATHS & CONFIG
# =========================
FLASH_PARQ = resource_path('models', 'CanESM5/ml_feature_withSRTM/train_table_flash_hail_withSRTM_strictLand.parquet')
OUT_ROOT   = resource_path('models', 'CanESM5/ml_xgboost_result')

SEED = 99
TEST_SIZE = 0.15
VAL_SIZE  = 0.15  # fraction of TOTAL; implemented via val_frac_of_trval

TARGET_COL = "LISOTD_Flash"

# Latitude constraint (apply BEFORE split)
LAT_COL = "Latitude"
LAT_MAX = 63.0  # keep samples within [-63, 63]

DROP_COLS = [
    "Latitude", "Longitude", "Land_Sea",   # no lat/lon; Land_Sea already strict land
    "LISOTD_Flash", "ni_HailPF"            # drop both targets from features
]

TOPK = 120

# "Rlo" preset base params
RLO_PARAMS = dict(
    booster="gbtree",
    objective="reg:squarederror",
    eval_metric="rmse",
    learning_rate=0.03,
    max_depth=5,
    min_child_weight=5.0,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.0,
    reg_lambda=1.0,
    gamma=0.0,
    tree_method="hist",
)

REG_SCALE = 10.0  # scales reg_lambda/reg_alpha

N_ESTIMATORS_MAX = 6000
EARLY_STOP_ROUNDS = 80

PLOT_SCATTER = True  # set False if you don't want plots


# =========================
# 1) helpers
# =========================
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def metrics_reg(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Robust regression metrics on finite pairs only."""
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[m]
    yp = y_pred[m]
    if yt.size == 0:
        return dict(r2=np.nan, rmse=np.nan, mae=np.nan, n=0)
    return dict(
        r2=float(r2_score(yt, yp)),
        rmse=float(np.sqrt(mean_squared_error(yt, yp))),
        mae=float(mean_absolute_error(yt, yp)),
        n=int(yt.size),
    )

def build_feature_list(df: pd.DataFrame) -> list[str]:
    """Select numeric feature columns excluding DROP_COLS and targets."""
    cols: list[str] = []
    for c in df.columns:
        if c in DROP_COLS:
            continue
        if c == TARGET_COL:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols

def apply_reg_scale(params: dict, scale: float) -> dict:
    """Scale regularization knobs."""
    p = dict(params)
    p["reg_lambda"] = float(p.get("reg_lambda", 1.0)) * float(scale)
    p["reg_alpha"]  = float(p.get("reg_alpha", 0.0))  * float(scale)
    return p

def split_train_val_test(n: int):
    """Random split: test first, then split trainval into train/val."""
    idx = np.arange(n)
    idx_trval, idx_te = train_test_split(
        idx, test_size=TEST_SIZE, random_state=SEED, shuffle=True
    )
    val_frac_of_trval = VAL_SIZE / (1.0 - TEST_SIZE)
    idx_tr, idx_va = train_test_split(
        idx_trval, test_size=val_frac_of_trval, random_state=SEED, shuffle=True
    )
    return idx_tr, idx_va, idx_te

def fit_rank_features_by_gain(
    X_tr: pd.DataFrame, y_tr: np.ndarray,
    X_va: pd.DataFrame, y_va: np.ndarray,
    base_params: dict
) -> list[str]:
    """Train a quick model on ALL features to get gain importance ranking."""
    dtr = xgb.DMatrix(X_tr, label=y_tr, feature_names=list(X_tr.columns))
    dva = xgb.DMatrix(X_va, label=y_va, feature_names=list(X_va.columns))

    bst = xgb.train(
        params=base_params,
        dtrain=dtr,
        num_boost_round=N_ESTIMATORS_MAX,
        evals=[(dva, "valid")],
        early_stopping_rounds=EARLY_STOP_ROUNDS,
        verbose_eval=False,
    )

    score = bst.get_score(importance_type="gain")
    gains = {f: float(score.get(f, 0.0)) for f in X_tr.columns}
    ranked = sorted(gains.items(), key=lambda kv: kv[1], reverse=True)
    # keep positive-gain first, then zero-gain
    return [k for k, v in ranked if v > 0.0] + [k for k, v in ranked if v == 0.0]

def train_final(
    X_tr: pd.DataFrame, y_tr: np.ndarray,
    X_va: pd.DataFrame, y_va: np.ndarray,
    X_te: pd.DataFrame, y_te: np.ndarray,
    feat_list: list[str],
    params: dict
):
    dtr = xgb.DMatrix(X_tr[feat_list], label=y_tr, feature_names=feat_list)
    dva = xgb.DMatrix(X_va[feat_list], label=y_va, feature_names=feat_list)
    dte = xgb.DMatrix(X_te[feat_list], label=y_te, feature_names=feat_list)

    bst = xgb.train(
        params=params,
        dtrain=dtr,
        num_boost_round=N_ESTIMATORS_MAX,
        evals=[(dtr, "train"), (dva, "valid")],
        early_stopping_rounds=EARLY_STOP_ROUNDS,
        verbose_eval=False,
    )

    best_it = int(bst.best_iteration)
    it_rng = (0, best_it + 1)

    pred_tr = bst.predict(dtr, iteration_range=it_rng)
    pred_va = bst.predict(dva, iteration_range=it_rng)
    pred_te = bst.predict(dte, iteration_range=it_rng)

    m_tr = metrics_reg(y_tr, pred_tr)
    m_va = metrics_reg(y_va, pred_va)
    m_te = metrics_reg(y_te, pred_te)

    out = {
        "best_iteration": best_it,
        "n_features": int(len(feat_list)),
        "train": m_tr,
        "valid": m_va,
        "test":  m_te,
        "gapTT_r2": float(m_tr["r2"] - m_te["r2"]) if np.isfinite(m_tr["r2"]) and np.isfinite(m_te["r2"]) else np.nan,
    }
    return bst, out, (pred_tr, pred_va, pred_te)

def save_gain_importance(bst: xgb.Booster, feat_list: list[str], out_csv: str) -> None:
    score = bst.get_score(importance_type="gain")
    gains = pd.Series({f: float(score.get(f, 0.0)) for f in feat_list}).sort_values(ascending=False)
    gains.rename("gain").to_csv(out_csv, header=True)

def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, out_png: str, title: str) -> None:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[m]
    yp = y_pred[m]
    if yt.size == 0:
        print(f"[WARN] {title}: no finite points to plot.")
        return

    plt.figure(figsize=(6, 6), dpi=150)
    plt.scatter(yt, yp, s=10, alpha=0.4)

    mn = float(min(np.min(yt), np.min(yp)))
    mx = float(max(np.max(yt), np.max(yp)))
    plt.plot([mn, mx], [mn, mx], linewidth=1.0)

    r2 = r2_score(yt, yp)
    plt.title(f"{title}\nR2={r2:.4f}  N={yt.size}")
    plt.xlabel("True")
    plt.ylabel("Pred")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


# =========================
# 2) MAIN
# =========================
def main():
    print("[LOAD] reading parquet ...")
    df = pd.read_parquet(FLASH_PARQ)

    # ----------------------
    # 2.1 Latitude filter
    # ----------------------
    if LAT_COL not in df.columns:
        raise KeyError(f"[ERROR] LAT_COL '{LAT_COL}' not found in dataframe columns.")
    if TARGET_COL not in df.columns:
        raise KeyError(f"[ERROR] TARGET_COL '{TARGET_COL}' not found in dataframe columns.")

    n_before = len(df)
    df = df[df[LAT_COL].abs() <= LAT_MAX].reset_index(drop=True)
    n_after_lat = len(df)
    print(f"[LAT FILTER] |{LAT_COL}| <= {LAT_MAX}° : {n_before} -> {n_after_lat}")

    if n_after_lat < 100:
        raise RuntimeError(f"[ERROR] Too few samples after lat filter: {n_after_lat}")

    # ----------------------
    # 2.2 Target finite filter (critical for stability)
    # ----------------------
    y_raw = df[TARGET_COL].to_numpy(dtype="float64")
    m_y = np.isfinite(y_raw)
    n_after_yfinite = int(m_y.sum())
    if n_after_yfinite < len(df):
        print(f"[WARN] drop rows with non-finite {TARGET_COL}: {len(df)} -> {n_after_yfinite}")
        df = df.loc[m_y].reset_index(drop=True)
        y_raw = df[TARGET_COL].to_numpy(dtype="float64")

    if len(df) < 100:
        raise RuntimeError(f"[ERROR] Too few samples after target finite filter: {len(df)}")

    # ----------------------
    # 2.3 Build features
    # ----------------------
    base_feats = build_feature_list(df)
    if len(base_feats) == 0:
        raise RuntimeError("[ERROR] No usable numeric feature columns found after filtering.")
    print(f"[INFO] base features = {len(base_feats)}")

    # ----------------------
    # 2.4 Split
    # ----------------------
    idx_tr, idx_va, idx_te = split_train_val_test(len(df))

    y = y_raw  # already float64 finite
    X = df[base_feats]  # DO NOT dropna; XGB handles NaN

    X_tr, y_tr = X.iloc[idx_tr], y[idx_tr]
    X_va, y_va = X.iloc[idx_va], y[idx_va]
    X_te, y_te = X.iloc[idx_te], y[idx_te]

    # ----------------------
    # 2.5 Params & output dirs
    # ----------------------
    params = apply_reg_scale(RLO_PARAMS, REG_SCALE)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(
        OUT_ROOT,
        f"FLASH_topk{TOPK}_Rlo_reg{REG_SCALE}_seed{SEED}_lat{int(LAT_MAX)}_{ts}"
    )
    model_dir = os.path.join(out_dir, "models")
    pred_dir  = os.path.join(out_dir, "predictions")
    imp_dir   = os.path.join(out_dir, "importance")
    feat_dir  = os.path.join(out_dir, "feature_lists")
    fig_dir   = os.path.join(out_dir, "figs")

    for p in [out_dir, model_dir, pred_dir, imp_dir, feat_dir, fig_dir]:
        ensure_dir(p)

    # ----------------------
    # 2.6 Rank features by gain (quick model on all features)
    # ----------------------
    ranked = fit_rank_features_by_gain(X_tr, y_tr, X_va, y_va, params)
    feat_list = ranked[:TOPK]
    print(f"[INFO] topk={TOPK} selected (ranked by gain).")

    feat_path = os.path.join(feat_dir, f"FLASH_topk{TOPK}_features.txt")
    with open(feat_path, "w", encoding="utf-8") as f:
        f.write("\n".join(feat_list))

    # ----------------------
    # 2.7 Train final model on TopK
    # ----------------------
    bst, info, (pred_tr, pred_va, pred_te) = train_final(
        X_tr, y_tr, X_va, y_va, X_te, y_te, feat_list, params
    )

    # Save model
    model_path = os.path.join(model_dir, f"FLASH_topk{TOPK}_reg{REG_SCALE}.json")
    bst.save_model(model_path)

    # Save importance
    gain_csv = os.path.join(imp_dir, f"FLASH_topk{TOPK}_reg{REG_SCALE}_gain.csv")
    save_gain_importance(bst, feat_list, gain_csv)

    # Save predictions
    pd.DataFrame({"y_true": y_tr, "y_pred": pred_tr}).to_csv(
        os.path.join(pred_dir, f"FLASH_topk{TOPK}_pred_train.csv"), index=False
    )
    pd.DataFrame({"y_true": y_va, "y_pred": pred_va}).to_csv(
        os.path.join(pred_dir, f"FLASH_topk{TOPK}_pred_valid.csv"), index=False
    )
    pd.DataFrame({"y_true": y_te, "y_pred": pred_te}).to_csv(
        os.path.join(pred_dir, f"FLASH_topk{TOPK}_pred_test.csv"), index=False
    )

    # Optional plots
    if PLOT_SCATTER:
        plot_scatter(y_tr, pred_tr, os.path.join(fig_dir, "scatter_train.png"), "FLASH Train")
        plot_scatter(y_va, pred_va, os.path.join(fig_dir, "scatter_valid.png"), "FLASH Valid")
        plot_scatter(y_te, pred_te, os.path.join(fig_dir, "scatter_test.png"),  "FLASH Test")

    # ----------------------
    # 2.8 Summary
    # ----------------------
    summary = {
        "target": "FLASH",
        "objective": params["objective"],
        "preset": "Rlo",
        "reg_scale": REG_SCALE,
        "topk": TOPK,
        "nfeat": info["n_features"],
        "best_iter": info["best_iteration"],
        "train": info["train"],
        "valid": info["valid"],
        "test":  info["test"],
        "gapTT_r2": info["gapTT_r2"],
        "lat_filter": {
            "lat_col": LAT_COL,
            "lat_max": LAT_MAX,
            "n_before": int(n_before),
            "n_after_lat": int(n_after_lat),
            "n_after_yfinite": int(len(df)),
        },
        "paths": {
            "out_dir": out_dir,
            "model_path": model_path,
            "feature_list": feat_path,
            "gain_csv": gain_csv,
            "pred_dir": pred_dir,
            "fig_dir": fig_dir,
        },
        "split": {
            "seed": SEED,
            "test_size": TEST_SIZE,
            "val_size": VAL_SIZE,
            "n_train": int(len(idx_tr)),
            "n_valid": int(len(idx_va)),
            "n_test":  int(len(idx_te)),
        },
        "notes": "Strict-land parquet as input. Lat filter applied BEFORE split. Random split, no lat/lon, no dropna. Target non-finite rows removed. Metrics/plots computed on finite pairs only."
    }

    with open(os.path.join(out_dir, "best_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame([{
        "target": "FLASH",
        "objective": params["objective"],
        "preset": "Rlo",
        "reg_scale": REG_SCALE,
        "topk": TOPK,
        "nfeat": info["n_features"],
        "best_iter": info["best_iteration"],
        "train_r2": info["train"]["r2"],
        "valid_r2": info["valid"]["r2"],
        "test_r2":  info["test"]["r2"],
        "gapTT_r2": info["gapTT_r2"],
        "train_rmse": info["train"]["rmse"],
        "valid_rmse": info["valid"]["rmse"],
        "test_rmse":  info["test"]["rmse"],
        "train_mae": info["train"]["mae"],
        "valid_mae": info["valid"]["mae"],
        "test_mae":  info["test"]["mae"],
        "train_n": info["train"]["n"],
        "valid_n": info["valid"]["n"],
        "test_n":  info["test"]["n"],
        "lat_max": LAT_MAX,
        "n_before": int(n_before),
        "n_after_lat": int(n_after_lat),
        "n_after_yfinite": int(len(df)),
        "model_path": model_path,
        "feature_list_path": feat_path,
        "gain_path": gain_csv,
        "out_dir": out_dir,
    }]).to_csv(os.path.join(out_dir, "metrics_flash_final.csv"), index=False)

    print("[DONE] Saved to:", out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
