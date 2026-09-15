# Run from repository root: python -m modeling.miroc6_02_train

# ============================================================
# Step 2. Train and save final MIROC6 XGBoost models
#         reproduced from best no-season tuning workflow
#
# 功能：
#   1. 按调参代码逻辑读取 MIROC6 historical features + labels_land_strict
#   2. 只使用 no-season features:
#        - elev
#        - env_* 中不包含 _DJF_ / _MAM_ / _JJA_ / _SON_ 的变量
#   3. 对当前目标变量 + no-season features 执行 finite/dropna 筛选
#   4. 读取 best_params_FLASH.json / best_params_HAIL.json
#   5. 使用调参时的 baseline gain importance CSV 选择 TopK 特征
#   6. 使用 SEED=42 进行 train / valid / test 划分
#   7. 用 TopK 特征训练最终 FLASH / HAIL XGBoost Booster
#   8. 保存 Booster 模型、feature list、预测结果、gain importance、metrics
#
# 关键说明：
#   - 输出路径保持不变：[configured path; see config.json]
#   - 不剔除 elev > 2000 m
#   - 不做目标变量 log transform
#   - 不重新用 all-feature 模型排序，而是使用调参时的 baseline gain CSV
#   - metrics 和 prediction CSV 中的 y_pred 使用 clip_negative 后的结果
#   - Booster 模型本身保存原始 XGBoost 模型；后续预测建议同样 clip 到 >=0
# ============================================================

from __future__ import annotations

from publication_config import resource_path

import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ============================================================
# 0. Path and global configuration
# ============================================================

MODEL_NAME = "MIROC6"

BASE_DIR = Path(resource_path('models', 'MIROC6'))

# 保留这个路径，便于记录；但为了严格复现调参代码，
# 本脚本实际训练时直接读取 FEATURE_HIST + LABEL_STRICT。
TRAIN_TABLE = (
    BASE_DIR
    / "feature"
    / "ml_train_historical"
    / "train_table_flash_hail_withSRTM_strictLand.parquet"
)

FEATURE_HIST = (
    BASE_DIR
    / "feature"
    / "ml_train_historical"
    / "features_historical_1995_2014.parquet"
)

LABEL_STRICT = (
    BASE_DIR
    / "feature"
    / "ml_train_historical"
    / "labels_land_strict.parquet"
)

BEST_FLASH_JSON = BASE_DIR / "TUNING_xgb_noSeason" / "best_params_FLASH.json"
BEST_HAIL_JSON = BASE_DIR / "TUNING_xgb_noSeason" / "best_params_HAIL.json"

BASELINE_FLASH_GAIN = (
    BASE_DIR
    / "RUNS_xgb_random_nolog_roi63_noSeason"
    / "FLASH_20260323_234226"
    / "feature_importance_gain.csv"
)

BASELINE_HAIL_GAIN = (
    BASE_DIR
    / "RUNS_xgb_random_nolog_roi63_noSeason"
    / "HAIL_20260323_234235"
    / "feature_importance_gain.csv"
)

# 输出路径不变
OUT_ROOT = BASE_DIR / "ml_xgboost_result"
OUT_DIR = OUT_ROOT / 'FINAL_XGB'

MODEL_DIR = OUT_DIR / "models"
FEATURE_DIR = OUT_DIR / "feature_lists"
PRED_DIR = OUT_DIR / "predictions"
IMP_DIR = OUT_DIR / "importance"
METRIC_DIR = OUT_DIR / "metrics"

for p in [OUT_DIR, MODEL_DIR, FEATURE_DIR, PRED_DIR, IMP_DIR, METRIC_DIR]:
    p.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. Configuration matched to tuning code
# ============================================================

SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

LAT_COL = "Latitude"
LON_COL = "Longitude"
LAND_COL = "Land_Sea"

TARGET_FLASH = "LISOTD_Flash"
TARGET_HAIL = "ni_HailPF"

ELEV_COL = "elev"

LAT_MAX = 63.0

NUM_BOOST_ROUND = 3000
EARLY_STOPPING_ROUNDS = 100

CLIP_NEGATIVE_PRED = True

EXPECTED_NOSEASON_FEATURES = 183
EXPECTED_USABLE_ROWS = 2592

BASE_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "seed": SEED,
}


# ============================================================
# 2. Utility functions
# ============================================================

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def metrics_reg(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute regression metrics on finite y_true/y_pred pairs.
    y_pred 应该已经按调参逻辑完成 clip_negative。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    m = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[m]
    yp = y_pred[m]

    if yt.size == 0:
        return {
            "r2": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "bias": np.nan,
            "n": 0,
        }

    return {
        "r2": float(r2_score(yt, yp)),
        "rmse": float(rmse(yt, yp)),
        "mae": float(mean_absolute_error(yt, yp)),
        "bias": float(np.mean(yp - yt)),
        "n": int(len(yt)),
    }


def choose_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    与调参代码完全一致：
      - elev 作为特征保留；
      - 只保留 env_ 开头的非季节统计特征；
      - 剔除 _DJF_ / _MAM_ / _JJA_ / _SON_。
    """
    out = []

    for c in df.columns:
        if c == ELEV_COL:
            out.append(c)
            continue

        if not str(c).startswith("env_"):
            continue

        if (
            ("_DJF_" in c) or
            ("_MAM_" in c) or
            ("_JJA_" in c) or
            ("_SON_" in c)
        ):
            continue

        out.append(c)

    return sorted(out)


def load_train_table_like_tuning(target_col: str):
    """
    按调参代码的逻辑构建当前目标变量的训练表。

    注意：
      调参代码不是读取 Step 1 的 train_table；
      而是重新读取 features_historical_1995_2014.parquet 与 labels_land_strict.parquet，
      然后按 target + no-season features dropna。
    """
    if not FEATURE_HIST.exists():
        raise FileNotFoundError(FEATURE_HIST)

    if not LABEL_STRICT.exists():
        raise FileNotFoundError(LABEL_STRICT)

    print("[READ] historical features:", FEATURE_HIST)
    df_feat = pd.read_parquet(FEATURE_HIST)

    print("[READ] strict-land labels:", LABEL_STRICT)
    df_lab = pd.read_parquet(LABEL_STRICT)

    required_lab_cols = [LAT_COL, LON_COL, target_col]
    missing_lab = [c for c in required_lab_cols if c not in df_lab.columns]
    if missing_lab:
        raise KeyError(f"Missing columns in label table: {missing_lab}")

    required_feat_cols = [LAT_COL, LON_COL]
    missing_feat = [c for c in required_feat_cols if c not in df_feat.columns]
    if missing_feat:
        raise KeyError(f"Missing columns in feature table: {missing_feat}")

    if ELEV_COL not in df_feat.columns:
        raise KeyError(
            f"Missing elevation column '{ELEV_COL}' in feature table. "
            "The tuning workflow allowed elevation to enter the feature set."
        )

    # 与调参代码一致：df_lab.merge(df_feat, ...)
    df = df_lab[[LAT_COL, LON_COL, target_col]].copy().merge(
        df_feat,
        on=[LAT_COL, LON_COL],
        how="inner",
    )

    n_before_lat = len(df)

    # 调参路径名中有 roi63；保留 |lat| <= 63 安全筛选。
    df = df.loc[df[LAT_COL].abs() <= LAT_MAX].copy()
    n_after_lat = len(df)

    feat_cols = choose_feature_columns(df)

    keep_cols = [LAT_COL, LON_COL, target_col] + feat_cols
    df = df[keep_cols].copy()

    # 与调参代码一致：replace inf -> nan，然后 dropna(target + feat_cols)
    df = df.replace([np.inf, -np.inf], np.nan)

    for c in [target_col] + feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    n_before_dropna = len(df)
    df = df.dropna(subset=[target_col] + feat_cols).reset_index(drop=True)
    n_after_dropna = len(df)

    print("\n[DATA FILTER]")
    print("Rows before lat filter :", n_before_lat)
    print("Rows after lat filter  :", n_after_lat)
    print("Rows before dropna     :", n_before_dropna)
    print("Rows after dropna      :", n_after_dropna)
    print("Selected no-season features:", len(feat_cols))

    if len(feat_cols) != EXPECTED_NOSEASON_FEATURES:
        print(
            f"[WARNING] no-season feature count = {len(feat_cols)}, "
            f"expected = {EXPECTED_NOSEASON_FEATURES}. "
            "Please check feature schema."
        )

    if n_after_dropna != EXPECTED_USABLE_ROWS:
        print(
            f"[WARNING] usable rows = {n_after_dropna}, "
            f"expected from tuning output = {EXPECTED_USABLE_ROWS}. "
            "If Step 1 or input files changed, metrics may differ."
        )

    return df, feat_cols, {
        "n_before_lat": int(n_before_lat),
        "n_after_lat": int(n_after_lat),
        "n_before_dropna": int(n_before_dropna),
        "n_after_dropna": int(n_after_dropna),
    }


def split_train_valid_test(df: pd.DataFrame, feature_cols: list[str], target_col: str):
    """
    与调参代码一致：
      先划分 train_valid / test；
      再从 train_valid 中划分 train / valid。
    返回 X/y 以及对应原始 index。
    """
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    idx_all = np.arange(len(df))

    X_train_valid, X_test, y_train_valid, y_test, idx_train_valid, idx_test = train_test_split(
        X,
        y,
        idx_all,
        test_size=TEST_SIZE,
        random_state=SEED,
        shuffle=True,
    )

    valid_ratio_inside = VAL_SIZE / (1.0 - TEST_SIZE)

    X_train, X_valid, y_train, y_valid, idx_train, idx_valid = train_test_split(
        X_train_valid,
        y_train_valid,
        idx_train_valid,
        test_size=valid_ratio_inside,
        random_state=SEED,
        shuffle=True,
    )

    return (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
        idx_train,
        idx_valid,
        idx_test,
    )


def load_best_params(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_xgb_params(best: dict) -> dict:
    """
    与调参代码保持一致：
      BASE_PARAMS + best params。
    """
    params = BASE_PARAMS.copy()

    update = {
        "eta": float(best.get("eta", best.get("learning_rate", 0.03))),
        "max_depth": int(best.get("max_depth", 5)),
        "min_child_weight": float(best.get("min_child_weight", 1.0)),
        "subsample": float(best.get("subsample", 1.0)),
        "colsample_bytree": float(best.get("colsample_bytree", 1.0)),
        "lambda": float(best.get("lambda", best.get("reg_lambda", 1.0))),
        "alpha": float(best.get("alpha", best.get("reg_alpha", 0.0))),
    }

    # 如果 best json 中有 gamma，则保留；否则不额外添加，避免偏离调参代码。
    if "gamma" in best:
        update["gamma"] = float(best["gamma"])

    params.update(update)

    return params


def parse_topk(best: dict):
    if "topK" in best:
        val = best["topK"]
    elif "topk" in best:
        val = best["topk"]
    elif "n_features" in best:
        val = best["n_features"]
    else:
        raise KeyError("Cannot find topK / topk / n_features in best_params JSON.")

    if isinstance(val, str) and val.upper() == "ALL":
        return None

    if pd.isna(val):
        return None

    return int(val)


def load_top_features(
    gain_csv: Path,
    all_feature_cols: list[str],
    topk,
) -> list[str]:
    """
    与调参代码 load_top_features 保持一致：
      1. 读取 baseline feature_importance_gain.csv；
      2. 只保留存在于 all_feature_cols 中的特征；
      3. 确保 elev 可以进入；
      4. 若 topk 为数值，则取前 topk，并在 elev 不在其中时追加 elev。
    """
    if not gain_csv.exists():
        raise FileNotFoundError(gain_csv)

    df_gain = pd.read_csv(gain_csv)

    if "feature" not in df_gain.columns:
        raise KeyError(f"{gain_csv} missing column: feature")

    gain_feats = [
        f for f in df_gain["feature"].astype(str).tolist()
        if f in all_feature_cols
    ]

    if ELEV_COL in all_feature_cols and ELEV_COL not in gain_feats:
        gain_feats.append(ELEV_COL)

    if topk is None:
        out = [c for c in all_feature_cols if c in set(gain_feats) or c == ELEV_COL]

        if len(out) < 10:
            out = all_feature_cols.copy()

        return out

    out = gain_feats[:int(topk)]

    if ELEV_COL in all_feature_cols and ELEV_COL not in out:
        out.append(ELEV_COL)

    return out


def train_final_model_like_tuning(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    y_test: pd.Series,
    feature_cols: list[str],
    params: dict,
):
    """
    与调参代码 run_one_trial 训练逻辑一致：
      - num_boost_round = 3000
      - early_stopping_rounds = 100
      - evals = [(dtrain, "train"), (dvalid, "valid")]
      - prediction 使用 iteration_range=(0, best_iteration+1)
      - metrics 前 clip negative prediction
    """
    dtrain = xgb.DMatrix(
        X_train[feature_cols],
        label=y_train,
        feature_names=feature_cols,
    )

    dvalid = xgb.DMatrix(
        X_valid[feature_cols],
        label=y_valid,
        feature_names=feature_cols,
    )

    dtest = xgb.DMatrix(
        X_test[feature_cols],
        label=y_test,
        feature_names=feature_cols,
    )

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )

    best_iteration = int(model.best_iteration)
    it_rng = (0, best_iteration + 1)

    pred_train_raw = model.predict(dtrain, iteration_range=it_rng)
    pred_valid_raw = model.predict(dvalid, iteration_range=it_rng)
    pred_test_raw = model.predict(dtest, iteration_range=it_rng)

    if CLIP_NEGATIVE_PRED:
        pred_train = np.clip(pred_train_raw, 0, None)
        pred_valid = np.clip(pred_valid_raw, 0, None)
        pred_test = np.clip(pred_test_raw, 0, None)
    else:
        pred_train = pred_train_raw
        pred_valid = pred_valid_raw
        pred_test = pred_test_raw

    metrics = {
        "best_iteration": best_iteration,
        "n_features": int(len(feature_cols)),
        "train": metrics_reg(y_train, pred_train),
        "valid": metrics_reg(y_valid, pred_valid),
        "test": metrics_reg(y_test, pred_test),
    }

    metrics["overfit_gap_train_test_r2"] = (
        metrics["train"]["r2"] - metrics["test"]["r2"]
        if np.isfinite(metrics["train"]["r2"]) and np.isfinite(metrics["test"]["r2"])
        else np.nan
    )

    metrics["valid_test_r2_diff"] = (
        metrics["valid"]["r2"] - metrics["test"]["r2"]
        if np.isfinite(metrics["valid"]["r2"]) and np.isfinite(metrics["test"]["r2"])
        else np.nan
    )

    preds = {
        "train_raw": pred_train_raw,
        "valid_raw": pred_valid_raw,
        "test_raw": pred_test_raw,
        "train": pred_train,
        "valid": pred_valid,
        "test": pred_test,
    }

    return model, metrics, preds


def save_feature_list(features: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(features))


def save_gain_importance(bst: xgb.Booster, features: list[str], path: Path) -> None:
    score = bst.get_score(importance_type="gain")

    df_gain = pd.DataFrame({
        "feature": features,
        "gain": [float(score.get(f, 0.0)) for f in features],
    }).sort_values("gain", ascending=False)

    df_gain.to_csv(path, index=False, encoding="utf-8-sig")


def save_prediction_csv(
    target_name: str,
    split_name: str,
    df_meta: pd.DataFrame,
    idx: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_raw: np.ndarray,
):
    out = pd.DataFrame({
        LAT_COL: df_meta.iloc[idx][LAT_COL].to_numpy(),
        LON_COL: df_meta.iloc[idx][LON_COL].to_numpy(),
        "y_true": np.asarray(y_true, dtype=float),
        "y_pred": np.asarray(y_pred, dtype=float),
        "y_pred_raw": np.asarray(y_pred_raw, dtype=float),
    })

    out_path = PRED_DIR / f"{target_name}_pred_{split_name}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    return out_path


def compare_with_best_json(metrics: dict, best: dict) -> dict:
    """
    保存当前最终训练结果与调参 best json 的差异。
    如果数据顺序、baseline gain、XGBoost 版本完全一致，结果应非常接近。
    """
    out = {}

    mapping = {
        "train_r2": ("train", "r2"),
        "valid_r2": ("valid", "r2"),
        "test_r2": ("test", "r2"),
        "train_rmse": ("train", "rmse"),
        "valid_rmse": ("valid", "rmse"),
        "test_rmse": ("test", "rmse"),
        "train_mae": ("train", "mae"),
        "valid_mae": ("valid", "mae"),
        "test_mae": ("test", "mae"),
    }

    for best_key, metric_path in mapping.items():
        if best_key not in best:
            continue

        split, metric_name = metric_path
        current_value = metrics[split][metric_name]
        best_value = float(best[best_key])

        out[f"{best_key}_from_best_json"] = best_value
        out[f"{best_key}_current"] = current_value
        out[f"{best_key}_diff_current_minus_best"] = current_value - best_value

    if "best_iteration" in best:
        out["best_iteration_from_best_json"] = int(best["best_iteration"])
        out["best_iteration_current"] = int(metrics["best_iteration"])
        out["best_iteration_diff_current_minus_best"] = (
            int(metrics["best_iteration"]) - int(best["best_iteration"])
        )

    return out


# ============================================================
# 3. Main training function for one target
# ============================================================

def run_one_target(
    target_name: str,
    target_col: str,
    best_json: Path,
    baseline_gain_csv: Path,
) -> dict:
    print("\n" + "=" * 100)
    print(f"[TRAIN] {MODEL_NAME} {target_name}")
    print("=" * 100)

    df, all_feature_cols, data_filter_info = load_train_table_like_tuning(target_col)

    print("\n[FEATURE]")
    print("All no-season candidate features:", len(all_feature_cols))
    print("First 20 no-season features:")
    print(all_feature_cols[:20])

    best = load_best_params(best_json)
    params = make_xgb_params(best)
    topk_requested = parse_topk(best)

    selected_features = load_top_features(
        gain_csv=baseline_gain_csv,
        all_feature_cols=all_feature_cols,
        topk=topk_requested,
    )

    print("\n[BEST PARAMS]")
    print("Best JSON:", best_json)
    print("Target in JSON:", best.get("target", None))
    print("topK requested:", topk_requested)
    print("n_features in JSON:", best.get("n_features", None))
    print("selected features actual:", len(selected_features))
    print("baseline gain CSV:", baseline_gain_csv)
    print("best_iteration in JSON:", best.get("best_iteration", None))

    print("\n[XGB PARAMS USED]")
    print(json.dumps(params, indent=2))

    if len(selected_features) == 0:
        raise RuntimeError("No selected features.")

    missing_selected = [f for f in selected_features if f not in all_feature_cols]
    if missing_selected:
        raise KeyError(f"Selected features not found in training table: {missing_selected[:20]}")

    (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
        idx_train,
        idx_valid,
        idx_test,
    ) = split_train_valid_test(df, selected_features, target_col)

    print("\n[SPLIT]")
    print("train:", len(idx_train))
    print("valid:", len(idx_valid))
    print("test :", len(idx_test))

    print("\n[SELECTED FEATURES]")
    print("First 30 selected features:")
    print(selected_features[:30])
    print("Last 20 selected features:")
    print(selected_features[-20:])

    print("\n[FINAL] Training final model using selected baseline-gain TopK features ...")

    bst, metrics, preds = train_final_model_like_tuning(
        X_train=X_train,
        X_valid=X_valid,
        X_test=X_test,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
        feature_cols=selected_features,
        params=params,
    )

    comparison = compare_with_best_json(metrics, best)

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    model_path = MODEL_DIR / f"{target_name}_model.json"
    feature_path = FEATURE_DIR / f"{target_name}_features.txt"
    all_feature_path = FEATURE_DIR / f"{target_name}_all_noSeason_features.txt"
    ranked_path = FEATURE_DIR / f"{target_name}_ranked_baseline_gain_features.txt"
    gain_path = IMP_DIR / f"{target_name}_gain.csv"
    summary_path = METRIC_DIR / f"{target_name}_summary.json"

    bst.save_model(model_path)

    save_feature_list(selected_features, feature_path)
    save_feature_list(all_feature_cols, all_feature_path)

    # 保存 baseline gain 排序后真正参与筛选的特征顺序
    baseline_order = load_top_features(
        gain_csv=baseline_gain_csv,
        all_feature_cols=all_feature_cols,
        topk=None,
    )
    save_feature_list(baseline_order, ranked_path)

    save_gain_importance(bst, selected_features, gain_path)

    pred_train_path = save_prediction_csv(
        target_name=target_name,
        split_name="train",
        df_meta=df,
        idx=idx_train,
        y_true=y_train.to_numpy(dtype=float),
        y_pred=preds["train"],
        y_pred_raw=preds["train_raw"],
    )

    pred_valid_path = save_prediction_csv(
        target_name=target_name,
        split_name="valid",
        df_meta=df,
        idx=idx_valid,
        y_true=y_valid.to_numpy(dtype=float),
        y_pred=preds["valid"],
        y_pred_raw=preds["valid_raw"],
    )

    pred_test_path = save_prediction_csv(
        target_name=target_name,
        split_name="test",
        df_meta=df,
        idx=idx_test,
        y_true=y_test.to_numpy(dtype=float),
        y_pred=preds["test"],
        y_pred_raw=preds["test_raw"],
    )

    split_index_path = PRED_DIR / f"{target_name}_split_indices.npz"
    np.savez_compressed(
        split_index_path,
        idx_train=idx_train.astype(np.int64),
        idx_valid=idx_valid.astype(np.int64),
        idx_test=idx_test.astype(np.int64),
    )

    summary = {
        "model_name": MODEL_NAME,
        "target_name": target_name,
        "target_col": target_col,
        "seed": SEED,
        "test_size": TEST_SIZE,
        "val_size": VAL_SIZE,
        "num_boost_round": NUM_BOOST_ROUND,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "clip_negative_pred_for_metrics_and_csv": CLIP_NEGATIVE_PRED,
        "lat_filter": {
            "lat_col": LAT_COL,
            "lat_max": LAT_MAX,
        },
        "data_filter_info": data_filter_info,
        "feature_policy": {
            "use_noSeason_features": True,
            "exclude_seasonal_features": ["_DJF_", "_MAM_", "_JJA_", "_SON_"],
            "include_elevation": ELEV_COL in all_feature_cols,
            "all_noSeason_feature_count": int(len(all_feature_cols)),
            "selected_feature_count": int(len(selected_features)),
            "topk_requested_from_best_json": None if topk_requested is None else int(topk_requested),
            "baseline_gain_csv": str(baseline_gain_csv),
            "feature_selection_method": "baseline_gain_csv_topK_as_used_in_tuning",
        },
        "best_params_json": str(best_json),
        "best_params_loaded": best,
        "xgb_params_used": params,
        "metrics": metrics,
        "comparison_with_best_params_json": comparison,
        "paths": {
            "model_path": str(model_path),
            "feature_list_path": str(feature_path),
            "all_noSeason_feature_list_path": str(all_feature_path),
            "ranked_baseline_gain_feature_path": str(ranked_path),
            "gain_path": str(gain_path),
            "pred_train_path": str(pred_train_path),
            "pred_valid_path": str(pred_valid_path),
            "pred_test_path": str(pred_test_path),
            "split_index_path": str(split_index_path),
        },
        "notes": (
            "Final model training follows the no-season tuning workflow. "
            "Features were selected from baseline gain CSV rather than from a newly trained ranking model. "
            "Rows with missing values in the current target or selected no-season candidate features were removed before split. "
            "No elevation >2000 m filtering was applied. "
            "Predictions in CSV and metrics are clipped to >=0, consistent with the tuning script. "
            "For future prediction, use the saved Booster and feature list, then clip negative predictions if required."
        ),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n[RESULT]")
    print(f"{target_name} best_iteration:", metrics["best_iteration"])
    print(f"{target_name} n_features:", metrics["n_features"])
    print(f"{target_name} train r2/rmse:", metrics["train"]["r2"], metrics["train"]["rmse"])
    print(f"{target_name} valid r2/rmse:", metrics["valid"]["r2"], metrics["valid"]["rmse"])
    print(f"{target_name} test  r2/rmse:", metrics["test"]["r2"], metrics["test"]["rmse"])
    print(f"{target_name} overfit gap train-test R2:", metrics["overfit_gap_train_test_r2"])

    print("\n[COMPARE WITH BEST JSON]")
    for k, v in comparison.items():
        print(f"{k}: {v}")

    print("\n[SAVED]")
    print("Model       :", model_path)
    print("Feature list:", feature_path)
    print("Gain        :", gain_path)
    print("Summary     :", summary_path)
    print("Pred test   :", pred_test_path)

    return summary


# ============================================================
# 4. Run FLASH and HAIL
# ============================================================

if __name__ == "__main__":
    t0 = time.time()

    flash_summary = run_one_target(
        target_name="FLASH",
        target_col=TARGET_FLASH,
        best_json=BEST_FLASH_JSON,
        baseline_gain_csv=BASELINE_FLASH_GAIN,
    )

    hail_summary = run_one_target(
        target_name="HAIL",
        target_col=TARGET_HAIL,
        best_json=BEST_HAIL_JSON,
        baseline_gain_csv=BASELINE_HAIL_GAIN,
    )

    # Save combined metrics table.
    rows = []

    for s in [flash_summary, hail_summary]:
        m = s["metrics"]
        c = s["comparison_with_best_params_json"]
        fp = s["feature_policy"]

        rows.append({
            "model_name": s["model_name"],
            "target_name": s["target_name"],
            "target_col": s["target_col"],
            "seed": s["seed"],

            "topk_requested": fp["topk_requested_from_best_json"],
            "n_features": m["n_features"],
            "all_noSeason_feature_count": fp["all_noSeason_feature_count"],

            "best_iteration": m["best_iteration"],

            "train_r2": m["train"]["r2"],
            "valid_r2": m["valid"]["r2"],
            "test_r2": m["test"]["r2"],

            "train_rmse": m["train"]["rmse"],
            "valid_rmse": m["valid"]["rmse"],
            "test_rmse": m["test"]["rmse"],

            "train_mae": m["train"]["mae"],
            "valid_mae": m["valid"]["mae"],
            "test_mae": m["test"]["mae"],

            "train_bias": m["train"]["bias"],
            "valid_bias": m["valid"]["bias"],
            "test_bias": m["test"]["bias"],

            "train_n": m["train"]["n"],
            "valid_n": m["valid"]["n"],
            "test_n": m["test"]["n"],

            "overfit_gap_train_test_r2": m["overfit_gap_train_test_r2"],
            "valid_test_r2_diff": m["valid_test_r2_diff"],

            "best_json_test_r2": c.get("test_r2_from_best_json", np.nan),
            "current_minus_best_json_test_r2": c.get("test_r2_diff_current_minus_best", np.nan),
            "best_json_valid_r2": c.get("valid_r2_from_best_json", np.nan),
            "current_minus_best_json_valid_r2": c.get("valid_r2_diff_current_minus_best", np.nan),
            "best_json_best_iteration": c.get("best_iteration_from_best_json", np.nan),
            "current_minus_best_json_best_iteration": c.get("best_iteration_diff_current_minus_best", np.nan),

            "model_path": s["paths"]["model_path"],
            "feature_list_path": s["paths"]["feature_list_path"],
            "gain_path": s["paths"]["gain_path"],
            "baseline_gain_csv": fp["baseline_gain_csv"],
        })

    metrics_df = pd.DataFrame(rows)
    metrics_path = METRIC_DIR / "metrics_xgb_final_seed42.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("[ALL DONE] Final MIROC6 XGBoost models saved.")
    print("=" * 100)
    print("Output directory:", OUT_DIR)
    print("Combined metrics:", metrics_path)
    print("\nMetrics preview:")
    print(metrics_df)

    print(f"\nElapsed time: {(time.time() - t0) / 60:.2f} min")
