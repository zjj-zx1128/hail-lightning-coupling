# Purpose: canesm5 07 standardize.
# Source: cmip6_figure_CanESM5_XGB_plot_latest_checked.ipynb; see docs/source_coverage.csv for cell provenance.
# Run order and required inputs: docs/workflow.md.
# Run from repository root: python -m modeling.canesm5_07_standardize

# -*- coding: utf-8 -*-
'\nStep A. Standardize the selected CanESM5 XGBoost outputs for prediction and plotting.\n\nThis cell keeps the original CanESM5 model-selection rule unchanged:\n- FLASH: latest directory matching FLASH_topk*_Rlo_reg*_seed*_lat*_*\n- HAIL : latest directory matching HAIL_topk*_Rlo_reg*_seed*_lat*_*\n\nIt does not retrain or re-select models. It only copies/renames the selected model outputs into\na BCC-style standard directory:\n    [configured path; see config.json]\n'

from __future__ import annotations

from publication_config import resource_path

import glob
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 0. Paths and standard output structure
# ============================================================

MODEL_NAME = "CanESM5"
BASE_DIR = Path(resource_path('models', 'CanESM5'))
OUT_ROOT = BASE_DIR / "ml_xgboost_result"
TRAIN_TABLE = BASE_DIR / "ml_feature_withSRTM" / "train_table_flash_hail_withSRTM_strictLand.parquet"

FINAL_DIR = OUT_ROOT / 'FINAL_XGB'
MODEL_DIR = FINAL_DIR / "models"
FEATURE_DIR = FINAL_DIR / "feature_lists"
PRED_DIR = FINAL_DIR / "predictions"
IMP_DIR = FINAL_DIR / "importance"
METRIC_DIR = FINAL_DIR / "metrics"

for p in [FINAL_DIR, MODEL_DIR, FEATURE_DIR, PRED_DIR, IMP_DIR, METRIC_DIR]:
    p.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. Helper functions
# ============================================================

def parse_timestamp_from_dirname(name: str):
    """Extract the trailing timestamp used by the original CanESM5 training script."""
    m = re.search(r"_(\d{8}_\d{6})$", name)
    return m.group(1) if m else None


def find_latest_run(root: Path, pattern: str) -> Path:
    cands = [Path(p) for p in glob.glob(str(root / pattern)) if Path(p).is_dir()]
    if not cands:
        raise FileNotFoundError(f"No run dirs found: {pattern} under {root}")

    def key(p: Path):
        ts = parse_timestamp_from_dirname(p.name)
        return ts if ts is not None else f"{p.stat().st_mtime:.6f}"

    return sorted(cands, key=key)[-1]


def pick_one(pattern: str) -> Path:
    xs = [Path(p) for p in glob.glob(pattern) if Path(p).is_file()]
    if not xs:
        raise FileNotFoundError(f"No file matches: {pattern}")
    return xs[0]


def read_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_best_iteration(summary: dict):
    for key in ["best_iteration", "best_iter"]:
        if key in summary and summary[key] is not None:
            return int(summary[key])
    metrics = summary.get("metrics", {})
    for key in ["best_iteration", "best_iter"]:
        if key in metrics and metrics[key] is not None:
            return int(metrics[key])
    return None


def copy_file(src: Path, dst: Path):
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[COPY] {src} -> {dst}")


def standardize_gain_csv(src: Path, dst: Path):
    if not src.exists():
        raise FileNotFoundError(src)

    df = pd.read_csv(src)
    cols = list(df.columns)

    if "feature" not in df.columns:
        first = cols[0]
        df = df.rename(columns={first: "feature"})

    if "gain" not in df.columns:
        candidates = [c for c in df.columns if c != "feature"]
        if not candidates:
            raise KeyError(f"Cannot infer gain column from {src}")
        df = df.rename(columns={candidates[0]: "gain"})

    df = df[["feature", "gain"]].copy()
    df["feature"] = df["feature"].astype(str)
    df["gain"] = pd.to_numeric(df["gain"], errors="coerce").fillna(0.0)
    df.to_csv(dst, index=False)
    print(f"[WRITE] standardized gain csv -> {dst}")


def standardize_target(target: str, run_dir: Path):
    target = target.upper()

    model_src = pick_one(str(run_dir / "models" / "*.json"))
    feat_src = pick_one(str(run_dir / "feature_lists" / "*_features.txt"))
    gain_src = pick_one(str(run_dir / "importance" / "*_gain.csv"))
    summary_src = run_dir / "best_summary.json"
    summary = read_json_if_exists(summary_src)

    copy_file(model_src, MODEL_DIR / f"{target}_model.json")
    copy_file(feat_src, FEATURE_DIR / f"{target}_features.txt")
    standardize_gain_csv(gain_src, IMP_DIR / f"{target}_gain.csv")

    # Standardize train/valid/test prediction CSVs.
    for split in ["train", "valid", "test"]:
        pred_src = pick_one(str(run_dir / "predictions" / f"*_{split}.csv"))
        pred_dst = PRED_DIR / f"{target}_pred_{split}.csv"
        dfp = pd.read_csv(pred_src)
        if not {"y_true", "y_pred"}.issubset(dfp.columns):
            raise KeyError(f"{pred_src} must contain y_true and y_pred columns")
        dfp[["y_true", "y_pred"]].to_csv(pred_dst, index=False)
        print(f"[WRITE] {pred_dst}")

    best_iteration = get_best_iteration(summary)
    std_summary = {
        "target": target,
        "model_name": MODEL_NAME,
        "source_run_dir": str(run_dir),
        "source_summary_json": str(summary_src),
        "metrics": {
            "best_iteration": best_iteration,
            "train": summary.get("train", {}),
            "valid": summary.get("valid", {}),
            "test": summary.get("test", {}),
            "gapTT_r2": summary.get("gapTT_r2", None),
        },
        "source_summary": summary,
    }

    with open(METRIC_DIR / f"{target}_summary.json", "w", encoding="utf-8") as f:
        json.dump(std_summary, f, indent=2)

    print(f"[SUMMARY] {target}: best_iteration = {best_iteration}")
    return std_summary


# ============================================================
# 2. Select original CanESM5 latest FLASH/HAIL runs and standardize
# ============================================================

flash_run = find_latest_run(OUT_ROOT, "FLASH_topk*_Rlo_reg*_seed*_lat*_*")
hail_run = find_latest_run(OUT_ROOT, "HAIL_topk*_Rlo_reg*_seed*_lat*_*")

print("=" * 100)
print("[STEP A] Standardize selected CanESM5 XGBoost outputs")
print("=" * 100)
print("Selected FLASH run:", flash_run)
print("Selected HAIL  run:", hail_run)
print("Standard output :", FINAL_DIR)

flash_summary = standardize_target("FLASH", flash_run)
hail_summary = standardize_target("HAIL", hail_run)

meta = {
    "model_name": MODEL_NAME,
    "train_table": str(TRAIN_TABLE),
    "final_dir": str(FINAL_DIR),
    "selected_flash_run": str(flash_run),
    "selected_hail_run": str(hail_run),
    "note": (
        "CanESM5 model selection is unchanged from the original workflow. "
        "This cell only reorganizes the selected latest outputs into a BCC-style standard directory."
    ),
}
with open(FINAL_DIR / "standardization_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print("\n[DONE] Standardized CanESM5 outputs saved to:")
print(FINAL_DIR)
