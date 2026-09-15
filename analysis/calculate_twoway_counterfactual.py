"""Calculate the two-way, 14-parameter counterfactual attribution used by Figure 5.

Run order: after model training and historical/SSP585 feature preparation, before
``analysis.regional_parameter_attribution`` and
``figures.figure_05_parameter_contributions``.

Run from the repository root:
    python -m analysis.calculate_twoway_counterfactual

Source: ``cmip6_pipeline.py`` supplied by the author. Only the
``attribute-parameters`` calculation path is retained. The original variable
named ``forward`` implements the manuscript's reverse experiment (historical
state with one future parameter); the original variable named ``backward``
implements the manuscript's forward experiment (future state with one
historical parameter). These established output column names are retained for
compatibility. Their mean, ``two_way``, is invariant to this label order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from publication_config import resource_path


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "attribution_models.json"
PARAMETERS = (
    "mucape", "mucin", "pw", "s06", "h_10c", "h_30c", "dh_10_30",
    "div500", "td_sfc", "theta_e", "cape", "cin", "flh", "k_index",
)


def load_config() -> dict:
    """Load relative model-artifact paths and resolve them below models/."""
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    for model in config["models"].values():
        for key in (
            "historical_features", "strict_land_labels", "flash_model",
            "hail_model", "flash_features", "hail_features",
        ):
            model[key] = resource_path("models", model[key])
        template = model["future_feature_template"]
        model["future_feature_template"] = resource_path("models", template)
    return config


def read_feature_list(path: str) -> list[str]:
    """Read a one-column CSV or line-based list of selected model features."""
    feature_path = Path(path)
    if feature_path.suffix.lower() == ".csv":
        frame = pd.read_csv(feature_path)
        column = "feature" if "feature" in frame.columns else frame.columns[0]
        return frame[column].astype(str).tolist()
    return [
        line.strip()
        for line in feature_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def key_columns(common: dict) -> list[str]:
    return [common["latitude_column"], common["longitude_column"]]


def align_to_strict_land(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    common: dict,
) -> pd.DataFrame:
    """Align each feature table to the same strict-land grid."""
    keys = key_columns(common)
    land = labels[keys].drop_duplicates()
    aligned = land.merge(features, on=keys, how="left", validate="one_to_one")
    if len(aligned) != len(land):
        raise ValueError("Strict-land alignment changed row count.")
    return aligned.sort_values(keys).reset_index(drop=True)


def load_booster(path: str) -> xgb.Booster:
    booster = xgb.Booster()
    booster.load_model(path)
    return booster


def predict(
    booster: xgb.Booster,
    frame: pd.DataFrame,
    features: list[str],
) -> np.ndarray:
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing {len(missing)} model features: {missing[:10]}")
    values = booster.predict(xgb.DMatrix(frame[features], feature_names=features))
    return np.clip(np.asarray(values, dtype=np.float64), 0.0, None)


def feature_parameter(feature: str) -> str | None:
    lower = feature.lower()
    for parameter in PARAMETERS:
        if f"env_{parameter}_" in lower or lower == parameter:
            return parameter
    return None


def parameter_features(features: list[str]) -> dict[str, list[str]]:
    grouped = {parameter: [] for parameter in PARAMETERS}
    for feature in features:
        parameter = feature_parameter(feature)
        if parameter is not None:
            grouped[parameter].append(feature)
    return grouped


def area_weighted_mean(values: np.ndarray, latitudes: pd.Series) -> float:
    valid = np.isfinite(values) & latitudes.notna().to_numpy()
    if not valid.any():
        return float("nan")
    weights = np.cos(np.deg2rad(latitudes.to_numpy()[valid]))
    return float(np.average(values[valid], weights=weights))


def parameter_gain_table(
    model_name: str,
    flash_model: xgb.Booster,
    hail_model: xgb.Booster,
) -> pd.DataFrame:
    """Aggregate fitted-model gain by the same 14 parameter groups."""
    rows = []
    for target, booster in (("flash", flash_model), ("hail", hail_model)):
        scores = booster.get_score(importance_type="gain")
        grouped = {parameter: 0.0 for parameter in PARAMETERS}
        for feature, score in scores.items():
            parameter = feature_parameter(feature)
            if parameter is not None:
                grouped[parameter] += float(score)
        total = sum(grouped.values())
        for parameter, gain in grouped.items():
            rows.append({
                "model": model_name,
                "target": target,
                "parameter": parameter,
                "gain": gain,
                "gain_fraction": gain / total if total else np.nan,
            })
    result = pd.DataFrame(rows)
    result["gain_rank"] = result.groupby("target")["gain"].rank(
        method="average", ascending=False
    )
    return result


def single_parameter_attribution(
    hist: pd.DataFrame,
    future: pd.DataFrame,
    flash_model: xgb.Booster,
    hail_model: xgb.Booster,
    flash_features: list[str],
    hail_features: list[str],
) -> tuple[
    dict[str, dict[str, dict[str, np.ndarray]]],
    dict[str, np.ndarray],
    dict[str, dict[str, np.ndarray]],
]:
    """Run both replacement directions and average them for each parameter."""
    all_features = sorted(set(flash_features) | set(hail_features))
    grouped = parameter_features(all_features)
    empty = [parameter for parameter, columns in grouped.items() if not columns]
    if empty:
        raise ValueError(f"Parameters without model features: {empty}")

    models = {"flash": flash_model, "hail": hail_model}
    feature_lists = {"flash": flash_features, "hail": hail_features}
    baseline = {
        target: predict(models[target], hist, feature_lists[target])
        for target in models
    }
    full_future = {
        target: predict(models[target], future, feature_lists[target])
        for target in models
    }
    total = {
        target: full_future[target] - baseline[target]
        for target in models
    }
    contributions = {
        target: {
            direction: {}
            for direction in ("forward", "backward", "two_way")
        }
        for target in models
    }

    for parameter, columns in grouped.items():
        forward_frame = hist[all_features].copy()
        forward_frame.loc[:, columns] = future[columns].to_numpy()
        backward_frame = future[all_features].copy()
        backward_frame.loc[:, columns] = hist[columns].to_numpy()

        for target in models:
            forward_prediction = predict(
                models[target], forward_frame, feature_lists[target]
            )
            backward_prediction = predict(
                models[target], backward_frame, feature_lists[target]
            )
            forward = forward_prediction - baseline[target]
            backward = full_future[target] - backward_prediction
            contributions[target]["forward"][parameter] = forward
            contributions[target]["backward"][parameter] = backward
            contributions[target]["two_way"][parameter] = (forward + backward) / 2.0
        print(f"  parameter {parameter}")

    residuals = {
        target: {
            direction: total[target]
            - sum(contributions[target][direction].values())
            for direction in ("forward", "backward", "two_way")
        }
        for target in models
    }
    return contributions, total, residuals


def calculate_model(model_name: str, config: dict) -> None:
    """Calculate all five decades and save Figure 5-compatible products."""
    common = config["common"]
    model = config["models"][model_name]
    labels = pd.read_parquet(model["strict_land_labels"])
    keys = key_columns(common)
    hist = align_to_strict_land(
        pd.read_parquet(model["historical_features"]), labels, common
    )
    flash_features = read_feature_list(model["flash_features"])
    hail_features = read_feature_list(model["hail_features"])
    flash_model = load_booster(model["flash_model"])
    hail_model = load_booster(model["hail_model"])

    unmapped = sorted(
        feature
        for feature in set(flash_features) | set(hail_features)
        if feature_parameter(feature) is None and feature.lower() != "elev"
    )
    if unmapped:
        raise ValueError(f"Unmapped dynamic features: {unmapped[:20]}")

    out_dir = (
        Path(resource_path("attribution", "models"))
        / model_name
        / "attribution_parameter14_single"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    gain_table = parameter_gain_table(model_name, flash_model, hail_model)
    gain_table.to_csv(out_dir / "parameter_gain.csv", index=False, encoding="utf-8-sig")
    summary_rows = []

    for start, end in common["decades"]:
        path = Path(model["future_feature_template"].format(start=start, end=end))
        if not path.exists():
            raise FileNotFoundError(f"{model_name} missing future feature table: {path}")
        future = align_to_strict_land(pd.read_parquet(path), labels, common)
        print(f"{model_name} {start}s: fourteen-parameter two-way attribution")
        contributions, total_by_target, residuals = single_parameter_attribution(
            hist, future, flash_model, hail_model, flash_features, hail_features
        )

        contribution_rows = []
        residual_rows = []
        for target in ("flash", "hail"):
            total = total_by_target[target]
            residual_part = hist[keys].copy()
            residual_part["model"] = model_name
            residual_part["decade"] = f"{start}s"
            residual_part["target"] = target
            residual_part["total_change"] = total
            for direction in ("forward", "backward", "two_way"):
                contribution_sum = sum(contributions[target][direction].values())
                residual_part[f"{direction}_sum"] = contribution_sum
                residual_part[f"{direction}_residual"] = residuals[target][direction]

            identity_error = float(np.max(np.abs(
                residual_part["two_way_residual"].to_numpy()
                - (
                    residual_part["forward_residual"].to_numpy()
                    + residual_part["backward_residual"].to_numpy()
                ) / 2.0
            )))
            print(
                f"{model_name} {start}s {target} "
                f"max two-way residual identity error={identity_error:.3e}"
            )
            residual_rows.append(residual_part)

            for parameter in PARAMETERS:
                part = hist[keys].copy()
                part["model"] = model_name
                part["decade"] = f"{start}s"
                part["target"] = target
                part["parameter"] = parameter
                for direction in ("forward", "backward", "two_way"):
                    values = contributions[target][direction][parameter]
                    part[f"{direction}_contribution"] = values
                    part[f"{direction}_absolute"] = np.abs(values)
                    summary_rows.append({
                        "model": model_name,
                        "decade": f"{start}s",
                        "target": target,
                        "direction": direction,
                        "parameter": parameter,
                        "signed_area_mean": area_weighted_mean(
                            values, hist[common["latitude_column"]]
                        ),
                        "absolute_area_mean": area_weighted_mean(
                            np.abs(values), hist[common["latitude_column"]]
                        ),
                    })
                contribution_rows.append(part)

        pd.concat(contribution_rows, ignore_index=True).to_parquet(
            out_dir / f"parameter_contributions_{start}_{end}.parquet", index=False
        )
        pd.concat(residual_rows, ignore_index=True).to_parquet(
            out_dir / f"grid_residuals_{start}_{end}.parquet", index=False
        )

    summary = pd.DataFrame(summary_rows).merge(
        gain_table,
        on=["model", "target", "parameter"],
        validate="many_to_one",
    )
    summary["attribution_rank"] = summary.groupby(
        ["decade", "target", "direction"]
    )["absolute_area_mean"].rank(method="average", ascending=False)
    summary.to_csv(
        out_dir / "parameter_area_summary.csv", index=False, encoding="utf-8-sig"
    )

    correlations = []
    for keys_value, group in summary.groupby(["decade", "target", "direction"]):
        decade, target, direction = keys_value
        correlations.append({
            "model": model_name,
            "decade": decade,
            "target": target,
            "direction": direction,
            "spearman_gain_vs_absolute_contribution": group["gain_rank"].corr(
                group["attribution_rank"]
            ),
        })
    pd.DataFrame(correlations).to_csv(
        out_dir / "gain_attribution_rank_correlations.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Completed: {model_name}; saved under {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate two-way counterfactual attribution for Figure 5"
    )
    parser.add_argument(
        "--model",
        choices=("all", "CanESM5", "MIROC6", "BCC-CSM2-MR"),
        default="all",
        help="Run one model or all three models (default: all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    names = tuple(config["models"]) if args.model == "all" else (args.model,)
    for name in names:
        calculate_model(name, config)


if __name__ == "__main__":
    main()
