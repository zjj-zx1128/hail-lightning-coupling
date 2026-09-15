# Purpose: regional parameter attribution.
# Source: cmip6_figure5_contribution.ipynb; see docs/source_coverage.csv for cell provenance.
# Run order and required inputs: docs/workflow.md.
# Run from repository root: python -m analysis.regional_parameter_attribution

# -*- coding: utf-8 -*-
"""
Figure_region_parameter_attribution.py

Regional counterfactual attribution analysis for CMIP6 lightning/hail projections.

Main workflow
-------------
1. Read existing grid-cell attribution parquet files.
2. Apply region bounds, land mask, and finite-value filtering.
3. Calculate area-weighted regional statistics for each model and decade.
4. Calculate five-decade summaries.
5. Calculate the multi-model mean (MMM) after regional averaging.
6. Export three CSV files.
7. Plot MMM regional attribution for flash and hail separately.

Important
---------
- This script does NOT retrain XGBoost models.
- This script does NOT rerun counterfactual attribution.
- The MMM is calculated after model-specific regional statistics.
- Positive and negative fractions are weighted by cos(latitude).
"""

from __future__ import annotations

from publication_config import resource_path

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Cartopy and Shapely are used only to construct the fallback land mask.
# If an existing land-mask table is supplied, these packages are not required.
try:
    import cartopy.io.shapereader as shpreader
    from shapely.geometry import Point
    from shapely.ops import unary_union
    try:
        from shapely import vectorized as shapely_vectorized
    except ImportError:
        shapely_vectorized = None
except ImportError:
    shpreader = None
    Point = None
    unary_union = None
    shapely_vectorized = None


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class Config:
    # Root containing:
    # models/CanESM5/attribution_parameter14_single/*.parquet
    project_root: Path = Path(resource_path('attribution', ''))

    # Output folder
    output_dir: Path = Path(resource_path('figures', 'regional_attribution'))

    # Optional existing land-mask table.
    # Supported formats: parquet or csv.
    # Required columns: latitude, longitude, and one of:
    # land_mask / is_land / land / mask
    # Set to None to construct a Natural Earth land mask.
    land_mask_file: Optional[Path] = None

    models: Tuple[str, ...] = (
        "CanESM5",
        "MIROC6",
        "BCC-CSM2-MR",
    )

    targets: Tuple[str, ...] = (
        "flash",
        "hail",
    )

    parameters: Tuple[str, ...] = (
        "mucape",
        "mucin",
        "pw",
        "s06",
        "h_10c",
        "h_30c",
        "dh_10_30",
        "div500",
        "td_sfc",
        "theta_e",
        "cape",
        "cin",
        "flh",
        "k_index",
    )

    decades: Tuple[str, ...] = (
        "2050_2059",
        "2060_2069",
        "2070_2079",
        "2080_2089",
        "2090_2099",
    )

    attribution_subdir: str = "attribution_parameter14_single"
    file_pattern: str = "parameter_contributions_{decade}.parquet"

    # Plot configuration
    dpi: int = 600
    figure_width: float = 18.0
    figure_height: float = 15.0
    contribution_scale: float = 1.0e4
    positive_color: str = "#D55E00"
    negative_color: str = "#0072B2"
    font_family: str = "Arial"


CONFIG = Config()


# Figure 1 regional definitions.
# Longitude convention is assumed to be [-180, 180).
REGIONS: Dict[str, Dict[str, float]] = {
    "FL1": {"lat_min": 24.0,  "lat_max": 35.0,  "lon_min": -98.0, "lon_max": -78.0},
    "FL2": {"lat_min": -10.0, "lat_max": 5.0,   "lon_min": -76.0, "lon_max": -58.0},
    "FL3": {"lat_min": -8.0,  "lat_max": 5.0,   "lon_min": 12.0,  "lon_max": 35.0},
    "FL4": {"lat_min": -12.0, "lat_max": 8.0,   "lon_min": 95.0,  "lon_max": 150.0},
    "HL1": {"lat_min": 35.0,  "lat_max": 47.0,  "lon_min": -104.0,"lon_max": -90.0},
    "HL2": {"lat_min": -42.0, "lat_max": -23.0, "lon_min": -68.0, "lon_max": -50.0},
    "HL3": {"lat_min": 35.0,  "lat_max": 55.0,  "lon_min": -10.0, "lon_max": 20.0},
    "HL4": {"lat_min": 35.0,  "lat_max": 55.0,  "lon_min": 112.0, "lon_max": 138.0},
}

REGION_ORDER: Tuple[str, ...] = (
    "FL1", "FL2", "FL3", "FL4",
    "HL1", "HL2", "HL3", "HL4",
)

REQUIRED_PARQUET_COLUMNS: Tuple[str, ...] = (
    "latitude",
    "longitude",
    "model",
    "decade",
    "target",
    "parameter",
    "forward_contribution",
    "backward_contribution",
    "two_way_contribution",
)


# =============================================================================
# 2. GENERAL UTILITIES
# =============================================================================

def standardize_longitude(values: pd.Series) -> pd.Series:
    """Convert longitude values to the [-180, 180) convention."""
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = ((arr + 180.0) % 360.0) - 180.0
    return pd.Series(arr, index=values.index, name=values.name)


def normalize_decade_label(value: object) -> str:
    """
    Normalize decade labels to YYYY_YYYY.

    Examples
    --------
    2050_2059 -> 2050_2059
    2050-2059 -> 2050_2059
    2050s     -> 2050_2059
    """
    text = str(value).strip()
    text = text.replace("-", "_").replace("–", "_")

    if text.endswith("s") and len(text) == 5 and text[:4].isdigit():
        start = int(text[:4])
        return f"{start}_{start + 9}"

    parts = text.split("_")
    if len(parts) == 2 and all(part.isdigit() for part in parts):
        return f"{int(parts[0]):04d}_{int(parts[1]):04d}"

    return text


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize known attribution-column names.

    The existing parquet files may use coordinate columns such as
    ``Latitude`` and ``Longitude``. These are normalized to lowercase names
    used internally by this script. Other known fields are matched
    case-insensitively as well.
    """
    canonical_names = {
        "latitude": "latitude",
        "longitude": "longitude",
        "model": "model",
        "decade": "decade",
        "target": "target",
        "parameter": "parameter",
        "forward_contribution": "forward_contribution",
        "backward_contribution": "backward_contribution",
        "two_way_contribution": "two_way_contribution",
    }

    rename_map = {}
    for column in df.columns:
        normalized = str(column).strip().lower()
        if normalized in canonical_names:
            rename_map[column] = canonical_names[normalized]

    standardized = df.rename(columns=rename_map).copy()

    duplicated = standardized.columns[standardized.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(
            "Column-name standardization created duplicate columns: "
            f"{duplicated}. Please inspect the parquet schema."
        )

    return standardized


def validate_columns(df: pd.DataFrame, required: Sequence[str], source: Path) -> None:
    """Raise a clear error when required columns are absent."""
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {source}:\n"
            f"  {missing}\n"
            f"Available columns after standardization:\n"
            f"  {list(df.columns)}"
        )


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Finite weighted mean."""
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return np.nan
    return float(np.sum(values[valid] * weights[valid]) / np.sum(weights[valid]))


def weighted_fraction(condition: np.ndarray, weights: np.ndarray) -> float:
    """Area-weighted fraction for a Boolean condition."""
    valid = np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return np.nan
    return float(np.sum(weights[valid] * condition[valid].astype(float)) / np.sum(weights[valid]))


# =============================================================================
# 3. READ ATTRIBUTION DATA
# =============================================================================

def attribution_file_path(config: Config, model: str, decade: str) -> Path:
    """Return the expected parquet path."""
    return (
        config.project_root
        / "models"
        / model
        / config.attribution_subdir
        / config.file_pattern.format(decade=decade)
    )


def inspect_first_parquet(config: Config) -> None:
    """Read one file and validate the declared parquet schema."""
    for model in config.models:
        for decade in config.decades:
            path = attribution_file_path(config, model, decade)
            if path.exists():
                sample = pd.read_parquet(path)
                sample = standardize_column_names(sample)
                validate_columns(sample, REQUIRED_PARQUET_COLUMNS, path)
                print("[SCHEMA CHECK] Required parquet fields are available.")
                print(f"[SCHEMA CHECK] Sample file: {path}")
                print(f"[SCHEMA CHECK] Columns: {list(sample.columns)}")
                return

    raise FileNotFoundError(
        "No attribution parquet file was found under the configured project root.\n"
        f"Configured root: {config.project_root}"
    )


def read_all_attribution(config: Config) -> pd.DataFrame:
    """Read and concatenate all configured model-decade parquet files."""
    frames: List[pd.DataFrame] = []
    missing_files: List[Path] = []

    for model in config.models:
        for decade in config.decades:
            path = attribution_file_path(config, model, decade)
            if not path.exists():
                missing_files.append(path)
                continue

            df = pd.read_parquet(path)
            df = standardize_column_names(df)
            validate_columns(df, REQUIRED_PARQUET_COLUMNS, path)

            # Standardize critical fields.
            df = df.loc[:, REQUIRED_PARQUET_COLUMNS].copy()
            df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
            df["longitude"] = standardize_longitude(df["longitude"])
            df["target"] = df["target"].astype(str).str.lower().str.strip()
            df["parameter"] = df["parameter"].astype(str).str.lower().str.strip()
            df["model"] = df["model"].astype(str).str.strip()
            df["decade"] = df["decade"].map(normalize_decade_label)

            for column in (
                "forward_contribution",
                "backward_contribution",
                "two_way_contribution",
            ):
                df[column] = pd.to_numeric(df[column], errors="coerce")

            # Trust directory/file metadata if embedded labels are absent or inconsistent.
            inconsistent_model = df["model"].notna() & (df["model"] != model)
            if inconsistent_model.any():
                warnings.warn(
                    f"{path}: embedded model labels differ from directory model. "
                    f"The directory model '{model}' will be used."
                )
            df["model"] = model

            normalized_decade = normalize_decade_label(decade)
            inconsistent_decade = df["decade"].notna() & (df["decade"] != normalized_decade)
            if inconsistent_decade.any():
                warnings.warn(
                    f"{path}: embedded decade labels differ from filename decade. "
                    f"The filename decade '{normalized_decade}' will be used."
                )
            df["decade"] = normalized_decade

            frames.append(df)

    if missing_files:
        missing_text = "\n".join(str(path) for path in missing_files)
        raise FileNotFoundError(
            "The following required parquet files are missing:\n"
            f"{missing_text}"
        )

    if not frames:
        raise RuntimeError("No attribution data were read.")

    data = pd.concat(frames, ignore_index=True)

    unknown_targets = sorted(set(data["target"].dropna()) - set(config.targets))
    unknown_parameters = sorted(set(data["parameter"].dropna()) - set(config.parameters))
    if unknown_targets:
        warnings.warn(f"Unconfigured targets found and ignored: {unknown_targets}")
    if unknown_parameters:
        warnings.warn(f"Unconfigured parameters found and ignored: {unknown_parameters}")

    data = data[
        data["target"].isin(config.targets)
        & data["parameter"].isin(config.parameters)
    ].copy()

    if data.empty:
        raise RuntimeError(
            "No rows remain after target/parameter filtering. "
            "Check target and parameter naming."
        )

    return data


# =============================================================================
# 4. LAND MASK
# =============================================================================

def read_existing_land_mask(path: Path) -> pd.DataFrame:
    """
    Read an existing tabular land mask.

    Required coordinate columns
    ---------------------------
    latitude, longitude

    Accepted mask columns
    ---------------------
    land_mask, is_land, land, mask
    """
    if not path.exists():
        raise FileNotFoundError(f"Land-mask file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        mask = pd.read_parquet(path)
    elif suffix == ".csv":
        mask = pd.read_csv(path)
    else:
        raise ValueError(
            "Existing land mask must be a .parquet or .csv table. "
            f"Received: {path}"
        )

    coordinate_columns = {"latitude", "longitude"}
    missing_coordinates = coordinate_columns - set(mask.columns)
    if missing_coordinates:
        raise ValueError(
            f"Land-mask file is missing coordinate columns: {sorted(missing_coordinates)}"
        )

    candidate_mask_columns = ("land_mask", "is_land", "land", "mask")
    mask_column = next(
        (column for column in candidate_mask_columns if column in mask.columns),
        None,
    )
    if mask_column is None:
        raise ValueError(
            "Land-mask file must contain one of these columns: "
            f"{candidate_mask_columns}"
        )

    out = mask[["latitude", "longitude", mask_column]].copy()
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = standardize_longitude(out["longitude"])
    out["is_land"] = out[mask_column].astype(bool)

    out = out.dropna(subset=["latitude", "longitude"])
    out = out.drop_duplicates(subset=["latitude", "longitude"], keep="last")
    return out[["latitude", "longitude", "is_land"]]


def build_natural_earth_land_mask(grid: pd.DataFrame) -> pd.DataFrame:
    """
    Construct a Natural Earth 110 m land mask for unique grid-cell centers.

    This reproduces a strict center-point land mask:
    a grid point is land only when its center is contained by a land polygon.
    """
    if shpreader is None or unary_union is None or Point is None:
        raise ImportError(
            "Natural Earth fallback land masking requires cartopy and shapely.\n"
            "Install them or set CONFIG.land_mask_file to an existing mask table."
        )

    grid = grid[["latitude", "longitude"]].drop_duplicates().copy()
    lon = grid["longitude"].to_numpy(dtype=float)
    lat = grid["latitude"].to_numpy(dtype=float)

    shapefile = shpreader.natural_earth(
        resolution="110m",
        category="physical",
        name="land",
    )
    land_geometry = unary_union(list(shpreader.Reader(shapefile).geometries()))

    if shapely_vectorized is not None:
        # contains() is strict: points exactly on polygon boundaries are False.
        is_land = shapely_vectorized.contains(land_geometry, lon, lat)
    else:
        prepared_points = [Point(x, y) for x, y in zip(lon, lat)]
        is_land = np.array(
            [land_geometry.contains(point) for point in prepared_points],
            dtype=bool,
        )

    grid["is_land"] = is_land
    return grid


def construct_land_mask(data: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Read an existing land mask or build a Natural Earth fallback mask."""
    grid = data[["latitude", "longitude"]].dropna().drop_duplicates().copy()

    if config.land_mask_file is not None:
        mask = read_existing_land_mask(config.land_mask_file)
        merged = grid.merge(
            mask,
            on=["latitude", "longitude"],
            how="left",
            validate="one_to_one",
        )

        missing = int(merged["is_land"].isna().sum())
        if missing:
            raise ValueError(
                f"The existing land mask does not cover {missing} attribution grid points. "
                "Coordinate precision and longitude convention should be checked."
            )

        print(f"[LAND MASK] Existing mask loaded: {config.land_mask_file}")
        return merged

    print("[LAND MASK] No existing mask configured; using Natural Earth 110 m.")
    return build_natural_earth_land_mask(grid)


# =============================================================================
# 5. REGION MASK AND STATISTICS
# =============================================================================

def region_coordinate_mask(
    latitude: pd.Series,
    longitude: pd.Series,
    bounds: Dict[str, float],
) -> pd.Series:
    """
    Return the rectangular region mask.

    Both lower and upper limits are inclusive, matching common map-box logic.
    """
    return (
        latitude.between(bounds["lat_min"], bounds["lat_max"], inclusive="both")
        & longitude.between(bounds["lon_min"], bounds["lon_max"], inclusive="both")
    )


def add_land_information(data: pd.DataFrame, land_mask: pd.DataFrame) -> pd.DataFrame:
    """Attach the Boolean land mask to attribution rows."""
    merged = data.merge(
        land_mask,
        on=["latitude", "longitude"],
        how="left",
        validate="many_to_one",
    )

    if merged["is_land"].isna().any():
        missing = int(merged["is_land"].isna().sum())
        raise ValueError(f"Land-mask information is missing for {missing} rows.")

    merged["is_land"] = merged["is_land"].astype(bool)
    return merged


def calculate_model_decade_statistics(
    data: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    """
    Calculate model-specific, decade-specific regional statistics.

    Valid analysis rows satisfy:
        region rectangle
        AND land mask
        AND finite two_way_contribution
    """
    records: List[Dict[str, object]] = []

    for region_name in REGION_ORDER:
        bounds = REGIONS[region_name]
        regional = data[
            region_coordinate_mask(data["latitude"], data["longitude"], bounds)
            & data["is_land"]
        ].copy()

        for model in config.models:
            for target in config.targets:
                for parameter in config.parameters:
                    for decade in config.decades:
                        decade_label = normalize_decade_label(decade)

                        subset = regional[
                            (regional["model"] == model)
                            & (regional["target"] == target)
                            & (regional["parameter"] == parameter)
                            & (regional["decade"] == decade_label)
                        ].copy()

                        subset = subset[
                            np.isfinite(subset["two_way_contribution"])
                            & np.isfinite(subset["latitude"])
                        ].copy()

                        values = subset["two_way_contribution"].to_numpy(dtype=float)
                        weights = np.cos(
                            np.deg2rad(subset["latitude"].to_numpy(dtype=float))
                        )

                        mean_contribution = weighted_mean(values, weights)
                        positive_fraction = weighted_fraction(values > 0, weights)
                        negative_fraction = weighted_fraction(values < 0, weights)

                        records.append(
                            {
                                "Model": model,
                                "Region": region_name,
                                "Target": target,
                                "Parameter": parameter,
                                "Decade": decade_label,
                                "MeanContribution": mean_contribution,
                                "PositiveFraction": positive_fraction,
                                "NegativeFraction": negative_fraction,
                                "GridNumber": int(len(subset)),
                            }
                        )

    result = pd.DataFrame.from_records(records)

    missing_groups = result["MeanContribution"].isna().sum()
    if missing_groups:
        warnings.warn(
            f"{missing_groups} model-region-target-parameter-decade groups "
            "contain no valid land-grid contribution."
        )

    return result


def append_mmm_by_decade(model_decade: pd.DataFrame) -> pd.DataFrame:
    """
    Append MMM rows after model-specific regional statistics.

    MMM is the arithmetic mean of the model-level regional statistics.
    GridNumber is retained as the rounded mean model grid count. A warning is
    emitted if model grid counts differ within an MMM group.
    """
    group_columns = ["Region", "Target", "Parameter", "Decade"]

    count_variability = (
        model_decade.groupby(group_columns, observed=True)["GridNumber"]
        .nunique(dropna=True)
    )
    inconsistent = count_variability[count_variability > 1]
    if not inconsistent.empty:
        warnings.warn(
            "GridNumber differs among models for some MMM groups. "
            "MMM GridNumber will be the rounded mean of model-specific counts."
        )

    mmm = (
        model_decade.groupby(group_columns, as_index=False, observed=True)
        .agg(
            MeanContribution=("MeanContribution", "mean"),
            PositiveFraction=("PositiveFraction", "mean"),
            NegativeFraction=("NegativeFraction", "mean"),
            GridNumber=("GridNumber", "mean"),
        )
    )
    mmm.insert(0, "Model", "MMM")
    mmm["GridNumber"] = mmm["GridNumber"].round().astype("Int64")

    combined = pd.concat([model_decade, mmm], ignore_index=True)
    return combined[
        [
            "Model",
            "Region",
            "Target",
            "Parameter",
            "Decade",
            "MeanContribution",
            "PositiveFraction",
            "NegativeFraction",
            "GridNumber",
        ]
    ]


def calculate_five_decade_summary(by_decade: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate five-decade summaries for each model, including MMM.

    Definitions
    -----------
    FiveDecadeMean:
        Arithmetic mean of the five decade-level regional mean contributions.

    FiveDecadeStd:
        Sample standard deviation across the five decade-level regional means.

    PositiveFraction / NegativeFraction:
        Arithmetic mean of the five decade-level area-weighted fractions.

    GridNumber:
        Rounded arithmetic mean of the five decade-level valid-grid counts.

    Rank:
        Descending rank within each Model-Region-Target group according to
        abs(FiveDecadeMean). Rank 1 has the largest absolute mean contribution.
    """
    group_columns = ["Model", "Region", "Target", "Parameter"]

    summary = (
        by_decade.groupby(group_columns, as_index=False, observed=True)
        .agg(
            FiveDecadeMean=("MeanContribution", "mean"),
            FiveDecadeStd=("MeanContribution", "std"),
            PositiveFraction=("PositiveFraction", "mean"),
            NegativeFraction=("NegativeFraction", "mean"),
            GridNumber=("GridNumber", "mean"),
        )
    )
    summary["GridNumber"] = summary["GridNumber"].round().astype("Int64")

    summary["Rank"] = (
        summary.groupby(["Model", "Region", "Target"], observed=True)[
            "FiveDecadeMean"
        ]
        .transform(lambda series: series.abs().rank(method="first", ascending=False))
        .astype("Int64")
    )

    return summary[
        [
            "Model",
            "Region",
            "Target",
            "Parameter",
            "FiveDecadeMean",
            "FiveDecadeStd",
            "PositiveFraction",
            "NegativeFraction",
            "GridNumber",
            "Rank",
        ]
    ]


def calculate_region_information(
    data: pd.DataFrame,
    land_mask: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the region-information table.

    Grid Number is the number of unique land grid-cell centers in each region
    on the attribution grid, independent of parameter validity.
    """
    grid = (
        data[["latitude", "longitude"]]
        .dropna()
        .drop_duplicates()
        .merge(
            land_mask,
            on=["latitude", "longitude"],
            how="left",
            validate="one_to_one",
        )
    )
    grid = grid[grid["is_land"].fillna(False)].copy()

    records: List[Dict[str, object]] = []
    for region_name in REGION_ORDER:
        bounds = REGIONS[region_name]
        selected = grid[
            region_coordinate_mask(grid["latitude"], grid["longitude"], bounds)
        ]

        records.append(
            {
                "Region": region_name,
                "Latitude Range": f"{bounds['lat_min']:g} to {bounds['lat_max']:g}",
                "Longitude Range": f"{bounds['lon_min']:g} to {bounds['lon_max']:g}",
                "Grid Number": int(
                    selected[["latitude", "longitude"]].drop_duplicates().shape[0]
                ),
            }
        )

    return pd.DataFrame.from_records(records)


# =============================================================================
# 6. CSV OUTPUT
# =============================================================================

def save_csv_outputs(
    by_decade: pd.DataFrame,
    summary: pd.DataFrame,
    region_information: pd.DataFrame,
    config: Config,
) -> None:
    """Save all requested CSV files."""
    config.output_dir.mkdir(parents=True, exist_ok=True)

    by_decade_path = config.output_dir / "region_parameter_by_decade.csv"
    summary_path = config.output_dir / "region_parameter_summary.csv"
    region_path = config.output_dir / "region_information.csv"

    by_decade.to_csv(by_decade_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    region_information.to_csv(region_path, index=False, encoding="utf-8-sig")

    print(f"[CSV] {by_decade_path}")
    print(f"[CSV] {summary_path}")
    print(f"[CSV] {region_path}")


# =============================================================================
# 7. PLOTTING
# =============================================================================

def configure_matplotlib(config: Config) -> None:
    """Apply a consistent manuscript plotting style."""
    plt.rcParams.update(
        {
            "font.family": config.font_family,
            "axes.unicode_minus": False,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 11,
            "savefig.dpi": config.dpi,
        }
    )


def parameter_display_name(parameter: str) -> str:
    """Return manuscript-friendly parameter labels."""
    labels = {
        "mucape": "MUCAPE",
        "mucin": "MUCIN",
        "pw": "PW",
        "s06": "S06",
        "h_10c": r"H$_{-10}$",
        "h_30c": r"H$_{-30}$",
        "dh_10_30": r"$\Delta$H$_{-10,-30}$",
        "div500": "Div500",
        "td_sfc": r"T$_{d,sfc}$",
        "theta_e": r"$\theta_e$",
        "cape": "CAPE",
        "cin": "CIN",
        "flh": "FLH",
        "k_index": "K index",
    }
    return labels.get(parameter, parameter)


def plot_regional_attribution(
    summary: pd.DataFrame,
    target: str,
    config: Config,
) -> Path:
    """
    Plot MMM five-decade mean contributions for one target.

    Bar length is abs(FiveDecadeMean).
    Bar color retains the sign.
    Text labels show signed contribution x 10^4.
    """
    configure_matplotlib(config)

    plot_data = summary[
        (summary["Model"] == "MMM")
        & (summary["Target"] == target)
    ].copy()

    expected_rows = len(REGION_ORDER) * len(config.parameters)
    if len(plot_data) != expected_rows:
        warnings.warn(
            f"Expected {expected_rows} MMM rows for target '{target}', "
            f"but found {len(plot_data)}."
        )

    panel_letters = list("abcdefgh")
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(config.figure_width, config.figure_height),
        sharex=False,
        constrained_layout=False,
    )
    axes = axes.ravel()

    # Use one common x limit across all regions for direct comparison.
    finite_abs_values = np.abs(
        plot_data["FiveDecadeMean"].to_numpy(dtype=float)
        * config.contribution_scale
    )
    finite_abs_values = finite_abs_values[np.isfinite(finite_abs_values)]
    global_max = float(finite_abs_values.max()) if finite_abs_values.size else 1.0
    x_upper = global_max * 1.25 if global_max > 0 else 1.0

    for index, (axis, region_name) in enumerate(zip(axes, REGION_ORDER)):
        region_data = plot_data[plot_data["Region"] == region_name].copy()

        # Preserve the Figure 5 parameter order rather than rank order.
        region_data["Parameter"] = pd.Categorical(
            region_data["Parameter"],
            categories=list(config.parameters),
            ordered=True,
        )
        region_data = region_data.sort_values("Parameter", ascending=False)

        signed_scaled = (
            region_data["FiveDecadeMean"].to_numpy(dtype=float)
            * config.contribution_scale
        )
        lengths = np.abs(signed_scaled)
        colors = np.where(
            signed_scaled >= 0,
            config.positive_color,
            config.negative_color,
        )
        y = np.arange(len(region_data))

        axis.barh(
            y,
            lengths,
            color=colors,
            edgecolor="none",
            height=0.72,
        )

        axis.set_yticks(y)
        axis.set_yticklabels(
            [parameter_display_name(str(value)) for value in region_data["Parameter"]]
        )
        axis.set_xlim(0, x_upper)
        axis.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.35)
        axis.set_axisbelow(True)

        for y_pos, length, signed_value in zip(y, lengths, signed_scaled):
            if not np.isfinite(signed_value):
                continue
            text_x = length + x_upper * 0.015
            axis.text(
                text_x,
                y_pos,
                f"{signed_value:+.2f}",
                va="center",
                ha="left",
                fontsize=9,
            )

        axis.set_title(
            f"({panel_letters[index]}) {region_name}",
            loc="left",
            pad=8,
            fontweight="bold",
        )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    for axis in axes[4:]:
        axis.set_xlabel(r"|Five-decade mean contribution| ($\times 10^{4}$)")

    # Figure-level legend matching Figure 5.
    positive_proxy = plt.Rectangle(
        (0, 0), 1, 1, facecolor=config.positive_color, edgecolor="none"
    )
    negative_proxy = plt.Rectangle(
        (0, 0), 1, 1, facecolor=config.negative_color, edgecolor="none"
    )
    fig.legend(
        [positive_proxy, negative_proxy],
        ["Positive contribution", "Negative contribution"],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.985),
    )

    fig.suptitle(
        f"Regional parameter contributions to future {target} change (MMM)",
        y=1.005,
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.012,
        r"Numbers show signed contribution ($\times 10^{4}$).",
        ha="center",
        va="bottom",
        fontsize=11,
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.07,
        top=0.925,
        wspace=0.48,
        hspace=0.28,
    )

    output_path = (
        config.output_dir
        / f"Figure_region_parameter_attribution_{target}_MMM.png"
    )
    fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[FIGURE] {output_path}")
    return output_path


# =============================================================================
# 8. QUALITY CONTROL
# =============================================================================

def run_quality_checks(
    data: pd.DataFrame,
    by_decade: pd.DataFrame,
    summary: pd.DataFrame,
    region_information: pd.DataFrame,
    config: Config,
) -> None:
    """Run concise consistency checks before writing final outputs."""
    duplicate_keys = [
        "latitude",
        "longitude",
        "model",
        "decade",
        "target",
        "parameter",
    ]
    duplicate_count = int(data.duplicated(subset=duplicate_keys, keep=False).sum())
    if duplicate_count:
        warnings.warn(
            f"{duplicate_count} attribution rows have duplicate coordinate/model/"
            "decade/target/parameter keys. They will be treated as separate rows."
        )

    fraction_columns = ["PositiveFraction", "NegativeFraction"]
    for table_name, table in (("by_decade", by_decade), ("summary", summary)):
        for column in fraction_columns:
            invalid = table[column].notna() & ~table[column].between(0.0, 1.0)
            if invalid.any():
                raise ValueError(
                    f"{table_name}.{column} contains values outside [0, 1]."
                )

    finite_fraction_rows = by_decade[
        by_decade["PositiveFraction"].notna()
        & by_decade["NegativeFraction"].notna()
    ]
    fraction_sum = (
        finite_fraction_rows["PositiveFraction"]
        + finite_fraction_rows["NegativeFraction"]
    )
    # Values equal to zero are excluded from both fractions, so the sum can be < 1.
    if (fraction_sum > 1.0 + 1e-10).any():
        raise ValueError("PositiveFraction + NegativeFraction exceeds 1.")

    expected_model_decade_rows = (
        len(config.models)
        * len(REGION_ORDER)
        * len(config.targets)
        * len(config.parameters)
        * len(config.decades)
    )
    actual_model_rows = int((by_decade["Model"] != "MMM").sum())
    if actual_model_rows != expected_model_decade_rows:
        warnings.warn(
            f"Expected {expected_model_decade_rows} model-specific decade rows, "
            f"but found {actual_model_rows}."
        )

    if (region_information["Grid Number"] <= 0).any():
        empty_regions = region_information.loc[
            region_information["Grid Number"] <= 0, "Region"
        ].tolist()
        raise ValueError(f"No land grid cells were found in regions: {empty_regions}")

    print("[QC] Quality checks completed.")


# =============================================================================
# 9. MAIN
# =============================================================================

def main(config: Config = CONFIG) -> None:
    """Execute the complete regional attribution workflow."""
    print("=" * 88)
    print("REGIONAL CMIP6 LIGHTNING/HAIL PARAMETER ATTRIBUTION")
    print("=" * 88)
    print(f"[CONFIG] Project root: {config.project_root}")
    print(f"[CONFIG] Output folder: {config.output_dir}")

    config.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Confirm parquet fields.
    inspect_first_parquet(config)

    # 2. Read existing attribution results.
    data = read_all_attribution(config)
    print(f"[DATA] Rows read: {len(data):,}")
    print(
        f"[DATA] Unique grid coordinates: "
        f"{data[['latitude', 'longitude']].drop_duplicates().shape[0]:,}"
    )

    # 3. Construct/read land mask.
    land_mask = construct_land_mask(data, config)
    data = add_land_information(data, land_mask)
    print(
        f"[LAND MASK] Land grid coordinates: "
        f"{land_mask.loc[land_mask['is_land'], ['latitude', 'longitude']].shape[0]:,}"
    )

    # 4. Model-specific regional statistics.
    model_decade = calculate_model_decade_statistics(data, config)

    # 5. Append MMM after regional averaging.
    by_decade = append_mmm_by_decade(model_decade)

    # 6. Five-decade summary and ranking.
    summary = calculate_five_decade_summary(by_decade)

    # 7. Region information.
    region_information = calculate_region_information(data, land_mask)

    # 8. Quality control.
    run_quality_checks(
        data=data,
        by_decade=by_decade,
        summary=summary,
        region_information=region_information,
        config=config,
    )

    # 9. CSV output.
    save_csv_outputs(
        by_decade=by_decade,
        summary=summary,
        region_information=region_information,
        config=config,
    )

    # 10. MMM figures, one for each target.
    for target in config.targets:
        plot_regional_attribution(
            summary=summary,
            target=target,
            config=config,
        )

    print("=" * 88)
    print("DONE")
    print("=" * 88)


if __name__ == "__main__":
    main()