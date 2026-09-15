# Purpose: pf regional density.
# Source: hdfcanshu_vs.ipynb; see docs/source_coverage.csv for cell provenance.
# Run order and required inputs: docs/workflow.md.
# Run from repository root: python -m analysis.pf_regional_density

# -*- coding: utf-8 -*-
"""
Calculate land PF density for eight regions using two versions:
1) ALL_LAND: retain all land PFs and all land area.
2) LAND_BELOW_2000M: retain land PFs and land area with DEM elevation < 2000 m.

Density unit: PFs per 10,000 km^2 yr^-1.
"""

from __future__ import annotations

from publication_config import resource_path

import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cartopy.io.shapereader as shpreader
import numpy as np
import pandas as pd
import rasterio
from pyproj import Geod
from rasterio.windows import Window, from_bounds
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
from shapely.ops import unary_union
from shapely.prepared import prep
from tqdm import tqdm
import shapely

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
INPUT_FOLDER = Path(resource_path('pf_processed', 'environment/filtered_land_regions'))
LAND_SHAPEFILE = Path(resource_path('geography', 'ne_10m_land/ne_10m_land.shp'))
ELEVATION_TIF = Path(resource_path('observations', 'data/elevation/elevation_1KMmn_SRTM.tif'))
OUTPUT_FOLDER = Path(resource_path('tables', ''))

OUTPUT_COMBINED_CSV = OUTPUT_FOLDER / "PF_density_by_region_two_versions.csv"
OUTPUT_ALL_LAND_CSV = OUTPUT_FOLDER / "PF_density_by_region_all_land.csv"
OUTPUT_BELOW_2000M_CSV = OUTPUT_FOLDER / "PF_density_by_region_land_below_2000m.csv"

OBSERVATION_YEARS = 10.0  # 2014-03 to 2024-02 = 120 months
ELEVATION_THRESHOLD_M = 2000.0  # strictly retain elevation < 2000 m

LAT_COLUMN = "LAT"
LON_COLUMN = "LON"
PF_ELEVATION_COLUMN = "ELEV"          # diagnostic only
PF_LAND_COLUMN = "LANDOCEAN"          # diagnostic only
PF_LAND_VALUE = 1

EARTH_RADIUS_KM = 6371.0088
GEOD = Geod(ellps="WGS84")

REGIONS: Dict[str, Dict[str, float]] = {
    "FL1": {"lat_min": 24.0,  "lat_max": 35.0,  "lon_min": -98.0,  "lon_max": -78.0},
    "FL2": {"lat_min": -10.0, "lat_max": 5.0,   "lon_min": -76.0,  "lon_max": -58.0},
    "FL3": {"lat_min": -8.0,  "lat_max": 5.0,   "lon_min": 12.0,   "lon_max": 35.0},
    "FL4": {"lat_min": -12.0, "lat_max": 8.0,   "lon_min": 95.0,   "lon_max": 150.0},
    "HL1": {"lat_min": 35.0,  "lat_max": 47.0,  "lon_min": -104.0, "lon_max": -90.0},
    "HL2": {"lat_min": -42.0, "lat_max": -23.0, "lon_min": -68.0,  "lon_max": -50.0},
    "HL3": {"lat_min": 35.0,  "lat_max": 55.0,  "lon_min": -10.0,  "lon_max": 20.0},
    "HL4": {"lat_min": 35.0,  "lat_max": 55.0,  "lon_min": 112.0,  "lon_max": 138.0},
}
REGION_ORDER: Tuple[str, ...] = (
    "HL1", "HL2", "HL3", "HL4", "FL1", "FL2", "FL3", "FL4"
)

# =============================================================================
# 2. UTILITIES
# =============================================================================
def normalize_lon_to_180(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return ((arr + 180.0) % 360.0) - 180.0


def parse_region_name(file_name: str) -> Optional[str]:
    match = re.match(r"^(FL|HL)([1-4])", file_name, flags=re.IGNORECASE)
    return None if match is None else f"{match.group(1).upper()}{match.group(2)}"


def validate_paths() -> None:
    missing = []
    for label, path in (
        ("PF input folder", INPUT_FOLDER),
        ("Natural Earth shapefile", LAND_SHAPEFILE),
        ("Elevation GeoTIFF", ELEVATION_TIF),
    ):
        if not path.exists():
            missing.append(f"{label}: {path}")
    if missing:
        raise FileNotFoundError("Missing configured paths:\n- " + "\n- ".join(missing))
    if OBSERVATION_YEARS <= 0:
        raise ValueError("OBSERVATION_YEARS must be > 0.")
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


def spherical_cell_area_km2(lat_center: float, dlat_deg: float, dlon_deg: float) -> float:
    lat_s = np.deg2rad(lat_center - abs(dlat_deg) / 2.0)
    lat_n = np.deg2rad(lat_center + abs(dlat_deg) / 2.0)
    dlon = np.deg2rad(abs(dlon_deg))
    return float(EARTH_RADIUS_KM ** 2 * dlon * abs(np.sin(lat_n) - np.sin(lat_s)))


def geometry_area_km2(geometry) -> float:
    if geometry is None or geometry.is_empty:
        return 0.0
    if isinstance(geometry, Polygon):
        area_m2, _ = GEOD.geometry_area_perimeter(geometry)
        return abs(float(area_m2)) / 1e6
    if isinstance(geometry, MultiPolygon):
        return float(sum(geometry_area_km2(g) for g in geometry.geoms))
    if isinstance(geometry, GeometryCollection):
        return float(sum(geometry_area_km2(g) for g in geometry.geoms
                         if isinstance(g, (Polygon, MultiPolygon))))
    return 0.0


def polygon_contains_xy(geometry, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if hasattr(shapely, "intersects_xy"):
        return np.asarray(shapely.intersects_xy(geometry, x, y), dtype=bool)
    if hasattr(shapely, "contains_xy"):
        return np.asarray(shapely.contains_xy(geometry, x, y), dtype=bool)
    from shapely.geometry import Point
    prepared = prep(geometry)
    return np.array([prepared.intersects(Point(float(xx), float(yy)))
                     for xx, yy in zip(x, y)], dtype=bool)

# =============================================================================
# 3. LAND POLYGON
# =============================================================================
def load_land_union(shapefile_path: Path):
    print(f"[LAND] Reading {shapefile_path}")
    geoms = list(shpreader.Reader(str(shapefile_path)).geometries())
    if not geoms:
        raise ValueError("No land geometries were read.")
    land_union = unary_union(geoms)
    if not land_union.is_valid:
        warnings.warn("Invalid land geometry; repairing with buffer(0).")
        land_union = land_union.buffer(0)
    return land_union

# =============================================================================
# 4. PF FILES
# =============================================================================
def discover_region_csv_files() -> Dict[str, List[Path]]:
    mapping = {region: [] for region in REGION_ORDER}
    for path in sorted(INPUT_FOLDER.glob("*.csv")):
        region = parse_region_name(path.name)
        if region in mapping:
            mapping[region].append(path)
    if all(not v for v in mapping.values()):
        raise FileNotFoundError(f"No regional CSV files found under {INPUT_FOLDER}")
    missing = [k for k, v in mapping.items() if not v]
    if missing:
        warnings.warn("Missing regional CSVs: " + ", ".join(missing))
    return mapping


def read_region_pf_files(paths: Sequence[Path], region_name: str) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path, low_memory=False)
        df["_source_file"] = path.name
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    missing = [c for c in (LAT_COLUMN, LON_COLUMN) if c not in data.columns]
    if missing:
        raise ValueError(f"{region_name} missing columns {missing}; available={list(data.columns)}")
    data[LAT_COLUMN] = pd.to_numeric(data[LAT_COLUMN], errors="coerce")
    data[LON_COLUMN] = pd.to_numeric(data[LON_COLUMN], errors="coerce")
    data[LON_COLUMN] = normalize_lon_to_180(data[LON_COLUMN].to_numpy())
    if PF_ELEVATION_COLUMN in data.columns:
        data[PF_ELEVATION_COLUMN] = pd.to_numeric(data[PF_ELEVATION_COLUMN], errors="coerce")
    if PF_LAND_COLUMN in data.columns:
        data[PF_LAND_COLUMN] = pd.to_numeric(data[PF_LAND_COLUMN], errors="coerce")
    return data

# =============================================================================
# 5. EVENT COUNTS
# =============================================================================
def sample_dem_elevation(src, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    coords = list(zip(lon.astype(float), lat.astype(float)))
    sampled = np.array([v[0] for v in src.sample(coords, indexes=1, masked=True)], dtype=float)
    if src.nodata is not None:
        sampled[np.isclose(sampled, src.nodata)] = np.nan
    sampled[~np.isfinite(sampled)] = np.nan
    return sampled


def count_region_pf_events(data: pd.DataFrame, region_name: str,
                           bounds: Dict[str, float], land_union, dem_src) -> Dict[str, object]:
    valid = data[np.isfinite(data[LAT_COLUMN]) & np.isfinite(data[LON_COLUMN])].copy()
    regional = valid[
        valid[LAT_COLUMN].between(bounds["lat_min"], bounds["lat_max"], inclusive="both") &
        valid[LON_COLUMN].between(bounds["lon_min"], bounds["lon_max"], inclusive="both")
    ].copy()

    lon = regional[LON_COLUMN].to_numpy(float)
    lat = regional[LAT_COLUMN].to_numpy(float)
    polygon_land = polygon_contains_xy(land_union, lon, lat)
    land_events = regional.loc[polygon_land].copy()

    sampled_elev = sample_dem_elevation(
        dem_src,
        land_events[LON_COLUMN].to_numpy(float),
        land_events[LAT_COLUMN].to_numpy(float),
    )
    below = np.isfinite(sampled_elev) & (sampled_elev < ELEVATION_THRESHOLD_M)

    original_land_count = np.nan
    land_disagreement = np.nan
    if PF_LAND_COLUMN in regional.columns:
        original_land = regional[PF_LAND_COLUMN].to_numpy(float) == PF_LAND_VALUE
        original_land_count = int(np.sum(original_land))
        land_disagreement = int(np.sum(original_land != polygon_land))

    original_below_count = np.nan
    elev_disagreement = np.nan
    if PF_ELEVATION_COLUMN in land_events.columns:
        original_elev = land_events[PF_ELEVATION_COLUMN].to_numpy(float)
        comparable = np.isfinite(original_elev) & np.isfinite(sampled_elev)
        original_below = original_elev < ELEVATION_THRESHOLD_M
        sampled_below = sampled_elev < ELEVATION_THRESHOLD_M
        original_below_count = int(np.sum(np.isfinite(original_elev) & original_below))
        elev_disagreement = int(np.sum(comparable & (original_below != sampled_below)))

    return {
        "Region": region_name,
        "SourceFileNumber": int(data["_source_file"].nunique()),
        "TotalPFRecords": int(len(data)),
        "ValidCoordinatePFCount": int(len(valid)),
        "InRectanglePFCount": int(len(regional)),
        "AllLandPFCount": int(len(land_events)),
        "LandBelow2000mPFCount": int(np.sum(below)),
        "LandPFWithValidDEMElevation": int(np.sum(np.isfinite(sampled_elev))),
        "OriginalLANDOCEANLandPFCount": original_land_count,
        "LandMaskDisagreementCount": land_disagreement,
        "OriginalELEVBelow2000mPFCount": original_below_count,
        "ElevationClassDisagreementCount": elev_disagreement,
    }

# =============================================================================
# 6. AREAS
# =============================================================================
def calculate_all_land_area_km2(land_union, bounds: Dict[str, float]) -> Dict[str, float]:
    region = box(bounds["lon_min"], bounds["lat_min"], bounds["lon_max"], bounds["lat_max"])
    intersection = land_union.intersection(region)
    rectangle_area = geometry_area_km2(region)
    land_area = geometry_area_km2(intersection)
    return {
        "RectangleArea_km2": rectangle_area,
        "AllLandArea_km2": land_area,
        "AllLandFractionOfRectangle": land_area / rectangle_area if rectangle_area > 0 else np.nan,
    }


def calculate_land_below_2000m_area_km2(dem_src, land_union,
                                         bounds: Dict[str, float]) -> Dict[str, float]:
    if dem_src.crs is None or not dem_src.crs.is_geographic:
        raise ValueError(f"DEM CRS must be geographic; got {dem_src.crs}")
    t = dem_src.transform
    if not np.isclose(t.b, 0.0) or not np.isclose(t.d, 0.0):
        raise ValueError("Rotated DEM grids are not supported.")

    raw = from_bounds(bounds["lon_min"], bounds["lat_min"],
                      bounds["lon_max"], bounds["lat_max"], transform=t)
    raw = raw.round_offsets().round_lengths()
    window = raw.intersection(Window(0, 0, dem_src.width, dem_src.height))

    dem = dem_src.read(1, window=window, masked=True)
    elevation = np.asarray(dem.filled(np.nan), dtype=float)
    if dem_src.nodata is not None:
        elevation[np.isclose(elevation, dem_src.nodata)] = np.nan

    wt = dem_src.window_transform(window)
    rows = np.arange(elevation.shape[0])
    cols = np.arange(elevation.shape[1])
    lon_centers = wt.c + (cols + 0.5) * wt.a
    lat_centers = wt.f + (rows + 0.5) * wt.e
    lon_centers = normalize_lon_to_180(lon_centers)
    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)

    in_region = (
        (lat_grid >= bounds["lat_min"]) & (lat_grid <= bounds["lat_max"]) &
        (lon_grid >= bounds["lon_min"]) & (lon_grid <= bounds["lon_max"])
    )
    land_mask = polygon_contains_xy(land_union, lon_grid.ravel(), lat_grid.ravel()).reshape(elevation.shape)
    valid_dem = np.isfinite(elevation)
    land_valid = in_region & land_mask & valid_dem
    below = land_valid & (elevation < ELEVATION_THRESHOLD_M)
    high = land_valid & (elevation >= ELEVATION_THRESHOLD_M)

    valid_area = below_area = high_area = 0.0
    for i, lat_c in enumerate(lat_centers):
        cell_area = spherical_cell_area_km2(float(lat_c), abs(wt.e), abs(wt.a))
        valid_area += np.sum(land_valid[i]) * cell_area
        below_area += np.sum(below[i]) * cell_area
        high_area += np.sum(high[i]) * cell_area

    return {
        "DEMLandValidArea_km2": float(valid_area),
        "LandBelow2000mArea_km2": float(below_area),
        "LandAtOrAbove2000mArea_km2": float(high_area),
        "Below2000mFractionOfDEMLand": below_area / valid_area if valid_area > 0 else np.nan,
        "HighElevationFractionOfDEMLand": high_area / valid_area if valid_area > 0 else np.nan,
        "DEMRows": int(elevation.shape[0]),
        "DEMColumns": int(elevation.shape[1]),
    }

# =============================================================================
# 7. DENSITY
# =============================================================================
def calculate_density(pf_count: int, area_km2: float) -> Dict[str, float]:
    if not np.isfinite(area_km2) or area_km2 <= 0:
        return {
            "PF_density_per_km2_per_year": np.nan,
            "PF_density_per_10000km2_per_year": np.nan,
            "PF_density_per_100000km2_per_year": np.nan,
        }
    d = float(pf_count) / float(area_km2) / OBSERVATION_YEARS
    return {
        "PF_density_per_km2_per_year": d,
        "PF_density_per_10000km2_per_year": d * 10000.0,
        "PF_density_per_100000km2_per_year": d * 100000.0,
    }

# =============================================================================
# 8. MAIN
# =============================================================================
def main() -> None:
    validate_paths()
    print("=" * 88)
    print("PF DENSITY: ALL LAND AND LAND BELOW 2000 M")
    print("=" * 88)
    land_union = load_land_union(LAND_SHAPEFILE)
    region_files = discover_region_csv_files()
    rows = []
    preview = []

    with rasterio.open(ELEVATION_TIF) as dem_src:
        print("[DEM]", ELEVATION_TIF)
        print("[DEM] CRS:", dem_src.crs)
        print("[DEM] Resolution:", dem_src.res)

        for region_name in tqdm(REGION_ORDER, desc="Calculating regional PF density"):
            paths = region_files.get(region_name, [])
            if not paths:
                continue
            bounds = REGIONS[region_name]
            pf_data = read_region_pf_files(paths, region_name)
            pf_info = count_region_pf_events(pf_data, region_name, bounds, land_union, dem_src)
            all_area = calculate_all_land_area_km2(land_union, bounds)
            low_area = calculate_land_below_2000m_area_km2(dem_src, land_union, bounds)

            all_density = calculate_density(pf_info["AllLandPFCount"], all_area["AllLandArea_km2"])
            low_density = calculate_density(pf_info["LandBelow2000mPFCount"], low_area["LandBelow2000mArea_km2"])

            common = {
                "Region": region_name,
                "Group": region_name[:2],
                "LatitudeRange": f"{bounds['lat_min']:g} to {bounds['lat_max']:g}",
                "LongitudeRange": f"{bounds['lon_min']:g} to {bounds['lon_max']:g}",
                "ObservationYears": OBSERVATION_YEARS,
                "ElevationThreshold_m": ELEVATION_THRESHOLD_M,
                "SourceFiles": "; ".join(p.name for p in paths),
                **pf_info,
                **all_area,
                **low_area,
            }
            rows.append({
                **common,
                "Version": "ALL_LAND",
                "PFCountUsed": pf_info["AllLandPFCount"],
                "AreaUsed_km2": all_area["AllLandArea_km2"],
                **all_density,
            })
            rows.append({
                **common,
                "Version": "LAND_BELOW_2000M",
                "PFCountUsed": pf_info["LandBelow2000mPFCount"],
                "AreaUsed_km2": low_area["LandBelow2000mArea_km2"],
                **low_density,
            })
            preview.append({
                "Region": region_name,
                "AllLandPFCount": pf_info["AllLandPFCount"],
                "AllLandArea_km2": all_area["AllLandArea_km2"],
                "AllLandDensity_per_10000km2_per_year": all_density["PF_density_per_10000km2_per_year"],
                "LandBelow2000mPFCount": pf_info["LandBelow2000mPFCount"],
                "LandBelow2000mArea_km2": low_area["LandBelow2000mArea_km2"],
                "Below2000mDensity_per_10000km2_per_year": low_density["PF_density_per_10000km2_per_year"],
            })

    result = pd.DataFrame(rows)
    result["Region"] = pd.Categorical(result["Region"], categories=REGION_ORDER, ordered=True)
    result["Version"] = pd.Categorical(result["Version"],
                                        categories=["ALL_LAND", "LAND_BELOW_2000M"], ordered=True)
    result = result.sort_values(["Region", "Version"]).reset_index(drop=True)

    all_land = result[result["Version"] == "ALL_LAND"].copy()
    below_2000 = result[result["Version"] == "LAND_BELOW_2000M"].copy()

    for df in (result, all_land, below_2000):
        float_cols = df.select_dtypes(include=[np.floating]).columns
        df[float_cols] = df[float_cols].round(8)

    result.to_csv(OUTPUT_COMBINED_CSV, index=False, encoding="utf-8-sig")
    all_land.to_csv(OUTPUT_ALL_LAND_CSV, index=False, encoding="utf-8-sig")
    below_2000.to_csv(OUTPUT_BELOW_2000M_CSV, index=False, encoding="utf-8-sig")

    print("\nRESULT PREVIEW")
    print(pd.DataFrame(preview).to_string(index=False))
    print("\n[OUTPUT]", OUTPUT_COMBINED_CSV)
    print("[OUTPUT]", OUTPUT_ALL_LAND_CSV)
    print("[OUTPUT]", OUTPUT_BELOW_2000M_CSV)


if __name__ == "__main__":
    main()