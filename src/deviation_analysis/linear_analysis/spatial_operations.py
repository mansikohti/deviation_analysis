import os
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union
from typing import Tuple, Dict

SQM_PER_HECTARE = 10000.0

def _safe_overlay(left: gpd.GeoDataFrame, right: gpd.GeoDataFrame, how: str) -> gpd.GeoDataFrame:
    """
    Perform spatial overlay with automatic geometry fixing.
    
    Args:
        left: Left GeoDataFrame
        right: Right GeoDataFrame
        how: Overlay method ('intersection', 'difference', etc.)
        
    Returns:
        Result of overlay operation
    """
    try:
        return gpd.overlay(left[["geometry"]], right[["geometry"]], how=how)
    except Exception:
        # Fix invalid geometries and retry
        left_fixed = gpd.GeoDataFrame(geometry=left.geometry.buffer(0), crs=left.crs)
        right_fixed = gpd.GeoDataFrame(geometry=right.geometry.buffer(0), crs=right.crs)
        return gpd.overlay(left_fixed, right_fixed, how=how)
    

def _save_overlay_results(
    results: Dict[str, gpd.GeoDataFrame],
    output_dir: str
) -> None:
    """Save overlay results as shapefiles and GeoJSON."""
    shp_folder = os.path.join(output_dir, "shapefile_outputs")
    geojson_folder = os.path.join(output_dir, "geojson_outputs")
    os.makedirs(shp_folder, exist_ok=True)
    os.makedirs(geojson_folder, exist_ok=True)

    for name, gdf in results.items():
        if gdf.empty:
            continue

        # Normalize geometry
        gdf = (
            gdf[["geometry"]]
            .explode(ignore_index=True)
            .dropna(subset=["geometry"])
            .drop_duplicates(subset="geometry")
            .reset_index(drop=True)
        )

        # Add attributes
        gdf["AREA_in_HA"] = gdf.geometry.area / SQM_PER_HECTARE
        gdf["AREA_NAME"] = [f"{name}_area_{i+1}" for i in range(len(gdf))]

        # Save files
        gdf.to_file(os.path.join(shp_folder, f"{name}.shp"))
        gdf.to_file(os.path.join(geojson_folder, f"{name}.geojson"), driver="GeoJSON")

def _create_summary_dataframe(
    actual: gpd.GeoDataFrame,
    planned: gpd.GeoDataFrame,
    area_type: str
) -> pd.DataFrame:
    """Create summary DataFrame with compliant and deviation areas."""
    # Explode to single parts
    actual_single = actual[["geometry"]].explode(ignore_index=True).copy()
    actual_single["geometry"] = actual_single.geometry.buffer(0)

    # Create union of planned areas
    planned_union = unary_union(planned.geometry)
    planned_empty = getattr(planned_union, "is_empty", False)

    summary_rows = []
    for i, geom in enumerate(actual_single.geometry):
        if geom is None or geom.is_empty:
            inter_area = 0.0
            actual_area = 0.0
        else:
            inter_area = 0.0 if planned_empty else geom.intersection(planned_union).area
            actual_area = geom.area

        compliant_ha = inter_area / SQM_PER_HECTARE
        deviation_ha = max(actual_area - inter_area, 0.0) / SQM_PER_HECTARE

        summary_rows.append({
            "AREA_TYPE": area_type,
            "AREA_NAME": f"{area_type}_area_{i+1}",
            "Compliant_Area (Ha)": compliant_ha,
            "Deviation_Area (Ha)": deviation_ha
        })

    return pd.DataFrame(summary_rows)

def generate_area_overlap(
    actual_file: gpd.GeoDataFrame,
    planned_file: gpd.GeoDataFrame,
    output_dir: str,
    area_type: str
) -> Tuple[Dict[str, gpd.GeoDataFrame], pd.DataFrame]:
    """
    Perform spatial overlay analysis between actual and planned areas.
    
    Args:
        actual_file: Actual area GeoDataFrame
        planned_file: Planned area GeoDataFrame
        output_dir: Output directory for results
        area_type: Type of area ('excavation' or 'dump')
        
    Returns:
        Tuple of (overlay_results_dict, summary_dataframe)
        
    Raises:
        ValueError: If geometry column missing or CRS invalid
    """
    # Validate inputs
    actual, planned = actual_file.copy(), planned_file.copy()
    
    if "geometry" not in actual.columns or "geometry" not in planned.columns:
        raise ValueError("Both GeoDataFrames must contain a 'geometry' column.")
    if actual.crs is None or planned.crs is None:
        raise ValueError("Both GeoDataFrames must have a valid CRS.")

    # Align CRS
    if actual.crs != planned.crs:
        planned = planned.to_crs(actual.crs)

    # Fix invalid geometries
    actual["geometry"] = actual.geometry.buffer(0)
    planned["geometry"] = planned.geometry.buffer(0)
    
    # Perform overlays
    planned_and_actual = _safe_overlay(planned, actual, "intersection")
    unplanned_and_actual = _safe_overlay(actual, planned, "difference")
    planned_not_actual = _safe_overlay(planned, actual, "difference")

    results = {
        f"planned_and_done_{area_type}": planned_and_actual,
        f"unplanned_and_done_{area_type}": unplanned_and_actual,
        f"planned_and_not_done_{area_type}": planned_not_actual,
    }

    # Save results
    _save_overlay_results(results, output_dir)

    # Create summary
    df = _create_summary_dataframe(actual, planned, area_type)

    return results, df