import glob
import sys
import os
import geopandas as gpd
import numpy as np
from tkinter import filedialog,messagebox
import tkinter as tk
import pandas as pd
from shapely.geometry import Point, LineString, MultiLineString,GeometryCollection
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Callable, Dict, List, Optional, Union
import re
from pathlib import Path
from shapely.geometry.base import BaseGeometry
import traceback

############# new functions ###########################
# def sanitize(name: str) -> str:
#     """Make the folder name safe by removing special characters."""
#     name = re.sub(r'[^A-Za-z0-9_-]+', '_', str(name))
#     return name.strip('_')


def create_sub_folder(section_gdf, section_number, sectional_analysis_folder):

    # get section names 
    row = section_gdf.iloc[section_number]
    section_name_from_row = f"{row.get('start_text', '')}_{row.get('end_text', '')}"
    readable_section_name = section_name_from_row
    print(readable_section_name)

    ## create the subfolders
    sub_section_folder = os.path.join(sectional_analysis_folder, readable_section_name)
    os.makedirs(sub_section_folder,exist_ok=True)
    related_folder = os.path.join(sub_section_folder, "related_files")
    os.makedirs(related_folder, exist_ok = True)

    return sub_section_folder, related_folder, section_name_from_row
























############# old funtions ##########################################

def _elevation_at_point(dtm_src: rasterio.io.DatasetReader, x: float, y: float) -> float:
    """
    Query elevation value from an opened DTM raster at (x, y).

    Parameters
    ----------
    dtm_src : rasterio.io.DatasetReader
        Opened raster dataset.
    x, y : float
        Coordinates for which elevation is queried.

    Returns
    -------
    float
        Elevation value from raster at (x, y).
    """
    value = list(dtm_src.sample([(x, y)]))[0][0]
    return float(value)

def _generate_chainages(length: float, interval: float) -> list:
    """Generate chainage (distance) values along a line at given interval."""
    if interval <= 0:
        raise ValueError("Interval must be positive.")
    chainages = [i * interval for i in range(int(length // interval) + 1)]
    if chainages[-1] < length:
        chainages.append(length)
    return chainages

def _ensure_crs_match(section: gpd.GeoDataFrame, dtm_src: rasterio.io.DatasetReader) -> gpd.GeoDataFrame:
    """Reproject section to match DTM CRS if needed."""
    if section.crs and section.crs != dtm_src.crs:
        return section.to_crs(dtm_src.crs)
    return section

def get_elevation_profile(
    dtm_path_itr1: str,
    dtm_path_itr2: str,
    section_gdf: gpd.GeoDataFrame,
    section_index: int,
    interval: float 
) -> pd.DataFrame:
    """
    Generate an elevation profile by sampling two DTMs along a section line.

    Parameters
    ----------
    dtm_path_itr1 : str
        Path to the first DTM (e.g., Iteration 1).
    dtm_path_itr2 : str
        Path to the second DTM (e.g., Iteration 2).
    section_gdf : gpd.GeoDataFrame
        GeoDataFrame containing section LineString geometries.
    section_index : int
        Index of the section to process within the GeoDataFrame.
    interval : float, optional
        Sampling interval (default 0.01 map units).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['chainage', 'x', 'y', 'z_itr1', 'z_itr2'].
    """
    # Validate input ---
    if section_index >= len(section_gdf):
        raise IndexError(f"Section index {section_index} out of range (max {len(section_gdf)-1}).")

    geometry = section_gdf.geometry.iloc[section_index]
    if not isinstance(geometry, LineString):
        raise TypeError("Selected feature is not a LineString geometry.")

    # Compute distances along line 
    length = geometry.length
    distances = _generate_chainages(length, interval)

    # fetcg elevation from DTM
    rows = []
    with rasterio.open(dtm_path_itr1) as dtm1, rasterio.open(dtm_path_itr2) as dtm2:
        # Reproject section line if needed
        geometry = _ensure_crs_match(section_gdf.iloc[[section_index]], dtm1).geometry.iloc[0]

        for d in distances:
            x, y = geometry.interpolate(d).coords[0]
            z1 = _elevation_at_point(dtm1, x, y)
            z2 = _elevation_at_point(dtm2, x, y)
            rows.append((d, x, y, z1, z2))

    return pd.DataFrame(rows, columns=["chainage", "x", "y", "z_itr1", "z_itr2"])



def extract_section_intersections(
    line_gdf: gpd.GeoDataFrame,
    output_folder_path: str,
    planned_and_done_excavation_path: Optional[str] = None,
    unplanned_and_done_excavation_path: Optional[str] = None,
    planned_and_used_dump_path: Optional[str] = None,
    unplanned_and_used_dump_path: Optional[str] = None,
):
    """
    Intersect a single line feature with up to four polygon layers and write each set of
    intersecting line segments to separate shapefiles.

    Returns
    -------
    (gdf1, gdf2, gdf3, gdf4), section_name
    """

    # validations
    if line_gdf is None or len(line_gdf) != 1:
        raise ValueError("Line GeoDataFrame must contain exactly one feature.")

    if line_gdf.geometry.is_empty.any():
        raise ValueError("Input line has empty geometry.")

    if not isinstance(line_gdf.geometry.iloc[0], LineString):
        raise TypeError("Input feature must be a LineString.")

    if line_gdf.crs is None:
        raise ValueError("Line GeoDataFrame must have a valid CRS.")

    # Section name (keep your original field usage & format)
    row = line_gdf.iloc[0]
    section_name = f"{row.get('start_text', '')}_{row.get('end_text', '')}"

    out_dir = Path(output_folder_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    line_geom: LineString = line_gdf.geometry.iloc[0]

    
    # helper functions
    def _output_path(layer_name: str) -> Path:
        # sanitize filename a bit
        safe_section = str(section_name).replace("/", "-").replace("\\", "-")
        return out_dir / f"section_{safe_section}_{layer_name}.shp"

    def _empty_gdf() -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(geometry=[], crs=line_gdf.crs)

    def _extract_line_parts(geom: BaseGeometry) -> List[LineString]:
        """
        Normalize intersection result into a list of LineStrings.
        Accepts LineString, MultiLineString, or GeometryCollection (possibly nested).
        """
        if geom.is_empty:
            return []

        if isinstance(geom, LineString):
            return [geom]

        if isinstance(geom, MultiLineString):
            return [g for g in geom.geoms if not g.is_empty]

        if isinstance(geom, GeometryCollection):
            parts: List[LineString] = []
            for g in geom.geoms:
                parts.extend(_extract_line_parts(g))
            return parts

        # other geometry types (e.g., Point/Polygon) don't contribute line segments
        return []

    def _read_polygon_layer(path: Optional[str]) -> Optional[gpd.GeoDataFrame]:
        """Read polygon layer if path exists; return None if not usable."""
        if not path or not os.path.exists(path):
            return None
        gdf = gpd.read_file(path)
        if gdf.empty or gdf.geometry.is_empty.all():
            return None
        return gdf

    def _reproject_to_line_crs(poly: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        if poly.crs != line_gdf.crs:
            return poly.to_crs(line_gdf.crs)
        return poly

    def _intersect_and_save(layer_name: str, poly_path: Optional[str]) -> gpd.GeoDataFrame:
        """
        Intersect `line_geom` with polygons at `poly_path`, join AREA_NAME if available,
        and save to shapefile. Returns the resulting GeoDataFrame (possibly empty).
        """
        dst = _output_path(layer_name)

        poly_gdf = _read_polygon_layer(poly_path)
        if poly_gdf is None:
            empty = _empty_gdf()
            empty.to_file(dst)
            return empty

        # ensure CRS match
        poly_gdf = _reproject_to_line_crs(poly_gdf)

        # union polygons (use unary_union for broad compatibility)
        poly_union = poly_gdf.unary_union

        # intersection
        inter_geom = line_geom.intersection(poly_union)
        line_parts = _extract_line_parts(inter_geom)

        if not line_parts:
            empty = _empty_gdf()
            empty.to_file(dst)
            return empty

        out_gdf = gpd.GeoDataFrame(geometry=line_parts, crs=line_gdf.crs)

        # attach AREA_NAME if present (spatial join)
        if "AREA_NAME" in poly_gdf.columns:
            out_gdf = out_gdf.reset_index(drop=True)
            out_gdf["__tmpid"] = out_gdf.index
            joined = gpd.sjoin(
                out_gdf,
                poly_gdf[["geometry", "AREA_NAME"]],
                how="left",
                predicate="intersects",
            ).drop(columns=["index_right"])
            out_gdf = joined.drop(columns="__tmpid", errors="ignore")

        # save
        out_gdf.to_file(dst)
        return out_gdf

    # process all four layers    
    lines_planned_and_done_excavation = _intersect_and_save(
        "planned_and_done_excavation", planned_and_done_excavation_path
    )
    lines_unplanned_and_done_excavation = _intersect_and_save(
        "unplanned_and_done_excavation", unplanned_and_done_excavation_path
    )
    lines_planned_and_used_dump = _intersect_and_save(
        "planned_and_used_dump", planned_and_used_dump_path
    )
    lines_unplanned_and_used_dump = _intersect_and_save(
        "unplanned_and_used_dump", unplanned_and_used_dump_path
    )

    return (
        lines_planned_and_done_excavation,
        lines_unplanned_and_done_excavation,
        lines_planned_and_used_dump,
        lines_unplanned_and_used_dump,
    ), section_name




def _ensure_label_columns_exist(df):
    """Ensure required label columns exist in DataFrame."""
    if "line_name" not in df.columns:
        df["line_name"] = np.nan
    if "area_name" not in df.columns:
        df["area_name"] = pd.Series([None] * len(df), dtype="object")
    if "label_name" not in df.columns:
        df["label_name"] = pd.Series([None] * len(df), dtype="object")


def _validate_elevation_columns(df):
    """Validate that elevation columns exist for threshold mode."""
    required = {"chainage", "x", "y", "z_itr1", "z_itr2"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing required columns for threshold mode: {missing}")


def _validate_line_gdf(gdf, key):
    """Validate line_gdf and return True if valid."""
    if gdf is None:
        print(f"Skipped: line_gdf is None for key '{key}'.")
        return False
    
    if getattr(gdf, "empty", False):
        print(f"Skipped: line_gdf has no features for key '{key}'.")
        return False
    
    if "AREA_NAME" not in gdf.columns:
        raise ValueError("line_gdf must contain an 'AREA_NAME' column.")
    
    return True


def _process_line_feature(section_df, line, area_name, key, idx, section_line, chainages, threshold):
    """Process a single line feature and update section_df."""
    start_chain, end_chain = _get_chainage_range(line, section_line, chainages)
    mask = (section_df["chainage"] >= start_chain) & (section_df["chainage"] <= end_chain)
    
    if threshold is not None and not _meets_threshold(section_df, mask, threshold):
        _clear_labels(section_df, mask)
        print(f"Removed label for '{key}' at feature {idx}: no elevation diff > {threshold}")
        return
    
    _apply_labels(section_df, mask, key, area_name)


def _get_chainage_range(line, section_line, chainages):
    """Calculate start and end chainage for a line feature."""
    start_pt = Point(line.coords[0])
    end_pt = Point(line.coords[-1])
    
    start_chain_exact = section_line.project(start_pt)
    end_chain_exact = section_line.project(end_pt)
    
    start_chain = chainages[np.argmin(np.abs(chainages - start_chain_exact))]
    end_chain = chainages[np.argmin(np.abs(chainages - end_chain_exact))]
    
    return sorted([start_chain, end_chain])


def _meets_threshold(df, mask, threshold):
    """Check if any elevation difference exceeds threshold."""
    diffs = (df.loc[mask, "z_itr1"] - df.loc[mask, "z_itr2"]).abs()
    return (diffs > threshold).any()


def _clear_labels(df, mask):
    """Clear all label columns for masked rows."""
    df.loc[mask, "line_name"] = np.nan
    df.loc[mask, "area_name"] = np.nan
    df.loc[mask, "label_name"] = np.nan


def _apply_labels(df, mask, key, area_name):
    """Apply labels to masked rows."""
    df.loc[mask, "line_name"] = key
    df.loc[mask, "area_name"] = area_name
    df.loc[mask, "label_name"] = _derive_label_name(area_name)


def _derive_label_name(area_name):
    """
    Derive clean label from area_name.
    
    Examples:
        'excavation_area_23' -> 'Excavation Area 23'
        'dump_area_5' -> 'Dump Area 5'
    """
    if not isinstance(area_name, str):
        return None
    
    lower = area_name.lower()
    
    kind = None
    if "excavation" in lower:
        kind = "Excavation Area"
    elif "dump" in lower:
        kind = "Dump Area"
    
    if not kind:
        return None
    
    match = re.search(r'(\d+)(?!.*\d)', area_name)
    
    if match:
        return f"{kind} {match.group(1)}"
    
    return kind


def add_labels_for_area(section_df, line_gdf, key, threshold=None, output_csv=None):
    """
    Label areas on section_df based on line_gdf geometries.
    
    Planned mode (threshold=None): Labels all areas.
    Unplanned mode (threshold set): Labels only areas where elevation change exceeds threshold.
    
    Parameters
    ----------
    section_df : pd.DataFrame
        DataFrame with columns: chainage, x, y, z_itr1, z_itr2
    line_gdf : gpd.GeoDataFrame
        GeoDataFrame with LineString geometries and AREA_NAME column
    key : str
        Identifier for the line_name column
    threshold : float, optional
        If provided, only label areas with |z_itr1 - z_itr2| > threshold
    output_csv : str, optional
        Path to save the updated DataFrame
    
    Returns
    -------
    pd.DataFrame
        Updated section_df with line_name, area_name, and label_name columns
    """
    _ensure_label_columns_exist(section_df)
    
    if threshold is not None:
        _validate_elevation_columns(section_df)
    
    if not _validate_line_gdf(line_gdf, key):
        return section_df
    
    section_line = LineString(section_df[['x', 'y']].to_numpy())
    chainages = section_df['chainage'].to_numpy()
    
    for idx, row in line_gdf.iterrows():
        if not isinstance(row.geometry, LineString):
            continue
        
        _process_line_feature(
            section_df=section_df,
            line=row.geometry,
            area_name=row.get("AREA_NAME"),
            key=key,
            idx=idx,
            section_line=section_line,
            chainages=chainages,
            threshold=threshold
        )
    
    if output_csv:
        section_df.to_csv(output_csv, index=False)
        print(f"Updated DataFrame exported to: {output_csv}")
    
    return section_df

