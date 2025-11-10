"""
Functions for processing elevation profiles from DTMs.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import Point, LineString
from shapely.ops import split

def get_elevation_profile(
    dtm_path1: str,
    dtm_path2: str,
    section_gdf: gpd.GeoDataFrame,
    section_number: int,
    interval: float = 0.1
) -> pd.DataFrame:
    """
    Get elevation profile along a section line from two DTMs.
    
    Args:
        dtm_path1: Path to first DTM
        dtm_path2: Path to second DTM
        section_gdf: GeoDataFrame containing section lines
        section_number: Index of section line to process
        interval: Sampling interval along section line
        
    Returns:
        DataFrame with columns:
        - chainage: Distance along section line
        - z_itr1: Elevation from first DTM
        - z_itr2: Elevation from second DTM
        - geometry: Point geometry at each sample location
    """
    # Extract single section line
    section = section_gdf.iloc[section_number]
    line = section.geometry
    
    # Create sampling points along line
    length = line.length
    distances = np.arange(0, length + interval, interval)
    points = [line.interpolate(d) for d in distances]
    coords = [(p.x, p.y) for p in points]
    
    # Sample DTMs
    with rasterio.open(dtm_path1) as src1, rasterio.open(dtm_path2) as src2:
        z1 = [z[0] for z in src1.sample(coords)]
        z2 = [z[0] for z in src2.sample(coords)]
    
    # Create output dataframe
    df = pd.DataFrame({
        "chainage": distances,
        "z_itr1": z1,
        "z_itr2": z2,
        "x": [x for x, y in coords],
        "y": [y for x, y in coords],
        "geometry": [Point(x, y) for x, y in coords]
    })
    
    # Add metadata columns
    df["line_name"] = ""  # Will be populated by classification functions
    df["area_name"] = ""  # Will be populated by classification functions
    
    return df