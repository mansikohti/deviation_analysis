import os
import geopandas as gpd
from typing import Tuple


SQM_PER_HECTARE = 10000.0


def _clean_gdf(input_path: str, prefix: str, out_name: str, cleaned_folder: str) -> gpd.GeoDataFrame:
    """
    Clean and standardize a GeoDataFrame.
    
    Args:
        input_path: Path to input shapefile
        prefix: Prefix for area names
        out_name: Output filename (without extension)
        cleaned_folder: Output directory
        
    Returns:
        Cleaned GeoDataFrame
        
    Raises:
        ValueError: If geometry column is missing
    """
    gdf = gpd.read_file(input_path)

    # Validate geometry column
    if "geometry" not in gdf.columns:
        raise ValueError(f"'geometry' column not found in {input_path}")
    
    # Keep only geometry and process
    gdf = (
        gdf[["geometry"]]
        .explode(index_parts=True, ignore_index=True)
        .drop_duplicates(subset="geometry")
        .reset_index(drop=True)
    )
    
    # Add attributes
    gdf["AREA_ha"] = gdf.geometry.area / SQM_PER_HECTARE
    gdf["AREA_NAME"] = [f"{prefix}_{i+1}" for i in range(len(gdf))]

    # Save cleaned data
    out_path = os.path.join(cleaned_folder, f"{out_name}.shp")
    gdf.to_file(out_path)

    return gdf

def get_data_preprocessed(
    act_exv: str,
    act_dump: str,
    pln_exv: str,
    pln_dump: str,
    output_dir: str
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, str]:
    """
    Preprocess actual and planned excavation/dump shapefiles.
    
    Args:
        act_exv: Path to actual excavation shapefile
        act_dump: Path to actual dump shapefile
        pln_exv: Path to planned excavation shapefile
        pln_dump: Path to planned dump shapefile
        output_dir: Output directory for cleaned files
        
    Returns:
        Tuple of (actual_exv, actual_dump, planned_exv, planned_dump, epsg_code)
    """
    # Create output subfolder
    sub_folder = os.path.join(output_dir, "preprocessed_inputs")
    os.makedirs(sub_folder, exist_ok=True)

    # Clean actual data
    clean_act_exv = _clean_gdf(act_exv, "actual_excavation_area", "actual_excavation", sub_folder)
    clean_act_dump = _clean_gdf(act_dump, "actual_dump_area", "actual_dump", sub_folder)

    # Clean planned data
    clean_pln_exv = _clean_gdf(pln_exv, "planned_excavation_area", "planned_excavation", sub_folder)
    clean_pln_dump = _clean_gdf(pln_dump, "planned_dump_area", "planned_dump", sub_folder)

    # Extract CRS
    gdf_check = gpd.read_file(act_exv)
    epsg_code = gdf_check.crs.to_string() if gdf_check.crs else None

    return clean_act_exv, clean_act_dump, clean_pln_exv, clean_pln_dump, epsg_code