import os
import glob
import re
import rasterio
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString, GeometryCollection
from typing import Optional, Tuple



def get_linear_outputs(output_dir):
    """
    Reads specific shapefile outputs from a given directory and returns them as GeoDataFrames.

    Args:
        output_dir (str): Base directory containing 'shapefile_outputs' folder.

    Returns:
        tuple: (planned_exc_gdf, unplanned_exc_gdf, planned_dump_gdf, unplanned_dump_gdf)
               Each is a GeoDataFrame or None if the file doesn't exist.
    """
    shape_folder = os.path.join(output_dir, "shapefile_outputs")

    base_names = [
        "planned_and_done_excavation",
        "unplanned_and_done_excavation",
        "planned_and_done_dump",
        "unplanned_and_done_dump",
    ]

    gdf_results = []

    for base in base_names:
        pattern = os.path.join(shape_folder, f"{base}.shp")
        matches = glob.glob(pattern)

        if matches:
            try:
                gdf = gpd.read_file(matches[0])
                gdf_results.append(gdf)
            except Exception as e:
                print(f"Error reading {base}: {e}")
                gdf_results.append(None)
        else:
            gdf_results.append(None)

    return tuple(gdf_results)

def elevation_at_point(dtm_src, x: float, y: float) -> float:
    """
    Returns the elevation value from an opened DTM raster at (x, y).

    Args:
        dtm_src (rasterio.DatasetReader): Open raster dataset.
        x (float): X coordinate.
        y (float): Y coordinate.

    Returns:
        float: Elevation value at the given coordinate.
    """
    try:
        return list(dtm_src.sample([(x, y)]))[0][0]
    except Exception as e:
        print(f"Error fetching elevation at ({x}, {y}): {e}")
        return float("nan")


def get_section_elevation_profiles(
    dtm_path1: str,
    dtm_path2: str,
    section_gdf: gpd.GeoDataFrame,
    section_number: int,
    interval: float = 0.1
) -> pd.DataFrame:
    """
    Samples elevation values along a section line from two DTM rasters.

    Args:
        dtm_path1 (str): Path to the first DTM raster (Iteration 1).
        dtm_path2 (str): Path to the second DTM raster (Iteration 2).
        section_gdf (GeoDataFrame): GeoDataFrame containing section LineStrings.
        section_number (int): Index of the section to process.
        interval (float, optional): Distance interval between sample points. Defaults to 0.01.

    Returns:
        pd.DataFrame: DataFrame containing ['chainage', 'x', 'y', 'z_itr1', 'z_itr2'].
    """
    # Extract target line
    try:
        line_gdf = section_gdf.iloc[[section_number]]
        geom = line_gdf.geometry.iloc[0]
    except Exception as e:
        raise ValueError(f"Invalid section index {section_number}: {e}")

    if not isinstance(geom, LineString):
        raise TypeError(f"Selected geometry at index {section_number} is not a LineString.")

    # Create distance intervals
    length = geom.length
    distances = [i * interval for i in range(int(length // interval) + 1)]
    if distances[-1] < length:
        distances.append(length)

    rows = []

    with rasterio.open(dtm_path1) as dtm1, rasterio.open(dtm_path2) as dtm2:
        # Reproject if CRS mismatches
        if line_gdf.crs and line_gdf.crs != dtm1.crs:
            print("Reprojecting section line to match DTM CRS...")
            line_gdf = line_gdf.to_crs(dtm1.crs)
            geom = line_gdf.geometry.iloc[0]

        for d in distances:
            pt = geom.interpolate(d)
            x, y = pt.x, pt.y
            z1 = elevation_at_point(dtm1, x, y)
            z2 = elevation_at_point(dtm2, x, y)
            rows.append((d, x, y, z1, z2))

    return pd.DataFrame(rows, columns=["chainage", "x", "y", "z_itr1", "z_itr2"])



def intersect_section_line_with_polygons(
            section_gdf: gpd.GeoDataFrame,
            section_number: int,
            output_folder_path: str,
            planned_and_done_excavation_gdf: Optional[gpd.GeoDataFrame] = None,
            unplanned_and_done_excavation_gdf: Optional[gpd.GeoDataFrame] = None,
            planned_and_used_dump_gdf: Optional[gpd.GeoDataFrame] = None,
            unplanned_and_used_dump_gdf: Optional[gpd.GeoDataFrame] = None,
            ) -> Tuple[Optional[gpd.GeoDataFrame], Optional[gpd.GeoDataFrame], Optional[gpd.GeoDataFrame], Optional[gpd.GeoDataFrame]]:
    """
    Intersect a single section line with four polygon layers and return
    individual GeoDataFrames. Also writes each result to a shapefile inside
    output_folder_path as '<name>_intersections.shp'.

    Returns tuple in same order as the input polygon args.
    """
    os.makedirs(output_folder_path, exist_ok=True)

    # validate section index
    if section_number < 0 or section_number >= len(section_gdf):
        raise IndexError(f"section_number {section_number} out of range (0..{len(section_gdf)-1})")

    line_gdf = section_gdf.iloc[[section_number]]
    line_geom = line_gdf.geometry.iloc[0]
    if not isinstance(line_geom, LineString):
        raise TypeError(f"Selected geometry at index {section_number} is not a LineString")

    def _write_empty_gdf(crs, out_fp):
        empty = gpd.GeoDataFrame(geometry=[], crs=crs)
        empty.to_file(out_fp)
        return empty

    def process(name: str, poly_gdf: Optional[gpd.GeoDataFrame]) -> Optional[gpd.GeoDataFrame]:
        out_fp = os.path.join(output_folder_path, f"{name}_intersections.shp")

        # missing file -> write empty and return None or empty GDF
        if poly_gdf is None:
            print(f"{name}: input is None — creating empty output: {out_fp}")
            return _write_empty_gdf(line_gdf.crs, out_fp)

        if poly_gdf.empty:
            print(f"{name}: polygon GeoDataFrame is empty — creating empty output: {out_fp}")
            return _write_empty_gdf(line_gdf.crs, out_fp)

        # ensure same CRS (use line CRS as canonical)
        if poly_gdf.crs != line_gdf.crs:
            poly_gdf = poly_gdf.to_crs(line_gdf.crs)

        # union all polygons into single geometry for faster intersection
        try:
            poly_union = poly_gdf.geometry.unary_union  # shapely geometry
        except Exception as e:
            print(f"{name}: failed to build union: {e}")
            return _write_empty_gdf(line_gdf.crs, out_fp)

        inter = line_geom.intersection(poly_union)

        segments = []

        # handle possible geometry types returned by intersection
        if inter.is_empty:
            print(f"{name}: No intersection.")
        elif isinstance(inter, LineString):
            segments.append(inter)
        elif isinstance(inter, MultiLineString):
            segments.extend(list(inter.geoms))
        elif isinstance(inter, GeometryCollection):
            # extract any lines within the collection
            for g in inter.geoms:
                if isinstance(g, (LineString, MultiLineString)):
                    if isinstance(g, LineString):
                        segments.append(g)
                    else:
                        segments.extend(list(g.geoms))
                    
        else:
            # could be a Point or other type — ignore silently but log
            print(f"{name}: intersection produced unsupported geometry type: {type(inter)}")

        # Print after processing all geometry types
        if segments:
            print(f"{name}: intersection present ({len(segments)} segment(s))")

        # Create GeoDataFrame in line CRS
        intersect_gdf = gpd.GeoDataFrame(geometry=segments, crs=line_gdf.crs)

        if not intersect_gdf.empty:
            # join attributes from polygons: spatial join with 'intersects'
            # ensure polygon has the attribute AREA_NAME (or use a fallback)
            join_cols = [c for c in poly_gdf.columns if c != poly_gdf.geometry.name]
            attr_gdf = poly_gdf[join_cols] if join_cols else poly_gdf[['geometry']]
            try:
                intersect_gdf = gpd.sjoin(intersect_gdf, poly_gdf[[*join_cols, 'geometry']] if join_cols else poly_gdf[['geometry']], how="left", predicate="intersects")
            except Exception as e:
                # older geopandas versions might not support 'predicate' argument name
                try:
                    intersect_gdf = gpd.sjoin(intersect_gdf, poly_gdf[[*join_cols, 'geometry']] if join_cols else poly_gdf[['geometry']], how="left", op="intersects")
                except Exception as ee:
                    print(f"{name}: spatial join failed: {e} / {ee}")

            # cleanup typical sjoin columns
            for col in ('index_right', ):
                if col in intersect_gdf.columns:
                    intersect_gdf = intersect_gdf.drop(columns=[col])

        # persist result
        try:
            intersect_gdf.to_file(out_fp)
        except Exception as e:
            print(f"{name}: failed to write {out_fp}: {e}")

        return intersect_gdf

    a = process("planned_and_done_excavation", planned_and_done_excavation_gdf)
    b = process("unplanned_and_done_excavation", unplanned_and_done_excavation_gdf)
    c = process("planned_and_used_dump", planned_and_used_dump_gdf)
    d = process("unplanned_and_used_dump", unplanned_and_used_dump_gdf)

    return (a, b, c, d)

def get_planned_area(
    df: pd.DataFrame,
    line_gdf,
    key: str,
):
    """
    Maps planned excavation/dump areas along a section line.
    Assigns line_name, area_name, and label_name columns to df 
    based on intersections with geometries in line_gdf.

    Args:
        df (pd.DataFrame): DataFrame with 'x', 'y', 'chainage' columns.
        line_gdf (GeoDataFrame): GeoDataFrame containing 'AREA_NAME' polygons or lines.
        key (str): Identifier for the current feature type (e.g. 'planned_and_done_excavation').

    Returns:
        pd.DataFrame: Updated DataFrame with 'line_name', 'area_name', and 'label_name' columns.
    """

    # Ensure required columns exist with correct dtype
    if "line_name" not in df.columns:
        df["line_name"] = np.nan
    # ensure dtype is object so string assignment won’t throw warning
    if df["line_name"].dtype != "object":
        df["line_name"] = df["line_name"].astype("object")

    if "area_name" not in df.columns:
        df["area_name"] = pd.Series([None] * len(df), dtype="object")

    if "label_name" not in df.columns:
        df["label_name"] = pd.Series([None] * len(df), dtype="object")

    # Early exits for missing or empty GeoDataFrame
    if line_gdf is None:
        print(f"Skipped: line_gdf is None for key '{key}'.")
        return df
    if line_gdf.empty:
        print(f"Skipped: line_gdf has no features for key '{key}'.")
        return df

    if "AREA_NAME" not in line_gdf.columns:
        raise ValueError("line_gdf must contain an 'AREA_NAME' column.")

    # Build the full section line
    section_line = LineString(df[['x', 'y']].to_numpy())
    chainages = df['chainage'].to_numpy()

    for idx, geom in line_gdf.iterrows():
        line = geom.geometry
        area_name = geom.get("AREA_NAME", None)

        if not isinstance(line, LineString):
            continue  # skip non-LineString geometries

        # Start & end points of the line
        start_pt = Point(line.coords[0])
        end_pt = Point(line.coords[-1])

        # Project onto section line
        start_chain_exact = section_line.project(start_pt)
        end_chain_exact = section_line.project(end_pt)

        # Snap to nearest sampled chainages
        start_chain = chainages[np.argmin(np.abs(chainages - start_chain_exact))]
        end_chain = chainages[np.argmin(np.abs(chainages - end_chain_exact))]

        # Ensure start <= end
        start_chain, end_chain = sorted([start_chain, end_chain])

        # Mask rows in section_df within this range
        mask = (df["chainage"] >= start_chain) & (df["chainage"] <= end_chain)

        # Assign values safely
        df.loc[mask, "line_name"] = key
        df.loc[mask, "area_name"] = area_name

        # Derive label_name from area_name (Excavation Area i / Dump Area i)
        label_value = None
        if isinstance(area_name, str):
            lower = area_name.lower()
            kind = None
            if "excavation" in lower:
                kind = "Excavation Area"
            elif "dump" in lower:
                kind = "Dump Area"

            # find the last integer in the string (e.g., area_23 -> 23)
            m = re.search(r'(\d+)(?!.*\d)', area_name)
            if kind and m:
                label_value = f"{kind} {m.group(1)}"
            elif kind:
                label_value = kind

        df.loc[mask, "label_name"] = label_value

    return df


def get_unplanned_area(
    df: pd.DataFrame,
    line_gdf,
    key: str,
    threshold: float,
):
    """
    Label unplanned areas only if max abs(z_itr1 - z_itr2) in the area exceeds `threshold`.
    Also create/populate `label_name` like "Excavation Area <i>" or "Dump Area <i>" (case-sensitive).
    """

    # Ensure required elevation columns exist
    required = {"chainage", "x", "y", "z_itr1", "z_itr2"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"section_df missing required columns: {missing}")

    # Ensure line_name, area_name, label_name columns exist with object dtype
    if "line_name" not in df.columns:
        df["line_name"] = pd.Series([None] * len(df), dtype="object")
    elif df["line_name"].dtype != "object":
        df["line_name"] = df["line_name"].astype("object")

    if "area_name" not in df.columns:
        df["area_name"] = pd.Series([None] * len(df), dtype="object")
    elif df["area_name"].dtype != "object":
        df["area_name"] = df["area_name"].astype("object")

    if "label_name" not in df.columns:
        df["label_name"] = pd.Series([None] * len(df), dtype="object")
    elif df["label_name"].dtype != "object":
        df["label_name"] = df["label_name"].astype("object")

    # Early exit if no line_gdf provided
    if line_gdf is None:
        print(f"Skipped: line_gdf is None for key '{key}'.")
        return df
    if getattr(line_gdf, "empty", False):
        print(f"Skipped: line_gdf has no features for key '{key}'.")
        return df

    has_area_name = "AREA_NAME" in line_gdf.columns

    # Build the full section line from the sampled coordinates
    section_line = LineString(df[['x', 'y']].to_numpy())
    chainages = df['chainage'].to_numpy()

    def derive_label(area_name: str) -> str:
        if not isinstance(area_name, str):
            return None
        lower = area_name.lower()
        kind = None
        if "excavation" in lower:
            kind = "Excavation Area"
        elif "dump" in lower:
            kind = "Dump Area"
        m = re.search(r'(\d+)(?!.*\d)', area_name)
        if kind and m:
            return f"{kind} {m.group(1)}"
        if kind:
            return kind
        return None

    for idx, row in line_gdf.iterrows():
        line = row.geometry
        if not isinstance(line, LineString):
            continue

        area_name = row["AREA_NAME"] if has_area_name else None

        # Start & end points of the blue line
        start_pt = Point(line.coords[0])
        end_pt = Point(line.coords[-1])

        # Project onto section line -> exact continuous chainage
        start_chain_exact = section_line.project(start_pt)
        end_chain_exact = section_line.project(end_pt)

        # Snap to nearest sampled chainages in section_df
        start_chain = chainages[np.argmin(np.abs(chainages - start_chain_exact))]
        end_chain = chainages[np.argmin(np.abs(chainages - end_chain_exact))]

        # Ensure start <= end
        start_chain, end_chain = sorted([start_chain, end_chain])

        # Mask rows in section_df within this range
        mask = (df["chainage"] >= start_chain) & (df["chainage"] <= end_chain)

        # Compute absolute diffs for this range
        diffs = (df.loc[mask, "z_itr1"] - df.loc[mask, "z_itr2"]).abs()

        if (diffs > threshold).any():
            # Significant deviation: keep the label and set area_name & label_name
            df.loc[mask, "line_name"] = key
            df.loc[mask, "area_name"] = area_name
            df.loc[mask, "label_name"] = derive_label(area_name)
        else:
            # No significant deviation: clear any label/area_name/label_name for this range
            df.loc[mask, "line_name"] = None
            df.loc[mask, "area_name"] = None
            df.loc[mask, "label_name"] = None
            print(f"Removed label for '{key}' at feature {idx}: no elevation diff > {threshold}")


    return df
