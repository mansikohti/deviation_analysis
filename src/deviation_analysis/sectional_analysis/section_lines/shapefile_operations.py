import os
import re
import shutil
import numpy as np
import geopandas as gpd


def clean_shapefile_text_fields_keep_only_ends(shp_path, start_col="start_text",
                                               end_col="end_text", backup=True,
                                               verbose=True):
    """Clean shapefile text fields and keep only start/end columns."""
    if not os.path.exists(shp_path):
        raise FileNotFoundError(shp_path)

    gdf = gpd.read_file(shp_path)
    if verbose:
        print(f"Loaded '{shp_path}' ({len(gdf)} features).")

    missing = [c for c in (start_col, end_col) if c not in gdf.columns]
    if missing:
        raise KeyError(f"Required column(s) missing: {missing}")

    main_pattern = re.compile(r'\{[^;]*;([^}]*)\}')
    token_pattern = re.compile(r"[A-Za-z0-9']+")

    def extract_label(value):
        """Extract clean label from text value."""
        if value is None:
            return value
        if isinstance(value, float) and np.isnan(value):
            return value
        
        s = str(value).strip()
        if len(s) <= 2:
            return s
        
        m = main_pattern.search(s)
        if m:
            return m.group(1).strip()
        
        tokens = token_pattern.findall(s)
        if tokens:
            return tokens[-1].strip()
        
        s2 = re.sub(r'[\{\}\\]', '', s)
        if ';' in s2:
            return s2.split(';')[-1].strip()
        return s2.strip()

    gdf[f"{start_col}_orig"] = gdf[start_col]
    gdf[f"{end_col}_orig"] = gdf[end_col]

    gdf[start_col] = gdf[start_col].apply(extract_label)
    gdf[end_col] = gdf[end_col].apply(extract_label)

    if verbose:
        for col in (start_col, end_col):
            orig = gdf[f"{col}_orig"].fillna("__NA__").astype(str)
            new = gdf[col].fillna("__NA__").astype(str)
            n_changed = int((orig != new).sum())
            print(f"Column '{col}': {n_changed} changed cells.")

    if backup:
        base_dir = os.path.dirname(shp_path) or "."
        base_name = os.path.splitext(os.path.basename(shp_path))[0]
        bak_dir = os.path.join(base_dir, f"{base_name}_bak")
        os.makedirs(bak_dir, exist_ok=True)
        
        exts = [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix"]
        for ext in exts:
            p = os.path.join(base_dir, base_name + ext)
            if os.path.exists(p):
                dst = os.path.join(bak_dir, os.path.basename(p))
                shutil.copy2(p, dst)
        
        if verbose:
            print(f"Backup written to: {bak_dir}")

    cols_to_keep = [start_col, end_col, gdf.geometry.name]
    if gdf.geometry.name in cols_to_keep:
        cols_to_keep = [c for c in cols_to_keep if c != gdf.geometry.name] + [gdf.geometry.name]

    cleaned_gdf = gdf[cols_to_keep].copy()
    cleaned_gdf.crs = gdf.crs

    cleaned_gdf.to_file(shp_path)
    return shp_path