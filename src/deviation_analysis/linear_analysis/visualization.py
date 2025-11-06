import os
import glob
from typing import Tuple, Dict, List, Optional
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

# Constants
SQM_PER_HECTARE = 10000.0
DEFAULT_COLORS = ["#0DA21C", "#E71D0E", "#E6D00E"]
DONUT_WIDTH = 0.4
LABEL_OFFSET_RADIUS = 0.8
PIE_START_ANGLE_DEG = 90
SAVEFIG_DPI = 300
AUTOPCT_FORMAT = "%1.1f%%"

# Category labels
LABEL_PLANNED_ACTIVE = "Planned & active"
LABEL_UNPLANNED_ACTIVE = "Unplanned & active"
LABEL_PLANNED_INACTIVE = "Planned & inactive"

CATEGORY_FILES = {
    "excavation": {
        LABEL_PLANNED_ACTIVE: "planned_and_done_excavation.shp",
        LABEL_UNPLANNED_ACTIVE: "unplanned_and_done_excavation.shp",
        LABEL_PLANNED_INACTIVE: "planned_and_not_done_excavation.shp",
    },
    "dump": {
        LABEL_PLANNED_ACTIVE: "planned_and_done_dump.shp",
        LABEL_UNPLANNED_ACTIVE: "unplanned_and_done_dump.shp",
        LABEL_PLANNED_INACTIVE: "planned_and_not_done_dump.shp",
    },
}


def _find_shapefiles(
    output_dir: str,
    category: str
) -> Dict[str, str]:
    """Find shapefiles for a given category."""
    candidate_dirs = [
        os.path.join(output_dir, "shapefile_outputs"),
        os.path.join(output_dir, "shape_file_outputs"),
    ]

    expected = CATEGORY_FILES[category]
    found = {}

    for label, fname in expected.items():
        for directory in candidate_dirs:
            if not os.path.isdir(directory):
                continue
            
            exact_path = os.path.join(directory, fname)
            if os.path.exists(exact_path):
                found[label] = exact_path
                break

    if not found:
        raise FileNotFoundError(
            f"No {category} shapefiles found in {candidate_dirs}."
        )

    return found


def _compute_areas(file_paths: Dict[str, str]) -> Dict[str, float]:
    """Compute areas in hectares from shapefiles."""
    areas = {}
    
    for label, fpath in file_paths.items():
        gdf = gpd.read_file(fpath)
        if "geometry" not in gdf.columns:
            continue
        
        try:
            area_sqm = gdf.geometry.area.sum()
        except Exception:
            area_sqm = gdf.geometry.buffer(0).area.sum()
        
        areas[label] = area_sqm / SQM_PER_HECTARE if area_sqm else 0.0

    if not areas or sum(areas.values()) <= 0:
        raise ValueError("Total area is zero; cannot create chart.")

    return areas


def _create_donut_chart(
    labels: List[str],
    sizes: List[float],
    colors: List[str],
    title: str,
    figsize: Tuple[int, int]
) -> plt.Figure:
    """Create a donut chart."""
    fig, ax = plt.subplots(figsize=figsize)
    
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct=AUTOPCT_FORMAT,
        startangle=PIE_START_ANGLE_DEG,
        colors=colors,
        wedgeprops=dict(width=DONUT_WIDTH),
    )

    # Position percentage text inside donut
    for i, autotext in enumerate(autotexts):
        angle = (wedges[i].theta2 + wedges[i].theta1) / 2.0
        x = np.cos(np.deg2rad(angle))
        y = np.sin(np.deg2rad(angle))
        autotext.set_position((LABEL_OFFSET_RADIUS * x, LABEL_OFFSET_RADIUS * y))
        autotext.set_color("white")
        autotext.set_weight("bold")

    ax.set_aspect("equal")
    plt.title(title)

    return fig


def plot_category_chart(
    output_dir: str,
    category: str,
    labels: Optional[Dict[str, str]] = None,
    colors: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (4, 4)
) -> Tuple[pd.DataFrame, str]:
    """
    Create a donut chart for excavation or dump analysis.
    
    Args:
        output_dir: Directory containing shapefiles
        category: 'excavation' or 'dump'
        labels: Optional label overrides
        colors: Optional color list
        figsize: Figure size in inches
        
    Returns:
        Tuple of (summary_dataframe, image_path)
        
    Raises:
        ValueError: If category is invalid or data is empty
        FileNotFoundError: If shapefiles not found
    """
    if category not in CATEGORY_FILES:
        raise ValueError("Invalid category. Use 'excavation' or 'dump'.")

    # Find and process shapefiles
    file_paths = _find_shapefiles(output_dir, category)
    areas = _compute_areas(file_paths)

    # Prepare data for plotting
    default_order = [LABEL_PLANNED_ACTIVE, LABEL_UNPLANNED_ACTIVE, LABEL_PLANNED_INACTIVE]
    labels_used = [lbl for lbl in default_order if lbl in areas]
    sizes = [areas[lbl] for lbl in labels_used]

    # Apply label overrides
    display_labels = (
        [labels.get(lbl, lbl) for lbl in labels_used]
        if isinstance(labels, dict)
        else labels_used
    )

    # Prepare colors
    chart_colors = (colors or DEFAULT_COLORS)[:len(sizes)]

    # Create chart
    title = f"{category.capitalize()} Distribution"
    fig = _create_donut_chart(display_labels, sizes, chart_colors, title, figsize)

    # Save chart
    img_path = os.path.join(output_dir, f"{category}_donut_chart.png")
    plt.savefig(img_path, dpi=SAVEFIG_DPI, bbox_inches="tight")
    plt.close(fig)

    # Create summary DataFrame
    df = pd.DataFrame({"Category": display_labels, "Area_ha": sizes})

    return df, img_path