from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import pandas as pd
import geopandas as gpd
import re
import traceback

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from deviation_analysis.sectional_analysis.preprocessing import *
from deviation_analysis.sectional_analysis.visulization_2 import plot_elevation_profile
from deviation_analysis.sectional_analysis.dxf_file_generation import export_elevation_profiles_to_dxf
from deviation_analysis.sectional_analysis.elevation_profile import get_elevation_profile
# TODO: Add extract_section_intersections implementation


def _sanitize(s: Optional[str]) -> str:
    if not s:
        return "section"
    s = str(s).replace("_", " ").strip()
    s = re.sub(r'[<>:"/\\|?*\n\r\t]+', "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or "section"


def _mkdirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def _discover_linear_layers(output_dir: Path) -> Tuple[Optional[Path], Optional[Path], Optional[Path], Optional[Path]]:
    """Find (done_exv, unplanned_exv, done_dump, unplanned_dump). Tries get_linear_outputs then filename heuristics."""
    # 1) Try user's helper if present
    try:
        from __main__ import get_linear_outputs  # works when running as a single script
        de, ue, dd, ud = get_linear_outputs(str(output_dir))
        return (Path(de) if de else None,
                Path(ue) if ue else None,
                Path(dd) if dd else None,
                Path(ud) if ud else None)
    except Exception:
        pass

    # 2) Heuristics
    targets = {
        "done_exv": ["done_exv", "planned_done_excavation", "planned_done_exv"],
        "unplanned_exv": ["unplanned_exv", "unplanned_done_excavation"],
        "done_dump": ["done_dump", "planned_used_dump"],
        "unplanned_dump": ["unplanned_dump", "unplanned_used_dump"],
    }
    found = {k: None for k in targets}
    for p in output_dir.rglob("*"):
        if p.suffix.lower() not in {".shp", ".gpkg", ".geojson"}:
            continue
        name = p.stem.lower()
        for key, keys in targets.items():
            if found[key] is None and any(k in name for k in keys):
                found[key] = p
    return found["done_exv"], found["unplanned_exv"], found["done_dump"], found["unplanned_dump"]


def _safe_plot(df: pd.DataFrame, itr1: str, itr2: str, section_name: str, save_path: Path) -> Tuple[float, str]:
    try:
        out = plot_elevation_profile(
            df, itr1=itr1, itr2=itr2, section_name=section_name, save_path=str(save_path), show=False
        )
        max_dev, dev_flag = 0.0, "no"
        if isinstance(out, tuple):
            if len(out) == 4:
                _, _, max_dev, dev_flag = out
            elif len(out) == 3:
                _, _, max_dev = out
                dev_flag = "yes" if (max_dev and max_dev > 0) else "no"
            elif len(out) == 2:
                max_dev, dev_flag = out
        return float(max_dev or 0.0), dev_flag or "no"
    except Exception as e:
        print(f" plot failed: {e}")
        return 0.0, "no"


def _safe_dxf(csv_path: Path, out_path: Path, section_name: str, deviation_threshold: float) -> Tuple[Path, Optional[float]]:
    try:
        exported = export_elevation_profiles_to_dxf(
            str(csv_path),
            out_path=str(out_path),
            chain_col="chainage",
            elev1_col="z_itr1",
            elev2_col="z_itr2",
            line_name_col="line_name",
            section_name=section_name,
            deviation_threshold=deviation_threshold
        )
        if isinstance(exported, (tuple, list)) and len(exported) >= 2:
            return Path(exported[0]), exported[1]
        return out_path, None
    except Exception as e:
        print(f" DXF export failed: {e}")
        return out_path, None


# def _assert_fixed_functions():
#     missing = []
#     for name in [
#         "sample_elevations_two_dtms",
#         "intersect_line_with_polygons_return_separately",
#         "add_labels_for_planned_area",
#         "add_labels_for_unplanned_area",
#         "plot_planned_and_unplanned_areas_with_numbered_legend",
#         "export_elevation_profiles_to_dxf",
#     ]:
#         if name not in globals() or not callable(globals()[name]):
#             missing.append(name)
#     if missing:
#         raise RuntimeError(f"Missing required fixed functions: {', '.join(missing)}")



def run_section_analysis(
    output_dir: str,
    section_line_path: str,
    dtm_itr1_path: str,
    dtm_itr2_path: str,
    *,
    interval: float = 0.1,
    itr_names: Tuple[str, str] = ("ITR1", "ITR2"),
    deviation_threshold: float = 0.0,
) -> Tuple[List[Dict[str, Any]], List[Optional[float]], str]:
    """
    End-to-end runner with fixed helper functions (no function args).
    Returns: (results, y_axis_heights, summary_csv_path)
    """
    #_assert_fixed_functions()

    output_dir = Path(output_dir)
    section_line_path = Path(section_line_path)
    dtm_itr1_path = Path(dtm_itr1_path)
    dtm_itr2_path = Path(dtm_itr2_path)

    base_out = output_dir / "sectional_deviation_analysis"
    _mkdirs(base_out)

    # read sections
    section_gdf = gpd.read_file(section_line_path)
    n_sections = len(section_gdf)
    print(f" Found {n_sections} sections in {section_line_path}")

    # discover polygon layers
    done_exv, unplanned_exv, done_dump, unplanned_dump = _discover_linear_layers(output_dir)
    print(done_exv, unplanned_exv, done_dump,unplanned_dump)
    results: List[Dict[str, Any]] = []
    y_axis_heights: List[Optional[float]] = []
    summary_rows: List[Dict[str, Any]] = []

    itr1, itr2 = itr_names

    for idx in range(n_sections):
        print(f"\n🚧 --- Processing section index {idx} ---")
        row = section_gdf.iloc[idx]
        base_name = f"{row.get('start_text', '')}_{row.get('end_text', '')}"
        section_readable = _sanitize(base_name)
        section_title = base_name

        # per-section folders
        sec_root = base_out / section_readable
        inter_dir = sec_root / "intersecting_lines"
        prof_dir = sec_root / "elevation_profile"
        data_dir = sec_root / "section_data"
        dxf_dir = sec_root / "dxf_output"
        _mkdirs(sec_root, inter_dir, prof_dir, data_dir, dxf_dir)

        try:
            # 1) sample
            df = get_elevation_profile(
                dtm_path1=str(dtm_itr1_path),
                dtm_path2=str(dtm_itr2_path),
                section_gdf=section_gdf,
                section_number=idx,
                interval=interval
            )

            # 2) intersections
            intersect_res = extract_section_intersections(
                line_gdf=section_gdf.iloc[[idx]],
                output_folder_path=str(inter_dir),
                planned_and_done_excavation_path=str(done_exv) if done_exv else None,
                unplanned_and_done_excavation_path=str(unplanned_exv) if unplanned_exv else None,
                planned_and_used_dump_path=str(done_dump) if done_dump else None,
                unplanned_and_used_dump_path=str(unplanned_dump) if unplanned_dump else None
            )

            lpde = lude = lpud = luud = None
            returned_name = None
            if isinstance(intersect_res, (tuple, list)):
                first = intersect_res[0]
                if isinstance(first, (tuple, list)) and len(first) >= 4:
                    lpde, lude, lpud, luud = first[:4]
                elif len(intersect_res) >= 4:
                    lpde, lude, lpud, luud = intersect_res[:4]
                if len(intersect_res) >= 2 and isinstance(intersect_res[1], str):
                    returned_name = intersect_res[1]
            section_title = returned_name or base_name

            # 3) classification
            csv_path = data_dir / "section_data.csv"
            df = add_labels_for_area(df, lpde, "planned_and_done_excavation", None, str(csv_path))
            df = add_labels_for_area(df, lpud, "planned_and_used_dump", None, str(csv_path))
            df = add_labels_for_area(df, lude, "unplanned_and_done_excavation", deviation_threshold, str(csv_path))
            df = add_labels_for_area(df, luud, "unplanned_and_used_dump", deviation_threshold, str(csv_path))

            # 4) plot
            png_name = f"elevation_profile {_sanitize(section_title)}.png"
            png_path = prof_dir / re.sub(r'[<>:\"/\\|?*\n\r\t]+', '', png_name).strip()
            max_dev, dev_flag = _safe_plot(df, itr1, itr2, section_title, png_path)
            print(f" Plot returned max_dev={max_dev}, dev_flag={dev_flag}")

            # 5) dxf
            dxf_name = f"elevation_profile_{_sanitize(section_title)}.dxf"
            dxf_path, y_height = _safe_dxf(csv_path, dxf_dir / dxf_name, section_title, deviation_threshold)

            # 6) summary
            summary_rows.append({
                "section_name": section_readable,
                "deviation_detected": dev_flag,
                "maximum_deviation": round(float(max_dev), 3),
            })

            results.append({
                "section_index": idx,
                "section_name": section_readable,
                "csv_path": str(csv_path),
                "png_path": str(png_path),
                "dxf_path": str(dxf_path),
                "y_axis_height": y_height,
                "maximum_deviation": float(max_dev),
                "deviation_detected": dev_flag,
                "success": True,
            })
            y_axis_heights.append(y_height)
            print(f" Section '{section_readable}' done — MaxDev={max_dev}, Deviation={dev_flag}")
 
        except Exception as e:
            traceback.print_exc()
            y_axis_heights.append(None)
            summary_rows.append({
                "section_name": section_readable,
                "deviation_detected": "no",
                "maximum_deviation": 0.0
            })
            results.append({
                "section_index": idx,
                "section_name": section_readable,
                "success": False,
                "error": str(e)
            })
            print(f" Error in section {idx}: {e}")

    # write summary
    summary_csv_path = base_out / "section_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)
    print("\n ------------------------------")
    print(f"Summary CSV saved at: {summary_csv_path}")
    print(" Summary CSV content preview:")
    print(pd.DataFrame(summary_rows).head())
    print("  ------------------------------")

    return results, y_axis_heights, str(summary_csv_path)


from pathlib import Path
import pandas as pd

# --- REQUIRED inputs (change these to your actual files/dirs) ---
output_dir = r"D:\2_Analytics\6_plan_vs_actual\6_nov_output_2"                                           # folder where results will be written
section_line_path = r"D:\2_Analytics\6_plan_vs_actual\6_nov_output_2\section_lines\section_line.shp"     # sections (LineString) with start_text/end_text
#dtm_itr1_path = r"D:\mine_project\inputs\dtm_itr1.tif"                                                    # first DTM
#dtm_itr2_path = r"D:\mine_project\inputs\dtm_itr2.tif"                                                     # second DTM


## Dtms
dtm_itr1_path = "D:/2_Analytics/6_plan_vs_actual/UTCL_data/UTCL_data/DEMs/dtm_1/DEM_itr_1.tif"
itr_1 =  2024                              

dtm_itr2_path = "D:/2_Analytics/6_plan_vs_actual/UTCL_data/UTCL_data/DEMs/dtm_2/DEM_itr_2.tif"
itr_2 = 2025                              


# --- Optional knobs ---
interval = 0.1                         # sampling resolution along each section (map units)
itr_names = ("ITR1", "ITR2")           # labels shown on the plot
deviation_threshold = 2                # used by add_labels_for_area for unplanned classification

# --- Run the pipeline ---
results, y_axis_heights, summary_csv_path = run_section_analysis(
    output_dir=output_dir,
    section_line_path=section_line_path,
    dtm_itr1_path=dtm_itr1_path,
    dtm_itr2_path=dtm_itr2_path,
    interval=interval,
    itr_names=itr_names,
    deviation_threshold=deviation_threshold,
)