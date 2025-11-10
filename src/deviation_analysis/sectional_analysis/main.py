import os
import sys
from typing import Optional
import geopandas as gpd

# Add src directory to Python path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from deviation_analysis.sectional_analysis.section_lines.pipeline import get_section_lines
from deviation_analysis.sectional_analysis.preprocessing import *
from deviation_analysis.sectional_analysis.spatial_operations import *
from deviation_analysis.sectional_analysis.visualization import *


if __name__ == "__main__":
    # Configuration
   
    OUTPUT_DIR = "D:/2_Analytics/6_plan_vs_actual/10_nov_output_1"
  
    # DWG file and ODA converter (for section lines with attributes)
    INPUT_FOLDER_DWG = "D:/2_Analytics/6_plan_vs_actual/raw_data_dwg_file/dwg_file"
    ODA_EXE = "C:/Program Files/ODA/ODAFileConverter 26.7.0/ODAFileConverter.exe"
    
    #DTMs (Digital Terrain Models)
    DTM_ITR1_PATH = "D:/2_Analytics/6_plan_vs_actual/UTCL_data/UTCL_data/DEMs/dtm_1/DEM_itr_1.tif"
    DTM_ITR2_PATH = "D:/2_Analytics/6_plan_vs_actual/UTCL_data/UTCL_data/DEMs/dtm_2/DEM_itr_2.tif"
    
    #Analysis Parameters
    ITR_1 = 2024
    ITR_2 = 2025
    DEVIATION_THRESHOLD = 2  # meters
    ELEVATION_PRIOFILE_INTERVAL = 0.1  #(10 cm)
    # Create output directory
    section_line_folder = os.path.join(OUTPUT_DIR, "section_lines")
    os.makedirs(section_line_folder, exist_ok=True)

    # Run pipeline
    cleaned_gdf = get_section_lines(
        input_folder=INPUT_FOLDER_DWG,
        section_line_folder=section_line_folder,
        oda_exe=ODA_EXE,
        crs=32644
    )
    print(section_line_folder)
    
    ## create the main sectional analysis folder 
    sectional_analysis_folder = os.path.join(OUTPUT_DIR, "sectional_analysis")
    os.makedirs(sectional_analysis_folder, exist_ok=True)


    ## loop for each analysis
    
    ## get number of sections 
    section_gdf = gpd.read_file(cleaned_gdf)
    n_sections  = len(section_gdf)

    ## get linear deviation output for sectional operations 
    planned_and_done_exc, unplanned_and_done_exc, planned_and_used_dump, unplanned_and_used_dump = get_linear_outputs(OUTPUT_DIR)

    for idx in range(n_sections):
        print(f"\n Processing section index {idx}")
        readable_section_name = f"section_{idx}"
        print(readable_section_name)
        try: 
            # create the subfolder within sectional analysis
            sub_section_folder, related_folder, section_name = create_sub_folder(section_gdf=section_gdf,
                            section_number=idx,
                            sectional_analysis_folder=sectional_analysis_folder)
            
            print(sub_section_folder)
            print(related_folder)
            print(section_name)
            
            # intersetions 
            (line_planned_and_done_excavation,
             line_unplanned_and_done_excavation,
             line_planned_and_used_dump,
             line_unplanned_and_used_dump) = intersect_section_line_with_polygons(section_gdf = section_gdf,
                                                   section_number = idx, 
                                                   output_folder_path = related_folder ,
                                                   planned_and_done_excavation_gdf = planned_and_done_exc,
                                                   unplanned_and_done_excavation_gdf = unplanned_and_done_exc,
                                                   planned_and_used_dump_gdf = planned_and_used_dump,
                                                   unplanned_and_used_dump_gdf =  unplanned_and_used_dump)


            # generate the df for each section
            df = get_section_elevation_profiles(
                                dtm_path1 = DTM_ITR1_PATH,
                                dtm_path2 = DTM_ITR2_PATH,
                                section_gdf =  section_gdf,
                                section_number = idx,
                                interval = ELEVATION_PRIOFILE_INTERVAL)

            df = get_planned_area(df=df, line_gdf=line_planned_and_done_excavation, key= "planned_and_done_excavation")
            df = get_planned_area(df=df, line_gdf=line_planned_and_used_dump, key="planned_and_done_dump")
            df = get_unplanned_area(df=df, line_gdf=line_unplanned_and_done_excavation, key = "unplanned_and_done_excavation", threshold=DEVIATION_THRESHOLD)
            df = get_unplanned_area(df=df, line_gdf=line_unplanned_and_used_dump, key="unplanned_and_used_dump",threshold=DEVIATION_THRESHOLD)

            df_path = os.path.join(sub_section_folder,"section_data.csv")
            df.to_csv(df_path, index= False)


            ## plot the data
            fig, ax, max_deviation, present =plot_elevation_profile(section_df = df, section_name =section_name)
            fig_path = os.path.join(sub_section_folder, "elevation_profile.png")
            fig.savefig(fig_path)


        except Exception as e:
            print("error", e)





