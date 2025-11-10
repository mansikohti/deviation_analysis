import os
from .dwg_converter import ODA_convertor
from .dxf_operations import get_layers_from_dxf, clean_section_lines_dxf, section_lines_to_shp
from .shapefile_operations import clean_shapefile_text_fields_keep_only_ends


def get_section_lines(input_folder, section_line_folder, oda_exe, crs):
    """Main function to extract section lines from DWG to shapefile."""
    related_file_folder = os.path.join(section_line_folder, "related_files")

    dxf_file = ODA_convertor(input_folder, related_file_folder, oda_exe)
    section_lines_dxf = get_layers_from_dxf(dxf_file, related_file_folder)
    clean_section_lines_path = clean_section_lines_dxf(section_lines_dxf, related_file_folder, prec=6)
    section_lines_shp = section_lines_to_shp(clean_section_lines_path, related_file_folder, 
                                             section_line_folder, output_name="section_line.shp", crs=crs)
    cleaned_gdf = clean_shapefile_text_fields_keep_only_ends(shp_path=section_lines_shp, backup=True, verbose=True)

    return cleaned_gdf




