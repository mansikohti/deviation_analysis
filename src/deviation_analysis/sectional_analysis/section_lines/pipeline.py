import os
from typing import Optional
from deviation_analysis.sectional_analysis.section_lines.converter import DWGConverter
from deviation_analysis.sectional_analysis.section_lines.layer_extractor import LayerExtractor
from deviation_analysis.sectional_analysis.section_lines.dxf_cleaner import DXFCleaner
from deviation_analysis.sectional_analysis.config.constants import SectionLineConstants


class SectionLinePipeline:
    """
    Orchestrate the complete section line processing pipeline.
    
    Pipeline steps:
    1. Convert DWG to DXF
    2. Extract section lines layer
    3. Clean and deduplicate entities
    4. Convert to Shapefile (will add in next iteration)
    5. Clean text fields (will add in next iteration)
    """
    
    def __init__(self, oda_exe_path: str, precision: int = 6):
        """
        Initialize pipeline.
        
        Args:
            oda_exe_path: Path to ODA File Converter executable
            precision: Coordinate precision for deduplication
        """
        self.converter = DWGConverter(oda_exe_path)
        self.layer_extractor = LayerExtractor()
        self.dxf_cleaner = DXFCleaner(precision)
        self.constants = SectionLineConstants()
    
    def process(
        self,
        input_folder: str,
        output_folder: str,
        crs: Optional[str] = None
    ) -> str:
        """
        Execute complete section line processing pipeline.
        
        Args:
            input_folder: Folder containing DWG file(s)
            output_folder: Output folder for all intermediate and final files
            crs: Coordinate reference system (for shapefile conversion)
            
        Returns:
            Path to final shapefile
        """
        print("\n" + "="*60)
        print(" SECTION LINE PROCESSING PIPELINE")
        print("="*60)
        
        # Setup folders
        related_files_folder = os.path.join(
            output_folder, 
            self.constants.RELATED_FILES_FOLDER
        )
        os.makedirs(related_files_folder, exist_ok=True)
        os.makedirs(output_folder, exist_ok=True)
        
        # Step 1: Convert DWG to DXF
        print("\n Step 1: Converting DWG to DXF...")
        dxf_file = self.converter.convert(input_folder, related_files_folder)
        
        if dxf_file == self.constants.PROCESS_NOT_COMPLETED:
            raise RuntimeError("DWG to DXF conversion failed")
        
        # Step 2: Extract section lines layer
        print("\n Step 2: Extracting section lines layer...")
        section_lines_dxf = self.layer_extractor.extract_section_lines_layer(
            dxf_file, 
            related_files_folder
        )
        
        if section_lines_dxf == self.constants.NO_SECTION_LINES_FOUND:
            raise RuntimeError("No section lines layer found in DXF")
        if section_lines_dxf == self.constants.PROCESS_NOT_COMPLETED:
            raise RuntimeError("Layer extraction failed")
        
        # Step 3: Clean and deduplicate
        print("\n Step 3: Cleaning and deduplicating entities...")
        clean_dxf_path = self.dxf_cleaner.clean(
            section_lines_dxf, 
            related_files_folder
        )
        
        print("\n" + "="*60)
        print(" DXF PROCESSING COMPLETE")
        print("="*60)
        print(f" Cleaned DXF: {clean_dxf_path}")
        
        # TODO: Steps 4 & 5 will be added in next iteration
        # 4. Convert to Shapefile
        # 5. Clean text fields
        
        return clean_dxf_path



# USAGE EXAMPLE

if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    
    # Configuration
    ODA_EXE = "C:/Program Files/ODA/ODAFileConverter 26.7.0/ODAFileConverter.exe"
    INPUT_DWG_FOLDER = "D:/2_Analytics/6_plan_vs_actual/raw_data_dwg_file/dwg_file"
    OUTPUT_FOLDER = "D:/2_Analytics/6_plan_vs_actual/6_nov_output_2/sectiona"
    EPSG_CODE = "EPSG:32644"  
    
    # Initialize pipeline
    pipeline = SectionLinePipeline(
        oda_exe_path=ODA_EXE,
        precision=6
    )
    
    # Execute pipeline
    try:
        result_path = pipeline.process(
            input_folder=INPUT_DWG_FOLDER,
            output_folder=OUTPUT_FOLDER,
            crs=EPSG_CODE
        )
        print(f"\n Success! Result: {result_path}")
    except Exception as e:
        print(f"\n Pipeline failed: {e}")