"""
Main execution script for Linear Analysis.
Performs plan vs actual deviation analysis for excavation and dump areas.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

try:
    # Try relative imports (when used as a package)
    from .preprocessing import get_data_preprocessed
    from .spatial_operations import generate_area_overlap
    from .visualization import plot_category_chart
    from .excel_operation import  merge_and_save_dataframes, update_excel_with_tables_charts
except ImportError:
    # Fall back to absolute imports (when run as a script)
    from preprocessing import get_data_preprocessed
    from spatial_operations import generate_area_overlap
    from visualization import plot_category_chart
    from excel_operation import  merge_and_save_dataframes, update_excel_with_tables_charts


# CONFIGURATION

class Config:
    """Configuration class for Linear Analysis."""
    
    # Output Configuration
    OUTPUT_DIR = "D:/2_Analytics/6_plan_vs_actual/6_nov_output_2"
    
    # Actual data from site (excavation and dump)
    ACTUAL_EXCAVATION = "D:/2_Analytics/6_plan_vs_actual/All_Inputs_UTCL/Actual Dump & EXAVATION  area/Actual Excavated area.shp"
    ACTUAL_DUMP = "D:/2_Analytics/6_plan_vs_actual/All_Inputs_UTCL/Actual Dump & EXAVATION  area/Actual Dump area.shp"
    
    # Planned/proposed data from client
    PLANNED_EXCAVATION = "D:/2_Analytics/6_plan_vs_actual/All_Inputs_UTCL/Proposed Dump & Pit Area from client/Proposed PIT AREA from client.shp"
    PLANNED_DUMP = "D:/2_Analytics/6_plan_vs_actual/All_Inputs_UTCL/Proposed Dump & Pit Area from client/Proposed Dump from client.shp"
    
    # DWG file and ODA converter (for section lines with attributes)
    INPUT_FOLDER_DWG = "D:/2_Analytics/6_plan_vs_actual/raw_data_dwg_file/dwg_file"
    ODA_EXE = "C:/Program Files/ODA/ODAFileConverter 26.7.0/ODAFileConverter.exe"
    
    # DTMs (Digital Terrain Models)
    DTM_ITR1_PATH = "D:/2_Analytics/6_plan_vs_actual/UTCL_data/UTCL_data/DEMs/dtm_1/DEM_itr_1.tif"
    DTM_ITR2_PATH = "D:/2_Analytics/6_plan_vs_actual/UTCL_data/UTCL_data/DEMs/dtm_2/DEM_itr_2.tif"
    
    # Analysis Parameters
    ITR_1 = 2024
    ITR_2 = 2025
    DEVIATION_THRESHOLD = 2  # meters
    
    @classmethod
    def to_dict(cls) -> Dict[str, any]:
        """Convert configuration to dictionary."""
        return {
            'output_dir': cls.OUTPUT_DIR,
            'actual_excavation': cls.ACTUAL_EXCAVATION,
            'actual_dump': cls.ACTUAL_DUMP,
            'planned_excavation': cls.PLANNED_EXCAVATION,
            'planned_dump': cls.PLANNED_DUMP,
            'input_folder_dwg': cls.INPUT_FOLDER_DWG,
            'oda_exe': cls.ODA_EXE,
            'dtm_itr1_path': cls.DTM_ITR1_PATH,
            'dtm_itr2_path': cls.DTM_ITR2_PATH,
            'itr_1': cls.ITR_1,
            'itr_2': cls.ITR_2,
            'deviation_threshold': cls.DEVIATION_THRESHOLD
        }
    
    @classmethod
    def validate_paths(cls) -> bool:
        """
        Validate that required input files exist.
        
        Returns:
            True if all required files exist
            
        Raises:
            FileNotFoundError: If required files are missing
        """
        required_files = [
            ('Actual Excavation', cls.ACTUAL_EXCAVATION),
            ('Actual Dump', cls.ACTUAL_DUMP),
            ('Planned Excavation', cls.PLANNED_EXCAVATION),
            ('Planned Dump', cls.PLANNED_DUMP),
        ]
        
        missing_files = []
        for name, path in required_files:
            if not os.path.exists(path):
                missing_files.append(f"{name}: {path}")
        
        if missing_files:
            raise FileNotFoundError(
                "Required input files not found:\n" + "\n".join(missing_files)
            )
        
        return True


# LOGGING SETUP
def setup_logging(output_dir: str, log_level: int = logging.INFO) -> logging.Logger:
    """
    Configure logging for the analysis pipeline.
    
    Args:
        output_dir: Directory for log files
        log_level: Logging level (default: INFO)
        
    Returns:
        Configured logger instance
    """
    # Create logs directory
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f"linear_analysis_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger


# WORKFLOW EXECUTION
def run_linear_analysis(config: Config, logger: logging.Logger) -> str:
    """
    Execute the complete linear analysis workflow.
    
    This function orchestrates the entire analysis pipeline:
    Preprocessing input shapefiles
    Spatial overlay analysis for excavation
    Spatial overlay analysis for dump
    Data merging and visualization
    Final report generation
    
    Args:
        config: Configuration object with all paths and parameters
        logger: Logger instance for progress tracking
        
    Returns:
        Path to the final Excel report
        
    Raises:
        Exception: If any step in the pipeline fails
    """
    output_dir = config.OUTPUT_DIR
    
    try:
    
        # STEP 1: PREPROCESS INPUT DATA
      
        logger.info("="*70)
        logger.info("[STEP 1/5] PREPROCESSING INPUT DATA")
        logger.info("="*70)
        
        actual_excavation, actual_dump, planned_excavation, planned_dump, epsg_code = (
            get_data_preprocessed(
                act_exv=config.ACTUAL_EXCAVATION,
                act_dump=config.ACTUAL_DUMP,
                pln_exv=config.PLANNED_EXCAVATION,
                pln_dump=config.PLANNED_DUMP,
                output_dir=output_dir
            )
        )
        
        logger.info(f"✓ Data preprocessed successfully")
        logger.info(f"  - CRS: {epsg_code}")
        logger.info(f"  - Actual Excavation features: {len(actual_excavation)}")
        logger.info(f"  - Actual Dump features: {len(actual_dump)}")
        logger.info(f"  - Planned Excavation features: {len(planned_excavation)}")
        logger.info(f"  - Planned Dump features: {len(planned_dump)}")
        
        # STEP 2: EXCAVATION ANALYSIS
        logger.info("\n" + "="*70)
        logger.info("[STEP 2/5] EXCAVATION SPATIAL ANALYSIS")
        logger.info("="*70)
        
        results_exv, df_exv = generate_area_overlap(
            actual_file=actual_excavation,
            planned_file=planned_excavation,
            output_dir=output_dir,
            area_type="excavation"
        )
        
        logger.info(f"✓ Excavation analysis complete")
        logger.info(f"  - Total areas analyzed: {len(df_exv)}")
        logger.info(f"  - Total compliant area: {df_exv['Compliant_Area (Ha)'].sum():.2f} Ha")
        logger.info(f"  - Total deviation area: {df_exv['Deviation_Area (Ha)'].sum():.2f} Ha")
        
       
        # STEP 3: DUMP ANALYSIS
        logger.info("\n" + "="*70)
        logger.info("[STEP 3/5] DUMP SPATIAL ANALYSIS")
        logger.info("="*70)
        
        results_dump, df_dump = generate_area_overlap(
            actual_file=actual_dump,
            planned_file=planned_dump,
            output_dir=output_dir,
            area_type="dump"
        )
        
        logger.info(f" Dump analysis complete")
        logger.info(f"  - Total areas analyzed: {len(df_dump)}")
        logger.info(f"  - Total compliant area: {df_dump['Compliant_Area (Ha)'].sum():.2f} Ha")
        logger.info(f"  - Total deviation area: {df_dump['Deviation_Area (Ha)'].sum():.2f} Ha")
        
     
        # STEP 4: MERGE AND VISUALIZE
        logger.info("\n" + "="*70)
        logger.info("[STEP 4/5] MERGING RESULTS AND CREATING VISUALIZATIONS")
        logger.info("="*70)
        
        # Merge excavation and dump data
        excel_path = merge_and_save_dataframes(df_exv, df_dump, output_dir)
        logger.info(f"✓ Excel summary created: {excel_path}")
        
        # Generate excavation chart
        df_excavation_chart, img_path_excavation = plot_category_chart(
            output_dir,
            category="excavation",
            labels=None,
            colors=None,
            figsize=(4, 4)
        )
        logger.info(f"✓ Excavation chart generated: {img_path_excavation}")
        
        # Generate dump chart
        df_dump_chart, img_path_dump = plot_category_chart(
            output_dir,
            category="dump",
            labels=None,
            colors=None,
            figsize=(4, 4)
        )
        logger.info(f"✓ Dump chart generated: {img_path_dump}")
        
        # STEP 5: FINAL REPORT
        logger.info("\n" + "="*70)
        logger.info("[STEP 5/5] GENERATING FINAL REPORT")
        logger.info("="*70)
        
        final_excel = update_excel_with_tables_charts(
            input_excel=excel_path,
            df_excavation=df_excavation_chart,
            img_path_excavation=img_path_excavation,
            df_dump=df_dump_chart,
            img_path_dump=img_path_dump
        )
        
        logger.info(f" Final report generated: {final_excel}")
        
        return final_excel
        
    except Exception as e:
        logger.error(f" Analysis failed at workflow execution", exc_info=True)
        raise


# MAIN ENTRY POINT

def main() -> int:
    """
    Main entry point for the linear analysis pipeline.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    logger = None
    
    try:
        # Create output directory
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        
        # Setup logging
        logger = setup_logging(Config.OUTPUT_DIR)
        
        # Print banner
        logger.info("\n" + "="*70)
        logger.info("LINEAR ANALYSIS PIPELINE - PLAN VS ACTUAL DEVIATION")
        logger.info("="*70)
        logger.info(f"Output Directory: {Config.OUTPUT_DIR}")
        logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*70 + "\n")
        
        # Validate input files
        logger.info("Validating input files...")
        Config.validate_paths()
        logger.info("✓ All required input files found\n")
        
        # Execute analysis
        final_report = run_linear_analysis(Config, logger)
        
        # Success summary
        logger.info("\n" + "="*70)
        logger.info(" ANALYSIS COMPLETED SUCCESSFULLY ")
        logger.info("="*70)
        logger.info(f"Final Report: {final_report}")
        logger.info(f"Output Directory: {Config.OUTPUT_DIR}")
        logger.info("="*70 + "\n")
        
        return 0
        
    except FileNotFoundError as e:
        if logger:
            logger.error(f" File not found error: {e}")
        else:
            print(f" File not found error: {e}", file=sys.stderr)
        return 1
        
    except Exception as e:
        if logger:
            logger.error(f" Unexpected error occurred: {e}", exc_info=True)
        else:
            print(f" Unexpected error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())