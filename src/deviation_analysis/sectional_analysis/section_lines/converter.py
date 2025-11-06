import os
import glob
import subprocess
from typing import Optional
from deviation_analysis.sectional_analysis.config.constants import SectionLineConstants


class DWGConverter:
    """Convert DWG files to DXF using ODA File Converter."""
    
    def __init__(self, oda_exe_path: str):
        """
        Initialize DWG to DXF converter.
        
        Args:
            oda_exe_path: Path to ODA File Converter executable
        """
        self.oda_exe_path = oda_exe_path
        self.constants = SectionLineConstants()
    
    def convert(
        self, 
        input_folder: str, 
        output_folder: str,
        acad_version: Optional[str] = None
    ) -> Optional[str]:
        """
        Convert DWG files to DXF format.
        
        Args:
            input_folder: Folder containing DWG files
            output_folder: Output folder for converted DXF files
            acad_version: Target AutoCAD version (default: ACAD2013)
            
        Returns:
            Path to first converted DXF file, or error message
        """
        os.makedirs(output_folder, exist_ok=True)
        
        version = acad_version or self.constants.ACAD_VERSION
        
        command = [
            self.oda_exe_path,
            input_folder,
            output_folder,
            version,
            "DXF",
            "0",
            "*.dwg"
        ]
        
        try:
            result = subprocess.run(
                command, 
                capture_output=True, 
                text=True,
                timeout=300 
            )
            
            # Find converted DXF files
            dxf_files = glob.glob(os.path.join(output_folder, "*.dxf"))
            
            if not dxf_files:
                print(f" No DXF files created. Process output:\n{result.stdout}")
                return self.constants.PROCESS_NOT_COMPLETED
            
            print(f" Converted DWG to DXF: {dxf_files[0]}")
            return dxf_files[0]
            
        except subprocess.TimeoutExpired:
            print(" Conversion timeout expired")
            return self.constants.PROCESS_NOT_COMPLETED
        except Exception as e:
            print(f" Conversion failed: {e}")
            return self.constants.PROCESS_NOT_COMPLETED
