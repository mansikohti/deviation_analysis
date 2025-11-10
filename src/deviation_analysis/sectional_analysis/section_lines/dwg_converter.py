import os
import subprocess
import glob


def ODA_convertor(input_folder, output_folder, oda_exe):
    """Convert DWG files to DXF format using ODA File Converter."""
    os.makedirs(output_folder, exist_ok=True)

    cmd = [
        oda_exe,
        input_folder,
        output_folder,
        "ACAD2013",
        "DXF",
        "0",
        "*.dwg"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    dxf_files = glob.glob(os.path.join(output_folder, "*.dxf"))

    if not dxf_files:
        return "process not completed"

    return dxf_files[0]