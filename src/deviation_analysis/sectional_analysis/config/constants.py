from typing import Set

class SectionLineConstants:
    # Constants for section line processing 
    
    # DXF Entity Types
    ALLOWED_ENTITY_TYPES: Set[str] = {"LINE", "LWPOLYLINE", "TEXT", "MTEXT"}
    DISALLOWED_ENTITY_TYPES: Set[str] = {"HATCH", "POLYLINE"}
    
    # AutoCAD Version
    ACAD_VERSION: str = "ACAD2013"
    
    # Default Precision
    DEFAULT_PRECISION: int = 6
    
    # File Names
    SECTION_LINES_DXF: str = "section_lines.dxf"
    SECTION_LINES_CLEAN_DXF: str = "section_lines_clean.dxf"
    LINES_DXF: str = "LINES.dxf"
    TEXTS_DXF: str = "TEXTS.dxf"
    SECTION_LINE_SHP: str = "section_lines.shp"
    
    # Column Names
    START_TEXT_COL: str = "start_text"
    END_TEXT_COL: str = "end_text"
    
    # Folder Names
    RELATED_FILES_FOLDER: str = "related_files"
    
    # Messages
    PROCESS_NOT_COMPLETED: str = "process not completed"
    NO_SECTION_LINES_FOUND: str = "no section lines found"