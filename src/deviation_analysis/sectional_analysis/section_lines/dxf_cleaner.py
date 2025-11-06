import os
import ezdxf
from typing import Set, Tuple, Dict, Optional
from deviation_analysis.sectional_analysis.config.constants import SectionLineConstants


class DXFCleaner:
    """Deduplicate and clean DXF entities."""
    
    def __init__(self, precision: Optional[int] = None):
        """
        Initialize DXF cleaner.
        
        Args:
            precision: Decimal precision for coordinate comparison
        """
        self.constants = SectionLineConstants()
        self.precision = precision or self.constants.DEFAULT_PRECISION
    
    def clean(
        self, 
        input_dxf: str, 
        output_folder: str
    ) -> str:
        """
        Deduplicate LINE, LWPOLYLINE, TEXT, and MTEXT entities.
        
        Args:
            input_dxf: Input DXF file path
            output_folder: Output folder for cleaned DXF
            
        Returns:
            Path to cleaned DXF file
        """
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, self.constants.SECTION_LINES_CLEAN_DXF)
        
        # Read source DXF
        doc_in = ezdxf.readfile(input_dxf)
        msp_in = doc_in.modelspace()
        
        # Extract unique geometries
        unique_lines = self._extract_unique_lines(msp_in)
        unique_texts = self._extract_unique_texts(msp_in, "TEXT")
        unique_mtexts = self._extract_unique_texts(msp_in, "MTEXT")
        
        # Write cleaned DXF
        self._write_cleaned_dxf(
            output_path, 
            unique_lines, 
            unique_texts, 
            unique_mtexts
        )
        
        print(f" Cleaned DXF saved: {output_path}")
        return output_path
    
    def _round(self, value: float) -> float:
        """Round value to precision."""
        return round(float(value), self.precision)
    
    def _point_to_tuple(self, point) -> Tuple[float, float, float]:
        """Convert point to rounded tuple (x, y, z)."""
        if hasattr(point, "x"):
            return (
                self._round(point.x),
                self._round(point.y),
                self._round(getattr(point, "z", 0.0))
            )
        
        if len(point) == 2:
            return (self._round(point[0]), self._round(point[1]), 0.0)
        
        return (
            self._round(point[0]),
            self._round(point[1]),
            self._round(point[2])
        )
    
    def _extract_unique_lines(self, msp) -> Set[Tuple]:
        """
        Extract unique line segments from LINE and LWPOLYLINE entities.
        
        Returns:
            Set of unique line segments as tuples of sorted points
        """
        line_set = set()
        
        # Process LINE entities
        for line in msp.query("LINE"):
            start = self._point_to_tuple(line.dxf.start)
            end = self._point_to_tuple(line.dxf.end)
            # Sort to ensure (A,B) == (B,A)
            key = tuple(sorted([start, end]))
            line_set.add(key)
        
        # Process LWPOLYLINE entities
        for polyline in msp.query("LWPOLYLINE"):
            points = list(polyline.get_points("xy"))
            
            if len(points) >= 2:
                # Add segments
                for i in range(len(points) - 1):
                    pt_a = (self._round(points[i][0]), self._round(points[i][1]), 0.0)
                    pt_b = (self._round(points[i + 1][0]), self._round(points[i + 1][1]), 0.0)
                    key = tuple(sorted([pt_a, pt_b]))
                    line_set.add(key)
                
                # Handle closed polylines
                if polyline.closed and len(points) > 2:
                    pt_a = (self._round(points[-1][0]), self._round(points[-1][1]), 0.0)
                    pt_b = (self._round(points[0][0]), self._round(points[0][1]), 0.0)
                    key = tuple(sorted([pt_a, pt_b]))
                    line_set.add(key)
        
        return line_set
    
    def _extract_unique_texts(self, msp, entity_type: str) -> Dict:
        """
        Extract unique text entities.
        
        Args:
            msp: Modelspace
            entity_type: "TEXT" or "MTEXT"
            
        Returns:
            Dictionary mapping unique keys to entities
        """
        text_dict = {}
        
        for entity in msp.query(entity_type):
            key = self._get_text_key(entity, entity_type)
            if key not in text_dict:
                text_dict[key] = entity
        
        return text_dict
    
    def _get_text_key(self, entity, entity_type: str) -> Tuple:
        """
        Generate unique key for text entity based on position and attributes.
        """
        insert = self._point_to_tuple(entity.dxf.insert)
        
        if entity_type == "TEXT":
            height = self._round(entity.dxf.height) if entity.dxf.hasattr("height") else 0.0
            rotation = self._round(entity.dxf.rotation) if entity.dxf.hasattr("rotation") else 0.0
            return ("TEXT", insert, height, rotation)
        else:  # MTEXT
            height = self._round(entity.dxf.char_height) if entity.dxf.hasattr("char_height") else 0.0
            width = self._round(entity.dxf.width) if entity.dxf.hasattr("width") else 0.0
            rotation = self._round(entity.dxf.rotation) if entity.dxf.hasattr("rotation") else 0.0
            return ("MTEXT", insert, height, width, rotation)
    
    def _write_cleaned_dxf(
        self,
        output_path: str,
        unique_lines: Set[Tuple],
        unique_texts: Dict,
        unique_mtexts: Dict
    ) -> None:
        """Write cleaned entities to new DXF file."""
        doc_out = ezdxf.new(setup=True)
        msp_out = doc_out.modelspace()
        
        # Add lines
        for start, end in unique_lines:
            msp_out.add_line(start, end)
        
        # Add TEXT entities
        for key, text_entity in unique_texts.items():
            _, insert, height, rotation = key
            new_text = msp_out.add_text(
                text_entity.dxf.text or "",
                dxfattribs={
                    "height": height,
                    "rotation": rotation,
                    "layer": text_entity.dxf.layer if text_entity.dxf.hasattr("layer") else "0",
                }
            )
            new_text.set_pos(insert)
        
        # Add MTEXT entities
        for key, mtext_entity in unique_mtexts.items():
            _, insert, height, width, rotation = key
            new_mtext = msp_out.add_mtext(
                mtext_entity.text or "",
                dxfattribs={
                    "char_height": height,
                    "width": width,
                    "rotation": rotation,
                    "layer": mtext_entity.dxf.layer if mtext_entity.dxf.hasattr("layer") else "0",
                }
            )
            new_mtext.set_location(insert)
        
        doc_out.saveas(output_path)