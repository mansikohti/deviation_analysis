import os
import ezdxf
from collections import defaultdict, Counter
from typing import Optional, List, Tuple
from deviation_analysis.sectional_analysis.config.constants import SectionLineConstants


class LayerExtractor:
    """Extract section line layer from DXF file."""
    
    def __init__(self):
        """Initialize layer extractor."""
        self.constants = SectionLineConstants()
    
    def extract_section_lines_layer(
        self, 
        input_dxf: str, 
        output_folder: str
    ) -> str:
        """
        Identify and extract the layer containing section lines.
        
        Args:
            input_dxf: Input DXF file path
            output_folder: Output folder for extracted layer
            
        Returns:
            Path to extracted section lines DXF, or error message
        """
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, self.constants.SECTION_LINES_DXF)
        
        try:
            doc = ezdxf.readfile(input_dxf)
        except Exception as e:
            print(f" Failed to read DXF: {e}")
            return self.constants.PROCESS_NOT_COMPLETED
        
        msp = doc.modelspace()
        
        # Group entities by layer
        layer_entities = self._group_entities_by_layer(msp)
        
        # Find candidate layers
        candidates = self._find_candidate_layers(layer_entities)
        
        if not candidates:
            print(" No layer matched the 'section lines + letters' criteria.")
            return self.constants.NO_SECTION_LINES_FOUND
        
        # Select best candidate and save
        return self._save_best_candidate(candidates, output_file)
    
    def _group_entities_by_layer(self, msp) -> dict:
        """Group entities by their layer."""
        layer_entities = defaultdict(list)
        for entity in msp:
            layer_entities[entity.dxf.layer].append(entity)
        return layer_entities
    
    def _find_candidate_layers(
        self, 
        layer_entities: dict
    ) -> List[Tuple[str, int, int, List]]:
        """
        Find layers that match section line criteria.
        
        Returns:
            List of tuples: (layer_name, text_count, total_entities, entities)
        """
        candidates = []
        
        for layer_name, entities in layer_entities.items():
            if self._is_valid_section_layer(entities):
                text_count = sum(
                    1 for e in entities 
                    if e.dxftype() in ("TEXT", "MTEXT")
                )
                total_entities = len(entities)
                candidates.append((layer_name, text_count, total_entities, entities))
        
        return candidates
    
    def _is_valid_section_layer(self, entities) -> bool:
        """
        Check if layer has the right mix of entities to be a section line layer.
        Must have both lines and text entities, and only allowed entity types.
        """
        # Count entity types
        type_counts = Counter(e.dxftype() for e in entities)
        
        # Must have both lines and text
        has_lines = any(
            type_counts[t] > 0 
            for t in ("LINE", "LWPOLYLINE")
        )
        has_text = any(
            type_counts[t] > 0 
            for t in ("TEXT", "MTEXT")
        )
        
        # Check for disallowed entity types
        allowed_types = self.constants.ALLOWED_ENTITY_TYPES
        disallowed_types = self.constants.DISALLOWED_ENTITY_TYPES
        has_disallowed = any(
            type_counts[t] > 0 
            for t in disallowed_types
        )
        only_allowed = all(
            etype in allowed_types 
            for etype in type_counts.keys()
        )
        
        return has_lines and has_text and not has_disallowed and only_allowed
        
    def _save_best_candidate(
        self,
        candidates: List[Tuple[str, int, int, List]],
        output_file: str
    ) -> str:
        """
        Select best candidate layer and save to new DXF.
        
        Args:
            candidates: List of (layer_name, text_count, total_entities, entities)
            output_file: Output DXF file path
            
        Returns:
            Path to saved DXF file
        """
        # Sort by text count and total entities
        candidates.sort(key=lambda x: (-x[1], -x[2]))
        best_layer = candidates[0]
        
        print(f" Selected layer '{best_layer[0]}' with {best_layer[1]} text entities")
        
        # Create new DXF with just the section line layer
        doc = ezdxf.new()
        msp = doc.modelspace()
        
        # Copy entities
        for entity in best_layer[3]:
            if entity.dxftype() in self.constants.ALLOWED_ENTITY_TYPES:
                try:
                    msp.add_entity(entity.copy())
                except Exception as e:
                    print(f" Warning: Failed to copy entity: {e}")
        
        # Save the file
        try:
            doc.saveas(output_file)
            return output_file
        except Exception as e:
            print(f" Failed to save DXF: {e}")
            return self.constants.PROCESS_NOT_COMPLETED