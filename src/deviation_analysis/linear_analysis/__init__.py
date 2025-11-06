"""
Linear Analysis Module
Performs plan vs actual analysis for excavation and dump areas.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Import main functions for easy access
from .preprocessing import get_data_preprocessed
from .spatial_operations import generate_area_overlap
from .visualization import plot_category_chart
from .excel_operation import get_merged_and_save_files, update_excel_with_tables_charts

# Define what gets imported with "from linear_analysis import *"
__all__ = [
    'get_data_preprocessed',
    'generate_area_overlap',
    'plot_category_chart',
    'get_merged_and_save_files',
    'update_excel_with_tables_charts',
]

