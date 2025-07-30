"""
Core modules for Retail Intelligence Platform
Provides data processing, forecasting, insights, and alerts functionality
"""

from .data_processor import RetailDataProcessor
from .model_selector import ModelSelector
from .forecasting import ForecastingEngine
from .insights import InsightsGenerator
from .alerts import AlertsGenerator

__all__ = [
    'RetailDataProcessor',
    'ModelSelector', 
    'ForecastingEngine',
    'InsightsGenerator',
    'AlertsGenerator'
]

__version__ = "1.0.0"
 
