"""
Climate data acquisition and processing package.
"""

from .data_manager import ClimateDataManager
from .pipeline import ClimateDataPipeline
from .processors.air_quality_processor import AirQualityProcessor
from .processors.meteorological_processor import MeteorologicalProcessor
from .processors.base_processor import BaseDataProcessor, DataQualityChecker

__all__ = [
    'ClimateDataManager',
    'ClimateDataPipeline', 
    'AirQualityProcessor',
    'MeteorologicalProcessor',
    'BaseDataProcessor',
    'DataQualityChecker'
]