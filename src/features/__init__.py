#!/usr/bin/env python3
"""
🌍 Universal Climate Feature Engineering Module - Day 5
src/features/__init__.py

Universal feature engineering system for climate data that works anywhere on Earth.
Provides location-independent climate indicators, regional adaptation, global comparisons,
and hemisphere-aware seasonal processing.
"""

from .universal_engine import UniversalFeatureEngine, create_universal_features
from .climate_indicators import ClimateIndicators
from .regional_adaptation import RegionalAdaptation
from .global_comparisons import GlobalComparisons
from .seasonal_adjustment import SeasonalAdjustment
from .feature_library import FeatureLibrary

__all__ = [
    'UniversalFeatureEngine',
    'create_universal_features',
    'ClimateIndicators',
    'RegionalAdaptation', 
    'GlobalComparisons',
    'SeasonalAdjustment',
    'FeatureLibrary'
]

__version__ = "1.0.0"
__author__ = "Climate Impact Predictor Team"
__description__ = "Universal climate feature engineering for global climate analysis"

# Feature engineering capabilities
CAPABILITIES = {
    "universal_indicators": [
        "climate_stress_index",
        "human_comfort_index", 
        "weather_extremity_index",
        "air_quality_health_impact",
        "climate_hazard_index"
    ],
    "regional_adaptation": [
        "regional_climate_stress_index",
        "temp_deviation_from_regional_norm",
        "local_percentile_rankings",
        "climate_zone_classification"
    ],
    "global_comparisons": [
        "global_percentile_rankings",
        "global_anomaly_detection",
        "climate_change_acceleration",
        "global_favorability_ranking"
    ],
    "seasonal_adjustment": [
        "hemisphere_aware_seasons",
        "seasonal_anomaly_detection",
        "seasonal_detrending",
        "seasonal_risk_indicators"
    ]
}

def get_capabilities():
    """Get overview of feature engineering capabilities."""
    return CAPABILITIES

def get_version_info():
    """Get version and module information."""
    return {
        "version": __version__,
        "description": __description__,
        "author": __author__,
        "total_feature_types": sum(len(features) for features in CAPABILITIES.values()),
        "feature_categories": len(CAPABILITIES)
    }