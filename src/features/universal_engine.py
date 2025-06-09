#!/usr/bin/env python3
"""
🌍 Universal Feature Engineering Engine - Day 5 Core System
src/features/universal_engine.py

Central intelligence system that creates location-independent climate features
with global context, regional adaptation, and seasonal awareness.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import warnings
from pathlib import Path

from .climate_indicators import ClimateIndicators
from .regional_adaptation import RegionalAdaptation
from .global_comparisons import GlobalComparisons
from .seasonal_adjustment import SeasonalAdjustment
from .feature_library import FeatureLibrary

logger = logging.getLogger(__name__)

class UniversalFeatureEngine:
    """
    🌍 Universal Feature Engineering Engine
    
    Creates intelligent, location-independent climate features that work anywhere on Earth.
    Provides global context, regional adaptation, and seasonal awareness for climate data.
    
    Key Capabilities:
    - Universal climate stress and comfort indices
    - Regional adaptation based on local climate norms
    - Global comparative metrics and percentiles
    - Hemisphere-aware seasonal adjustments
    - Extensible feature library with documentation
    """
    
    def __init__(self, global_data_path: Optional[Path] = None):
        """
        Initialize Universal Feature Engine.
        
        Args:
            global_data_path: Path to global climate baselines (optional)
        """
        self.climate_indicators = ClimateIndicators()
        self.regional_adaptation = RegionalAdaptation()
        self.global_comparisons = GlobalComparisons(global_data_path)
        self.seasonal_adjustment = SeasonalAdjustment()
        self.feature_library = FeatureLibrary()
        
        # Engine metrics and tracking
        self.processing_stats = {
            "locations_processed": 0,
            "features_created": 0,
            "avg_processing_time": 0.0,
            "regional_adaptations": 0,
            "global_comparisons": 0
        }
        
        logger.info("🌍 Universal Feature Engine initialized")
        logger.info(f"📚 Feature Library: {len(self.feature_library.get_all_features())} features available")
    
    def engineer_features(
        self, 
        data: pd.DataFrame, 
        location_info: Dict[str, Any],
        feature_groups: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Engineer universal features for any location on Earth.
        
        Args:
            data: Raw climate data (multi-source integrated)
            location_info: Location metadata (lat, lon, name, country, etc.)
            feature_groups: Specific feature groups to create (optional)
            
        Returns:
            Enhanced DataFrame with universal features
        """
        start_time = datetime.now()
        
        logger.info(f"🔬 Engineering universal features for {location_info.get('name', 'Unknown Location')}")
        logger.info(f"📊 Input data: {data.shape[0]} records, {data.shape[1]} features")
        
        # Validate inputs
        self._validate_inputs(data, location_info)
        
        # Create working copy
        enhanced_data = data.copy()
        initial_features = enhanced_data.shape[1]
        
        # Determine which feature groups to create
        if feature_groups is None:
            feature_groups = ["climate_indicators", "regional_adaptation", 
                            "global_comparisons", "seasonal_adjustment"]
        
        try:
            # Phase 1: Climate Indicators (Universal stress/comfort indices)
            if "climate_indicators" in feature_groups:
                logger.info("🌡️ Creating universal climate indicators...")
                try:
                    enhanced_data = self.climate_indicators.create_indicators(
                        enhanced_data, location_info
                    )
                    logger.info(f"   ✅ Added {enhanced_data.shape[1] - initial_features} climate indicators")
                except Exception as e:
                    logger.warning(f"   ⚠️ Climate indicators failed: {e}")
                    logger.info(f"   ⚙️ Continuing without climate indicator features")
            
            # Phase 2: Regional Adaptation (Local context and baselines)
            if "regional_adaptation" in feature_groups:
                logger.info("🗺️ Applying regional adaptation...")
                try:
                    enhanced_data = self.regional_adaptation.adapt_features(
                        enhanced_data, location_info
                    )
                    self.processing_stats["regional_adaptations"] += 1
                    logger.info(f"   ✅ Applied regional context for {location_info.get('climate_zone', 'unknown')} climate")
                except Exception as e:
                    logger.warning(f"   ⚠️ Regional adaptation failed: {e}")
                    logger.info(f"   ⚙️ Continuing without regional adaptation features")
            
            # Phase 3: Global Comparisons (World-scale benchmarking)
            if "global_comparisons" in feature_groups:
                logger.info("🌍 Creating global comparative metrics...")
                try:
                    enhanced_data = self.global_comparisons.create_comparisons(
                        enhanced_data, location_info
                    )
                    self.processing_stats["global_comparisons"] += 1
                    logger.info(f"   ✅ Added global percentiles and rankings")
                except Exception as e:
                    logger.warning(f"   ⚠️ Global comparisons failed: {e}")
                    logger.info(f"   ⚙️ Continuing without global comparison features")
            
            # Phase 4: Seasonal Adjustment (Hemisphere-aware processing)
            if "seasonal_adjustment" in feature_groups:
                logger.info("📅 Applying seasonal adjustments...")
                try:
                    enhanced_data = self.seasonal_adjustment.adjust_features(
                        enhanced_data, location_info
                    )
                    logger.info(f"   ✅ Applied hemisphere-aware seasonal processing")
                except Exception as e:
                    logger.warning(f"   ⚠️ Seasonal adjustment failed: {e}")
                    logger.info(f"   ⚙️ Continuing without seasonal adjustment features")
            
            # Add feature documentation metadata
            enhanced_data = self._add_feature_metadata(enhanced_data)
            
            # Update processing statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_processing_stats(enhanced_data, processing_time)
            
            final_features = enhanced_data.shape[1]
            features_added = final_features - initial_features
            
            logger.info(f"🎉 Universal feature engineering complete!")
            logger.info(f"📈 Features: {initial_features} → {final_features} (+{features_added})")
            logger.info(f"⏱️ Processing time: {processing_time:.2f} seconds")
            
            return enhanced_data
            
        except Exception as e:
            logger.error(f"❌ Universal feature engineering failed: {e}")
            raise
    
    def engineer_batch_features(
        self, 
        data_batch: Dict[str, Tuple[pd.DataFrame, Dict[str, Any]]],
        feature_groups: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Engineer features for multiple locations in batch.
        
        Args:
            data_batch: Dict of {location_key: (data, location_info)}
            feature_groups: Feature groups to create
            
        Returns:
            Dict of {location_key: enhanced_data}
        """
        logger.info(f"🔄 Batch processing {len(data_batch)} locations...")
        
        results = {}
        total_start = datetime.now()
        
        for location_key, (data, location_info) in data_batch.items():
            try:
                logger.info(f"📍 Processing {location_key}...")
                enhanced_data = self.engineer_features(data, location_info, feature_groups)
                results[location_key] = enhanced_data
                
            except Exception as e:
                logger.error(f"❌ Failed to process {location_key}: {e}")
                # Continue with other locations
                continue
        
        total_time = (datetime.now() - total_start).total_seconds()
        success_rate = len(results) / len(data_batch) * 100
        
        logger.info(f"🎯 Batch processing complete!")
        logger.info(f"✅ Success rate: {success_rate:.1f}% ({len(results)}/{len(data_batch)})")
        logger.info(f"⏱️ Total time: {total_time:.2f} seconds")
        
        return results
    
    def get_feature_explanations(self, feature_names: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Get human-readable explanations of features.
        
        Args:
            feature_names: Specific features to explain (optional)
            
        Returns:
            Dict of {feature_name: explanation}
        """
        return self.feature_library.get_feature_explanations(feature_names)
    
    def get_processing_report(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics and insights."""
        return {
            "processing_stats": self.processing_stats,
            "feature_library_size": len(self.feature_library.get_all_features()),
            "available_feature_groups": [
                "climate_indicators", "regional_adaptation", 
                "global_comparisons", "seasonal_adjustment"
            ],
            "engine_capabilities": {
                "universal_indicators": True,
                "regional_adaptation": True,
                "global_benchmarking": True,
                "seasonal_awareness": True,
                "batch_processing": True,
                "feature_documentation": True
            }
        }
    
    def _validate_inputs(self, data: pd.DataFrame, location_info: Dict[str, Any]):
        """Validate inputs for feature engineering."""
        if data.empty:
            raise ValueError("Input data cannot be empty")
        
        required_location_fields = ["latitude", "longitude"]
        missing_fields = [field for field in required_location_fields 
                         if field not in location_info]
        
        if missing_fields:
            raise ValueError(f"Missing required location fields: {missing_fields}")
        
        # Validate coordinate ranges
        lat, lon = location_info["latitude"], location_info["longitude"]
        if not (-90 <= lat <= 90):
            raise ValueError(f"Invalid latitude: {lat}")
        if not (-180 <= lon <= 180):
            raise ValueError(f"Invalid longitude: {lon}")
    
    def _add_feature_metadata(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add feature metadata to DataFrame attributes."""
        if not hasattr(data, 'attrs'):
            data.attrs = {}
        
        data.attrs.update({
            "universal_features_version": "1.0",
            "feature_engineering_date": datetime.now().isoformat(),
            "total_features": data.shape[1],
            "feature_groups": ["climate_indicators", "regional_adaptation", 
                             "global_comparisons", "seasonal_adjustment"],
            "engine": "UniversalFeatureEngine"
        })
        
        return data
    
    def _update_processing_stats(self, data: pd.DataFrame, processing_time: float):
        """Update internal processing statistics."""
        self.processing_stats["locations_processed"] += 1
        self.processing_stats["features_created"] += data.shape[1]
        
        # Update average processing time
        current_avg = self.processing_stats["avg_processing_time"]
        locations_count = self.processing_stats["locations_processed"]
        
        self.processing_stats["avg_processing_time"] = (
            (current_avg * (locations_count - 1) + processing_time) / locations_count
        )


def create_universal_features(
    data: pd.DataFrame, 
    location_info: Dict[str, Any],
    feature_groups: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Convenience function to create universal features.
    
    Args:
        data: Climate data DataFrame
        location_info: Location metadata
        feature_groups: Feature groups to create
        
    Returns:
        Enhanced DataFrame with universal features
    """
    engine = UniversalFeatureEngine()
    return engine.engineer_features(data, location_info, feature_groups)


if __name__ == "__main__":
    # Example usage and testing
    print("🌍 Universal Feature Engineering Engine - Day 5")
    print("=" * 60)
    
    # Create sample data for testing
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Sample location info
    sample_location = {
        "name": "Berlin",
        "country": "Germany", 
        "latitude": 52.52,
        "longitude": 13.41,
        "climate_zone": "temperate"
    }
    
    # Sample climate data
    dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")
    sample_data = pd.DataFrame({
        "temperature_2m": np.random.normal(10, 15, len(dates)),
        "precipitation": np.random.exponential(2, len(dates)),
        "relative_humidity": np.random.normal(65, 20, len(dates)),
        "wind_speed_2m": np.random.exponential(3, len(dates)),
        "pm2_5": np.random.exponential(15, len(dates))
    }, index=dates)
    
    # Test the engine
    engine = UniversalFeatureEngine()
    enhanced_data = engine.engineer_features(sample_data, sample_location)
    
    print(f"✅ Test successful!")
    print(f"📊 Original features: {sample_data.shape[1]}")
    print(f"📈 Enhanced features: {enhanced_data.shape[1]}")
    print(f"🎯 Features added: {enhanced_data.shape[1] - sample_data.shape[1]}")