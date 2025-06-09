#!/usr/bin/env python3
"""
🌍 Global Comparisons - Day 5 Feature Engineering
src/features/global_comparisons.py

Creates global comparative metrics and percentiles that show how any location
compares to the rest of the world. Provides global context and rankings
for climate conditions anywhere on Earth.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class GlobalComparisons:
    """
    🌍 Global Climate Comparison System
    
    Creates global comparative metrics:
    - Global percentile rankings for current conditions
    - Cross-regional comparisons
    - Global anomaly detection
    - World climate rankings
    - Climate change acceleration indicators
    """
    
    def __init__(self, global_data_path: Optional[Path] = None):
        """
        Initialize Global Comparisons system.
        
        Args:
            global_data_path: Path to global climate baselines (optional)
        """
        # Global climate statistics (rough estimates - in production, use real global data)
        self.global_baselines = {
            "temperature_2m": {
                "global_mean": 14.0,  # Global average temperature
                "global_std": 20.0,   # Global temperature variation
                "extreme_cold": -50,  # Record cold
                "extreme_hot": 55,    # Record hot
                "percentiles": {
                    "p01": -30, "p05": -20, "p10": -10, "p25": 0,
                    "p50": 14, "p75": 25, "p90": 35, "p95": 40, "p99": 50
                }
            },
            "precipitation": {
                "global_mean": 2.5,   # Daily average
                "global_std": 10.0,
                "extreme_high": 300,  # Record daily precipitation
                "percentiles": {
                    "p01": 0, "p05": 0, "p10": 0, "p25": 0,
                    "p50": 0, "p75": 2, "p90": 8, "p95": 15, "p99": 50
                }
            },
            "relative_humidity": {
                "global_mean": 65.0,
                "global_std": 25.0,
                "percentiles": {
                    "p01": 10, "p05": 20, "p10": 30, "p25": 45,
                    "p50": 65, "p75": 80, "p90": 90, "p95": 95, "p99": 98
                }
            },
            "wind_speed_2m": {
                "global_mean": 3.5,
                "global_std": 4.0,
                "extreme_high": 85,   # Record wind speed
                "percentiles": {
                    "p01": 0, "p05": 0.5, "p10": 1, "p25": 2,
                    "p50": 3, "p75": 5, "p90": 8, "p95": 12, "p99": 20
                }
            },
            "pm2_5": {
                "global_mean": 18.0,  # Global urban average
                "global_std": 25.0,
                "extreme_high": 500,  # Hazardous levels
                "percentiles": {
                    "p01": 2, "p05": 5, "p10": 8, "p25": 12,
                    "p50": 18, "p75": 28, "p90": 45, "p95": 65, "p99": 120
                }
            }
        }
        
        # Climate change indicators (global trends)
        self.climate_trends = {
            "temperature_warming_rate": 0.18,  # °C per decade
            "precipitation_variability_increase": 0.05,  # 5% per decade
            "extreme_event_frequency_increase": 0.10  # 10% per decade
        }
        
        # Load external global data if provided
        if global_data_path and global_data_path.exists():
            self._load_global_data(global_data_path)
        
        logger.info("🌍 Global Comparisons initialized with world climate baselines")
    
    def create_comparisons(self, data: pd.DataFrame, location_info: Dict[str, Any]) -> pd.DataFrame:
        """
        Create global comparative metrics for location data.
        
        Args:
            data: Climate data with regional features
            location_info: Location metadata
            
        Returns:
            DataFrame with global comparison features
        """
        logger.info(f"📊 Creating global comparisons for {location_info.get('name', 'Unknown Location')}")
        
        enhanced_data = data.copy()
        
        # Create global percentile rankings
        enhanced_data = self._create_global_percentiles(enhanced_data)
        
        # Create global anomaly indicators
        enhanced_data = self._create_global_anomalies(enhanced_data)
        
        # Create cross-regional comparisons
        enhanced_data = self._create_cross_regional_comparisons(enhanced_data, location_info)
        
        # Create climate change acceleration indicators
        enhanced_data = self._create_climate_change_indicators(enhanced_data, location_info)
        
        # Create global ranking features
        enhanced_data = self._create_global_rankings(enhanced_data)
        
        # Create world context features
        enhanced_data = self._create_world_context(enhanced_data, location_info)
        
        logger.info("✅ Global comparison features created")
        return enhanced_data
    
    def _create_global_percentiles(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create global percentile rankings for climate variables."""
        logger.debug("   Creating global percentiles...")
        
        for variable in ['temperature_2m', 'precipitation', 'relative_humidity', 'wind_speed_2m', 'pm2_5']:
            if variable in data.columns and variable in self.global_baselines:
                baselines = self.global_baselines[variable]
                percentiles = baselines.get("percentiles", {})
                
                if percentiles:
                    # Calculate which global percentile current values represent
                    global_percentile = self._calculate_percentile_from_thresholds(
                        data[variable], percentiles
                    )
                    data[f'{variable}_global_percentile'] = global_percentile
                    
                    # Create categorical rankings
                    data[f'{variable}_global_rank'] = pd.cut(
                        global_percentile,
                        bins=[0, 10, 25, 75, 90, 100],
                        labels=['Bottom_10%', 'Low', 'Normal', 'High', 'Top_10%']
                    )
                    
                    # Extreme value indicators
                    data[f'{variable}_is_global_extreme'] = (
                        (global_percentile <= 1) | (global_percentile >= 99)
                    ).astype(int)
        
        return data
    
    def _create_global_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create global anomaly indicators."""
        logger.debug("   Creating global anomaly indicators...")
        
        for variable in ['temperature_2m', 'precipitation', 'relative_humidity']:
            if variable in data.columns and variable in self.global_baselines:
                baselines = self.global_baselines[variable]
                global_mean = baselines["global_mean"]
                global_std = baselines["global_std"]
                
                # Standard deviations from global mean
                global_anomaly = (data[variable] - global_mean) / global_std
                data[f'{variable}_global_anomaly_sigma'] = global_anomaly
                
                # Anomaly magnitude (absolute)
                data[f'{variable}_global_anomaly_magnitude'] = abs(global_anomaly)
                
                # Anomaly categories (with error handling)
                try:
                    data[f'{variable}_global_anomaly_category'] = pd.cut(
                        abs(global_anomaly),
                        bins=[0, 0.5, 1.0, 2.0, 3.0, float('inf')],
                        labels=['Normal', 'Slight', 'Moderate', 'Strong', 'Extreme']
                    )
                except Exception as e:
                    logger.debug(f"Could not create anomaly category for {variable}: {e}")
                    data[f'{variable}_global_anomaly_category'] = 'Normal'
                
                # Direction of anomaly (with safe conversion)
                anomaly_direction = np.where(
                    global_anomaly > 0.5, 'Above_Normal',
                    np.where(global_anomaly < -0.5, 'Below_Normal', 'Normal')
                )
                data[f'{variable}_global_anomaly_direction'] = anomaly_direction
        
        return data
    
    def _create_cross_regional_comparisons(self, data: pd.DataFrame, location_info: Dict[str, Any]) -> pd.DataFrame:
        """Create comparisons with similar regions globally."""
        logger.debug("   Creating cross-regional comparisons...")
        
        latitude = abs(location_info.get("latitude", 0))
        
        # Define latitude bands for comparison
        if latitude < 15:
            comparison_zone = "equatorial"
            comparison_temp_norm = 27
        elif latitude < 30:
            comparison_zone = "tropical"
            comparison_temp_norm = 25
        elif latitude < 45:
            comparison_zone = "subtropical"
            comparison_temp_norm = 18
        elif latitude < 60:
            comparison_zone = "temperate"
            comparison_temp_norm = 12
        else:
            comparison_zone = "polar"
            comparison_temp_norm = 0
        
        data['global_comparison_zone'] = comparison_zone
        data['zone_temp_norm'] = comparison_temp_norm
        
        # Compare to similar latitude regions
        if 'temperature_2m' in data.columns:
            data['temp_vs_similar_latitudes'] = data['temperature_2m'] - comparison_temp_norm
            temp_unusual = abs(data['temp_vs_similar_latitudes']) > 10
            if hasattr(temp_unusual, 'astype'):
                data['temp_unusual_for_latitude'] = temp_unusual.astype(int)
            else:
                data['temp_unusual_for_latitude'] = int(temp_unusual) if isinstance(temp_unusual, bool) else 0
        
        # Seasonal comparison (compare to global seasonal patterns)
        if not data.empty:
            month = data.index.month
            hemisphere = "northern" if location_info.get("latitude", 0) >= 0 else "southern"
            
            # Create seasonal temperature expectations for hemisphere
            if hemisphere == "northern":
                seasonal_temp_adjustment = np.cos((month - 7) * np.pi / 6) * 10  # Peak in July
            else:
                seasonal_temp_adjustment = np.cos((month - 1) * np.pi / 6) * 10  # Peak in January
            
            data['global_seasonal_temp_expectation'] = comparison_temp_norm + seasonal_temp_adjustment
            
            if 'temperature_2m' in data.columns:
                data['temp_vs_global_seasonal_norm'] = (
                    data['temperature_2m'] - data['global_seasonal_temp_expectation']
                )
        
        return data
    
    def _create_climate_change_indicators(self, data: pd.DataFrame, location_info: Dict[str, Any]) -> pd.DataFrame:
        """Create climate change acceleration indicators."""
        logger.debug("   Creating climate change indicators...")
        
        # Temperature trend indicators
        if 'temperature_2m' in data.columns and len(data) > 30:
            # Calculate recent warming trend
            window = min(365, len(data))
            recent_temp_trend = self._calculate_trend(data['temperature_2m'], window)
            data['local_warming_trend'] = recent_temp_trend
            
            # Compare to global warming rate
            global_warming_rate = self.climate_trends["temperature_warming_rate"]
            annual_trend = recent_temp_trend * 365  # Convert daily trend to annual
            data['warming_vs_global_rate'] = annual_trend / global_warming_rate
            
            # Climate change acceleration indicator
            data['climate_change_acceleration'] = np.clip(
                data['warming_vs_global_rate'] / 2, 0, 2  # 0-2 scale
            )
        
        # Extreme event frequency
        if 'weather_extremity_index' in data.columns:
            # Count recent extreme events
            window = min(90, len(data))  # 3 months
            extreme_threshold = 80  # 80+ on extremity index
            
            recent_extremes = (data['weather_extremity_index'] > extreme_threshold).rolling(
                window=window
            ).sum()
            data['recent_extreme_event_frequency'] = recent_extremes
            
            # Compare to expected baseline
            expected_extremes_per_period = window * 0.05  # 5% of days expected to be extreme
            data['extreme_events_vs_expected'] = recent_extremes / expected_extremes_per_period
        
        # Precipitation variability increase
        if 'precipitation' in data.columns and len(data) > 60:
            # Recent precipitation variability
            window = min(90, len(data))
            recent_precip_variability = data['precipitation'].rolling(window=window).std()
            historical_precip_variability = data['precipitation'].expanding().std()
            
            data['precipitation_variability_change'] = (
                recent_precip_variability / historical_precip_variability
            )
        
        return data
    
    def _create_global_rankings(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create global ranking features."""
        logger.debug("   Creating global rankings...")
        
        # Overall climate favorability ranking (global context)
        ranking_components = []
        
        if 'climate_stress_index' in data.columns:
            # Invert stress (lower stress = higher ranking)
            stress_ranking = 100 - data['climate_stress_index']
            ranking_components.append(stress_ranking)
        
        if 'air_quality_health_impact' in data.columns:
            # Invert air quality impact (lower impact = higher ranking)
            air_ranking = 100 - data['air_quality_health_impact']
            ranking_components.append(air_ranking)
        
        if 'weather_extremity_index' in data.columns:
            # Invert extremity (less extreme = higher ranking)
            weather_ranking = 100 - data['weather_extremity_index']
            ranking_components.append(weather_ranking)
        
        if ranking_components:
            global_favorability_ranking = np.mean(ranking_components, axis=0)
            data['global_climate_favorability_ranking'] = global_favorability_ranking
            
            # Categorical global ranking (with error handling)
            try:
                data['global_climate_tier'] = pd.cut(
                    global_favorability_ranking,
                    bins=[0, 20, 40, 60, 80, 100],
                    labels=['Poor', 'Below_Average', 'Average', 'Good', 'Excellent']
                )
            except Exception as e:
                logger.debug(f"Could not create global climate tier: {e}")
                data['global_climate_tier'] = 'Average'
        
        return data
    
    def _create_world_context(self, data: pd.DataFrame, location_info: Dict[str, Any]) -> pd.DataFrame:
        """Add world context features."""
        logger.debug("   Adding world context...")
        
        # Oceanic vs continental influence
        # Rough estimation based on location name
        location_name = location_info.get("name", "").lower()
        country = location_info.get("country", "").lower()
        
        # Coastal indicators (simplified)
        coastal_indicators = ['coast', 'beach', 'port', 'bay', 'sea', 'ocean']
        is_coastal = any(indicator in location_name for indicator in coastal_indicators)
        
        # Island nations (simplified)
        island_countries = ['japan', 'uk', 'australia', 'new zealand', 'philippines', 'indonesia']
        is_island_nation = any(island in country for island in island_countries)
        
        oceanic_influence = float(is_coastal or is_island_nation)
        data['oceanic_influence'] = oceanic_influence
        data['continental_influence'] = 1.0 - oceanic_influence
        
        # Global position indicators (with safe conversions)
        latitude = location_info.get("latitude", 0)
        longitude = location_info.get("longitude", 0)
        
        data['northern_hemisphere'] = int(latitude >= 0)
        data['southern_hemisphere'] = int(latitude < 0)
        data['eastern_hemisphere'] = int(longitude >= 0)
        data['western_hemisphere'] = int(longitude < 0)
        
        # Distance from major climate influences
        data['distance_from_equator_normalized'] = abs(latitude) / 90  # 0-1
        
        tropical_influence = max(0, 1.0 - (abs(latitude) / 30))  # Strong up to 30° lat
        data['tropical_influence'] = tropical_influence
        
        polar_influence = max(0, (abs(latitude) - 60) / 30) if abs(latitude) > 60 else 0  # Strong beyond 60° lat
        data['polar_influence'] = polar_influence
        
        return data
    
    def _calculate_percentile_from_thresholds(self, values: pd.Series, percentiles: Dict[str, float]) -> pd.Series:
        """Calculate percentile rank based on predefined thresholds."""
        # Sort percentile thresholds
        sorted_percentiles = sorted([(float(k[1:]), v) for k, v in percentiles.items()])
        
        percentile_ranks = np.zeros(len(values), dtype=float)
        
        for i, value in enumerate(values):
            if pd.isna(value):
                percentile_ranks[i] = np.nan
                continue
            
            # Find which percentile bucket this value falls into
            for j, (percentile, threshold) in enumerate(sorted_percentiles):
                if value <= threshold:
                    if j == 0:
                        percentile_ranks[i] = percentile / 2  # Below lowest threshold
                    else:
                        # Interpolate between thresholds
                        prev_percentile, prev_threshold = sorted_percentiles[j-1]
                        ratio = (value - prev_threshold) / (threshold - prev_threshold + 1e-10)  # Avoid division by zero
                        percentile_ranks[i] = prev_percentile + ratio * (percentile - prev_percentile)
                    break
            else:
                # Value is above highest threshold
                percentile_ranks[i] = 99.5
        
        return pd.Series(percentile_ranks, index=values.index)
    
    def _calculate_trend(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling trend (slope) in series."""
        def trend_slope(x):
            if len(x) < 2:
                return 0
            try:
                return np.polyfit(range(len(x)), x, 1)[0]
            except (np.linalg.LinAlgError, TypeError):
                return 0
        
        return series.rolling(window=window, min_periods=2).apply(trend_slope, raw=False)
    
    def _load_global_data(self, data_path: Path):
        """Load external global climate data."""
        try:
            with open(data_path, 'r') as f:
                external_data = json.load(f)
            
            # Update baselines with external data
            if "baselines" in external_data:
                self.global_baselines.update(external_data["baselines"])
            
            if "trends" in external_data:
                self.climate_trends.update(external_data["trends"])
            
            logger.info(f"📊 Loaded external global data from {data_path}")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not load external global data: {e}")
            logger.info("📊 Using default global baselines")


if __name__ == "__main__":
    # Test the global comparisons
    print("🌍 Testing Global Comparisons")
    print("=" * 45)
    
    # Create sample data with some features
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    sample_data = pd.DataFrame({
        "temperature_2m": np.random.normal(20, 10, len(dates)),
        "precipitation": np.random.exponential(2, len(dates)),
        "relative_humidity": np.random.normal(60, 20, len(dates)),
        "wind_speed_2m": np.random.exponential(3, len(dates)),
        "pm2_5": np.random.exponential(15, len(dates)),
        "climate_stress_index": np.random.uniform(0, 100, len(dates)),
        "weather_extremity_index": np.random.uniform(0, 100, len(dates))
    }, index=dates)
    
    # Test locations
    test_locations = [
        {"name": "Berlin", "country": "Germany", "latitude": 52.52, "longitude": 13.41},
        {"name": "Singapore", "country": "Singapore", "latitude": 1.35, "longitude": 103.82},
        {"name": "Sydney", "country": "Australia", "latitude": -33.87, "longitude": 151.21}
    ]
    
    comparisons = GlobalComparisons()
    
    for location in test_locations:
        print(f"\n📍 Testing {location['name']} ({location['latitude']:.1f}°)")
        enhanced_data = comparisons.create_comparisons(sample_data, location)
        print(f"   Original features: {sample_data.shape[1]}")
        print(f"   Enhanced features: {enhanced_data.shape[1]}")
        print(f"   Global features added: {enhanced_data.shape[1] - sample_data.shape[1]}")
        
        # Show some example global comparisons
        if 'temperature_2m_global_percentile' in enhanced_data.columns:
            avg_temp_percentile = enhanced_data['temperature_2m_global_percentile'].mean()
            print(f"   Avg temperature global percentile: {avg_temp_percentile:.1f}%")
        
        if 'global_climate_favorability_ranking' in enhanced_data.columns:
            avg_ranking = enhanced_data['global_climate_favorability_ranking'].mean()
            print(f"   Climate favorability ranking: {avg_ranking:.1f}/100")