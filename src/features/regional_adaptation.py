#!/usr/bin/env python3
"""
🗺️ Regional Adaptation - Day 5 Feature Engineering
src/features/regional_adaptation.py

Adapts climate features based on regional context and local climate norms.
Provides regional baselines, adaptation coefficients, and local context
that makes climate data meaningful for each specific location.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import warnings

logger = logging.getLogger(__name__)

class RegionalAdaptation:
    """
    🗺️ Regional Climate Adaptation System
    
    Adapts climate features based on regional context:
    - Regional climate baselines and norms
    - Local percentile rankings
    - Climate zone-specific adaptations
    - Regional weighting of features
    - Local context awareness
    """
    
    def __init__(self):
        """Initialize Regional Adaptation system."""
        # Climate zone definitions and characteristics
        self.climate_zones = {
            "tropical": {
                "temp_range": (20, 35),
                "humidity_norm": 80,
                "temp_importance": 0.6,
                "humidity_importance": 0.8,
                "precipitation_importance": 0.9
            },
            "arid": {
                "temp_range": (10, 45),
                "humidity_norm": 30,
                "temp_importance": 0.9,
                "humidity_importance": 0.6,
                "precipitation_importance": 1.0
            },
            "temperate": {
                "temp_range": (-5, 30),
                "humidity_norm": 60,
                "temp_importance": 0.8,
                "humidity_importance": 0.7,
                "precipitation_importance": 0.7
            },
            "continental": {
                "temp_range": (-20, 35),
                "humidity_norm": 55,
                "temp_importance": 0.9,
                "humidity_importance": 0.6,
                "precipitation_importance": 0.8
            },
            "polar": {
                "temp_range": (-40, 10),
                "humidity_norm": 70,
                "temp_importance": 1.0,
                "humidity_importance": 0.5,
                "precipitation_importance": 0.6
            }
        }
        
        # Regional air quality norms (rough estimates)
        self.regional_air_quality_norms = {
            "industrial": {"pm2_5": 25, "pm10": 50},
            "urban": {"pm2_5": 20, "pm10": 40},
            "suburban": {"pm2_5": 12, "pm10": 25},
            "rural": {"pm2_5": 8, "pm10": 20},
            "remote": {"pm2_5": 5, "pm10": 15}
        }
        
        logger.info("🗺️ Regional Adaptation initialized with climate zone intelligence")
    
    def adapt_features(self, data: pd.DataFrame, location_info: Dict[str, Any]) -> pd.DataFrame:
        """
        Adapt features based on regional context.
        
        Args:
            data: Climate data with universal features
            location_info: Location metadata including climate zone
            
        Returns:
            DataFrame with regionally adapted features
        """
        logger.info(f"🌍 Applying regional adaptation for {location_info.get('name', 'Unknown Location')}")
        
        enhanced_data = data.copy()
        
        # Determine climate zone
        climate_zone = self._determine_climate_zone(location_info)
        logger.debug(f"   Climate zone: {climate_zone}")
        
        # Create regional baselines
        enhanced_data = self._create_regional_baselines(enhanced_data, climate_zone)
        
        # Calculate local percentiles
        enhanced_data = self._calculate_local_percentiles(enhanced_data)
        
        # Apply regional weighting
        enhanced_data = self._apply_regional_weighting(enhanced_data, climate_zone)
        
        # Create regional anomaly indicators
        enhanced_data = self._create_regional_anomalies(enhanced_data, climate_zone)
        
        # Add regional context features
        enhanced_data = self._add_regional_context(enhanced_data, location_info, climate_zone)
        
        logger.info(f"✅ Regional adaptation complete for {climate_zone} climate zone")
        return enhanced_data
    
    def _determine_climate_zone(self, location_info: Dict[str, Any]) -> str:
        """Determine climate zone based on location information."""
        # Use provided climate zone if available
        if "climate_zone" in location_info:
            zone = location_info["climate_zone"].lower()
            if zone in self.climate_zones:
                return zone
        
        # Estimate climate zone from latitude
        latitude = location_info.get("latitude", 0)
        abs_lat = abs(latitude)
        
        if abs_lat >= 66.5:  # Arctic/Antarctic Circle
            return "polar"
        elif abs_lat >= 45:  # Mid-latitudes
            return "continental"
        elif abs_lat >= 30:  # Subtropics
            return "temperate"
        elif abs_lat >= 15:  # Tropics edge
            return "arid"  # Could be arid or tropical
        else:  # Equatorial
            return "tropical"
    
    def _create_regional_baselines(self, data: pd.DataFrame, climate_zone: str) -> pd.DataFrame:
        """Create regional climate baselines."""
        logger.debug("   Creating regional baselines...")
        
        zone_config = self.climate_zones[climate_zone]
        
        # Temperature baseline (expected normal range for this zone)
        if 'temperature_2m' in data.columns:
            temp_min, temp_max = zone_config["temp_range"]
            data['regional_temp_baseline_min'] = temp_min
            data['regional_temp_baseline_max'] = temp_max
            data['regional_temp_baseline_center'] = (temp_min + temp_max) / 2
            
            # How far from regional normal
            data['temp_deviation_from_regional_norm'] = (
                data['temperature_2m'] - data['regional_temp_baseline_center']
            )
        
        # Humidity baseline
        if 'relative_humidity' in data.columns:
            humidity_norm = zone_config["humidity_norm"]
            data['regional_humidity_baseline'] = humidity_norm
            data['humidity_deviation_from_regional_norm'] = (
                data['relative_humidity'] - humidity_norm
            )
        
        return data
    
    def _calculate_local_percentiles(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate where current conditions rank locally (percentiles)."""
        logger.debug("   Calculating local percentiles...")
        
        # Calculate rolling percentiles (what percentile is today vs recent history)
        window = min(365, len(data))  # 1 year or all available data
        
        for column in ['temperature_2m', 'relative_humidity', 'precipitation', 'wind_speed_2m']:
            if column in data.columns:
                # Local percentile ranking
                local_percentile = data[column].rolling(window=window, min_periods=30).rank(pct=True) * 100
                data[f'{column}_local_percentile'] = local_percentile
                
                # Recent trend (is it getting more/less extreme locally?)
                recent_avg = data[column].rolling(window=30).mean()
                historical_avg = data[column].rolling(window=window).mean()
                data[f'{column}_recent_vs_historical'] = recent_avg - historical_avg
        
        # Air quality local percentiles
        for pollutant in ['pm2_5', 'pm10', 'ozone', 'carbon_monoxide']:
            if pollutant in data.columns:
                local_percentile = data[pollutant].rolling(window=window, min_periods=30).rank(pct=True) * 100
                data[f'{pollutant}_local_percentile'] = local_percentile
        
        return data
    
    def _apply_regional_weighting(self, data: pd.DataFrame, climate_zone: str) -> pd.DataFrame:
        """Apply regional importance weighting to features."""
        logger.debug("   Applying regional weighting...")
        
        zone_config = self.climate_zones[climate_zone]
        
        # Create regionally weighted stress indices
        weighted_components = []
        
        # Temperature stress (weighted by regional importance)
        if 'temperature_stress' in data.columns:
            temp_weight = zone_config["temp_importance"]
            weighted_temp_stress = data['temperature_stress'] * temp_weight
            data['regional_weighted_temp_stress'] = weighted_temp_stress
            weighted_components.append(weighted_temp_stress)
        
        # Humidity stress (weighted by regional importance)
        if 'humidity_stress' in data.columns:
            humidity_weight = zone_config["humidity_importance"]
            weighted_humidity_stress = data['humidity_stress'] * humidity_weight
            data['regional_weighted_humidity_stress'] = weighted_humidity_stress
            weighted_components.append(weighted_humidity_stress)
        
        # Precipitation stress (weighted by regional importance)
        if 'precipitation_stress' in data.columns:
            precip_weight = zone_config["precipitation_importance"]
            weighted_precip_stress = data['precipitation_stress'] * precip_weight
            data['regional_weighted_precip_stress'] = weighted_precip_stress
            weighted_components.append(weighted_precip_stress)
        
        # Create overall regional climate stress
        if weighted_components:
            regional_climate_stress = np.mean(weighted_components, axis=0)
            data['regional_climate_stress_index'] = regional_climate_stress
        
        return data
    
    def _create_regional_anomalies(self, data: pd.DataFrame, climate_zone: str) -> pd.DataFrame:
        """Create regional anomaly indicators."""
        logger.debug("   Creating regional anomaly indicators...")
        
        # Temperature anomalies relative to regional expectations
        if 'temperature_2m' in data.columns and 'regional_temp_baseline_center' in data.columns:
            # Is temperature unusually high/low for this climate zone?
            temp_anomaly_magnitude = abs(data['temp_deviation_from_regional_norm'])
            zone_temp_range = self.climate_zones[climate_zone]["temp_range"]
            expected_variation = (zone_temp_range[1] - zone_temp_range[0]) / 4  # Quarter of range
            
            data['regional_temp_anomaly_severity'] = temp_anomaly_magnitude / expected_variation
            data['is_regional_temp_anomaly'] = (data['regional_temp_anomaly_severity'] > 1.5).astype(int)
        
        # Precipitation anomalies (relative to local patterns)
        if 'precipitation' in data.columns:
            # Monthly precipitation anomaly
            monthly_normal = data.groupby(data.index.month)['precipitation'].transform('mean')
            data['monthly_precip_anomaly'] = data['precipitation'] - monthly_normal
            data['monthly_precip_anomaly_ratio'] = data['precipitation'] / (monthly_normal + 0.1)  # Avoid div by 0
        
        # Compound anomalies (multiple factors unusual at once)
        anomaly_indicators = []
        for col in ['is_regional_temp_anomaly']:
            if col in data.columns:
                anomaly_indicators.append(data[col])
        
        if len(anomaly_indicators) > 1:
            compound_anomaly = np.sum(anomaly_indicators, axis=0)
            data['compound_regional_anomaly'] = compound_anomaly
        
        return data
    
    def _add_regional_context(self, data: pd.DataFrame, location_info: Dict[str, Any], climate_zone: str) -> pd.DataFrame:
        """Add regional context features."""
        logger.debug("   Adding regional context...")
        
        # Add climate zone as categorical feature
        data['climate_zone'] = climate_zone
        
        # Add hemisphere information
        hemisphere = "northern" if location_info.get("latitude", 0) >= 0 else "southern"
        data['hemisphere'] = hemisphere
        
        # Add latitude-based features
        latitude = abs(location_info.get("latitude", 0))
        data['abs_latitude'] = latitude
        data['latitude_zone'] = pd.cut(
            [latitude], 
            bins=[0, 23.5, 35, 50, 66.5, 90],
            labels=['Tropical', 'Subtropical', 'Temperate', 'Subarctic', 'Arctic']
        )[0]
        
        # Add distance from equator (climate influence)
        data['distance_from_equator'] = latitude
        data['equatorial_influence'] = 100 - (latitude / 90 * 100)  # 0-100, higher = more equatorial
        
        # Add seasonal context (month-based)
        if not data.empty:
            data['month'] = data.index.month
            
            # Define seasons based on hemisphere
            if hemisphere == "northern":
                season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                            3: 'Spring', 4: 'Spring', 5: 'Spring',
                            6: 'Summer', 7: 'Summer', 8: 'Summer',
                            9: 'Fall', 10: 'Fall', 11: 'Fall'}
            else:  # Southern hemisphere
                season_map = {12: 'Summer', 1: 'Summer', 2: 'Summer',
                            3: 'Fall', 4: 'Fall', 5: 'Fall',
                            6: 'Winter', 7: 'Winter', 8: 'Winter',
                            9: 'Spring', 10: 'Spring', 11: 'Spring'}
            
            data['regional_season'] = data['month'].map(season_map)
        
        # Add regional air quality context (if we can estimate region type)
        region_type = self._estimate_region_type(location_info)
        if region_type and region_type in self.regional_air_quality_norms:
            norms = self.regional_air_quality_norms[region_type]
            data['regional_type'] = region_type
            
            for pollutant, norm_value in norms.items():
                if pollutant in data.columns:
                    data[f'{pollutant}_regional_norm'] = norm_value
                    data[f'{pollutant}_vs_regional_norm'] = data[pollutant] / norm_value
        
        return data
    
    def _estimate_region_type(self, location_info: Dict[str, Any]) -> Optional[str]:
        """Estimate region type (urban, rural, etc.) from location info."""
        # This is a simplified estimation - in practice, you'd use more sophisticated data
        country = location_info.get("country", "").lower()
        name = location_info.get("name", "").lower()
        
        # Major cities (rough heuristic)
        major_cities = ['london', 'paris', 'berlin', 'tokyo', 'new york', 'beijing', 'mumbai', 'delhi']
        if any(city in name for city in major_cities):
            return "urban"
        
        # Industrial regions (very rough heuristic)
        if any(term in name for term in ['industrial', 'port', 'refinery']):
            return "industrial"
        
        # Default to suburban for cities, rural for others
        if any(term in name for term in ['city', 'town']):
            return "suburban"
        
        return "rural"
    
    def get_adaptation_summary(self, data: pd.DataFrame, climate_zone: str) -> Dict[str, Any]:
        """Get summary of regional adaptations applied."""
        adaptations = {
            "climate_zone": climate_zone,
            "zone_characteristics": self.climate_zones[climate_zone],
            "regional_features_added": [],
            "baseline_features": [],
            "percentile_features": [],
            "weighted_features": [],
            "anomaly_features": []
        }
        
        # Identify which adaptation features were added
        for col in data.columns:
            if 'regional' in col.lower():
                adaptations["regional_features_added"].append(col)
            if 'baseline' in col.lower():
                adaptations["baseline_features"].append(col)
            if 'percentile' in col.lower():
                adaptations["percentile_features"].append(col)
            if 'weighted' in col.lower():
                adaptations["weighted_features"].append(col)
            if 'anomaly' in col.lower():
                adaptations["anomaly_features"].append(col)
        
        return adaptations


if __name__ == "__main__":
    # Test the regional adaptation
    print("🗺️ Testing Regional Adaptation")
    print("=" * 40)
    
    # Create sample data with some universal features
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    sample_data = pd.DataFrame({
        "temperature_2m": np.random.normal(20, 10, len(dates)),
        "relative_humidity": np.random.normal(60, 20, len(dates)),
        "precipitation": np.random.exponential(2, len(dates)),
        "temperature_stress": np.random.uniform(0, 100, len(dates)),
        "humidity_stress": np.random.uniform(0, 100, len(dates))
    }, index=dates)
    
    # Test locations with different climate zones
    test_locations = [
        {"name": "Berlin", "latitude": 52.52, "longitude": 13.41, "climate_zone": "temperate"},
        {"name": "Dubai", "latitude": 25.27, "longitude": 55.31, "climate_zone": "arid"},
        {"name": "Singapore", "latitude": 1.35, "longitude": 103.82, "climate_zone": "tropical"}
    ]
    
    adapter = RegionalAdaptation()
    
    for location in test_locations:
        print(f"\n📍 Testing {location['name']} ({location['climate_zone']})")
        adapted_data = adapter.adapt_features(sample_data, location)
        print(f"   Original features: {sample_data.shape[1]}")
        print(f"   Adapted features: {adapted_data.shape[1]}")
        print(f"   Regional features added: {adapted_data.shape[1] - sample_data.shape[1]}")
        
        # Show adaptation summary
        summary = adapter.get_adaptation_summary(adapted_data, location['climate_zone'])
        print(f"   Climate zone importance - Temp: {summary['zone_characteristics']['temp_importance']}, "
              f"Humidity: {summary['zone_characteristics']['humidity_importance']}")