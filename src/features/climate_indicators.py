#!/usr/bin/env python3
"""
🌡️ Universal Climate Indicators - Day 5 Feature Engineering
src/features/climate_indicators.py

Creates location-independent climate stress and comfort indices that work
anywhere on Earth. These indicators provide universal metrics for climate
conditions regardless of local climate norms.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import warnings

logger = logging.getLogger(__name__)

class ClimateIndicators:
    """
    🌡️ Universal Climate Indicators Creator
    
    Creates location-independent climate stress and comfort indices:
    - Climate Stress Index (0-100): Overall environmental stress
    - Human Comfort Index (0-100): Human thermal comfort
    - Weather Extremity Index (0-100): How extreme current conditions are
    - Air Quality Health Impact (0-100): Health risk from air pollution
    - Climate Variability Index (0-100): How variable conditions are
    """
    
    def __init__(self):
        """Initialize Climate Indicators creator."""
        # Universal comfort and stress thresholds
        self.comfort_thresholds = {
            "temperature": {"optimal": (18, 24), "uncomfortable": (5, 35), "dangerous": (-10, 45)},
            "humidity": {"optimal": (40, 60), "uncomfortable": (20, 80), "dangerous": (10, 95)},
            "wind_speed": {"optimal": (0, 5), "uncomfortable": (10, 20), "dangerous": (25, 50)},
            "precipitation": {"optimal": (0, 5), "uncomfortable": (20, 50), "dangerous": (100, 300)}
        }
        
        # Air quality thresholds (WHO guidelines)
        self.air_quality_thresholds = {
            "pm2_5": {"good": 15, "moderate": 35, "unhealthy": 55, "hazardous": 150},
            "pm10": {"good": 45, "moderate": 75, "unhealthy": 155, "hazardous": 250},
            "ozone": {"good": 100, "moderate": 160, "unhealthy": 215, "hazardous": 265},
            "carbon_monoxide": {"good": 9, "moderate": 15, "unhealthy": 30, "hazardous": 50}
        }
        
        logger.info("🌡️ Climate Indicators initialized with universal thresholds")
    
    def create_indicators(self, data: pd.DataFrame, location_info: Dict[str, Any]) -> pd.DataFrame:
        """
        Create universal climate indicators for any location.
        
        Args:
            data: Climate data DataFrame
            location_info: Location metadata
            
        Returns:
            DataFrame with universal climate indicators added
        """
        logger.info("🔬 Creating universal climate indicators...")
        
        enhanced_data = data.copy()
        
        # Create each universal indicator
        enhanced_data = self._create_climate_stress_index(enhanced_data)
        enhanced_data = self._create_human_comfort_index(enhanced_data)
        enhanced_data = self._create_weather_extremity_index(enhanced_data)
        enhanced_data = self._create_air_quality_health_impact(enhanced_data)
        enhanced_data = self._create_climate_variability_index(enhanced_data)
        enhanced_data = self._create_thermal_indices(enhanced_data)
        enhanced_data = self._create_precipitation_indices(enhanced_data)
        enhanced_data = self._create_composite_indices(enhanced_data)
        
        logger.info(f"✅ Created {enhanced_data.shape[1] - data.shape[1]} universal climate indicators")
        return enhanced_data
    
    def _create_climate_stress_index(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create overall climate stress index (0-100)."""
        logger.debug("   Creating Climate Stress Index...")
        
        stress_components = []
        
        # Temperature stress
        if 'temperature_2m' in data.columns:
            temp_stress = self._calculate_temperature_stress(data['temperature_2m'])
            stress_components.append(temp_stress)
            data['temperature_stress'] = temp_stress
        
        # Humidity stress
        if 'relative_humidity' in data.columns:
            humidity_stress = self._calculate_humidity_stress(data['relative_humidity'])
            stress_components.append(humidity_stress)
            data['humidity_stress'] = humidity_stress
        
        # Wind stress
        if 'wind_speed_2m' in data.columns:
            wind_stress = self._calculate_wind_stress(data['wind_speed_2m'])
            stress_components.append(wind_stress)
            data['wind_stress'] = wind_stress
        
        # Precipitation stress
        if 'precipitation' in data.columns:
            precip_stress = self._calculate_precipitation_stress(data['precipitation'])
            stress_components.append(precip_stress)
            data['precipitation_stress'] = precip_stress
        
        # Combine into overall stress index
        if stress_components:
            # Weight by availability and importance
            weights = [0.4, 0.3, 0.2, 0.1][:len(stress_components)]
            weights = np.array(weights) / sum(weights)  # Normalize
            
            climate_stress = sum(w * component for w, component in zip(weights, stress_components))
            data['climate_stress_index'] = climate_stress
        
        return data
    
    def _create_human_comfort_index(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create human thermal comfort index (0-100, higher = more comfortable)."""
        logger.debug("   Creating Human Comfort Index...")
        
        comfort_components = []
        
        # Temperature comfort
        if 'temperature_2m' in data.columns:
            temp_comfort = self._calculate_temperature_comfort(data['temperature_2m'])
            comfort_components.append(temp_comfort)
        
        # Humidity comfort
        if 'relative_humidity' in data.columns:
            humidity_comfort = self._calculate_humidity_comfort(data['relative_humidity'])
            comfort_components.append(humidity_comfort)
        
        # Wind comfort (gentle breeze is good, strong wind is bad)
        if 'wind_speed_2m' in data.columns:
            wind_comfort = self._calculate_wind_comfort(data['wind_speed_2m'])
            comfort_components.append(wind_comfort)
        
        # Combine into overall comfort index
        if comfort_components:
            human_comfort = np.mean(comfort_components, axis=0)
            data['human_comfort_index'] = human_comfort
        
        return data
    
    def _create_weather_extremity_index(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create weather extremity index (0-100, higher = more extreme)."""
        logger.debug("   Creating Weather Extremity Index...")
        
        extremity_scores = []
        
        # Temperature extremity (based on global ranges)
        if 'temperature_2m' in data.columns:
            temp_extremity = self._calculate_extremity_score(
                data['temperature_2m'], global_min=-40, global_max=50
            )
            extremity_scores.append(temp_extremity)
        
        # Precipitation extremity
        if 'precipitation' in data.columns:
            precip_extremity = self._calculate_extremity_score(
                data['precipitation'], global_min=0, global_max=500, is_log=True
            )
            extremity_scores.append(precip_extremity)
        
        # Wind extremity
        if 'wind_speed_2m' in data.columns:
            wind_extremity = self._calculate_extremity_score(
                data['wind_speed_2m'], global_min=0, global_max=50
            )
            extremity_scores.append(wind_extremity)
        
        # Combine extremity scores
        if extremity_scores:
            weather_extremity = np.max(extremity_scores, axis=0)  # Take maximum extremity
            data['weather_extremity_index'] = weather_extremity
        
        return data
    
    def _create_air_quality_health_impact(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create air quality health impact index (0-100, higher = worse health impact)."""
        logger.debug("   Creating Air Quality Health Impact...")
        
        health_impacts = []
        
        # PM2.5 health impact (most important)
        if 'pm2_5' in data.columns:
            pm25_impact = self._calculate_air_quality_impact(
                data['pm2_5'], self.air_quality_thresholds['pm2_5']
            )
            health_impacts.append(pm25_impact * 0.4)  # 40% weight
        
        # PM10 health impact
        if 'pm10' in data.columns:
            pm10_impact = self._calculate_air_quality_impact(
                data['pm10'], self.air_quality_thresholds['pm10']
            )
            health_impacts.append(pm10_impact * 0.25)  # 25% weight
        
        # Ozone health impact
        if 'ozone' in data.columns:
            ozone_impact = self._calculate_air_quality_impact(
                data['ozone'], self.air_quality_thresholds['ozone']
            )
            health_impacts.append(ozone_impact * 0.2)  # 20% weight
        
        # Carbon monoxide health impact
        if 'carbon_monoxide' in data.columns:
            co_impact = self._calculate_air_quality_impact(
                data['carbon_monoxide'], self.air_quality_thresholds['carbon_monoxide']
            )
            health_impacts.append(co_impact * 0.15)  # 15% weight
        
        # Combine health impacts
        if health_impacts:
            total_health_impact = sum(health_impacts)
            data['air_quality_health_impact'] = np.clip(total_health_impact, 0, 100)
        
        return data
    
    def _create_climate_variability_index(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create climate variability index (0-100, higher = more variable)."""
        logger.debug("   Creating Climate Variability Index...")
        
        # Calculate rolling variability for key parameters
        window = min(30, len(data) // 4)  # 30 days or 1/4 of data
        
        variability_components = []
        
        if 'temperature_2m' in data.columns:
            temp_var = data['temperature_2m'].rolling(window=window).std()
            temp_variability = self._normalize_to_0_100(temp_var, expected_max=10)
            variability_components.append(temp_variability)
        
        if 'precipitation' in data.columns:
            precip_var = data['precipitation'].rolling(window=window).std()
            precip_variability = self._normalize_to_0_100(precip_var, expected_max=50)
            variability_components.append(precip_variability)
        
        if 'wind_speed_2m' in data.columns:
            wind_var = data['wind_speed_2m'].rolling(window=window).std()
            wind_variability = self._normalize_to_0_100(wind_var, expected_max=5)
            variability_components.append(wind_variability)
        
        if variability_components:
            climate_variability = np.mean(variability_components, axis=0)
            data['climate_variability_index'] = climate_variability
        
        return data
    
    def _create_thermal_indices(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create thermal comfort indices."""
        logger.debug("   Creating thermal indices...")
        
        # Heat Index (if both temperature and humidity available)
        if all(col in data.columns for col in ['temperature_2m', 'relative_humidity']):
            heat_index = self._calculate_heat_index(data['temperature_2m'], data['relative_humidity'])
            data['heat_index'] = heat_index
            
            # Heat stress category
            data['heat_stress_level'] = pd.cut(
                heat_index,
                bins=[-np.inf, 27, 32, 41, 54, np.inf],
                labels=['Safe', 'Caution', 'Extreme_Caution', 'Danger', 'Extreme_Danger']
            )
        
        # Wind Chill (if both temperature and wind available)
        if all(col in data.columns for col in ['temperature_2m', 'wind_speed_2m']):
            wind_chill = self._calculate_wind_chill(data['temperature_2m'], data['wind_speed_2m'])
            data['wind_chill_index'] = wind_chill
        
        return data
    
    def _create_precipitation_indices(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create precipitation-related indices."""
        logger.debug("   Creating precipitation indices...")
        
        if 'precipitation' in data.columns:
            # Drought stress index (based on recent precipitation)
            window = min(30, len(data) // 4)
            recent_precip = data['precipitation'].rolling(window=window).sum()
            drought_stress = 100 - self._normalize_to_0_100(recent_precip, expected_max=100)
            data['drought_stress_index'] = drought_stress
            
            # Flood risk index (based on intense precipitation)
            daily_max = data['precipitation'].rolling(window=3).max()  # 3-day max
            flood_risk = self._normalize_to_0_100(daily_max, expected_max=100)
            data['flood_risk_index'] = flood_risk
        
        return data
    
    def _create_composite_indices(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create composite indices combining multiple factors."""
        logger.debug("   Creating composite indices...")
        
        # Overall Climate Hazard Index
        hazard_components = []
        for col in ['climate_stress_index', 'weather_extremity_index', 'air_quality_health_impact']:
            if col in data.columns:
                hazard_components.append(data[col])
        
        if hazard_components:
            climate_hazard = np.mean(hazard_components, axis=0)
            data['climate_hazard_index'] = climate_hazard
        
        # Overall Climate Favorability Index (inverse of hazard)
        if 'climate_hazard_index' in data.columns:
            data['climate_favorability_index'] = 100 - data['climate_hazard_index']
        
        return data
    
    # Helper methods for calculations
    def _calculate_temperature_stress(self, temperature: pd.Series) -> pd.Series:
        """Calculate temperature stress (0-100)."""
        optimal_min, optimal_max = self.comfort_thresholds["temperature"]["optimal"]
        uncomfortable_min, uncomfortable_max = self.comfort_thresholds["temperature"]["uncomfortable"]
        dangerous_min, dangerous_max = self.comfort_thresholds["temperature"]["dangerous"]
        
        stress = np.zeros_like(temperature)
        
        # Below optimal
        mask = temperature < optimal_min
        stress[mask] = np.clip((optimal_min - temperature[mask]) / (optimal_min - uncomfortable_min) * 50, 0, 50)
        
        # Above optimal
        mask = temperature > optimal_max
        stress[mask] = np.clip((temperature[mask] - optimal_max) / (uncomfortable_max - optimal_max) * 50, 0, 50)
        
        # Extreme conditions
        mask = (temperature < dangerous_min) | (temperature > dangerous_max)
        stress[mask] = np.clip(stress[mask] + 50, 0, 100)
        
        return pd.Series(stress, index=temperature.index)
    
    def _calculate_temperature_comfort(self, temperature: pd.Series) -> pd.Series:
        """Calculate temperature comfort (0-100, higher = more comfortable)."""
        optimal_min, optimal_max = self.comfort_thresholds["temperature"]["optimal"]
        
        # Distance from optimal range
        distance = np.minimum(
            np.abs(temperature - optimal_min),
            np.abs(temperature - optimal_max)
        )
        distance = np.where(
            (temperature >= optimal_min) & (temperature <= optimal_max),
            0,  # In optimal range
            np.minimum(
                np.abs(temperature - optimal_min),
                np.abs(temperature - optimal_max)
            )
        )
        
        # Convert distance to comfort (0-100)
        comfort = 100 - np.clip(distance * 3, 0, 100)
        return pd.Series(comfort, index=temperature.index)
    
    def _calculate_humidity_stress(self, humidity: pd.Series) -> pd.Series:
        """Calculate humidity stress (0-100)."""
        optimal_min, optimal_max = self.comfort_thresholds["humidity"]["optimal"]
        
        stress = np.zeros_like(humidity)
        
        # Too dry
        mask = humidity < optimal_min
        stress[mask] = (optimal_min - humidity[mask]) / optimal_min * 50
        
        # Too humid
        mask = humidity > optimal_max
        stress[mask] = (humidity[mask] - optimal_max) / (100 - optimal_max) * 50
        
        return pd.Series(np.clip(stress, 0, 100), index=humidity.index)
    
    def _calculate_humidity_comfort(self, humidity: pd.Series) -> pd.Series:
        """Calculate humidity comfort (0-100)."""
        return 100 - self._calculate_humidity_stress(humidity)
    
    def _calculate_wind_stress(self, wind_speed: pd.Series) -> pd.Series:
        """Calculate wind stress (0-100)."""
        optimal_max = self.comfort_thresholds["wind_speed"]["optimal"][1]
        uncomfortable_max = self.comfort_thresholds["wind_speed"]["uncomfortable"][1]
        
        stress = np.clip((wind_speed - optimal_max) / (uncomfortable_max - optimal_max) * 100, 0, 100)
        return pd.Series(stress, index=wind_speed.index)
    
    def _calculate_wind_comfort(self, wind_speed: pd.Series) -> pd.Series:
        """Calculate wind comfort (0-100)."""
        optimal_min, optimal_max = self.comfort_thresholds["wind_speed"]["optimal"]
        
        # Gentle breeze is comfortable, no wind or strong wind is uncomfortable
        comfort = np.where(
            (wind_speed >= optimal_min) & (wind_speed <= optimal_max),
            100 - np.abs(wind_speed - (optimal_min + optimal_max) / 2) * 10,
            100 - np.clip(np.maximum(0, wind_speed - optimal_max) * 5, 0, 100)
        )
        
        return pd.Series(np.clip(comfort, 0, 100), index=wind_speed.index)
    
    def _calculate_precipitation_stress(self, precipitation: pd.Series) -> pd.Series:
        """Calculate precipitation stress (0-100)."""
        # Rolling 7-day precipitation for stress calculation
        rolling_precip = precipitation.rolling(window=7).sum()
        
        # Drought stress (lack of precipitation)
        drought_stress = np.clip((7 - rolling_precip) / 7 * 50, 0, 50)
        
        # Flood stress (too much precipitation)
        flood_stress = np.clip((rolling_precip - 50) / 50 * 50, 0, 50)
        
        total_stress = drought_stress + flood_stress
        return pd.Series(np.clip(total_stress, 0, 100), index=precipitation.index)
    
    def _calculate_extremity_score(self, values: pd.Series, global_min: float, 
                                 global_max: float, is_log: bool = False) -> pd.Series:
        """Calculate how extreme values are on a global scale."""
        if is_log:
            # For highly skewed data like precipitation
            log_values = np.log1p(values)
            log_min, log_max = np.log1p(global_min), np.log1p(global_max)
            normalized = (log_values - log_min) / (log_max - log_min)
        else:
            normalized = (values - global_min) / (global_max - global_min)
        
        # Distance from center (0.5) indicates extremity
        extremity = np.abs(normalized - 0.5) * 2
        return pd.Series(np.clip(extremity * 100, 0, 100), index=values.index)
    
    def _calculate_air_quality_impact(self, pollutant: pd.Series, thresholds: Dict[str, float]) -> pd.Series:
        """Calculate health impact from air quality (0-100)."""
        conditions = [
            pollutant <= thresholds['good'],
            pollutant <= thresholds['moderate'],
            pollutant <= thresholds['unhealthy'],
            pollutant <= thresholds['hazardous']
        ]
        
        choices = [10, 35, 65, 90]  # Health impact scores
        
        impact = np.select(conditions, choices, default=100)
        return pd.Series(impact, index=pollutant.index)
    
    def _calculate_heat_index(self, temp_c: pd.Series, humidity: pd.Series) -> pd.Series:
        """Calculate heat index from temperature and humidity."""
        # Convert to Fahrenheit for calculation
        temp_f = temp_c * 9/5 + 32
        
        # Simplified heat index calculation
        hi = 0.5 * (temp_f + 61.0 + ((temp_f - 68.0) * 1.2) + (humidity * 0.094))
        
        # For high temperatures, use more complex formula
        mask = hi >= 80
        if mask.any():
            T = temp_f[mask]
            R = humidity[mask]
            
            hi[mask] = (
                -42.379 + 2.04901523 * T + 10.14333127 * R
                - 0.22475541 * T * R - 6.83783e-3 * T**2
                - 5.481717e-2 * R**2 + 1.22874e-3 * T**2 * R
                + 8.5282e-4 * T * R**2 - 1.99e-6 * T**2 * R**2
            )
        
        # Convert back to Celsius
        return (hi - 32) * 5/9
    
    def _calculate_wind_chill(self, temp_c: pd.Series, wind_kmh: pd.Series) -> pd.Series:
        """Calculate wind chill index."""
        # Convert to Fahrenheit and mph for calculation
        temp_f = temp_c * 9/5 + 32
        wind_mph = wind_kmh * 0.621371
        
        # Wind chill formula (only applies below 50°F with wind > 3 mph)
        mask = (temp_f <= 50) & (wind_mph >= 3)
        
        wind_chill = temp_c.copy()
        
        if mask.any():
            T = temp_f[mask]
            V = wind_mph[mask]
            
            wc_f = 35.74 + 0.6215 * T - 35.75 * (V ** 0.16) + 0.4275 * T * (V ** 0.16)
            wind_chill[mask] = (wc_f - 32) * 5/9
        
        return wind_chill
    
    def _normalize_to_0_100(self, values: pd.Series, expected_max: float) -> pd.Series:
        """Normalize values to 0-100 scale."""
        normalized = (values / expected_max) * 100
        return np.clip(normalized, 0, 100)


if __name__ == "__main__":
    # Test the climate indicators
    print("🌡️ Testing Universal Climate Indicators")
    print("=" * 50)
    
    # Create sample data
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    sample_data = pd.DataFrame({
        "temperature_2m": np.random.normal(20, 10, len(dates)),
        "relative_humidity": np.random.normal(60, 20, len(dates)),
        "wind_speed_2m": np.random.exponential(3, len(dates)),
        "precipitation": np.random.exponential(2, len(dates)),
        "pm2_5": np.random.exponential(15, len(dates))
    }, index=dates)
    
    # Test indicators
    indicators = ClimateIndicators()
    enhanced_data = indicators.create_indicators(sample_data, {"name": "Test Location"})
    
    print(f"✅ Original features: {sample_data.shape[1]}")
    print(f"📈 Enhanced features: {enhanced_data.shape[1]}")
    print(f"🎯 New indicators: {enhanced_data.shape[1] - sample_data.shape[1]}")
    
    # Show some indicator values
    if 'climate_stress_index' in enhanced_data.columns:
        print(f"📊 Climate Stress Index range: {enhanced_data['climate_stress_index'].min():.1f} - {enhanced_data['climate_stress_index'].max():.1f}")
    
    if 'human_comfort_index' in enhanced_data.columns:
        print(f"😌 Human Comfort Index range: {enhanced_data['human_comfort_index'].min():.1f} - {enhanced_data['human_comfort_index'].max():.1f}")