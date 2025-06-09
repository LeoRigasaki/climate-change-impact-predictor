#!/usr/bin/env python3
"""
📅 Seasonal Adjustment - Day 5 Feature Engineering
src/features/seasonal_adjustment.py

Handles hemisphere-aware seasonal processing and adjustments.
Creates seasonal baselines, detrended features, and seasonal anomaly detection
that works correctly for both Northern and Southern hemispheres.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import warnings
from scipy import signal

logger = logging.getLogger(__name__)

class SeasonalAdjustment:
    """
    📅 Hemisphere-Aware Seasonal Adjustment System
    
    Creates seasonal adjustments and features:
    - Hemisphere-aware seasonal definitions
    - Dynamic seasonal baselines
    - Seasonal anomaly detection
    - Deseasonalized trends
    - Seasonal intensity indicators
    """
    
    def __init__(self):
        """Initialize Seasonal Adjustment system."""
        # Hemisphere-specific seasonal definitions
        self.northern_seasons = {
            "Winter": [12, 1, 2],
            "Spring": [3, 4, 5], 
            "Summer": [6, 7, 8],
            "Fall": [9, 10, 11]
        }
        
        self.southern_seasons = {
            "Summer": [12, 1, 2],
            "Fall": [3, 4, 5],
            "Winter": [6, 7, 8], 
            "Spring": [9, 10, 11]
        }
        
        # Temperature expectations by season (Northern Hemisphere baseline)
        self.seasonal_temp_patterns = {
            "Winter": {"min": -5, "max": 10, "peak_month": 1},
            "Spring": {"min": 5, "max": 20, "peak_month": 4},
            "Summer": {"min": 15, "max": 30, "peak_month": 7},
            "Fall": {"min": 5, "max": 20, "peak_month": 10}
        }
        
        # Precipitation patterns (more rain in temperate winters, dry summers)
        self.seasonal_precip_patterns = {
            "Winter": 1.2,  # 20% above average
            "Spring": 1.1,  # 10% above average
            "Summer": 0.8,  # 20% below average
            "Fall": 1.0     # Average
        }
        
        logger.info("📅 Seasonal Adjustment initialized with hemisphere awareness")
    
    def adjust_features(self, data: pd.DataFrame, location_info: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply seasonal adjustments to features.
        
        Args:
            data: Climate data with regional/global features
            location_info: Location metadata including latitude
            
        Returns:
            DataFrame with seasonal adjustment features
        """
        logger.info(f"📅 Applying seasonal adjustments for {location_info.get('name', 'Unknown Location')}")
        
        enhanced_data = data.copy()
        
        # Determine hemisphere and seasonal calendar
        hemisphere = self._determine_hemisphere(location_info)
        logger.debug(f"   Hemisphere: {hemisphere}")
        
        # Create seasonal calendar
        enhanced_data = self._create_seasonal_calendar(enhanced_data, hemisphere)
        
        # Calculate seasonal baselines
        enhanced_data = self._calculate_seasonal_baselines(enhanced_data, hemisphere)
        
        # Create seasonal anomaly indicators
        enhanced_data = self._create_seasonal_anomalies(enhanced_data)
        
        # Apply seasonal detrending
        enhanced_data = self._apply_seasonal_detrending(enhanced_data)
        
        # Create seasonal intensity indicators
        enhanced_data = self._create_seasonal_intensity_indicators(enhanced_data, location_info)
        
        # Add seasonal climate risk features
        enhanced_data = self._create_seasonal_risk_features(enhanced_data, hemisphere)
        
        logger.info("✅ Seasonal adjustment features created")
        return enhanced_data
    
    def _determine_hemisphere(self, location_info: Dict[str, Any]) -> str:
        """Determine hemisphere from latitude."""
        latitude = location_info.get("latitude", 0)
        return "northern" if latitude >= 0 else "southern"
    
    def _create_seasonal_calendar(self, data: pd.DataFrame, hemisphere: str) -> pd.DataFrame:
        """Create hemisphere-appropriate seasonal calendar."""
        logger.debug("   Creating seasonal calendar...")
        
        if data.empty:
            return data
        
        # Get appropriate seasonal definitions
        seasons = self.northern_seasons if hemisphere == "northern" else self.southern_seasons
        
        # Add month and season columns
        data['month'] = data.index.month
        data['day_of_year'] = data.index.dayofyear
        
        # Map months to seasons
        season_mapping = {}
        for season, months in seasons.items():
            for month in months:
                season_mapping[month] = season
        
        data['hemisphere_season'] = data['month'].map(season_mapping)
        
        # Add seasonal progression (0-1 through each season)
        seasonal_progress = []
        for month in data['month']:
            season = season_mapping[month]
            season_months = seasons[season]
            position_in_season = season_months.index(month)
            progress = position_in_season / len(season_months)
            seasonal_progress.append(progress)
        
        data['seasonal_progress'] = seasonal_progress
        
        # Adjust day of year for Southern Hemisphere (shift by 6 months)
        if hemisphere == "southern":
            data['adjusted_day_of_year'] = ((data['day_of_year'] + 182) % 365) + 1
        else:
            data['adjusted_day_of_year'] = data['day_of_year']
        
        return data
    
    def _calculate_seasonal_baselines(self, data: pd.DataFrame, hemisphere: str) -> pd.DataFrame:
        """Calculate seasonal baselines for each variable."""
        logger.debug("   Calculating seasonal baselines...")
        
        # Calculate seasonal means for each variable
        for variable in ['temperature_2m', 'precipitation', 'relative_humidity', 'wind_speed_2m']:
            if variable in data.columns and 'hemisphere_season' in data.columns:
                # Seasonal means
                seasonal_means = data.groupby('hemisphere_season')[variable].mean()
                data[f'{variable}_seasonal_baseline'] = data['hemisphere_season'].map(seasonal_means)
                
                # Deviation from seasonal baseline
                data[f'{variable}_seasonal_deviation'] = (
                    data[variable] - data[f'{variable}_seasonal_baseline']
                )
                
                # Monthly baselines (more granular)
                monthly_means = data.groupby('month')[variable].mean()
                data[f'{variable}_monthly_baseline'] = data['month'].map(monthly_means)
                data[f'{variable}_monthly_deviation'] = (
                    data[variable] - data[f'{variable}_monthly_baseline']
                )
        
        # Create seasonal temperature expectations based on hemisphere
        if 'temperature_2m' in data.columns:
            expected_temps = self._calculate_expected_seasonal_temperature(data, hemisphere)
            data['seasonal_temp_expectation'] = expected_temps
            data['temp_vs_seasonal_expectation'] = (
                data['temperature_2m'] - data['seasonal_temp_expectation']
            )
        
        return data
    
    def _create_seasonal_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create seasonal anomaly indicators."""
        logger.debug("   Creating seasonal anomaly indicators...")
        
        for variable in ['temperature_2m', 'precipitation', 'relative_humidity']:
            if f'{variable}_seasonal_deviation' in data.columns:
                seasonal_dev = data[f'{variable}_seasonal_deviation']
                
                # Calculate seasonal anomaly magnitude
                seasonal_std = data.groupby('hemisphere_season')[variable].transform('std')
                seasonal_anomaly_magnitude = abs(seasonal_dev) / (seasonal_std + 1e-6)
                data[f'{variable}_seasonal_anomaly_magnitude'] = seasonal_anomaly_magnitude
                
                # Seasonal anomaly categories (with error handling)
                try:
                    data[f'{variable}_seasonal_anomaly_category'] = pd.cut(
                        seasonal_anomaly_magnitude,
                        bins=[0, 0.5, 1.0, 2.0, 3.0, float('inf')],
                        labels=['Normal', 'Slight', 'Moderate', 'Strong', 'Extreme']
                    )
                except Exception as e:
                    logger.debug(f"Could not create seasonal anomaly category for {variable}: {e}")
                    data[f'{variable}_seasonal_anomaly_category'] = 'Normal'
                
                # Identify exceptional seasonal conditions
                exceptional_mask = seasonal_anomaly_magnitude > 2.0
                if hasattr(exceptional_mask, 'astype'):
                    data[f'{variable}_exceptional_for_season'] = exceptional_mask.astype(int)
                else:
                    data[f'{variable}_exceptional_for_season'] = int(exceptional_mask) if isinstance(exceptional_mask, bool) else 0
        
        # Combined seasonal anomaly indicator
        anomaly_columns = [col for col in data.columns if 'exceptional_for_season' in col]
        if anomaly_columns:
            data['multiple_seasonal_anomalies'] = data[anomaly_columns].sum(axis=1)
            exceptional_conditions = data['multiple_seasonal_anomalies'] >= 2
            if hasattr(exceptional_conditions, 'astype'):
                data['is_exceptional_seasonal_conditions'] = exceptional_conditions.astype(int)
            else:
                data['is_exceptional_seasonal_conditions'] = int(exceptional_conditions) if isinstance(exceptional_conditions, bool) else 0
        
        return data
    
    def _apply_seasonal_detrending(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply seasonal detrending to reveal underlying trends."""
        logger.debug("   Applying seasonal detrending...")
        
        for variable in ['temperature_2m', 'precipitation', 'relative_humidity']:
            if variable in data.columns and len(data) > 365:
                try:
                    # Simple seasonal decomposition
                    deseasonalized = self._simple_seasonal_decompose(data[variable])
                    data[f'{variable}_deseasonalized'] = deseasonalized
                    
                    # Calculate long-term trend
                    if len(data) > 30:
                        window = min(365, len(data) // 2)  # 1 year or half the data
                        long_term_trend = data[f'{variable}_deseasonalized'].rolling(
                            window=window, center=True
                        ).mean()
                        data[f'{variable}_long_term_trend'] = long_term_trend
                        
                        # Deviation from long-term trend
                        data[f'{variable}_trend_deviation'] = (
                            data[f'{variable}_deseasonalized'] - long_term_trend
                        )
                
                except Exception as e:
                    logger.debug(f"   Could not detrend {variable}: {e}")
                    continue
        
        return data
    
    def _create_seasonal_intensity_indicators(self, data: pd.DataFrame, location_info: Dict[str, Any]) -> pd.DataFrame:
        """Create indicators of seasonal intensity and transitions."""
        logger.debug("   Creating seasonal intensity indicators...")
        
        # Seasonal transition indicators
        if 'hemisphere_season' in data.columns:
            # Detect season changes
            season_changes = data['hemisphere_season'] != data['hemisphere_season'].shift(1)
            if hasattr(season_changes, 'astype'):
                data['seasonal_transition'] = season_changes.astype(int)
            else:
                data['seasonal_transition'] = int(season_changes) if isinstance(season_changes, bool) else 0
            
            # Days since season start
            season_groups = data['hemisphere_season'].ne(data['hemisphere_season'].shift()).cumsum()
            data['days_into_season'] = data.groupby(season_groups).cumcount() + 1
        
        # Temperature seasonal intensity
        if 'temperature_2m' in data.columns and 'hemisphere_season' in data.columns:
            # How "intense" is this season (how far from annual average)
            annual_temp_mean = data['temperature_2m'].mean()
            seasonal_temp_means = data.groupby('hemisphere_season')['temperature_2m'].mean()
            data['season_temperature_intensity'] = abs(
                data['hemisphere_season'].map(seasonal_temp_means) - annual_temp_mean
            )
            
            # Seasonal temperature range
            seasonal_temp_ranges = data.groupby('hemisphere_season')['temperature_2m'].apply(
                lambda x: x.max() - x.min()
            )
            data['season_temperature_range'] = data['hemisphere_season'].map(seasonal_temp_ranges)
        
        # Precipitation seasonal patterns
        if 'precipitation' in data.columns and 'hemisphere_season' in data.columns:
            # Seasonal precipitation concentration
            seasonal_precip_totals = data.groupby('hemisphere_season')['precipitation'].sum()
            annual_precip_total = data['precipitation'].sum()
            
            seasonal_precip_share = seasonal_precip_totals / annual_precip_total
            data['seasonal_precipitation_share'] = data['hemisphere_season'].map(seasonal_precip_share)
            
            # Is this a "wet" or "dry" season for this location?
            median_seasonal_share = 0.25  # Equal distribution would be 25% per season
            wet_season_mask = data['seasonal_precipitation_share'] > median_seasonal_share * 1.5
            dry_season_mask = data['seasonal_precipitation_share'] < median_seasonal_share * 0.5
            
            if hasattr(wet_season_mask, 'astype'):
                data['is_wet_season'] = wet_season_mask.astype(int)
            else:
                data['is_wet_season'] = int(wet_season_mask) if isinstance(wet_season_mask, bool) else 0
                
            if hasattr(dry_season_mask, 'astype'):
                data['is_dry_season'] = dry_season_mask.astype(int)
            else:
                data['is_dry_season'] = int(dry_season_mask) if isinstance(dry_season_mask, bool) else 0
        
        return data
    
    def _create_seasonal_risk_features(self, data: pd.DataFrame, hemisphere: str) -> pd.DataFrame:
        """Create seasonal climate risk features."""
        logger.debug("   Creating seasonal risk features...")
        
        if 'hemisphere_season' in data.columns:
            # Define risk periods for different hazards (Northern Hemisphere baseline)
            risk_seasons = {
                "heat_risk": "Summer",
                "cold_risk": "Winter", 
                "storm_risk": "Fall" if hemisphere == "northern" else "Summer",
                "drought_risk": "Summer",
                "flood_risk": "Spring" if hemisphere == "northern" else "Fall"
            }
            
            # Adjust for Southern Hemisphere
            if hemisphere == "southern":
                risk_seasons["heat_risk"] = "Summer"  # Still Summer (Dec-Feb)
                risk_seasons["cold_risk"] = "Winter"  # Still Winter (Jun-Aug)
            
            # Create risk indicators
            for risk_type, risk_season in risk_seasons.items():
                risk_mask = data['hemisphere_season'] == risk_season
                if hasattr(risk_mask, 'astype'):
                    data[f'is_{risk_type}_season'] = risk_mask.astype(int)
                else:
                    data[f'is_{risk_type}_season'] = int(risk_mask) if isinstance(risk_mask, bool) else 0
        
        # Compound seasonal risks
        risk_columns = [col for col in data.columns if col.startswith('is_') and col.endswith('_risk_season')]
        if risk_columns:
            data['seasonal_risk_count'] = data[risk_columns].sum(axis=1)
            high_risk_mask = data['seasonal_risk_count'] >= 2
            if hasattr(high_risk_mask, 'astype'):
                data['is_high_risk_season'] = high_risk_mask.astype(int)
            else:
                data['is_high_risk_season'] = int(high_risk_mask) if isinstance(high_risk_mask, bool) else 0
        
        return data
    
    def _calculate_expected_seasonal_temperature(self, data: pd.DataFrame, hemisphere: str) -> pd.Series:
        """Calculate expected temperature based on seasonal patterns."""
        if 'adjusted_day_of_year' not in data.columns:
            return pd.Series(0, index=data.index)
        
        # Create sinusoidal temperature expectation
        day_of_year = data['adjusted_day_of_year']
        
        # Base temperature calculation (sinusoidal)
        # Peak in summer (day 182 for Northern, adjusted for Southern)
        if hemisphere == "northern":
            peak_day = 182  # Summer solstice
        else:
            peak_day = 365  # Already adjusted, so this is summer solstice for Southern
        
        # Sinusoidal temperature pattern
        angle = 2 * np.pi * (day_of_year - peak_day) / 365
        seasonal_temp = 15 + 10 * np.cos(angle)  # 15°C average, ±10°C variation
        
        return seasonal_temp
    
    def _simple_seasonal_decompose(self, series: pd.Series, period: int = 365) -> pd.Series:
        """Simple seasonal decomposition to remove seasonal component."""
        if len(series) < period * 2:
            return series  # Not enough data for seasonal decomposition
        
        # Calculate seasonal component using moving averages
        # This is a simplified version - in production, use statsmodels.seasonal_decompose
        try:
            # Calculate trend using centered moving average
            trend = series.rolling(window=period, center=True).mean()
            
            # Calculate seasonal component
            detrended = series - trend
            seasonal_means = detrended.groupby(detrended.index.dayofyear).mean()
            
            # Create seasonal component for full series
            seasonal = pd.Series(index=series.index, dtype=float)
            for i, day_of_year in enumerate(series.index.dayofyear):
                if day_of_year in seasonal_means:
                    seasonal.iloc[i] = seasonal_means[day_of_year]
                else:
                    seasonal.iloc[i] = 0
            
            # Remove seasonal component
            deseasonalized = series - seasonal
            
            return deseasonalized
            
        except Exception as e:
            logger.debug(f"Seasonal decomposition failed: {e}")
            return series
    
    def get_seasonal_summary(self, data: pd.DataFrame, hemisphere: str) -> Dict[str, Any]:
        """Get summary of seasonal adjustments applied."""
        summary = {
            "hemisphere": hemisphere,
            "seasonal_features_added": [],
            "seasons_detected": [],
            "seasonal_baselines_created": [],
            "anomaly_features": [],
            "detrended_features": []
        }
        
        # Identify seasonal features
        for col in data.columns:
            if 'seasonal' in col.lower():
                summary["seasonal_features_added"].append(col)
            if 'baseline' in col.lower():
                summary["seasonal_baselines_created"].append(col)
            if 'anomaly' in col.lower() and 'seasonal' in col.lower():
                summary["anomaly_features"].append(col)
            if 'deseasonalized' in col.lower() or 'trend' in col.lower():
                summary["detrended_features"].append(col)
        
        # Get seasons present in data
        if 'hemisphere_season' in data.columns:
            summary["seasons_detected"] = data['hemisphere_season'].unique().tolist()
        
        return summary


if __name__ == "__main__":
    # Test the seasonal adjustment
    print("📅 Testing Seasonal Adjustment")
    print("=" * 40)
    
    # Create sample data spanning full year
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    
    # Create seasonal temperature pattern
    day_of_year = dates.dayofyear
    seasonal_temp = 15 + 10 * np.cos(2 * np.pi * (day_of_year - 182) / 365)
    random_noise = np.random.normal(0, 3, len(dates))
    
    sample_data = pd.DataFrame({
        "temperature_2m": seasonal_temp + random_noise,
        "precipitation": np.random.exponential(2, len(dates)),
        "relative_humidity": np.random.normal(60, 15, len(dates))
    }, index=dates)
    
    # Test locations in different hemispheres
    test_locations = [
        {"name": "Berlin", "latitude": 52.52, "longitude": 13.41},  # Northern
        {"name": "Sydney", "latitude": -33.87, "longitude": 151.21}  # Southern
    ]
    
    adjuster = SeasonalAdjustment()
    
    for location in test_locations:
        hemisphere = "northern" if location["latitude"] >= 0 else "southern"
        print(f"\n📍 Testing {location['name']} ({hemisphere} hemisphere)")
        
        adjusted_data = adjuster.adjust_features(sample_data, location)
        print(f"   Original features: {sample_data.shape[1]}")
        print(f"   Adjusted features: {adjusted_data.shape[1]}")
        print(f"   Seasonal features added: {adjusted_data.shape[1] - sample_data.shape[1]}")
        
        # Show seasonal summary
        summary = adjuster.get_seasonal_summary(adjusted_data, hemisphere)
        print(f"   Seasons detected: {summary['seasons_detected']}")
        print(f"   Baselines created: {len(summary['seasonal_baselines_created'])}")
        
        # Show seasonal temperature pattern
        if 'hemisphere_season' in adjusted_data.columns:
            seasonal_temps = adjusted_data.groupby('hemisphere_season')['temperature_2m'].mean()
            print(f"   Seasonal temperatures: {dict(seasonal_temps.round(1))}")