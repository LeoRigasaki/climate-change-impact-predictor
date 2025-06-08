"""
Meteorological data processor for NASA POWER API responses.
Handles meteorological data transformation and feature engineering.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime

from .base_processor import BaseDataProcessor

logger = logging.getLogger(__name__)

class MeteorologicalProcessor(BaseDataProcessor):
    """Processor for NASA POWER meteorological data."""
    
    def __init__(self):
        super().__init__("MeteorologicalProcessor")
    
    def process(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Process raw meteorological data into standardized DataFrame."""
        self.validate_input(data)
        
        logger.info("Processing meteorological data...")
        
        # Extract parameter data
        parameters = data.get('properties', {}).get('parameter', {})
        
        if not parameters:
            raise ValueError("No parameter data found in meteorological response")
        
        # Convert to DataFrame
        df = pd.DataFrame(parameters)
        
        # Convert index to datetime with explicit format
        df.index = pd.to_datetime(df.index, format='%Y%m%d')
        df.index.name = 'datetime'
        
        # Standardize column names
        column_mapping = {
            'T2M': 'temperature_2m',
            'PRECTOTCORR': 'precipitation',
            'WS2M': 'wind_speed_2m',
            'RH2M': 'relative_humidity'
        }
        
        df.rename(columns=column_mapping, inplace=True)
        
        # Calculate derived meteorological features
        df = self._calculate_meteorological_features(df)
        
        # Add processing metadata
        df = self.add_processing_metadata(df)
        
        # Add source metadata
        if '_metadata' in data:
            df.attrs.update(data['_metadata'])
        
        logger.info(f"Meteorological data processed: {df.shape[0]} records, {df.shape[1]} features")
        return df
    
    def _calculate_meteorological_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived meteorological features."""
        
        # Temperature features
        if 'temperature_2m' in df.columns:
            # Temperature categories
            df['temp_category'] = pd.cut(
                df['temperature_2m'],
                bins=[-float('inf'), 0, 10, 20, 30, float('inf')],
                labels=['Freezing', 'Cold', 'Cool', 'Warm', 'Hot']
            )
            
            # Moving averages
            df['temp_7d_avg'] = df['temperature_2m'].rolling(window=7, min_periods=1).mean()
            df['temp_30d_avg'] = df['temperature_2m'].rolling(window=30, min_periods=1).mean()
            
            # Temperature anomalies (deviation from 30-day average)
            df['temp_anomaly'] = df['temperature_2m'] - df['temp_30d_avg']
            
            # Temperature extremes detection
            df['is_hot_day'] = (df['temperature_2m'] > 30).astype(int)
            df['is_cold_day'] = (df['temperature_2m'] < 0).astype(int)
            
            # Growing degree days (base 10°C for general agriculture)
            df['growing_degree_days'] = np.maximum(0, df['temperature_2m'] - 10)
            
            # Heating degree days (base 18°C)
            df['heating_degree_days'] = np.maximum(0, 18 - df['temperature_2m'])
            
            # Cooling degree days (base 18°C)
            df['cooling_degree_days'] = np.maximum(0, df['temperature_2m'] - 18)
        
        # Heat Index calculation (if humidity available)
        if all(col in df.columns for col in ['temperature_2m', 'relative_humidity']):
            df['heat_index'] = self._calculate_heat_index(
                df['temperature_2m'], df['relative_humidity']
            )
            
            # Heat stress categories
            df['heat_stress_category'] = pd.cut(
                df['heat_index'],
                bins=[-float('inf'), 27, 32, 39, 46, float('inf')],
                labels=['No_Risk', 'Caution', 'Extreme_Caution', 'Danger', 'Extreme_Danger']
            )
        
        # Precipitation features
        if 'precipitation' in df.columns:
            # Precipitation categories
            df['precip_category'] = pd.cut(
                df['precipitation'],
                bins=[0, 0.1, 2.5, 10, 50, float('inf')],
                labels=['None', 'Light', 'Moderate', 'Heavy', 'Extreme'],
                include_lowest=True
            )
            
            # Cumulative precipitation
            df['precip_7d_sum'] = df['precipitation'].rolling(window=7, min_periods=1).sum()
            df['precip_30d_sum'] = df['precipitation'].rolling(window=30, min_periods=1).sum()
            
            # Dry spell analysis
            df['is_dry_day'] = (df['precipitation'] < 0.1).astype(int)
            
            # Calculate consecutive dry days
            df['dry_spell_length'] = self._calculate_consecutive_days(df['is_dry_day'])
            
            # Wet spell analysis
            df['is_wet_day'] = (df['precipitation'] >= 1.0).astype(int)
            df['wet_spell_length'] = self._calculate_consecutive_days(df['is_wet_day'])
            
            # Precipitation intensity (when it rains)
            df['precip_intensity'] = df['precipitation'].where(df['precipitation'] > 0.1)
            
            # Monthly precipitation percentiles
            df['precip_percentile'] = df.groupby(df.index.month)['precipitation'].rank(pct=True)
        
        # Wind features
        if 'wind_speed_2m' in df.columns:
            # Wind categories (Beaufort scale simplified)
            df['wind_category'] = pd.cut(
                df['wind_speed_2m'],
                bins=[0, 1, 3, 7, 12, float('inf')],
                labels=['Calm', 'Light', 'Moderate', 'Strong', 'Very_Strong']
            )
            
            # Wind power potential (simplified)
            df['wind_power_potential'] = np.power(df['wind_speed_2m'], 3)
            
            # High wind day indicator
            df['is_windy_day'] = (df['wind_speed_2m'] > 7).astype(int)
        
        # Humidity features
        if 'relative_humidity' in df.columns:
            # Humidity categories
            df['humidity_category'] = pd.cut(
                df['relative_humidity'],
                bins=[0, 30, 60, 80, 100],
                labels=['Dry', 'Comfortable', 'Humid', 'Very_Humid'],
                include_lowest=True
            )
            
            # Comfort index (simplified)
            if 'temperature_2m' in df.columns:
                # Simplified comfort calculation
                df['comfort_index'] = 100 - abs(df['temperature_2m'] - 22) * 2 - abs(df['relative_humidity'] - 50)
                df['comfort_index'] = np.clip(df['comfort_index'], 0, 100)
        
        # Multi-variable weather indices
        if all(col in df.columns for col in ['temperature_2m', 'precipitation', 'wind_speed_2m']):
            # Weather severity index
            temp_stress = np.abs(df['temperature_2m'] - 20) / 20  # 20°C as comfortable baseline
            precip_stress = np.minimum(df['precipitation'] / 10, 1)  # Cap at 10mm
            wind_stress = np.minimum(df['wind_speed_2m'] / 10, 1)  # Cap at 10 m/s
            
            df['weather_severity_index'] = (temp_stress + precip_stress + wind_stress) / 3 * 100
        
        # Time-based features
        df['month'] = df.index.month
        df['day_of_year'] = df.index.dayofyear
        df['week_of_year'] = df.index.isocalendar().week
        
        # Season mapping
        df['season'] = df['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        # Seasonal anomalies
        if 'temperature_2m' in df.columns:
            seasonal_temp_avg = df.groupby('season')['temperature_2m'].transform('mean')
            df['seasonal_temp_anomaly'] = df['temperature_2m'] - seasonal_temp_avg
        
        if 'precipitation' in df.columns:
            seasonal_precip_avg = df.groupby('season')['precipitation'].transform('mean')
            df['seasonal_precip_anomaly'] = df['precipitation'] - seasonal_precip_avg
        
        # Weather pattern indicators
        if all(col in df.columns for col in ['temperature_2m', 'precipitation']):
            # Hot and dry conditions
            df['hot_dry_conditions'] = (
                (df['temperature_2m'] > df['temperature_2m'].quantile(0.75)) &
                (df['precipitation'] < df['precipitation'].quantile(0.25))
            ).astype(int)
            
            # Cold and wet conditions
            df['cold_wet_conditions'] = (
                (df['temperature_2m'] < df['temperature_2m'].quantile(0.25)) &
                (df['precipitation'] > df['precipitation'].quantile(0.75))
            ).astype(int)
        
        return df
    
    def _calculate_heat_index(self, temp_c: pd.Series, humidity: pd.Series) -> pd.Series:
        """Calculate heat index from temperature (Celsius) and relative humidity."""
        # Convert Celsius to Fahrenheit
        temp_f = temp_c * 9/5 + 32
        
        # Heat index calculation (simplified Rothfusz equation)
        hi = temp_f.copy()
        
        # Only calculate for temperatures above 80°F (26.7°C)
        mask = temp_f >= 80
        
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
    
    def _calculate_consecutive_days(self, binary_series: pd.Series) -> pd.Series:
        """Calculate consecutive days for binary conditions (1=True, 0=False)."""
        # Create groups where the condition changes
        groups = (binary_series != binary_series.shift()).cumsum()
        
        # For each group, calculate cumulative sum only when condition is True
        consecutive = binary_series.groupby(groups).cumsum()
        
        # Set to 0 when condition is False
        consecutive = consecutive * binary_series
        
        return consecutive
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all engineered features."""
        return {
            # Temperature features
            'temp_category': 'Temperature category (Freezing/Cold/Cool/Warm/Hot)',
            'temp_7d_avg': '7-day rolling average temperature',
            'temp_30d_avg': '30-day rolling average temperature',
            'temp_anomaly': 'Temperature deviation from 30-day average',
            'is_hot_day': 'Binary indicator for hot days (>30°C)',
            'is_cold_day': 'Binary indicator for cold days (<0°C)',
            'growing_degree_days': 'Growing degree days (base 10°C)',
            'heating_degree_days': 'Heating degree days (base 18°C)',
            'cooling_degree_days': 'Cooling degree days (base 18°C)',
            
            # Heat stress features
            'heat_index': 'Heat index combining temperature and humidity',
            'heat_stress_category': 'Heat stress risk category',
            
            # Precipitation features
            'precip_category': 'Precipitation intensity category',
            'precip_7d_sum': '7-day cumulative precipitation',
            'precip_30d_sum': '30-day cumulative precipitation',
            'is_dry_day': 'Binary indicator for dry days (<0.1mm)',
            'dry_spell_length': 'Consecutive dry days count',
            'is_wet_day': 'Binary indicator for wet days (>=1.0mm)',
            'wet_spell_length': 'Consecutive wet days count',
            'precip_intensity': 'Precipitation intensity on rainy days only',
            'precip_percentile': 'Monthly precipitation percentile rank',
            
            # Wind features
            'wind_category': 'Wind speed category (Beaufort scale)',
            'wind_power_potential': 'Wind power potential (speed cubed)',
            'is_windy_day': 'Binary indicator for windy days (>7 m/s)',
            
            # Humidity features
            'humidity_category': 'Relative humidity category',
            'comfort_index': 'Human comfort index (0-100)',
            
            # Weather indices
            'weather_severity_index': 'Overall weather severity index',
            
            # Temporal features
            'month': 'Month of year (1-12)',
            'day_of_year': 'Day of year (1-365)',
            'week_of_year': 'Week of year (1-52)',
            'season': 'Season (Winter/Spring/Summer/Fall)',
            'seasonal_temp_anomaly': 'Temperature anomaly relative to season',
            'seasonal_precip_anomaly': 'Precipitation anomaly relative to season',
            
            # Weather patterns
            'hot_dry_conditions': 'Hot and dry conditions indicator',
            'cold_wet_conditions': 'Cold and wet conditions indicator'
        }