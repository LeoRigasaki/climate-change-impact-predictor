"""
Air Quality data processor for Open-Meteo API responses.
Handles air quality data transformation and feature engineering.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime

from .base_processor import BaseDataProcessor

logger = logging.getLogger(__name__)

class AirQualityProcessor(BaseDataProcessor):
    """Processor for Open-Meteo air quality data."""
    
    def __init__(self):
        super().__init__("AirQualityProcessor")
    
    def process(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Process raw air quality data into standardized DataFrame."""
        self.validate_input(data)
        
        logger.info("Processing air quality data...")
        
        # Extract hourly data
        hourly_data = data.get('hourly', {})
        current_data = data.get('current', {})
        
        if not hourly_data:
            raise ValueError("No hourly data found in air quality response")
        
        # Create DataFrame from hourly data
        df = pd.DataFrame(hourly_data)
        
        # Convert time column to datetime
        if 'time' in df.columns:
            df['datetime'] = pd.to_datetime(df['time'])
            df.set_index('datetime', inplace=True)
            df.drop('time', axis=1, inplace=True)
        
        # Standardize column names
        df = self._standardize_column_names(df)
        
        # Add current conditions if available
        if current_data:
            df = self._add_current_conditions(df, current_data)
        
        # Calculate derived features
        df = self._calculate_air_quality_features(df)
        
        # Add metadata
        df = self.add_processing_metadata(df)
        
        # Add source metadata
        if '_metadata' in data:
            df.attrs.update(data['_metadata'])
        
        logger.info(f"Air quality data processed: {df.shape[0]} records, {df.shape[1]} features")
        return df
    
    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names for consistency."""
        column_mapping = {
            'pm2_5': 'pm2_5',
            'pm10': 'pm10',
            'carbon_monoxide': 'co',
            'carbon_dioxide': 'co2',
            'nitrogen_dioxide': 'no2',
            'sulphur_dioxide': 'so2',
            'ozone': 'o3',
            'ammonia': 'nh3',
            'dust': 'dust',
            'uv_index': 'uv_index',
            'methane': 'ch4'
        }
        
        # Rename columns that exist
        existing_mappings = {k: v for k, v in column_mapping.items() if k in df.columns}
        df.rename(columns=existing_mappings, inplace=True)
        
        return df
    
    def _add_current_conditions(self, df: pd.DataFrame, current_data: Dict[str, Any]) -> pd.DataFrame:
        """Add current conditions to the latest timestamp."""
        if df.empty:
            return df
        
        # Get the latest timestamp
        latest_time = df.index.max()
        
        # Add current AQI values if available
        aqi_columns = ['european_aqi', 'us_aqi']
        for col in aqi_columns:
            if col in current_data:
                df.loc[latest_time, col] = current_data[col]
        
        return df
    
    def _calculate_air_quality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived air quality features."""
        
        # Air Quality Index calculations
        if 'pm2_5' in df.columns and 'pm10' in df.columns:
            # PM ratio (indicator of combustion vs dust)
            df['pm_ratio'] = df['pm2_5'] / df['pm10'].replace(0, np.nan)
        
        # Pollution load index (weighted sum of major pollutants)
        pollutant_weights = {
            'pm2_5': 0.3,
            'pm10': 0.2,
            'no2': 0.2,
            'o3': 0.15,
            'so2': 0.1,
            'co': 0.05
        }
        
        pollution_components = []
        for pollutant, weight in pollutant_weights.items():
            if pollutant in df.columns:
                # Normalize to 0-100 scale (approximate)
                if pollutant == 'pm2_5':
                    normalized = (df[pollutant] / 35) * 100  # WHO guideline
                elif pollutant == 'pm10':
                    normalized = (df[pollutant] / 50) * 100  # WHO guideline
                elif pollutant == 'no2':
                    normalized = (df[pollutant] / 40) * 100  # WHO guideline
                elif pollutant == 'o3':
                    normalized = (df[pollutant] / 100) * 100  # WHO guideline
                elif pollutant == 'so2':
                    normalized = (df[pollutant] / 20) * 100  # WHO guideline
                elif pollutant == 'co':
                    normalized = (df[pollutant] / 10000) * 100  # WHO guideline
                else:
                    continue
                
                pollution_components.append(normalized * weight)
        
        if pollution_components:
            df['pollution_index'] = sum(pollution_components)
        
        # Health risk categories
        if 'pm2_5' in df.columns:
            df['pm2_5_risk'] = pd.cut(
                df['pm2_5'],
                bins=[0, 12, 35, 55, 150, float('inf')],
                labels=['Good', 'Moderate', 'Unhealthy_Sensitive', 'Unhealthy', 'Hazardous'],
                include_lowest=True
            )
        
        # Time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        
        # Moving averages (8-hour and 24-hour)
        for col in ['pm2_5', 'pm10', 'o3', 'no2']:
            if col in df.columns:
                df[f'{col}_8h_avg'] = df[col].rolling(window=8, min_periods=1).mean()
                df[f'{col}_24h_avg'] = df[col].rolling(window=24, min_periods=1).mean()
        
        return df

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
        
        # Convert index to datetime
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
        
        # Heat Index calculation (if humidity available)
        if all(col in df.columns for col in ['temperature_2m', 'relative_humidity']):
            df['heat_index'] = self._calculate_heat_index(
                df['temperature_2m'], df['relative_humidity']
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
            
            # Dry spell indicator (consecutive days without precipitation)
            df['is_dry_day'] = df['precipitation'] < 0.1
            df['dry_spell_length'] = df['is_dry_day'].groupby(
                (df['is_dry_day'] != df['is_dry_day'].shift()).cumsum()
            ).cumsum()
        
        # Wind features
        if 'wind_speed_2m' in df.columns:
            # Wind categories (Beaufort scale simplified)
            df['wind_category'] = pd.cut(
                df['wind_speed_2m'],
                bins=[0, 1, 3, 7, 12, float('inf')],
                labels=['Calm', 'Light', 'Moderate', 'Strong', 'Very_Strong']
            )
        
        # Time-based features
        df['month'] = df.index.month
        df['season'] = df['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
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