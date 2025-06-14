# src/data/processors/weather_forecast_processor.py
"""
Weather Forecast data processor for OpenMeteo Forecast API responses.
Handles weather forecast transformation and feature engineering.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime, timedelta

from .base_processor import BaseDataProcessor

logger = logging.getLogger(__name__)

class WeatherForecastProcessor(BaseDataProcessor):
    """Processor for OpenMeteo weather forecast data."""
    
    def __init__(self):
        super().__init__("WeatherForecastProcessor")
    
    def process(self, data: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Process raw weather forecast data into standardized DataFrames."""
        self.validate_input(data)
        
        logger.info("Processing weather forecast data...")
        
        results = {}
        
        # Process hourly forecast data
        if 'hourly' in data and data['hourly']:
            results['hourly'] = self._process_hourly_data(data['hourly'])
            logger.info(f"Processed hourly forecast: {results['hourly'].shape[0]} records")
        
        # Process daily forecast data
        if 'daily' in data and data['daily']:
            results['daily'] = self._process_daily_data(data['daily'])
            logger.info(f"Processed daily forecast: {results['daily'].shape[0]} records")
        
        # Process current conditions
        if 'current' in data and data['current']:
            results['current'] = self._process_current_data(data['current'])
            logger.info("Processed current conditions")
        
        # Add metadata to all DataFrames
        if '_metadata' in data:
            for df_key, df in results.items():
                if isinstance(df, pd.DataFrame):
                    df.attrs.update(data['_metadata'])
                    df.attrs['processing_type'] = df_key
        
        logger.info(f"Weather forecast processing complete: {len(results)} datasets")
        return results
    
    def _process_hourly_data(self, hourly_data: Dict[str, Any]) -> pd.DataFrame:
        """Process hourly forecast data."""
        if not hourly_data or 'time' not in hourly_data:
            raise ValueError("No hourly time data found in forecast response")
        
        # Create DataFrame from hourly data
        df = pd.DataFrame(hourly_data)
        
        # Convert time column to datetime
        df['datetime'] = pd.to_datetime(df['time'])
        df.set_index('datetime', inplace=True)
        df.drop('time', axis=1, inplace=True)
        
        # Standardize column names
        df = self._standardize_column_names(df)
        
        # Calculate forecast-specific features
        df = self._calculate_forecast_features(df)
        
        # Add temporal features
        df = self._add_temporal_features(df)
        
        # Add metadata
        df = self.add_processing_metadata(df)
        
        return df
    
    def _process_daily_data(self, daily_data: Dict[str, Any]) -> pd.DataFrame:
        """Process daily forecast data."""
        if not daily_data or 'time' not in daily_data:
            raise ValueError("No daily time data found in forecast response")
        
        # Create DataFrame from daily data
        df = pd.DataFrame(daily_data)
        
        # Convert time column to datetime
        df['date'] = pd.to_datetime(df['time'])
        df.set_index('date', inplace=True)
        df.drop('time', axis=1, inplace=True)
        
        # Calculate daily forecast features
        df = self._calculate_daily_features(df)
        
        # Add temporal features
        df['day_of_week'] = df.index.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        df['days_from_today'] = (df.index - pd.Timestamp.now().normalize()).days
        
        # Add metadata
        df = self.add_processing_metadata(df)
        
        return df
    
    def _process_current_data(self, current_data: Dict[str, Any]) -> pd.DataFrame:
        """Process current conditions data."""
        # Convert current data to single-row DataFrame
        current_df = pd.DataFrame([current_data])
        
        if 'time' in current_df.columns:
            current_df['datetime'] = pd.to_datetime(current_df['time'])
            current_df.set_index('datetime', inplace=True)
            current_df.drop('time', axis=1, inplace=True)
        
        # Standardize column names
        current_df = self._standardize_column_names(current_df)
        
        # Add metadata
        current_df = self.add_processing_metadata(current_df)
        
        return current_df
    
    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names for consistency."""
        column_mapping = {
            'temperature_2m': 'temperature_2m',
            'relative_humidity_2m': 'relative_humidity',
            'apparent_temperature': 'apparent_temperature',
            'precipitation_probability': 'precipitation_probability',
            'precipitation': 'precipitation',
            'wind_speed_10m': 'wind_speed',
            'wind_gusts_10m': 'wind_gusts',
            'surface_pressure': 'surface_pressure',
            'cloud_cover': 'cloud_cover',
            'visibility': 'visibility',
            'weather_code': 'weather_code'
        }
        
        # Rename columns that exist
        existing_mappings = {k: v for k, v in column_mapping.items() if k in df.columns}
        df.rename(columns=existing_mappings, inplace=True)
        
        return df
    
    def _calculate_forecast_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate forecast-specific features from hourly data."""
        
        # Heat Index calculation (temperature + humidity)
        if 'temperature_2m' in df.columns and 'relative_humidity' in df.columns:
            # Simplified heat index calculation
            T = df['temperature_2m']
            RH = df['relative_humidity']
            df['heat_index'] = T + 0.5 * (T - 14.4) * (RH / 100)
        
        # Weather comfort index
        if 'apparent_temperature' in df.columns:
            df['comfort_index'] = np.where(
                (df['apparent_temperature'] >= 18) & (df['apparent_temperature'] <= 24), 100,
                np.where(
                    (df['apparent_temperature'] >= 15) & (df['apparent_temperature'] <= 27), 80,
                    np.where(
                        (df['apparent_temperature'] >= 10) & (df['apparent_temperature'] <= 30), 60,
                        40
                    )
                )
            )
        
        # Wind chill factor
        if 'temperature_2m' in df.columns and 'wind_speed' in df.columns:
            T = df['temperature_2m']
            V = df['wind_speed']
            # Wind chill for temperatures below 10°C
            wind_chill = 13.12 + 0.6215 * T - 11.37 * (V ** 0.16) + 0.3965 * T * (V ** 0.16)
            df['wind_chill'] = np.where(T < 10, wind_chill, T)
        
        # Precipitation intensity categories
        if 'precipitation' in df.columns:
            df['precipitation_intensity'] = pd.cut(
                df['precipitation'],
                bins=[0, 0.1, 2.5, 10, 50, float('inf')],
                labels=['None', 'Light', 'Moderate', 'Heavy', 'Extreme'],
                include_lowest=True
            )
        
        # Weather severity index (combines multiple factors)
        severity_factors = []
        
        if 'temperature_2m' in df.columns:
            temp_severity = np.where(
                (df['temperature_2m'] < 0) | (df['temperature_2m'] > 35), 3,
                np.where(
                    (df['temperature_2m'] < 5) | (df['temperature_2m'] > 30), 2,
                    1
                )
            )
            severity_factors.append(temp_severity)
        
        if 'wind_speed' in df.columns:
            wind_severity = np.where(
                df['wind_speed'] > 50, 3,
                np.where(df['wind_speed'] > 25, 2, 1)
            )
            severity_factors.append(wind_severity)
        
        if 'precipitation' in df.columns:
            precip_severity = np.where(
                df['precipitation'] > 10, 3,
                np.where(df['precipitation'] > 2.5, 2, 1)
            )
            severity_factors.append(precip_severity)
        
        if severity_factors:
            df['weather_severity'] = np.mean(severity_factors, axis=0)
        
        return df
    
    def _calculate_daily_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily forecast features."""
        
        # Temperature range
        if 'temperature_2m_max' in df.columns and 'temperature_2m_min' in df.columns:
            df['temperature_range'] = df['temperature_2m_max'] - df['temperature_2m_min']
            df['temperature_mean'] = (df['temperature_2m_max'] + df['temperature_2m_min']) / 2
        
        # Apparent temperature range
        if 'apparent_temperature_max' in df.columns and 'apparent_temperature_min' in df.columns:
            df['apparent_temperature_range'] = df['apparent_temperature_max'] - df['apparent_temperature_min']
            df['apparent_temperature_mean'] = (df['apparent_temperature_max'] + df['apparent_temperature_min']) / 2
        
        # Heat stress risk
        if 'apparent_temperature_max' in df.columns:
            df['heat_stress_risk'] = np.where(
                df['apparent_temperature_max'] > 35, 'High',
                np.where(
                    df['apparent_temperature_max'] > 30, 'Moderate',
                    np.where(df['apparent_temperature_max'] > 25, 'Low', 'None')
                )
            )
        
        # Cold stress risk
        if 'apparent_temperature_min' in df.columns:
            df['cold_stress_risk'] = np.where(
                df['apparent_temperature_min'] < -10, 'High',
                np.where(
                    df['apparent_temperature_min'] < 0, 'Moderate',
                    np.where(df['apparent_temperature_min'] < 5, 'Low', 'None')
                )
            )
        
        # UV risk category
        if 'uv_index_max' in df.columns:
            df['uv_risk'] = pd.cut(
                df['uv_index_max'],
                bins=[0, 3, 6, 8, 11, float('inf')],
                labels=['Low', 'Moderate', 'High', 'Very High', 'Extreme'],
                include_lowest=True
            )
        
        # Precipitation risk
        if 'precipitation_probability_max' in df.columns:
            df['precipitation_risk'] = np.where(
                df['precipitation_probability_max'] > 70, 'High',
                np.where(
                    df['precipitation_probability_max'] > 40, 'Moderate',
                    np.where(df['precipitation_probability_max'] > 20, 'Low', 'Minimal')
                )
            )
        
        # Overall weather comfort score (0-100)
        comfort_factors = []
        
        if 'temperature_mean' in df.columns:
            temp_comfort = 100 - np.abs(df['temperature_mean'] - 20) * 3
            comfort_factors.append(np.clip(temp_comfort, 0, 100))
        
        if 'precipitation_probability_max' in df.columns:
            precip_comfort = 100 - df['precipitation_probability_max']
            comfort_factors.append(np.clip(precip_comfort, 0, 100))
        
        if 'wind_speed_10m_max' in df.columns:
            wind_comfort = 100 - np.clip(df['wind_speed_10m_max'] - 15, 0, 50) * 2
            comfort_factors.append(np.clip(wind_comfort, 0, 100))
        
        if comfort_factors:
            df['weather_comfort_score'] = np.mean(comfort_factors, axis=0)
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features to forecast data."""
        
        # Hour-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        df['is_daytime'] = (df['hour'] >= 6) & (df['hour'] <= 20)
        
        # Time until forecast
        now = pd.Timestamp.now()
        df['hours_from_now'] = (df.index - now).total_seconds() / 3600
        df['days_from_now'] = df['hours_from_now'] / 24
        
        # Forecast confidence (decreases with time)
        df['forecast_confidence'] = np.clip(100 - df['days_from_now'] * 10, 20, 100)
        
        return df