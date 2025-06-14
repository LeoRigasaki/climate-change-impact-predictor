# src/api/open_meteo_forecast.py
"""
OpenMeteo Weather Forecast API client.
Handles weather forecast data acquisition including temperature, precipitation, wind forecasts.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from .base_client import BaseAPIClient
from config.settings import API_CONFIGS

logger = logging.getLogger(__name__)

class OpenMeteoForecastClient(BaseAPIClient):
    """Client for OpenMeteo Weather Forecast API."""
    
    def __init__(self):
        config = API_CONFIGS["open_meteo_forecast"]
        super().__init__(
            base_url=config["base_url"],
            timeout=config["timeout"],
            rate_limit=config["rate_limit"]
        )
    
    def fetch_forecast_data(
        self,
        latitude: float,
        longitude: float,
        forecast_days: int = 7,
        past_days: int = 0,
        include_current: bool = True,
        timezone: str = "auto"
    ) -> Dict[str, Any]:
        """
        Fetch weather forecast data from OpenMeteo Forecast API.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate  
            forecast_days: Number of forecast days (1-16)
            past_days: Number of past days (0-92) 
            include_current: Whether to include current conditions
            timezone: Timezone for data (auto, GMT, or specific timezone)
        
        Returns:
            Dictionary containing API response with hourly, daily, and current data
        """
        # Validate inputs
        self.validate_coordinates(latitude, longitude)
        
        if not (1 <= forecast_days <= 16):
            raise ValueError("forecast_days must be between 1 and 16")
        
        if not (0 <= past_days <= 92):
            raise ValueError("past_days must be between 0 and 92")
        
        # Build request parameters
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "timezone": timezone,
            "forecast_days": forecast_days,
            "past_days": past_days
        }
        
        # Hourly parameters - comprehensive weather data
        hourly_params = [
            "temperature_2m", "relative_humidity_2m", "apparent_temperature",
            "precipitation_probability", "precipitation", "wind_speed_10m", 
            "wind_gusts_10m", "surface_pressure", "cloud_cover", "visibility"
        ]
        params["hourly"] = ",".join(hourly_params)
        
        # Daily parameters - daily summaries and extremes
        daily_params = [
            "temperature_2m_max", "temperature_2m_min", 
            "apparent_temperature_max", "apparent_temperature_min",
            "uv_index_max", "precipitation_sum", "precipitation_hours",
            "precipitation_probability_max", "wind_speed_10m_max", 
            "wind_gusts_10m_max"
        ]
        params["daily"] = ",".join(daily_params)
        
        # Current conditions if requested
        if include_current:
            current_params = [
                "temperature_2m", "relative_humidity_2m", "apparent_temperature",
                "wind_speed_10m", "wind_gusts_10m", "precipitation", 
                "weather_code", "surface_pressure"
            ]
            params["current"] = ",".join(current_params)
        
        logger.info(f"Fetching weather forecast for coordinates ({latitude}, {longitude})")
        logger.info(f"Forecast days: {forecast_days}, Past days: {past_days}")
        
        # Make API request
        response = self._make_request(self.base_url, params=params)
        
        # Add metadata
        response["_metadata"] = {
            "source": "OpenMeteo Weather Forecast API",
            "fetch_time": datetime.now().isoformat(),
            "coordinates": {"latitude": latitude, "longitude": longitude},
            "forecast_days": forecast_days,
            "past_days": past_days,
            "timezone": timezone,
            "parameters": {
                "hourly": hourly_params,
                "daily": daily_params,
                "current": current_params if include_current else []
            }
        }
        
        logger.info("Weather forecast data fetched successfully")
        return response
    
    def fetch_data(self, **kwargs) -> Dict[str, Any]:
        """
        Abstract method implementation - delegates to fetch_forecast_data.
        This satisfies the BaseAPIClient abstract method requirement.
        """
        # Extract common parameters
        latitude = kwargs.get('latitude')
        longitude = kwargs.get('longitude')
        forecast_days = kwargs.get('forecast_days', 7)
        past_days = kwargs.get('past_days', 0)
        
        if latitude is None or longitude is None:
            raise ValueError("latitude and longitude are required parameters")
        
        return self.fetch_forecast_data(
            latitude=latitude,
            longitude=longitude,
            forecast_days=forecast_days,
            past_days=past_days
        )
    
    def get_available_parameters(self) -> Dict[str, List[str]]:
        """Get list of available weather forecast parameters by category."""
        return {
            "hourly": [
                "temperature_2m", "relative_humidity_2m", "apparent_temperature",
                "precipitation_probability", "precipitation", "wind_speed_10m", 
                "wind_gusts_10m", "surface_pressure", "cloud_cover", "visibility",
                "shortwave_radiation", "direct_radiation", "diffuse_radiation",
                "windDirection_10m", "windDirection_80m", "windSpeed_80m",
                "temperature_80m", "soil_temperature_0cm", "soil_moisture_0_1cm"
            ],
            "daily": [
                "temperature_2m_max", "temperature_2m_min", 
                "apparent_temperature_max", "apparent_temperature_min",
                "temperature_2m_mean", "apparent_temperature_mean",
                "sunrise", "sunset", "daylight_duration", "sunshine_duration",
                "uv_index_max", "uv_index_clear_sky_max",
                "precipitation_sum", "rain_sum", "showers_sum", "snowfall_sum",
                "precipitation_hours", "precipitation_probability_max",
                "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant",
                "shortwave_radiation_sum", "et0_fao_evapotranspiration"
            ],
            "current": [
                "temperature_2m", "relative_humidity_2m", "apparent_temperature",
                "is_day", "precipitation", "rain", "showers", "snowfall",
                "weather_code", "cloud_cover", "pressure_msl", "surface_pressure",
                "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"
            ]
        }
    
    def fetch_simple_forecast(
        self,
        latitude: float,
        longitude: float,
        days: int = 3
    ) -> Dict[str, Any]:
        """
        Fetch simplified weather forecast with essential parameters only.
        Useful for quick predictions and reduced API calls.
        """
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "forecast_days": days,
            "hourly": "temperature_2m,precipitation_probability,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
            "current": "temperature_2m,weather_code"
        }
        
        logger.info(f"Fetching simple {days}-day forecast for ({latitude}, {longitude})")
        
        response = self._make_request(self.base_url, params=params)
        response["_metadata"] = {
            "source": "OpenMeteo Weather Forecast API (Simple)",
            "fetch_time": datetime.now().isoformat(),
            "forecast_type": "simple",
            "forecast_days": days
        }
        
        return response