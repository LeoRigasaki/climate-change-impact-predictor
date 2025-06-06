"""
Open-Meteo Air Quality API client.
Handles air quality data acquisition including pollutants, greenhouse gases, and AQI.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base_api import BaseAPIClient
from config.settings import API_CONFIGS, AIR_QUALITY_PARAMS

logger = logging.getLogger(__name__)

class OpenMeteoClient(BaseAPIClient):
    """Client for Open-Meteo Air Quality API."""
    
    def __init__(self):
        config = API_CONFIGS["open_meteo"]
        super().__init__(
            base_url=config["base_url"],
            timeout=config["timeout"],
            rate_limit=config["rate_limit"]
        )
    
    def fetch_data(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        parameters: Optional[List[str]] = None,
        include_current: bool = True
    ) -> Dict[str, Any]:
        """
        Fetch air quality data from Open-Meteo API.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate  
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            parameters: List of parameters to fetch (defaults to all available)
            include_current: Whether to include current conditions
        
        Returns:
            Dictionary containing API response
        """
        # Validate inputs
        self.validate_coordinates(latitude, longitude)
        self.validate_date_range(start_date, end_date)
        
        # Use default parameters if none specified
        if parameters is None:
            parameters = AIR_QUALITY_PARAMS
        
        # Build request parameters
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(parameters)
        }
        
        # Add current conditions if requested
        if include_current:
            current_params = [
                "european_aqi", "us_aqi", "pm10", "pm2_5", 
                "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", 
                "ozone", "uv_index", "ammonia", "dust"
            ]
            params["current"] = ",".join(current_params)
        
        logger.info(f"Fetching air quality data for coordinates ({latitude}, {longitude})")
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Parameters: {parameters}")
        
        # Make API request
        response = self._make_request(self.base_url, params=params)
        
        # Add metadata
        response["_metadata"] = {
            "source": "Open-Meteo Air Quality API",
            "fetch_time": datetime.now().isoformat(),
            "coordinates": {"latitude": latitude, "longitude": longitude},
            "date_range": {"start": start_date, "end": end_date},
            "parameters": parameters
        }
        
        logger.info("Air quality data fetched successfully")
        return response
    
    def get_available_parameters(self) -> List[str]:
        """Get list of available air quality parameters."""
        return AIR_QUALITY_PARAMS.copy()