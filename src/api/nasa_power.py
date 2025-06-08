"""
NASA POWER API client.
Handles meteorological data acquisition including temperature, precipitation, wind, and humidity.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base_client import BaseAPIClient
from config.settings import API_CONFIGS, NASA_POWER_PARAMS

logger = logging.getLogger(__name__)

class NASAPowerClient(BaseAPIClient):
    """Client for NASA POWER API."""
    
    def __init__(self):
        config = API_CONFIGS["nasa_power"]
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
        community: str = "RE"
    ) -> Dict[str, Any]:
        """
        Fetch meteorological data from NASA POWER API.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate  
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            parameters: List of parameters to fetch
            community: NASA POWER community (AG, RE, SB)
        
        Returns:
            Dictionary containing API response
        """
        # Validate inputs
        self.validate_coordinates(latitude, longitude)
        start_dt, end_dt = self.validate_date_range(start_date, end_date)
        
        # Use default parameters if none specified
        if parameters is None:
            parameters = NASA_POWER_PARAMS
        
        # Convert dates to NASA POWER format (YYYYMMDD)
        start_formatted = start_dt.strftime("%Y%m%d")
        end_formatted = end_dt.strftime("%Y%m%d")
        
        # Build request parameters
        params = {
            "parameters": ",".join(parameters),
            "community": community,
            "longitude": longitude,
            "latitude": latitude,
            "start": start_formatted,
            "end": end_formatted,
            "format": "JSON"
        }
        
        logger.info(f"Fetching NASA POWER data for coordinates ({latitude}, {longitude})")
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Parameters: {parameters}")
        logger.info(f"Community: {community}")
        
        # Make API request
        response = self._make_request(self.base_url, params=params)
        
        # Add metadata
        response["_metadata"] = {
            "source": "NASA POWER API",
            "fetch_time": datetime.now().isoformat(),
            "coordinates": {"latitude": latitude, "longitude": longitude},
            "date_range": {"start": start_date, "end": end_date},
            "parameters": parameters,
            "community": community
        }
        
        logger.info("NASA POWER data fetched successfully")
        return response
    
    def get_available_parameters(self) -> List[str]:
        """Get list of available meteorological parameters."""
        return NASA_POWER_PARAMS.copy()