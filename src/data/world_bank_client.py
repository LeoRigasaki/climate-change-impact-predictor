"""
World Bank Climate Change Knowledge Portal (CCKP) API client.
Handles IPCC CMIP6 climate projection data including temperature extremes and scenarios.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base_api import BaseAPIClient
from config.settings import API_CONFIGS

logger = logging.getLogger(__name__)

class WorldBankClient(BaseAPIClient):
    """Client for World Bank CCKP API."""
    
    def __init__(self):
        config = API_CONFIGS["world_bank"]
        super().__init__(
            base_url=config["base_url"],
            timeout=config["timeout"],
            rate_limit=config["rate_limit"]
        )
    
    def fetch_data(self, **kwargs) -> Dict[str, Any]:
        """
        Abstract method implementation - delegates to fetch_climate_projections.
        Required by BaseAPIClient abstract class.
        """
        return self.fetch_climate_projections(**kwargs)
    
    def fetch_climate_projections(
        self,
        countries: str = "all_countries",
        scenario: str = "ssp245",
        variable: str = "txx",
        time_period: str = "2015-2100",
        model: str = "ensemble_all_mean"
    ) -> Dict[str, Any]:
        """
        Fetch climate projection data from World Bank CCKP API.
        
        Args:
            countries: Country selection ("all_countries" or specific codes)
            scenario: SSP scenario (ssp126, ssp245, ssp370, ssp585)
            variable: Climate variable (txx, tas, pr, etc.)
            time_period: Time period for data (e.g., "2015-2100")
            model: Model selection (ensemble_all_mean or specific model)
        
        Returns:
            Dictionary containing API response
        """
        
        # Build API endpoint URL
        endpoint_path = (
            f"cmip6-x0.25_timeseries_{variable}_timeseries_annual_"
            f"{time_period}_median_{scenario}_{model}/{countries}"
        )
        
        url = f"{self.base_url}/{endpoint_path}"
        
        params = {"_format": "json"}
        
        logger.info(f"Fetching climate projections from World Bank CCKP")
        logger.info(f"Countries: {countries}")
        logger.info(f"Scenario: {scenario}")
        logger.info(f"Variable: {variable}")
        logger.info(f"Time period: {time_period}")
        logger.info(f"Model: {model}")
        
        # Make API request
        response = self._make_request(url, params=params)
        
        # Add metadata
        response["_metadata"] = {
            "source": "World Bank CCKP API",
            "fetch_time": datetime.now().isoformat(),
            "countries": countries,
            "scenario": scenario,
            "variable": variable,
            "time_period": time_period,
            "model": model
        }
        
        logger.info("Climate projection data fetched successfully")
        return response
    
    def get_available_scenarios(self) -> List[str]:
        """Get list of available SSP scenarios."""
        return ["ssp126", "ssp245", "ssp370", "ssp585"]
    
    def get_available_variables(self) -> List[str]:
        """Get list of available climate variables."""
        return [
            "tas",      # Mean temperature
            "tasmax",   # Maximum temperature  
            "tasmin",   # Minimum temperature
            "txx",      # Temperature extremes
            "pr",       # Precipitation
            "rx1day",   # Maximum 1-day precipitation
            "cdd",      # Consecutive dry days
            "hi35"      # Heat index above 35°C
        ]