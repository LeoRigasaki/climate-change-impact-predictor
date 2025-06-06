"""
Data Manager for coordinating all climate data sources.
Provides unified interface for data acquisition from multiple APIs.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from .open_meteo_client import OpenMeteoClient
from .nasa_power_client import NASAPowerClient
from config.settings import RAW_DATA_DIR, DEFAULT_LOCATIONS

logger = logging.getLogger(__name__)

class ClimateDataManager:
    """Unified manager for all climate data sources."""
    
    def __init__(self):
        self.open_meteo = OpenMeteoClient()
        self.nasa_power = NASAPowerClient()
        
        # Ensure data directories exist
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    def fetch_air_quality_data(
        self,
        location: str,
        start_date: str,
        end_date: str,
        save: bool = True
    ) -> Dict[str, Any]:
        """Fetch air quality data for a named location."""
        
        if location not in DEFAULT_LOCATIONS:
            raise ValueError(f"Unknown location: {location}. Available: {list(DEFAULT_LOCATIONS.keys())}")
        
        coords = DEFAULT_LOCATIONS[location]
        
        logger.info(f"Fetching air quality data for {location}")
        data = self.open_meteo.fetch_data(
            latitude=coords["latitude"],
            longitude=coords["longitude"],
            start_date=start_date,
            end_date=end_date
        )
        
        if save:
            self._save_data(data, f"air_quality_{location}_{start_date}_{end_date}")
        
        return data
    
    def fetch_meteorological_data(
        self,
        location: str,
        start_date: str,
        end_date: str,
        save: bool = True
    ) -> Dict[str, Any]:
        """Fetch meteorological data for a named location."""
        
        if location not in DEFAULT_LOCATIONS:
            raise ValueError(f"Unknown location: {location}. Available: {list(DEFAULT_LOCATIONS.keys())}")
        
        coords = DEFAULT_LOCATIONS[location]
        
        logger.info(f"Fetching meteorological data for {location}")
        data = self.nasa_power.fetch_data(
            latitude=coords["latitude"],
            longitude=coords["longitude"],
            start_date=start_date,
            end_date=end_date
        )
        
        if save:
            self._save_data(data, f"meteorological_{location}_{start_date}_{end_date}")
        
        return data
    
    def fetch_all_data(
        self,
        location: str,
        start_date: str,
        end_date: str,
        save: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch all available data for a location."""
        
        logger.info(f"Fetching all climate data for {location}")
        
        results = {}
        
        try:
            results["air_quality"] = self.fetch_air_quality_data(
                location, start_date, end_date, save=save
            )
        except Exception as e:
            logger.error(f"Failed to fetch air quality data: {e}")
            results["air_quality"] = None
        
        try:
            results["meteorological"] = self.fetch_meteorological_data(
                location, start_date, end_date, save=save
            )
        except Exception as e:
            logger.error(f"Failed to fetch meteorological data: {e}")
            results["meteorological"] = None
        
        return results
    
    def _save_data(self, data: Dict[str, Any], filename: str):
        """Save data to JSON file."""
        filepath = RAW_DATA_DIR / f"{filename}.json"
        
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Data saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save data to {filepath}: {e}")
    
    def get_available_locations(self) -> List[str]:
        """Get list of available default locations."""
        return list(DEFAULT_LOCATIONS.keys())