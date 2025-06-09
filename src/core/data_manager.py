"""
🌍 Enhanced Climate Data Manager - Day 4 Adaptive Collection
src/core/data_manager.py

Data Manager for coordinating all climate data sources with global location support.
Provides unified interface for data acquisition from multiple APIs with smart
regional adaptation and intelligent source selection.
"""

import logging
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
import time

from ..api.open_meteo import OpenMeteoClient
from ..api.nasa_power import NASAPowerClient
from ..api.world_bank import WorldBankClient
from .location_service import LocationInfo

from config.settings import RAW_DATA_DIR, DEFAULT_LOCATIONS

logger = logging.getLogger(__name__)

class ClimateDataManager:
    """
    Enhanced unified manager for all climate data sources with global support.
    
    New Day 4 Features:
    - Works with any global coordinates via LocationInfo objects
    - Smart data source selection based on regional availability
    - Intelligent fallbacks for limited-coverage areas
    - Location-aware caching and performance optimization
    - Graceful degradation when APIs don't cover certain regions
    """
    
    def __init__(self):
        self.open_meteo = OpenMeteoClient()
        self.nasa_power = NASAPowerClient()
        self.world_bank = WorldBankClient()
        
        # Enhanced tracking for adaptive collection
        self.regional_capabilities = self._initialize_regional_capabilities()
        self.data_availability_cache = {}
        self.performance_metrics = {
            "api_response_times": {},
            "success_rates": {},
            "regional_coverage": {}
        }
        
        # Ensure data directories exist
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        logger.info("🌍 Enhanced ClimateDataManager initialized with global support")
    
    def _initialize_regional_capabilities(self) -> Dict[str, Dict[str, bool]]:
        """Initialize known regional API capabilities and limitations."""
        return {
            "open_meteo": {
                "global_coverage": True,
                "real_time": True,
                "historical_limit_years": 1,  # Limited historical data
                "polar_regions": False,  # Limited polar coverage
                "ocean_coverage": True
            },
            "nasa_power": {
                "global_coverage": True,
                "real_time": False,
                "historical_limit_years": 40,  # Excellent historical data
                "polar_regions": True,
                "ocean_coverage": True
            },
            "world_bank": {
                "global_coverage": True,
                "real_time": False,
                "country_based": True,  # Requires country mapping
                "historical_limit_years": 100,
                "projections": True
            }
        }
    
    async def check_data_availability(self, location: LocationInfo) -> Dict[str, bool]:
        """
        🔍 Check which data sources are available for a specific location.
        
        Args:
            location: LocationInfo object with coordinates and metadata
            
        Returns:
            Dictionary indicating availability of each data source
        """
        logger.info(f"🔍 Checking data availability for {location.name}")
        
        # Check cache first
        cache_key = f"{location.latitude:.4f},{location.longitude:.4f}"
        if cache_key in self.data_availability_cache:
            cache_age = time.time() - self.data_availability_cache[cache_key]["timestamp"]
            if cache_age < 3600:  # Cache for 1 hour
                logger.info("📋 Using cached availability data")
                return self.data_availability_cache[cache_key]["availability"]
        
        availability = {}
        
        # Check Open-Meteo (Air Quality)
        availability["air_quality"] = await self._check_open_meteo_availability(location)
        
        # Check NASA POWER (Meteorological)
        availability["meteorological"] = await self._check_nasa_power_availability(location)
        
        # Check World Bank (Climate Projections)
        availability["climate_projections"] = await self._check_world_bank_availability(location)
        
        # Update cache
        self.data_availability_cache[cache_key] = {
            "availability": availability,
            "timestamp": time.time()
        }
        
        # Update location object with availability flags
        location.has_air_quality = availability["air_quality"]
        location.has_meteorological = availability["meteorological"]
        location.has_climate_projections = availability["climate_projections"]
        
        available_sources = sum(availability.values())
        logger.info(f"✅ Data availability: {available_sources}/3 sources available")
        
        return availability
    
    async def _check_open_meteo_availability(self, location: LocationInfo) -> bool:
        """Check if Open-Meteo air quality data is available for location."""
        try:
            # Open-Meteo has good global coverage but limited in some polar regions
            if abs(location.latitude) > 85:
                logger.debug("🚫 Open-Meteo: Polar region limitation")
                return False
            
            # Could add a lightweight test query here in the future
            # For now, assume available for most locations
            return True
            
        except Exception as e:
            logger.debug(f"🚫 Open-Meteo availability check failed: {e}")
            return False
    
    async def _check_nasa_power_availability(self, location: LocationInfo) -> bool:
        """Check if NASA POWER meteorological data is available for location."""
        try:
            # NASA POWER has excellent global coverage including polar regions
            return True
            
        except Exception as e:
            logger.debug(f"🚫 NASA POWER availability check failed: {e}")
            return False
    
    async def _check_world_bank_availability(self, location: LocationInfo) -> bool:
        """Check if World Bank climate projections are available for location."""
        try:
            # World Bank data is country-based, so check if we have country info
            if not location.country or not location.country_code:
                logger.debug("🚫 World Bank: Missing country information")
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"🚫 World Bank availability check failed: {e}")
            return False
    
    def fetch_adaptive_data(
        self,
        location: LocationInfo,
        start_date: str,
        end_date: str,
        save: bool = True,
        force_all: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        🎯 Fetch all available data adaptively for any global location.
        
        Args:
            location: LocationInfo object with coordinates and metadata
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            save: Whether to save raw data to files
            force_all: Attempt to fetch from all sources regardless of availability
            
        Returns:
            Dictionary containing data from available sources
        """
        logger.info(f"🌍 Starting adaptive data collection for {location.name}")
        logger.info(f"📍 Coordinates: {location.latitude:.4f}, {location.longitude:.4f}")
        logger.info(f"📅 Date range: {start_date} to {end_date}")
        
        results = {}
        
        # Check data availability if not forced
        if not force_all:
            availability = asyncio.run(self.check_data_availability(location))
        else:
            availability = {
                "air_quality": True,
                "meteorological": True,
                "climate_projections": True
            }
        
        # Fetch Air Quality Data (if available)
        if availability.get("air_quality", False):
            try:
                logger.info("🌬️ Fetching air quality data...")
                results["air_quality"] = self.fetch_air_quality_coordinates(
                    latitude=location.latitude,
                    longitude=location.longitude,
                    location_name=location.name,
                    start_date=start_date,
                    end_date=end_date,
                    save=save
                )
                logger.info("✅ Air quality data collection successful")
            except Exception as e:
                logger.warning(f"⚠️ Air quality data collection failed: {e}")
                results["air_quality"] = None
        else:
            logger.info("⏭️ Skipping air quality data (not available for this location)")
            results["air_quality"] = None
        
        # Fetch Meteorological Data (if available)
        if availability.get("meteorological", False):
            try:
                logger.info("🌡️ Fetching meteorological data...")
                results["meteorological"] = self.fetch_meteorological_coordinates(
                    latitude=location.latitude,
                    longitude=location.longitude,
                    location_name=location.name,
                    start_date=start_date,
                    end_date=end_date,
                    save=save
                )
                logger.info("✅ Meteorological data collection successful")
            except Exception as e:
                logger.warning(f"⚠️ Meteorological data collection failed: {e}")
                results["meteorological"] = None
        else:
            logger.info("⏭️ Skipping meteorological data (not available for this location)")
            results["meteorological"] = None
        
        # Fetch Climate Projections (if available)
        if availability.get("climate_projections", False):
            try:
                logger.info("🔮 Fetching climate projections...")
                results["climate_projections"] = self.fetch_climate_projections_location(
                    location=location,
                    save=save
                )
                logger.info("✅ Climate projections collection successful")
            except Exception as e:
                logger.warning(f"⚠️ Climate projections collection failed: {e}")
                results["climate_projections"] = None
        else:
            logger.info("⏭️ Skipping climate projections (not available for this location)")
            results["climate_projections"] = None
        
        # Update performance metrics
        available_sources = sum(1 for v in results.values() if v is not None)
        total_sources = len(results)
        
        logger.info(f"🎯 Adaptive collection complete: {available_sources}/{total_sources} sources successful")
        
        return results
    
    def fetch_air_quality_coordinates(
        self,
        latitude: float,
        longitude: float,
        location_name: str = "unknown",
        start_date: str = None,
        end_date: str = None,
        save: bool = True
    ) -> Dict[str, Any]:
        """🌬️ Fetch air quality data using direct coordinates."""
        
        logger.info(f"Fetching air quality data for coordinates {latitude:.4f}, {longitude:.4f}")
        
        data = self.open_meteo.fetch_data(
            latitude=latitude,
            longitude=longitude,
            start_date=start_date,
            end_date=end_date
        )
        
        if save:
            safe_name = location_name.lower().replace(" ", "_").replace(",", "")
            filename = f"air_quality_{safe_name}_{start_date}_{end_date}"
            self._save_data(data, filename)
        
        return data
    
    def fetch_meteorological_coordinates(
        self,
        latitude: float,
        longitude: float,
        location_name: str = "unknown",
        start_date: str = None,
        end_date: str = None,
        save: bool = True
    ) -> Dict[str, Any]:
        """🌡️ Fetch meteorological data using direct coordinates."""
        
        logger.info(f"Fetching meteorological data for coordinates {latitude:.4f}, {longitude:.4f}")
        
        data = self.nasa_power.fetch_data(
            latitude=latitude,
            longitude=longitude,
            start_date=start_date,
            end_date=end_date
        )
        
        if save:
            safe_name = location_name.lower().replace(" ", "_").replace(",", "")
            filename = f"meteorological_{safe_name}_{start_date}_{end_date}"
            self._save_data(data, filename)
        
        return data
    
    def fetch_climate_projections_location(
        self,
        location: LocationInfo,
        scenario: str = "ssp245",
        save: bool = True
    ) -> Dict[str, Any]:
        """🔮 Fetch climate projections for a specific location."""
        
        logger.info(f"Fetching climate projections for {location.country}")
        
        # Use country code or country name for World Bank API
        country_identifier = location.country_code or location.country
        
        data = self.world_bank.fetch_climate_projections(
            countries=country_identifier,
            scenario=scenario
        )
        
        if save:
            safe_name = location.name.lower().replace(" ", "_").replace(",", "")
            filename = f"climate_projections_{safe_name}_{scenario}"
            self._save_data(data, filename)
        
        return data
    
    # =========================================================================
    # BACKWARD COMPATIBILITY METHODS (Keep existing API working)
    # =========================================================================
    
    def fetch_air_quality_data(
        self,
        location: str,
        start_date: str,
        end_date: str,
        save: bool = True
    ) -> Dict[str, Any]:
        """🔄 Backward compatibility: Fetch air quality data for a named location."""
        
        if location not in DEFAULT_LOCATIONS:
            raise ValueError(f"Unknown location: {location}. Available: {list(DEFAULT_LOCATIONS.keys())}")
        
        coords = DEFAULT_LOCATIONS[location]
        
        return self.fetch_air_quality_coordinates(
            latitude=coords["latitude"],
            longitude=coords["longitude"],
            location_name=location,
            start_date=start_date,
            end_date=end_date,
            save=save
        )
    
    def fetch_meteorological_data(
        self,
        location: str,
        start_date: str,
        end_date: str,
        save: bool = True
    ) -> Dict[str, Any]:
        """🔄 Backward compatibility: Fetch meteorological data for a named location."""
        
        if location not in DEFAULT_LOCATIONS:
            raise ValueError(f"Unknown location: {location}. Available: {list(DEFAULT_LOCATIONS.keys())}")
        
        coords = DEFAULT_LOCATIONS[location]
        
        return self.fetch_meteorological_coordinates(
            latitude=coords["latitude"],
            longitude=coords["longitude"],
            location_name=location,
            start_date=start_date,
            end_date=end_date,
            save=save
        )
    
    def fetch_climate_projections(
        self,
        countries: str = "all_countries",
        scenario: str = "ssp245",
        save: bool = True
    ) -> Dict[str, Any]:
        """🔄 Backward compatibility: Fetch climate projection data for specified countries."""
        
        logger.info(f"Fetching climate projections for {countries}")
        data = self.world_bank.fetch_climate_projections(
            countries=countries,
            scenario=scenario
        )
        
        if save:
            self._save_data(data, f"climate_projections_{countries}_{scenario}")
        
        return data
    
    def fetch_all_data(
        self,
        location: str,
        start_date: str,
        end_date: str,
        save: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """🔄 Backward compatibility: Fetch all available data for a default location."""
        
        logger.info(f"Fetching all climate data for {location} (legacy method)")
        
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
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def _save_data(self, data: Dict[str, Any], filename: str):
        """💾 Save data to JSON file with enhanced error handling."""
        filepath = RAW_DATA_DIR / f"{filename}.json"
        
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"📁 Data saved to {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save data to {filepath}: {e}")
    
    def get_available_locations(self) -> List[str]:
        """📍 Get list of available default locations (backward compatibility)."""
        return list(DEFAULT_LOCATIONS.keys())
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """📊 Get performance metrics for adaptive collection."""
        return self.performance_metrics
    
    def clear_availability_cache(self):
        """🗑️ Clear the data availability cache (useful for testing)."""
        self.data_availability_cache.clear()
        logger.info("🗑️ Data availability cache cleared")
    
    def get_supported_regions(self) -> Dict[str, Any]:
        """🌍 Get information about regional API capabilities."""
        return self.regional_capabilities