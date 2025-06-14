# src/core/data_manager.py
"""
🌍 Enhanced Climate Data Manager - Day 8 Weather Forecast Integration
src/core/data_manager.py

Data Manager for coordinating all climate data sources with weather forecast support.
Now includes 4 data sources: Air Quality + Weather Forecast + Meteorological + Climate Projections
"""

import logging
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import time

from ..api.open_meteo import OpenMeteoClient
from ..api.open_meteo_forecast import OpenMeteoForecastClient
from ..api.nasa_power import NASAPowerClient
from ..api.world_bank import WorldBankClient
from .location_service import LocationInfo

from config.settings import RAW_DATA_DIR, DEFAULT_LOCATIONS

logger = logging.getLogger(__name__)

class ClimateDataManager:
    """
    Enhanced unified manager for all climate data sources with weather forecast support.
    
    New Day 8 Features:
    - Weather forecast integration (7-day predictions)
    - Combined current + forecast data collection
    - Forecast-aware data availability checking
    - Enhanced adaptive collection with prediction capabilities
    """
    
    def __init__(self):
        self.open_meteo = OpenMeteoClient()
        self.open_meteo_forecast = OpenMeteoForecastClient()  # NEW WEATHER FORECAST API
        self.nasa_power = NASAPowerClient()
        self.world_bank = WorldBankClient()
        
        # Enhanced tracking for adaptive collection + forecasts
        self.regional_capabilities = self._initialize_regional_capabilities()
        self.data_availability_cache = {}
        self.performance_metrics = {
            "api_response_times": {},
            "success_rates": {},
            "regional_coverage": {},
            "forecast_accuracy": {}  # NEW
        }
        
        # Ensure data directories exist
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        logger.info("🌍 Enhanced ClimateDataManager initialized with weather forecast support")
    
    def _initialize_regional_capabilities(self) -> Dict[str, Dict[str, bool]]:
        """Initialize known regional API capabilities including forecasts."""
        return {
            "air_quality": {
                "global_coverage": True,
                "real_time": True,
                "historical_limit_years": 1,
                "polar_regions": False,  # Limited polar coverage
                "ocean_coverage": True,
                "forecast": False
            },
            "weather_forecast": {  # NEW
                "global_coverage": True,
                "real_time": True,
                "polar_regions": True,  # Better polar coverage than air quality
                "ocean_coverage": True,
                "forecast": True,
                "max_forecast_days": 16,
                "historical_limit_years": 0  # Primarily forecast-focused
            },
            "meteorological": {
                "global_coverage": True,
                "real_time": False,
                "historical_limit_years": 40,
                "polar_regions": True,
                "ocean_coverage": True,
                "forecast": False
            },
            "climate_projections": {
                "global_coverage": True,
                "real_time": False,
                "country_based": True,
                "historical_limit_years": 100,
                "projections": True,
                "forecast": True
            }
        }
    
    async def check_data_availability(self, location: LocationInfo) -> Dict[str, bool]:
        """
        🔍 Check data availability for location including weather forecasts.
        
        Args:
            location: LocationInfo object with coordinates and metadata
            
        Returns:
            Dictionary indicating availability of each data source including forecasts
        """
        logger.info(f"🔍 Checking data availability for {location.name} (including forecasts)")
        
        # Check cache first
        cache_key = f"{location.latitude:.4f},{location.longitude:.4f}"
        if cache_key in self.data_availability_cache:
            cache_age = time.time() - self.data_availability_cache[cache_key]["timestamp"]
            if cache_age < 3600:  # Cache for 1 hour
                logger.info("📋 Using cached availability data")
                return self.data_availability_cache[cache_key]["availability"]
        
        availability = {}
        
        # Check Air Quality (existing)
        availability["air_quality"] = await self._check_open_meteo_availability(location)
        
        # Check Weather Forecast (NEW)
        availability["weather_forecast"] = await self._check_weather_forecast_availability(location)
        
        # Check NASA POWER (existing)
        availability["meteorological"] = await self._check_nasa_power_availability(location)
        
        # Check World Bank (existing)
        availability["climate_projections"] = await self._check_world_bank_availability(location)
        
        # Update cache
        self.data_availability_cache[cache_key] = {
            "availability": availability,
            "timestamp": time.time()
        }
        
        # Update location object with availability flags
        location.has_air_quality = availability["air_quality"]
        location.has_weather_forecast = availability["weather_forecast"]  # NEW
        location.has_meteorological = availability["meteorological"]
        location.has_climate_projections = availability["climate_projections"]
        
        available_sources = sum(availability.values())
        logger.info(f"✅ Data availability: {available_sources}/4 sources available (including forecasts)")
        
        return availability
    
    async def _check_weather_forecast_availability(self, location: LocationInfo) -> bool:
        """Check if weather forecast data is available for location."""
        try:
            # Weather forecasts have excellent global coverage
            # Only limitation is extreme polar regions (but less restrictive than air quality)
            if abs(location.latitude) > 89:
                logger.debug("🚫 Weather Forecast: Extreme polar region limitation")
                return False
            
            # OpenMeteo weather forecast has global coverage
            return True
            
        except Exception as e:
            logger.debug(f"🚫 Weather forecast availability check failed: {e}")
            return False
    
    async def fetch_weather_forecast_data(
        self,
        location: LocationInfo,
        forecast_days: int = 7,
        past_days: int = 0,
        save: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        📊 Fetch weather forecast data for a location.
        
        Args:
            location: LocationInfo object
            forecast_days: Number of forecast days (1-16)
            past_days: Number of past days (0-92)
            save: Whether to save raw data
            
        Returns:
            Dictionary containing forecast data or None if failed
        """
        logger.info(f"🌤️ Fetching {forecast_days}-day weather forecast for {location.name}")
        
        try:
            start_time = time.time()
            
            # Fetch forecast data
            data = self.open_meteo_forecast.fetch_forecast_data(
                latitude=location.latitude,
                longitude=location.longitude,
                forecast_days=forecast_days,
                past_days=past_days
            )
            
            fetch_time = time.time() - start_time
            self.performance_metrics["api_response_times"]["weather_forecast"] = fetch_time
            
            if save:
                # Save raw forecast data
                safe_name = location.name.lower().replace(" ", "_").replace(",", "")
                filename = f"weather_forecast_{safe_name}_{forecast_days}d"
                self._save_data(data, filename)
                logger.info(f"💾 Weather forecast data saved")
            
            # Update success metrics
            if "weather_forecast" not in self.performance_metrics["success_rates"]:
                self.performance_metrics["success_rates"]["weather_forecast"] = []
            self.performance_metrics["success_rates"]["weather_forecast"].append(True)
            
            logger.info(f"✅ Weather forecast data fetched successfully in {fetch_time:.2f}s")
            return data
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch weather forecast for {location.name}: {e}")
            
            # Update failure metrics
            if "weather_forecast" not in self.performance_metrics["success_rates"]:
                self.performance_metrics["success_rates"]["weather_forecast"] = []
            self.performance_metrics["success_rates"]["weather_forecast"].append(False)
            
            return None
    
    async def fetch_adaptive_data(
        self,
        location: LocationInfo,
        start_date: str,
        end_date: str,
        forecast_days: int = 7,
        save: bool = True,
        force_all: bool = False
    ) -> Dict[str, Any]:
        """
        🎯 Enhanced adaptive data collection including weather forecasts.
        
        Args:
            location: LocationInfo object
            start_date: Historical data start date
            end_date: Historical data end date
            forecast_days: Number of forecast days
            save: Whether to save raw data
            force_all: Whether to attempt all sources regardless of availability
            
        Returns:
            Dictionary with data from all available sources including forecasts
        """
        logger.info(f"🌍 Enhanced adaptive data collection for {location.name}")
        logger.info(f"📍 Coordinates: {location.latitude:.4f}, {location.longitude:.4f}")
        logger.info(f"📅 Historical: {start_date} to {end_date}, Forecast: {forecast_days} days")
        
        results = {}
        
        # Check availability first (unless forcing)
        if not force_all:
            availability = await self.check_data_availability(location)
        else:
            availability = {
                "air_quality": True, 
                "weather_forecast": True,  # NEW
                "meteorological": True, 
                "climate_projections": True
            }
        
        # Collect Air Quality (existing)
        if availability.get("air_quality", False):
            try:
                logger.info("🌬️ Fetching air quality data...")
                air_quality_data = self.fetch_air_quality_coordinates(
                    latitude=location.latitude,
                    longitude=location.longitude,
                    location_name=location.name,
                    start_date=start_date,
                    end_date=end_date,
                    save=save
                )
                results["air_quality"] = air_quality_data
                logger.info("✅ Air quality data collection successful")
            except Exception as e:
                logger.warning(f"⚠️ Air quality collection failed: {e}")
                results["air_quality"] = None
        else:
            logger.info("⏭️ Skipping air quality data (not available for this location)")
            results["air_quality"] = None
        
        # Collect Weather Forecast (NEW)
        if availability.get("weather_forecast", False):
            try:
                logger.info("🌤️ Fetching weather forecast data...")
                forecast_data = await self.fetch_weather_forecast_data(
                    location=location,
                    forecast_days=forecast_days,
                    save=save
                )
                results["weather_forecast"] = forecast_data
                logger.info(f"✅ {forecast_days}-day weather forecast collected")
            except Exception as e:
                logger.warning(f"⚠️ Weather forecast collection failed: {e}")
                results["weather_forecast"] = None
        else:
            logger.info("⏭️ Skipping weather forecast data (not available for this location)")
            results["weather_forecast"] = None
        
        # Collect Meteorological (existing)
        if availability.get("meteorological", False):
            try:
                logger.info("🌡️ Fetching meteorological data...")
                met_data = self.fetch_meteorological_coordinates(
                    latitude=location.latitude,
                    longitude=location.longitude,
                    location_name=location.name,
                    start_date=start_date,
                    end_date=end_date,
                    save=save
                )
                results["meteorological"] = met_data
                logger.info("✅ Meteorological data collection successful")
            except Exception as e:
                logger.warning(f"⚠️ Meteorological collection failed: {e}")
                results["meteorological"] = None
        else:
            logger.info("⏭️ Skipping meteorological data (not available for this location)")
            results["meteorological"] = None
        
        # Collect Climate Projections (existing)
        if availability.get("climate_projections", False):
            try:
                logger.info("🔮 Fetching climate projections...")
                climate_data = self.fetch_climate_projections_location(
                    location=location,
                    save=save
                )
                results["climate_projections"] = climate_data
                logger.info("✅ Climate projections collection successful")
            except Exception as e:
                logger.warning(f"⚠️ Climate projections collection failed: {e}")
                results["climate_projections"] = None
        else:
            logger.info("⏭️ Skipping climate projections (not available for this location)")
            results["climate_projections"] = None
        
        # Summary
        successful_sources = len([r for r in results.values() if r is not None])
        total_sources = len(results)
        
        logger.info(f"🎯 Enhanced adaptive collection complete: {successful_sources}/{total_sources} sources successful")
        
        return results
    
    async def fetch_complete_dataset(
        self,
        location: LocationInfo,
        historical_days: int = 30,
        forecast_days: int = 7
    ) -> Dict[str, Any]:
        """
        📊 Fetch complete dataset combining historical data and forecasts.
        Perfect for Day 8 ML model training and prediction.
        
        Args:
            location: LocationInfo object
            historical_days: Days of historical data
            forecast_days: Days of forecast data
            
        Returns:
            Complete dataset with historical + forecast data
        """
        logger.info(f"📊 Fetching complete dataset for {location.name}")
        logger.info(f"📈 Historical: {historical_days} days, Forecast: {forecast_days} days")
        
        # Calculate date ranges
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=historical_days)).strftime("%Y-%m-%d")
        
        # Fetch all data
        results = await self.fetch_adaptive_data(
            location=location,
            start_date=start_date,
            end_date=end_date,
            forecast_days=forecast_days,
            save=True
        )
        
        # Add metadata about the complete dataset
        results["_dataset_metadata"] = {
            "location": {
                "name": location.name,
                "latitude": location.latitude,
                "longitude": location.longitude,
                "country": location.country
            },
            "time_coverage": {
                "historical_start": start_date,
                "historical_end": end_date,
                "historical_days": historical_days,
                "forecast_days": forecast_days,
                "total_coverage_days": historical_days + forecast_days
            },
            "data_sources": [k for k, v in results.items() if v is not None and k != "_dataset_metadata"],
            "collection_time": datetime.now().isoformat(),
            "purpose": "Complete dataset for ML model training and prediction"
        }
        
        logger.info(f"✅ Complete dataset ready: {historical_days + forecast_days} days total coverage")
        return results
    
    # =========================================================================
    # EXISTING API COMPATIBILITY METHODS (Keep all existing methods)
    # =========================================================================
    
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
    
    def fetch_adaptive_data_sync(
        self,
        location: LocationInfo,
        start_date: str,
        end_date: str,
        forecast_days: int = 7,  # NEW PARAMETER
        save: bool = True,
        force_all: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        🔄 Synchronous wrapper for fetch_adaptive_data (for backward compatibility).
        """
        try:
            # Try to get existing event loop
            asyncio.get_running_loop()
            # If we're in an async context, we need to handle this differently
            # Use a thread to run the async code
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: asyncio.run(self.fetch_adaptive_data(location, start_date, end_date, forecast_days, save, force_all))
                )
                return future.result()
        except RuntimeError:
            # No event loop running, safe to create one
            return asyncio.run(self.fetch_adaptive_data(location, start_date, end_date, forecast_days, save, force_all))
    
    def check_data_availability_sync(self, location: LocationInfo) -> Dict[str, bool]:
        """
        🔄 Synchronous wrapper for check_data_availability (for backward compatibility).
        """
        try:
            # Try to get existing event loop
            asyncio.get_running_loop()
            # If we're in an async context, we need to handle this differently
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: asyncio.run(self.check_data_availability(location))
                )
                return future.result()
        except RuntimeError:
            # No event loop running, safe to create one
            return asyncio.run(self.check_data_availability(location))
    
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
        """📊 Get performance metrics for adaptive collection including forecasts."""
        return self.performance_metrics
    
    def clear_availability_cache(self):
        """🗑️ Clear the data availability cache (useful for testing)."""
        self.data_availability_cache.clear()
        logger.info("🗑️ Data availability cache cleared")
    
    def get_supported_regions(self) -> Dict[str, Any]:
        """🌍 Get information about regional API capabilities including forecasts."""
        return self.regional_capabilities
    
    def get_forecast_capabilities(self) -> Dict[str, Any]:
        """🌤️ Get information about weather forecast capabilities."""
        return {
            "max_forecast_days": 16,
            "available_parameters": {
                "hourly": ["temperature_2m", "precipitation", "wind_speed_10m", "humidity"],
                "daily": ["temperature_max", "temperature_min", "precipitation_sum", "uv_index"],
                "current": ["temperature_2m", "weather_code", "wind_speed"]
            },
            "global_coverage": True,
            "polar_regions": True,
            "update_frequency": "hourly"
        }