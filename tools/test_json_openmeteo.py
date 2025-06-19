#!/usr/bin/env python3
"""
🧪 Enhanced JSON to Climate Data Test - Day 8 Viability Check
tools/test_json_openmeteo.py

Enhanced test: JSON cities → Geocoding → ClimateDataManager → Complete Climate Data
Tests if we can expand dataset using JSON city names + your complete climate data pipeline.
"""

import sys
import json
import asyncio
import logging
import requests
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.core.data_manager import ClimateDataManager
from src.core.location_service import LocationService, LocationInfo

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedJSONTester:
    """
    🧪 Enhanced test: Can we use JSON cities + ClimateDataManager to expand our dataset?
    
    Test Flow:
    1. Load JSON cities/capitals data
    2. Extract sample cities 
    3. Get coordinates using LocationService
    4. Fetch COMPLETE climate data using ClimateDataManager.fetch_adaptive_data()
    5. Report success rate and comprehensive data quality
    
    This gets you: Air Quality + Weather Forecast + Meteorological + Climate Projections
    """
    
    def __init__(self):
        self.location_service = LocationService()
        self.climate_manager = ClimateDataManager()  # Complete climate data pipeline
        
        self.results = {
            "cities_tested": 0,
            "geocoding_success": 0,
            "climate_success": 0,
            "total_records": 0,
            "successful_cities": [],
            "failed_cities": [],
            "data_source_summary": {},
            "test_duration": 0
        }
        
        logger.info("🧪 Enhanced JSON→ClimateDataManager Tester initialized")
    
    def load_json_cities(self, json_file_path: str, max_cities: int = 10) -> List[Dict[str, str]]:
        """
        Load cities from JSON file (either cities or capitals format).
        
        Args:
            json_file_path: Path to JSON file or URL
            max_cities: Maximum cities to test (keep small for quick test)
            
        Returns:
            List of {"city": "name", "country": "country"} dictionaries
        """
        logger.info(f"📂 Loading JSON cities from: {json_file_path}")
        
        cities = []
        
        try:
            # Load JSON data (from file or URL)
            if json_file_path.startswith("http"):
                logger.info("   🌐 Downloading from URL...")
                response = requests.get(json_file_path, timeout=30)
                response.raise_for_status()
                data = response.json()
            else:
                logger.info("   📄 Loading from file...")
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
            logger.info(f"   ✅ Loaded JSON with {len(data)} countries")
            
            # Extract cities from different JSON formats
            cities_extracted = 0
            
            for country_data in data:
                if cities_extracted >= max_cities:
                    break
                    
                country = country_data.get("country", "Unknown")
                
                # Handle different JSON formats
                if "cities" in country_data:
                    # Format: country-by-cities.json
                    city_list = country_data["cities"]
                    if city_list and len(city_list) > 0:
                        # Take first city from each country
                        cities.append({
                            "city": city_list[0],
                            "country": country
                        })
                        cities_extracted += 1
                        logger.info(f"      • {city_list[0]}, {country}")
                        
                elif "capital" in country_data:
                    # Format: country-by-capital-city.json (if it exists)
                    capital = country_data["capital"]
                    if capital:
                        cities.append({
                            "city": capital,
                            "country": country
                        })
                        cities_extracted += 1
                        logger.info(f"      • {capital}, {country}")
            
            logger.info(f"   🎯 Extracted {len(cities)} cities for testing")
            
        except Exception as e:
            logger.error(f"   ❌ Failed to load JSON: {e}")
            
        return cities
    
    async def test_city_complete_climate_data(self, city: str, country: str) -> Optional[Dict]:
        """
        Test getting COMPLETE climate data for a single city using ClimateDataManager.
        
        Args:
            city: City name
            country: Country name
            
        Returns:
            Complete climate data summary or None if failed
        """
        full_name = f"{city}, {country}"
        
        try:
            # Step 1: Get coordinates using LocationService
            location = self.location_service.geocode_location(full_name)
            
            if not location or not location.latitude or not location.longitude:
                logger.warning(f"   ❌ Geocoding failed: {full_name}")
                return None
            
            lat, lon = location.latitude, location.longitude
            logger.info(f"   📍 {city}: {lat:.3f}, {lon:.3f}")
            
            # Step 2: Get COMPLETE climate data using ClimateDataManager
            # This gets: Air Quality + Weather Forecast + Meteorological + Climate Projections
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            
            # Create LocationInfo object for the data manager
            location_info = location
            
            # Use your complete climate data pipeline
            logger.info(f"   🌍 Fetching complete climate data for {city}...")
            complete_climate_data = await self.climate_manager.fetch_adaptive_data(
                location=location_info,
                start_date=start_date,
                end_date=end_date,
                forecast_days=7,  # Get 7-day forecast
                save=False  # Don't save test data
            )
            
            if not complete_climate_data:
                logger.warning(f"   ❌ No climate data: {city}")
                return None
            
            # Analyze the complete multi-source data
            data_summary = self._analyze_complete_climate_data(complete_climate_data)
            data_summary["city"] = city
            data_summary["country"] = country
            data_summary["coordinates"] = {"lat": lat, "lon": lon}
            
            # Log comprehensive results
            sources = ", ".join(data_summary["available_sources"])
            temp = data_summary["avg_temp"]
            records = data_summary["total_records"]
            
            logger.info(f"   ✅ {city}: {records} records, {temp:.1f}°C, sources: {sources}")
            
            return data_summary
            
        except Exception as e:
            logger.warning(f"   ❌ Failed {full_name}: {e}")
            return None
    
    def _analyze_complete_climate_data(self, complete_data: Dict) -> Dict:
        """Analyze complete climate data from ClimateDataManager.fetch_adaptive_data()."""
        
        summary = {
            "total_records": 0,
            "available_sources": [],
            "data_quality": "unknown",
            
            # Weather data (from weather forecast or meteorological)
            "avg_temp": 0.0,
            "avg_humidity": 0.0,
            "avg_precipitation": 0.0,
            "avg_wind_speed": 0.0,
            
            # Air quality data
            "avg_pm25": 0.0,
            "avg_co2": 0.0,
            "avg_ozone": 0.0,
            
            # Data source breakdown
            "source_details": {}
        }
        
        try:
            # Analyze each data source
            for source_name, source_data in complete_data.items():
                if source_name.startswith("_") or not source_data:  # Skip metadata and null sources
                    continue
                
                summary["available_sources"].append(source_name)
                source_summary = {"records": 0, "parameters": []}
                
                # Process different data source types
                if source_name == "air_quality":
                    source_summary = self._analyze_air_quality_data(source_data)
                    if source_summary["avg_pm25"] > 0:
                        summary["avg_pm25"] = source_summary["avg_pm25"]
                    if source_summary["avg_co2"] > 0:
                        summary["avg_co2"] = source_summary["avg_co2"]
                    if source_summary["avg_ozone"] > 0:
                        summary["avg_ozone"] = source_summary["avg_ozone"]
                
                elif source_name == "weather_forecast":
                    source_summary = self._analyze_weather_forecast_data(source_data)
                    if source_summary["avg_temp"] != 0:
                        summary["avg_temp"] = source_summary["avg_temp"]
                    if source_summary["avg_humidity"] > 0:
                        summary["avg_humidity"] = source_summary["avg_humidity"]
                    if source_summary["avg_precipitation"] >= 0:
                        summary["avg_precipitation"] = source_summary["avg_precipitation"]
                    if source_summary["avg_wind_speed"] > 0:
                        summary["avg_wind_speed"] = source_summary["avg_wind_speed"]
                
                elif source_name == "meteorological":
                    source_summary = self._analyze_meteorological_data(source_data)
                    # Use meteorological temp if weather forecast not available
                    if summary["avg_temp"] == 0 and source_summary["avg_temp"] != 0:
                        summary["avg_temp"] = source_summary["avg_temp"]
                
                elif source_name == "climate_projections":
                    source_summary = self._analyze_climate_projections_data(source_data)
                
                summary["source_details"][source_name] = source_summary
                summary["total_records"] += source_summary["records"]
            
            # Overall data quality assessment
            source_count = len(summary["available_sources"])
            if source_count >= 3 and summary["total_records"] > 200:
                summary["data_quality"] = "excellent"
            elif source_count >= 2 and summary["total_records"] > 100:
                summary["data_quality"] = "good"
            elif source_count >= 1 and summary["total_records"] > 0:
                summary["data_quality"] = "limited"
            else:
                summary["data_quality"] = "failed"
        
        except Exception as e:
            logger.warning(f"Complete data analysis failed: {e}")
            summary["data_quality"] = "error"
        
        return summary
    
    def _analyze_air_quality_data(self, air_data: Dict) -> Dict:
        """Analyze air quality data component."""
        summary = {"records": 0, "parameters": [], "avg_pm25": 0.0, "avg_co2": 0.0, "avg_ozone": 0.0}
        
        try:
            if "hourly" in air_data:
                hourly = air_data["hourly"]
                
                if "time" in hourly:
                    summary["records"] = len(hourly["time"])
                
                # PM2.5
                if "pm2_5" in hourly and hourly["pm2_5"]:
                    pm25_values = [p for p in hourly["pm2_5"] if p is not None]
                    if pm25_values:
                        summary["avg_pm25"] = sum(pm25_values) / len(pm25_values)
                        summary["parameters"].append("pm2_5")
                
                # CO2
                if "carbon_dioxide" in hourly and hourly["carbon_dioxide"]:
                    co2_values = [c for c in hourly["carbon_dioxide"] if c is not None]
                    if co2_values:
                        summary["avg_co2"] = sum(co2_values) / len(co2_values)
                        summary["parameters"].append("carbon_dioxide")
                
                # Ozone
                if "ozone" in hourly and hourly["ozone"]:
                    ozone_values = [o for o in hourly["ozone"] if o is not None]
                    if ozone_values:
                        summary["avg_ozone"] = sum(ozone_values) / len(ozone_values)
                        summary["parameters"].append("ozone")
        
        except Exception as e:
            logger.warning(f"Air quality analysis failed: {e}")
        
        return summary
    
    def _analyze_weather_forecast_data(self, weather_data: Dict) -> Dict:
        """Analyze weather forecast data component."""
        summary = {"records": 0, "parameters": [], "avg_temp": 0.0, "avg_humidity": 0.0, 
                  "avg_precipitation": 0.0, "avg_wind_speed": 0.0}
        
        try:
            if "hourly" in weather_data:
                hourly = weather_data["hourly"]
                
                if "time" in hourly:
                    summary["records"] = len(hourly["time"])
                
                # Temperature
                if "temperature_2m" in hourly and hourly["temperature_2m"]:
                    temp_values = [t for t in hourly["temperature_2m"] if t is not None]
                    if temp_values:
                        summary["avg_temp"] = sum(temp_values) / len(temp_values)
                        summary["parameters"].append("temperature_2m")
                
                # Humidity
                if "relative_humidity_2m" in hourly and hourly["relative_humidity_2m"]:
                    humidity_values = [h for h in hourly["relative_humidity_2m"] if h is not None]
                    if humidity_values:
                        summary["avg_humidity"] = sum(humidity_values) / len(humidity_values)
                        summary["parameters"].append("relative_humidity_2m")
                
                # Precipitation
                if "precipitation" in hourly and hourly["precipitation"]:
                    precip_values = [p for p in hourly["precipitation"] if p is not None]
                    if precip_values:
                        summary["avg_precipitation"] = sum(precip_values) / len(precip_values)
                        summary["parameters"].append("precipitation")
                
                # Wind speed
                if "wind_speed_10m" in hourly and hourly["wind_speed_10m"]:
                    wind_values = [w for w in hourly["wind_speed_10m"] if w is not None]
                    if wind_values:
                        summary["avg_wind_speed"] = sum(wind_values) / len(wind_values)
                        summary["parameters"].append("wind_speed_10m")
        
        except Exception as e:
            logger.warning(f"Weather forecast analysis failed: {e}")
        
        return summary
    
    def _analyze_meteorological_data(self, met_data: Dict) -> Dict:
        """Analyze meteorological data component."""
        summary = {"records": 0, "parameters": [], "avg_temp": 0.0}
        
        try:
            # NASA POWER data structure
            if "properties" in met_data and "parameter" in met_data["properties"]:
                parameters = met_data["properties"]["parameter"]
                
                # Temperature data
                if "T2M" in parameters:
                    temp_data = parameters["T2M"]
                    if temp_data:
                        temp_values = [float(v) for v in temp_data.values() if v is not None]
                        if temp_values:
                            summary["avg_temp"] = sum(temp_values) / len(temp_values)
                            summary["records"] = len(temp_values)
                            summary["parameters"].append("T2M")
        
        except Exception as e:
            logger.warning(f"Meteorological analysis failed: {e}")
        
        return summary
    
    def _analyze_climate_projections_data(self, proj_data: Dict) -> Dict:
        """Analyze climate projections data component."""
        summary = {"records": 0, "parameters": [], "projection_scenarios": 0}
        
        try:
            if "data" in proj_data:
                data = proj_data["data"]
                if isinstance(data, list):
                    summary["records"] = len(data)
                    summary["projection_scenarios"] = len(data)
                elif isinstance(data, dict):
                    summary["records"] = 1
                    summary["projection_scenarios"] = 1
        
        except Exception as e:
            logger.warning(f"Climate projections analysis failed: {e}")
        
        return summary
    
    async def run_test(self, json_source: str, max_cities: int = 10) -> Dict:
        """
        Run the complete test: JSON → Cities → Complete Climate Data.
        
        Args:
            json_source: Path to JSON file or URL
            max_cities: Number of cities to test (keep small)
            
        Returns:
            Test results summary
        """
        start_time = time.time()
        
        logger.info("🧪 Starting Enhanced JSON→ClimateDataManager Test")
        logger.info("=" * 60)
        
        # Step 1: Load cities from JSON
        print("\n1️⃣ LOADING CITIES FROM JSON")
        cities = self.load_json_cities(json_source, max_cities)
        
        if not cities:
            logger.error("❌ No cities loaded - test failed")
            return {"success": False, "error": "No cities loaded"}
        
        self.results["cities_tested"] = len(cities)
        
        # Step 2: Test each city with complete climate data collection
        print(f"\n2️⃣ TESTING {len(cities)} CITIES WITH COMPLETE CLIMATE DATA")
        
        data_source_counts = {}
        
        for i, city_info in enumerate(cities, 1):
            city = city_info["city"]
            country = city_info["country"]
            
            logger.info(f"🌍 Testing {i}/{len(cities)}: {city}, {country}")
            
            # Test complete climate data collection
            climate_result = await self.test_city_complete_climate_data(city, country)
            
            if climate_result:
                self.results["successful_cities"].append(climate_result)
                self.results["climate_success"] += 1
                self.results["total_records"] += climate_result["total_records"]
                self.results["geocoding_success"] += 1
                
                # Track data source availability
                for source in climate_result["available_sources"]:
                    data_source_counts[source] = data_source_counts.get(source, 0) + 1
                    
            else:
                self.results["failed_cities"].append({
                    "city": city,
                    "country": country,
                    "reason": "geocoding_or_climate_failed"
                })
        
        # Step 3: Calculate final metrics
        self.results["test_duration"] = time.time() - start_time
        self.results["data_source_summary"] = data_source_counts
        
        return self.results
    
    def print_results(self):
        """Print comprehensive test results."""
        
        print("\n" + "=" * 70)
        print("🧪 ENHANCED JSON→CLIMATE DATA TEST RESULTS")
        print("=" * 70)
        
        # Success rates
        cities_tested = self.results["cities_tested"]
        geocoding_rate = self.results["geocoding_success"] / cities_tested if cities_tested > 0 else 0
        climate_rate = self.results["climate_success"] / cities_tested if cities_tested > 0 else 0
        
        print(f"📊 PERFORMANCE METRICS:")
        print(f"   Cities Tested: {cities_tested}")
        print(f"   Geocoding Success: {self.results['geocoding_success']}/{cities_tested} ({geocoding_rate:.1%})")
        print(f"   Climate Data Success: {self.results['climate_success']}/{cities_tested} ({climate_rate:.1%})")
        print(f"   Total Climate Records: {self.results['total_records']:,}")
        print(f"   Test Duration: {self.results['test_duration']:.1f} seconds")
        
        # Data source availability
        if self.results["data_source_summary"]:
            print(f"\n📡 DATA SOURCE AVAILABILITY:")
            for source, count in self.results["data_source_summary"].items():
                availability = count / self.results["climate_success"] if self.results["climate_success"] > 0 else 0
                print(f"   • {source}: {count}/{self.results['climate_success']} cities ({availability:.1%})")
        
        # Show successful cities with detailed data
        if self.results["successful_cities"]:
            print(f"\n✅ SUCCESSFUL CITIES (Detailed):")
            for city_data in self.results["successful_cities"]:
                quality = city_data["data_quality"]
                records = city_data["total_records"]
                temp = city_data["avg_temp"]
                pm25 = city_data["avg_pm25"]
                sources = len(city_data["available_sources"])
                
                print(f"   • {city_data['city']}, {city_data['country']}:")
                print(f"     📊 {records} records, {sources} sources ({quality})")
                print(f"     🌡️ Temp: {temp:.1f}°C, PM2.5: {pm25:.1f} μg/m³")
        
        # Show failed cities
        if self.results["failed_cities"]:
            print(f"\n❌ FAILED CITIES:")
            for city_data in self.results["failed_cities"]:
                print(f"   • {city_data['city']}, {city_data['country']}: {city_data['reason']}")
        
        # Enhanced recommendation based on comprehensive data
        print(f"\n🎯 RECOMMENDATION:")
        
        avg_sources = sum(len(city["available_sources"]) for city in self.results["successful_cities"]) / max(len(self.results["successful_cities"]), 1)
        has_temperature = any(city["avg_temp"] != 0 for city in self.results["successful_cities"])
        
        if geocoding_rate >= 0.8 and climate_rate >= 0.7 and avg_sources >= 2 and has_temperature:
            print("   ✅ PROCEED: Excellent success rates - JSON expansion highly viable!")
            print("   💡 Use JSON cities to significantly expand your dataset")
            print("   🌟 Multi-source data collection working perfectly")
            print("   🚀 Day 8: Train on 6 rich cities + 20-50 JSON cities with complete climate data")
            
        elif geocoding_rate >= 0.6 and climate_rate >= 0.5 and avg_sources >= 1.5:
            print("   ⚠️  SELECTIVE: Good success - use best cities selectively")
            print("   💡 Cherry-pick most reliable cities with multiple data sources")
            print("   🚀 Day 8: Focus on your 6 cities + 10-15 reliable JSON cities")
            
        else:
            print("   ❌ SKIP: Low success rates or data quality issues")
            print("   💡 Your 6 high-quality cities are better for ML training")
            print("   🚀 Day 8: Build advanced ML on your existing high-quality dataset")


async def main():
    """Main test function with enhanced climate data collection."""
    
    print("🧪 Enhanced JSON→Complete Climate Data Viability Test")
    print("Testing: JSON cities → Coordinates → ClimateDataManager → Complete Climate Data")
    print("=" * 80)
    
    # Initialize enhanced tester
    tester = EnhancedJSONTester()
    
    # JSON source options
    json_sources = {
        "cities": "https://raw.githubusercontent.com/samayo/country-json/master/src/country-by-cities.json",
        "capitals": "https://raw.githubusercontent.com/samayo/country-json/master/src/country-by-capital-city.json"
    }
    
    # Let user choose or default to cities
    print("📂 Available JSON sources:")
    print("   1. country-by-cities.json (cities by country)")
    print("   2. country-by-capital-city.json (capital cities)")
    print()
    
    # Default to cities for automatic testing
    selected_source = json_sources["cities"]
    test_name = "cities"
    
    print(f"🎯 Testing with: {test_name}")
    print(f"📥 Source: {selected_source}")
    print("🌍 Using ClimateDataManager for complete climate data collection")
    
    try:
        # Run the enhanced test (limit to 5 cities for thorough testing)
        results = await tester.run_test(selected_source, max_cities=5)
        
        # Print comprehensive results
        tester.print_results()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = Path("logs") / f"enhanced_climate_test_{timestamp}.json"
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Detailed results saved: {results_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"Enhanced test failed: {e}")
        print(f"❌ Test failed: {e}")
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    print("🧪 Starting Enhanced JSON→Complete Climate Data Test...")
    print("⏱️  Expected duration: 3-8 minutes for 5 cities (comprehensive data)")
    print()
    
    result = asyncio.run(main())
    
    print("\n🎯 ENHANCED TEST COMPLETE")
    print("📋 Use results above to decide Day 8 strategy:")
    print("   ✅ PROCEED → Use JSON to expand dataset with multi-source data")
    print("   ⚠️  SELECTIVE → Use JSON selectively with best data sources") 
    print("   ❌ SKIP → Focus on existing 6-city high-quality dataset")
    print("\n🚀 Ready for Day 8 ML implementation with complete climate data!")