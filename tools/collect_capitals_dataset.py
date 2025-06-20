#!/usr/bin/env python3
"""
🌍 Global Cities Climate Data Collector - Day 9 Enhanced Dataset
tools/collect_cities_data.py

Collects climate data for ALL cities globally with temporal variation.
Enhanced version supporting 63,904+ cities with multiple time periods.
"""

import sys
import json
import asyncio
import logging
import requests
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.core.location_service import LocationService, LocationInfo
from src.core.data_manager import ClimateDataManager

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class GlobalCitiesCollector:
    """
    🌍 Global Cities Climate Data Collector
    
    Enhanced version collecting climate data for ALL cities worldwide
    with multiple time periods for comprehensive training dataset.
    """
    
    def __init__(self):
        self.location_service = LocationService()
        self.data_manager = ClimateDataManager()
        
        # Collection tracking
        self.collection_stats = {
            "total_cities": 0,
            "successful_collections": 0,
            "failed_collections": 0,
            "total_records": 0,
            "start_time": None,
            "countries_processed": [],
            "time_periods": 0
        }
        
        # Output directories
        self.output_dir = Path("data/cities_global")
        self.output_dir.mkdir(exist_ok=True)
        
        print("🌍 Global Cities Climate Data Collector initialized")
    
    def download_all_cities(self) -> List[Dict[str, str]]:
        """Download complete global cities list from JSON source."""
        
        print("📥 Downloading global cities database...")
        
        url = "https://raw.githubusercontent.com/samayo/country-json/master/src/country-by-cities.json"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            all_cities = []
            
            for country_data in data:
                country = country_data.get("country", "Unknown")
                cities_list = country_data.get("cities", [])
                
                for city in cities_list:
                    if city and city.strip():
                        all_cities.append({
                            "city": city.strip(),
                            "country": country,
                            "type": "city"
                        })
            
            print(f"✅ Downloaded {len(all_cities)} cities from {len(data)} countries")
            return all_cities
            
        except Exception as e:
            print(f"❌ Failed to download cities: {e}")
            return []
    
    def select_cities(self, all_cities: List[Dict], mode: str) -> List[Dict]:
        """Select cities based on collection mode."""
        
        if mode == "test":
            selected = all_cities[:50]
            print(f"🎯 Test mode: {len(selected)} cities")
            
        elif mode == "medium":
            selected = all_cities[:1000]
            print(f"🎯 Medium dataset: {len(selected)} cities")
            
        elif mode == "large":
            selected = all_cities[:10000]
            print(f"🎯 Large dataset: {len(selected)} cities")
            
        elif mode == "all":
            selected = all_cities
            print(f"🎯 ALL CITIES MODE: {len(selected)} cities")
            print(f"⚠️  This will take 15-30 HOURS to complete!")
            
            confirm = input("Continue with full collection? (yes/no): ").lower()
            if confirm != "yes":
                print("Collection cancelled")
                return []
        
        else:
            selected = all_cities[:100]
            print(f"🎯 Default: {len(selected)} cities")
        
        return selected
    
    def generate_time_periods(self, num_periods: int = 4) -> List[Dict[str, str]]:
        """Generate multiple time periods for temporal variation."""
        
        periods = []
        base_date = datetime.now()
        
        for i in range(num_periods):
            # Go back in time: current week, 1 month ago, 3 months ago, 6 months ago
            if i == 0:
                days_back = 7  # Last week
            elif i == 1:
                days_back = 30  # 1 month ago
            elif i == 2:
                days_back = 90  # 3 months ago
            else:
                days_back = 180  # 6 months ago
            
            end_date = base_date - timedelta(days=days_back)
            start_date = end_date - timedelta(days=7)  # 7-day periods
            
            periods.append({
                "period_name": f"period_{i+1}",
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "description": f"Period {i+1} ({days_back} days ago)"
            })
        
        return periods
    
    async def collect_city_climate_data(self, city_info: Dict[str, str], time_period: Dict[str, str]) -> Optional[Dict]:
        """Collect climate data for a single city in a specific time period."""
        
        city = city_info["city"]
        country = city_info["country"]
        full_name = f"{city}, {country}"
        
        try:
            # Geocode the city
            location = await self.location_service.geocode_location(full_name)
            
            if not location or not location.latitude or not location.longitude:
                return None
            
            # Collect climate data for this time period
            climate_data = await self.data_manager.fetch_adaptive_data(
                location=location,
                start_date=time_period["start_date"],
                end_date=time_period["end_date"],
                forecast_days=7,
                save=False
            )
            
            # Process the data
            processed_data = self._process_climate_data(climate_data, location, city_info, time_period)
            
            return processed_data
            
        except Exception as e:
            return None
    
    def _process_climate_data(self, climate_data: Dict, location: LocationInfo, 
                            city_info: Dict, time_period: Dict) -> Dict:
        """Process raw climate data with temporal information."""
        
        processed = {
            "location": {
                "city": city_info["city"],
                "country": city_info["country"],
                "latitude": location.latitude,
                "longitude": location.longitude,
                "type": city_info["type"]
            },
            "time_period": {
                "period_name": time_period["period_name"],
                "start_date": time_period["start_date"],
                "end_date": time_period["end_date"],
                "description": time_period["description"]
            },
            "features": {},
            "targets": {},
            "metadata": {
                "collection_time": datetime.now().isoformat(),
                "sources_available": []
            }
        }
        
        record_count = 0
        
        # Process Air Quality Data (same as before, but with -999 fix)
        if climate_data.get("air_quality"):
            aq_data = climate_data["air_quality"]
            if "hourly" in aq_data:
                hourly = aq_data["hourly"]
                
                # Temperature with -999 filter
                if "temperature_2m" in hourly and hourly["temperature_2m"]:
                    temps = [t for t in hourly["temperature_2m"] if t is not None and t != -999.0]
                    if temps:
                        processed["features"]["current_temp"] = temps[-1]
                        processed["targets"]["temp_avg"] = sum(temps) / len(temps)
                        processed["targets"]["temp_max"] = max(temps)
                        processed["targets"]["temp_min"] = min(temps)
                
                # PM2.5 with -999 filter
                if "pm2_5" in hourly and hourly["pm2_5"]:
                    pm25_data = [p for p in hourly["pm2_5"] if p is not None and p != -999.0]
                    if pm25_data:
                        processed["features"]["current_pm25"] = pm25_data[-1]
                        processed["targets"]["aqi_avg"] = sum(pm25_data) / len(pm25_data)
                
                # Humidity with -999 filter
                if "relative_humidity_2m" in hourly and hourly["relative_humidity_2m"]:
                    humidity = [h for h in hourly["relative_humidity_2m"] if h is not None and h != -999.0]
                    if humidity:
                        processed["features"]["current_humidity"] = humidity[-1]
                
                record_count += len([t for t in hourly.get("time", []) if t])
                processed["metadata"]["sources_available"].append("air_quality")
        
        # Process Weather Forecast
        if climate_data.get("weather_forecast"):
            forecast = climate_data["weather_forecast"]
            
            if "daily" in forecast:
                daily = forecast["daily"]
                
                # Extract forecast targets with -999 filter
                for key, target_base in [
                    ("temperature_2m_max", "temp_forecast_7day"),
                    ("temperature_2m_min", "temp_min_forecast_7day"),
                    ("precipitation_probability_max", "precip_prob_7day"),
                    ("uv_index_max", "uv_index_7day")
                ]:
                    if key in daily and daily[key]:
                        values = [v for v in daily[key][:7] if v is not None and v != -999.0]
                        if values:
                            for i, val in enumerate(values):
                                processed["targets"][f"target_{target_base}_day{i+1}"] = val
                
                processed["metadata"]["sources_available"].append("weather_forecast")
        
        # Process NASA Meteorological Data with -999 fix
        if climate_data.get("meteorological"):
            met_data = climate_data["meteorological"]
            if "properties" in met_data and "parameter" in met_data["properties"]:
                params = met_data["properties"]["parameter"]
                
                # Temperature with -999 filter
                if "T2M" in params:
                    temps = [t for t in params["T2M"].values() if t is not None and t != -999.0]
                    if temps:
                        processed["features"]["historical_temp_avg"] = sum(temps) / len(temps)
                
                # Wind with -999 filter
                if "WS2M" in params:
                    winds = [w for w in params["WS2M"].values() if w is not None and w != -999.0]
                    if winds:
                        processed["features"]["wind_avg"] = sum(winds) / len(winds)
                
                processed["metadata"]["sources_available"].append("meteorological")
        
        # Add geographic features
        processed["features"]["latitude"] = location.latitude
        processed["features"]["longitude"] = location.longitude
        
        # Climate zone classification
        lat = abs(location.latitude)
        if lat > 66.5:
            climate_zone = "polar"
        elif lat > 35:
            climate_zone = "temperate"
        elif lat > 23.5:
            climate_zone = "subtropical"
        else:
            climate_zone = "tropical"
        
        processed["features"]["climate_zone"] = climate_zone
        
        # Add temporal features
        start_date = datetime.strptime(time_period["start_date"], "%Y-%m-%d")
        processed["features"]["month"] = start_date.month
        processed["features"]["season"] = (start_date.month % 12) // 3  # 0=winter, 1=spring, etc.
        
        processed["metadata"]["total_records"] = record_count
        
        return processed
    
    async def collect_all_cities_data(self, collection_mode: str = "medium") -> Dict:
        """Collect climate data for cities with temporal variation."""
        
        print("🌍 Starting Global Cities Climate Data Collection")
        print("=" * 60)
        
        self.collection_stats["start_time"] = time.time()
        
        # Download all cities
        all_cities = self.download_all_cities()
        if not all_cities:
            return {"success": False, "error": "Failed to download cities"}
        
        # Select cities based on mode
        selected_cities = self.select_cities(all_cities, collection_mode)
        if not selected_cities:
            return {"success": False, "error": "No cities selected"}
        
        # Generate time periods
        time_periods = self.generate_time_periods(4)  # 4 time periods
        
        self.collection_stats["total_cities"] = len(selected_cities)
        self.collection_stats["time_periods"] = len(time_periods)
        total_collections = len(selected_cities) * len(time_periods)
        
        print(f"📊 Collection Plan:")
        print(f"   Cities: {len(selected_cities)}")
        print(f"   Time periods: {len(time_periods)}")
        print(f"   Total collections: {total_collections}")
        print(f"   Estimated time: {total_collections * 10 // 60} minutes")
        print()
        
        # Collect data
        all_city_data = []
        collection_count = 0
        
        for period in time_periods:
            print(f"📅 {period['description']}")
            
            for i, city_info in enumerate(selected_cities, 1):
                collection_count += 1
                city = city_info["city"]
                country = city_info["country"]
                
                if collection_count % 100 == 0:
                    print(f"🌍 Progress: {collection_count}/{total_collections} ({collection_count/total_collections*100:.1f}%)")
                
                # Collect climate data for this city in this time period
                city_data = await self.collect_city_climate_data(city_info, period)
                
                if city_data:
                    all_city_data.append(city_data)
                    self.collection_stats["successful_collections"] += 1
                    self.collection_stats["total_records"] += city_data["metadata"]["total_records"]
                    
                    if country not in self.collection_stats["countries_processed"]:
                        self.collection_stats["countries_processed"].append(country)
                else:
                    self.collection_stats["failed_collections"] += 1
        
        # Save collected data
        self._save_cities_dataset(all_city_data, collection_mode)
        
        # Print final summary
        self._print_collection_summary()
        
        return {
            "cities_data": all_city_data,
            "stats": self.collection_stats
        }
    
    def _save_cities_dataset(self, cities_data: List[Dict], mode: str):
        """Save the collected cities dataset."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save complete dataset
        output_file = self.output_dir / f"global_cities_climate_{mode}_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump({
                "collection_metadata": self.collection_stats,
                "cities_data": cities_data
            }, f, indent=2, default=str)
        
        print(f"\n💾 Dataset saved: {output_file}")
        
        # Save as CSV for ML training
        self._save_cities_csv(cities_data, mode, timestamp)
    
    def _save_cities_csv(self, cities_data: List[Dict], mode: str, timestamp: str):
        """Save flattened dataset as CSV for ML training."""
        
        csv_rows = []
        
        for city_data in cities_data:
            row = {}
            
            # Location info
            row.update(city_data["location"])
            
            # Time period info
            row.update({f"time_{k}": v for k, v in city_data["time_period"].items()})
            
            # Features
            row.update({f"feature_{k}": v for k, v in city_data["features"].items()})
            
            # Targets
            for k, v in city_data["targets"].items():
                if isinstance(v, list):
                    for i, val in enumerate(v):
                        row[f"target_{k}_day{i+1}"] = val
                else:
                    row[f"target_{k}"] = v
            
            # Metadata
            row["sources_count"] = len(city_data["metadata"]["sources_available"])
            row["total_records"] = city_data["metadata"]["total_records"]
            
            csv_rows.append(row)
        
        if csv_rows:
            df = pd.DataFrame(csv_rows)
            csv_file = self.output_dir / f"global_cities_features_{mode}_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
            print(f"💾 ML Training CSV: {csv_file}")
            print(f"📊 Dataset shape: {df.shape} (rows × columns)")
    
    def _print_collection_summary(self):
        """Print comprehensive collection summary."""
        
        duration = time.time() - self.collection_stats["start_time"]
        success_rate = self.collection_stats["successful_collections"] / (
            self.collection_stats["successful_collections"] + self.collection_stats["failed_collections"]
        ) if (self.collection_stats["successful_collections"] + self.collection_stats["failed_collections"]) > 0 else 0
        
        print("\n" + "=" * 60)
        print("🌍 GLOBAL CITIES COLLECTION COMPLETE")
        print("=" * 60)
        
        print(f"📊 COLLECTION STATISTICS:")
        print(f"   Total Collections Attempted: {self.collection_stats['successful_collections'] + self.collection_stats['failed_collections']}")
        print(f"   Successful Collections: {self.collection_stats['successful_collections']}")
        print(f"   Failed Collections: {self.collection_stats['failed_collections']}")
        print(f"   Success Rate: {success_rate:.1%}")
        print(f"   Total Climate Records: {self.collection_stats['total_records']:,}")
        print(f"   Collection Duration: {duration/60:.1f} minutes")
        
        print(f"\n🌍 COVERAGE:")
        print(f"   Countries Processed: {len(self.collection_stats['countries_processed'])}")
        print(f"   Time Periods: {self.collection_stats['time_periods']}")
        print(f"   Average Records per Collection: {self.collection_stats['total_records'] // max(self.collection_stats['successful_collections'], 1)}")
        
        print(f"\n🚀 MACHINE LEARNING DATASET:")
        print(f"   ✅ Massive temporal + geographic dataset")
        print(f"   ✅ Global climate pattern coverage")
        print(f"   ✅ Multiple time periods for seasonal learning")
        print(f"   ✅ Ready for advanced neural network training")


async def main():
    """Main collection function with enhanced options."""
    
    print("🌍 Global Cities Climate Data Collector - Day 9 Enhanced")
    print("Collect climate data for ALL cities with temporal variation")
    print()
    
    collector = GlobalCitiesCollector()
    
    # Collection options
    print("📋 Collection Options:")
    print("   1. Test (50 cities × 4 time periods = 200 samples)")
    print("   2. Medium (1,000 cities × 4 time periods = 4,000 samples)")
    print("   3. Large (10,000 cities × 4 time periods = 40,000 samples)")
    print("   4. ALL CITIES (63,904 cities × 4 time periods = 255,616 samples)")
    print()
    
    choice = input("Choose option (1-4): ").strip()
    
    mode_map = {"1": "test", "2": "medium", "3": "large", "4": "all"}
    mode = mode_map.get(choice, "test")
    
    if mode == "all":
        print("⚠️  WARNING: This will take 15-30 HOURS and collect 250k+ samples!")
    
    print()
    
    try:
        # Start collection
        results = await collector.collect_all_cities_data(mode)
        
        if results.get("success", True):
            print(f"\n🎉 SUCCESS! Collected data for {len(results['cities_data'])} city/time combinations")
            print("📁 Check data/cities_global/ directory for output files")
            print("\n🚀 Ready for Day 9 neural network training!")
        else:
            print(f"\n❌ Collection failed: {results.get('error', 'Unknown error')}")
        
        return True
        
    except KeyboardInterrupt:
        print("\n🛑 Collection interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Collection failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Starting Global Cities Climate Data Collection...")
    
    result = asyncio.run(main())
    
    if result:
        print("\n✅ COLLECTION COMPLETE")
        print("🔥 Ready for Day 9 neural network training with massive dataset!")
    else:
        print("\n⚠️ Collection incomplete")
        print("💡 Try smaller dataset size or check error messages")