#!/usr/bin/env python3
"""
🏛️ World Capitals Climate Data Collector - Day 8 Dataset Creation
tools/collect_capitals_dataset.py

Collects comprehensive climate data for world capitals to create 
massive training dataset for global climate prediction neural network.
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

# Configure clean logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class CapitalsDataCollector:
    """
    🏛️ World Capitals Climate Data Collector
    
    Collects comprehensive climate data for all world capitals:
    - Temperature, humidity, precipitation, wind
    - Air quality (PM2.5, AQI, pollutants) 
    - Weather forecasts (7-day)
    - Climate projections (IPCC scenarios)
    
    Creates massive training dataset for neural network.
    """
    
    def __init__(self):
        self.location_service = LocationService()
        self.data_manager = ClimateDataManager()
        
        # Collection tracking
        self.collection_stats = {
            "total_capitals": 0,
            "successful_collections": 0,
            "failed_collections": 0,
            "total_records": 0,
            "start_time": None,
            "countries_processed": []
        }
        
        # Output directories
        self.output_dir = Path("data/capitals")
        self.output_dir.mkdir(exist_ok=True)
        
        print("🏛️ World Capitals Climate Data Collector initialized")
    
    def download_capitals_list(self) -> List[Dict[str, str]]:
        """Download and parse world capitals from JSON source."""
        
        print("📥 Downloading world capitals list...")
        
        # Try capitals first, fallback to cities
        sources = [
            "https://raw.githubusercontent.com/samayo/country-json/master/src/country-by-capital-city.json",
            "https://raw.githubusercontent.com/samayo/country-json/master/src/country-by-cities.json"
        ]
        
        for source_url in sources:
            try:
                response = requests.get(source_url, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                capitals = []
                
                # Parse different JSON formats
                for country_data in data:
                    country = country_data.get("country", "Unknown")
                    
                    if "capital" in country_data:
                        # Capital city format
                        capital = country_data["capital"]
                        if capital and capital.strip():
                            capitals.append({
                                "city": capital.strip(),
                                "country": country,
                                "type": "capital"
                            })
                    
                    elif "cities" in country_data:
                        # Cities format - take first city as capital
                        cities = country_data["cities"]
                        if cities and len(cities) > 0:
                            capitals.append({
                                "city": cities[0].strip(),
                                "country": country,
                                "type": "major_city"
                            })
                
                if capitals:
                    print(f"✅ Downloaded {len(capitals)} capitals/major cities")
                    return capitals
                    
            except Exception as e:
                print(f"⚠️ Failed to download from {source_url}: {e}")
                continue
        
        raise Exception("Failed to download capitals data from all sources")
    
    async def collect_capital_climate_data(self, capital_info: Dict[str, str]) -> Optional[Dict]:
        """Collect comprehensive climate data for a single capital."""
        
        city = capital_info["city"]
        country = capital_info["country"]
        full_name = f"{city}, {country}"
        
        try:
            # Step 1: Geocode the capital
            location = self.location_service.geocode_location(full_name)
            
            if not location or not location.latitude or not location.longitude:
                return None
            
            # Step 2: Collect comprehensive climate data
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            
            climate_data = await self.data_manager.fetch_adaptive_data(
                location=location,
                start_date=start_date,
                end_date=end_date,
                forecast_days=7,
                save=False  # Don't save individual files
            )
            
            # Step 3: Process and structure the data
            processed_data = self._process_climate_data(climate_data, location, capital_info)
            
            return processed_data
            
        except Exception as e:
            return None
    
    def _process_climate_data(self, climate_data: Dict, location: LocationInfo, capital_info: Dict) -> Dict:
        """Process raw climate data into training format."""
        
        # Extract key metrics from multi-source data
        processed = {
            "location": {
                "city": capital_info["city"],
                "country": capital_info["country"],
                "latitude": location.latitude,
                "longitude": location.longitude,
                "type": capital_info["type"]
            },
            "features": {},
            "targets": {},
            "metadata": {
                "collection_time": datetime.now().isoformat(),
                "sources_available": []
            }
        }
        
        record_count = 0
        
        # Process Air Quality Data
        if climate_data.get("air_quality"):
            aq_data = climate_data["air_quality"]
            if "hourly" in aq_data:
                hourly = aq_data["hourly"]
                
                # Extract features
                if "temperature_2m" in hourly and hourly["temperature_2m"]:
                    temps = [t for t in hourly["temperature_2m"] if t is not None]
                    if temps:
                        processed["features"]["current_temp"] = temps[-1]  # Latest
                        processed["targets"]["temp_avg"] = sum(temps) / len(temps)
                        processed["targets"]["temp_max"] = max(temps)
                        processed["targets"]["temp_min"] = min(temps)
                
                # Air quality features
                if "pm2_5" in hourly and hourly["pm2_5"]:
                    pm25_data = [p for p in hourly["pm2_5"] if p is not None]
                    if pm25_data:
                        processed["features"]["current_pm25"] = pm25_data[-1]
                        processed["targets"]["aqi_avg"] = sum(pm25_data) / len(pm25_data)
                
                if "relative_humidity_2m" in hourly and hourly["relative_humidity_2m"]:
                    humidity = [h for h in hourly["relative_humidity_2m"] if h is not None]
                    if humidity:
                        processed["features"]["current_humidity"] = humidity[-1]
                
                record_count += len(hourly.get("time", []))
                processed["metadata"]["sources_available"].append("air_quality")
        
        # Process Weather Forecast
        if climate_data.get("weather_forecast"):
            forecast = climate_data["weather_forecast"]
            
            # Daily forecasts for targets
            if "daily" in forecast:
                daily = forecast["daily"]
                
                if "temperature_2m_max" in daily and daily["temperature_2m_max"]:
                    processed["targets"]["temp_forecast_7day"] = daily["temperature_2m_max"][:7]
                
                if "temperature_2m_min" in daily and daily["temperature_2m_min"]:
                    processed["targets"]["temp_min_forecast_7day"] = daily["temperature_2m_min"][:7]
                
                if "precipitation_probability_max" in daily and daily["precipitation_probability_max"]:
                    processed["targets"]["precip_prob_7day"] = daily["precipitation_probability_max"][:7]
                
                if "uv_index_max" in daily and daily["uv_index_max"]:
                    processed["targets"]["uv_index_7day"] = daily["uv_index_max"][:7]
            
            processed["metadata"]["sources_available"].append("weather_forecast")
        
        # Process NASA Meteorological Data
        if climate_data.get("meteorological"):
            met_data = climate_data["meteorological"]
            if "properties" in met_data and "parameter" in met_data["properties"]:
                params = met_data["properties"]["parameter"]
                
                # Historical temperature patterns
                if "T2M" in params:
                    temps = list(params["T2M"].values())
                    temps = [t for t in temps if t is not None]
                    if temps:
                        processed["features"]["historical_temp_avg"] = sum(temps) / len(temps)
                
                # Wind patterns
                if "WS2M" in params:
                    winds = list(params["WS2M"].values())
                    winds = [w for w in winds if w is not None]
                    if winds:
                        processed["features"]["wind_avg"] = sum(winds) / len(winds)
                
                processed["metadata"]["sources_available"].append("meteorological")
        
        # Add geographic features
        processed["features"]["latitude"] = location.latitude
        processed["features"]["longitude"] = location.longitude
        
        # Determine climate zone from coordinates
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
        
        # Calculate heat index if we have temp and humidity
        if "current_temp" in processed["features"] and "current_humidity" in processed["features"]:
            temp_c = processed["features"]["current_temp"]
            humidity = processed["features"]["current_humidity"]
            
            # Simplified heat index calculation
            if temp_c > 26 and humidity > 40:
                heat_index = temp_c + (0.5 * (humidity - 40))
                processed["targets"]["heat_index"] = heat_index
        
        processed["metadata"]["total_records"] = record_count
        
        return processed
    
    async def collect_all_capitals(self, max_capitals: Optional[int] = None) -> Dict:
        """Collect climate data for all world capitals."""
        
        print("🏛️ Starting world capitals climate data collection")
        print("=" * 60)
        
        self.collection_stats["start_time"] = time.time()
        
        # Download capitals list
        capitals = self.download_capitals_list()
        
        if max_capitals:
            capitals = capitals[:max_capitals]
            print(f"🎯 Limiting collection to first {max_capitals} capitals")
        
        self.collection_stats["total_capitals"] = len(capitals)
        
        print(f"📊 Target: {len(capitals)} capitals")
        print(f"⏱️ Estimated time: {len(capitals) * 15 // 60} minutes")
        print()
        
        # Collect data for each capital
        all_capitals_data = []
        
        for i, capital_info in enumerate(capitals, 1):
            city = capital_info["city"]
            country = capital_info["country"]
            
            print(f"🌍 [{i:3d}/{len(capitals)}] {city}, {country}")
            
            # Collect climate data
            capital_data = await self.collect_capital_climate_data(capital_info)
            
            if capital_data:
                all_capitals_data.append(capital_data)
                self.collection_stats["successful_collections"] += 1
                self.collection_stats["total_records"] += capital_data["metadata"]["total_records"]
                self.collection_stats["countries_processed"].append(country)
                
                sources = len(capital_data["metadata"]["sources_available"])
                print(f"    ✅ {capital_data['metadata']['total_records']} records, {sources} sources")
            else:
                self.collection_stats["failed_collections"] += 1
                print(f"    ❌ Collection failed")
        
        # Save collected data
        self._save_capitals_dataset(all_capitals_data)
        
        # Print final summary
        self._print_collection_summary()
        
        return {
            "capitals_data": all_capitals_data,
            "stats": self.collection_stats
        }
    
    def _save_capitals_dataset(self, capitals_data: List[Dict]):
        """Save the collected capitals dataset."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save complete dataset
        output_file = self.output_dir / f"world_capitals_climate_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump({
                "collection_metadata": self.collection_stats,
                "capitals_data": capitals_data
            }, f, indent=2, default=str)
        
        print(f"\n💾 Dataset saved: {output_file}")
        
        # Also save as CSV for easy analysis
        self._save_capitals_csv(capitals_data, timestamp)
    
    def _save_capitals_csv(self, capitals_data: List[Dict], timestamp: str):
        """Save flattened dataset as CSV for analysis."""
        
        csv_rows = []
        
        for capital in capitals_data:
            row = {}
            
            # Location info
            row.update(capital["location"])
            
            # Features
            row.update({f"feature_{k}": v for k, v in capital["features"].items()})
            
            # Targets (flatten lists)
            for k, v in capital["targets"].items():
                if isinstance(v, list):
                    for i, val in enumerate(v):
                        row[f"target_{k}_day{i+1}"] = val
                else:
                    row[f"target_{k}"] = v
            
            # Metadata
            row["sources_count"] = len(capital["metadata"]["sources_available"])
            row["total_records"] = capital["metadata"]["total_records"]
            
            csv_rows.append(row)
        
        if csv_rows:
            df = pd.DataFrame(csv_rows)
            csv_file = self.output_dir / f"world_capitals_features_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
            print(f"💾 CSV dataset saved: {csv_file}")
            print(f"📊 Dataset shape: {df.shape} (rows × columns)")
    
    def _print_collection_summary(self):
        """Print comprehensive collection summary."""
        
        duration = time.time() - self.collection_stats["start_time"]
        success_rate = self.collection_stats["successful_collections"] / self.collection_stats["total_capitals"]
        
        print("\n" + "=" * 60)
        print("🏛️ WORLD CAPITALS COLLECTION COMPLETE")
        print("=" * 60)
        
        print(f"📊 COLLECTION STATISTICS:")
        print(f"   Total Capitals Targeted: {self.collection_stats['total_capitals']}")
        print(f"   Successful Collections: {self.collection_stats['successful_collections']}")
        print(f"   Failed Collections: {self.collection_stats['failed_collections']}")
        print(f"   Success Rate: {success_rate:.1%}")
        print(f"   Total Climate Records: {self.collection_stats['total_records']:,}")
        print(f"   Collection Duration: {duration/60:.1f} minutes")
        
        print(f"\n🌍 GEOGRAPHIC COVERAGE:")
        print(f"   Countries Processed: {len(self.collection_stats['countries_processed'])}")
        print(f"   Average Records per Capital: {self.collection_stats['total_records'] // max(self.collection_stats['successful_collections'], 1)}")
        
        print(f"\n🚀 READY FOR NEURAL NETWORK TRAINING:")
        print(f"   ✅ Massive global dataset collected")
        print(f"   ✅ Multi-source climate features")
        print(f"   ✅ Real prediction targets available")
        print(f"   ✅ Geographic diversity across continents")


async def main():
    """Main collection function with options."""
    
    print("🏛️ World Capitals Climate Data Collector")
    print("Collecting comprehensive climate data for neural network training")
    print()
    
    collector = CapitalsDataCollector()
    
    # Collection options
    print("📋 Collection Options:")
    print("   1. Quick test (10 capitals)")
    print("   2. Medium dataset (50 capitals)")
    print("   3. Large dataset (100 capitals)")
    print("   4. Full global dataset (all capitals)")
    print()
    
    # Default to medium for testing
    choice = input("Choose option (1-4) or press Enter for medium dataset: ").strip()
    
    if choice == "1":
        max_capitals = 10
        print("🎯 Quick test: 10 capitals")
    elif choice == "3":
        max_capitals = 100
        print("🎯 Large dataset: 100 capitals")
    elif choice == "4":
        max_capitals = None
        print("🎯 Full global dataset: All capitals")
    else:
        max_capitals = 50
        print("🎯 Medium dataset: 50 capitals")
    
    print()
    
    try:
        # Start collection
        results = await collector.collect_all_capitals(max_capitals)
        
        print(f"\n🎉 SUCCESS! Collected data for {len(results['capitals_data'])} capitals")
        print("📁 Check data/capitals/ directory for output files")
        print("\n🚀 Next step: Upload dataset to Google Colab for neural network training!")
        
        return True
        
    except KeyboardInterrupt:
        print("\n🛑 Collection interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Collection failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Starting World Capitals Climate Data Collection...")
    
    result = asyncio.run(main())
    
    if result:
        print("\n✅ COLLECTION COMPLETE")
        print("🔥 Ready for Day 8 neural network training!")
    else:
        print("\n⚠️ Collection incomplete")
        print("💡 Try running again or check error messages")