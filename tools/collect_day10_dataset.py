#!/usr/bin/env python3
"""
tools/collect_day10_dataset.py
Day 10: Advanced ML Dataset Collection

Collects climate data for ≤150 world capitals for LSTM and multi-output models.
"""

import sys
import json
import asyncio
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


class Day10DataCollector:
    """Clean data collector for Day 10 advanced ML models."""
    
    def __init__(self):
        self.location_service = LocationService()
        self.data_manager = ClimateDataManager()
        self.output_dir = Path("data/day10")
        self.output_dir.mkdir(exist_ok=True)
    
    def get_world_capitals(self, max_count: int = 150) -> List[Dict]:
        """Get list of world capitals from reliable source."""
        try:
            # Use REST Countries API for reliable capital data
            response = requests.get("https://restcountries.com/v3.1/all?fields=name,capital,latlng", timeout=10)
            countries = response.json()
            
            capitals = []
            for country in countries:
                if country.get('capital') and country.get('latlng'):
                    capital_name = country['capital'][0] if isinstance(country['capital'], list) else country['capital']
                    capitals.append({
                        'city': capital_name,
                        'country': country['name']['common'],
                        'type': 'capital'
                    })
            
            # Shuffle and limit to max_count
            import random
            random.shuffle(capitals)
            return capitals[:max_count]
            
        except:
            # Fallback to static list if API fails
            return self.get_fallback_capitals()[:max_count]
    
    def get_fallback_capitals(self) -> List[Dict]:
        """Fallback list of major world capitals."""
        return [
            {'city': 'London', 'country': 'United Kingdom', 'type': 'capital'},
            {'city': 'Paris', 'country': 'France', 'type': 'capital'},
            {'city': 'Berlin', 'country': 'Germany', 'type': 'capital'},
            {'city': 'Madrid', 'country': 'Spain', 'type': 'capital'},
            {'city': 'Rome', 'country': 'Italy', 'type': 'capital'},
            {'city': 'Tokyo', 'country': 'Japan', 'type': 'capital'},
            {'city': 'Beijing', 'country': 'China', 'type': 'capital'},
            {'city': 'Delhi', 'country': 'India', 'type': 'capital'},
            {'city': 'Moscow', 'country': 'Russia', 'type': 'capital'},
            {'city': 'Washington D.C.', 'country': 'United States', 'type': 'capital'},
            {'city': 'Ottawa', 'country': 'Canada', 'type': 'capital'},
            {'city': 'Brasília', 'country': 'Brazil', 'type': 'capital'},
            {'city': 'Buenos Aires', 'country': 'Argentina', 'type': 'capital'},
            {'city': 'Mexico City', 'country': 'Mexico', 'type': 'capital'},
            {'city': 'Cairo', 'country': 'Egypt', 'type': 'capital'},
            {'city': 'Johannesburg', 'country': 'South Africa', 'type': 'capital'},
            {'city': 'Lagos', 'country': 'Nigeria', 'type': 'capital'},
            {'city': 'Sydney', 'country': 'Australia', 'type': 'capital'},
            {'city': 'Bangkok', 'country': 'Thailand', 'type': 'capital'},
            {'city': 'Jakarta', 'country': 'Indonesia', 'type': 'capital'},
            {'city': 'Seoul', 'country': 'South Korea', 'type': 'capital'},
            {'city': 'Singapore', 'country': 'Singapore', 'type': 'capital'},
            {'city': 'Dubai', 'country': 'UAE', 'type': 'capital'},
            {'city': 'Mumbai', 'country': 'India', 'type': 'capital'},
            {'city': 'Istanbul', 'country': 'Turkey', 'type': 'capital'},
            {'city': 'Stockholm', 'country': 'Sweden', 'type': 'capital'},
            {'city': 'Oslo', 'country': 'Norway', 'type': 'capital'},
            {'city': 'Copenhagen', 'country': 'Denmark', 'type': 'capital'},
            {'city': 'Helsinki', 'country': 'Finland', 'type': 'capital'},
            {'city': 'Vienna', 'country': 'Austria', 'type': 'capital'},
            {'city': 'Zurich', 'country': 'Switzerland', 'type': 'capital'},
            {'city': 'Amsterdam', 'country': 'Netherlands', 'type': 'capital'},
            {'city': 'Brussels', 'country': 'Belgium', 'type': 'capital'},
            {'city': 'Prague', 'country': 'Czech Republic', 'type': 'capital'},
            {'city': 'Warsaw', 'country': 'Poland', 'type': 'capital'},
            {'city': 'Budapest', 'country': 'Hungary', 'type': 'capital'},
            {'city': 'Athens', 'country': 'Greece', 'type': 'capital'},
            {'city': 'Lisbon', 'country': 'Portugal', 'type': 'capital'},
            {'city': 'Dublin', 'country': 'Ireland', 'type': 'capital'},
            {'city': 'Edinburgh', 'country': 'Scotland', 'type': 'capital'},
            {'city': 'Reykjavik', 'country': 'Iceland', 'type': 'capital'},
            {'city': 'Tehran', 'country': 'Iran', 'type': 'capital'},
            {'city': 'Riyadh', 'country': 'Saudi Arabia', 'type': 'capital'},
            {'city': 'Doha', 'country': 'Qatar', 'type': 'capital'},
            {'city': 'Kuwait City', 'country': 'Kuwait', 'type': 'capital'},
            {'city': 'Tel Aviv', 'country': 'Israel', 'type': 'capital'},
            {'city': 'Beirut', 'country': 'Lebanon', 'type': 'capital'},
            {'city': 'Amman', 'country': 'Jordan', 'type': 'capital'},
            {'city': 'Baghdad', 'country': 'Iraq', 'type': 'capital'},
            {'city': 'Kabul', 'country': 'Afghanistan', 'type': 'capital'},
            {'city': 'Islamabad', 'country': 'Pakistan', 'type': 'capital'},
            {'city': 'Kathmandu', 'country': 'Nepal', 'type': 'capital'},
            {'city': 'Dhaka', 'country': 'Bangladesh', 'type': 'capital'},
            {'city': 'Colombo', 'country': 'Sri Lanka', 'type': 'capital'},
            {'city': 'Male', 'country': 'Maldives', 'type': 'capital'},
            {'city': 'Yangon', 'country': 'Myanmar', 'type': 'capital'},
            {'city': 'Vientiane', 'country': 'Laos', 'type': 'capital'},
            {'city': 'Phnom Penh', 'country': 'Cambodia', 'type': 'capital'},
            {'city': 'Hanoi', 'country': 'Vietnam', 'type': 'capital'},
            {'city': 'Manila', 'country': 'Philippines', 'type': 'capital'},
            {'city': 'Kuala Lumpur', 'country': 'Malaysia', 'type': 'capital'},
            {'city': 'Brunei', 'country': 'Brunei', 'type': 'capital'},
            {'city': 'Dili', 'country': 'East Timor', 'type': 'capital'},
            {'city': 'Darwin', 'country': 'Australia', 'type': 'capital'},
            {'city': 'Wellington', 'country': 'New Zealand', 'type': 'capital'},
            {'city': 'Suva', 'country': 'Fiji', 'type': 'capital'},
            {'city': 'Port Moresby', 'country': 'Papua New Guinea', 'type': 'capital'},
            {'city': 'Honiara', 'country': 'Solomon Islands', 'type': 'capital'},
            {'city': 'Port Vila', 'country': 'Vanuatu', 'type': 'capital'},
            {'city': 'Nuku\'alofa', 'country': 'Tonga', 'type': 'capital'},
            {'city': 'Apia', 'country': 'Samoa', 'type': 'capital'},
            {'city': 'Tarawa', 'country': 'Kiribati', 'type': 'capital'},
            {'city': 'Funafuti', 'country': 'Tuvalu', 'type': 'capital'},
            {'city': 'Yaren', 'country': 'Nauru', 'type': 'capital'},
            {'city': 'Ngerulmud', 'country': 'Palau', 'type': 'capital'},
            {'city': 'Majuro', 'country': 'Marshall Islands', 'type': 'capital'}
        ]
    
    async def collect_capital_data(self, capital_info: Dict) -> Optional[Dict]:
        """Collect climate data for one capital matching Day 8 CSV structure."""
        city = capital_info["city"]
        country = capital_info["country"]
        full_name = f"{city}, {country}"
        
        try:
            # Geocode location
            location = self.location_service.geocode_location(full_name)
            if not location:
                return None
            
            # Collect comprehensive climate data (30 days historical + 7 days forecast for LSTM)
            end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")  # Yesterday
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")  # 30 days ago
            
            climate_data = await self.data_manager.fetch_adaptive_data(
                location=location,
                start_date=start_date,
                end_date=end_date,
                forecast_days=7,
                save=False
            )
            
            # Process into Day 8 CSV format
            return self.process_to_csv_format(climate_data, location, capital_info)
            
        except Exception as e:
            return None
    
    def process_to_csv_format(self, climate_data: Dict, location: LocationInfo, capital_info: Dict) -> Dict:
        """Process climate data into exact Day 8 CSV column format using working data access patterns."""
        
        # Initialize row with location info
        row = {
            'city': capital_info['city'],
            'country': capital_info['country'],
            'latitude': location.latitude,
            'longitude': location.longitude,
            'type': capital_info['type']
        }
        
        # Initialize all columns with default values
        columns = [
            'feature_current_pm25', 'feature_historical_temp_avg', 'feature_wind_avg',
            'feature_latitude', 'feature_longitude', 'feature_climate_zone',
            'target_aqi_avg'
        ]
        
        # Add 7-day forecast columns
        for day in range(1, 8):
            columns.extend([
                f'target_temp_forecast_7day_day{day}',
                f'target_temp_min_forecast_7day_day{day}',
                f'target_precip_prob_7day_day{day}',
                f'target_uv_index_7day_day{day}'
            ])
        
        columns.extend(['sources_count', 'total_records'])
        
        # Initialize with None/0
        for col in columns:
            row[col] = None
        
        # Extract features
        row['feature_latitude'] = location.latitude
        row['feature_longitude'] = location.longitude
        row['feature_climate_zone'] = self.determine_climate_zone(location.latitude)
        
        # Process air quality data (for current PM2.5 and AQI)
        if climate_data.get('air_quality') and climate_data['air_quality'].get('hourly'):
            aq_hourly = climate_data['air_quality']['hourly']
            
            # Current PM2.5
            if 'pm2_5' in aq_hourly and aq_hourly['pm2_5']:
                pm25_values = [v for v in aq_hourly['pm2_5'] if v is not None]
                if pm25_values:
                    row['feature_current_pm25'] = pm25_values[-1]
                    row['target_aqi_avg'] = sum(pm25_values) / len(pm25_values)
        
        # Process NASA Meteorological Data (for historical temp and wind)
        if climate_data.get('meteorological'):
            met_data = climate_data['meteorological']
            if 'properties' in met_data and 'parameter' in met_data['properties']:
                params = met_data['properties']['parameter']
                
                # Historical temperature average from NASA T2M parameter
                if 'T2M' in params:
                    temp_values = list(params['T2M'].values())
                    temp_values = [t for t in temp_values if t is not None]
                    if temp_values:
                        row['feature_historical_temp_avg'] = sum(temp_values) / len(temp_values)
                
                # Wind average from NASA WS2M parameter
                if 'WS2M' in params:
                    wind_values = list(params['WS2M'].values())
                    wind_values = [w for w in wind_values if w is not None]
                    if wind_values:
                        row['feature_wind_avg'] = sum(wind_values) / len(wind_values)
        
        # Process forecast data (for temperature and precipitation forecasts)
        if climate_data.get('weather_forecast') and climate_data['weather_forecast'].get('daily'):
            daily_forecast = climate_data['weather_forecast']['daily']
            
            # Temperature forecasts
            if 'temperature_2m_max' in daily_forecast and daily_forecast['temperature_2m_max']:
                for i, temp in enumerate(daily_forecast['temperature_2m_max'][:7], 1):
                    if temp is not None:
                        row[f'target_temp_forecast_7day_day{i}'] = temp
            
            # Min temperature forecasts
            if 'temperature_2m_min' in daily_forecast and daily_forecast['temperature_2m_min']:
                for i, temp in enumerate(daily_forecast['temperature_2m_min'][:7], 1):
                    if temp is not None:
                        row[f'target_temp_min_forecast_7day_day{i}'] = temp
            
            # Precipitation probability (try multiple field names)
            precip_field = None
            for field_name in ['precipitation_probability_max', 'precipitation_sum', 'precipitation', 'rain']:
                if field_name in daily_forecast and daily_forecast[field_name]:
                    precip_field = field_name
                    break
            
            if precip_field:
                for i, precip in enumerate(daily_forecast[precip_field][:7], 1):
                    if precip is not None:
                        row[f'target_precip_prob_7day_day{i}'] = precip
            
            # UV Index
            if 'uv_index_max' in daily_forecast and daily_forecast['uv_index_max']:
                for i, uv in enumerate(daily_forecast['uv_index_max'][:7], 1):
                    if uv is not None:
                        row[f'target_uv_index_7day_day{i}'] = uv
        
        # Count sources and records
        sources_count = 0
        total_records = 0
        
        for source_name, source_data in climate_data.items():
            if source_data and isinstance(source_data, dict):
                sources_count += 1
                
                # Count records from different data structures
                if 'hourly' in source_data and source_data['hourly']:
                    # Air quality hourly data
                    hourly_data = source_data['hourly']
                    if isinstance(hourly_data, dict) and hourly_data:
                        first_key = next(iter(hourly_data))
                        if isinstance(hourly_data[first_key], list):
                            total_records += len(hourly_data[first_key])
                
                elif 'daily' in source_data and source_data['daily']:
                    # Weather forecast daily data
                    daily_data = source_data['daily']
                    if isinstance(daily_data, dict) and daily_data:
                        first_key = next(iter(daily_data))
                        if isinstance(daily_data[first_key], list):
                            total_records += len(daily_data[first_key])
                
                elif 'properties' in source_data and 'parameter' in source_data['properties']:
                    # NASA meteorological data
                    params = source_data['properties']['parameter']
                    if isinstance(params, dict) and params:
                        first_param = next(iter(params.values()))
                        if isinstance(first_param, dict):
                            total_records += len(first_param)
        
        row['sources_count'] = sources_count
        row['total_records'] = total_records
        
        return row
    
    def determine_climate_zone(self, latitude: float) -> str:
        """Determine climate zone from latitude."""
        abs_lat = abs(latitude)
        if abs_lat <= 23.5:
            return "tropical"
        elif abs_lat <= 35:
            return "subtropical"
        elif abs_lat <= 50:
            return "temperate"
        elif abs_lat <= 66.5:
            return "continental"
        else:
            return "polar"
    
    async def collect_dataset(self, max_capitals: int = 150) -> str:
        """Collect complete dataset and save to CSV."""
        
        print(f"🌍 Collecting Day 10 dataset for {max_capitals} capitals...")
        
        # Get capitals list
        capitals = self.get_world_capitals(max_capitals)
        
        # Collect data
        successful_rows = []
        
        for i, capital in enumerate(capitals, 1):
            print(f"[{i:3d}/{len(capitals)}] {capital['city']}, {capital['country']}")
            
            row_data = await self.collect_capital_data(capital)
            if row_data:
                successful_rows.append(row_data)
                
                # Debug: Show which key fields are populated
                key_fields = ['feature_historical_temp_avg', 'feature_wind_avg', 'target_precip_prob_7day_day1']
                populated_fields = [field for field in key_fields if row_data.get(field) is not None]
                sources = row_data.get('sources_count', 0)
                
                print(f"    ✅ Success - {sources} sources, fields: {populated_fields}")
            else:
                print(f"    ❌ Failed")
            
            # Brief pause to respect API limits
            await asyncio.sleep(0.5)
        
        # Save to CSV
        if successful_rows:
            df = pd.DataFrame(successful_rows)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file = self.output_dir / f"day10_capitals_dataset_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
            
            print(f"\n✅ Dataset saved: {csv_file}")
            print(f"📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            print(f"🎯 Success rate: {len(successful_rows)}/{len(capitals)} ({len(successful_rows)/len(capitals)*100:.1f}%)")
            
            return str(csv_file)
        else:
            print("❌ No data collected")
            return None


async def main():
    """Main collection function."""
    collector = Day10DataCollector()
    
    # Collect dataset (default 150 capitals)
    csv_file = await collector.collect_dataset(150)
    
    if csv_file:
        print(f"\n🚀 Ready for Day 10 advanced ML!")
        print(f"📁 Upload this file to Google Colab: {csv_file}")
    else:
        print("❌ Collection failed")


if __name__ == "__main__":
    asyncio.run(main())