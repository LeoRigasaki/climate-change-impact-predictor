#!/usr/bin/env python3
"""
tools/fetch_city_data.py

Test script to fetch and display climate data for multiple cities.
Shows raw data and app intelligence results for comparison.
"""

import sys
from pathlib import Path
import asyncio
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.core.location_service import LocationService
from src.core.data_manager import ClimateDataManager
from src.core.pipeline import ClimateDataPipeline
from src.features.universal_engine import UniversalFeatureEngine

class CityDataFetcher:
    """Fetch and display climate data for cities."""
    
    def __init__(self):
        self.location_service = LocationService()
        self.data_manager = ClimateDataManager()
        self.pipeline = ClimateDataPipeline()
        self.feature_engine = UniversalFeatureEngine()
    
    async def fetch_city_data(self, city_query):
        """Fetch complete data for a city."""
        
        # Resolve location
        location = self.location_service.geocode_location(city_query)
        if not location:
            print(f"ERROR: Could not resolve location: {city_query}")
            return None
        
        print(f"LOCATION: {location.name}, {location.country}")
        print(f"COORDINATES: {location.latitude:.4f}, {location.longitude:.4f}")
        
        # Data collection
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        
        try:
            collection_results = await self.data_manager.fetch_adaptive_data(
                location=location,
                start_date=start_date,
                end_date=end_date,
                save=False
            )
            
            successful_sources = len([r for r in collection_results.values() if r is not None])
            print(f"DATA_SOURCES: {successful_sources}/3")
            
            # Check data availability
            availability = await self.data_manager.check_data_availability(location)
            print(f"AVAILABILITY: Air={availability.get('air_quality', False)}, Weather={availability.get('meteorological', False)}, Climate={availability.get('climate_projections', False)}")
            
            # Process through pipeline with proper error handling
            try:
                pipeline_results = await self.pipeline.process_global_location(
                    location=location,
                    start_date=start_date,
                    end_date=end_date,
                    skip_collection=False
                )
                print(f"PIPELINE_SUCCESS: {len(pipeline_results)} result keys")
                
                # Debug raw data sources first
                print("RAW_DATA_ANALYSIS:")
                for key in ['air_quality', 'meteorological', 'climate_projections']:
                    if key in pipeline_results and pipeline_results[key]:
                        data_source = pipeline_results[key]
                        print(f"  {key.upper()}: {type(data_source)}")
                        
                        if isinstance(data_source, dict):
                            if 'data' in data_source:
                                raw_data = data_source['data']
                                print(f"    DATA_TYPE: {type(raw_data)}")
                                
                                # Fix data format if it's a list
                                if isinstance(raw_data, list):
                                    print(f"    WARNING: Data is list with {len(raw_data)} items, converting to DataFrame")
                                    if len(raw_data) > 0:
                                        # Convert list to DataFrame
                                        try:
                                            if isinstance(raw_data[0], dict):
                                                df = pd.DataFrame(raw_data)
                                                print(f"    CONVERTED: {len(df)} records, {len(df.columns)} columns")
                                                # Update the pipeline result
                                                pipeline_results[key]['data'] = df
                                            else:
                                                print(f"    ERROR: List items are {type(raw_data[0])}, not dict")
                                        except Exception as convert_error:
                                            print(f"    CONVERSION_ERROR: {convert_error}")
                                elif isinstance(raw_data, pd.DataFrame):
                                    print(f"    DATAFRAME: {len(raw_data)} records, {len(raw_data.columns)} columns")
                                    
                                    # Check for missing value indicators
                                    if 'temperature_2m' in raw_data.columns:
                                        temp_col = raw_data['temperature_2m']
                                        invalid_temps = (temp_col < -100) | (temp_col > 60)
                                        if invalid_temps.any():
                                            print(f"    WARNING: {invalid_temps.sum()} invalid temperature values found")
                                            print(f"    TEMP_RANGE: {temp_col.min():.1f}°C to {temp_col.max():.1f}°C")
                                            # Clean the data
                                            raw_data['temperature_2m'] = raw_data['temperature_2m'].replace([-999, -9999], np.nan)
                                            cleaned_temps = raw_data['temperature_2m'].dropna()
                                            if len(cleaned_temps) > 0:
                                                print(f"    CLEANED_TEMP_RANGE: {cleaned_temps.min():.1f}°C to {cleaned_temps.max():.1f}°C")
                                else:
                                    print(f"    UNEXPECTED_TYPE: {type(raw_data)}")
                            else:
                                print(f"    NO_DATA_KEY: Available keys: {list(data_source.keys())}")
                        else:
                            print(f"    NOT_DICT: {data_source}")
                    else:
                        print(f"  {key.upper()}: Missing or None")
                
                # Check for integrated data
                if 'integrated' in pipeline_results and pipeline_results['integrated']:
                    integrated_data = pipeline_results['integrated']['data']
                    print(f"\nINTEGRATED_DATA: {len(integrated_data)} records, {len(integrated_data.columns)} features")
                    
                    # Analyze temperature data specifically
                    if 'temperature_2m' in integrated_data.columns:
                        temp_data = integrated_data['temperature_2m'].replace([-999, -9999], np.nan)
                        valid_temps = temp_data.dropna()
                        
                        if len(valid_temps) > 0:
                            print(f"TEMPERATURE_ANALYSIS:")
                            print(f"  Valid records: {len(valid_temps)}/{len(temp_data)}")
                            print(f"  Mean: {valid_temps.mean():.1f}°C")
                            print(f"  Range: {valid_temps.min():.1f}°C to {valid_temps.max():.1f}°C")
                            
                            # Climate risk calculation
                            avg_temp = valid_temps.mean()
                            risk_score = self._calculate_climate_risk(avg_temp, location)
                            print(f"CLIMATE_RISK: {risk_score}/100")
                            
                            # Global comparison
                            global_comparison = self._compare_to_global_average(avg_temp)
                            print(f"VS_GLOBAL: {global_comparison}")
                            
                        else:
                            print("TEMPERATURE_ANALYSIS: No valid temperature data")
                    
                    # Air quality analysis
                    if 'pm2_5' in integrated_data.columns:
                        pm25_data = integrated_data['pm2_5'].replace([-999, -9999], np.nan)
                        valid_pm25 = pm25_data.dropna()
                        
                        if len(valid_pm25) > 0:
                            print(f"AIR_QUALITY:")
                            print(f"  PM2.5 Mean: {valid_pm25.mean():.1f} μg/m³")
                            print(f"  PM2.5 Max: {valid_pm25.max():.1f} μg/m³")
                            air_quality_rating = self._get_air_quality_rating(valid_pm25.mean())
                            print(f"  Rating: {air_quality_rating}")
                    
                    # Climate zone
                    climate_zone = self._determine_climate_zone(location, valid_temps.mean() if len(valid_temps) > 0 else None)
                    print(f"CLIMATE_ZONE: {climate_zone}")
                    
                    # Hemisphere and season
                    hemisphere = "Northern" if location.latitude >= 0 else "Southern"
                    season = self._get_current_season(hemisphere)
                    print(f"HEMISPHERE: {hemisphere}")
                    print(f"SEASON: {season}")
                    
                else:
                    print("\nINTEGRATION_ISSUE: No integrated data in pipeline results")
                    
            except Exception as pipeline_error:
                print(f"PIPELINE_ERROR: {pipeline_error}")
                import traceback
                print("FULL_TRACEBACK:")
                traceback.print_exc()
                return location
                
        except Exception as e:
            print(f"COLLECTION_ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            
        print("-" * 50)
        return location
    
    def _calculate_climate_risk(self, avg_temp, location):
        """Calculate climate risk based on temperature and location."""
        if avg_temp is None or pd.isna(avg_temp):
            return 50
        
        # Base risk on temperature extremes
        if avg_temp < 0:  # Very cold
            return max(10, min(30, 30 - avg_temp))
        elif avg_temp < 10:  # Cold
            return max(20, min(40, 40 - avg_temp))
        elif avg_temp < 25:  # Moderate
            return max(30, min(50, 30 + avg_temp))
        elif avg_temp < 35:  # Hot
            return max(60, min(80, 40 + avg_temp))
        else:  # Extreme heat
            return min(100, 60 + (avg_temp - 35) * 2)
    
    def _compare_to_global_average(self, avg_temp):
        """Compare temperature to global average."""
        if avg_temp is None or pd.isna(avg_temp):
            return "No temperature data"
        
        global_avg = 15.0  # Current global average ~15°C
        diff = avg_temp - global_avg
        
        if diff > 25:
            return "Extremely above global average"
        elif diff > 15:
            return "Well above global average"
        elif diff > 5:
            return "Above global average"
        elif diff > -5:
            return "Near global average"
        elif diff > -15:
            return "Below global average"
        else:
            return "Well below global average"
    
    def _get_air_quality_rating(self, pm25_avg):
        """Get air quality rating based on PM2.5."""
        if pm25_avg is None or pd.isna(pm25_avg):
            return "No data"
        elif pm25_avg <= 12:
            return "Good"
        elif pm25_avg <= 35:
            return "Moderate"
        elif pm25_avg <= 55:
            return "Unhealthy for sensitive"
        elif pm25_avg <= 150:
            return "Unhealthy"
        else:
            return "Hazardous"
    
    def _determine_climate_zone(self, location, avg_temp):
        """Determine climate zone based on latitude and temperature."""
        lat = abs(location.latitude)
        
        if lat > 66.5:
            return "Polar"
        elif lat > 23.5:
            if avg_temp is not None and not pd.isna(avg_temp):
                if avg_temp > 35:
                    return "Hot Desert"
                elif avg_temp > 25:
                    return "Subtropical"
                elif avg_temp > 10:
                    return "Temperate"
                else:
                    return "Cold"
            else:
                return "Temperate"
        else:
            return "Tropical"
    
    def _get_current_season(self, hemisphere):
        """Get current season based on hemisphere."""
        month = datetime.now().month
        if hemisphere == "Northern":
            if month in [12, 1, 2]:
                return "Winter"
            elif month in [3, 4, 5]:
                return "Spring"
            elif month in [6, 7, 8]:
                return "Summer"
            else:
                return "Autumn"
        else:
            if month in [12, 1, 2]:
                return "Summer"
            elif month in [3, 4, 5]:
                return "Autumn"
            elif month in [6, 7, 8]:
                return "Winter"
            else:
                return "Spring"

async def main():
    """Test multiple cities."""
    
    fetcher = CityDataFetcher()
    
    test_cities = [
        "Death Valley, California, USA",  # Hottest place
        "Lahore, Pakistan",              # User's location - test first since it works
        "Reykjavik, Iceland",            # Cold place
        "Abadan, Iran",                  # Extreme heat 
    ]
    
    print("CLIMATE DATA ANALYSIS - JUNE 11, 2025")
    print("=" * 50)
    
    for city in test_cities:
        print(f"TESTING: {city}")
        await fetcher.fetch_city_data(city)

if __name__ == "__main__":
    asyncio.run(main())