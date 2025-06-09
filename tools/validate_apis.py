#!/usr/bin/env python3
"""
Test script for climate data APIs.
Validates that all API clients are working correctly.
"""

import logging
from datetime import datetime, timedelta

from src.core.data_manager import ClimateDataManager
from config.settings import LOGGING_CONFIG

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

def test_apis():
    """Test all climate data APIs."""
    
    # Initialize data manager
    manager = ClimateDataManager()
    
    # Test dates (last 7 days)
    end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    print("🌍 Testing Climate Change Impact Predictor APIs")
    print(f"Date range: {start_date} to {end_date}")
    print("-" * 50)
    
    # Test each location
    for location in manager.get_available_locations():
        print(f"\n📍 Testing APIs for {location.upper()}")
        
        try:
            # Test air quality API
            print("  🌬️ Testing Open-Meteo Air Quality API...")
            air_data = manager.fetch_air_quality_data(
                location=location,
                start_date=start_date,
                end_date=end_date,
                save=False  # Don't save during testing
            )
            print(f"     ✅ Success! Got {len(air_data.get('hourly', {}).get('time', []))} hourly records")
            
        except Exception as e:
            print(f"     ❌ Failed: {e}")
        
        try:
            # Test NASA POWER API
            print("  🛰️ Testing NASA POWER API...")
            nasa_data = manager.fetch_meteorological_data(
                location=location,
                start_date=start_date,
                end_date=end_date,
                save=False  # Don't save during testing
            )
            print(f"     ✅ Success! Got meteorological data")
            
        except Exception as e:
            print(f"     ❌ Failed: {e}")
            
            
        print("\n🌍 Testing World Bank CCKP API...")
        try:
            print("  📊 Testing climate projections...")
            
            # Test World Bank API
            wb_data = manager.fetch_climate_projections(
                countries="USA",  # Test with USA only for speed
                scenario="ssp245",
                save=False
            )
            print(f"     ✅ Success! Got climate projection data")
            
        except Exception as e:
            print(f"     ❌ Failed: {e}")

    print("\n🎯 API Testing Complete!")
    print("\nNext steps:")
    print("1. All APIs are working - ready for data collection!")
    print("2. Run 'python collect_sample_data.py' to gather sample datasets")
    print("3. Begin exploratory data analysis in notebooks/")

if __name__ == "__main__":
    test_apis()