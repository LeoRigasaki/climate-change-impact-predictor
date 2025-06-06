#!/usr/bin/env python3
"""
Sample data collection script for Climate Change Impact Predictor.
Collects representative datasets from all three major APIs.
"""

import logging
from datetime import datetime, timedelta

from src.data.data_manager import ClimateDataManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

def collect_sample_datasets():
    """Collect sample datasets from all APIs for development and testing."""
    
    print("🌍 Climate Change Impact Predictor - Sample Data Collection")
    print("=" * 60)
    
    # Initialize data manager
    manager = ClimateDataManager()
    
    # Define collection parameters
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    locations = ["berlin", "houston"]  # Collect for 2 representative locations
    
    print(f"📅 Collection Period: {start_date} to {end_date}")
    print(f"📍 Locations: {', '.join([loc.title() for loc in locations])}")
    print()
    
    # Collect Air Quality Data
    print("🌬️ Collecting Air Quality Data (Open-Meteo)")
    print("-" * 40)
    for location in locations:
        try:
            print(f"  📍 {location.title()}... ", end="")
            data = manager.fetch_air_quality_data(
                location=location,
                start_date=start_date,
                end_date=end_date,
                save=True
            )
            records = len(data.get('hourly', {}).get('time', []))
            print(f"✅ {records} records")
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    print()
    
    # Collect Meteorological Data  
    print("🛰️ Collecting Meteorological Data (NASA POWER)")
    print("-" * 40)
    for location in locations:
        try:
            print(f"  📍 {location.title()}... ", end="")
            data = manager.fetch_meteorological_data(
                location=location,
                start_date=start_date,
                end_date=end_date,
                save=True
            )
            print("✅ Complete")
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    print()
    
    # Collect Climate Projections
    print("🌍 Collecting Climate Projections (World Bank CCKP)")
    print("-" * 40)
    scenarios = ["ssp245", "ssp585"]  # Moderate and high scenarios
    
    for scenario in scenarios:
        try:
            print(f"  📊 Scenario {scenario.upper()}... ", end="")
            data = manager.fetch_climate_projections(
                countries="all_countries",
                scenario=scenario,
                save=True
            )
            print("✅ Complete")
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    print()
    print("🎯 Sample Data Collection Complete!")
    print()
    print("📂 Data saved to:")
    print("   • data/raw/ - All raw API responses") 
    print("   • Organized by source and date range")
    print()
    print("🚀 Next Steps:")
    print("   1. Explore data in notebooks/")
    print("   2. Begin Day 2: Data processing pipeline")
    print("   3. Start feature engineering")

if __name__ == "__main__":
    collect_sample_datasets()