#!/usr/bin/env python3
"""
🌍 Simple Climate Prediction CLI - Fixed Version
tools/predict_climate_fixed.py

Usage: python tools/predict_climate_fixed.py "City Name"
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from src.models.climate_predictor import GlobalClimatePredictorFixed
except ImportError:
    print("❌ Could not import climate predictor")
    print("Make sure model files are in ./models/ directory")
    sys.exit(1)


def predict_single_city(city_name: str):
    """Generate prediction for a single city."""
    
    try:
        predictor = GlobalClimatePredictorFixed()
        forecast = predictor.predict_climate(city_name)
        
        if not forecast["success"]:
            print(f"❌ {forecast['error']}")
            return
        
        # Extract key information
        loc = forecast["location"]
        temps = forecast["forecasts"]["temperature_3day"]
        current = forecast["current_conditions"]
        uv = forecast["forecasts"]["uv_index_3day"][0]["uv_index"]
        precip = forecast["forecasts"]["precipitation_3day"][0]["probability"]
        
        # Print formatted output
        print(f"\n📍 {loc['city']}, {loc['country']} ({loc['coordinates']})")
        
        print(f"\n🌡️ Temperature Forecast (Next 3 Days):")
        for temp in temps:
            print(f"   Day {temp['day']}: {temp['min']:.1f}°C / {temp['max']:.1f}°C")
        
        print(f"\n💨 Air Quality: AQI {current['aqi_prediction']:.0f} ({current['aqi_category']})")
        print(f"🌧️ Precipitation: {precip:.0f}% chance today")
        print(f"☀️ UV Index: {uv:.1f} today")
        print(f"🔥 Heat Index: {current['heat_index']:.1f}°C ({current['heat_category']})")
        
        if forecast["health_alerts"]:
            print(f"\n⚠️ Health Alerts:")
            for alert in forecast["health_alerts"]:
                print(f"   {alert}")
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")


def main():
    """Main CLI function."""
    
    parser = argparse.ArgumentParser(
        description="🌍 Global Climate Prediction System (Fixed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/predict_climate_fixed.py "Lahore"
  python tools/predict_climate_fixed.py "London"
  python tools/predict_climate_fixed.py "New York"
        """
    )
    
    parser.add_argument(
        "city", 
        help="City name to predict climate for"
    )
    
    args = parser.parse_args()
    
    print("🌍 Global Climate Prediction System (Fixed)")
    print("=" * 45)
    
    predict_single_city(args.city)


if __name__ == "__main__":
    main()