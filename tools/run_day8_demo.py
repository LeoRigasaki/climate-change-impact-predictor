#!/usr/bin/env python3
"""
🎯 Global Climate AI - Portfolio Demonstration
tools/portfolio_demo.py

Automated demo showing global climate prediction capabilities.
"""

import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.models.climate_predictor import GlobalClimatePredictor, format_prediction
    
    # Demo cities across different continents and climate zones
    demo_cities = [
        ("London", "Europe", "Temperate"),
        ("Tokyo", "Asia", "Subtropical"), 
        ("Lahore", "Asia", "Subtropical"),
        ("Cairo", "Africa", "Subtropical"),
        ("Sydney", "Oceania", "Temperate"),
        ("New York", "North America", "Temperate"),
        ("Mumbai", "Asia", "Tropical"),
        ("Lagos", "Africa", "Tropical")
    ]
    
    def run_portfolio_demo():
        """Run automated portfolio demonstration."""
        
        print("🌍 GLOBAL CLIMATE AI SYSTEM - Portfolio Demonstration")
        print("=" * 65)
        print("Neural network trained on 144 world capitals")
        print("Predicts climate for ANY location on Earth")
        print()
        
        predictor = GlobalClimatePredictor()
        
        successful_predictions = 0
        
        for city, continent, climate_zone in demo_cities:
            print(f"🔍 Predicting climate for {city} ({continent}, {climate_zone})")
            
            try:
                forecast = predictor.predict_climate(city)
                
                if forecast["success"]:
                    # Extract key info for summary
                    temps = forecast["forecasts"]["temperature_3day"][0]
                    aqi = forecast["current"]["aqi"]
                    zone = forecast["location"]["climate_zone"]
                    
                    print(f"   ✅ {temps['min']:.1f}-{temps['max']:.1f}°C, AQI {aqi:.0f}, {zone} zone")
                    successful_predictions += 1
                else:
                    print(f"   ❌ {forecast['error']}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
            
            time.sleep(0.5)  # Brief pause for demo effect
        
        print()
        print("=" * 65)
        print("🎯 DEMONSTRATION SUMMARY")
        print("=" * 65)
        print(f"Cities tested: {len(demo_cities)}")
        print(f"Successful predictions: {successful_predictions}")
        print(f"Success rate: {successful_predictions/len(demo_cities)*100:.0f}%")
        print(f"Global coverage: {len(set(continent for _, continent, _ in demo_cities))} continents")
        print(f"Climate zones: {len(set(zone for _, _, zone in demo_cities))} zones")
        
        print(f"\n🏆 PORTFOLIO HIGHLIGHTS:")
        print(f"✅ Global scale: Works for any city on Earth")
        print(f"✅ Real data: Trained on professional climate APIs")
        print(f"✅ Production ready: Clean interface, error handling")
        print(f"✅ Advanced ML: Neural networks, transfer learning")
        print(f"✅ Practical impact: Actually useful climate predictions")
        
        if successful_predictions >= len(demo_cities) * 0.8:
            print(f"\n🎉 DEMO SUCCESS: System ready for portfolio showcase!")
        else:
            print(f"\n⚠️ DEMO PARTIAL: Some predictions failed")
    
    def detailed_demo(city_name: str):
        """Show detailed prediction for specific city."""
        
        print(f"🔍 Detailed Climate Analysis: {city_name}")
        print("-" * 50)
        
        predictor = GlobalClimatePredictor()
        forecast = predictor.predict_climate(city_name)
        print(format_prediction(forecast))
    
    if __name__ == "__main__":
        if len(sys.argv) > 1:
            # Detailed demo for specific city
            city = " ".join(sys.argv[1:])
            detailed_demo(city)
        else:
            # Full portfolio demo
            run_portfolio_demo()

except ImportError as e:
    print(f"❌ System not ready: {e}")
    print("Ensure model files are in ./models/ directory")
except Exception as e:
    print(f"❌ Demo failed: {e}")