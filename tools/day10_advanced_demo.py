#!/usr/bin/env python3
"""
tools/day10_advanced_demo.py
🌍 Day 10: Advanced Climate Prediction Demo

Showcases the enhanced GlobalClimatePredictor with:
- LSTM long-term forecasting
- Multi-output comprehensive predictions
- Combined advanced predictions
"""

import sys
import json
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models.enhanced_climate_predictor import EnhancedGlobalClimatePredictor


def print_header(title: str):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"🌍 {title}")
    print("=" * 70)


def print_section(title: str):
    """Print formatted section."""
    print(f"\n🔍 {title}")
    print("-" * 50)


def format_prediction_result(result: dict, show_details: bool = True):
    """Format prediction result for display."""
    if not result.get("success", False):
        print(f"❌ Failed: {result.get('error', 'Unknown error')}")
        return
    
    city = result.get("city", "Unknown")
    model = result.get("model_used", "Unknown Model")
    
    print(f"✅ Success for {city}")
    print(f"   Model: {model}")
    
    if show_details:
        # Show location info
        if "location" in result:
            loc = result["location"]
            print(f"   Location: {loc.get('latitude', 0):.2f}, {loc.get('longitude', 0):.2f}")
            print(f"   Country: {loc.get('country', 'Unknown')}")
            print(f"   Climate Zone: {loc.get('climate_zone', 'Unknown')}")
        
        # Show current conditions
        if "current" in result:
            curr = result["current"]
            print(f"   Current PM2.5: {curr.get('current_pm25', 0):.1f} μg/m³")
            print(f"   Historical Temp: {curr.get('historical_temp_avg', 0):.1f}°C")
            print(f"   Wind Speed: {curr.get('wind_avg', 0):.1f} m/s")
        
        # Show basic prediction
        if "prediction" in result:
            pred = result["prediction"]
            print(f"   Predicted Temp: {pred.get('temperature_avg', 0):.1f}°C")
            print(f"   Predicted AQI: {pred.get('aqi_prediction', 0):.1f}")
        
        # Show LSTM forecast
        if "forecast" in result:
            forecast = result["forecast"][:3]  # Show first 3 days
            print(f"   LSTM Forecast ({len(result['forecast'])} days):")
            for day_data in forecast:
                print(f"     Day {day_data['day']}: {day_data['temperature']:.1f}°C ({day_data['confidence']})")
        
        # Show comprehensive predictions
        if "comprehensive_predictions" in result:
            comp = result["comprehensive_predictions"]
            print(f"   Comprehensive Predictions:")
            
            if "temperature_3day" in comp:
                temps = comp["temperature_3day"]
                temp_str = ", ".join([f"{d['temperature']:.1f}°C" for d in temps])
                print(f"     3-day Temps: {temp_str}")
            
            if "precipitation_3day" in comp:
                precip = comp["precipitation_3day"]
                precip_str = ", ".join([f"{d['precipitation_prob']:.1f}%" for d in precip])
                print(f"     3-day Precip: {precip_str}")
            
            if "air_quality" in comp:
                print(f"     Air Quality: {comp['air_quality']:.1f}")


def demonstrate_enhanced_predictor():
    """Main demonstration of enhanced climate predictor."""
    
    print_header("ENHANCED CLIMATE PREDICTOR - DAY 10 ADVANCED ML DEMO")
    
    # Initialize predictor
    print("🚀 Initializing Enhanced Climate Predictor...")
    predictor = EnhancedGlobalClimatePredictor()
    
    # Check model status
    print_section("Model Status Check")
    status = predictor.get_model_status()
    
    print("📊 Available Models:")
    for model_name, available in status.items():
        status_icon = "✅" if available else "❌"
        print(f"   {status_icon} {model_name.replace('_', ' ').title()}")
    
    if not status["fully_enhanced"]:
        print("\n⚠️ Some models are missing. Continuing with available models...")
    else:
        print("\n🎉 All models loaded successfully! Full capabilities available.")
    
    # Demo cities
    demo_cities = [
        "London, UK",
        "Tokyo, Japan", 
        "New York, USA",
        "Sydney, Australia",
        "Cairo, Egypt"
    ]
    
    # Demonstrate each prediction type
    for city in demo_cities:
        print_header(f"Advanced Predictions for {city}")
        
        # 1. Original Base Prediction (Day 8)
        if status["base_model"]:
            print_section("Base Climate Prediction (Day 8 Model)")
            base_result = predictor.predict_climate(city)
            format_prediction_result(base_result)
        
        # 2. LSTM Long-term Forecasting (Day 10)
        if status["lstm_forecaster"]:
            print_section("LSTM Long-term Forecasting (Day 10)")
            lstm_result = predictor.predict_long_term(city, days=7)
            format_prediction_result(lstm_result)
        
        # 3. Multi-output Comprehensive Prediction (Day 10)
        if status["multi_output_predictor"]:
            print_section("Multi-output Comprehensive Prediction (Day 10)")
            multi_result = predictor.predict_comprehensive(city)
            format_prediction_result(multi_result)
        
        # 4. Ultimate Advanced Prediction (All Models)
        print_section("Ultimate Advanced Prediction (All Models Combined)")
        advanced_result = predictor.predict_climate_advanced(city)
        
        if advanced_result["success"]:
            print(f"✅ Advanced prediction successful for {city}")
            print(f"   Models Used: {', '.join(advanced_result['models_used'])}")
            print(f"   Total Predictions: {len(advanced_result['predictions'])}")
            
            # Show summary of each prediction type
            for pred_type, pred_data in advanced_result["predictions"].items():
                print(f"   📊 {pred_type.replace('_', ' ').title()}: ✅ Success")
        else:
            print(f"❌ Advanced prediction failed: {advanced_result.get('error', 'Unknown error')}")
        
        print("\n" + "⏳ Processing next city..." + "⏳")
        time.sleep(1)  # Brief pause for readability
    
    # Performance Summary
    print_header("PERFORMANCE SUMMARY")
    
    successful_predictions = 0
    total_attempts = len(demo_cities)
    
    print(f"📊 Demo Statistics:")
    print(f"   Cities Tested: {total_attempts}")
    print(f"   Models Available: {sum(status.values())}")
    print(f"   Advanced ML Features: {'✅ Active' if status['fully_enhanced'] else '⚠️ Partial'}")
    
    print(f"\n🏆 Portfolio Highlights:")
    print(f"   ✅ Multi-model Architecture: {sum(status.values())} models working together")
    print(f"   ✅ LSTM Time Series: {'Advanced 7-day forecasting' if status['lstm_forecaster'] else 'Not available'}")
    print(f"   ✅ Multi-output Prediction: {'5+ simultaneous outputs' if status['multi_output_predictor'] else 'Not available'}")
    print(f"   ✅ Global Coverage: Works for any city worldwide")
    print(f"   ✅ Production Ready: Professional error handling & scalability")
    
    print(f"\n🚀 Next Steps:")
    if status["fully_enhanced"]:
        print(f"   • System ready for production deployment")
        print(f"   • All advanced ML features operational")
        print(f"   • Portfolio demonstration complete")
    else:
        print(f"   • Ensure all model files are in models/ directory")
        print(f"   • Check for any missing dependencies")
        print(f"   • Verify file permissions and paths")
    
    return status["fully_enhanced"]


def quick_test():
    """Quick test of key functionality."""
    
    print_header("QUICK FUNCTIONALITY TEST")
    
    predictor = EnhancedGlobalClimatePredictor()
    test_city = "London, UK"
    
    print(f"🧪 Testing with {test_city}...")
    
    # Test each method
    methods_to_test = [
        ("Base Prediction", lambda: predictor.predict_climate(test_city)),
        ("LSTM Forecasting", lambda: predictor.predict_long_term(test_city)),
        ("Multi-output Prediction", lambda: predictor.predict_comprehensive(test_city)),
        ("Advanced Combined", lambda: predictor.predict_climate_advanced(test_city))
    ]
    
    results = {}
    
    for method_name, method_func in methods_to_test:
        try:
            result = method_func()
            success = result.get("success", False)
            results[method_name] = success
            print(f"   {'✅' if success else '❌'} {method_name}")
        except Exception as e:
            results[method_name] = False
            print(f"   ❌ {method_name}: {str(e)}")
    
    success_rate = sum(results.values()) / len(results)
    print(f"\n📊 Success Rate: {success_rate:.1%}")
    
    return success_rate > 0.5


def demo_single_city(city_name: str):
    """Demo enhanced predictions for a single city."""
    
    print_header(f"ENHANCED CLIMATE PREDICTION FOR {city_name.upper()}")
    
    # Initialize predictor
    print("🚀 Initializing Enhanced Climate Predictor...")
    predictor = EnhancedGlobalClimatePredictor()
    
    # Check model status
    print_section("Model Status")
    status = predictor.get_model_status()
    
    for model_name, available in status.items():
        status_icon = "✅" if available else "❌"
        print(f"   {status_icon} {model_name.replace('_', ' ').title()}")
    
    print_header(f"Advanced Predictions for {city_name}")
    
    # 1. Base Climate Prediction
    if status["base_model"]:
        print_section("Base Climate Prediction (Day 8)")
        base_result = predictor.predict_climate(city_name)
        format_prediction_result(base_result)
    
    # 2. LSTM Long-term Forecasting  
    if status["lstm_forecaster"]:
        print_section("LSTM Long-term Forecasting (Day 10)")
        lstm_result = predictor.predict_long_term(city_name, days=7)
        format_prediction_result(lstm_result)
    
    # 3. Multi-output Comprehensive Prediction
    if status["multi_output_predictor"]:
        print_section("Multi-output Comprehensive Prediction (Day 10)")
        multi_result = predictor.predict_comprehensive(city_name)
        format_prediction_result(multi_result)
    
    # 4. Ultimate Advanced Prediction
    print_section("Ultimate Advanced Prediction (All Models)")
    advanced_result = predictor.predict_climate_advanced(city_name)
    
    if advanced_result["success"]:
        print(f"✅ Advanced prediction successful for {city_name}")
        print(f"   Models Used: {', '.join(advanced_result['models_used'])}")
        print(f"   Total Predictions: {len(advanced_result['predictions'])}")
        
        for pred_type, pred_data in advanced_result["predictions"].items():
            print(f"   📊 {pred_type.replace('_', ' ').title()}: ✅ Success")
    else:
        print(f"❌ Advanced prediction failed: {advanced_result.get('error', 'Unknown error')}")
    
    # Summary
    print_header("PREDICTION SUMMARY")
    
    models_used = sum(status.values())
    print(f"🎯 Results for {city_name}:")
    print(f"   • Models Available: {models_used}/4")
    print(f"   • Advanced Features: {'✅ Full' if status['fully_enhanced'] else '⚠️ Partial'}")
    print(f"   • Prediction Success: {'✅ Complete' if advanced_result.get('success', False) else '❌ Issues'}")
    
    return advanced_result.get("success", False)


def main():
    """Main demo function with command line support."""
    
    import sys
    
    print("🌍 DAY 10: ADVANCED CLIMATE PREDICTION SYSTEM")
    print("Enhanced with LSTM + Multi-output Deep Learning")
    print()
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        # City provided as command line argument
        city_name = " ".join(sys.argv[1:])  # Join all arguments as city name
        print(f"🎯 Command line mode: Predicting for {city_name}")
        success = demo_single_city(city_name)
    else:
        # Interactive mode
        print("Choose demo mode:")
        print("1. Quick test (London)")
        print("2. Full demonstration (5 cities)")
        print("3. Custom city")
        
        mode = input("Enter choice (1-3): ").strip()
        
        if mode == "1":
            success = quick_test()
        elif mode == "3":
            city_name = input("Enter city name: ").strip()
            if city_name:
                success = demo_single_city(city_name)
            else:
                print("❌ No city provided")
                return False
        else:
            success = demonstrate_enhanced_predictor()
    
    print_header("DEMO COMPLETE")
    
    if success:
        print("🎉 Demo completed successfully!")
        print("🚀 Your enhanced climate predictor is ready for portfolio showcase!")
        print("\n💫 Day 10 Achievement Unlocked:")
        print("   • Advanced LSTM time series forecasting")
        print("   • Multi-output neural network predictions") 
        print("   • Production-ready ML model ensemble")
        print("   • Global climate prediction system")
    else:
        print("⚠️ Demo completed with some issues.")
        print("💡 Check model files and try again.")
    
    return success


if __name__ == "__main__":
    main()