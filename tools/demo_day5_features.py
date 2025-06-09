#!/usr/bin/env python3
"""
🌍 Day 5 Universal Feature Engineering Demo
tools/demo_day5_features.py

Interactive demonstration of the universal feature engineering system.
Shows how the system creates intelligent, location-independent climate features
that work anywhere on Earth with global context and regional adaptation.
"""

import sys
from pathlib import Path
import logging
from typing import Dict
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.universal_engine import UniversalFeatureEngine
from src.features.feature_library import FeatureLibrary
from src.core.data_manager import ClimateDataManager
from src.core.location_service import LocationService

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

logger = logging.getLogger(__name__)

class Day5Demo:
    """
    🌍 Interactive Day 5 Universal Feature Engineering Demo
    
    Demonstrates the power of universal climate feature engineering:
    - Location-independent climate indicators  
    - Regional adaptation and context
    - Global comparative metrics
    - Hemisphere-aware seasonal processing
    - Comprehensive feature documentation
    """
    
    def __init__(self):
        """Initialize the demo system."""
        print("🌍 Initializing Day 5 Universal Feature Engineering Demo...")
        print("=" * 65)
        
        self.engine = UniversalFeatureEngine()
        self.library = FeatureLibrary()
        self.data_manager = ClimateDataManager()
        self.location_service = LocationService()
        
        print("✅ Universal Feature Engine loaded")
        print("✅ Feature Library initialized")
        print("✅ Data Manager ready")
        print("✅ Location Service active")
        print()
    
    async def run_interactive_demo(self):
        """Run interactive demonstration."""
        print("🎯 Day 5 Universal Feature Engineering - Interactive Demo")
        print("=" * 60)
        print()
        
        # Demo 1: Universal Features for Different Locations
        await self.demo_global_locations()
        
        # Demo 2: Feature Library and Documentation
        await self.demo_feature_library()
        
        # Demo 3: Regional Adaptation Intelligence
        await self.demo_regional_adaptation()
        
        # Demo 4: Seasonal Awareness
        await self.demo_seasonal_awareness()
        
        # Demo 5: Real-time Feature Engineering
        await self.demo_real_time_processing()
        
        print("\n🎉 Day 5 Demo Complete!")
        print("✨ Universal Feature Engineering System Ready for Production!")
    
    async def demo_global_locations(self):
        """Demonstrate universal features working globally."""
        print("🌍 DEMO 1: Universal Features for Global Locations")
        print("=" * 55)
        
        # Test locations across different climate zones and hemispheres
        demo_locations = [
            {"name": "Reykjavik", "country": "Iceland", "latitude": 64.13, "longitude": -21.95},
            {"name": "Singapore", "country": "Singapore", "latitude": 1.35, "longitude": 103.82},
            {"name": "Cairo", "country": "Egypt", "latitude": 30.03, "longitude": 31.23},
            {"name": "Sydney", "country": "Australia", "latitude": -33.87, "longitude": 151.21}
        ]
        
        for i, location in enumerate(demo_locations, 1):
            print(f"\n📍 Location {i}: {location['name']}, {location['country']}")
            print(f"   Coordinates: {location['latitude']:.2f}°, {location['longitude']:.2f}°")
            
            # Create sample climate data for this location
            sample_data = self._create_location_appropriate_data(location)
            
            # Generate universal features
            start_time = time.time()
            enhanced_data = self.engine.engineer_features(sample_data, location)
            processing_time = time.time() - start_time
            
            # Show results
            original_features = sample_data.shape[1]
            final_features = enhanced_data.shape[1]
            features_added = final_features - original_features
            
            print(f"   ⚡ Processing time: {processing_time:.2f} seconds")
            print(f"   📈 Features: {original_features} → {final_features} (+{features_added})")
            
            # Show key universal indicators
            self._show_key_indicators(enhanced_data, location)
        
        input("\n🔍 Press Enter to continue to Feature Library demo...")
    
    async def demo_feature_library(self):
        """Demonstrate feature library and documentation."""
        print("\n📚 DEMO 2: Feature Library and Documentation")
        print("=" * 48)
        
        # Show library statistics
        all_features = self.library.get_all_features()
        categories = self.library.feature_categories
        
        print(f"📊 Feature Library Statistics:")
        print(f"   • Total documented features: {len(all_features)}")
        print(f"   • Feature categories: {len(categories)}")
        print(f"   • High importance features: {len(self.library.get_high_importance_features())}")
        
        # Demonstrate search functionality
        print(f"\n🔍 Feature Search Examples:")
        search_terms = ["stress", "global", "seasonal"]
        
        for term in search_terms:
            results = self.library.search_features(term)
            print(f"   • '{term}': {len(results)} features found")
            for feature in results[:2]:  # Show first 2
                description = self.library.feature_catalog[feature]["description"]
                print(f"     - {feature}: {description[:60]}...")
        
        # Show feature explanation example
        print(f"\n📖 Feature Explanation Example:")
        feature_guide = self.library.get_feature_usage_guide("climate_stress_index")
        print(f"   Feature: climate_stress_index")
        print(f"   Description: {feature_guide['description']}")
        print(f"   Scale: {feature_guide['scale']}")
        print(f"   Good range: {feature_guide['good_range']}")
        print(f"   Use cases: {', '.join(feature_guide['use_cases'][:2])}")
        
        input("\n🔍 Press Enter to continue to Regional Adaptation demo...")
    
    async def demo_regional_adaptation(self):
        """Demonstrate regional adaptation intelligence."""
        print("\n🗺️ DEMO 3: Regional Adaptation Intelligence")
        print("=" * 45)
        
        # Compare same weather conditions in different climate zones
        print("🌡️ Same Temperature, Different Contexts:")
        print("   Showing how 25°C is interpreted differently by climate zone...")
        
        test_locations = [
            {"name": "Tropical City", "latitude": 0, "longitude": 0, "climate_zone": "tropical"},
            {"name": "Arctic City", "latitude": 70, "longitude": 0, "climate_zone": "polar"},
            {"name": "Desert City", "latitude": 30, "longitude": 0, "climate_zone": "arid"}
        ]
        
        # Create identical weather data
        base_data = pd.DataFrame({
            "temperature_2m": [25.0] * 30,  # Constant 25°C
            "relative_humidity": [60.0] * 30,
            "precipitation": [1.0] * 30,
            "wind_speed_2m": [3.0] * 30,
            "pm2_5": [20.0] * 30
        }, index=pd.date_range("2024-01-01", periods=30))
        
        print()
        for location in test_locations:
            enhanced_data = self.engine.engineer_features(base_data, location)
            
            # Show regional interpretation
            climate_stress = enhanced_data['climate_stress_index'].mean()
            comfort = enhanced_data['human_comfort_index'].mean()
            climate_zone = enhanced_data['climate_zone'].iloc[0]
            
            print(f"📍 {location['name']} ({climate_zone} zone):")
            print(f"   Climate Stress Index: {climate_stress:.1f}/100")
            print(f"   Human Comfort Index: {comfort:.1f}/100")
            
            if 'temp_deviation_from_regional_norm' in enhanced_data.columns:
                temp_deviation = enhanced_data['temp_deviation_from_regional_norm'].mean()
                print(f"   Temperature vs Regional Normal: {temp_deviation:+.1f}°C")
            
            print()
        
        input("🔍 Press Enter to continue to Seasonal Awareness demo...")
    
    async def demo_seasonal_awareness(self):
        """Demonstrate hemisphere-aware seasonal processing."""
        print("\n📅 DEMO 4: Hemisphere-Aware Seasonal Processing")
        print("=" * 50)
        
        print("🌍 Comparing Seasons: July in Different Hemispheres")
        print("   July is summer in Northern Hemisphere, winter in Southern...")
        
        # Create July data
        july_dates = pd.date_range("2024-07-01", "2024-07-31")
        july_data = pd.DataFrame({
            "temperature_2m": np.random.normal(20, 5, len(july_dates)),
            "precipitation": np.random.exponential(2, len(july_dates)),
            "relative_humidity": np.random.normal(65, 15, len(july_dates))
        }, index=july_dates)
        
        hemisphere_locations = [
            {"name": "Berlin", "country": "Germany", "latitude": 52.52, "longitude": 13.41},
            {"name": "Sydney", "country": "Australia", "latitude": -33.87, "longitude": 151.21}
        ]
        
        print()
        for location in hemisphere_locations:
            enhanced_data = self.engine.engineer_features(july_data, location)
            
            hemisphere = "Northern" if location["latitude"] >= 0 else "Southern"
            season = enhanced_data['hemisphere_season'].iloc[0] if 'hemisphere_season' in enhanced_data.columns else "Unknown"
            
            print(f"📍 {location['name']} ({hemisphere} Hemisphere):")
            print(f"   July Season: {season}")
            
            if 'temperature_2m_seasonal_deviation' in enhanced_data.columns:
                seasonal_deviation = enhanced_data['temperature_2m_seasonal_deviation'].mean()
                print(f"   Temperature vs Seasonal Normal: {seasonal_deviation:+.1f}°C")
            
            if 'is_heat_risk_season' in enhanced_data.columns:
                heat_risk = enhanced_data['is_heat_risk_season'].iloc[0]
                print(f"   Heat Risk Season: {'Yes' if heat_risk else 'No'}")
            
            print()
        
        input("🔍 Press Enter to continue to Real-time Processing demo...")
    
    async def demo_real_time_processing(self):
        """Demonstrate real-time feature engineering."""
        print("\n⚡ DEMO 5: Real-time Feature Engineering Performance")
        print("=" * 55)
        
        print("🔄 Processing Multiple Locations in Real-time...")
        
        # Simulate real-time processing for multiple locations
        locations = [
            {"name": "New York", "latitude": 40.71, "longitude": -74.01},
            {"name": "London", "latitude": 51.51, "longitude": -0.13},
            {"name": "Tokyo", "latitude": 35.68, "longitude": 139.69},
            {"name": "São Paulo", "latitude": -23.55, "longitude": -46.64},
            {"name": "Mumbai", "latitude": 19.08, "longitude": 72.88}
        ]
        
        # Create sample data for batch processing
        sample_data = self._create_sample_data()
        data_batch = {f"location_{i}": (sample_data, loc) for i, loc in enumerate(locations)}
        
        print(f"📊 Processing {len(locations)} locations simultaneously...")
        
        # Time the batch processing
        start_time = time.time()
        batch_results = self.engine.engineer_batch_features(data_batch)
        processing_time = time.time() - start_time
        
        # Show performance metrics
        print(f"   ⚡ Total processing time: {processing_time:.2f} seconds")
        print(f"   📈 Average time per location: {processing_time/len(locations):.2f} seconds")
        print(f"   ✅ Success rate: {len(batch_results)}/{len(locations)} locations")
        
        # Show sample features from one location
        if batch_results:
            sample_key = list(batch_results.keys())[0]
            sample_result = batch_results[sample_key]
            print(f"\n📋 Sample Features Generated (for {locations[0]['name']}):")
            
            # Show key universal features
            key_features = [
                "climate_stress_index", "human_comfort_index", "weather_extremity_index",
                "temperature_2m_global_percentile", "hemisphere_season"
            ]
            
            for feature in key_features:
                if feature in sample_result.columns:
                    value = sample_result[feature].iloc[0]
                    if isinstance(value, (int, float)):
                        print(f"   • {feature}: {value:.1f}")
                    else:
                        print(f"   • {feature}: {value}")
        
        # Show engine statistics
        print(f"\n📊 Engine Performance Statistics:")
        engine_report = self.engine.get_processing_report()
        stats = engine_report["processing_stats"]
        
        print(f"   • Locations processed: {stats['locations_processed']}")
        print(f"   • Features created: {stats['features_created']}")
        print(f"   • Average processing time: {stats['avg_processing_time']:.2f} seconds")
        print(f"   • Regional adaptations: {stats['regional_adaptations']}")
        print(f"   • Global comparisons: {stats['global_comparisons']}")
        
        print(f"\n✨ Universal Feature Engineering System Ready for Production!")
    
    def _create_location_appropriate_data(self, location: Dict) -> pd.DataFrame:
        """Create climate data appropriate for location's climate."""
        dates = pd.date_range("2024-01-01", periods=90, freq="D")
        
        # Adjust data based on latitude (rough climate approximation)
        latitude = abs(location["latitude"])
        
        if latitude > 60:  # Polar
            temp_base = -5
            temp_var = 15
            humidity = 75
        elif latitude > 40:  # Temperate
            temp_base = 12
            temp_var = 12
            humidity = 65
        elif latitude > 20:  # Subtropical
            temp_base = 22
            temp_var = 8
            humidity = 55
        else:  # Tropical
            temp_base = 27
            temp_var = 4
            humidity = 80
        
        return pd.DataFrame({
            "temperature_2m": np.random.normal(temp_base, temp_var, 90),
            "precipitation": np.random.exponential(2, 90),
            "relative_humidity": np.clip(np.random.normal(humidity, 15, 90), 0, 100),
            "wind_speed_2m": np.random.exponential(3, 90),
            "pm2_5": np.random.exponential(15, 90)
        }, index=dates)
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create generic sample climate data."""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        
        return pd.DataFrame({
            "temperature_2m": np.random.normal(15, 10, 30),
            "precipitation": np.random.exponential(2, 30),
            "relative_humidity": np.clip(np.random.normal(60, 20, 30), 0, 100),
            "wind_speed_2m": np.random.exponential(3, 30),
            "pm2_5": np.random.exponential(15, 30)
        }, index=dates)
    
    def _show_key_indicators(self, data: pd.DataFrame, location: Dict):
        """Show key universal indicators for a location."""
        print("   🎯 Key Universal Indicators:")
        
        # Climate Stress Index
        if "climate_stress_index" in data.columns:
            stress = data["climate_stress_index"].mean()
            stress_level = "Low" if stress < 30 else "Moderate" if stress < 70 else "High"
            print(f"      • Climate Stress: {stress:.1f}/100 ({stress_level})")
        
        # Human Comfort Index
        if "human_comfort_index" in data.columns:
            comfort = data["human_comfort_index"].mean()
            comfort_level = "Excellent" if comfort > 80 else "Good" if comfort > 60 else "Poor"
            print(f"      • Human Comfort: {comfort:.1f}/100 ({comfort_level})")
        
        # Global Temperature Percentile
        if "temperature_2m_global_percentile" in data.columns:
            global_percentile = data["temperature_2m_global_percentile"].mean()
            print(f"      • Global Temperature Percentile: {global_percentile:.1f}%")
        
        # Regional Context
        if "climate_zone" in data.columns:
            climate_zone = data["climate_zone"].iloc[0]
            print(f"      • Climate Zone: {climate_zone.title()}")
        
        # Seasonal Context
        if "hemisphere_season" in data.columns:
            season = data["hemisphere_season"].iloc[0]
            hemisphere = "Northern" if location["latitude"] >= 0 else "Southern"
            print(f"      • Current Season: {season} ({hemisphere} Hemisphere)")


async def main():
    """Main demo function."""
    try:
        demo = Day5Demo()
        await demo.run_interactive_demo()
        
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user. Thanks for exploring Day 5!")
    except Exception as e:
        print(f"\n❌ Demo encountered an error: {e}")
        logger.error(f"Demo failed: {e}")


if __name__ == "__main__":
    print("🌍 Day 5: Universal Feature Engineering Demo")
    print("=" * 50)
    print("This demo showcases the universal climate feature engineering system")
    print("that creates intelligent, location-independent features for any location on Earth.")
    print()
    print("🎯 Demo Features:")
    print("   • Universal climate indicators (stress, comfort, extremity)")
    print("   • Regional adaptation and local context")
    print("   • Global comparative metrics and rankings")
    print("   • Hemisphere-aware seasonal processing")
    print("   • Comprehensive feature documentation")
    print("   • Real-time performance optimization")
    print()
    
    import asyncio
    asyncio.run(main())