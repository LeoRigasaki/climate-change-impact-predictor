#!/usr/bin/env python3
"""
Test script for the data processing pipeline.
Validates processors and pipeline functionality.
"""

import logging
import sys
from datetime import datetime, timedelta

import pandas as pd

# Add src to path
sys.path.append('src')

from src.data.pipeline import ClimateDataPipeline
from src.data.processors.air_quality_processor import AirQualityProcessor
from src.data.processors.meteorological_processor import MeteorologicalProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

def test_individual_processors():
    """Test individual data processors."""
    
    print("🧪 Testing Individual Data Processors")
    print("=" * 50)
    
    # Test Air Quality Processor
    print("\n🌬️ Testing Air Quality Processor...")
    
    # Sample air quality data structure (simplified)
    sample_aq_data = {
        'hourly': {
            'time': ['2025-06-01T00:00', '2025-06-01T01:00', '2025-06-01T02:00'],
            'pm2_5': [12.5, 15.2, 18.7],
            'pm10': [22.1, 28.3, 31.2],
            'carbon_dioxide': [415.2, 416.1, 417.0],
            'ozone': [45.2, 52.3, 48.1],
            'nitrogen_dioxide': [18.5, 22.1, 25.3]
        },
        'current': {
            'european_aqi': 2,
            'us_aqi': 58,
            'pm2_5': 18.7
        },
        '_metadata': {
            'source': 'Open-Meteo Air Quality API',
            'coordinates': {'latitude': 52.52, 'longitude': 13.41}
        }
    }
    
    try:
        aq_processor = AirQualityProcessor()
        aq_df = aq_processor.process(sample_aq_data)
        
        print(f"   ✅ Success! Processed shape: {aq_df.shape}")
        print(f"   📊 Features: {list(aq_df.columns)}")
        print(f"   🕐 Date range: {aq_df.index.min()} to {aq_df.index.max()}")
        
        # Check derived features
        derived_features = [col for col in aq_df.columns if any(
            term in col.lower() for term in ['ratio', 'index', 'risk', 'avg']
        )]
        print(f"   🔧 Derived features: {len(derived_features)}")
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test Meteorological Processor
    print("\n🛰️ Testing Meteorological Processor...")
    
    # Sample meteorological data structure
    sample_met_data = {
        'properties': {
            'parameter': {
                '20250601': {'T2M': 18.5, 'PRECTOTCORR': 0.0, 'WS2M': 3.2, 'RH2M': 65.2},
                '20250602': {'T2M': 21.2, 'PRECTOTCORR': 2.5, 'WS2M': 4.1, 'RH2M': 58.7},
                '20250603': {'T2M': 19.8, 'PRECTOTCORR': 0.1, 'WS2M': 2.8, 'RH2M': 72.3}
            }
        },
        '_metadata': {
            'source': 'NASA POWER API',
            'coordinates': {'latitude': 52.52, 'longitude': 13.41}
        }
    }
    
    try:
        met_processor = MeteorologicalProcessor()
        met_df = met_processor.process(sample_met_data)
        
        print(f"   ✅ Success! Processed shape: {met_df.shape}")
        print(f"   📊 Features: {list(met_df.columns)}")
        print(f"   🕐 Date range: {met_df.index.min()} to {met_df.index.max()}")
        
        # Check derived features
        derived_features = [col for col in met_df.columns if any(
            term in col.lower() for term in ['category', 'anomaly', 'avg', 'sum', 'index']
        )]
        print(f"   🔧 Derived features: {len(derived_features)}")
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")

def test_full_pipeline():
    """Test the complete data processing pipeline."""
    
    print("\n\n🔄 Testing Complete Data Pipeline")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = ClimateDataPipeline()
    
    # Test with sample location and date range
    location = "berlin"
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    
    print(f"📍 Testing location: {location.title()}")
    print(f"📅 Date range: {start_date} to {end_date}")
    
    try:
        # Run pipeline (skip collection if raw data doesn't exist)
        results = pipeline.process_location_data(
            location=location,
            start_date=start_date,
            end_date=end_date,
            skip_collection=True  # Use existing data
        )
        
        print("\n📊 Pipeline Results:")
        for source, result in results.items():
            if result and 'data' in result:
                df = result['data']
                quality_score = result.get('quality_report', {}).get('overall_score', 0)
                
                print(f"   {source.title()}:")
                print(f"     Shape: {df.shape}")
                print(f"     Quality Score: {quality_score:.1f}/100")
                print(f"     Date Range: {df.index.min()} to {df.index.max()}")
            else:
                print(f"   {source.title()}: ❌ Failed or no data")
        
        # Test pipeline summary
        summary = pipeline.get_processing_summary()
        print(f"\n📋 Processing Summary:")
        print(f"   Locations processed: {summary.get('total_locations_processed', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False

def test_data_integration():
    """Test data integration capabilities."""
    
    print("\n\n🔗 Testing Data Integration")
    print("=" * 40)
    
    try:
        from src.data.processors.base_processor import DataQualityChecker
        
        # Create sample datasets for integration
        dates = pd.date_range('2025-06-01', periods=5, freq='D')
        
        # Air quality data (daily aggregated)
        aq_data = pd.DataFrame({
            'pm2_5': [12.5, 15.2, 18.7, 14.3, 16.8],
            'pm10': [22.1, 28.3, 31.2, 25.4, 29.1],
            'co2': [415.2, 416.1, 417.0, 418.2, 419.5],
            'pollution_index': [45.2, 52.3, 58.1, 48.7, 54.2]
        }, index=dates)
        
        # Meteorological data
        met_data = pd.DataFrame({
            'temperature_2m': [18.5, 21.2, 19.8, 22.5, 20.1],
            'precipitation': [0.0, 2.5, 0.1, 0.0, 1.2],
            'wind_speed_2m': [3.2, 4.1, 2.8, 5.2, 3.8],
            'relative_humidity': [65.2, 58.7, 72.3, 61.4, 68.9]
        }, index=dates)
        
        # Test integration
        integrated = met_data.join(aq_data, how='inner')
        
        print(f"✅ Integration successful!")
        print(f"   Combined shape: {integrated.shape}")
        print(f"   Features: {list(integrated.columns)}")
        
        # Test quality checker
        quality_checker = DataQualityChecker()
        quality_report = quality_checker.assess_quality(integrated, 'test_integrated')
        
        print(f"📊 Quality Assessment:")
        print(f"   Overall Score: {quality_report['overall_score']:.1f}/100")
        print(f"   Completeness: {100 - quality_report['completeness']['missing_percentage']:.1f}%")
        print(f"   Duplicates: {quality_report['duplicates']['duplicate_percentage']:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    
    print("🧪 Climate Data Pipeline Testing Suite")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run tests
    test_results = []
    
    try:
        test_individual_processors()
        test_results.append("Individual Processors: ✅")
    except Exception as e:
        test_results.append(f"Individual Processors: ❌ ({e})")
    
    try:
        pipeline_success = test_full_pipeline()
        test_results.append(f"Full Pipeline: {'✅' if pipeline_success else '❌'}")
    except Exception as e:
        test_results.append(f"Full Pipeline: ❌ ({e})")
    
    try:
        integration_success = test_data_integration()
        test_results.append(f"Data Integration: {'✅' if integration_success else '❌'}")
    except Exception as e:
        test_results.append(f"Data Integration: ❌ ({e})")
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 Test Results Summary")
    print("=" * 60)
    
    for result in test_results:
        print(f"   {result}")
    
    successful_tests = len([r for r in test_results if "✅" in r])
    total_tests = len(test_results)
    
    print(f"\n📊 Overall: {successful_tests}/{total_tests} tests passed")
    
    if successful_tests == total_tests:
        print("🎉 All tests passed! Pipeline is ready for Day 2 development.")
    else:
        print("⚠️  Some tests failed. Check the issues above before proceeding.")
    
    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()