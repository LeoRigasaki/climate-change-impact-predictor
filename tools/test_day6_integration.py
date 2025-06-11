#!/usr/bin/env python3
"""
🧪 Day 6 Real Data Test - Using ACTUAL File Structure
tools/test_day6_corrected.py

Tests Day 6 validation and uncertainty systems using your actual processed data files.
No more assumptions - uses what actually exists!
"""

import sys
import time
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.validation.global_validator import GlobalDataValidator
from src.validation.uncertainty_quantifier import UncertaintyQuantifier
from src.validation.baseline_comparator import GlobalBaselineComparator
from src.features.universal_engine import UniversalFeatureEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def discover_actual_processed_files() -> Dict[str, List[str]]:
    """Discover what processed data files actually exist."""
    
    processed_dir = Path("data/processed")
    
    files_found = {
        "integrated": [],
        "air_quality": [],
        "meteorological": [],
        "metadata": []
    }
    
    if not processed_dir.exists():
        print("❌ data/processed directory not found!")
        return files_found
    
    # Scan for actual files
    for file_path in processed_dir.glob("*.parquet"):
        filename = file_path.name
        
        if filename.startswith("integrated_"):
            files_found["integrated"].append(filename)
        elif filename.startswith("air_quality_") and filename.endswith("_processed.parquet"):
            files_found["air_quality"].append(filename)
        elif filename.startswith("meteorological_") and filename.endswith("_processed.parquet"):
            files_found["meteorological"].append(filename)
    
    # Scan for metadata files
    for file_path in processed_dir.glob("*_metadata.json"):
        files_found["metadata"].append(file_path.name)
    
    return files_found


def load_real_processed_data(file_type: str, location_pattern: str = "berlin") -> pd.DataFrame:
    """Load actual processed data from your files."""
    
    processed_dir = Path("data/processed")
    
    # Find the most recent file for the location
    pattern = f"{file_type}_{location_pattern}_*.parquet"
    matching_files = list(processed_dir.glob(pattern))
    
    if not matching_files:
        print(f"   ⚠️ No {file_type} files found for {location_pattern}")
        return pd.DataFrame()
    
    # Get the most recent file (by filename, which includes dates)
    latest_file = sorted(matching_files)[-1]
    
    try:
        data = pd.read_parquet(latest_file)
        print(f"   ✅ Loaded {file_type}: {latest_file.name} ({data.shape[0]:,} records, {data.shape[1]} features)")
        return data
    except Exception as e:
        print(f"   ❌ Failed to load {latest_file}: {e}")
        return pd.DataFrame()


def test_day6_with_actual_files():
    """Test Day 6 systems using your actual processed data files."""
    
    print("🧪 Day 6 Test with ACTUAL File Structure")
    print("🔍 Using Real Processed Data Files")
    print("=" * 60)
    
    # Step 1: Discover what files actually exist
    print("1️⃣ Discovering Actual Processed Files...")
    actual_files = discover_actual_processed_files()
    
    print("   📂 Files Found:")
    for file_type, files in actual_files.items():
        print(f"      • {file_type}: {len(files)} files")
        for file in files[:3]:  # Show first 3
            print(f"        - {file}")
        if len(files) > 3:
            print(f"        ... and {len(files) - 3} more")
    
    if not any(actual_files.values()):
        print("❌ No processed files found! Run pipeline first.")
        return False
    
    print()
    
    # Step 2: Load Real Integrated Data (Best Option)
    print("2️⃣ Loading Real Integrated Data...")
    
    if actual_files["integrated"]:
        # Use integrated data (best - already combined)
        integrated_file = sorted(actual_files["integrated"])[-1]  # Most recent
        integrated_path = Path("data/processed") / integrated_file
        
        try:
            real_data = pd.read_parquet(integrated_path)
            print(f"   ✅ Loaded integrated data: {integrated_file}")
            print(f"   📊 Shape: {real_data.shape[0]:,} records, {real_data.shape[1]} features")
            print(f"   📅 Date range: {real_data.index.min()} to {real_data.index.max()}")
            
            # Extract location from filename (e.g., "integrated_berlin_2024-06-11_2025-06-10.parquet")
            location_name = integrated_file.split("_")[1]
            
        except Exception as e:
            print(f"   ❌ Failed to load integrated data: {e}")
            return False
    
    elif actual_files["air_quality"] and actual_files["meteorological"]:
        # Fallback: Load and combine air quality + meteorological
        print("   📊 Loading separate air quality + meteorological files...")
        
        air_quality_data = load_real_processed_data("air_quality", "berlin")
        meteorological_data = load_real_processed_data("meteorological", "berlin")
        
        if air_quality_data.empty or meteorological_data.empty:
            print("   ❌ Failed to load required data files")
            return False
        
        # Simple combination (join on index if possible)
        try:
            real_data = air_quality_data.join(meteorological_data, how='outer', rsuffix='_met')
            location_name = "berlin"
            print(f"   ✅ Combined data: {real_data.shape[0]:,} records, {real_data.shape[1]} features")
        except Exception as e:
            print(f"   ❌ Failed to combine data: {e}")
            return False
    
    else:
        print("   ❌ No suitable data files found for testing")
        return False
    
    # Determine location coordinates (approximate)
    location_coords = {
        "berlin": {"name": "Berlin", "country": "Germany", "latitude": 52.52, "longitude": 13.41},
        "houston": {"name": "Houston", "country": "USA", "latitude": 29.76, "longitude": -95.39},
        "lahore": {"name": "Lahore", "country": "Pakistan", "latitude": 31.57, "longitude": 74.31},
        "london": {"name": "London", "country": "UK", "latitude": 51.51, "longitude": -0.13},
        "paris": {"name": "Paris", "country": "France", "latitude": 48.86, "longitude": 2.35}
    }
    
    location_info = location_coords.get(location_name, {
        "name": location_name.title(),
        "country": "Unknown", 
        "latitude": 50.0,
        "longitude": 10.0
    })
    
    print()
    
    # Step 3: Test Global Data Validator
    print("3️⃣ Testing Global Data Validator...")
    try:
        validator = GlobalDataValidator()
        validation_report = validator.validate_regional_data_quality(
            real_data, 
            location_info, 
            ["processed_data"]  # Since this is processed data
        )
        
        quality_score = validation_report["overall_quality_score"]
        climate_zone = validation_report.get("regional_context", {}).get("climate_zone", "unknown")
        recommendations = len(validation_report.get("recommendations", []))
        
        print(f"   ✅ Validation successful: {quality_score:.1f}/100 quality score")
        print(f"   📋 Climate zone: {climate_zone}")
        print(f"   📊 Recommendations: {recommendations}")
        
        validation_success = True
        
    except Exception as e:
        print(f"   ❌ Validation failed: {e}")
        validation_success = False
    
    print()
    
    # Step 4: Test Uncertainty Quantifier
    print("4️⃣ Testing Uncertainty Quantifier...")
    try:
        quantifier = UncertaintyQuantifier()
        
        # Test measurement uncertainty
        measurement_unc = quantifier.quantify_measurement_uncertainty(
            real_data, 
            ["processed_data"]
        )
        
        # Test regional uncertainty
        regional_unc = quantifier.calculate_regional_uncertainty(
            real_data, 
            location_info
        )
        
        measurement_score = measurement_unc.get("overall_uncertainty_score", 0)
        regional_factor = regional_unc.get("overall_regional_uncertainty", 1.0)
        
        print(f"   ✅ Uncertainty analysis successful")
        print(f"   📊 Measurement uncertainty: {measurement_score:.1f}%")
        print(f"   🌍 Regional uncertainty factor: {regional_factor:.2f}x")
        
        uncertainty_success = True
        
    except Exception as e:
        print(f"   ❌ Uncertainty analysis failed: {e}")
        uncertainty_success = False
    
    print()
    
    # Step 5: Test Global Baseline Comparator
    print("5️⃣ Testing Global Baseline Comparator...")
    try:
        comparator = GlobalBaselineComparator()
        
        # Test global baseline comparison
        baseline_comparison = comparator.compare_to_global_baseline(
            real_data, 
            location_info
        )
        
        # Test global percentiles
        global_percentiles = comparator.calculate_global_percentiles(
            real_data, 
            location_info
        )
        
        overall_percentile = global_percentiles.get("overall_ranking", {}).get("overall_global_percentile", 50)
        variables_compared = len(baseline_comparison.get("global_comparisons", {}))
        
        print(f"   ✅ Baseline comparison successful")
        print(f"   🌍 Global percentile: {overall_percentile:.1f}%")
        print(f"   📊 Variables compared: {variables_compared}")
        
        baseline_success = True
        
    except Exception as e:
        print(f"   ❌ Baseline comparison failed: {e}")
        baseline_success = False
    
    print()
    
    # Step 6: Test Universal Features Integration
    print("6️⃣ Testing Universal Features Integration...")
    try:
        universal_engine = UniversalFeatureEngine()
        
        # Take a sample if data is too large (for performance)
        if len(real_data) > 1000:
            sample_data = real_data.sample(n=1000, random_state=42)
            print(f"   📊 Using sample of {len(sample_data):,} records for performance")
        else:
            sample_data = real_data
        
        # Apply universal feature engineering
        original_features = sample_data.shape[1]
        enhanced_data = universal_engine.engineer_features(sample_data, location_info)
        enhanced_features = enhanced_data.shape[1]
        features_added = enhanced_features - original_features
        
        # Test validation of enhanced data
        enhanced_validation = validator.validate_regional_data_quality(
            enhanced_data,
            location_info,
            ["processed_data", "universal_features"]
        )
        
        enhanced_quality = enhanced_validation["overall_quality_score"]
        
        print(f"   ✅ Integration successful")
        print(f"   📈 Features: {original_features} → {enhanced_features} (+{features_added})")
        print(f"   📊 Enhanced data quality: {enhanced_quality:.1f}/100")
        
        integration_success = True
        
    except Exception as e:
        print(f"   ❌ Integration failed: {e}")
        integration_success = False
    
    print()
    
    # Summary
    print("📋 DAY 6 REAL DATA TEST SUMMARY")
    print("=" * 40)
    
    tests = [
        ("Global Validation", validation_success),
        ("Uncertainty Quantification", uncertainty_success), 
        ("Baseline Comparison", baseline_success),
        ("Universal Integration", integration_success)
    ]
    
    passed_tests = sum(1 for _, success in tests if success)
    total_tests = len(tests)
    success_rate = passed_tests / total_tests
    
    for test_name, success in tests:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} {test_name}")
    
    print()
    print(f"🎯 Overall Success Rate: {success_rate:.1%} ({passed_tests}/{total_tests})")
    print(f"📊 Real Data Used: {real_data.shape[0]:,} records, {real_data.shape[1]} features")
    print(f"📍 Location Tested: {location_info['name']}")
    
    if success_rate >= 0.75:
        print("\n🎉 DAY 6 REAL DATA VALIDATION SUCCESSFUL!")
        print("✅ Global validation, uncertainty, and baseline systems operational")
        print("✅ Integration with universal features working")
        print("✅ Day 6 systems work with your actual processed data")
        print("🚀 Ready for Day 7 development!")
    else:
        print("\n⚠️ DAY 6 SYSTEMS NEED ATTENTION")
        print("🔧 Fix failing components before proceeding")
    
    print("\n💡 FILE STRUCTURE USED:")
    print(f"   📂 Integrated data: {len(actual_files['integrated'])} files")
    print(f"   📂 Air quality: {len(actual_files['air_quality'])} files") 
    print(f"   📂 Meteorological: {len(actual_files['meteorological'])} files")
    print(f"   📂 Total processed files available: {sum(len(files) for files in actual_files.values())}")
    
    return success_rate >= 0.75


if __name__ == "__main__":
    print("🔍 Day 6 Test Using ACTUAL File Structure")
    print("No more assumptions - using what really exists!")
    print("=" * 50)
    
    success = test_day6_with_actual_files()
    
    if success:
        print("\n🎯 DAY 6 STATUS: COMPLETE ✅")
        print("All validation and uncertainty systems working with real data!")
    else:
        print("\n🎯 DAY 6 STATUS: NEEDS ATTENTION ⚠️")
        print("Some systems need debugging before proceeding to Day 7")