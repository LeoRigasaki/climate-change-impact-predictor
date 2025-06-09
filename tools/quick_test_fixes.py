#!/usr/bin/env python3
"""
🧪 Quick Test for Day 5 Bug Fixes
tools/quick_test_fixes.py

Quick validation that the global comparisons bug is fixed.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_global_comparisons_fix():
    """Test that global comparisons module works without boolean errors."""
    print("🔧 Testing Global Comparisons Bug Fixes...")
    
    try:
        from src.features.universal_engine import UniversalFeatureEngine
        
        # Create test data
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        test_data = pd.DataFrame({
            "temperature_2m": np.random.normal(20, 5, 30),
            "precipitation": np.random.exponential(2, 30),
            "relative_humidity": np.random.uniform(40, 80, 30),
            "wind_speed_2m": np.random.exponential(3, 30),
            "pm2_5": np.random.exponential(15, 30)
        }, index=dates)
        
        # Test location
        location_info = {
            "name": "Test City",
            "country": "Test Country", 
            "latitude": 40.0,
            "longitude": 0.0
        }
        
        # Test the engine
        engine = UniversalFeatureEngine()
        enhanced_data = engine.engineer_features(test_data, location_info)
        
        # Check if global comparison features were created
        global_features = [col for col in enhanced_data.columns if 'global' in col.lower()]
        
        print(f"✅ Test passed!")
        print(f"📊 Original features: {test_data.shape[1]}")
        print(f"📈 Enhanced features: {enhanced_data.shape[1]}")
        print(f"🌍 Global features created: {len(global_features)}")
        
        if len(global_features) > 0:
            print(f"🎉 Global comparisons bug FIXED!")
            print("📋 Global features created:")
            for feature in global_features[:5]:  # Show first 5
                print(f"   • {feature}")
        else:
            print("⚠️ Global features still not being created")
        
        return True, len(global_features)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False, 0

def main():
    """Run quick test."""
    print("🔧 Day 5 Bug Fix Validation")
    print("=" * 40)
    
    success, global_features_count = test_global_comparisons_fix()
    
    print("\n" + "=" * 40)
    if success and global_features_count > 0:
        print("🎉 BUG FIXES SUCCESSFUL!")
        print("✅ Global comparisons module working")
        print("✅ Ready to run full test suite")
        print("\nRun: python -m tools.test_day5_features")
    elif success:
        print("⚠️ Partial Success")
        print("✅ No crashes, but global features still missing")
        print("🔧 May need additional debugging")
    else:
        print("❌ Bug fixes incomplete")
        print("🔧 Additional work needed")

if __name__ == "__main__":
    main()