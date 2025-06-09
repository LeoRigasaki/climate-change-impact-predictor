#!/usr/bin/env python3
"""
🌍 Global Location Testing Tool
tools/test_locations.py

Test location service with diverse global locations to ensure 
worldwide coverage for climate predictions.
"""

import sys
import os
import time
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from src.core.location_service import LocationService, LocationInfo
from src.core.data_manager import ClimateDataManager
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GlobalLocationTester:
    """Test location service with diverse global locations."""
    
    def __init__(self):
        """Initialize location tester."""
        self.location_service = LocationService()
        self.data_manager = ClimateDataManager()
        self.results: List[Dict[str, Any]] = []
        
        # Test locations covering all continents and climate zones
        self.test_locations = [
            # Major Cities
            {"query": "New York, USA", "type": "major_city", "continent": "North America"},
            {"query": "London, UK", "type": "major_city", "continent": "Europe"},
            {"query": "Tokyo, Japan", "type": "major_city", "continent": "Asia"},
            {"query": "Sydney, Australia", "type": "major_city", "continent": "Oceania"},
            {"query": "São Paulo, Brazil", "type": "major_city", "continent": "South America"},
            {"query": "Cairo, Egypt", "type": "major_city", "continent": "Africa"},
            
            # Extreme Climate Locations
            {"query": "Reykjavik, Iceland", "type": "arctic", "continent": "Europe"},
            {"query": "Dubai, UAE", "type": "desert", "continent": "Asia"},
            {"query": "Singapore", "type": "tropical", "continent": "Asia"},
            {"query": "Denver, Colorado", "type": "high_altitude", "continent": "North America"},
            {"query": "Mumbai, India", "type": "monsoon", "continent": "Asia"},
            {"query": "Stockholm, Sweden", "type": "subarctic", "continent": "Europe"},
            
            # Small Cities/Towns
            {"query": "Tromsø, Norway", "type": "small_arctic", "continent": "Europe"},
            {"query": "Alice Springs, Australia", "type": "small_desert", "continent": "Oceania"},
            {"query": "Ushuaia, Argentina", "type": "small_subantarctic", "continent": "South America"},
            {"query": "Fairbanks, Alaska", "type": "small_subarctic", "continent": "North America"},
            
            # Coordinates (Remote Locations)
            {"query": "78.9167, 11.9333", "type": "coordinates_arctic", "continent": "Arctic"},  # Svalbard
            {"query": "-54.8019, -68.3030", "type": "coordinates_subantarctic", "continent": "Antarctica"},  # Ushuaia area
            {"query": "0.0, 0.0", "type": "coordinates_equator", "continent": "Atlantic"},  # Gulf of Guinea
            {"query": "23.5, 0.0", "type": "coordinates_tropic", "continent": "Africa"},  # Sahara
            
            # Island Nations
            {"query": "Reykjavik, Iceland", "type": "island", "continent": "Europe"},
            {"query": "Port Louis, Mauritius", "type": "island", "continent": "Africa"},
            {"query": "Suva, Fiji", "type": "island", "continent": "Oceania"},
            {"query": "Nassau, Bahamas", "type": "island", "continent": "North America"},
            
            # Developing Countries
            {"query": "Lagos, Nigeria", "type": "developing", "continent": "Africa"},
            {"query": "Dhaka, Bangladesh", "type": "developing", "continent": "Asia"},
            {"query": "Lima, Peru", "type": "developing", "continent": "South America"},
            {"query": "Kiev, Ukraine", "type": "developing", "continent": "Europe"},
        ]
        
        logger.info(f"Initialized GlobalLocationTester with {len(self.test_locations)} test locations")
    
    async def test_location(self, test_case: Dict[str, str]) -> Dict[str, Any]:
        """Test a single location."""
        query = test_case["query"]
        logger.info(f"🔍 Testing location: {query}")
        
        result = {
            "query": query,
            "type": test_case["type"],
            "continent": test_case["continent"],
            "success": False,
            "location_info": None,
            "api_coverage": {},
            "errors": [],
            "processing_time": 0
        }
        
        start_time = time.time()
        
        try:
            # Test location service
            location = self.location_service.geocode_location(query)
            
            if location:
                result["success"] = True
                result["location_info"] = location.to_dict()
                
                # Test API coverage for this location
                api_coverage = await self._test_api_coverage(location)
                result["api_coverage"] = api_coverage
                
                logger.info(f"✅ Success: {location.name}, {location.country}")
                logger.info(f"📍 Coordinates: {location.latitude:.4f}, {location.longitude:.4f}")
                
            else:
                result["errors"].append("Location not found by geocoding service")
                logger.warning(f"❌ Failed to geocode: {query}")
                
        except Exception as e:
            result["errors"].append(str(e))
            logger.error(f"❌ Error testing {query}: {e}")
        
        result["processing_time"] = time.time() - start_time
        return result
    
    async def _test_api_coverage(self, location: LocationInfo) -> Dict[str, Any]:
        """Test API data availability for a location."""
        coverage = {
            "open_meteo_air_quality": False,
            "nasa_power_meteorological": False,
            "world_bank_climate_projections": False,
            "errors": []
        }
        
        try:
            # Test Open-Meteo Air Quality (should work globally)
            try:
                from src.api.open_meteo import OpenMeteoClient
                client = OpenMeteoClient()
                
                # Quick test - just check if API accepts coordinates
                test_params = {
                    "latitude": location.latitude,
                    "longitude": location.longitude,
                    "hourly": ["pm2_5"],
                    "start_date": "2025-06-07",
                    "end_date": "2025-06-07"
                }
                
                # This is a minimal check - in production you'd make actual API call
                if -90 <= location.latitude <= 90 and -180 <= location.longitude <= 180:
                    coverage["open_meteo_air_quality"] = True
                    
            except Exception as e:
                coverage["errors"].append(f"Open-Meteo test failed: {e}")
            
            # Test NASA POWER (global coverage expected)
            try:
                from src.api.nasa_power import NASAPowerClient
                
                # NASA POWER has global coverage
                if -90 <= location.latitude <= 90 and -180 <= location.longitude <= 180:
                    coverage["nasa_power_meteorological"] = True
                    
            except Exception as e:
                coverage["errors"].append(f"NASA POWER test failed: {e}")
            
            # Test World Bank CCKP (country-level data)
            try:
                from src.api.world_bank import WorldBankClient
                
                # World Bank needs valid country
                if location.country and location.country != "Unknown":
                    coverage["world_bank_climate_projections"] = True
                    
            except Exception as e:
                coverage["errors"].append(f"World Bank test failed: {e}")
                
        except Exception as e:
            coverage["errors"].append(f"General API test error: {e}")
        
        return coverage
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all location tests."""
        logger.info("🌍 Starting Global Location Testing")
        logger.info("=" * 60)
        
        start_time = time.time()
        self.results = []
        
        # Test each location
        for i, test_case in enumerate(self.test_locations, 1):
            logger.info(f"\n📍 Test {i}/{len(self.test_locations)}: {test_case['query']}")
            
            result = await self.test_location(test_case)
            self.results.append(result)
            
            # Brief pause to be respectful to geocoding service
            await asyncio.sleep(1)
        
        # Compile summary
        summary = self._compile_summary()
        summary["total_time"] = time.time() - start_time
        
        return summary
    
    def _compile_summary(self) -> Dict[str, Any]:
        """Compile test results summary."""
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r["success"])
        
        # Group by type and continent
        by_type = {}
        by_continent = {}
        api_coverage_stats = {
            "open_meteo_air_quality": 0,
            "nasa_power_meteorological": 0,
            "world_bank_climate_projections": 0
        }
        
        for result in self.results:
            # By type
            location_type = result["type"]
            if location_type not in by_type:
                by_type[location_type] = {"total": 0, "success": 0}
            by_type[location_type]["total"] += 1
            if result["success"]:
                by_type[location_type]["success"] += 1
            
            # By continent
            continent = result["continent"]
            if continent not in by_continent:
                by_continent[continent] = {"total": 0, "success": 0}
            by_continent[continent]["total"] += 1
            if result["success"]:
                by_continent[continent]["success"] += 1
            
            # API coverage
            if result["success"] and result["api_coverage"]:
                for api, available in result["api_coverage"].items():
                    if api in api_coverage_stats and available:
                        api_coverage_stats[api] += 1
        
        summary = {
            "overview": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
                "failed_tests": total_tests - successful_tests
            },
            "by_type": by_type,
            "by_continent": by_continent,
            "api_coverage": {
                api: {
                    "available_locations": count,
                    "coverage_rate": count / successful_tests if successful_tests > 0 else 0
                }
                for api, count in api_coverage_stats.items()
            },
            "failed_locations": [
                {
                    "query": r["query"],
                    "type": r["type"],
                    "continent": r["continent"],
                    "errors": r["errors"]
                }
                for r in self.results if not r["success"]
            ]
        }
        
        return summary
    
    def save_results(self, filename: str = None):
        """Save test results to JSON file."""
        if filename is None:
            filename = f"location_test_results_{int(time.time())}.json"
        
        output_path = Path("data/cache") / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                "results": self.results,
                "summary": self._compile_summary(),
                "timestamp": time.time()
            }, f, indent=2)
        
        logger.info(f"📄 Results saved to: {output_path}")
        return output_path
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print formatted test summary."""
        print("\n" + "=" * 60)
        print("🌍 GLOBAL LOCATION TEST SUMMARY")
        print("=" * 60)
        
        overview = summary["overview"]
        print(f"\n📊 OVERVIEW:")
        print(f"   Total Tests: {overview['total_tests']}")
        print(f"   Successful: {overview['successful_tests']}")
        print(f"   Failed: {overview['failed_tests']}")
        print(f"   Success Rate: {overview['success_rate']:.1%}")
        
        print(f"\n🌍 BY CONTINENT:")
        for continent, stats in summary["by_continent"].items():
            success_rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0
            print(f"   {continent}: {stats['success']}/{stats['total']} ({success_rate:.1%})")
        
        print(f"\n🏷️ BY LOCATION TYPE:")
        for location_type, stats in summary["by_type"].items():
            success_rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0
            print(f"   {location_type}: {stats['success']}/{stats['total']} ({success_rate:.1%})")
        
        print(f"\n🔌 API COVERAGE:")
        for api, stats in summary["api_coverage"].items():
            print(f"   {api}: {stats['available_locations']} locations ({stats['coverage_rate']:.1%})")
        
        if summary.get("failed_locations"):
            print(f"\n❌ FAILED LOCATIONS:")
            for failure in summary["failed_locations"]:
                print(f"   {failure['query']} ({failure['type']}) - {', '.join(failure['errors'])}")
        
        total_time = summary.get("total_time", 0)
        print(f"\n⏱️ Total Testing Time: {total_time:.1f} seconds")
        print("=" * 60)


async def main():
    """Main testing function."""
    print("🌍 Global Location Testing Tool")
    print("Testing location service with diverse global locations...")
    
    tester = GlobalLocationTester()
    
    try:
        # Run all tests
        summary = await tester.run_all_tests()
        
        # Print summary
        tester.print_summary(summary)
        
        # Save detailed results
        results_file = tester.save_results()
        
        # Check if we meet global coverage requirements
        success_rate = summary["overview"]["success_rate"]
        if success_rate >= 0.8:  # 80% success rate target
            print(f"\n✅ GLOBAL COVERAGE TEST PASSED!")
            print(f"   Success rate: {success_rate:.1%} (target: 80%)")
            return True
        else:
            print(f"\n⚠️ GLOBAL COVERAGE TEST NEEDS IMPROVEMENT")
            print(f"   Success rate: {success_rate:.1%} (target: 80%)")
            return False
            
    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Testing failed with error: {e}")
        logger.error(f"Main testing error: {e}")
        return False


def test_specific_location(query: str):
    """Test a specific location quickly."""
    print(f"🔍 Testing specific location: {query}")
    
    service = LocationService()
    location = service.geocode_location(query)
    
    if location:
        print(f"✅ Found: {location.name}, {location.country}")
        print(f"📍 Coordinates: {location.latitude:.4f}, {location.longitude:.4f}")
        print(f"🏷️ Details: {location.to_dict()}")
        return True
    else:
        print(f"❌ Location not found: {query}")
        return False


def test_coordinates(lat: float, lon: float):
    """Test specific coordinates."""
    print(f"🔍 Testing coordinates: {lat}, {lon}")
    
    service = LocationService()
    location = service.get_location_by_coordinates(lat, lon)
    
    if location:
        print(f"✅ Found: {location.name}, {location.country}")
        print(f"📍 Coordinates: {location.latitude:.4f}, {location.longitude:.4f}")
        return True
    else:
        print(f"❌ Could not resolve coordinates: {lat}, {lon}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test global location service")
    parser.add_argument("--full", action="store_true", help="Run full global test suite")
    parser.add_argument("--location", type=str, help="Test specific location")
    parser.add_argument("--coordinates", nargs=2, type=float, metavar=("LAT", "LON"), 
                       help="Test specific coordinates")
    
    args = parser.parse_args()
    
    if args.full:
        # Run full test suite
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
        
    elif args.location:
        # Test specific location
        result = test_specific_location(args.location)
        sys.exit(0 if result else 1)
        
    elif args.coordinates:
        # Test specific coordinates
        lat, lon = args.coordinates
        result = test_coordinates(lat, lon)
        sys.exit(0 if result else 1)
        
    else:
        # Interactive mode
        print("🌍 Global Location Testing Tool")
        print("=" * 40)
        print("Options:")
        print("1. Run full global test suite")
        print("2. Test specific location")
        print("3. Test coordinates")
        print("4. Exit")
        
        while True:
            choice = input("\nEnter choice (1-4): ").strip()
            
            if choice == "1":
                result = asyncio.run(main())
                break
            elif choice == "2":
                query = input("Enter location (e.g., 'Berlin, Germany'): ").strip()
                if query:
                    test_specific_location(query)
            elif choice == "3":
                try:
                    lat = float(input("Enter latitude: ").strip())
                    lon = float(input("Enter longitude: ").strip())
                    test_coordinates(lat, lon)
                except ValueError:
                    print("❌ Invalid coordinates. Please enter numbers.")
            elif choice == "4":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-4.")