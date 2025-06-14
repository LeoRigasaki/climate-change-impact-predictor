#!/usr/bin/env python3
"""
🌍 Day 7: Comprehensive Continental Testing Suite
tools/test_day7_continental.py

Tests the Climate Change Impact Predictor system across all 7 continents
with challenging locations to ensure global reliability and performance.
"""

import sys
import time
import asyncio
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
import statistics

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.core.location_service import LocationService
from src.core.data_manager import ClimateDataManager
from src.core.pipeline import ClimateDataPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Day7ContinentalTester:
    """
    🌍 Comprehensive continental testing system for global climate system validation.
    
    Features:
    - Tests all 7 continents with diverse locations
    - Performance benchmarking by region
    - Edge case validation (remote areas, special names)
    - Data availability assessment globally
    - Comprehensive reporting for portfolio showcase
    """
    
    def __init__(self):
        self.location_service = LocationService()
        self.data_manager = ClimateDataManager()
        self.pipeline = ClimateDataPipeline()
        
        # Continental test locations (challenging and diverse)
        self.continental_locations = {
            "North America": [
                "New York, USA", "Los Angeles, USA", "Mexico City, Mexico", 
                "Vancouver, Canada", "Anchorage, Alaska", "Nuuk, Greenland",
                "Yellowknife, Canada", "Miami, USA"
            ],
            "South America": [
                "São Paulo, Brazil", "Buenos Aires, Argentina", "Lima, Peru",
                "Bogotá, Colombia", "Quito, Ecuador", "La Paz, Bolivia",
                "Santiago, Chile", "Manaus, Brazil"
            ],
            "Europe": [
                "London, UK", "Berlin, Germany", "Moscow, Russia",
                "Stockholm, Sweden", "Athens, Greece", "Reykjavik, Iceland",
                "Oslo, Norway", "Lisbon, Portugal"
            ],
            "Africa": [
                "Cairo, Egypt", "Lagos, Nigeria", "Cape Town, South Africa",
                "Nairobi, Kenya", "Casablanca, Morocco", "Addis Ababa, Ethiopia",
                "Kinshasa, DR Congo", "Tunis, Tunisia"
            ],
            "Asia": [
                "Tokyo, Japan", "Mumbai, India", "Beijing, China",
                "Bangkok, Thailand", "Seoul, South Korea", "Jakarta, Indonesia",
                "Dhaka, Bangladesh", "Manila, Philippines"
            ],
            "Oceania": [
                "Sydney, Australia", "Auckland, New Zealand", "Suva, Fiji",
                "Port Moresby, Papua New Guinea", "Honolulu, USA", "Perth, Australia",
                "Melbourne, Australia", "Wellington, New Zealand"
            ],
            "Antarctica": [
                # Research stations (challenging edge cases)
                "McMurdo Station, Antarctica", "Rothera Research Station, Antarctica",
                "Halley Research Station, Antarctica"
            ]
        }
        
        # Performance benchmarks
        self.performance_targets = {
            "geocoding_time": 5.0,  # seconds
            "data_availability_time": 8.0,  # seconds
            "data_collection_time": 30.0,  # seconds
            "success_rate": 0.80,  # 80% minimum
            "cache_speedup": 2.0  # 2x minimum
        }
        
        # Test results storage
        self.test_results = {
            "continental_tests": {},
            "performance_metrics": {},
            "edge_cases": {},
            "global_summary": {},
            "recommendations": []
        }
        
        logger.info("🌍 Day 7 Continental Tester initialized")
        logger.info(f"📍 Testing {sum(len(locs) for locs in self.continental_locations.values())} global locations")
    
    async def run_continental_testing(self) -> Dict[str, Any]:
        """🌍 Run comprehensive continental testing suite."""
        
        print("🌍 DAY 7: COMPREHENSIVE CONTINENTAL TESTING")
        print("=" * 60)
        print("Testing climate system reliability across all 7 continents...")
        print()
        
        start_time = time.time()
        
        # Test each continent
        for continent, locations in self.continental_locations.items():
            print(f"\n🗺️ TESTING: {continent.upper()}")
            print("-" * 40)
            
            continent_results = await self._test_continent(continent, locations)
            self.test_results["continental_tests"][continent] = continent_results
            
            # Print continent summary
            success_rate = continent_results["success_rate"]
            avg_time = continent_results["avg_response_time"]
            status_emoji = "✅" if success_rate >= 0.8 else "⚠️" if success_rate >= 0.6 else "❌"
            
            print(f"{status_emoji} {continent}: {success_rate:.1%} success rate, {avg_time:.1f}s avg response")
        
        # Run edge case testing
        print(f"\n🚀 TESTING: EDGE CASES")
        print("-" * 40)
        edge_results = await self._test_edge_cases()
        self.test_results["edge_cases"] = edge_results
        
        # Calculate global metrics
        await self._calculate_global_metrics()
        
        # Generate comprehensive report
        total_time = time.time() - start_time
        self.test_results["total_test_time"] = total_time
        
        print(f"\n🎯 CONTINENTAL TESTING COMPLETE")
        print(f"⏱️ Total Time: {total_time:.1f} seconds")
        
        return self.test_results
    
    async def _test_continent(self, continent: str, locations: List[str]) -> Dict[str, Any]:
        """Test all locations in a specific continent."""
        
        results = {
            "locations_tested": len(locations),
            "successful_locations": 0,
            "failed_locations": [],
            "response_times": [],
            "data_availability": {},
            "success_rate": 0.0,
            "avg_response_time": 0.0,
            "best_performance": None,
            "worst_performance": None
        }
        
        for location_name in locations:
            try:
                location_start = time.time()
                
                # Test 1: Location Resolution
                print(f"  📍 Testing {location_name}... ", end="", flush=True)
                location = self.location_service.geocode_location(location_name)
                
                if not location:
                    print("❌ Location not found")
                    results["failed_locations"].append(location_name)
                    continue
                
                # Test 2: Data Availability Check
                availability = await self.data_manager.check_data_availability(location)
                available_sources = sum(availability.values())
                
                # Test 3: Quick Data Collection Test (minimal)
                try:
                    # Try a small data collection to verify APIs work
                    end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
                    start_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
                    
                    quick_data = await self.data_manager.fetch_adaptive_data(
                        location=location,
                        start_date=start_date,
                        end_date=end_date,
                        save=False,
                        force_all=False
                    )
                    
                    collection_successful = len([d for d in quick_data.values() if d is not None]) > 0
                    
                except Exception as e:
                    collection_successful = False
                    logger.warning(f"Data collection failed for {location_name}: {e}")
                
                location_time = time.time() - location_start
                results["response_times"].append(location_time)
                
                # Store availability data
                results["data_availability"][location_name] = {
                    "available_sources": available_sources,
                    "details": availability,
                    "collection_test": collection_successful
                }
                
                # Success criteria: location found + at least 1 data source + collection works
                if available_sources > 0 and collection_successful:
                    results["successful_locations"] += 1
                    print(f"✅ {location_time:.1f}s ({available_sources}/4 sources)")
                else:
                    print(f"⚠️ {location_time:.1f}s (limited data: {available_sources}/4 sources)")
                    results["failed_locations"].append(f"{location_name} (limited data)")
                
            except Exception as e:
                location_time = time.time() - location_start
                results["response_times"].append(location_time)
                results["failed_locations"].append(f"{location_name} (error: {str(e)[:50]})")
                print(f"❌ {location_time:.1f}s (error)")
                logger.error(f"Location test failed for {location_name}: {e}")
        
        # Calculate continent metrics
        if results["response_times"]:
            results["success_rate"] = results["successful_locations"] / results["locations_tested"]
            results["avg_response_time"] = statistics.mean(results["response_times"])
            results["best_performance"] = min(results["response_times"])
            results["worst_performance"] = max(results["response_times"])
        
        return results
    
    async def _test_edge_cases(self) -> Dict[str, Any]:
        """Test challenging edge cases for system robustness."""
        
        edge_cases = {
            "Special Characters": [
                "São Paulo, Brazil", "Zürich, Switzerland", "Kraków, Poland",
                "Malmö, Sweden", "Røros, Norway"
            ],
            "Remote Locations": [
                "Tristan da Cunha", "Easter Island, Chile", "Svalbard, Norway",
                "Pitcairn Islands", "Saint Helena"
            ],
            "Coordinate Edge Cases": [
                (0, 0),  # Null Island
                (90, 0),  # North Pole
                (-90, 0),  # South Pole
                (23.5, 0),  # Tropic of Cancer
                (-23.5, 0)  # Tropic of Capricorn
            ],
            "Ambiguous Names": [
                "Paris", "London", "Moscow", "Berlin"  # Could match multiple countries
            ]
        }
        
        results = {
            "categories_tested": len(edge_cases),
            "category_results": {},
            "overall_robustness": 0.0
        }
        
        total_tests = 0
        total_successful = 0
        
        for category, test_cases in edge_cases.items():
            print(f"  🧪 Testing {category}...")
            
            category_results = {
                "tests_run": len(test_cases),
                "successful": 0,
                "failed": [],
                "special_notes": []
            }
            
            for test_case in test_cases:
                try:
                    if isinstance(test_case, tuple):
                        # Coordinate test
                        lat, lon = test_case
                        location = self.location_service.get_location_by_coordinates(lat, lon)
                        test_name = f"({lat}, {lon})"
                    else:
                        # Name test
                        location = self.location_service.geocode_location(test_case)
                        test_name = test_case
                    
                    if location:
                        category_results["successful"] += 1
                        print(f"    ✅ {test_name}")
                        
                        # Special handling for ambiguous names
                        if category == "Ambiguous Names":
                            category_results["special_notes"].append(
                                f"{test_name} → {location.name}, {location.country}"
                            )
                    else:
                        category_results["failed"].append(test_name)
                        print(f"    ❌ {test_name}")
                
                except Exception as e:
                    category_results["failed"].append(f"{test_name} (error)")
                    print(f"    ❌ {test_name} (error: {str(e)[:30]})")
            
            results["category_results"][category] = category_results
            total_tests += category_results["tests_run"]
            total_successful += category_results["successful"]
        
        results["overall_robustness"] = total_successful / total_tests if total_tests > 0 else 0.0
        
        return results
    
    async def _calculate_global_metrics(self):
        """Calculate comprehensive global performance metrics."""
        
        # Aggregate continental results
        all_response_times = []
        total_locations = 0
        total_successful = 0
        continental_success_rates = []
        
        for continent, results in self.test_results["continental_tests"].items():
            all_response_times.extend(results["response_times"])
            total_locations += results["locations_tested"]
            total_successful += results["successful_locations"]
            continental_success_rates.append(results["success_rate"])
        
        # Calculate global metrics
        global_metrics = {
            "total_locations_tested": total_locations,
            "global_success_rate": total_successful / total_locations if total_locations > 0 else 0.0,
            "avg_continental_success_rate": statistics.mean(continental_success_rates) if continental_success_rates else 0.0,
            "global_avg_response_time": statistics.mean(all_response_times) if all_response_times else 0.0,
            "fastest_response": min(all_response_times) if all_response_times else 0.0,
            "slowest_response": max(all_response_times) if all_response_times else 0.0,
            "response_time_std": statistics.stdev(all_response_times) if len(all_response_times) > 1 else 0.0,
            "edge_case_robustness": self.test_results["edge_cases"]["overall_robustness"],
            "performance_benchmarks_met": {}
        }
        
        # Check performance benchmarks
        benchmarks = self.performance_targets
        global_metrics["performance_benchmarks_met"] = {
            "geocoding_speed": global_metrics["global_avg_response_time"] <= benchmarks["geocoding_time"],
            "success_rate": global_metrics["global_success_rate"] >= benchmarks["success_rate"],
            "robustness": global_metrics["edge_case_robustness"] >= 0.7
        }
        
        # Overall system readiness
        benchmarks_met = sum(global_metrics["performance_benchmarks_met"].values())
        total_benchmarks = len(global_metrics["performance_benchmarks_met"])
        global_metrics["system_readiness_score"] = benchmarks_met / total_benchmarks
        
        self.test_results["global_summary"] = global_metrics
        
        # Generate recommendations
        self._generate_recommendations()
    
    def _generate_recommendations(self):
        """Generate actionable recommendations based on test results."""
        
        recommendations = []
        metrics = self.test_results["global_summary"]
        
        # Performance recommendations
        if metrics["global_avg_response_time"] > self.performance_targets["geocoding_time"]:
            recommendations.append(
                f"🔧 Performance: Average response time ({metrics['global_avg_response_time']:.1f}s) "
                f"exceeds target ({self.performance_targets['geocoding_time']}s). "
                f"Consider implementing connection pooling and advanced caching."
            )
        
        # Success rate recommendations
        if metrics["global_success_rate"] < self.performance_targets["success_rate"]:
            recommendations.append(
                f"📊 Reliability: Global success rate ({metrics['global_success_rate']:.1%}) "
                f"below target ({self.performance_targets['success_rate']:.1%}). "
                f"Review failed locations and improve error handling."
            )
        
        # Edge case recommendations
        if metrics["edge_case_robustness"] < 0.8:
            recommendations.append(
                f"🧪 Robustness: Edge case handling ({metrics['edge_case_robustness']:.1%}) "
                f"needs improvement. Focus on special character handling and ambiguous name resolution."
            )
        
        # Continental performance variance
        continental_rates = [
            results["success_rate"] 
            for results in self.test_results["continental_tests"].values()
        ]
        
        if len(continental_rates) > 1 and statistics.stdev(continental_rates) > 0.2:
            recommendations.append(
                f"🌍 Regional Consistency: High variance in continental performance detected. "
                f"Investigate regional API limitations and implement region-specific optimizations."
            )
        
        # Positive achievements
        if metrics["system_readiness_score"] >= 0.8:
            recommendations.append(
                f"🎉 Achievement: System demonstrates excellent global readiness ({metrics['system_readiness_score']:.1%}). "
                f"Ready for production deployment and Phase 2 development."
            )
        
        self.test_results["recommendations"] = recommendations
    
    def print_comprehensive_results(self):
        """Print detailed test results for Day 7 validation."""
        
        print("\n" + "=" * 80)
        print("🌍 DAY 7 COMPREHENSIVE CONTINENTAL TESTING RESULTS")
        print("=" * 80)
        
        metrics = self.test_results["global_summary"]
        
        # Global Summary
        readiness_score = metrics["system_readiness_score"]
        readiness_emoji = "🎉" if readiness_score >= 0.9 else "✅" if readiness_score >= 0.8 else "⚠️" if readiness_score >= 0.7 else "❌"
        
        print(f"\n🎯 GLOBAL SYSTEM READINESS: {readiness_emoji} {readiness_score:.1%}")
        print(f"📊 Locations Tested: {metrics['total_locations_tested']}")
        print(f"✅ Global Success Rate: {metrics['global_success_rate']:.1%}")
        print(f"⚡ Average Response Time: {metrics['global_avg_response_time']:.1f}s")
        print(f"🧪 Edge Case Robustness: {metrics['edge_case_robustness']:.1%}")
        
        # Continental Breakdown
        print(f"\n🗺️ CONTINENTAL PERFORMANCE BREAKDOWN:")
        for continent, results in self.test_results["continental_tests"].items():
            success_rate = results["success_rate"]
            avg_time = results["avg_response_time"]
            status_emoji = "✅" if success_rate >= 0.8 else "⚠️" if success_rate >= 0.6 else "❌"
            
            print(f"   {status_emoji} {continent:15} {success_rate:>6.1%} success | {avg_time:>5.1f}s avg | "
                  f"{results['successful_locations']:>2}/{results['locations_tested']:>2} locations")
        
        # Performance Benchmarks
        print(f"\n⚡ PERFORMANCE BENCHMARKS:")
        benchmarks = metrics["performance_benchmarks_met"]
        for benchmark, met in benchmarks.items():
            status = "✅" if met else "❌"
            print(f"   {status} {benchmark.replace('_', ' ').title()}")
        
        # Edge Cases Summary
        print(f"\n🧪 EDGE CASE TESTING:")
        edge_results = self.test_results["edge_cases"]
        for category, results in edge_results["category_results"].items():
            success_rate = results["successful"] / results["tests_run"] if results["tests_run"] > 0 else 0
            status_emoji = "✅" if success_rate >= 0.8 else "⚠️" if success_rate >= 0.6 else "❌"
            print(f"   {status_emoji} {category:20} {success_rate:>6.1%} ({results['successful']:>2}/{results['tests_run']:>2})")
        
        # Recommendations
        if self.test_results["recommendations"]:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(self.test_results["recommendations"], 1):
                print(f"   {i}. {rec}")
        
        # Day 7 Success Assessment
        print(f"\n" + "=" * 80)
        
        day7_success = (
            metrics["system_readiness_score"] >= 0.8 and
            metrics["global_success_rate"] >= 0.75 and
            metrics["global_avg_response_time"] <= 10.0
        )
        
        if day7_success:
            print("🎉 DAY 7 SUCCESS CRITERIA MET!")
            print("   ✅ Global system reliability confirmed")
            print("   ✅ Continental coverage validated") 
            print("   ✅ Performance benchmarks achieved")
            print("   ✅ Edge case robustness demonstrated")
            print("   ✅ System ready for Phase 2 (ML Models)")
            print()
            print("🚀 READY TO PROCEED TO DAY 8: Global Model Architecture")
        else:
            print("⚠️ DAY 7 NEEDS ATTENTION")
            print("   Focus on recommendations before proceeding to Day 8")
        
        return day7_success
    
    def save_results(self):
        """Save comprehensive test results for documentation."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = Path("logs") / f"day7_continental_results_{timestamp}.json"
        
        # Ensure logs directory exists
        results_file.parent.mkdir(exist_ok=True)
        
        # Save detailed results
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved: {results_file}")
        return results_file


async def main():
    """Main testing function for Day 7 continental validation."""
    
    print("🌍 Day 7: Continental Testing Suite")
    print("Testing global climate system across all continents...")
    print()
    
    tester = Day7ContinentalTester()
    
    try:
        # Run comprehensive continental testing
        results = await tester.run_continental_testing()
        
        # Print detailed results
        success = tester.print_comprehensive_results()
        
        # Save results for documentation
        tester.save_results()
        
        return success
        
    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        logger.error(f"Continental testing error: {e}")
        return False


if __name__ == "__main__":
    print("🌍 Starting Day 7 Comprehensive Continental Testing...")
    
    result = asyncio.run(main())
    
    if result:
        print("\n🎯 CONTINENTAL TESTING: SUCCESS ✅")
        print("Your climate system is globally ready!")
    else:
        print("\n🎯 CONTINENTAL TESTING: NEEDS IMPROVEMENT ⚠️")
        print("Review recommendations and address issues.")
    
    print("\n💡 Next Step: Performance Optimization")
    print("Run: python tools/test_day7_continental.py")