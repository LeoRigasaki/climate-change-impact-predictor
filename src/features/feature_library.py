#!/usr/bin/env python3
"""
📚 Feature Library - Day 5 Feature Engineering
src/features/feature_library.py

Comprehensive documentation and registry of all universal climate features.
Provides feature explanations, categories, and usage guidelines for the
complete universal feature engineering system.
"""

import logging
from typing import Dict, List, Optional, Any
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class FeatureLibrary:
    """
    📚 Universal Climate Feature Library and Documentation System
    
    Provides comprehensive documentation for all universal climate features:
    - Feature explanations and interpretations
    - Feature categories and groupings
    - Usage guidelines and best practices
    - Feature importance rankings
    - Feature relationships and dependencies
    """
    
    def __init__(self):
        """Initialize Feature Library with comprehensive documentation."""
        self.feature_catalog = self._build_comprehensive_catalog()
        self.feature_categories = self._define_feature_categories()
        self.feature_importance = self._define_feature_importance()
        self.feature_relationships = self._define_feature_relationships()
        
        logger.info(f"📚 Feature Library initialized with {len(self.feature_catalog)} documented features")
    
    def _build_comprehensive_catalog(self) -> Dict[str, Dict[str, Any]]:
        """Build comprehensive catalog of all universal features."""
        return {
            # ========================================
            # CLIMATE INDICATORS (Universal Indices)
            # ========================================
            "climate_stress_index": {
                "description": "Overall environmental stress level (0-100, higher = more stressful)",
                "interpretation": "Combines temperature, humidity, wind, and precipitation stress into single metric",
                "scale": "0-100",
                "good_range": "0-30",
                "concerning_range": "70-100",
                "use_cases": ["Health impact assessment", "Comfort evaluation", "Risk screening"],
                "category": "climate_indicators"
            },
            
            "human_comfort_index": {
                "description": "Human thermal comfort level (0-100, higher = more comfortable)",
                "interpretation": "Measures how comfortable current conditions are for humans",
                "scale": "0-100",
                "good_range": "70-100",
                "concerning_range": "0-30",
                "use_cases": ["Tourism planning", "Outdoor activity safety", "HVAC optimization"],
                "category": "climate_indicators"
            },
            
            "weather_extremity_index": {
                "description": "How extreme current weather is globally (0-100, higher = more extreme)",
                "interpretation": "Measures deviation from normal global weather patterns",
                "scale": "0-100",
                "good_range": "0-30",
                "concerning_range": "80-100",
                "use_cases": ["Extreme weather detection", "Insurance risk", "Emergency planning"],
                "category": "climate_indicators"
            },
            
            "air_quality_health_impact": {
                "description": "Health risk from air pollution (0-100, higher = worse health impact)",
                "interpretation": "Combines PM2.5, PM10, ozone, CO into health risk score",
                "scale": "0-100",
                "good_range": "0-25",
                "concerning_range": "60-100",
                "use_cases": ["Health alerts", "Exercise recommendations", "Mask usage guidance"],
                "category": "climate_indicators"
            },
            
            "climate_variability_index": {
                "description": "How variable recent climate has been (0-100, higher = more variable)",
                "interpretation": "Measures short-term climate instability and unpredictability",
                "scale": "0-100",
                "good_range": "0-40",
                "concerning_range": "70-100",
                "use_cases": ["Agricultural planning", "Climate adaptation", "Ecosystem monitoring"],
                "category": "climate_indicators"
            },
            
            "heat_index": {
                "description": "Apparent temperature combining heat and humidity effects (°C)",
                "interpretation": "What temperature feels like to human body accounting for humidity",
                "scale": "°C",
                "good_range": "18-27°C",
                "concerning_range": ">35°C",
                "use_cases": ["Heat stress warnings", "Outdoor work safety", "Sports medicine"],
                "category": "climate_indicators"
            },
            
            "climate_hazard_index": {
                "description": "Overall climate hazard level (0-100, higher = more hazardous)",
                "interpretation": "Composite of stress, extremity, and air quality impacts",
                "scale": "0-100",
                "good_range": "0-30",
                "concerning_range": "70-100",
                "use_cases": ["Emergency management", "Risk assessment", "Public health"],
                "category": "climate_indicators"
            },
            
            "climate_favorability_index": {
                "description": "Overall climate favorability (0-100, higher = more favorable)",
                "interpretation": "Inverse of climate hazard - how favorable conditions are",
                "scale": "0-100",
                "good_range": "70-100",
                "concerning_range": "0-30",
                "use_cases": ["Location selection", "Tourism", "Quality of life assessment"],
                "category": "climate_indicators"
            },
            
            # ========================================
            # REGIONAL ADAPTATION (Local Context)
            # ========================================
            "regional_temp_baseline_center": {
                "description": "Expected normal temperature for this climate zone (°C)",
                "interpretation": "Regional reference point for temperature expectations",
                "scale": "°C",
                "use_cases": ["Regional comparisons", "Climate context", "Anomaly detection"],
                "category": "regional_adaptation"
            },
            
            "temp_deviation_from_regional_norm": {
                "description": "How much temperature deviates from regional normal (°C)",
                "interpretation": "Positive = warmer than regional normal, negative = cooler",
                "scale": "°C",
                "good_range": "-5 to +5°C",
                "concerning_range": ">±15°C",
                "use_cases": ["Regional climate monitoring", "Adaptation planning"],
                "category": "regional_adaptation"
            },
            
            "temperature_2m_local_percentile": {
                "description": "Local temperature percentile (0-100, based on local history)",
                "interpretation": "Where today's temperature ranks in local historical record",
                "scale": "0-100%",
                "good_range": "25-75%",
                "concerning_range": "<5% or >95%",
                "use_cases": ["Local anomaly detection", "Historical context"],
                "category": "regional_adaptation"
            },
            
            "regional_climate_stress_index": {
                "description": "Climate stress weighted by regional importance (0-100)",
                "interpretation": "Climate stress accounting for what matters most in this region",
                "scale": "0-100",
                "good_range": "0-30",
                "concerning_range": "70-100",
                "use_cases": ["Regional risk assessment", "Adaptation priorities"],
                "category": "regional_adaptation"
            },
            
            "is_regional_temp_anomaly": {
                "description": "Binary indicator of regional temperature anomaly (0/1)",
                "interpretation": "1 = Temperature is unusual for this climate zone",
                "scale": "Binary",
                "use_cases": ["Anomaly alerts", "Climate monitoring", "Research"],
                "category": "regional_adaptation"
            },
            
            "climate_zone": {
                "description": "Climate zone classification (tropical, arid, temperate, etc.)",
                "interpretation": "Broad climate classification for regional context",
                "scale": "Categorical",
                "use_cases": ["Regional grouping", "Climate analysis", "Adaptation planning"],
                "category": "regional_adaptation"
            },
            
            # ========================================
            # GLOBAL COMPARISONS (World Context)
            # ========================================
            "temperature_2m_global_percentile": {
                "description": "Global temperature percentile (0-100, vs worldwide conditions)",
                "interpretation": "Where today's temperature ranks globally",
                "scale": "0-100%",
                "good_range": "25-75%",
                "concerning_range": "<1% or >99%",
                "use_cases": ["Global monitoring", "Climate research", "International comparisons"],
                "category": "global_comparisons"
            },
            
            "temperature_2m_global_rank": {
                "description": "Global temperature ranking category",
                "interpretation": "Categorical ranking vs global temperature distribution",
                "scale": "Bottom_10%, Low, Normal, High, Top_10%",
                "use_cases": ["Communication", "Risk categorization", "Public awareness"],
                "category": "global_comparisons"
            },
            
            "temperature_2m_global_anomaly_sigma": {
                "description": "Temperature anomaly in standard deviations from global mean",
                "interpretation": "How many standard deviations from global average",
                "scale": "Standard deviations",
                "good_range": "-1 to +1σ",
                "concerning_range": ">±3σ",
                "use_cases": ["Climate science", "Extreme event detection", "Research"],
                "category": "global_comparisons"
            },
            
            "global_climate_favorability_ranking": {
                "description": "Global climate favorability ranking (0-100)",
                "interpretation": "How favorable climate is compared to rest of world",
                "scale": "0-100",
                "good_range": "60-100",
                "concerning_range": "0-20",
                "use_cases": ["Location comparison", "Migration decisions", "Investment"],
                "category": "global_comparisons"
            },
            
            "climate_change_acceleration": {
                "description": "Local climate change acceleration vs global rate (0-2)",
                "interpretation": "How fast local warming compares to global average",
                "scale": "0-2 (1 = global average)",
                "good_range": "0.8-1.2",
                "concerning_range": ">1.5",
                "use_cases": ["Climate change monitoring", "Adaptation urgency"],
                "category": "global_comparisons"
            },
            
            "global_comparison_zone": {
                "description": "Global latitude zone for comparison (equatorial, tropical, etc.)",
                "interpretation": "Which global zone this location belongs to",
                "scale": "Categorical",
                "use_cases": ["Global analysis", "Cross-regional studies"],
                "category": "global_comparisons"
            },
            
            # ========================================
            # SEASONAL ADJUSTMENT (Temporal Context)
            # ========================================
            "hemisphere_season": {
                "description": "Hemisphere-corrected season (Winter, Spring, Summer, Fall)",
                "interpretation": "Correct seasonal classification for hemisphere",
                "scale": "Categorical",
                "use_cases": ["Seasonal analysis", "Tourism", "Agriculture"],
                "category": "seasonal_adjustment"
            },
            
            "temperature_2m_seasonal_deviation": {
                "description": "Temperature deviation from seasonal baseline (°C)",
                "interpretation": "How much warmer/cooler than normal for this season",
                "scale": "°C",
                "good_range": "-3 to +3°C",
                "concerning_range": ">±10°C",
                "use_cases": ["Seasonal anomaly detection", "Weather forecasting"],
                "category": "seasonal_adjustment"
            },
            
            "temperature_2m_seasonal_anomaly_magnitude": {
                "description": "Magnitude of seasonal temperature anomaly (standard deviations)",
                "interpretation": "How unusual temperature is for this season",
                "scale": "Standard deviations",
                "good_range": "0-1σ",
                "concerning_range": ">2σ",
                "use_cases": ["Seasonal monitoring", "Agricultural alerts"],
                "category": "seasonal_adjustment"
            },
            
            "temperature_2m_deseasonalized": {
                "description": "Temperature with seasonal cycle removed (°C)",
                "interpretation": "Temperature trend without normal seasonal variation",
                "scale": "°C",
                "use_cases": ["Climate trend analysis", "Long-term monitoring"],
                "category": "seasonal_adjustment"
            },
            
            "seasonal_transition": {
                "description": "Binary indicator of season change (0/1)",
                "interpretation": "1 = Season just changed, 0 = within season",
                "scale": "Binary",
                "use_cases": ["Seasonal planning", "Transition monitoring"],
                "category": "seasonal_adjustment"
            },
            
            "is_wet_season": {
                "description": "Binary indicator of wet season (0/1)",
                "interpretation": "1 = This is the wet season for this location",
                "scale": "Binary",
                "use_cases": ["Agricultural planning", "Flood preparedness"],
                "category": "seasonal_adjustment"
            },
            
            "is_heat_risk_season": {
                "description": "Binary indicator of heat risk season (0/1)",
                "interpretation": "1 = Season with highest heat-related risks",
                "scale": "Binary",
                "use_cases": ["Health warnings", "Cooling planning"],
                "category": "seasonal_adjustment"
            },
            
            # ========================================
            # DERIVED STRESS INDICATORS
            # ========================================
            "temperature_stress": {
                "description": "Temperature-specific stress level (0-100)",
                "interpretation": "Stress caused by temperature alone",
                "scale": "0-100",
                "category": "stress_indicators"
            },
            
            "humidity_stress": {
                "description": "Humidity-specific stress level (0-100)",
                "interpretation": "Stress caused by humidity levels",
                "scale": "0-100",
                "category": "stress_indicators"
            },
            
            "wind_stress": {
                "description": "Wind-specific stress level (0-100)",
                "interpretation": "Stress caused by wind conditions",
                "scale": "0-100",
                "category": "stress_indicators"
            },
            
            "precipitation_stress": {
                "description": "Precipitation-related stress level (0-100)",
                "interpretation": "Stress from too much or too little precipitation",
                "scale": "0-100",
                "category": "stress_indicators"
            }
        }
    
    def _define_feature_categories(self) -> Dict[str, Dict[str, Any]]:
        """Define feature categories and their characteristics."""
        return {
            "climate_indicators": {
                "description": "Universal climate stress and comfort indices",
                "purpose": "Provide location-independent climate assessments",
                "key_features": [
                    "climate_stress_index", "human_comfort_index", "weather_extremity_index",
                    "air_quality_health_impact", "climate_hazard_index"
                ],
                "importance": "High - Core universal metrics"
            },
            
            "regional_adaptation": {
                "description": "Regional context and local baselines",
                "purpose": "Adapt features to local climate norms and expectations",
                "key_features": [
                    "regional_climate_stress_index", "temp_deviation_from_regional_norm",
                    "local_percentile", "climate_zone"
                ],
                "importance": "High - Essential for local context"
            },
            
            "global_comparisons": {
                "description": "Global benchmarking and world rankings",
                "purpose": "Show how location compares to rest of world",
                "key_features": [
                    "global_percentile", "global_rank", "global_anomaly_sigma",
                    "climate_change_acceleration"
                ],
                "importance": "Medium - Important for global context"
            },
            
            "seasonal_adjustment": {
                "description": "Hemisphere-aware seasonal processing",
                "purpose": "Handle seasonal patterns correctly for any hemisphere",
                "key_features": [
                    "hemisphere_season", "seasonal_deviation", "deseasonalized",
                    "seasonal_anomaly_magnitude"
                ],
                "importance": "High - Critical for temporal analysis"
            },
            
            "stress_indicators": {
                "description": "Individual stress component indicators",
                "purpose": "Break down overall stress into components",
                "key_features": [
                    "temperature_stress", "humidity_stress", "wind_stress", "precipitation_stress"
                ],
                "importance": "Medium - Useful for detailed analysis"
            }
        }
    
    def _define_feature_importance(self) -> Dict[str, int]:
        """Define feature importance rankings (1-10, 10 = most important)."""
        return {
            # Core universal indicators (highest importance)
            "climate_stress_index": 10,
            "human_comfort_index": 9,
            "climate_hazard_index": 9,
            "air_quality_health_impact": 9,
            
            # Regional context (very important)
            "regional_climate_stress_index": 8,
            "temp_deviation_from_regional_norm": 8,
            "climate_zone": 8,
            "hemisphere_season": 8,
            
            # Global context (important)
            "temperature_2m_global_percentile": 7,
            "global_climate_favorability_ranking": 7,
            "climate_change_acceleration": 7,
            
            # Detailed metrics (moderately important)
            "weather_extremity_index": 6,
            "climate_variability_index": 6,
            "temperature_2m_seasonal_deviation": 6,
            "heat_index": 6,
            
            # Component indicators (useful but not critical)
            "temperature_stress": 5,
            "humidity_stress": 5,
            "local_percentile": 5,
            "seasonal_anomaly_magnitude": 5,
            
            # Specialized indicators (contextually important)
            "is_regional_temp_anomaly": 4,
            "seasonal_transition": 4,
            "is_wet_season": 4,
            "global_rank": 4,
            
            # Technical features (background context)
            "deseasonalized": 3,
            "global_anomaly_sigma": 3,
            "baseline": 3
        }
    
    def _define_feature_relationships(self) -> Dict[str, List[str]]:
        """Define relationships between features."""
        return {
            "climate_stress_index": [
                "temperature_stress", "humidity_stress", "wind_stress", "precipitation_stress"
            ],
            "regional_climate_stress_index": [
                "climate_stress_index", "climate_zone", "regional_adaptation"
            ],
            "global_climate_favorability_ranking": [
                "climate_hazard_index", "global_percentile", "global_comparisons"
            ],
            "seasonal_adjustment": [
                "hemisphere_season", "seasonal_deviation", "seasonal_anomaly"
            ],
            "climate_change_acceleration": [
                "deseasonalized", "long_term_trend", "global_warming_rate"
            ]
        }
    
    def get_feature_explanations(self, feature_names: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Get human-readable explanations of features.
        
        Args:
            feature_names: Specific features to explain (optional)
            
        Returns:
            Dict of {feature_name: explanation}
        """
        if feature_names is None:
            # Return all feature explanations
            return {name: info["description"] for name, info in self.feature_catalog.items()}
        
        explanations = {}
        for feature_name in feature_names:
            if feature_name in self.feature_catalog:
                explanations[feature_name] = self.feature_catalog[feature_name]["description"]
            else:
                # Try to find partial matches
                matches = [name for name in self.feature_catalog.keys() if feature_name in name]
                if matches:
                    explanations[feature_name] = f"Similar features: {', '.join(matches[:3])}"
                else:
                    explanations[feature_name] = "Feature not found in library"
        
        return explanations
    
    def get_features_by_category(self, category: str) -> List[str]:
        """Get all features in a specific category."""
        return [name for name, info in self.feature_catalog.items() 
                if info.get("category") == category]
    
    def get_high_importance_features(self, min_importance: int = 7) -> List[str]:
        """Get features above specified importance threshold."""
        high_importance = []
        for feature_name, importance in self.feature_importance.items():
            if importance >= min_importance:
                high_importance.append(feature_name)
        
        return sorted(high_importance, key=lambda x: self.feature_importance.get(x, 0), reverse=True)
    
    def get_feature_usage_guide(self, feature_name: str) -> Dict[str, Any]:
        """Get comprehensive usage guide for a feature."""
        if feature_name not in self.feature_catalog:
            return {"error": f"Feature '{feature_name}' not found in library"}
        
        feature_info = self.feature_catalog[feature_name].copy()
        
        # Add importance and relationships
        feature_info["importance_score"] = self.feature_importance.get(feature_name, 0)
        feature_info["related_features"] = self.feature_relationships.get(feature_name, [])
        
        # Add category information
        category = feature_info.get("category", "unknown")
        if category in self.feature_categories:
            feature_info["category_info"] = self.feature_categories[category]
        
        return feature_info
    
    def generate_feature_report(self, features_present: List[str]) -> Dict[str, Any]:
        """Generate comprehensive report of features present in dataset."""
        report = {
            "total_features": len(features_present),
            "categories_present": {},
            "importance_distribution": {},
            "missing_high_importance": [],
            "feature_coverage": {}
        }
        
        # Analyze categories present
        for feature in features_present:
            if feature in self.feature_catalog:
                category = self.feature_catalog[feature].get("category", "unknown")
                if category not in report["categories_present"]:
                    report["categories_present"][category] = []
                report["categories_present"][category].append(feature)
        
        # Analyze importance distribution
        importance_counts = {i: 0 for i in range(1, 11)}
        for feature in features_present:
            importance = self.feature_importance.get(feature, 0)
            if importance > 0:
                importance_counts[importance] += 1
        
        report["importance_distribution"] = importance_counts
        
        # Check for missing high-importance features
        high_importance = self.get_high_importance_features(min_importance=8)
        missing_high = [f for f in high_importance if f not in features_present]
        report["missing_high_importance"] = missing_high
        
        # Calculate feature coverage by category
        for category, category_info in self.feature_categories.items():
            category_features = self.get_features_by_category(category)
            present_in_category = [f for f in category_features if f in features_present]
            coverage = len(present_in_category) / len(category_features) if category_features else 0
            
            report["feature_coverage"][category] = {
                "coverage_percentage": coverage * 100,
                "features_present": len(present_in_category),
                "features_total": len(category_features),
                "missing_features": [f for f in category_features if f not in features_present]
            }
        
        return report
    
    def get_all_features(self) -> List[str]:
        """Get list of all documented features."""
        return list(self.feature_catalog.keys())
    
    def search_features(self, search_term: str) -> List[str]:
        """Search for features by name or description."""
        search_term = search_term.lower()
        matches = []
        
        for feature_name, feature_info in self.feature_catalog.items():
            # Search in feature name
            if search_term in feature_name.lower():
                matches.append(feature_name)
                continue
            
            # Search in description
            if search_term in feature_info.get("description", "").lower():
                matches.append(feature_name)
                continue
            
            # Search in use cases
            use_cases = feature_info.get("use_cases", [])
            if any(search_term in use_case.lower() for use_case in use_cases):
                matches.append(feature_name)
        
        return matches


if __name__ == "__main__":
    # Test the feature library
    print("📚 Testing Universal Feature Library")
    print("=" * 50)
    
    library = FeatureLibrary()
    
    print(f"📊 Total features documented: {len(library.get_all_features())}")
    print(f"📂 Categories available: {len(library.feature_categories)}")
    
    # Test feature search
    print(f"\n🔍 Searching for 'stress' features:")
    stress_features = library.search_features("stress")
    for feature in stress_features[:5]:  # Show first 5
        print(f"   • {feature}")
    
    # Test high importance features
    print(f"\n⭐ High importance features (8+):")
    high_importance = library.get_high_importance_features(min_importance=8)
    for feature in high_importance[:5]:  # Show first 5
        importance = library.feature_importance.get(feature, 0)
        print(f"   • {feature} (importance: {importance})")
    
    # Test feature explanation
    print(f"\n📖 Example feature explanation:")
    explanation = library.get_feature_usage_guide("climate_stress_index")
    print(f"   {explanation.get('description', 'No description')}")
    print(f"   Scale: {explanation.get('scale', 'Unknown')}")
    print(f"   Good range: {explanation.get('good_range', 'Unknown')}")
    
    # Test category breakdown
    print(f"\n📂 Features by category:")
    for category, info in library.feature_categories.items():
        features_in_category = library.get_features_by_category(category)
        print(f"   • {category}: {len(features_in_category)} features")